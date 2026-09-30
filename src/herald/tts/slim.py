"""Shrink a coqui-tts training checkpoint to what inference needs.

A checkpoint written by the coqui ``Trainer`` holds the model weights (``"model"``, ~1.9 GB
for XTTS-v2) plus the optimizer, scheduler and scaler state, the config and the step
counters: ~5.6 GB in total. Inference only reads ``"model"``
(``Xtts.get_compatible_checkpoint_state_dict``) and herald never resumes a training run, so
:func:`slim_checkpoint` keeps just that entry.

torch is imported inside the function, so this module imports without it.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from herald.errors import MISSING_STACK_MESSAGE, CheckpointError, DependencyError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlimResult:
    source: Path
    path: Path  # the slim file (``source`` itself when it was already slim or replaced)
    size_before: int
    size_after: int
    already_slim: bool = False


def format_size(n_bytes: float) -> str:
    """``n_bytes`` for humans: ``"812 bytes"``, ``"12.00 MB"``, ``"5.61 GB"``."""
    size = float(n_bytes)
    for unit in ("bytes", "KB", "MB"):
        if size < 1000:
            return f"{size:.0f} bytes" if unit == "bytes" else f"{size:.2f} {unit}"
        size /= 1000
    return f"{size:.2f} GB"


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise DependencyError(MISSING_STACK_MESSAGE) from exc
    return torch


def _load(torch: Any, path: Path) -> Any:
    """Memory-map ``path``, so a 5.6 GB file does not have to fit in RAM."""
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"{path} could not be read as a checkpoint: {exc}") from exc


def _tensor_bytes(torch: Any, state_dict: dict[str, Any]) -> int:
    return sum(t.numel() * t.element_size() for t in state_dict.values() if torch.is_tensor(t))


def _same_tensor(torch: Any, a: Any, b: Any) -> bool:
    """Same shape, dtype and values, compared bit for bit (so a NaN equals itself)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def _verify(torch: Any, path: Path, model: dict[str, Any]) -> None:
    """Reload ``path`` and check it holds exactly ``model``: same keys, shapes, dtypes, values.

    Both sides are memory-mapped, so this reads the model part of the checkpoint twice.
    """
    saved = _load(torch, path)
    if (
        not isinstance(saved, dict)
        or set(saved) != {"model"}
        or not isinstance(saved["model"], dict)
    ):
        raise CheckpointError(f"Verification failed: {path} is not a {{'model': ...}} checkpoint")
    saved_model = saved["model"]
    if set(saved_model) != set(model):
        raise CheckpointError(f"Verification failed: {path} has a different set of weights")
    for key, tensor in model.items():
        other = saved_model[key]
        if torch.is_tensor(tensor):
            same = torch.is_tensor(other) and _same_tensor(torch, tensor, other)
        else:
            same = type(other) is type(tensor)
        if not same:
            raise CheckpointError(f"Verification failed: {key!r} differs in {path}")


def slim_checkpoint(source: Path, dest: Path | None = None, *, replace: bool = False) -> SlimResult:
    """Write a copy of the training checkpoint ``source`` that holds only its model weights.

    The default ``dest`` is ``<name>.slim.pth`` next to ``source`` (an existing file of that
    name is overwritten); with ``replace=True`` the slim file takes the place of ``source``
    instead, and ``dest`` must not be given. A checkpoint that already holds nothing but
    ``"model"`` is left alone (``already_slim``, nothing is written).

    The slim file is written to ``<dest>.part``, reloaded and compared (same keys, shapes,
    dtypes and values) and only then renamed into place, so an interrupted or failed run
    leaves no partial file and ``source`` is never touched unless the verified replacement
    is done.
    Raises :class:`CheckpointError` if ``source`` is not a coqui checkpoint or the disk is
    clearly too small.

    The file is unpickled with ``weights_only=False`` (it holds config objects), which can
    run arbitrary code: only slim checkpoints you trust, such as your own training runs.
    """
    if replace and dest is not None:
        raise ValueError("dest and replace are mutually exclusive")
    source = Path(source)
    if not source.is_file():
        raise CheckpointError(f"Checkpoint not found: {source}")
    if replace:
        target = source
    else:
        target = Path(dest) if dest is not None else source.with_name(source.stem + ".slim.pth")
        if target.resolve() == source.resolve():
            raise ValueError("dest is the source file: use replace=True to slim it in place")

    torch = _import_torch()
    size_before = source.stat().st_size
    state = _load(torch, source)
    model = state.get("model") if isinstance(state, dict) else None
    if not isinstance(model, dict):
        raise CheckpointError(f"{source} is not a coqui training checkpoint: no 'model' state dict")
    if set(state) == {"model"}:
        logger.info("%s is already slim (%s)", source, format_size(size_before))
        return SlimResult(source, source, size_before, size_before, already_slim=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    needed = _tensor_bytes(torch, model)
    free = shutil.disk_usage(target.parent).free
    if free < needed:
        raise CheckpointError(
            f"Not enough disk space in {target.parent} for the slim checkpoint: "
            f"{format_size(needed)} needed, {format_size(free)} free"
        )

    part = target.with_name(target.name + ".part")
    try:
        torch.save({"model": model}, part)
        _verify(torch, part, model)
        if replace:
            # The source is memory-mapped; Windows cannot replace a mapped file.
            del state, model
            gc.collect()
        os.replace(part, target)
    except (OSError, RuntimeError) as exc:  # torch.save reports a full disk as RuntimeError
        raise CheckpointError(f"Could not write {target}: {exc}") from exc
    finally:
        part.unlink(missing_ok=True)  # nothing left to remove once the rename has happened

    size_after = target.stat().st_size
    logger.info(
        "Slimmed %s: %s -> %s (%s)",
        source,
        format_size(size_before),
        format_size(size_after),
        target,
    )
    return SlimResult(source, target, size_before, size_after)
