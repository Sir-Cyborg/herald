"""Download the original XTTS-v2 weights into the project (once)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import requests

from herald.config import DEFAULT_CHECKPOINT_URL
from herald.errors import CheckpointError

logger = logging.getLogger(__name__)

BASE_FILES = (
    "vocab.json",
    "config.json",
    "model.pth",  # ~1.9 GB, needed for the base model and for fine-tuning
    "dvae.pth",
    "mel_stats.pth",
    "speakers_xtts.pth",
)

# What each use needs. Fine-tuned inference only reads the base config and vocabulary.
FINETUNED_INFERENCE_FILES = ("config.json", "vocab.json")
BASE_INFERENCE_FILES = ("config.json", "vocab.json", "model.pth")

_CHUNK_SIZE = 1 << 20  # 1 MiB
_LOG_EVERY_BYTES = 100 * _CHUNK_SIZE


def ensure_base_checkpoints(
    checkpoint_dir: Path,
    *,
    files: Iterable[str] = BASE_FILES,
    base_url: str = DEFAULT_CHECKPOINT_URL,
    timeout: float = 30.0,
) -> Path:
    """Download the listed files into ``checkpoint_dir`` unless they are already there."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        dest = checkpoint_dir / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        logger.info("Downloading %s ...", name)
        _download(f"{base_url.rstrip('/')}/{name}", dest, timeout)
    return checkpoint_dir


def _download(url: str, dest: Path, timeout: float) -> None:
    """Stream ``url`` to ``dest``.

    Data goes to ``<dest>.part`` first and is renamed only when complete, so an
    interrupted download is never mistaken for a finished file on the next run.
    """
    part = dest.with_name(dest.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length") or 0)
            # A Content-Encoding makes the length refer to the compressed body.
            check_size = total > 0 and not resp.headers.get("Content-Encoding")
            written = 0
            next_log = _LOG_EVERY_BYTES
            with open(part, "wb") as f:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    f.write(chunk)
                    written += len(chunk)
                    if written >= next_log:
                        logger.info("  %s: %d MiB%s", dest.name, written >> 20, _of(total))
                        next_log += _LOG_EVERY_BYTES
        if check_size and written != total:
            raise CheckpointError(
                f"Download of {url} was truncated: got {written} of {total} bytes"
            )
        part.replace(dest)
    except requests.RequestException as exc:
        raise CheckpointError(f"Failed to download {url}: {exc}") from exc
    finally:
        part.unlink(missing_ok=True)


def _of(total: int) -> str:
    return f" of {total >> 20} MiB" if total else ""
