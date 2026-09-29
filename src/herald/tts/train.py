"""Fine-tune the XTTS-v2 GPT on a voice dataset.

The training data is a single ``metadata.csv``: the train/eval split is derived from it
(seeded, so it is reproducible) instead of being read from pre-split files. When training
ends the best model is written, without its optimizer state, to
``<models_dir>/<model_name>/`` (never over an existing model: that one is renamed with a
timestamp) and the run directory's checkpoints, which are ~3x bigger, are deleted.

The parameters, the samples, the split and the trainer configuration are plain Python and
need no heavy dependency; only :func:`run_training` imports torch and coqui-tts.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from herald.config import DEFAULT_CHECKPOINT_URL
from herald.dataset.metadata import (
    METADATA_FILE,
    MetadataItem,
    load_metadata,
    resolve_audio_path,
    split_items,
)
from herald.errors import (
    MISSING_STACK_MESSAGE,
    CheckpointError,
    ConfigError,
    DependencyError,
    MetadataError,
)
from herald.tts import checkpoints, slim
from herald.tts.engine import prepare_torch_env, resolve_finetuned_checkpoint

logger = logging.getLogger(__name__)

DEFAULT_LR = 5e-6
DEFAULT_EVAL_FRACTION = 0.1
DEFAULT_SEED = 42
BEST_MODEL_NAME = "best_model.pth"
CONFIG_NAME = "config.json"

# "full" is the configuration of the original full run; "smoke" the quick sanity run
# (a few dozen samples, one epoch) used to check that training does not crash.
# A checkpoint (model + optimizer state) is ~5.6 GB and the Trainer writes the new one before
# deleting the oldest, so ``save_n_checkpoints`` N means N + 1 of them on disk at once: 1 is
# the smallest footprint that still leaves a fallback if a run stops before a best model.
PRESETS: dict[str, dict[str, Any]] = {
    "full": {
        "epochs": 5,
        "batch_size": 2,
        "grad_accum_steps": 16,
        "print_step": 25,
        "plot_step": 100,
        "save_step": 250,
        "save_n_checkpoints": 1,
        "print_eval": False,
        "max_train_samples": None,
        "max_eval_samples": None,
    },
    "smoke": {
        "epochs": 1,
        "batch_size": 1,
        "grad_accum_steps": 4,
        "print_step": 5,
        "plot_step": 50,
        "save_step": 500,
        "save_n_checkpoints": 1,
        "print_eval": True,
        "max_train_samples": 60,
        "max_eval_samples": 20,
    },
}


@dataclass(frozen=True)
class TrainParams:
    dataset_dir: Path
    checkpoint_dir: Path
    checkpoint_url: str
    runs_dir: Path
    models_dir: Path
    metadata_file: str
    eval_fraction: float
    seed: int
    speaker_name: str
    model_name: str  # the trained model goes to <models_dir>/<model_name>/
    language: str
    run_name: str
    project_name: str
    epochs: int
    batch_size: int
    eval_batch_size: int
    grad_accum_steps: int
    lr: float
    num_workers: int
    print_step: int
    plot_step: int
    save_step: int
    save_n_checkpoints: int
    print_eval: bool
    max_train_samples: int | None
    max_eval_samples: int | None
    keep_checkpoints: bool = False  # keep the run directory's checkpoints after promotion

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "batch_size",
            "eval_batch_size",
            "grad_accum_steps",
            "print_step",
            "plot_step",
            "save_step",
        ):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1, got {getattr(self, name)}")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ConfigError(f"lr must be a positive number, got {self.lr}")
        if self.num_workers < 0:
            raise ConfigError(f"num_workers must not be negative, got {self.num_workers}")
        if not 0 <= self.eval_fraction < 1:
            raise ConfigError(f"eval_fraction must be in [0, 1), got {self.eval_fraction}")
        for name, minimum in (("max_train_samples", 1), ("max_eval_samples", 0)):
            value = getattr(self, name)
            if value is not None and value < minimum:
                raise ConfigError(f"{name} must be at least {minimum}, got {value}")
        _check_plain_name("speaker name", self.speaker_name)
        _check_plain_name("model name", self.model_name)


def _check_plain_name(label: str, name: str) -> None:
    """Reject names that are not a single path component (the model name is a directory)."""
    if name in ("", ".", "..") or "/" in name or "\\" in name or Path(name).name != name:
        raise ConfigError(f"Invalid {label} {name!r}: use a plain name without path separators")


def make_params(
    *,
    dataset_dir: Path,
    checkpoint_dir: Path,
    runs_dir: Path,
    models_dir: Path,
    checkpoint_url: str = DEFAULT_CHECKPOINT_URL,
    preset: str = "full",
    language: str = "en",
    metadata_file: str = METADATA_FILE,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    seed: int = DEFAULT_SEED,
    speaker_name: str | None = None,
    model_name: str | None = None,
    run_name: str | None = None,
    project_name: str | None = None,
    keep_checkpoints: bool = False,
    **overrides: Any,
) -> TrainParams:
    """Build :class:`TrainParams` from a preset plus explicit overrides.

    Overrides set to ``None`` are ignored, so CLI options that were not given fall back
    to the preset. The speaker defaults to the dataset directory name.
    ``eval_fraction`` and ``seed`` control the train/eval split of ``metadata_file``.

    The trained model goes to ``models_dir/model_name``. By default that is the speaker for
    the ``full`` preset and ``<speaker>_<preset>`` for the others, so a quick ``smoke`` run
    never ends up next to (let alone over) the real voice.
    """
    if preset not in PRESETS:
        raise ConfigError(f"Unknown preset {preset!r}: choose from {', '.join(PRESETS)}")
    dataset_dir = Path(dataset_dir)
    speaker = speaker_name or dataset_dir.resolve().name
    values: dict[str, Any] = {
        "lr": DEFAULT_LR,
        "num_workers": 2,
        **PRESETS[preset],
        **{k: v for k, v in overrides.items() if v is not None},
    }
    values.setdefault("eval_batch_size", values["batch_size"])
    return TrainParams(
        dataset_dir=dataset_dir,
        checkpoint_dir=Path(checkpoint_dir),
        checkpoint_url=checkpoint_url,
        runs_dir=Path(runs_dir),
        models_dir=Path(models_dir),
        metadata_file=metadata_file,
        eval_fraction=eval_fraction,
        seed=seed,
        speaker_name=speaker,
        model_name=model_name or (speaker if preset == "full" else f"{speaker}_{preset}"),
        language=language,
        run_name=run_name or f"{speaker}_{preset}",
        project_name=project_name or f"{speaker}_xtts",
        keep_checkpoints=keep_checkpoints,
        **values,
    )


# --- dataset --------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetReport:
    """What :func:`run_training` will use: rows after the split and the sample limits."""

    train_rows: int
    eval_rows: int
    missing_audio: tuple[str, ...]  # of those rows, sorted


def _select_rows(
    params: TrainParams,
) -> tuple[list[MetadataItem], list[MetadataItem], tuple[str, ...]]:
    """The train and eval rows a run uses, and those among them whose audio is missing.

    Reads the metadata file, splits it and then applies ``max_train_samples`` /
    ``max_eval_samples`` (the smoke preset), so eval never overlaps train.
    """
    metadata_path = params.dataset_dir / params.metadata_file
    items = load_metadata(metadata_path)
    if not items:
        raise MetadataError(f"{metadata_path}: no usable rows")
    train_items, eval_items = split_items(items, params.eval_fraction, params.seed)
    train_items = train_items[: params.max_train_samples]
    eval_items = eval_items[: params.max_eval_samples]

    missing = []
    for item in [*train_items, *eval_items]:
        try:
            resolve_audio_path(item["audio_file"], params.dataset_dir)
        except FileNotFoundError:
            missing.append(item["audio_file"])
    return train_items, eval_items, tuple(sorted(missing))


def validate_dataset(params: TrainParams) -> DatasetReport:
    """Check the rows :func:`run_training` would use: how many, and that their audio exists."""
    train_items, eval_items, missing = _select_rows(params)
    return DatasetReport(len(train_items), len(eval_items), missing)


def build_samples(items: Sequence[MetadataItem], params: TrainParams) -> list[dict[str, Any]]:
    """Turn metadata rows into the sample dicts that coqui-tts datasets expect.

    ``audio_file``, ``text`` and ``language`` are what the XTTS dataset reads; the other
    keys are the ones ``TTS.tts.datasets.load_tts_samples`` would add.
    """
    root_path = str(params.dataset_dir.resolve())
    samples = []
    for item in items:
        audio_file = str(resolve_audio_path(item["audio_file"], params.dataset_dir))
        unique_name = Path(item["audio_file"]).with_suffix("")
        samples.append(
            {
                "audio_file": audio_file,
                "text": item["text"],
                "speaker_name": params.speaker_name,
                "language": params.language,
                "root_path": root_path,
                "audio_unique_name": f"{params.speaker_name}#{unique_name}",
            }
        )
    return samples


# --- trainer configuration --------------------------------------------------------------------


def model_args_kwargs(checkpoint_dir: Path) -> dict[str, Any]:
    """Keyword arguments for ``GPTArgs``."""
    checkpoint_dir = Path(checkpoint_dir)
    return {
        "max_conditioning_length": 132_300,  # 6 s at 22.05 kHz
        "min_conditioning_length": 66_150,  # 3 s
        "max_wav_length": 255_995,  # ~11.6 s
        "max_text_length": 200,
        "mel_norm_file": str(checkpoint_dir / "mel_stats.pth"),
        "dvae_checkpoint": str(checkpoint_dir / "dvae.pth"),
        "xtts_checkpoint": str(checkpoint_dir / "model.pth"),
        "tokenizer_file": str(checkpoint_dir / "vocab.json"),
        "gpt_num_audio_tokens": 1026,
        "gpt_start_audio_token": 1024,
        "gpt_stop_audio_token": 1025,
        "gpt_use_masking_gt_prompt_approach": True,
        "gpt_use_perceiver_resampler": True,
    }


AUDIO_CONFIG_KWARGS = {
    "sample_rate": 22_050,
    "dvae_sample_rate": 22_050,
    "output_sample_rate": 24_000,
}


def trainer_config_kwargs(params: TrainParams, *, run_eval: bool = True) -> dict[str, Any]:
    """Keyword arguments for ``GPTTrainerConfig``, except ``model_args`` and ``audio``.

    ``run_eval=False`` is for a run without eval samples: the trainer then skips the
    evaluation and picks the best model by training loss.
    """
    return {
        "output_path": str(params.runs_dir),
        "run_name": params.run_name,
        "project_name": params.project_name,
        "dashboard_logger": "tensorboard",
        "batch_size": params.batch_size,
        "eval_batch_size": params.eval_batch_size,
        "num_loader_workers": params.num_workers,
        "num_eval_loader_workers": params.num_workers,
        "print_step": params.print_step,
        "plot_step": params.plot_step,
        "log_model_step": 100,
        "save_step": params.save_step,
        "save_n_checkpoints": params.save_n_checkpoints,
        "save_checkpoints": True,
        "print_eval": params.print_eval,
        "run_eval": run_eval,
        "optimizer": "AdamW",
        "optimizer_wd_only_on_weights": True,
        "optimizer_params": {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        "lr": params.lr,
        "lr_scheduler": "MultiStepLR",
        "lr_scheduler_params": {
            "milestones": [50_000, 150_000, 300_000],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        "epochs": params.epochs,
        "test_sentences": [],
    }


def _import_training_stack() -> SimpleNamespace:
    prepare_torch_env()
    try:
        import torch
        from trainer import Trainer, TrainerArgs
        from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
        from TTS.tts.models.xtts import XttsAudioConfig
    except ImportError as exc:
        raise DependencyError(MISSING_STACK_MESSAGE) from exc
    return SimpleNamespace(
        torch=torch,
        Trainer=Trainer,
        TrainerArgs=TrainerArgs,
        GPTArgs=GPTArgs,
        GPTTrainer=GPTTrainer,
        GPTTrainerConfig=GPTTrainerConfig,
        XttsAudioConfig=XttsAudioConfig,
    )


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    model_path: Path | None  # where the best model was promoted to; None if none was found
    freed_bytes: int = 0  # size of the run directory's checkpoints that were deleted


_STEPPED_BEST_MODEL = re.compile(r"best_model_(\d+)\.pth")
_RUN_CHECKPOINT_PATTERNS = ("checkpoint_*.pth", "best_model*.pth")


def _find_best_checkpoint(run_dir: Path) -> Path:
    """The checkpoint to promote from a run directory.

    coqui's Trainer keeps ``best_model_<step>.pth`` plus an identical ``best_model.pth``
    copy, so the numbered file is preferred; :func:`resolve_finetuned_checkpoint` covers
    the remaining layouts (only ``best_model.pth``, or only ``checkpoint_<step>.pth``).
    """
    stepped = [
        (int(m.group(1)), path)
        for path in run_dir.glob("best_model_*.pth")
        if (m := _STEPPED_BEST_MODEL.fullmatch(path.name))
    ]
    if stepped:
        return max(stepped)[1]
    return resolve_finetuned_checkpoint(run_dir)


def _backup_tag(dest_dir: Path) -> str:
    """A timestamp (plus a counter if needed) that no backup in ``dest_dir`` uses yet."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag, n = stamp, 0
    while (dest_dir / f"best_model.{tag}.pth").exists() or (
        dest_dir / f"config.{tag}.json"
    ).exists():
        n += 1
        tag = f"{stamp}-{n}"
    return tag


def _stage_model(source: Path, partial: Path) -> None:
    """Put the model of the checkpoint ``source`` at ``partial``.

    Normally that is a slim copy (no optimizer state) and ``source`` stays where it is.
    If slimming is impossible for any reason, or ``source`` is already slim, ``source``
    itself is moved there: the model is never lost. If nothing could be put at ``partial``,
    raises :class:`CheckpointError` saying where the model still is.
    """
    try:
        result = slim.slim_checkpoint(source, partial)
    except Exception as exc:
        logger.warning("Could not slim %s (%s); moving the full checkpoint instead", source, exc)
    except BaseException:  # Ctrl-C: do not leave a half-promoted model behind
        partial.unlink(missing_ok=True)
        raise
    else:
        if not result.already_slim:
            return
    try:
        shutil.move(source, partial)
    except BaseException as exc:
        partial.unlink(missing_ok=True)
        if not isinstance(exc, Exception):  # Ctrl-C: clean up, then let it through
            raise
        raise CheckpointError(
            f"Could not move the trained model to {partial.parent} ({exc}); it is still at {source}"
        ) from exc


def promote_best_model(run_dir: Path, dest_dir: Path) -> Path | None:
    """Turn the best checkpoint of a training run into ``dest_dir/best_model.pth``.

    The promoted file is a slim copy of the checkpoint (:func:`herald.tts.slim.slim_checkpoint`:
    only the model weights, about a third of its size) and the run's ``config.json`` is
    copied next to it. The run directory is left as it was (see :func:`clean_run_dir`),
    except when slimming fails or is unnecessary: then the checkpoint file itself is moved
    (and the run's identical ``best_model.pth`` copy removed if it has the same size), with
    a warning, so the model is never lost.

    A model already in ``dest_dir`` is never overwritten or deleted: it is renamed to
    ``best_model.<YYYYmmdd-HHMMSS>.pth`` (and its ``config.json`` to ``config.<same>.json``,
    with a ``-<n>`` suffix if that name is taken), in whatever format it has.

    Returns the new model path, or ``None`` (with a warning) when the run holds no
    checkpoint. If the model cannot be put in place, raises :class:`CheckpointError`
    saying where it still is; no partial file is left in ``dest_dir``.
    """
    run_dir, dest_dir = Path(run_dir), Path(dest_dir)
    try:
        source = _find_best_checkpoint(run_dir)
    except CheckpointError as exc:
        logger.warning("No model to promote: %s", exc)
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / BEST_MODEL_NAME
    size = source.stat().st_size
    # Under a temporary name first, so that no truncated file ever looks like a finished model.
    partial = dest.with_name(dest.name + ".part")
    _stage_model(source, partial)

    if dest.exists():
        tag = _backup_tag(dest_dir)
        dest.replace(dest_dir / f"best_model.{tag}.pth")
        old_config = dest_dir / CONFIG_NAME
        if old_config.exists():
            old_config.replace(dest_dir / f"config.{tag}.json")
        logger.warning("The model already in %s was kept as best_model.%s.pth", dest_dir, tag)
    partial.replace(dest)

    config = run_dir / CONFIG_NAME
    if config.is_file():
        shutil.copy2(config, dest_dir / CONFIG_NAME)
    duplicate = run_dir / BEST_MODEL_NAME
    if not source.exists() and duplicate.is_file() and duplicate.stat().st_size == size:
        duplicate.unlink()  # source was moved: its identical copy is not needed either
        logger.info("Removed %s, an identical copy of the model that was moved", duplicate)
    logger.info("Best model written to %s", dest)
    return dest


def clean_run_dir(run_dir: Path) -> int:
    """Delete the checkpoints of a finished run and return the number of bytes freed.

    Only ``checkpoint_*.pth`` and ``best_model*.pth`` directly inside ``run_dir`` are
    removed; the config, logs and TensorBoard events stay.
    """
    run_dir = Path(run_dir)
    freed = files = 0
    for pattern in _RUN_CHECKPOINT_PATTERNS:
        for path in sorted(run_dir.glob(pattern)):
            if path.is_file() and not path.is_symlink():
                freed += path.stat().st_size
                path.unlink()
                files += 1
    if files:
        logger.info(
            "Deleted %d checkpoint file(s) from %s: %s freed",
            files,
            run_dir,
            slim.format_size(freed),
        )
    return freed


def run_training(params: TrainParams) -> TrainResult:
    """Fine-tune XTTS-v2 and promote the best model to ``<models_dir>/<model_name>/``.

    Unless ``params.keep_checkpoints`` is set, the checkpoints of the run are deleted once the
    model is safely in place (never when no model was promoted).
    """
    train_items, eval_items, missing = _select_rows(params)
    if missing:
        sample = ", ".join(missing[:5])
        raise MetadataError(
            f"{len(missing)} audio file(s) referenced by the metadata are missing "
            f"from {params.dataset_dir} (e.g. {sample})"
        )
    train_samples = build_samples(train_items, params)
    eval_samples = build_samples(eval_items, params)
    logger.info("train samples: %d, eval samples: %d", len(train_samples), len(eval_samples))
    if not eval_samples:
        logger.warning("No eval samples: the best model is picked by training loss")

    checkpoints.ensure_base_checkpoints(params.checkpoint_dir, base_url=params.checkpoint_url)
    s = _import_training_stack()
    if s.torch.cuda.is_available():
        logger.info("Training on CUDA")
    else:
        logger.warning("No CUDA device: training on the CPU (the trainer has no MPS support)")

    params.runs_dir.mkdir(parents=True, exist_ok=True)
    config = s.GPTTrainerConfig(
        model_args=s.GPTArgs(**model_args_kwargs(params.checkpoint_dir)),
        audio=s.XttsAudioConfig(**AUDIO_CONFIG_KWARGS),
        **trainer_config_kwargs(params, run_eval=bool(eval_samples)),
    )
    model = s.GPTTrainer.init_from_config(config)
    trainer = s.Trainer(
        s.TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=params.grad_accum_steps,
        ),
        config,
        output_path=str(params.runs_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        # Otherwise the Trainer parses our own sys.argv (`herald train --lr ...`).
        parse_command_line_args=False,
    )
    trainer.fit()

    run_dir = Path(trainer.output_path)
    model_path = promote_best_model(run_dir, params.models_dir / params.model_name)
    freed_bytes = 0
    if model_path is not None and not params.keep_checkpoints:
        if model_path.parent.resolve() == run_dir.resolve():
            logger.warning(
                "The model is in the run directory %s: not deleting checkpoints", run_dir
            )
        else:
            freed_bytes = clean_run_dir(run_dir)
    return TrainResult(run_dir=run_dir, model_path=model_path, freed_bytes=freed_bytes)
