"""Project layout: where the dataset, models, training runs and outputs live.

::

    dataset/<speaker>/audio/*.wav + metadata.csv
    models/xtts_v2/        base weights
    models/<speaker>/      fine-tuned voice (best_model.pth)
    runs/                  training logs and TensorBoard data
    output/                generated audio
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

ROOT_ENV = "HERALD_PROJECT_ROOT"

# Default locations, relative to the project root.
DEFAULT_DATASET_SUBDIR = Path("dataset") / "frieren"
DEFAULT_MODELS_SUBDIR = Path("models")
DEFAULT_RUNS_SUBDIR = Path("runs")
DEFAULT_OUTPUT_SUBDIR = Path("output")

# The base XTTS-v2 weights, relative to the models directory.
BASE_MODEL_SUBDIR = Path("xtts_v2")


def _source_checkout_root() -> Path | None:
    """The repository root when herald runs from a source checkout (e.g. `pip install -e .`)."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "herald").is_dir():
        return candidate
    return None


def project_root(env: Mapping[str, str] | None = None, cwd: Path | None = None) -> Path:
    """Resolve the project root.

    Order: ``HERALD_PROJECT_ROOT``, then the source checkout herald is running from,
    then the current working directory (a regular ``pip install`` or a container).
    """
    env = os.environ if env is None else env
    explicit = env.get(ROOT_ENV)
    if explicit:
        return Path(explicit).expanduser().resolve()
    return _source_checkout_root() or (cwd or Path.cwd()).resolve()
