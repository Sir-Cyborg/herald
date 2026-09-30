"""Runtime settings, read from ``HERALD_*`` environment variables.

Every command line option that has an environment counterpart takes its default from
here, so the precedence is: CLI option > environment variable > built-in default.

The project root only anchors the built-in defaults. A relative path given in an
environment variable is used as it is, that is, relative to the current directory.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from herald import paths
from herald.errors import ConfigError

ENV_PREFIX = "HERALD_"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_TIMEOUT = 120.0
DEFAULT_CHECKPOINT_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main"

DEFAULT_SYSTEM_PROMPT = (
    "You are Frieren, an elf mage who has lived for over a thousand years. "
    "You speak calmly, a bit detached, sometimes dryly funny, and you often "
    "reflect on the long passage of time. Keep answers SHORT, 1-2 sentences, "
    "since they will be spoken out loud."
)


@dataclass(frozen=True)
class Settings:
    """Resolved settings. Build with :meth:`from_env`."""

    project_root: Path
    dataset_dir: Path
    models_dir: Path  # fine-tuned voices: <models_dir>/<speaker>/best_model.pth
    checkpoint_dir: Path  # the base XTTS-v2 weights
    runs_dir: Path
    output_dir: Path
    tools_dir: Path  # your own tool scripts (*.py) for the chat assistant
    checkpoint: Path | None = None  # fine-tuned checkpoint (file or run directory) to speak with
    checkpoint_url: str = DEFAULT_CHECKPOINT_URL
    device: str = "auto"
    language: str = "en"
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_timeout_raw: str | None = None  # HERALD_OLLAMA_TIMEOUT as given, see ollama_timeout
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @property
    def ollama_timeout(self) -> float:
        """The Ollama timeout in seconds.

        Validated when read rather than in :meth:`from_env`, so that a bad value breaks only
        the command that uses it (chat), not ``--help`` or ``train``.
        """
        return _parse_timeout(self.ollama_timeout_raw)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read settings from ``env`` (defaults to ``os.environ``)."""
        env = os.environ if env is None else env

        def get(name: str) -> str | None:
            value = env.get(ENV_PREFIX + name)
            return value if value else None  # an empty variable counts as unset

        def get_path(name: str) -> Path | None:
            value = get(name)
            return Path(value).expanduser() if value else None

        root = paths.project_root(env)
        models_dir = get_path("MODELS_DIR") or root / paths.DEFAULT_MODELS_SUBDIR
        return cls(
            project_root=root,
            dataset_dir=get_path("DATASET_DIR") or root / paths.DEFAULT_DATASET_SUBDIR,
            models_dir=models_dir,
            checkpoint_dir=get_path("CHECKPOINT_DIR") or models_dir / paths.BASE_MODEL_SUBDIR,
            runs_dir=get_path("RUNS_DIR") or root / paths.DEFAULT_RUNS_SUBDIR,
            output_dir=get_path("OUTPUT_DIR") or root / paths.DEFAULT_OUTPUT_SUBDIR,
            tools_dir=get_path("TOOLS_DIR") or root / paths.DEFAULT_TOOLS_SUBDIR,
            checkpoint=get_path("CHECKPOINT"),
            checkpoint_url=get("CHECKPOINT_URL") or DEFAULT_CHECKPOINT_URL,
            device=get("DEVICE") or "auto",
            language=get("LANGUAGE") or "en",
            ollama_url=get("OLLAMA_URL") or DEFAULT_OLLAMA_URL,
            ollama_model=get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
            ollama_timeout_raw=get("OLLAMA_TIMEOUT"),
            system_prompt=get("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
        )


def _parse_timeout(raw: str | None) -> float:
    if raw is None:
        return DEFAULT_OLLAMA_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{ENV_PREFIX}OLLAMA_TIMEOUT must be a number, got {raw!r}") from None
    if value <= 0:
        raise ConfigError(f"{ENV_PREFIX}OLLAMA_TIMEOUT must be positive, got {raw!r}")
    return value
