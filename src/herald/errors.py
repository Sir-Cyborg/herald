"""Exceptions that the CLI reports as a one-line message instead of a traceback."""


class HeraldError(Exception):
    """Base class for expected, user-facing failures."""


class ConfigError(HeraldError):
    """An environment variable or option has an invalid value."""


class MetadataError(HeraldError):
    """A dataset metadata file is missing or malformed."""


class CheckpointError(HeraldError):
    """A model checkpoint could not be found or downloaded."""


class DependencyError(HeraldError):
    """The heavy TTS stack (torch, coqui-tts) is not importable."""


class ProfileError(HeraldError):
    """A voice profile is missing, malformed, or has an unknown or invalid setting."""


class OllamaError(HeraldError):
    """The Ollama server is unreachable or returned an unusable answer."""


# Shown when torch / coqui-tts cannot be imported; shared by the engine and the trainer.
MISSING_STACK_MESSAGE = (
    "torch / coqui-tts are not installed. Install herald with its dependencies "
    "(`pip install -e .`) or use the Docker image."
)
