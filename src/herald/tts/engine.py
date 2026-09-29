"""XTTS-v2 inference: base or fine-tuned model, short and long text, whole or streamed.

Everything that needs torch or coqui-tts is imported lazily inside functions. The text
splitting and WAV writing are pure numpy/stdlib and need no model.
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
import threading
import time
import wave
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from herald.errors import MISSING_STACK_MESSAGE, CheckpointError, ConfigError, DependencyError
from herald.tts import checkpoints

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000  # XTTS-v2 output rate
DEFAULT_MAX_CHARS = 250  # the longest chunk we ever synthesize (the English limit)
DEFAULT_PAUSE_MS = 150
DEFAULT_TEMPERATURE = 0.7

# Per-language character limits of the XTTS tokenizer (``char_limits`` in
# TTS/tts/layers/xtts/tokenizer.py): it warns that longer text may come out truncated.
# Keep in sync with the library; a test compares the two when coqui-tts is installed.
LANGUAGE_CHAR_LIMITS = {
    "en": 250,
    "de": 253,
    "fr": 273,
    "es": 239,
    "it": 213,
    "pt": 203,
    "pl": 224,
    "zh": 82,
    "ar": 166,
    "cs": 186,
    "ru": 182,
    "nl": 251,
    "tr": 226,
    "ja": 71,
    "hu": 224,
    "ko": 95,
    "hi": 150,
}

_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")
# A sentence end (. ! ? and friends), any closing quotes or brackets glued to it, then a space.
_SENTENCE_END = re.compile(r"([.!?…。！？][\"'”’»)\]}）」』]*)\s+")


# --- pure helpers -------------------------------------------------------------------------


def default_max_chars(language: str) -> int:
    """The longest chunk, in characters, to synthesize in one call for ``language``.

    Case-insensitive, and a region suffix is ignored ("pt-br" and "zh_CN" count as "pt" and
    "zh"). Never more than ``DEFAULT_MAX_CHARS``; an unknown language gets that value.
    """
    base = re.split(r"[-_]", language.strip().lower(), maxsplit=1)[0]
    return min(LANGUAGE_CHAR_LIMITS.get(base, DEFAULT_MAX_CHARS), DEFAULT_MAX_CHARS)


def _split_sentences(text: str) -> list[str]:
    """Cut ``text`` after each sentence end, keeping closing quotes and brackets with it.

    Abbreviations such as "Dr." are not recognised: they end a sentence like any other period.
    """
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        sentences.append(text[start : match.end(1)])
        start = match.end()
    sentences.append(text[start:])
    return sentences


def split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    Text that already fits is returned as one chunk. Longer text is cut at sentence
    boundaries and consecutive sentences are packed greedily into chunks. A single
    sentence longer than ``max_chars`` is wrapped at word boundaries. Whitespace is
    normalised; blank input gives an empty list.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in _split_sentences(text):
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                textwrap.wrap(sentence, width=max_chars, break_on_hyphens=False, tabsize=1)
            )
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _silence(pause_ms: float, sample_rate: int) -> np.ndarray:
    """``pause_ms`` of silence as a new float32 array (empty for a zero pause)."""
    if pause_ms < 0:
        raise ValueError("pause_ms must not be negative")
    return np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32)


def save_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write mono float samples in [-1, 1] as a 16-bit PCM WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes((pcm * 32767.0).astype("<i2").tobytes())
    return path


def pick_device(preferred: str, *, cuda: bool, mps: bool) -> str:
    """Choose a device: an explicit choice wins, otherwise cuda > mps > cpu."""
    if not _DEVICE_RE.match(preferred):
        raise ConfigError(f"Invalid device {preferred!r}: use auto, cpu, mps, cuda or cuda:N")
    if preferred != "auto":
        return preferred
    if cuda:
        return "cuda"
    if mps:
        return "mps"
    return "cpu"


def resolve_finetuned_checkpoint(path: Path) -> Path:
    """Turn a checkpoint file or a training run directory into a checkpoint file.

    For a directory, prefers ``best_model.pth``, then the highest ``best_model_<step>.pth``,
    then the highest ``checkpoint_<step>.pth``.
    """
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise CheckpointError(f"Checkpoint not found: {path}")

    best = path / "best_model.pth"
    if best.is_file():
        return best
    for prefix in ("best_model", "checkpoint"):
        stepped = [
            (int(m.group(1)), p)
            for p in path.glob(f"{prefix}_*.pth")
            if (m := re.fullmatch(rf"{prefix}_(\d+)\.pth", p.name))
        ]
        if stepped:
            return max(stepped)[1]
    raise CheckpointError(f"No best_model*.pth or checkpoint_*.pth found in {path}")


# --- model ----------------------------------------------------------------------------------


def prepare_torch_env() -> None:
    """Let unsupported MPS ops fall back to the CPU. Must run before torch is imported."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def select_device(preferred: str = "auto") -> str:
    """Resolve ``preferred`` to a concrete device, importing torch only for ``auto``."""
    if preferred != "auto":
        return pick_device(preferred, cuda=False, mps=False)
    prepare_torch_env()
    try:
        import torch
    except ImportError as exc:
        raise DependencyError(MISSING_STACK_MESSAGE) from exc
    return pick_device(
        preferred, cuda=torch.cuda.is_available(), mps=torch.backends.mps.is_available()
    )


def _to_numpy(wav: Any) -> np.ndarray:
    if hasattr(wav, "detach"):  # a torch tensor
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


class XttsEngine:
    """A loaded XTTS model bound to one speaker (its conditioning latents).

    ``model`` only needs an ``inference()`` method compatible with ``Xtts.inference``,
    which keeps this class testable without loading any weights.

    Meant to be created once and reused for many requests. It is safe to share between
    threads: XTTS keeps per-call state inside the model, so calls are run one at a time.
    """

    def __init__(
        self,
        model: Any,
        gpt_cond_latent: Any,
        speaker_embedding: Any,
        *,
        language: str = "en",
        temperature: float = DEFAULT_TEMPERATURE,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.model = model
        self.gpt_cond_latent = gpt_cond_latent
        self.speaker_embedding = speaker_embedding
        self.language = language
        self.temperature = temperature
        self.sample_rate = sample_rate
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize one short piece of text (see ``DEFAULT_MAX_CHARS``)."""
        with self._lock:
            start = time.perf_counter()
            # Xtts.inference already runs under torch.inference_mode(); no wrapper is needed.
            out = self.model.inference(
                text=text,
                language=self.language,
                gpt_cond_latent=self.gpt_cond_latent,
                speaker_embedding=self.speaker_embedding,
                temperature=self.temperature,
            )
            elapsed = time.perf_counter() - start
        wav = _to_numpy(out["wav"])
        seconds = len(wav) / self.sample_rate
        logger.debug(
            "Synthesized %.1f s of audio in %.1f s (%.2fx real time)",
            seconds,
            elapsed,
            seconds / elapsed if elapsed else 0.0,
        )
        return wav

    def synth_stream(
        self,
        text: str,
        pause_ms: float = DEFAULT_PAUSE_MS,
        max_chars: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Synthesize text of any length, yielding audio as soon as each chunk is ready.

        The text is split as in :func:`split_text` into chunks of at most ``max_chars``
        characters; ``None`` means ``default_max_chars(self.language)``, looked up on each
        call. Every yielded item is a mono float32 array at ``sample_rate``: the audio of
        one chunk, or ``pause_ms`` of silence between two chunks (never after the last one;
        a zero pause yields no silence). The next chunk is only synthesized when the caller
        asks for the next item, so playback or transmission can start after the first chunk
        instead of after the whole text, and stopping the iteration early skips the
        remaining synthesis.

        This is a generator, so nothing is checked or synthesized before the first item is
        requested: empty text, ``max_chars < 1`` or a negative ``pause_ms`` raise
        ``ValueError`` on the first iteration.
        """
        if max_chars is None:
            max_chars = default_max_chars(self.language)
        chunks = split_text(text, max_chars)
        if not chunks:
            raise ValueError("Nothing to synthesize: the text is empty")
        _silence(pause_ms, self.sample_rate)  # reject a bad pause before any synthesis
        logger.info("Synthesizing %d chunk(s)", len(chunks))
        for i, chunk in enumerate(chunks, 1):
            logger.debug("[chunk %d/%d] %s", i, len(chunks), chunk)
            yield self.synthesize(chunk)
            if i < len(chunks):
                silence = _silence(pause_ms, self.sample_rate)
                if len(silence):
                    yield silence

    def synth_long(
        self,
        text: str,
        pause_ms: float = DEFAULT_PAUSE_MS,
        max_chars: int | None = None,
    ) -> np.ndarray:
        """Synthesize text of any length and return it as one array (see ``synth_stream``)."""
        return np.concatenate(list(self.synth_stream(text, pause_ms, max_chars)))


def load_engine(
    checkpoint_dir: Path,
    reference_wavs: Sequence[str | Path],
    *,
    finetuned_checkpoint: Path | None = None,
    device: str = "auto",
    language: str = "en",
    temperature: float = DEFAULT_TEMPERATURE,
    checkpoint_url: str | None = None,
) -> XttsEngine:
    """Load the base XTTS-v2 model, or a fine-tuned checkpoint, and condition it on a voice.

    Missing base files are downloaded into ``checkpoint_dir``. A fine-tuned checkpoint only
    needs the base ``config.json`` and ``vocab.json``, so the large ``model.pth`` is
    fetched just for the base model.
    """
    if not reference_wavs:
        raise ValueError("At least one reference wav is needed to condition the voice")
    checkpoint_dir = Path(checkpoint_dir)
    device = select_device(device)
    url_kwargs = {"base_url": checkpoint_url} if checkpoint_url else {}
    checkpoints.ensure_base_checkpoints(
        checkpoint_dir,
        files=(
            checkpoints.FINETUNED_INFERENCE_FILES
            if finetuned_checkpoint
            else checkpoints.BASE_INFERENCE_FILES
        ),
        **url_kwargs,
    )

    prepare_torch_env()
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
    except ImportError as exc:
        raise DependencyError(MISSING_STACK_MESSAGE) from exc

    start = time.perf_counter()
    config = XttsConfig()
    config.load_json(str(checkpoint_dir / "config.json"))
    model = Xtts.init_from_config(config)
    if finetuned_checkpoint:
        logger.info("Loading fine-tuned checkpoint %s", finetuned_checkpoint)
        model.load_checkpoint(
            config,
            checkpoint_path=str(finetuned_checkpoint),
            vocab_path=str(checkpoint_dir / "vocab.json"),
            use_deepspeed=False,
        )
    else:
        logger.info("Loading base XTTS-v2 from %s", checkpoint_dir)
        model.load_checkpoint(config, checkpoint_dir=str(checkpoint_dir), use_deepspeed=False)
    # Move to the device first: the conditioning latents are computed on the model's device.
    model.to(device)
    model.eval()
    logger.info("Model loaded on %s in %.1f s", device, time.perf_counter() - start)

    logger.info("Computing voice conditioning from %d reference clip(s)", len(reference_wavs))
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[str(p) for p in reference_wavs]
    )
    return XttsEngine(
        model,
        gpt_cond_latent,
        speaker_embedding,
        language=language,
        temperature=temperature,
    )
