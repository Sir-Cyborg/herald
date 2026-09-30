"""Summarise the audio clips of a dataset and flag the ones XTTS-v2 fine-tuning skips.

The trainer never complains about a bad clip: it silently moves on to the next one. So a
dataset made of clips that are too short or too long trains on much less data than it
seems, and nobody finds out. :func:`inspect_audio` reads only the WAV headers (stdlib
``wave``, no decoding), so it is fast for thousands of files, and reports what would be
skipped before hours of training are spent.

The limits below are those of the XTTS-v2 GPT trainer (``TTS.tts.layers.xtts.trainer``).
"""

from __future__ import annotations

import logging
import wave
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_RATE_TRAIN = 22_050  # the trainer resamples every clip to this rate
MAX_WAV_SAMPLES = 255_995  # longer clips (at SAMPLE_RATE_TRAIN) are skipped
MIN_CLIP_SECONDS = 0.5  # shorter clips are skipped
MIN_CONDITIONING_SECONDS = 3.0  # shorter clips give a shorter voice reference to condition on
MAX_CLIP_SECONDS = MAX_WAV_SAMPLES / SAMPLE_RATE_TRAIN  # ~11.6 s


@dataclass(frozen=True)
class AudioStats:
    """Header statistics of the readable clips of a dataset.

    Paths are given as ``inspect_audio`` was told to report them.
    """

    clips_checked: int  # clips whose header could be read; the fields below describe these
    unreadable: tuple[str, ...]  # not a WAV that ``wave`` can read (see inspect_audio)
    total_seconds: float
    sample_rates: dict[int, int]  # rate -> number of clips
    channels: dict[int, int]  # channel count -> number of clips
    too_short: tuple[str, ...]  # shorter than MIN_CLIP_SECONDS: the trainer skips them
    too_long: tuple[str, ...]  # longer than MAX_CLIP_SECONDS: the trainer skips them
    weak_reference: int  # from MIN_CLIP_SECONDS up to 3 s: usable, but teach the voice less


def inspect_audio(
    paths: Sequence[Path],
    *,
    root: Path | None = None,
    min_seconds: float = MIN_CLIP_SECONDS,
    max_seconds: float = MAX_CLIP_SECONDS,
    reference_seconds: float = MIN_CONDITIONING_SECONDS,
) -> AudioStats:
    """Read the header of every clip in ``paths`` and summarise them.

    A clip counts as ``unreadable`` if ``wave`` cannot open it: a truncated or damaged
    file, but also anything that is not integer PCM WAV (MP3, FLAC, float WAV, ...). Those
    are not necessarily broken, they are just not inspected and left out of the statistics.

    ``root`` makes the reported paths relative to it (when they are inside it).
    """
    unreadable: list[str] = []
    too_short: list[str] = []
    too_long: list[str] = []
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    weak_reference = 0
    total_seconds = 0.0
    clips_checked = 0

    for path in paths:
        path = Path(path)
        label = str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)
        try:
            with wave.open(str(path), "rb") as clip:
                rate, n_channels, frames = (
                    clip.getframerate(),
                    clip.getnchannels(),
                    clip.getnframes(),
                )
        except (wave.Error, EOFError, OSError) as exc:
            logger.debug("%s: not inspected (%s)", path, exc)
            unreadable.append(label)
            continue
        if rate <= 0:
            unreadable.append(label)
            continue

        seconds = frames / rate
        clips_checked += 1
        total_seconds += seconds
        sample_rates[rate] += 1
        channels[n_channels] += 1
        if seconds < min_seconds:
            too_short.append(label)
        elif seconds > max_seconds:
            too_long.append(label)
        elif seconds < reference_seconds:
            weak_reference += 1

    return AudioStats(
        clips_checked=clips_checked,
        unreadable=tuple(unreadable),
        total_seconds=total_seconds,
        sample_rates=dict(sorted(sample_rates.items())),
        channels=dict(sorted(channels.items())),
        too_short=tuple(too_short),
        too_long=tuple(too_long),
        weak_reference=weak_reference,
    )
