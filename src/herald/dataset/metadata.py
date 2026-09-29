"""Read a dataset's metadata file, locate its audio and split it into train and eval.

A dataset directory looks like::

    <dataset_dir>/
      audio/000000.wav ...
      metadata.csv          # one row per clip

``metadata.csv`` is the only file: reference clips are picked from it for synthesis and
fine-tuning derives its train/eval split from it (:func:`split_items`), so there are no
pre-split files to keep in sync.

The file has two columns, ``audio_file`` and ``text``, with or without a header row and
separated by ``|``, a tab or a comma.
"""

from __future__ import annotations

import csv
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, TypeVar

from herald.errors import ConfigError, MetadataError

logger = logging.getLogger(__name__)

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")
AUDIO_SUBDIR = "audio"
METADATA_FILE = "metadata.csv"

# Checked in this order against the first line, so "|" wins over "," in transcripts
# that contain commas.
_DELIMITERS = ("|", "\t", ",")


T = TypeVar("T")


class MetadataItem(TypedDict):
    audio_file: str
    text: str


def _not_utf8(path: Path) -> MetadataError:
    return MetadataError(f"{path} is not valid UTF-8 text; save it as UTF-8")


def sniff_delimiter(path: Path) -> str:
    """Guess the column delimiter from the first line of ``path``."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            first_line = f.readline()
    except UnicodeDecodeError:
        raise _not_utf8(path) from None
    for delim in _DELIMITERS:
        if delim in first_line:
            return delim
    return ","


def _read_rows(path: Path, delim: str) -> list[list[str]]:
    """The non-blank rows of a delimited file, one per physical line.

    Quoted fields are honoured (a transcript may be written as ``"Well, well."``), but a
    field that starts with a quote and never closes it would silently swallow the
    following lines, so a record that spans several lines is an error.
    """
    rows: list[list[str]] = []
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=delim)
            while True:
                first_line = reader.line_num + 1
                try:
                    row = next(reader)
                except StopIteration:
                    break
                if reader.line_num > first_line:
                    raise MetadataError(
                        f"{path}: the row at line {first_line} runs over lines "
                        f"{first_line}-{reader.line_num}: a quote is not closed. "
                        "Fix the quote in that row (a literal quote is written twice)."
                    )
                if row:
                    rows.append(row)
    except UnicodeDecodeError:
        raise _not_utf8(path) from None
    except csv.Error as exc:
        raise MetadataError(f"{path}: {exc}") from exc
    return rows


def load_metadata(csv_path: Path) -> list[MetadataItem]:
    """Read a two-column ``audio_file|text`` file.

    The delimiter is auto-detected. A header row is assumed unless the first cell of the
    first row looks like an audio file name (ends in ``.wav``, ``.mp3``, ...).
    Rows that are too short or have an empty audio path or text are skipped. A row whose
    quoted field runs over several lines, or a file that is not UTF-8, is a
    :class:`MetadataError`.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise MetadataError(f"Metadata file not found: {csv_path}")

    delim = sniff_delimiter(csv_path)
    rows = _read_rows(csv_path, delim)
    if not rows:
        return []

    has_header = not rows[0][0].strip().lower().endswith(AUDIO_EXTS)
    if has_header:
        header = [c.strip().lower() for c in rows[0]]
        try:
            col_audio = next(
                i for i, c in enumerate(header) if any(k in c for k in ("audio", "file", "path"))
            )
            col_text = next(i for i, c in enumerate(header) if "text" in c)
        except StopIteration:
            raise MetadataError(
                f"{csv_path}: cannot find an audio column (audio/file/path) and a text column "
                f"in header {rows[0]!r}"
            ) from None
        data_rows = rows[1:]
    else:
        col_audio, col_text = 0, 1
        data_rows = rows

    items: list[MetadataItem] = []
    skipped = 0
    for row in data_rows:
        if len(row) <= max(col_audio, col_text):
            skipped += 1
            continue
        audio_file, text = row[col_audio].strip(), row[col_text].strip()
        if not audio_file or not text:
            skipped += 1
            continue
        items.append({"audio_file": audio_file, "text": text})

    logger.info(
        "%s: %d rows read, %d skipped (delimiter=%r, header=%s)",
        csv_path.name,
        len(items),
        skipped,
        delim,
        has_header,
    )
    return items


def resolve_audio_path(audio_file: str | Path, dataset_dir: Path) -> Path:
    """Find ``audio_file`` for a dataset.

    Tried in order: the path itself (only if absolute), relative to ``dataset_dir``,
    relative to ``dataset_dir/audio`` and finally just the file name inside
    ``dataset_dir/audio``. Relative paths are never resolved against the current
    directory, so a stray ``audio/`` folder there cannot shadow the dataset.
    """
    p = Path(audio_file)
    dataset_dir = Path(dataset_dir)
    candidates = [p] if p.is_absolute() else []
    candidates += [
        dataset_dir / p,
        dataset_dir / AUDIO_SUBDIR / p,
        dataset_dir / AUDIO_SUBDIR / p.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Audio file not found: {audio_file} (tried: {tried})")


def pick_reference_wavs(
    metadata_csv: Path, dataset_dir: Path, n: int = 3, seed: int = 42
) -> list[str]:
    """Pick ``n`` reproducible reference clips (as absolute paths) from a metadata file."""
    items = load_metadata(metadata_csv)
    if not items:
        raise MetadataError(f"{metadata_csv}: no usable rows to pick reference clips from")
    random.Random(seed).shuffle(items)
    return [str(resolve_audio_path(item["audio_file"], dataset_dir)) for item in items[:n]]


def split_items(
    items: Sequence[T], eval_fraction: float, seed: int = 42
) -> tuple[list[T], list[T]]:
    """Split ``items`` into ``(train, eval)`` reproducibly.

    The items are shuffled with ``random.Random(seed)`` (the input is not modified) and the
    first ``round(len(items) * eval_fraction)`` become the eval set, so the same input,
    fraction and seed always give the same split and the two sets never overlap.
    Whenever ``eval_fraction`` is positive and there are at least two items, the eval set
    has at least one item; the train set is never left empty. A fraction of 0 gives an
    empty eval set. ``eval_fraction`` must be in ``[0, 1)``.
    """
    if not 0 <= eval_fraction < 1:
        raise ConfigError(f"eval_fraction must be in [0, 1), got {eval_fraction}")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n_eval = round(len(shuffled) * eval_fraction)
    if eval_fraction > 0:
        n_eval = max(n_eval, 1)
    n_eval = max(0, min(n_eval, len(shuffled) - 1))  # keep at least one training item
    return shuffled[n_eval:], shuffled[:n_eval]
