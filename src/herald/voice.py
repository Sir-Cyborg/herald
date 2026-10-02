"""Getting a voice: pick the reference clips and load the XTTS model.

Shared by ``herald synthesize`` and ``herald chat``. Heavy imports (torch, coqui-tts) happen
inside :func:`load_engine`, so importing this module is cheap.
"""

from __future__ import annotations

import argparse
import logging

from herald.config import Settings
from herald.errors import ConfigError

logger = logging.getLogger(__name__)


def load_engine(args: argparse.Namespace, settings: Settings):
    """Pick reference clips and load the base or fine-tuned model."""
    from herald.dataset import metadata
    from herald.tts import engine

    if args.reference_wav:
        for wav in args.reference_wav:
            if not wav.is_file():
                raise FileNotFoundError(f"Reference wav not found: {wav}")
        references = [str(w) for w in args.reference_wav]
    else:
        metadata_csv = args.dataset_dir / metadata.METADATA_FILE
        if not metadata_csv.is_file():
            raise ConfigError(
                f"No dataset found at {args.dataset_dir} (no {metadata.METADATA_FILE}): pass "
                "--reference-wav CLIP.wav (repeatable, a few seconds of the voice) "
                "or --dataset-dir DIR"
            )
        references = metadata.pick_reference_wavs(
            metadata_csv, args.dataset_dir, n=args.num_references, seed=args.seed
        )
    logger.info("Reference clips: %s", ", ".join(references))

    finetuned = engine.resolve_finetuned_checkpoint(args.checkpoint) if args.checkpoint else None
    return engine.load_engine(
        args.checkpoint_dir,
        references,
        finetuned_checkpoint=finetuned,
        device=args.device,
        language=args.language,
        temperature=args.temperature,
        checkpoint_url=settings.checkpoint_url,
    )
