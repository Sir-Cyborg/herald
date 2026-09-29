"""Real-model smoke tests. They need the downloaded weights, so CI skips them.

Run locally with:  pytest -m integration
They skip themselves when the dataset or the weights are not where herald expects them.
"""

import numpy as np
import pytest

from herald.config import Settings
from herald.dataset import metadata
from herald.tts import engine

pytestmark = pytest.mark.integration

SENTENCE = "Hello there. This is a short test of the cloned voice."


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings.from_env()


@pytest.fixture(scope="module")
def references(settings) -> list[str]:
    csv_path = settings.dataset_dir / metadata.METADATA_FILE
    if not csv_path.is_file():
        pytest.skip(f"no dataset at {settings.dataset_dir}")
    return metadata.pick_reference_wavs(csv_path, settings.dataset_dir, n=1)


def check_speech(eng: engine.XttsEngine, tmp_path) -> None:
    wav = eng.synth_long(SENTENCE, pause_ms=100)
    seconds = len(wav) / eng.sample_rate
    assert 0.5 < seconds < 30, f"implausible duration: {seconds:.1f}s"
    assert np.isfinite(wav).all()
    assert np.abs(wav).max() > 0.01, "the output is silent"
    assert engine.save_wav(tmp_path / "smoke.wav", wav, eng.sample_rate).stat().st_size > 1000


def test_base_model(settings, references, tmp_path):
    if not (settings.checkpoint_dir / "model.pth").is_file():
        pytest.skip("base weights not downloaded (herald download-checkpoints)")
    eng = engine.load_engine(
        settings.checkpoint_dir, references, device=settings.device, language=settings.language
    )
    check_speech(eng, tmp_path)


def test_finetuned_model(settings, references, tmp_path):
    voices = sorted(settings.models_dir.glob("*/best_model*.pth"))
    if not voices:
        pytest.skip(f"no fine-tuned voice under {settings.models_dir} (<speaker>/best_model.pth)")
    if not (settings.checkpoint_dir / "config.json").is_file():
        pytest.skip("base config.json/vocab.json not downloaded")
    checkpoint = engine.resolve_finetuned_checkpoint(voices[-1].parent)
    eng = engine.load_engine(
        settings.checkpoint_dir,
        references,
        finetuned_checkpoint=checkpoint,
        device=settings.device,
        language=settings.language,
    )
    check_speech(eng, tmp_path)
