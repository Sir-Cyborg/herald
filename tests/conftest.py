from pathlib import Path

import pytest


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """A tiny dataset: 5 clips (empty files are enough, nothing decodes them) and metadata."""
    root = tmp_path / "speaker"
    (root / "audio").mkdir(parents=True)
    lines = []
    for i in range(5):
        (root / "audio" / f"{i:06d}.wav").write_bytes(b"RIFF")
        lines.append(f"audio/{i:06d}.wav|Sentence number {i}, with a comma.")
    (root / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _keep_torch_env_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """`prepare_torch_env` sets PYTORCH_ENABLE_MPS_FALLBACK for the whole process; undo it
    after each test so it cannot leak into the ones that follow."""
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK")
