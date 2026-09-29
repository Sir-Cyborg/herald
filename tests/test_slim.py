"""Shrinking training checkpoints to their model weights (tiny synthetic checkpoints only)."""

import sys
import types
from pathlib import Path

import pytest

from herald.errors import CheckpointError, DependencyError
from herald.tts import slim


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def make_state(torch, *, with_training_state: bool = True) -> dict:
    """What the coqui Trainer saves, with tiny tensors: model weights + optimizer + extras."""
    generator = torch.Generator().manual_seed(0)
    weights = torch.randn(128, 128, generator=generator)
    state = {
        "model": {
            "gpt.weight": weights,
            "gpt.bias": torch.arange(128, dtype=torch.float16),
            "counter": torch.tensor(7),
        }
    }
    if with_training_state:
        state.update(
            optimizer={"state": {0: {"exp_avg": weights * 2, "exp_avg_sq": weights * 3}}},
            scheduler={"last_epoch": 4},
            scaler=None,
            config={"run_name": "test", "lr": 5e-6},
            step=250,
            epoch=1,
        )
    return state


@pytest.fixture
def checkpoint(torch, tmp_path) -> Path:
    path = tmp_path / "best_model.pth"
    torch.save(make_state(torch), path)
    return path


def load(torch, path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def files_in(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir())


class TestSlimCheckpoint:
    def test_keeps_only_the_model_weights(self, torch, checkpoint):
        original = load(torch, checkpoint)["model"]

        result = slim.slim_checkpoint(checkpoint)

        assert result.source == checkpoint
        assert result.path == checkpoint.with_name("best_model.slim.pth")
        assert result.already_slim is False
        slimmed = load(torch, result.path)
        assert set(slimmed) == {"model"}
        assert set(slimmed["model"]) == set(original)
        for key, tensor in original.items():
            assert slimmed["model"][key].dtype == tensor.dtype
            assert torch.equal(slimmed["model"][key], tensor)

    def test_the_slim_file_is_smaller_and_the_source_is_untouched(self, torch, checkpoint):
        before = checkpoint.read_bytes()

        result = slim.slim_checkpoint(checkpoint)

        assert result.size_before == len(before)
        assert result.size_after == result.path.stat().st_size
        assert result.size_after < result.size_before / 2
        assert checkpoint.read_bytes() == before
        assert files_in(checkpoint.parent) == ["best_model.pth", "best_model.slim.pth"]

    def test_explicit_destination(self, torch, checkpoint, tmp_path):
        dest = tmp_path / "out" / "voice.pth"  # its directory is created

        result = slim.slim_checkpoint(checkpoint, dest)

        assert result.path == dest
        assert set(load(torch, dest)) == {"model"}
        assert files_in(dest.parent) == ["voice.pth"]

    def test_an_already_slim_checkpoint_is_left_alone(self, torch, tmp_path):
        path = tmp_path / "slim.pth"
        torch.save(make_state(torch, with_training_state=False), path)

        result = slim.slim_checkpoint(path)

        assert result.already_slim is True
        assert result.path == path
        assert result.size_before == result.size_after == path.stat().st_size
        assert files_in(tmp_path) == ["slim.pth"]

    def test_replace_puts_the_slim_file_in_place_of_the_source(self, torch, checkpoint):
        original = load(torch, checkpoint)["model"]
        size = checkpoint.stat().st_size

        result = slim.slim_checkpoint(checkpoint, replace=True)

        assert result.path == checkpoint
        assert result.size_before == size
        assert result.size_after == checkpoint.stat().st_size < size / 2
        assert files_in(checkpoint.parent) == ["best_model.pth"]
        slimmed = load(torch, checkpoint)
        assert set(slimmed) == {"model"}
        assert all(torch.equal(slimmed["model"][k], v) for k, v in original.items())

    def test_dest_and_replace_are_mutually_exclusive(self, torch, checkpoint, tmp_path):
        with pytest.raises(ValueError, match="mutually exclusive"):
            slim.slim_checkpoint(checkpoint, tmp_path / "x.pth", replace=True)

    def test_the_destination_cannot_be_the_source(self, torch, checkpoint):
        with pytest.raises(ValueError, match="replace=True"):
            slim.slim_checkpoint(checkpoint, checkpoint)

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(CheckpointError, match="not found"):
            slim.slim_checkpoint(tmp_path / "nope.pth")

    def test_a_file_that_is_not_a_checkpoint(self, torch, tmp_path):
        path = tmp_path / "junk.pth"
        path.write_bytes(b"this is not a torch file at all")
        with pytest.raises(CheckpointError, match="could not be read"):
            slim.slim_checkpoint(path)
        assert files_in(tmp_path) == ["junk.pth"]

    @pytest.mark.parametrize(
        "content", [{"weights": {}}, {"model": "not a dict"}, [1, 2, 3]], ids=str
    )
    def test_a_torch_file_without_a_model_state_dict(self, torch, tmp_path, content):
        path = tmp_path / "other.pth"
        torch.save(content, path)
        with pytest.raises(CheckpointError, match="no 'model' state dict"):
            slim.slim_checkpoint(path)
        assert files_in(tmp_path) == ["other.pth"]

    def test_without_torch(self, tmp_path, monkeypatch):
        path = tmp_path / "x.pth"
        path.write_bytes(b"data")
        monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` fail
        with pytest.raises(DependencyError, match="torch"):
            slim.slim_checkpoint(path)


class TestFailures:
    """Whatever goes wrong: no partial file, and the source is never lost."""

    @pytest.fixture
    def broken_save(self, torch, monkeypatch):
        """Make ``torch.save`` write half a file, then fail with ``error``."""

        def install(error: BaseException):
            def save(obj, path, *args, **kwargs):
                Path(path).write_bytes(b"half a checkpoint")
                raise error

            monkeypatch.setattr(torch, "save", save)

        return install

    @pytest.mark.parametrize("replace", [False, True])
    def test_a_failed_write_leaves_nothing_behind(self, checkpoint, broken_save, replace):
        before = checkpoint.read_bytes()
        broken_save(RuntimeError("PytorchStreamWriter failed writing file: disk full"))

        with pytest.raises(CheckpointError, match="Could not write"):
            slim.slim_checkpoint(checkpoint, replace=replace)

        assert files_in(checkpoint.parent) == ["best_model.pth"]
        assert checkpoint.read_bytes() == before

    def test_an_interrupt_leaves_nothing_behind(self, checkpoint, broken_save):
        before = checkpoint.read_bytes()
        broken_save(KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            slim.slim_checkpoint(checkpoint, replace=True)

        assert files_in(checkpoint.parent) == ["best_model.pth"]
        assert checkpoint.read_bytes() == before

    @pytest.mark.parametrize("replace", [False, True])
    def test_a_slim_file_that_does_not_match_is_never_used(
        self, torch, checkpoint, monkeypatch, replace
    ):
        before = checkpoint.read_bytes()
        real_save = torch.save

        def lossy_save(obj, path, *args, **kwargs):  # e.g. weights silently changing dtype
            model = {k: v.double() if v.is_floating_point() else v for k, v in obj["model"].items()}
            real_save({"model": model}, path, *args, **kwargs)

        monkeypatch.setattr(torch, "save", lossy_save)

        with pytest.raises(CheckpointError, match="Verification failed"):
            slim.slim_checkpoint(checkpoint, replace=replace)

        assert files_in(checkpoint.parent) == ["best_model.pth"]
        assert checkpoint.read_bytes() == before

    def test_a_slim_file_with_missing_weights_is_never_used(self, torch, checkpoint, monkeypatch):
        real_save = torch.save

        def truncated_save(obj, path, *args, **kwargs):
            (first, *_), model = list(obj["model"]), dict(obj["model"])
            del model[first]
            real_save({"model": model}, path, *args, **kwargs)

        monkeypatch.setattr(torch, "save", truncated_save)

        with pytest.raises(CheckpointError, match="different set of weights"):
            slim.slim_checkpoint(checkpoint)

        assert files_in(checkpoint.parent) == ["best_model.pth"]

    def test_not_enough_disk_space_is_reported_before_writing(self, torch, checkpoint, monkeypatch):
        monkeypatch.setattr(
            slim.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=1000)
        )

        with pytest.raises(CheckpointError, match="Not enough disk space"):
            slim.slim_checkpoint(checkpoint)

        assert files_in(checkpoint.parent) == ["best_model.pth"]


class TestFormatSize:
    @pytest.mark.parametrize(
        ("n_bytes", "text"),
        [
            (0, "0 bytes"),
            (812, "812 bytes"),
            (12_345, "12.35 KB"),
            (12_000_000, "12.00 MB"),
            (5_607_906_782, "5.61 GB"),
            (2_500_000_000_000, "2500.00 GB"),
        ],
    )
    def test_format_size(self, n_bytes, text):
        assert slim.format_size(n_bytes) == text
