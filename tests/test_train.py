"""Fine-tuning parameters, samples, train/eval split, trainer wiring, model promotion.

Nothing here trains: the torch / coqui-tts stack is replaced by fakes.
"""

import inspect
import logging
import sys
import types
import wave
from datetime import datetime
from pathlib import Path

import pytest

from herald.config import DEFAULT_CHECKPOINT_URL
from herald.dataset.audio_stats import AudioStats
from herald.dataset.metadata import METADATA_FILE, load_metadata, split_items
from herald.errors import CheckpointError, ConfigError, MetadataError
from herald.tts import slim, train


def write_dataset(root: Path, clips: int) -> Path:
    """A dataset of ``clips`` clips (empty files are enough, nothing decodes them)."""
    (root / "audio").mkdir(parents=True)
    lines = []
    for i in range(clips):
        (root / "audio" / f"{i:06d}.wav").write_bytes(b"RIFF")
        lines.append(f"audio/{i:06d}.wav|Sentence number {i}, with a comma.")
    (root / METADATA_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def write_wav(path: Path, seconds: float, *, rate: int = 22_050, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as clip:
        clip.setnchannels(channels)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(b"\x00\x00" * channels * round(seconds * rate))


@pytest.fixture
def train_dataset(tmp_path: Path) -> Path:
    """20 clips, enough for a 10 % eval split of 2 clips."""
    return write_dataset(tmp_path / "speaker", 20)


@pytest.fixture
def big_dataset(tmp_path: Path) -> Path:
    """100 clips: a 10 % split (90 / 10) is larger than the smoke limits allow for train."""
    return write_dataset(tmp_path / "big", 100)


def params_for(dataset: Path, tmp_path: Path, **kwargs) -> train.TrainParams:
    kwargs.setdefault("models_dir", tmp_path / "models")
    return train.make_params(
        dataset_dir=dataset, checkpoint_dir=tmp_path / "ckpt", runs_dir=tmp_path / "runs", **kwargs
    )


class TestMakeParams:
    def test_full_preset_matches_the_original_run(self, dataset_dir, tmp_path):
        p = params_for(dataset_dir, tmp_path)
        assert (p.epochs, p.batch_size, p.eval_batch_size, p.grad_accum_steps) == (5, 2, 2, 16)
        assert (p.lr, p.num_workers) == (5e-6, 2)
        assert (p.print_step, p.plot_step, p.save_step, p.save_n_checkpoints) == (25, 100, 250, 1)
        assert (p.max_train_samples, p.max_eval_samples) == (None, None)

    def test_smoke_preset_is_small(self, dataset_dir, tmp_path):
        p = params_for(dataset_dir, tmp_path, preset="smoke")
        assert (p.epochs, p.batch_size, p.grad_accum_steps) == (1, 1, 4)
        assert (p.max_train_samples, p.max_eval_samples) == (60, 20)
        assert p.run_name.endswith("_smoke")

    def test_directories_and_split_defaults(self, dataset_dir, tmp_path):
        p = params_for(dataset_dir, tmp_path)
        assert p.dataset_dir == dataset_dir
        assert p.models_dir == tmp_path / "models"
        assert (p.metadata_file, p.eval_fraction, p.seed) == ("metadata.csv", 0.1, 42)

    def test_explicit_split_settings(self, dataset_dir, tmp_path):
        p = params_for(dataset_dir, tmp_path, metadata_file="m.csv", eval_fraction=0.2, seed=7)
        assert (p.metadata_file, p.eval_fraction, p.seed) == ("m.csv", 0.2, 7)

    def test_names_default_to_the_dataset_directory(self, dataset_dir, tmp_path):
        p = params_for(dataset_dir, tmp_path)
        assert p.speaker_name == "speaker"
        assert p.model_name == "speaker"
        assert p.run_name == "speaker_full"
        assert p.project_name == "speaker_xtts"

    def test_a_smoke_run_gets_its_own_model_directory(self, dataset_dir, tmp_path):
        p = params_for(dataset_dir, tmp_path, preset="smoke", speaker_name="frieren")
        assert (p.speaker_name, p.model_name) == ("frieren", "frieren_smoke")

    def test_checkpoints_are_deleted_after_a_run_unless_asked_otherwise(
        self, dataset_dir, tmp_path
    ):
        assert params_for(dataset_dir, tmp_path).keep_checkpoints is False
        assert params_for(dataset_dir, tmp_path, keep_checkpoints=True).keep_checkpoints is True

    def test_checkpoint_url(self, dataset_dir, tmp_path):
        assert params_for(dataset_dir, tmp_path).checkpoint_url == DEFAULT_CHECKPOINT_URL
        p = params_for(dataset_dir, tmp_path, checkpoint_url="https://mirror.test/xtts")
        assert p.checkpoint_url == "https://mirror.test/xtts"

    def test_speaker_name_of_a_relative_dataset_directory(self, dataset_dir, tmp_path, monkeypatch):
        monkeypatch.chdir(dataset_dir)
        assert params_for(Path("."), tmp_path).speaker_name == "speaker"

    def test_explicit_names(self, dataset_dir, tmp_path):
        p = params_for(
            dataset_dir,
            tmp_path,
            speaker_name="frieren",
            model_name="frieren_v2",
            run_name="try1",
            project_name="proj",
        )
        assert (p.speaker_name, p.model_name) == ("frieren", "frieren_v2")
        assert (p.run_name, p.project_name) == ("try1", "proj")

    def test_overrides_win_and_none_is_ignored(self, dataset_dir, tmp_path):
        p = params_for(dataset_dir, tmp_path, preset="smoke", epochs=3, batch_size=None, lr=1e-5)
        assert (p.epochs, p.batch_size, p.lr) == (3, 1, 1e-5)

    def test_eval_batch_size_follows_the_batch_size(self, dataset_dir, tmp_path):
        assert params_for(dataset_dir, tmp_path, batch_size=6).eval_batch_size == 6
        assert (
            params_for(dataset_dir, tmp_path, batch_size=6, eval_batch_size=1).eval_batch_size == 1
        )

    def test_unknown_preset(self, dataset_dir, tmp_path):
        with pytest.raises(ConfigError, match="Unknown preset"):
            params_for(dataset_dir, tmp_path, preset="huge")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("epochs", 0),
            ("batch_size", 0),
            ("grad_accum_steps", 0),
            ("lr", 0),
            ("lr", float("inf")),
            ("lr", float("nan")),
            ("num_workers", -1),
            ("print_step", 0),
            ("plot_step", 0),
            ("save_step", 0),
            ("eval_fraction", -0.1),
            ("eval_fraction", 1),
            ("max_train_samples", 0),
            ("max_eval_samples", -1),
        ],
    )
    def test_invalid_values(self, dataset_dir, tmp_path, field, value):
        with pytest.raises(ConfigError, match=field):
            params_for(dataset_dir, tmp_path, **{field: value})

    @pytest.mark.parametrize("name", [".", "..", "a/b", "/abs", "a\\b"])
    @pytest.mark.parametrize("field", ["speaker", "model"])
    def test_names_must_be_plain_directory_names(self, dataset_dir, tmp_path, field, name):
        with pytest.raises(ConfigError, match=f"{field} name"):
            params_for(dataset_dir, tmp_path, **{f"{field}_name": name})


class TestBuildSamples:
    def test_samples_have_the_keys_coqui_expects(self, train_dataset, tmp_path):
        params = params_for(train_dataset, tmp_path, speaker_name="frieren", language="ja")
        items = [{"audio_file": "audio/000003.wav", "text": "Hello, there."}]

        (sample,) = train.build_samples(items, params)

        assert sample == {
            "audio_file": str((train_dataset / "audio" / "000003.wav").resolve()),
            "text": "Hello, there.",
            "speaker_name": "frieren",
            "language": "ja",
            "root_path": str(train_dataset.resolve()),
            "audio_unique_name": "frieren#audio/000003",
        }

    def test_an_audio_file_outside_the_dataset_is_fine(self, train_dataset, tmp_path):
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"RIFF")
        items = [{"audio_file": str(outside), "text": "Hi"}]

        (sample,) = train.build_samples(items, params_for(train_dataset, tmp_path))

        assert sample["audio_file"] == str(outside.resolve())

    def test_missing_audio_raises(self, train_dataset, tmp_path):
        items = [{"audio_file": "audio/ghost.wav", "text": "Hi"}]
        with pytest.raises(FileNotFoundError, match="ghost.wav"):
            train.build_samples(items, params_for(train_dataset, tmp_path))


class TestValidateDataset:
    def test_counts_rows_of_the_split(self, train_dataset, tmp_path):
        report = train.validate_dataset(params_for(train_dataset, tmp_path))
        assert (report.train_rows, report.eval_rows, report.missing_audio) == (18, 2, ())

    def test_eval_fraction_changes_the_split(self, train_dataset, tmp_path):
        report = train.validate_dataset(params_for(train_dataset, tmp_path, eval_fraction=0.25))
        assert (report.train_rows, report.eval_rows) == (15, 5)

    def test_no_eval_rows_with_a_zero_fraction(self, train_dataset, tmp_path):
        report = train.validate_dataset(params_for(train_dataset, tmp_path, eval_fraction=0))
        assert (report.train_rows, report.eval_rows) == (20, 0)

    def test_reports_missing_audio_sorted(self, train_dataset, tmp_path):
        (train_dataset / "audio" / "000015.wav").unlink()
        (train_dataset / "audio" / "000002.wav").unlink()
        report = train.validate_dataset(params_for(train_dataset, tmp_path))
        assert report.missing_audio == ("audio/000002.wav", "audio/000015.wav")

    def test_counts_are_those_of_a_smoke_run(self, big_dataset, tmp_path):
        report = train.validate_dataset(params_for(big_dataset, tmp_path, preset="smoke"))
        assert (report.train_rows, report.eval_rows) == (60, 10)  # split 90 / 10, smoke max 60 / 20

    def test_only_the_rows_a_smoke_run_uses_are_checked(self, big_dataset, tmp_path):
        params = params_for(big_dataset, tmp_path, preset="smoke")
        train_rows, eval_rows = split_items(load_metadata(big_dataset / METADATA_FILE), 0.1, 42)
        used = {r["audio_file"] for r in [*train_rows[:60], *eval_rows]}
        unused = next(r["audio_file"] for r in train_rows[60:])
        assert unused not in used
        (big_dataset / unused).unlink()
        assert train.validate_dataset(params).missing_audio == ()

        (big_dataset / train_rows[0]["audio_file"]).unlink()
        assert train.validate_dataset(params).missing_audio == (train_rows[0]["audio_file"],)

    def test_metadata_file_is_configurable(self, train_dataset, tmp_path):
        (train_dataset / "other.csv").write_text("audio/000000.wav|One\naudio/000001.wav|Two\n")
        report = train.validate_dataset(
            params_for(train_dataset, tmp_path, metadata_file="other.csv")
        )
        assert (report.train_rows, report.eval_rows) == (1, 1)

    def test_metadata_file_is_required(self, train_dataset, tmp_path):
        (train_dataset / METADATA_FILE).unlink()
        with pytest.raises(MetadataError, match="not found"):
            train.validate_dataset(params_for(train_dataset, tmp_path))

    def test_empty_metadata_is_an_error(self, train_dataset, tmp_path):
        (train_dataset / METADATA_FILE).write_text("")
        with pytest.raises(MetadataError, match="no usable rows"):
            train.validate_dataset(params_for(train_dataset, tmp_path))


class TestDatasetAudioReport:
    @pytest.fixture
    def wav_dataset(self, train_dataset) -> Path:
        """20 real WAV clips: 17 good ones and a too short, a too long and a short-ish one."""
        for i in range(20):
            write_wav(train_dataset / "audio" / f"{i:06d}.wav", 4)
        write_wav(train_dataset / "audio" / "000000.wav", 0.3)
        write_wav(train_dataset / "audio" / "000001.wav", 12)
        write_wav(train_dataset / "audio" / "000002.wav", 2, rate=44_100, channels=2)
        return train_dataset

    def test_reports_what_the_trainer_would_skip(self, wav_dataset, tmp_path):
        report = train.validate_dataset(params_for(wav_dataset, tmp_path))

        audio = report.audio
        assert audio is not None
        assert audio.clips_checked == 20
        assert audio.too_short == ("audio/000000.wav",)  # relative to the dataset
        assert audio.too_long == ("audio/000001.wav",)
        assert audio.weak_reference == 1
        assert audio.sample_rates == {22_050: 19, 44_100: 1}
        assert audio.channels == {1: 19, 2: 1}
        assert audio.total_seconds == pytest.approx(17 * 4 + 0.3 + 12 + 2)
        assert (report.train_rows, report.eval_rows, report.missing_audio) == (18, 2, ())

    def test_only_the_rows_a_smoke_run_uses_are_inspected(self, big_dataset, tmp_path):
        # The clips of big_dataset are not WAV files, so each inspected clip is "unreadable".
        report = train.validate_dataset(params_for(big_dataset, tmp_path, preset="smoke"))

        assert report.audio is not None
        assert len(report.audio.unreadable) == report.train_rows + report.eval_rows == 70

    def test_missing_files_are_not_counted_as_unreadable(self, train_dataset, tmp_path):
        (train_dataset / "audio" / "000005.wav").unlink()

        report = train.validate_dataset(params_for(train_dataset, tmp_path))

        assert report.missing_audio == ("audio/000005.wav",)
        assert report.audio is not None and len(report.audio.unreadable) == 19

    def test_counts_transcripts_that_may_be_too_long(self, train_dataset, tmp_path):
        rows = [f"audio/{i:06d}.wav|{'a' * n}" for i, n in enumerate([200, 201, 500, 10])]
        (train_dataset / METADATA_FILE).write_text("\n".join(rows) + "\n")

        report = train.validate_dataset(params_for(train_dataset, tmp_path))

        assert report.long_text_rows == 2

    def test_long_transcripts_outside_the_used_rows_are_ignored(self, big_dataset, tmp_path):
        params = params_for(big_dataset, tmp_path, preset="smoke")
        items = load_metadata(big_dataset / METADATA_FILE)
        train_rows, _ = split_items(items, 0.1, 42)
        unused = train_rows[60]["audio_file"]  # the smoke preset trains on the first 60 only
        rows = [
            f"{i['audio_file']}|{'a' * 300 if i['audio_file'] == unused else 'ok'}" for i in items
        ]
        (big_dataset / METADATA_FILE).write_text("\n".join(rows) + "\n")

        assert train.validate_dataset(params).long_text_rows == 0

    def test_run_training_logs_the_warnings(
        self, wav_dataset, tmp_path, monkeypatch, mocker, caplog
    ):
        run_dir = tmp_path / "runs" / "run"
        stack = make_fake_stack([], run_dir)
        monkeypatch.setattr(train, "_import_training_stack", lambda: stack)
        monkeypatch.setattr(train.slim, "slim_checkpoint", fake_slim)
        mocker.patch("herald.tts.checkpoints.ensure_base_checkpoints")

        with caplog.at_level(logging.WARNING, logger="herald.tts.train"):
            train.run_training(params_for(wav_dataset, tmp_path))

        assert "1 clip(s) are shorter than 0.5 s" in caplog.text
        assert "1 clip(s) are longer than 11.6 s" in caplog.text


class TestDatasetReportWarnings:
    def audio(self, **kwargs) -> AudioStats:
        values = dict(
            clips_checked=10,
            unreadable=(),
            total_seconds=60.0,
            sample_rates={22_050: 10},
            channels={1: 10},
            too_short=(),
            too_long=(),
            weak_reference=0,
        )
        return AudioStats(**{**values, **kwargs})

    def test_a_clean_dataset_has_no_warnings(self):
        assert train.DatasetReport(9, 1, (), self.audio()).warnings() == []

    def test_a_report_without_audio_statistics(self):
        assert train.DatasetReport(9, 1, ()).warnings() == []
        assert train.DatasetReport(9, 1, ()).audio is None
        assert train.DatasetReport(9, 1, ()).long_text_rows == 0

    def test_every_kind_of_problem_gets_a_line(self):
        audio = self.audio(
            unreadable=("a.flac",),
            too_short=("s1.wav", "s2.wav"),
            too_long=("l1.wav",),
            weak_reference=4,
        )

        lines = train.DatasetReport(9, 1, (), audio, long_text_rows=3).warnings()

        assert len(lines) == 5
        joined = "\n".join(lines)
        assert "1 audio file(s) are not WAV files" in joined
        assert (
            "2 clip(s) are shorter than 0.5 s: the trainer skips them (e.g. s1.wav, s2.wav)"
            in joined
        )
        assert "1 clip(s) are longer than 11.6 s: the trainer skips them" in joined
        assert "4 clip(s) are shorter than 3 s: usable, but they teach the voice less" in joined
        assert "3 transcript(s) are longer than 200 characters: the trainer may skip them" in joined

    def test_only_a_few_examples_are_listed(self):
        audio = self.audio(too_short=tuple(f"{i}.wav" for i in range(10)))
        (line,) = train.DatasetReport(9, 1, (), audio).warnings()
        assert "10 clip(s)" in line and "(e.g. 0.wav, 1.wav, 2.wav)" in line


class TestConfigKwargs:
    def test_trainer_config_for_the_full_preset(self, dataset_dir, tmp_path):
        kwargs = train.trainer_config_kwargs(params_for(dataset_dir, tmp_path))
        assert kwargs["output_path"] == str(tmp_path / "runs")
        assert kwargs["run_name"] == "speaker_full"
        assert kwargs["epochs"] == 5
        assert kwargs["batch_size"] == kwargs["eval_batch_size"] == 2
        assert kwargs["num_loader_workers"] == kwargs["num_eval_loader_workers"] == 2
        assert kwargs["lr"] == 5e-6
        assert kwargs["optimizer"] == "AdamW"
        assert kwargs["lr_scheduler_params"]["milestones"] == [50_000, 150_000, 300_000]
        assert kwargs["test_sentences"] == []
        assert kwargs["run_eval"] is True
        assert "model_args" not in kwargs and "audio" not in kwargs

    def test_evaluation_can_be_switched_off(self, dataset_dir, tmp_path):
        kwargs = train.trainer_config_kwargs(params_for(dataset_dir, tmp_path), run_eval=False)
        assert kwargs["run_eval"] is False

    def test_model_args_point_into_the_checkpoint_dir(self, tmp_path):
        kwargs = train.model_args_kwargs(tmp_path)
        assert kwargs["xtts_checkpoint"] == str(tmp_path / "model.pth")
        assert kwargs["dvae_checkpoint"] == str(tmp_path / "dvae.pth")
        assert kwargs["mel_norm_file"] == str(tmp_path / "mel_stats.pth")
        assert kwargs["tokenizer_file"] == str(tmp_path / "vocab.json")
        assert kwargs["max_wav_length"] == 255_995
        assert kwargs["min_conditioning_length"] == 66_150
        assert kwargs["max_text_length"] == 200
        assert train.AUDIO_CONFIG_KWARGS["sample_rate"] == 22_050


class FrozenDatetime:
    """Stands in for ``datetime`` in the train module: always the same instant."""

    @staticmethod
    def now() -> datetime:
        return datetime(2026, 9, 29, 20, 30, 15)


def fake_slim(source, dest=None, *, replace=False):
    """A stand-in for :func:`slim.slim_checkpoint`: b"slim:" + the checkpoint's bytes."""
    source, dest = Path(source), Path(dest)
    data = b"slim:" + source.read_bytes()
    dest.write_bytes(data)
    return slim.SlimResult(source, dest, source.stat().st_size, len(data))


@pytest.fixture
def fake_slimmer(monkeypatch):
    monkeypatch.setattr(train.slim, "slim_checkpoint", fake_slim)


@pytest.fixture
def slim_fails(monkeypatch):
    def fail(source, dest=None, *, replace=False):
        raise CheckpointError("cannot slim")

    monkeypatch.setattr(train.slim, "slim_checkpoint", fail)


@pytest.mark.usefixtures("fake_slimmer")
class TestPromoteBestModel:
    @pytest.fixture
    def run_dir(self, tmp_path) -> Path:
        run = tmp_path / "runs" / "speaker_full-today"
        run.mkdir(parents=True)
        (run / "best_model.pth").write_bytes(b"new weights")
        (run / "config.json").write_text('{"run": 1}')
        return run

    def test_promotes_a_slim_copy_and_copies_the_config(self, run_dir, tmp_path):
        dest_dir = tmp_path / "models" / "speaker"

        result = train.promote_best_model(run_dir, dest_dir)

        assert result == dest_dir / "best_model.pth"
        assert result.read_bytes() == b"slim:new weights"
        assert (dest_dir / "config.json").read_text() == '{"run": 1}'
        assert sorted(p.name for p in dest_dir.iterdir()) == ["best_model.pth", "config.json"]
        # The run directory is left as it was: clean_run_dir removes what is not needed.
        assert (run_dir / "best_model.pth").read_bytes() == b"new weights"

    def test_real_trainer_layout_promotes_the_numbered_file(self, run_dir, tmp_path):
        # coqui's Trainer writes best_model_<step>.pth and an identical best_model.pth,
        # plus the latest checkpoints.
        (run_dir / "best_model_100.pth").write_bytes(b"step 100")
        (run_dir / "best_model_250.pth").write_bytes(b"step 250")
        (run_dir / "best_model.pth").write_bytes(b"step 250")
        (run_dir / "checkpoint_250.pth").write_bytes(b"checkpoint")

        result = train.promote_best_model(run_dir, tmp_path / "out")

        assert result is not None and result.read_bytes() == b"slim:step 250"
        assert sorted(p.name for p in run_dir.glob("*.pth")) == [
            "best_model.pth",
            "best_model_100.pth",
            "best_model_250.pth",
            "checkpoint_250.pth",
        ]

    def test_falls_back_to_the_latest_checkpoint(self, run_dir, tmp_path):
        (run_dir / "best_model.pth").unlink()
        (run_dir / "checkpoint_100.pth").write_bytes(b"step 100")
        (run_dir / "checkpoint_250.pth").write_bytes(b"step 250")

        result = train.promote_best_model(run_dir, tmp_path / "out")

        assert result is not None and result.read_bytes() == b"slim:step 250"

    def test_an_existing_model_and_config_are_kept_with_a_timestamp(
        self, run_dir, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(train, "datetime", FrozenDatetime)
        dest_dir = tmp_path / "out"
        dest_dir.mkdir()
        (dest_dir / "best_model.pth").write_bytes(b"old weights")  # any format
        (dest_dir / "config.json").write_text('{"run": 0}')

        result = train.promote_best_model(run_dir, dest_dir)

        assert result is not None and result.read_bytes() == b"slim:new weights"
        assert (dest_dir / "config.json").read_text() == '{"run": 1}'
        assert (dest_dir / "best_model.20260929-203015.pth").read_bytes() == b"old weights"
        assert (dest_dir / "config.20260929-203015.json").read_text() == '{"run": 0}'

    def test_no_model_is_ever_deleted_by_repeated_promotions(self, run_dir, tmp_path, monkeypatch):
        monkeypatch.setattr(train, "datetime", FrozenDatetime)  # every backup gets the same tag
        dest_dir = tmp_path / "out"
        for generation in range(4):
            (run_dir / "best_model.pth").write_bytes(f"generation {generation}".encode())
            train.promote_best_model(run_dir, dest_dir)

        models = {p.name: p.read_bytes() for p in dest_dir.glob("best_model*.pth")}
        assert models == {
            "best_model.pth": b"slim:generation 3",
            "best_model.20260929-203015.pth": b"slim:generation 0",
            "best_model.20260929-203015-1.pth": b"slim:generation 1",
            "best_model.20260929-203015-2.pth": b"slim:generation 2",
        }

    def test_an_interrupt_while_slimming_leaves_no_partial_model(
        self, run_dir, tmp_path, monkeypatch
    ):
        def interrupted(source, dest=None, *, replace=False):
            Path(dest).write_bytes(b"trunc")  # the rename had just happened
            raise KeyboardInterrupt

        monkeypatch.setattr(train.slim, "slim_checkpoint", interrupted)

        with pytest.raises(KeyboardInterrupt):
            train.promote_best_model(run_dir, tmp_path / "out")

        assert list((tmp_path / "out").iterdir()) == []
        assert (run_dir / "best_model.pth").read_bytes() == b"new weights"

    def test_no_checkpoint_gives_none_and_a_warning(self, tmp_path, caplog):
        run = tmp_path / "empty_run"
        run.mkdir()

        with caplog.at_level(logging.WARNING, logger="herald.tts.train"):
            assert train.promote_best_model(run, tmp_path / "out") is None

        assert "No model to promote" in caplog.text
        assert not (tmp_path / "out").exists()

    def test_a_missing_run_directory_gives_none(self, tmp_path):
        assert train.promote_best_model(tmp_path / "nope", tmp_path / "out") is None

    def test_a_run_without_config_still_promotes_the_model(self, run_dir, tmp_path):
        (run_dir / "config.json").unlink()
        result = train.promote_best_model(run_dir, tmp_path / "out")
        assert result is not None and result.is_file()
        assert not (tmp_path / "out" / "config.json").exists()


class TestPromoteWhenSlimmingIsNotPossible:
    """The model is never lost: the full checkpoint is moved instead."""

    @pytest.fixture
    def run_dir(self, tmp_path) -> Path:
        run = tmp_path / "runs" / "speaker_full-today"
        run.mkdir(parents=True)
        (run / "best_model_250.pth").write_bytes(b"step 250")
        (run / "best_model.pth").write_bytes(b"step 250")  # the Trainer's identical copy
        (run / "checkpoint_250.pth").write_bytes(b"checkpoint")
        (run / "config.json").write_text("{}")
        return run

    def test_the_checkpoint_is_moved_and_its_copy_removed(
        self, run_dir, tmp_path, slim_fails, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="herald.tts.train"):
            result = train.promote_best_model(run_dir, tmp_path / "out")

        assert result is not None and result.read_bytes() == b"step 250"
        assert "Could not slim" in caplog.text
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
            "best_model.pth",
            "config.json",
        ]
        assert sorted(p.name for p in run_dir.glob("*.pth")) == ["checkpoint_250.pth"]

    def test_a_best_model_copy_of_another_size_is_not_removed(self, run_dir, tmp_path, slim_fails):
        (run_dir / "best_model.pth").write_bytes(b"something else entirely")

        train.promote_best_model(run_dir, tmp_path / "out")

        assert (run_dir / "best_model.pth").read_bytes() == b"something else entirely"

    def test_an_already_slim_checkpoint_is_moved_not_copied(self, run_dir, tmp_path, monkeypatch):
        def already_slim(source, dest=None, *, replace=False):
            size = Path(source).stat().st_size
            return slim.SlimResult(Path(source), Path(source), size, size, already_slim=True)

        monkeypatch.setattr(train.slim, "slim_checkpoint", already_slim)

        result = train.promote_best_model(run_dir, tmp_path / "out")

        assert result is not None and result.read_bytes() == b"step 250"
        assert not (run_dir / "best_model_250.pth").exists()

    def test_without_torch_the_checkpoint_is_moved(self, run_dir, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` fails

        result = train.promote_best_model(run_dir, tmp_path / "out")

        assert result is not None and result.read_bytes() == b"step 250"

    def test_a_failed_move_leaves_the_model_where_it_is(
        self, run_dir, tmp_path, slim_fails, monkeypatch
    ):
        dest_dir = tmp_path / "out"
        dest_dir.mkdir()
        (dest_dir / "best_model.pth").write_bytes(b"old weights")

        def failing_move(src, dst):
            Path(dst).write_bytes(b"trunc")  # what an interrupted copy leaves behind
            raise OSError("No space left on device")

        monkeypatch.setattr(train.shutil, "move", failing_move)

        with pytest.raises(CheckpointError, match="still at .*best_model_250.pth") as excinfo:
            train.promote_best_model(run_dir, dest_dir)

        assert "No space left" in str(excinfo.value)
        assert (run_dir / "best_model_250.pth").read_bytes() == b"step 250"
        assert sorted(p.name for p in dest_dir.iterdir()) == ["best_model.pth"]
        assert (dest_dir / "best_model.pth").read_bytes() == b"old weights"

    def test_an_interrupted_move_leaves_no_partial_file(
        self, run_dir, tmp_path, slim_fails, monkeypatch
    ):
        def interrupted_move(src, dst):
            Path(dst).write_bytes(b"trunc")
            raise KeyboardInterrupt

        monkeypatch.setattr(train.shutil, "move", interrupted_move)

        with pytest.raises(KeyboardInterrupt):
            train.promote_best_model(run_dir, tmp_path / "out")

        assert list((tmp_path / "out").iterdir()) == []


class TestPromoteWithRealSlimming:
    def test_the_promoted_model_has_no_optimizer_state(self, tmp_path):
        torch = pytest.importorskip("torch")
        run_dir = tmp_path / "runs" / "run"
        run_dir.mkdir(parents=True)
        weights = {"gpt.weight": torch.randn(64, 64)}
        state = {"model": weights, "optimizer": {"exp_avg": torch.randn(64, 64) * 2}, "step": 10}
        torch.save(state, run_dir / "best_model_10.pth")
        (run_dir / "best_model.pth").write_bytes((run_dir / "best_model_10.pth").read_bytes())

        result = train.promote_best_model(run_dir, tmp_path / "models" / "voice")

        assert result is not None
        promoted = torch.load(result, map_location="cpu", weights_only=False)
        assert set(promoted) == {"model"}
        assert torch.equal(promoted["model"]["gpt.weight"], weights["gpt.weight"])
        assert result.stat().st_size < (run_dir / "best_model_10.pth").stat().st_size
        assert [p.name for p in result.parent.iterdir()] == ["best_model.pth"]


class TestCleanRunDir:
    def test_removes_only_checkpoints_of_the_run_directory(self, tmp_path):
        run = tmp_path / "run"
        (run / "sub").mkdir(parents=True)
        for name, size in {
            "checkpoint_250.pth": 100,
            "checkpoint_500.pth": 100,
            "best_model.pth": 30,
            "best_model_250.pth": 30,
            "config.json": 5,
            "trainer_0_log.txt": 5,
            "events.out.tfevents.1": 5,
            "dvae.pth": 7,
            "sub/checkpoint_1.pth": 9,
        }.items():
            (run / name).write_bytes(b"x" * size)

        freed = train.clean_run_dir(run)

        assert freed == 260
        assert sorted(p.name for p in run.iterdir()) == [
            "config.json",
            "dvae.pth",
            "events.out.tfevents.1",
            "sub",
            "trainer_0_log.txt",
        ]
        assert (run / "sub" / "checkpoint_1.pth").exists()

    def test_an_empty_or_missing_directory_frees_nothing(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert train.clean_run_dir(tmp_path / "empty") == 0
        assert train.clean_run_dir(tmp_path / "missing") == 0


class TestRealCoquiClasses:
    """Our keyword arguments against the installed coqui-tts (skipped where it is not).

    The fake stack of the tests below cannot notice a renamed or removed field; this builds
    the real config objects (no weights are read, nothing is trained).
    """

    def test_trainer_config_and_arguments_are_accepted(self, dataset_dir, tmp_path):
        pytest.importorskip("trainer")
        gpt_trainer = pytest.importorskip("TTS.tts.layers.xtts.trainer.gpt_trainer")
        xtts = pytest.importorskip("TTS.tts.models.xtts")
        from trainer import Trainer, TrainerArgs

        params = params_for(dataset_dir, tmp_path, preset="smoke", lr=1e-5)
        config = gpt_trainer.GPTTrainerConfig(
            model_args=gpt_trainer.GPTArgs(**train.model_args_kwargs(tmp_path / "ckpt")),
            audio=xtts.XttsAudioConfig(**train.AUDIO_CONFIG_KWARGS),
            **train.trainer_config_kwargs(params, run_eval=False),
        )
        args = TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=params.grad_accum_steps,
        )

        assert (config.epochs, config.lr, config.run_eval) == (1, 1e-5, False)
        assert config.model_args.xtts_checkpoint == str(tmp_path / "ckpt" / "model.pth")
        assert args.grad_accum_steps == params.grad_accum_steps
        # The keyword arguments run_training passes to the Trainer.
        accepted = inspect.signature(Trainer.__init__).parameters
        for name in ("output_path", "model", "train_samples", "eval_samples"):
            assert name in accepted
        assert "parse_command_line_args" in accepted


class Recorder:
    """A class that remembers its constructor arguments."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


# What the coqui Trainer leaves in a run directory (contents only matter for their size).
REAL_LAYOUT = {
    "best_model_10.pth": b"weights",
    "best_model.pth": b"weights",  # identical copy of the numbered file
    "checkpoint_5.pth": b"ckpt",
    "checkpoint_10.pth": b"ckpt",
    "trainer_0_log.txt": b"log",
}
REAL_LAYOUT_CHECKPOINT_BYTES = 7 + 7 + 4 + 4


def make_fake_stack(events: list, run_dir: Path, *, files: dict[str, bytes] | None = None):
    files = REAL_LAYOUT if files is None else files

    class FakeTrainer(Recorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.output_path = run_dir
            events.append(("trainer_init", self))

        def fit(self):
            events.append(("fit", self))
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text("{}")
            for name, content in files.items():
                (run_dir / name).write_bytes(content)

    class FakeGPTTrainer:
        @staticmethod
        def init_from_config(config):
            events.append(("init_model", config))
            return "model"

    return types.SimpleNamespace(
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
        Trainer=FakeTrainer,
        TrainerArgs=Recorder,
        GPTArgs=Recorder,
        GPTTrainer=FakeGPTTrainer,
        GPTTrainerConfig=Recorder,
        XttsAudioConfig=Recorder,
    )


@pytest.mark.usefixtures("fake_slimmer")
class TestRunTraining:
    @pytest.fixture
    def run_dir(self, tmp_path) -> Path:
        return tmp_path / "runs" / "speaker_full-today"

    @pytest.fixture
    def events(self, monkeypatch, mocker, run_dir):
        events: list = []
        monkeypatch.setattr(
            train, "_import_training_stack", lambda: make_fake_stack(events, run_dir)
        )
        mocker.patch("herald.tts.checkpoints.ensure_base_checkpoints")
        return events

    def by_name(self, events, name):
        return [obj for n, obj in events if n == name]

    def trainer_kwargs(self, events) -> dict:
        (trainer,) = self.by_name(events, "trainer_init")
        return trainer.kwargs

    def test_wires_everything_and_fits(self, train_dataset, tmp_path, events, run_dir):
        params = params_for(train_dataset, tmp_path, speaker_name="frieren", grad_accum_steps=8)

        result = train.run_training(params)

        (trainer,) = self.by_name(events, "trainer_init")
        trainer_args, config = trainer.args
        assert trainer_args.kwargs["grad_accum_steps"] == 8
        assert trainer_args.kwargs["restore_path"] is None
        assert trainer.kwargs["model"] == "model"
        assert trainer.kwargs["output_path"] == str(tmp_path / "runs")
        assert len(trainer.kwargs["train_samples"]) == 18
        assert len(trainer.kwargs["eval_samples"]) == 2
        # Otherwise the Trainer would parse `herald train ...` from sys.argv.
        assert trainer.kwargs["parse_command_line_args"] is False
        assert self.by_name(events, "fit") == [trainer]

        assert config.kwargs["run_name"] == "frieren_full"
        assert config.kwargs["run_eval"] is True
        assert config.kwargs["model_args"].kwargs["xtts_checkpoint"] == str(
            tmp_path / "ckpt" / "model.pth"
        )
        assert config.kwargs["audio"].kwargs["output_sample_rate"] == 24_000
        assert result.run_dir == run_dir

    def test_promotes_the_best_model_and_deletes_the_checkpoints(
        self, train_dataset, tmp_path, events, run_dir
    ):
        result = train.run_training(params_for(train_dataset, tmp_path, speaker_name="frieren"))

        model = tmp_path / "models" / "frieren" / "best_model.pth"
        assert result == train.TrainResult(
            run_dir=run_dir, model_path=model, freed_bytes=REAL_LAYOUT_CHECKPOINT_BYTES
        )
        assert model.read_bytes() == b"slim:weights"
        assert (model.parent / "config.json").is_file()
        # Checkpoints are gone; the config and the logs of the run stay.
        assert sorted(p.name for p in run_dir.iterdir()) == ["config.json", "trainer_0_log.txt"]

    def test_keep_checkpoints_keeps_everything_in_the_run_directory(
        self, train_dataset, tmp_path, events, run_dir
    ):
        params = params_for(train_dataset, tmp_path, keep_checkpoints=True)

        result = train.run_training(params)

        assert result.freed_bytes == 0
        assert result.model_path is not None and result.model_path.is_file()
        assert sorted(p.name for p in run_dir.iterdir()) == sorted([*REAL_LAYOUT, "config.json"])

    def test_the_latest_checkpoint_is_promoted_when_no_best_model_exists(
        self, train_dataset, tmp_path, monkeypatch, mocker, run_dir
    ):
        # A run that stopped before its first epoch ended.
        files = {"checkpoint_5.pth": b"old", "checkpoint_10.pth": b"latest"}
        stack = make_fake_stack([], run_dir, files=files)
        monkeypatch.setattr(train, "_import_training_stack", lambda: stack)
        mocker.patch("herald.tts.checkpoints.ensure_base_checkpoints")

        result = train.run_training(params_for(train_dataset, tmp_path))

        assert result.model_path is not None
        assert result.model_path.read_bytes() == b"slim:latest"
        assert result.freed_bytes == len(b"old") + len(b"latest")

    def test_no_model_found_deletes_nothing(
        self, train_dataset, tmp_path, monkeypatch, mocker, run_dir
    ):
        stack = make_fake_stack([], run_dir, files={"notes.txt": b"hi"})
        monkeypatch.setattr(train, "_import_training_stack", lambda: stack)
        mocker.patch("herald.tts.checkpoints.ensure_base_checkpoints")

        result = train.run_training(params_for(train_dataset, tmp_path))

        assert result == train.TrainResult(run_dir=run_dir, model_path=None)
        assert (run_dir / "notes.txt").exists()

    def test_a_failed_promotion_deletes_nothing(
        self, train_dataset, tmp_path, events, run_dir, slim_fails, monkeypatch
    ):
        def failing_move(src, dst):
            raise OSError("No space left on device")

        monkeypatch.setattr(train.shutil, "move", failing_move)

        with pytest.raises(CheckpointError, match="still at"):
            train.run_training(params_for(train_dataset, tmp_path))

        assert sorted(p.name for p in run_dir.iterdir()) == sorted([*REAL_LAYOUT, "config.json"])

    def test_nothing_outside_the_run_directory_is_deleted(
        self, train_dataset, tmp_path, events, run_dir
    ):
        models = tmp_path / "models"
        (models / "frieren").mkdir(parents=True)
        (models / "other").mkdir()
        untouched = {
            models / "frieren" / "best_model.20200101-000000.pth": b"old backup",
            models / "frieren" / "checkpoint_1.pth": b"a stray file",
            models / "other" / "best_model.pth": b"another voice",
            tmp_path / "ckpt" / "model.pth": b"base model",
        }
        (tmp_path / "ckpt").mkdir()
        for path, content in untouched.items():
            path.write_bytes(content)
        (models / "frieren" / "best_model.pth").write_bytes(b"current voice")

        train.run_training(params_for(train_dataset, tmp_path, speaker_name="frieren"))

        assert {p: p.read_bytes() for p in untouched} == untouched
        backups = sorted(p.name for p in (models / "frieren").glob("best_model.2*.pth"))
        assert len(backups) == 2  # the old backup and the rotated current voice
        assert (models / "frieren" / "best_model.pth").read_bytes() == b"slim:weights"

    def test_checkpoints_are_kept_if_the_model_is_in_the_run_directory(
        self, train_dataset, tmp_path, events, run_dir, caplog
    ):
        params = params_for(
            train_dataset, tmp_path, models_dir=run_dir.parent, model_name=run_dir.name
        )

        with caplog.at_level(logging.WARNING, logger="herald.tts.train"):
            result = train.run_training(params)

        assert result.freed_bytes == 0
        assert "not deleting checkpoints" in caplog.text
        assert (run_dir / "best_model.pth").read_bytes() == b"slim:weights"
        assert (run_dir / "checkpoint_10.pth").exists()

    def test_a_smoke_run_leaves_the_real_voice_alone(self, train_dataset, tmp_path, events):
        voice = tmp_path / "models" / "frieren"
        voice.mkdir(parents=True)
        (voice / "best_model.pth").write_bytes(b"the real voice")

        result = train.run_training(
            params_for(train_dataset, tmp_path, preset="smoke", speaker_name="frieren")
        )

        assert result.model_path == tmp_path / "models" / "frieren_smoke" / "best_model.pth"
        assert [p.name for p in voice.iterdir()] == ["best_model.pth"]
        assert (voice / "best_model.pth").read_bytes() == b"the real voice"

    def test_samples_are_complete_and_resolved(self, train_dataset, tmp_path, events):
        train.run_training(params_for(train_dataset, tmp_path, speaker_name="frieren"))

        kwargs = self.trainer_kwargs(events)
        sample = kwargs["train_samples"][0]
        assert Path(sample["audio_file"]).is_absolute()
        assert sample["speaker_name"] == "frieren"
        assert sample["language"] == "en"
        assert sample["text"].startswith("Sentence number ")

    def test_train_and_eval_are_disjoint_and_cover_the_dataset(
        self, train_dataset, tmp_path, events
    ):
        train.run_training(params_for(train_dataset, tmp_path))

        kwargs = self.trainer_kwargs(events)
        train_files = [s["audio_file"] for s in kwargs["train_samples"]]
        eval_files = [s["audio_file"] for s in kwargs["eval_samples"]]
        assert set(train_files).isdisjoint(eval_files)
        assert len(set(train_files) | set(eval_files)) == 20

    def test_the_split_follows_the_seed(self, train_dataset, tmp_path, events):
        def eval_files(seed: int) -> tuple[str, ...]:
            train.run_training(params_for(train_dataset, tmp_path, seed=seed))
            eval_samples = self.by_name(events, "trainer_init")[-1].kwargs["eval_samples"]
            return tuple(s["audio_file"] for s in eval_samples)

        assert eval_files(1) == eval_files(1)
        assert len({eval_files(seed) for seed in range(5)}) > 1

    def test_sample_limits_apply_after_the_split(self, train_dataset, tmp_path, events):
        params = params_for(train_dataset, tmp_path, max_train_samples=3, max_eval_samples=1)
        train.run_training(params)

        kwargs = self.trainer_kwargs(events)
        assert len(kwargs["train_samples"]) == 3
        assert len(kwargs["eval_samples"]) == 1

    def test_without_eval_samples_evaluation_is_switched_off(self, train_dataset, tmp_path, events):
        train.run_training(params_for(train_dataset, tmp_path, eval_fraction=0))

        (trainer,) = self.by_name(events, "trainer_init")
        _, config = trainer.args
        assert len(trainer.kwargs["train_samples"]) == 20
        assert trainer.kwargs["eval_samples"] == []
        assert config.kwargs["run_eval"] is False

    def test_missing_audio_fails_before_anything_heavy(
        self, train_dataset, tmp_path, events, mocker
    ):
        ensure = mocker.patch("herald.tts.checkpoints.ensure_base_checkpoints")
        (train_dataset / "audio" / "000000.wav").unlink()

        with pytest.raises(MetadataError, match="1 audio file"):
            train.run_training(params_for(train_dataset, tmp_path))

        assert events == []
        ensure.assert_not_called()

    def test_base_checkpoints_are_ensured(self, train_dataset, tmp_path, events, mocker):
        ensure = mocker.patch("herald.tts.checkpoints.ensure_base_checkpoints")
        params = params_for(train_dataset, tmp_path, checkpoint_url="https://mirror.test/xtts")

        train.run_training(params)

        ensure.assert_called_once_with(tmp_path / "ckpt", base_url="https://mirror.test/xtts")
