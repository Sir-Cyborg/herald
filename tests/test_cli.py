"""CLI: --help for every command, argument wiring with the heavy modules replaced by fakes."""

import io
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from herald import cli
from herald.config import DEFAULT_CHECKPOINT_URL, DEFAULT_OLLAMA_TIMEOUT
from herald.errors import CheckpointError, HeraldError, OllamaError
from herald.tts.slim import SlimResult
from herald.tts.train import TrainResult

COMMANDS = ["synthesize", "train", "slim", "chat", "download-checkpoints"]


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Keep the host's HERALD_* variables out, and write outputs under tmp_path."""
    for name in list(os.environ):
        if name.startswith("HERALD_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HERALD_PROJECT_ROOT", str(tmp_path / "project"))


@pytest.fixture(autouse=True)
def silent_player(monkeypatch):
    """Never play sound from a test. Tests that check playback patch play_wav themselves."""
    from herald import audio_playback

    monkeypatch.setattr(audio_playback, "play_wav", lambda path: True)


def run(argv, capsys):
    code = cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


class FakeEngine:
    sample_rate = 1000

    def __init__(self):
        self.calls = []

    def synth_long(self, text, pause_ms, max_chars):
        self.calls.append({"text": text, "pause_ms": pause_ms, "max_chars": max_chars})
        return np.zeros(2500, dtype=np.float32)  # 2.5 s at 1 kHz


@pytest.fixture
def fake_load(monkeypatch):
    """Replace engine.load_engine; records its arguments and returns a FakeEngine."""
    from herald.tts import engine

    calls = []
    fake = FakeEngine()

    def load_engine(checkpoint_dir, reference_wavs, **kwargs):
        calls.append({"checkpoint_dir": checkpoint_dir, "reference_wavs": reference_wavs, **kwargs})
        return fake

    monkeypatch.setattr(engine, "load_engine", load_engine)
    fake.load_calls = calls
    return fake


def voice_args(dataset_dir, tmp_path):
    return ["--dataset-dir", str(dataset_dir), "--checkpoint-dir", str(tmp_path / "ckpt")]


# --- help ---------------------------------------------------------------------------------------


def test_verbose_enables_debug_for_herald_only():
    import logging

    herald_logger = logging.getLogger("herald")
    before = (herald_logger.level, logging.getLogger().level)
    try:
        cli._configure_logging(True)
        assert herald_logger.isEnabledFor(logging.DEBUG)
        # Third-party libraries (numba, matplotlib, ...) stay quiet at the debug level.
        assert not logging.getLogger("numba").isEnabledFor(logging.DEBUG)
        cli._configure_logging(False)
        assert not herald_logger.isEnabledFor(logging.DEBUG)
    finally:
        herald_logger.setLevel(before[0])
        logging.getLogger().setLevel(before[1])


def test_top_level_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in COMMANDS:
        assert command in out


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_has_help(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--help"])
    assert exc.value.code == 0
    assert f"herald {command}" in capsys.readouterr().out


def test_help_shows_environment_defaults(monkeypatch, capsys):
    monkeypatch.setenv("HERALD_LANGUAGE", "it")
    with pytest.raises(SystemExit):
        cli.main(["synthesize", "--help"])
    assert "(default: it)" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("herald ")


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["--help"], *([c, "--help"] for c in COMMANDS)])
def test_help_does_not_import_the_heavy_stack(argv):
    code = (
        "import sys\n"
        "from herald.cli import main\n"
        f"try:\n    main({argv!r})\nexcept SystemExit:\n    pass\n"
        "heavy = [m for m in ('torch', 'TTS', 'trainer', 'transformers', 'numpy', 'requests')"
        " if m in sys.modules]\n"
        "print('HEAVY:', heavy)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip().splitlines()[-1] == "HEAVY: []"


# --- synthesize ---------------------------------------------------------------------------------


class TestSynthesize:
    def test_wiring(self, dataset_dir, tmp_path, fake_load, capsys):
        out_wav = tmp_path / "out" / "hello.wav"
        code, out, _ = run(
            [
                "synthesize",
                "Hello there.",
                *voice_args(dataset_dir, tmp_path),
                "-o",
                str(out_wav),
                "--device",
                "cpu",
                "--language",
                "it",
                "--temperature",
                "0.5",
                "--num-references",
                "2",
                "--pause-ms",
                "50",
                "--max-chars",
                "120",
            ],
            capsys,
        )

        assert code == 0
        (call,) = fake_load.load_calls
        assert call["checkpoint_dir"] == tmp_path / "ckpt"
        assert len(call["reference_wavs"]) == 2
        assert all(Path(p).is_file() and str(dataset_dir) in p for p in call["reference_wavs"])
        assert call["finetuned_checkpoint"] is None
        assert (call["device"], call["language"], call["temperature"]) == ("cpu", "it", 0.5)
        assert fake_load.calls == [{"text": "Hello there.", "pause_ms": 50, "max_chars": 120}]
        with wave.open(str(out_wav)) as f:
            assert (f.getframerate(), f.getnframes()) == (1000, 2500)
        assert "2.5s" in out

    def test_max_chars_is_left_to_the_engine_unless_given(
        self, dataset_dir, tmp_path, fake_load, capsys
    ):
        argv = ["synthesize", "Hi.", "-o", str(tmp_path / "o.wav")]
        run(argv + voice_args(dataset_dir, tmp_path), capsys)
        run([*argv, "--max-chars", "99", *voice_args(dataset_dir, tmp_path)], capsys)
        # None means "the character limit of the language" (decided by the engine).
        assert [c["max_chars"] for c in fake_load.calls] == [None, 99]

    def test_default_output_is_a_timestamped_file_in_the_output_dir(
        self, dataset_dir, tmp_path, fake_load, monkeypatch, capsys
    ):
        monkeypatch.setenv("HERALD_OUTPUT_DIR", str(tmp_path / "outputs"))
        code, _, _ = run(["synthesize", "Hi.", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 0
        (wav,) = (tmp_path / "outputs").glob("herald_*.wav")
        assert wav.stat().st_size > 44

    def test_finetuned_checkpoint_from_a_run_directory(
        self, dataset_dir, tmp_path, fake_load, capsys
    ):
        run_dir = tmp_path / "runs" / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "best_model.pth").write_bytes(b"x")
        code, _, _ = run(
            ["synthesize", "Hi.", *voice_args(dataset_dir, tmp_path)]
            + ["--checkpoint", str(run_dir), "-o", str(tmp_path / "o.wav")],
            capsys,
        )
        assert code == 0
        assert fake_load.load_calls[0]["finetuned_checkpoint"] == run_dir / "best_model.pth"

    def test_finetuned_voice_from_a_models_directory(
        self, dataset_dir, tmp_path, fake_load, capsys
    ):
        voice = tmp_path / "models" / "frieren"
        voice.mkdir(parents=True)
        (voice / "best_model.pth").write_bytes(b"x")
        code, _, _ = run(
            ["synthesize", "Hi.", "--checkpoint", str(voice), "-o", str(tmp_path / "o.wav")]
            + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 0
        assert fake_load.load_calls[0]["finetuned_checkpoint"] == voice / "best_model.pth"

    def test_checkpoint_defaults_to_the_environment(
        self, dataset_dir, tmp_path, fake_load, monkeypatch, capsys
    ):
        voice = tmp_path / "voice.pth"
        voice.write_bytes(b"x")
        monkeypatch.setenv("HERALD_CHECKPOINT", str(voice))
        code, _, _ = run(
            ["synthesize", "Hi.", "-o", str(tmp_path / "o.wav")]
            + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 0
        assert fake_load.load_calls[0]["finetuned_checkpoint"] == voice

    def test_checkpoint_option_beats_the_environment(
        self, dataset_dir, tmp_path, fake_load, monkeypatch, capsys
    ):
        monkeypatch.setenv("HERALD_CHECKPOINT", str(tmp_path / "from_env.pth"))
        chosen = tmp_path / "chosen.pth"
        chosen.write_bytes(b"x")
        code, _, _ = run(
            ["synthesize", "Hi.", "--checkpoint", str(chosen), "-o", str(tmp_path / "o.wav")]
            + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 0
        assert fake_load.load_calls[0]["finetuned_checkpoint"] == chosen

    def test_missing_checkpoint_is_a_one_line_error(self, dataset_dir, tmp_path, fake_load, capsys):
        code, _, err = run(
            ["synthesize", "Hi.", "--checkpoint", str(tmp_path / "nope")]
            + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 1
        assert "Checkpoint not found" in err
        assert fake_load.load_calls == []

    def test_base_weights_default_to_the_models_directory(
        self, dataset_dir, tmp_path, fake_load, monkeypatch, capsys
    ):
        argv = [
            "synthesize",
            "Hi.",
            "--dataset-dir",
            str(dataset_dir),
            "-o",
            str(tmp_path / "o.wav"),
        ]
        run(argv, capsys)
        assert fake_load.load_calls[0]["checkpoint_dir"] == tmp_path / "project/models/xtts_v2"

        monkeypatch.setenv("HERALD_MODELS_DIR", str(tmp_path / "elsewhere"))
        run(argv, capsys)
        assert fake_load.load_calls[1]["checkpoint_dir"] == tmp_path / "elsewhere/xtts_v2"

    def test_explicit_reference_wavs_replace_dataset_picks(
        self, dataset_dir, tmp_path, fake_load, capsys
    ):
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"RIFF")
        code, _, _ = run(
            ["synthesize", "Hi.", "--reference-wav", str(ref), "-o", str(tmp_path / "o.wav")]
            + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 0
        assert fake_load.load_calls[0]["reference_wavs"] == [str(ref)]

    def test_reference_wavs_need_no_dataset(self, tmp_path, fake_load, capsys):
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"RIFF")
        argv = ["synthesize", "Hi.", "--reference-wav", str(ref), "-o", str(tmp_path / "o.wav")]
        code, _, _ = run(argv + voice_args(tmp_path / "no_dataset", tmp_path), capsys)
        assert code == 0
        assert fake_load.load_calls[0]["reference_wavs"] == [str(ref)]

    def test_no_dataset_and_no_reference_is_a_helpful_error(self, tmp_path, fake_load, capsys):
        # A fresh clone has no dataset (it is not in git): say what to do about it.
        nowhere = tmp_path / "no_dataset"
        code, _, err = run(["synthesize", "Hi.", *voice_args(nowhere, tmp_path)], capsys)
        assert code == 1
        assert err.count("\n") == 1  # one line
        assert f"No dataset found at {nowhere}" in err
        assert "--reference-wav" in err and "--dataset-dir" in err
        assert fake_load.load_calls == []

    def test_missing_reference_wav(self, dataset_dir, tmp_path, fake_load, capsys):
        code, _, err = run(
            ["synthesize", "Hi.", "--reference-wav", str(tmp_path / "ghost.wav")]
            + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 1
        assert "ghost.wav" in err
        assert fake_load.load_calls == []

    def test_text_from_a_file(self, dataset_dir, tmp_path, fake_load, capsys):
        text_file = tmp_path / "in.txt"
        text_file.write_text("From a file.\n", encoding="utf-8")
        code, _, _ = run(
            ["synthesize", "--text-file", str(text_file), "-o", str(tmp_path / "o.wav")]
            + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 0
        assert fake_load.calls[0]["text"] == "From a file.\n"

    def test_text_from_stdin(self, dataset_dir, tmp_path, fake_load, monkeypatch, capsys):
        monkeypatch.setattr(sys, "stdin", io.StringIO("Piped text."))
        code, _, _ = run(
            ["synthesize", "-o", str(tmp_path / "o.wav")] + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 0
        assert fake_load.calls[0]["text"] == "Piped text."

    def test_a_byte_order_mark_is_not_part_of_the_text(
        self, dataset_dir, tmp_path, fake_load, capsys
    ):
        text_file = tmp_path / "notepad.txt"
        text_file.write_bytes(b"\xef\xbb\xbfFrom Notepad.")
        code, _, _ = run(
            ["synthesize", "--text-file", str(text_file), "-o", str(tmp_path / "o.wav")]
            + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 0
        assert fake_load.calls[0]["text"] == "From Notepad."

    def test_text_and_text_file_together_are_rejected(
        self, dataset_dir, tmp_path, fake_load, capsys
    ):
        text_file = tmp_path / "in.txt"
        text_file.write_text("From a file.", encoding="utf-8")
        code, _, err = run(
            ["synthesize", "Inline.", "--text-file", str(text_file)]
            + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 1
        assert "either TEXT or --text-file" in err
        assert fake_load.load_calls == []

    def test_text_file_must_be_utf8(self, dataset_dir, tmp_path, fake_load, capsys):
        text_file = tmp_path / "in.txt"
        text_file.write_bytes(b"\xff\xfe\x00bad")
        code, _, err = run(
            ["synthesize", "--text-file", str(text_file)] + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 1
        assert "not a UTF-8 text file" in err
        assert "Traceback" not in err

    def test_empty_text_argument_does_not_fall_back_to_stdin(
        self, dataset_dir, tmp_path, fake_load, monkeypatch, capsys
    ):
        monkeypatch.setattr(sys, "stdin", io.StringIO("Piped text."))
        code, _, err = run(["synthesize", "", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 1
        assert "empty" in err
        assert fake_load.load_calls == []

    def test_no_text_on_a_terminal(self, dataset_dir, tmp_path, fake_load, monkeypatch, capsys):
        stdin = io.StringIO()
        stdin.isatty = lambda: True
        monkeypatch.setattr(sys, "stdin", stdin)
        code, _, err = run(["synthesize", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 1
        assert "No text given" in err
        assert fake_load.load_calls == []

    def test_blank_text_is_rejected_before_loading_the_model(
        self, dataset_dir, tmp_path, fake_load, capsys
    ):
        code, _, err = run(["synthesize", "   ", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 1
        assert "empty" in err
        assert fake_load.load_calls == []

    def test_play_flag(self, dataset_dir, tmp_path, fake_load, mocker, capsys):
        play = mocker.patch("herald.audio_playback.play_wav", return_value=True)
        out_wav = tmp_path / "o.wav"
        run(
            ["synthesize", "Hi.", "--play", "-o", str(out_wav)] + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        play.assert_called_once_with(out_wav)

    def test_playback_failure_is_only_a_note(
        self, dataset_dir, tmp_path, fake_load, mocker, capsys
    ):
        mocker.patch("herald.audio_playback.play_wav", return_value=False)
        code, _, err = run(
            ["synthesize", "Hi.", "--play", "-o", str(tmp_path / "o.wav")]
            + voice_args(dataset_dir, tmp_path),
            capsys,
        )
        assert code == 0
        assert "No audio player" in err


class TestNumericOptions:
    """Bad numbers are usage errors (exit 2) before anything heavy happens."""

    @pytest.mark.parametrize(
        ("option", "value", "message"),
        [
            ("--max-chars", "0", "must be at least 1"),
            ("--num-references", "0", "must be at least 1"),
            ("--pause-ms", "-1", "must be at least 0"),
            ("--temperature", "0", "must be greater than 0"),
            ("--temperature", "nan", "must be greater than 0"),
            ("--max-chars", "many", "invalid int value"),
        ],
    )
    def test_synthesize(self, option, value, message, fake_load, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["synthesize", "Hi.", option, value])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert f"argument {option}: " in err
        assert message in err
        assert fake_load.load_calls == []

    @pytest.mark.parametrize("value", ["0", "-5", "soon"])
    def test_ollama_timeout(self, value, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["chat", "--ollama-timeout", value])
        assert exc.value.code == 2
        assert "--ollama-timeout" in capsys.readouterr().err


# --- train --------------------------------------------------------------------------------------


class TestTrain:
    @pytest.fixture
    def trained(self, monkeypatch, tmp_path):
        """Replace run_training; record what make_params and run_training receive."""
        from herald.tts import train

        seen = SimpleNamespace(kwargs=[], params=[], result=None)
        seen.result = TrainResult(
            run_dir=tmp_path / "runs" / "finished",
            model_path=tmp_path / "models" / "frieren" / "best_model.pth",
        )
        real_make_params = train.make_params

        def make_params(**kwargs):
            seen.kwargs.append(kwargs)
            return real_make_params(**kwargs)

        def run_training(params):
            seen.params.append(params)
            return seen.result

        monkeypatch.setattr(train, "make_params", make_params)
        monkeypatch.setattr(train, "run_training", run_training)
        return seen

    def test_defaults_to_the_full_preset(self, dataset_dir, tmp_path, trained, capsys):
        code, _, _ = run(["train", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 0
        (p,) = trained.params
        assert (p.epochs, p.batch_size, p.grad_accum_steps) == (5, 2, 16)
        (kwargs,) = trained.kwargs
        assert kwargs["preset"] == "full"
        assert kwargs["dataset_dir"] == dataset_dir
        assert kwargs["checkpoint_dir"] == tmp_path / "ckpt"
        assert kwargs["runs_dir"] == tmp_path / "project" / "runs"
        assert kwargs["models_dir"] == tmp_path / "project" / "models"
        assert kwargs["checkpoint_url"] == DEFAULT_CHECKPOINT_URL
        assert kwargs["keep_checkpoints"] is False  # leftovers are deleted unless asked otherwise
        assert (kwargs["metadata_file"], kwargs["eval_fraction"], kwargs["seed"]) == (
            "metadata.csv",
            0.1,
            42,
        )

    def test_smoke_and_overrides(self, dataset_dir, tmp_path, trained, capsys):
        argv = ["train", "--smoke", "--epochs", "2", "--lr", "1e-5", "--speaker-name", "frieren"]
        argv += ["--runs-dir", str(tmp_path / "r")]
        argv += ["--metadata-file", "all.csv", "--eval-fraction", "0.25", "--seed", "7"]
        code, _, _ = run(argv + voice_args(dataset_dir, tmp_path), capsys)
        assert code == 0
        (p,) = trained.params
        assert (p.epochs, p.batch_size, p.max_train_samples) == (2, 1, 60)
        assert (p.lr, p.speaker_name, p.run_name) == (1e-5, "frieren", "frieren_smoke")
        (kwargs,) = trained.kwargs
        assert kwargs["runs_dir"] == tmp_path / "r"
        assert (kwargs["metadata_file"], kwargs["eval_fraction"], kwargs["seed"]) == (
            "all.csv",
            0.25,
            7,
        )

    def test_models_dir_comes_from_the_environment(
        self, dataset_dir, tmp_path, trained, monkeypatch, capsys
    ):
        monkeypatch.setenv("HERALD_MODELS_DIR", str(tmp_path / "voices"))
        run(["train", "--dataset-dir", str(dataset_dir)], capsys)
        (kwargs,) = trained.kwargs
        assert kwargs["models_dir"] == tmp_path / "voices"
        assert kwargs["checkpoint_dir"] == tmp_path / "voices" / "xtts_v2"

    def test_keep_checkpoints_reaches_make_params(self, dataset_dir, tmp_path, trained, capsys):
        run(["train", "--keep-checkpoints", *voice_args(dataset_dir, tmp_path)], capsys)
        assert trained.kwargs[0]["keep_checkpoints"] is True

    def test_reports_the_size_of_the_voice_and_the_space_freed(
        self, dataset_dir, tmp_path, trained, capsys
    ):
        model = trained.result.model_path
        model.parent.mkdir(parents=True)
        model.write_bytes(b"x" * 1500)
        trained.result = TrainResult(trained.result.run_dir, model, freed_bytes=11_200_000_000)
        code, out, _ = run(["train", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 0
        assert f"Fine-tuned voice: {model} (1.5 KB)" in out
        assert "Freed 11.2 GB of checkpoints" in out

    def test_nothing_is_said_about_freed_space_when_nothing_was_freed(
        self, dataset_dir, tmp_path, trained, capsys
    ):
        _, out, _ = run(["train", *voice_args(dataset_dir, tmp_path)], capsys)
        assert "Freed" not in out

    def test_there_is_no_models_dir_option(self, dataset_dir, tmp_path, trained, capsys):
        # HERALD_MODELS_DIR is the single knob: an option would not move the base weights.
        with pytest.raises(SystemExit) as exc:
            cli.main(["train", "--models-dir", str(tmp_path)])
        assert exc.value.code == 2
        assert trained.params == []

    def test_the_checkpoint_url_comes_from_the_environment(
        self, dataset_dir, tmp_path, trained, monkeypatch, capsys
    ):
        monkeypatch.setenv("HERALD_CHECKPOINT_URL", "http://mirror/xtts")
        run(["train", *voice_args(dataset_dir, tmp_path)], capsys)
        assert trained.kwargs[0]["checkpoint_url"] == "http://mirror/xtts"

    def test_reports_the_run_and_how_to_use_the_voice(self, dataset_dir, tmp_path, trained, capsys):
        code, out, _ = run(["train", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 0
        voice = tmp_path / "models" / "frieren"
        assert f"Training finished. Run directory: {tmp_path / 'runs' / 'finished'}" in out
        assert f"Fine-tuned voice: {voice / 'best_model.pth'}" in out
        # The printed command must work as it stands: options that are not the defaults
        # (here the dataset and the base weights) are part of it.
        voice_q, dataset_q, ckpt_q = (
            cli._shell_quote(str(p)) for p in (voice, dataset_dir, tmp_path / "ckpt")
        )
        expected = (
            f'herald synthesize "Hello." --checkpoint {voice_q} '
            f"--dataset-dir {dataset_q} --checkpoint-dir {ckpt_q}"
        )
        assert expected in out

    def test_hint_is_short_when_everything_is_default(
        self, dataset_dir, tmp_path, trained, monkeypatch, capsys
    ):
        monkeypatch.setenv("HERALD_DATASET_DIR", str(dataset_dir))
        monkeypatch.chdir(tmp_path)  # the voice is under the working directory: relative path
        code, out, _ = run(["train"], capsys)
        assert code == 0
        voice = os.path.join("models", "frieren")  # relative to the working directory
        assert f'Use it with:  herald synthesize "Hello." --checkpoint {voice}\n' in out

    def test_hint_quotes_paths_with_spaces(self, dataset_dir, tmp_path, trained, capsys):
        voice = tmp_path / "my voices"
        trained.result = TrainResult(run_dir=tmp_path, model_path=voice / "best_model.pth")
        _, out, _ = run(["train", *voice_args(dataset_dir, tmp_path)], capsys)
        assert f"--checkpoint {cli._shell_quote(str(voice))} " in out
        assert cli._shell_quote(str(voice)) != str(voice)  # it did get quoted

    def test_shell_quoting_follows_the_platform(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")  # cmd.exe does not understand single quotes
        assert cli._shell_quote("my voices") == '"my voices"'
        assert cli._shell_quote("plain") == "plain"
        monkeypatch.setattr(os, "name", "posix")
        assert cli._shell_quote("my voices") == "'my voices'"
        assert cli._shell_quote("plain") == "plain"

    def test_no_saved_model(self, dataset_dir, tmp_path, trained, capsys):
        trained.result = TrainResult(run_dir=tmp_path / "finished", model_path=None)
        code, out, _ = run(["train", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 0
        assert f"Run directory: {tmp_path / 'finished'}" in out
        assert "No fine-tuned model was saved" in out
        assert "herald synthesize" not in out

    def test_dry_run_checks_the_dataset_without_training(
        self, dataset_dir, tmp_path, trained, capsys
    ):
        code, out, _ = run(["train", "--dry-run", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 0
        assert trained.params == []
        train_rows, eval_rows = (
            int(n) for n in re.findall(r"^(?:train|eval) rows: (\d+)$", out, re.M)
        )
        assert train_rows + eval_rows == 5 and eval_rows >= 1  # the 5 rows of metadata.csv
        assert "dataset OK" in out
        assert "epochs: 5" in out

    @pytest.fixture
    def audit(self, monkeypatch):
        """Make validate_dataset return a canned report; call it with the findings to report."""
        from herald.dataset.audio_stats import AudioStats
        from herald.tts import train

        def install(*, audio=True, long_text_rows=0, missing=(), **stats):
            findings = {
                "clips_checked": 1299,
                "unreadable": (),
                "total_seconds": 5220.0,
                "sample_rates": {44100: 1299},
                "channels": {1: 1299},
                "too_short": (),
                "too_long": (),
                "weak_reference": 0,
                **stats,
            }
            report = train.DatasetReport(
                train_rows=1169,
                eval_rows=130,
                missing_audio=tuple(missing),
                audio=AudioStats(**findings) if audio else None,
                long_text_rows=long_text_rows,
            )
            monkeypatch.setattr(train, "validate_dataset", lambda params: report)

        return install

    def dry_run(self, dataset_dir, tmp_path, capsys):
        return run(["train", "--dry-run", *voice_args(dataset_dir, tmp_path)], capsys)

    def test_dry_run_summarizes_the_audio(self, dataset_dir, tmp_path, trained, audit, capsys):
        audit()
        code, out, _ = self.dry_run(dataset_dir, tmp_path, capsys)
        assert code == 0
        assert "Dataset check:" in out
        assert "speech: 87.0 min in 1299 clip(s)" in out
        assert "sample rates: 44100 Hz x1299" in out
        assert "channels: mono x1299" in out
        assert "warning" not in out  # nothing to complain about
        # The check sits between the row counts and the verdict.
        assert out.index("eval rows: 130") < out.index("Dataset check:") < out.index("dataset OK")

    def test_dry_run_lists_every_sample_rate_and_channel_count(
        self, dataset_dir, tmp_path, trained, audit, capsys
    ):
        audit(sample_rates={22050: 3, 44100: 10}, channels={1: 12, 2: 1})
        _, out, _ = self.dry_run(dataset_dir, tmp_path, capsys)
        assert "sample rates: 22050 Hz x3, 44100 Hz x10" in out
        assert "channels: mono x12, stereo x1" in out

    def test_dry_run_warns_but_still_succeeds(self, dataset_dir, tmp_path, trained, audit, capsys):
        audit(
            long_text_rows=4,
            too_short=("audio/a.wav", "audio/b.wav"),
            too_long=("audio/c.wav",),
            weak_reference=17,
            unreadable=("audio/d.mp3",),
        )
        code, out, _ = self.dry_run(dataset_dir, tmp_path, capsys)
        assert code == 0  # warnings never change the exit code
        assert (
            "2 clip(s) shorter than 0.5 s will be skipped by the trainer (e.g. audio/a.wav)" in out
        )
        assert "1 clip(s) longer than ~11.6 s will be skipped (e.g. audio/c.wav)" in out
        assert "4 transcript(s) over 200 characters may be skipped" in out
        assert "17 clip(s) shorter than 3 s: usable, but they teach the voice less" in out
        assert "1 file(s) are not WAV and were not inspected" in out
        assert "dataset OK" in out

    def test_dry_run_reports_missing_audio_after_the_check(
        self, dataset_dir, tmp_path, trained, audit, capsys
    ):
        audit(missing=("audio/gone.wav",))
        code, out, _ = self.dry_run(dataset_dir, tmp_path, capsys)
        assert code == 1
        assert out.index("Dataset check:") < out.index("missing audio: 1 (e.g. audio/gone.wav)")
        assert "dataset OK" not in out

    def test_dry_run_without_audio_findings(self, dataset_dir, tmp_path, trained, audit, capsys):
        audit(audio=False)
        code, out, _ = self.dry_run(dataset_dir, tmp_path, capsys)
        assert code == 0
        assert "Dataset check" not in out and "dataset OK" in out
        audit(audio=False, long_text_rows=2)  # transcripts are checked even so
        _, out, _ = self.dry_run(dataset_dir, tmp_path, capsys)
        assert "2 transcript(s) over 200 characters may be skipped" in out

    def test_dry_run_fails_on_missing_audio(self, dataset_dir, tmp_path, trained, capsys):
        (dataset_dir / "audio" / "000000.wav").unlink()
        code, out, _ = run(["train", "--dry-run", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 1
        assert "missing audio: 1" in out
        assert "dataset OK" not in out

    def test_dry_run_with_a_missing_metadata_file(self, dataset_dir, tmp_path, trained, capsys):
        argv = ["train", "--dry-run", "--metadata-file", "nope.csv"]
        code, _, err = run(argv + voice_args(dataset_dir, tmp_path), capsys)
        assert code == 1
        assert "nope.csv" in err and "Traceback" not in err

    def test_invalid_eval_fraction_is_a_clean_error(self, dataset_dir, tmp_path, trained, capsys):
        argv = ["train", "--dry-run", "--eval-fraction", "1.5"]
        code, _, err = run(argv + voice_args(dataset_dir, tmp_path), capsys)
        assert code == 1
        assert "eval_fraction" in err

    def test_invalid_option_is_a_clean_error(self, dataset_dir, tmp_path, trained, capsys):
        code, _, err = run(["train", "--epochs", "0", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 1
        assert "epochs must be at least 1" in err
        assert trained.params == []


# --- chat ---------------------------------------------------------------------------------------


class TestChat:
    @pytest.fixture
    def assistant(self, monkeypatch):
        """Replace Assistant with a script of replies; an exception in it is raised instead."""
        from herald import assistant as assistant_module

        seen = SimpleNamespace(created=[], said=[])
        replies = iter(["First reply.", OllamaError("Ollama is down"), "Third reply.", "Fourth."])

        class FakeAssistant:
            def __init__(self, llm, system_prompt, **kwargs):
                seen.created.append({"llm": llm, "system_prompt": system_prompt, **kwargs})

            def respond(self, user_text):
                seen.said.append(user_text)
                reply = next(replies)
                if isinstance(reply, Exception):
                    raise reply
                return reply

        monkeypatch.setattr(assistant_module, "Assistant", FakeAssistant)
        return seen

    @pytest.fixture
    def typed(self, monkeypatch):
        """Feed the given lines to input(); EOF afterwards."""

        def feed(*lines):
            it = iter(lines)

            def fake_input(prompt=""):
                try:
                    return next(it)
                except StopIteration:
                    raise EOFError from None

            monkeypatch.setattr("builtins.input", fake_input)

        return feed

    @pytest.fixture(autouse=True)
    def scratch_root(self, tmp_path, monkeypatch):
        """Where tempfile puts the chat's scratch directory, so that tests can look inside."""
        root = tmp_path / "scratch"
        root.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(root))
        return root

    def chat_args(self, dataset_dir, tmp_path, *extra):
        return ["chat", *voice_args(dataset_dir, tmp_path), *extra]

    def keep(self, tmp_path):
        """The option that keeps the replies, and the folder they end up in."""
        return ["--save-dir", str(tmp_path / "wavs")]

    def test_conversation_loop(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, mocker, capsys
    ):
        play = mocker.patch("herald.audio_playback.play_wav", return_value=True)
        typed("Hello", "Are you there?", "Still there?", "")

        code, out, err = run(self.chat_args(dataset_dir, tmp_path, *self.keep(tmp_path)), capsys)

        assert code == 0
        assert len(assistant.created) == 1  # one conversation for the whole session
        assert assistant.said == ["Hello", "Are you there?", "Still there?"]
        assert "Herald: First reply." in out and "Herald: Third reply." in out
        assert "error: Ollama is down" in err  # the failed turn does not end the session
        # Only the replies that arrived are spoken, saved and played.
        assert [c["text"] for c in fake_load.calls] == ["First reply.", "Third reply."]
        assert [c["max_chars"] for c in fake_load.calls] == [None, None]
        saved = sorted((tmp_path / "wavs").glob("chat_*.wav"))
        assert [p.name[-7:-4] for p in saved] == ["001", "003"]  # turn 2 had no reply
        assert [c.args[0] for c in play.call_args_list] == saved

    def test_replies_are_kept_only_with_save_dir(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, mocker, scratch_root, capsys
    ):
        mocker.patch("herald.audio_playback.play_wav", return_value=True)
        typed("a", "b", "c")
        run(self.chat_args(dataset_dir, tmp_path, *self.keep(tmp_path)), capsys)
        assert len(list((tmp_path / "wavs").iterdir())) == 2  # exactly one file per spoken reply
        assert list(scratch_root.iterdir()) == []  # no scratch directory was needed

    def test_replies_are_played_from_one_scratch_file_and_nothing_is_left(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, mocker, scratch_root, capsys
    ):
        played = []

        def play(path):
            played.append((path, sorted(p.name for p in path.parent.iterdir())))
            return True

        mocker.patch("herald.audio_playback.play_wav", side_effect=play)
        typed("a", "b", "c")

        code, out, _ = run(self.chat_args(dataset_dir, tmp_path), capsys)

        assert code == 0
        assert "Herald: First reply." in out and "Herald: Third reply." in out
        (first, listing1), (second, listing2) = played
        assert first == second  # every reply is written over the previous one
        assert first.parent.parent == scratch_root
        assert first.parent.name.startswith("herald-chat-")
        assert listing1 == listing2 == [first.name]  # at most one reply is ever on disk
        assert list(scratch_root.iterdir()) == []  # the scratch directory is gone
        assert not (tmp_path / "project" / "output").exists()  # ...and nothing went to output/

    def test_the_scratch_directory_is_removed_on_ctrl_c(
        self, dataset_dir, tmp_path, fake_load, assistant, mocker, monkeypatch, scratch_root, capsys
    ):
        play = mocker.patch("herald.audio_playback.play_wav", return_value=True)
        lines = iter(["Hello"])

        def input_then_interrupt(prompt=""):
            try:
                return next(lines)
            except StopIteration:
                raise KeyboardInterrupt from None

        monkeypatch.setattr("builtins.input", input_then_interrupt)
        code, _, _ = run(self.chat_args(dataset_dir, tmp_path), capsys)
        assert code == 130
        assert play.call_count == 1  # a reply had been written to the scratch directory
        assert list(scratch_root.iterdir()) == []

    def test_the_scratch_directory_is_removed_after_a_failed_synthesis(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, scratch_root, capsys
    ):
        def broken(text, pause_ms, max_chars):
            raise RuntimeError("model exploded")

        fake_load.synth_long = broken
        typed("a", "b", "c")
        code, _, err = run(self.chat_args(dataset_dir, tmp_path), capsys)
        assert code == 0
        assert err.count("could not speak the reply: model exploded") == 2
        assert list(scratch_root.iterdir()) == []

    def test_no_play_without_save_dir_is_a_text_only_chat(
        self, tmp_path, fake_load, assistant, typed, mocker, scratch_root, capsys
    ):
        play = mocker.patch("herald.audio_playback.play_wav")
        typed("a", "b", "c")

        # No dataset, no weights: nothing is needed because nothing is spoken.
        code, out, err = run(self.chat_args(tmp_path / "no_dataset", tmp_path, "--no-play"), capsys)

        assert code == 0
        assert "Herald: First reply." in out and "Herald: Third reply." in out
        assert fake_load.load_calls == [] and fake_load.calls == []  # the model is never loaded
        play.assert_not_called()
        assert err.count("chatting in text only") == 1
        assert "Audio is neither played nor saved (--no-play without --save-dir)" in err
        assert "Add --save-dir DIR to keep WAV files." in err
        assert list(scratch_root.iterdir()) == []

    def test_no_play_with_save_dir_saves_without_playing(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, mocker, capsys
    ):
        play = mocker.patch("herald.audio_playback.play_wav")
        typed("a", "b", "c")
        args = self.chat_args(dataset_dir, tmp_path, "--no-play", *self.keep(tmp_path))
        code, _, err = run(args, capsys)
        assert code == 0
        assert len(list((tmp_path / "wavs").glob("chat_*.wav"))) == 2
        assert len(fake_load.load_calls) == 1
        play.assert_not_called()
        assert "text only" not in err

    def test_a_failed_synthesis_does_not_end_the_session(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, mocker, capsys
    ):
        mocker.patch("herald.audio_playback.play_wav", return_value=True)
        real_synth = fake_load.synth_long

        def flaky(text, pause_ms, max_chars):
            if text == "First reply.":
                raise RuntimeError("model exploded")
            return real_synth(text, pause_ms, max_chars)

        fake_load.synth_long = flaky
        typed("one", "two", "three", "")

        code, out, err = run(self.chat_args(dataset_dir, tmp_path, *self.keep(tmp_path)), capsys)

        assert code == 0
        assert "Herald: First reply." in out  # the text is shown even if it could not be spoken
        assert "could not speak the reply: model exploded" in err
        assert "Herald: Third reply." in out  # ...and the conversation carries on
        assert len(list((tmp_path / "wavs").glob("chat_*.wav"))) == 1

    def test_ollama_settings_reach_the_client(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, capsys
    ):
        typed("Hi")
        extra = ["--ollama-url", "http://ollama:11434", "--ollama-model", "mistral"]
        extra += ["--ollama-timeout", "9", "--system-prompt", "Be brief.", "--no-play"]
        run(self.chat_args(dataset_dir, tmp_path, *extra), capsys)
        (created,) = assistant.created
        client = created["llm"]
        assert (client.base_url, client.model, client.timeout) == (
            "http://ollama:11434",
            "mistral",
            9,
        )
        assert created["system_prompt"] == "Be brief."  # the Assistant owns the system prompt

    def test_system_prompt_from_a_file(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, capsys
    ):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("  You are a pirate.\n", encoding="utf-8")
        typed("Hi")
        run(self.chat_args(dataset_dir, tmp_path, "--system-prompt-file", str(prompt_file)), capsys)
        (created,) = assistant.created
        assert created["system_prompt"] == "You are a pirate."

    def test_system_prompt_file_may_start_with_a_byte_order_mark(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, capsys
    ):
        prompt_file = tmp_path / "notepad.txt"
        prompt_file.write_bytes(b"\xef\xbb\xbfYou are a pirate.\r\n")
        typed()
        run(self.chat_args(dataset_dir, tmp_path, "--system-prompt-file", str(prompt_file)), capsys)
        assert assistant.created[0]["system_prompt"] == "You are a pirate."

    @pytest.mark.parametrize(
        ("env", "option", "expected"),
        [
            (None, None, DEFAULT_OLLAMA_TIMEOUT),
            ("7.5", None, 7.5),
            ("7.5", "9", 9),
            ("soon", "9", 9),  # the option wins, so the broken variable is never looked at
        ],
    )
    def test_ollama_timeout_precedence(
        self,
        env,
        option,
        expected,
        dataset_dir,
        tmp_path,
        fake_load,
        assistant,
        typed,
        monkeypatch,
        capsys,
    ):
        if env is not None:
            monkeypatch.setenv("HERALD_OLLAMA_TIMEOUT", env)
        typed()
        extra = ["--ollama-timeout", option] if option else []
        code, _, _ = run(self.chat_args(dataset_dir, tmp_path, *extra), capsys)
        assert code == 0
        assert assistant.created[0]["llm"].timeout == expected

    def test_system_prompt_file_must_be_utf8(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, capsys
    ):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_bytes(b"\xff\xfe\x00")
        code, _, err = run(
            self.chat_args(dataset_dir, tmp_path, "--system-prompt-file", str(prompt_file)), capsys
        )
        assert code == 1
        assert "not a UTF-8 text file" in err
        assert fake_load.load_calls == []  # fails before the model is loaded

    @pytest.mark.parametrize(
        ("extra", "turns"), [([], 10), (["--history", "3"], 3), (["--history", "0"], 0)]
    )
    def test_history_setting_reaches_the_assistant(
        self, extra, turns, dataset_dir, tmp_path, fake_load, assistant, typed, capsys
    ):
        typed()
        run(self.chat_args(dataset_dir, tmp_path, *extra), capsys)
        assert assistant.created[0]["history_turns"] == turns

    def test_missing_player_is_mentioned_once(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, mocker, capsys
    ):
        mocker.patch("herald.audio_playback.play_wav", return_value=False)
        typed("a", "b", "c")
        _, _, err = run(self.chat_args(dataset_dir, tmp_path), capsys)
        assert err.count("No audio player available") == 1
        # Nothing is kept, so it must not send the user looking for a folder of replies.
        assert "use --save-dir DIR to keep the replies as WAV files" in err

    def test_missing_player_with_save_dir_says_where_the_replies_are(
        self, dataset_dir, tmp_path, fake_load, assistant, typed, mocker, capsys
    ):
        mocker.patch("herald.audio_playback.play_wav", return_value=False)
        typed("a", "b", "c")
        _, _, err = run(self.chat_args(dataset_dir, tmp_path, *self.keep(tmp_path)), capsys)
        assert err.count("No audio player available") == 1
        assert f"replies are saved in {tmp_path / 'wavs'}" in err

    def test_ctrl_c_ends_the_session_quietly(
        self, dataset_dir, tmp_path, fake_load, assistant, monkeypatch, capsys
    ):
        def interrupted(prompt=""):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", interrupted)
        code, _, _ = run(self.chat_args(dataset_dir, tmp_path), capsys)
        assert code == 130

    def test_with_the_real_assistant_history_stays_clean_after_an_ollama_error(
        self, dataset_dir, tmp_path, fake_load, typed, mocker, capsys
    ):
        """CLI + Assistant + OllamaClient together; only the HTTP call is faked."""

        def answer(text):
            body = {"message": {"role": "assistant", "content": text}}
            return SimpleNamespace(ok=True, status_code=200, json=lambda: body, text=str(body))

        post = mocker.patch(
            "requests.post",
            side_effect=[answer("First."), requests.ConnectionError("down"), answer("Third.")],
        )
        typed("one", "two", "three")
        code, out, err = run(self.chat_args(dataset_dir, tmp_path), capsys)

        assert code == 0
        assert "Herald: First." in out and "Herald: Third." in out
        assert "error: Cannot connect to Ollama" in err
        sent = [call.kwargs["json"]["messages"] for call in post.call_args_list]
        assert [m["content"] for m in sent[2]][1:] == ["one", "First.", "three"]  # no "two"


# --- slim ---------------------------------------------------------------------------------------


class TestSlim:
    BEFORE, AFTER = 5_600_000_000, 1_900_000_000

    @pytest.fixture
    def slimmed(self, monkeypatch):
        """Replace slim_checkpoint (it needs torch and a real checkpoint); record its calls."""
        from herald.tts import slim

        seen = SimpleNamespace(calls=[], already_slim=False, error=None)

        def slim_checkpoint(source, dest=None, *, replace=False):
            seen.calls.append({"source": source, "dest": dest, "replace": replace})
            if seen.error:
                raise seen.error
            path = source if replace else dest or source.with_suffix(".slim.pth")
            if seen.already_slim:
                return SlimResult(source, path, self.AFTER, self.AFTER, already_slim=True)
            return SlimResult(source, path, self.BEFORE, self.AFTER)

        monkeypatch.setattr(slim, "slim_checkpoint", slim_checkpoint)
        return seen

    @pytest.fixture
    def checkpoint(self, tmp_path):
        path = tmp_path / "models" / "frieren" / "best_model.pth"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        return path

    def test_a_copy_is_written_next_to_the_original_by_default(self, checkpoint, slimmed, capsys):
        code, out, _ = run(["slim", str(checkpoint)], capsys)
        assert code == 0
        assert slimmed.calls == [{"source": checkpoint, "dest": None, "replace": False}]
        slim_path = checkpoint.with_suffix(".slim.pth")
        assert out.strip() == (f"{checkpoint} -> {slim_path}: 5.6 GB -> 1.9 GB (saved 3.7 GB)")

    def test_a_voice_directory_is_resolved_to_its_best_model(self, checkpoint, slimmed, capsys):
        code, _, _ = run(["slim", str(checkpoint.parent)], capsys)
        assert code == 0
        assert slimmed.calls[0]["source"] == checkpoint

    def test_a_run_directory_is_resolved_to_the_latest_best_model(self, tmp_path, slimmed, capsys):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for step in (100, 2000):
            (run_dir / f"best_model_{step}.pth").write_bytes(b"x")
        run(["slim", str(run_dir)], capsys)
        assert slimmed.calls[0]["source"] == run_dir / "best_model_2000.pth"

    def test_output_path(self, checkpoint, tmp_path, slimmed, capsys):
        target = tmp_path / "voices" / "frieren.pth"
        code, out, _ = run(["slim", str(checkpoint), "-o", str(target)], capsys)
        assert code == 0
        assert slimmed.calls == [{"source": checkpoint, "dest": target, "replace": False}]
        assert f"{checkpoint} -> {target}: " in out

    def test_replace(self, checkpoint, slimmed, capsys):
        code, out, _ = run(["slim", str(checkpoint), "--replace"], capsys)
        assert code == 0
        assert slimmed.calls == [{"source": checkpoint, "dest": None, "replace": True}]
        assert out.strip() == (
            f"{checkpoint}: 5.6 GB -> 1.9 GB (saved 3.7 GB), the original was replaced"
        )

    def test_output_and_replace_are_mutually_exclusive(self, checkpoint, slimmed, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["slim", str(checkpoint), "-o", str(tmp_path / "o.pth"), "--replace"])
        assert exc.value.code == 2
        assert "not allowed with" in capsys.readouterr().err
        assert slimmed.calls == []

    def test_a_checkpoint_argument_is_required(self, slimmed, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["slim"])
        assert exc.value.code == 2

    def test_already_slim(self, checkpoint, slimmed, capsys):
        slimmed.already_slim = True
        code, out, _ = run(["slim", str(checkpoint)], capsys)
        assert code == 0
        assert out.strip() == f"{checkpoint}: already slim (1.9 GB), nothing to do"

    def test_slimming_errors_are_one_line(self, checkpoint, slimmed, capsys):
        slimmed.error = CheckpointError("not a valid checkpoint")
        code, out, err = run(["slim", str(checkpoint)], capsys)
        assert code == 1
        assert err.strip().endswith("error: not a valid checkpoint")
        assert "Traceback" not in err and out == ""

    def test_missing_checkpoint(self, tmp_path, slimmed, capsys):
        code, _, err = run(["slim", str(tmp_path / "nope.pth")], capsys)
        assert code == 1
        assert "Checkpoint not found" in err and "nope.pth" in err
        assert slimmed.calls == []


@pytest.mark.parametrize(
    ("size", "text"),
    [
        (0, "0 B"),
        (999, "999 B"),
        (1_500, "1.5 KB"),
        (2_500_000, "2.5 MB"),
        (1_900_000_000, "1.9 GB"),
        (5_600_000_000, "5.6 GB"),
        (11_200_000_000, "11.2 GB"),
        (2_000_000_000_000, "2.0 TB"),
    ],
)
def test_human_size(size, text):
    assert cli._human_size(size) == text


# --- download-checkpoints ------------------------------------------------------------------------


def test_download_checkpoints(tmp_path, mocker, capsys):
    ensure = mocker.patch(
        "herald.tts.checkpoints.ensure_base_checkpoints", return_value=tmp_path / "ckpt"
    )
    code, out, _ = run(
        [
            "download-checkpoints",
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--base-url",
            "http://m",
        ],
        capsys,
    )
    assert code == 0
    ensure.assert_called_once_with(tmp_path / "ckpt", base_url="http://m")
    assert str(tmp_path / "ckpt") in out


def test_download_defaults_to_the_models_directory(tmp_path, mocker, capsys):
    ensure = mocker.patch("herald.tts.checkpoints.ensure_base_checkpoints")
    run(["download-checkpoints"], capsys)
    assert ensure.call_args.args == (tmp_path / "project" / "models" / "xtts_v2",)


def test_download_uses_the_environment_url(tmp_path, mocker, monkeypatch, capsys):
    monkeypatch.setenv("HERALD_CHECKPOINT_URL", "http://mirror/xtts")
    ensure = mocker.patch("herald.tts.checkpoints.ensure_base_checkpoints")
    run(["download-checkpoints", "--checkpoint-dir", str(tmp_path)], capsys)
    assert ensure.call_args.kwargs["base_url"] == "http://mirror/xtts"


# --- error handling -------------------------------------------------------------------------------


class TestErrors:
    def test_expected_errors_are_one_line(self, tmp_path, mocker, capsys):
        mocker.patch(
            "herald.tts.checkpoints.ensure_base_checkpoints", side_effect=HeraldError("no network")
        )
        code, out, err = run(["download-checkpoints", "--checkpoint-dir", str(tmp_path)], capsys)
        assert code == 1
        assert err.strip().endswith("error: no network")
        assert "Traceback" not in err

    def test_os_errors_are_one_line(self, tmp_path, mocker, capsys):
        mocker.patch(
            "herald.tts.checkpoints.ensure_base_checkpoints", side_effect=PermissionError("denied")
        )
        code, _, err = run(["download-checkpoints", "--checkpoint-dir", str(tmp_path)], capsys)
        assert code == 1
        assert "error: denied" in err

    def test_bugs_are_not_swallowed(self, tmp_path, mocker, capsys):
        mocker.patch("herald.tts.checkpoints.ensure_base_checkpoints", side_effect=KeyError("bug"))
        with pytest.raises(KeyError):
            cli.main(["download-checkpoints", "--checkpoint-dir", str(tmp_path)])

    def test_invalid_environment_stops_chat(
        self, dataset_dir, tmp_path, fake_load, monkeypatch, capsys
    ):
        monkeypatch.setenv("HERALD_OLLAMA_TIMEOUT", "soon")
        code, _, err = run(["chat", *voice_args(dataset_dir, tmp_path)], capsys)
        assert code == 1
        assert "HERALD_OLLAMA_TIMEOUT" in err
        assert fake_load.load_calls == []  # it fails before the model is loaded

    @pytest.mark.parametrize(
        "argv",
        [
            ["--help"],
            ["chat", "--help"],
            ["train", "--dry-run"],
            ["synthesize", "Hi.", "-o", "out.wav"],
            ["download-checkpoints"],
        ],
    )
    def test_the_ollama_timeout_only_matters_to_chat(
        self, argv, dataset_dir, tmp_path, fake_load, mocker, monkeypatch, capsys
    ):
        monkeypatch.setenv("HERALD_OLLAMA_TIMEOUT", "soon")
        mocker.patch("herald.tts.checkpoints.ensure_base_checkpoints")
        monkeypatch.chdir(tmp_path)
        if argv[0] != "--help" and "--help" not in argv:
            argv = [*argv, *voice_args(dataset_dir, tmp_path)]
        try:
            code = cli.main(argv)
        except SystemExit as exc:  # --help
            code = exc.code
        assert code == 0
        assert "HERALD_OLLAMA_TIMEOUT" not in capsys.readouterr().err

    def test_keyboard_interrupt(self, tmp_path, mocker, capsys):
        mocker.patch(
            "herald.tts.checkpoints.ensure_base_checkpoints", side_effect=KeyboardInterrupt
        )
        code, _, _ = run(["download-checkpoints", "--checkpoint-dir", str(tmp_path)], capsys)
        assert code == 130
