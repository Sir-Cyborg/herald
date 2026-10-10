"""CLI: --help for every command, argument wiring with the heavy modules replaced by fakes."""

import io
import os
import re
import subprocess
import sys
import tomllib
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from herald import cli
from herald.config import DEFAULT_CHECKPOINT_URL
from herald.errors import CheckpointError, HeraldError
from herald.tools import LoadReport, ToolRegistry
from herald.tts.slim import SlimResult
from herald.tts.train import TrainResult

SHOUT_TOOL = """\
from herald.tools import tool


@tool
def shout(text: str, times: int = 1) -> str:
    \"\"\"Repeat the text in capitals.

    Args:
        text: What to shout.
        times: How often (once if not said).
    \"\"\"
    return " ".join([text.upper()] * times)
"""

WHISPER_TOOL = """\
from herald.tools import tool


@tool(triggers=["whisper", "quiet"])
def whisper(text: str) -> str:
    \"\"\"Repeat the text in lower case.\"\"\"
    return text.lower()
"""

COMMANDS = [
    "synthesize",
    "train",
    "slim",
    "chat",
    "tools",
    "profiles",
    "new-profile",
    "doctor",
    "download-checkpoints",
]


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


def make_profile(tmp_path, name, text=""):
    """Write ``profiles/<name>.toml`` in the test project and return its path."""
    folder = tmp_path / "project" / "profiles"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def handled(monkeypatch):
    """Replace the handlers of synthesize, chat and train with recorders of their arguments."""
    seen = SimpleNamespace(args=[])

    def record(args, settings):
        seen.args.append(args)
        return 0

    for name in ("_cmd_synthesize", "_cmd_chat", "_cmd_train"):
        monkeypatch.setattr(cli, name, record)
    return seen


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


class TestOutputThatCannotShowEverything:
    """On Windows the console or a pipe can have a narrow code page: a reply with an emoji must
    come out as ``?``, not crash the chat with UnicodeEncodeError."""

    def narrow(self, encoding):
        return io.TextIOWrapper(io.BytesIO(), encoding=encoding, write_through=True)

    @pytest.mark.parametrize("encoding", ["ascii", "cp1252", "cp850"])
    def test_main_makes_printing_safe(self, encoding, monkeypatch):
        out, err = self.narrow(encoding), self.narrow(encoding)
        # Without it, printing the character fails: this is what the fix is for.
        with pytest.raises(UnicodeEncodeError):
            print("\U0001f600", file=self.narrow(encoding))

        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        with pytest.raises(SystemExit):
            cli.main(["--version"])
        print("Herald: \U0001f600 done", file=sys.stdout)
        print("error: \U0001f600", file=sys.stderr)
        assert out.buffer.getvalue().endswith(b"Herald: ? done\n")
        assert err.buffer.getvalue() == b"error: ?\n"

    def test_text_the_console_can_show_is_untouched(self, monkeypatch):
        out = self.narrow("cp1252")
        monkeypatch.setattr(sys, "stdout", out)
        cli._make_output_robust()
        print("caf\u00e9", file=sys.stdout)
        assert out.buffer.getvalue() == "caf\u00e9\n".encode("cp1252")

    def test_it_happens_before_anything_else_so_even_errors_are_safe(self, monkeypatch):
        err = self.narrow("ascii")
        monkeypatch.setattr(sys, "stderr", err)
        monkeypatch.setenv("HERALD_OLLAMA_TIMEOUT", "soon")  # fine until chat needs it
        code = cli.main(["new-profile", "bad name \u00e9\U0001f600"])
        assert code == 1
        assert b"Invalid profile name" in err.buffer.getvalue()

    def test_streams_that_cannot_be_reconfigured_are_left_alone(self, monkeypatch):
        class Plain:  # no reconfigure at all
            def write(self, text):
                return len(text)

            def flush(self):
                pass

        class Stubborn(Plain):
            def reconfigure(self, **kwargs):
                raise ValueError("cannot change the encoding after reading")

        monkeypatch.setattr(sys, "stdout", Plain())
        monkeypatch.setattr(sys, "stderr", Stubborn())
        cli._make_output_robust()  # must not raise


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

    # --- the profile that a successful training leaves behind -------------------------------

    @pytest.fixture
    def project_voice(self, tmp_path, trained):
        """A trained voice inside the project, so that its paths come out relative."""
        voice = tmp_path / "project" / "models" / "frieren"
        trained.result = TrainResult(tmp_path / "runs" / "finished", voice / "best_model.pth")
        return voice

    def profile_file(self, tmp_path, name="frieren"):
        return tmp_path / "project" / "profiles" / f"{name}.toml"

    def test_a_profile_is_created_for_the_new_voice(self, tmp_path, trained, project_voice, capsys):
        dataset = tmp_path / "project" / "dataset" / "frieren"
        argv = ["train", "--dataset-dir", str(dataset), "--language", "it"]
        code, out, _ = run([*argv, "--checkpoint-dir", str(tmp_path / "ckpt")], capsys)
        assert code == 0
        values = tomllib.loads(self.profile_file(tmp_path).read_text(encoding="utf-8"))
        assert re.fullmatch(
            r"Created by herald train on \d{4}-\d{2}-\d{2}", values.pop("description")
        )
        assert values == {
            "checkpoint": "models/frieren",
            "dataset": "dataset/frieren",
            "language": "it",
        }
        assert (
            "Profile created: profiles/frieren.toml (use it with: herald chat --profile frieren)\n"
        ) in out

    def test_the_created_profile_is_usable(self, tmp_path, trained, project_voice, capsys):
        project_voice.mkdir(parents=True)
        (project_voice / "best_model.pth").write_bytes(b"x")
        dataset = tmp_path / "project" / "dataset" / "frieren"
        dataset.mkdir(parents=True)
        (dataset / "metadata.csv").write_text("audio/a.wav|Hi.\n", encoding="utf-8")
        run(
            ["train", "--dataset-dir", str(dataset), "--checkpoint-dir", str(tmp_path / "c")],
            capsys,
        )
        code, out, _ = run(["profiles"], capsys)
        assert code == 0 and "frieren" in out and "!" not in out  # no problems reported

    def test_an_existing_profile_is_never_touched_nor_mentioned(
        self, tmp_path, trained, project_voice, capsys
    ):
        mine = make_profile(tmp_path, "frieren", 'description = "My own edits"\nlanguage = "ja"\n')
        code, out, err = run(["train", *voice_args(tmp_path / "speaker", tmp_path)], capsys)
        assert code == 0
        assert mine.read_text(encoding="utf-8") == 'description = "My own edits"\nlanguage = "ja"\n'
        assert "Profile created" not in out and "profile" not in err.lower()

    def test_a_second_training_keeps_the_first_profile(
        self, tmp_path, trained, project_voice, capsys
    ):
        argv = ["train", *voice_args(tmp_path / "speaker", tmp_path)]
        run([*argv, "--language", "it"], capsys)
        first = self.profile_file(tmp_path).read_text(encoding="utf-8")
        _, out, _ = run([*argv, "--language", "es"], capsys)
        assert self.profile_file(tmp_path).read_text(encoding="utf-8") == first
        assert "Profile created" not in out

    def test_a_smoke_run_gets_its_own_profile_and_leaves_the_real_one_alone(
        self, tmp_path, trained, capsys
    ):
        real = make_profile(tmp_path, "frieren", 'language = "en"\n')
        smoke_voice = tmp_path / "project" / "models" / "frieren_smoke"
        trained.result = TrainResult(tmp_path / "runs" / "r", smoke_voice / "best_model.pth")
        code, out, _ = run(
            ["train", "--smoke", *voice_args(tmp_path / "speaker", tmp_path)], capsys
        )
        assert code == 0
        assert real.read_text(encoding="utf-8") == 'language = "en"\n'
        values = tomllib.loads(self.profile_file(tmp_path, "frieren_smoke").read_text("utf-8"))
        assert values["checkpoint"] == "models/frieren_smoke"
        assert "herald chat --profile frieren_smoke" in out

    def test_no_profile_when_no_model_was_saved(self, tmp_path, trained, capsys):
        trained.result = TrainResult(tmp_path / "finished", None)
        run(["train", *voice_args(tmp_path / "speaker", tmp_path)], capsys)
        assert not (tmp_path / "project" / "profiles").exists()

    def test_a_voice_folder_that_cannot_be_a_profile_name_only_gets_a_warning(
        self, tmp_path, trained, capsys
    ):
        voice = tmp_path / "project" / "models" / "my voice"
        trained.result = TrainResult(tmp_path / "finished", voice / "best_model.pth")
        code, out, err = run(["train", *voice_args(tmp_path / "speaker", tmp_path)], capsys)
        assert code == 0  # the training worked
        assert "warning: no profile was created" in err and "my voice" in err
        assert "Profile created" not in out

    def test_a_profile_given_to_train_supplies_the_defaults(
        self, tmp_path, trained, project_voice, capsys
    ):
        make_profile(tmp_path, "mario", 'language = "it"\nspeaker_name = "mario"\n')
        argv = ["train", "--profile", "mario", "--checkpoint-dir", str(tmp_path / "ckpt")]
        code, _, _ = run(argv, capsys)
        assert code == 0
        assert trained.kwargs[0]["language"] == "it"
        assert trained.params[0].speaker_name == "mario"

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


# --- tools --------------------------------------------------------------------------------------


class TestTools:
    @pytest.fixture
    def tools_dir(self, tmp_path):
        path = tmp_path / "mine"
        path.mkdir()
        (path / "shout.py").write_text(SHOUT_TOOL, encoding="utf-8")
        return path

    def test_lists_builtin_and_own_tools_with_their_parameters(self, tools_dir, capsys):
        code, out, err = run(["tools", "--tools-dir", str(tools_dir)], capsys)
        assert code == 0 and err == ""
        assert "set_timer  (builtin)" in out
        assert "seconds (integer, required): " in out
        assert "message (string): " in out  # optional: no "required"
        assert f"shout  ({tools_dir / 'shout.py'})" in out
        assert "  Repeat the text in capitals." in out
        assert "    text (string, required): What to shout." in out
        assert "    times (integer): How often (once if not said)." in out

    def test_says_when_each_tool_is_offered(self, tools_dir, capsys):
        (tools_dir / "whisper.py").write_text(WHISPER_TOOL, encoding="utf-8")
        _, out, _ = run(["tools", "--tools-dir", str(tools_dir)], capsys)
        blocks = {block.splitlines()[0].split()[0]: block for block in out.strip().split("\n\n")}
        # A tool without trigger words is offered on every message...
        assert "\n  always offered\n" in blocks["shout"] + "\n"
        # ...one with them only when the message contains one of them (the full list is shown).
        assert "\n  offered when the message contains: whisper, quiet\n" in blocks["whisper"] + "\n"
        # The built-in timer has trigger words too, so that a small model is not tempted to set
        # timers all the time.
        assert "\n  offered when the message contains: " in blocks["set_timer"]

    def test_a_broken_script_is_reported_and_fails_the_command(self, tools_dir, capsys):
        (tools_dir / "broken.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
        code, out, err = run(["tools", "--tools-dir", str(tools_dir)], capsys)
        assert code == 1
        assert "shout  (" in out  # the good tools are still listed
        assert err.count("error: ") == 1 and "broken.py" in err and "boom" in err

    def test_a_missing_tools_directory_is_not_an_error(self, tmp_path, capsys):
        code, out, _ = run(["tools", "--tools-dir", str(tmp_path / "nope")], capsys)
        assert code == 0
        assert "set_timer  (builtin)" in out

    def test_default_directory_and_environment(self, tmp_path, monkeypatch, capsys):
        import herald.tools

        seen = []
        real = herald.tools.load_tools

        def load_tools(user_dir, context, **kwargs):
            seen.append(user_dir)
            return real(user_dir, context, **kwargs)

        monkeypatch.setattr(herald.tools, "load_tools", load_tools)
        run(["tools"], capsys)
        monkeypatch.setenv("HERALD_TOOLS_DIR", str(tmp_path / "from_env"))
        run(["tools"], capsys)
        assert seen == [tmp_path / "project" / "tools", tmp_path / "from_env"]

    def test_nothing_found(self, monkeypatch, capsys):
        import herald.tools

        nothing = LoadReport(ToolRegistry(), (), ())
        monkeypatch.setattr(herald.tools, "load_tools", lambda user_dir, context, **kw: nothing)
        code, out, _ = run(["tools"], capsys)
        assert code == 0
        assert out.strip() == "No tools found."

    def test_the_scheduler_is_shut_down(self, monkeypatch, capsys):
        import herald.tools.scheduler

        stopped = []

        class FakeScheduler:
            def shutdown(self):
                stopped.append(True)
                return 0

        monkeypatch.setattr(herald.tools.scheduler, "Scheduler", FakeScheduler)
        run(["tools"], capsys)
        assert stopped == [True]


# --- profiles -----------------------------------------------------------------------------------


class TestProfileDefaults:
    """``--profile``: command line > profile > environment > built-in default."""

    def test_the_profile_supplies_the_defaults_of_its_options(self, tmp_path, handled, capsys):
        make_profile(
            tmp_path,
            "mario",
            """
            language = "it"
            temperature = 0.3
            device = "cpu"
            num_references = 2
            ollama_model = "mistral"
            history = 4
            system_prompt = "Be Mario."
            """,
        )
        code, _, _ = run(["chat", "--profile", "mario"], capsys)
        assert code == 0
        (args,) = handled.args
        assert (args.language, args.temperature, args.device) == ("it", 0.3, "cpu")
        assert (args.num_references, args.ollama_model, args.history) == (2, "mistral", 4)
        assert args.system_prompt == "Be Mario."

    def test_the_command_line_beats_the_profile(self, tmp_path, handled, capsys):
        make_profile(
            tmp_path,
            "mario",
            'language = "it"\nollama_model = "mistral"\nsystem_prompt = "Be Mario."\n',
        )
        argv = ["chat", "--profile", "mario", "--language", "en", "--ollama-model", "phi"]
        run([*argv, "--system-prompt", "Be Luigi."], capsys)
        (args,) = handled.args
        assert (args.language, args.ollama_model) == ("en", "phi")
        assert args.system_prompt == "Be Luigi."

    def test_the_profile_beats_the_environment_which_beats_the_built_in_default(
        self, tmp_path, handled, monkeypatch, capsys
    ):
        make_profile(tmp_path, "mario", 'language = "it"\n')
        monkeypatch.setenv("HERALD_LANGUAGE", "de")  # the profile has a language: it wins
        monkeypatch.setenv("HERALD_DEVICE", "cpu")  # the profile has none: the environment does
        monkeypatch.setenv("HERALD_OLLAMA_MODEL", "phi")
        monkeypatch.setenv("HERALD_SYSTEM_PROMPT", "From the environment.")
        run(["chat", "--profile", "mario"], capsys)
        (args,) = handled.args
        assert (args.language, args.device) == ("it", "cpu")
        assert (args.ollama_model, args.system_prompt) == ("phi", "From the environment.")
        assert (args.temperature, args.history) == (0.7, 10)  # nobody said: built-in

    def test_a_system_prompt_file_on_the_command_line_still_wins(self, tmp_path, handled, capsys):
        make_profile(tmp_path, "mario", 'system_prompt = "Be Mario."\n')
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("Be Luigi.", encoding="utf-8")
        code, _, _ = run(
            ["chat", "--profile", "mario", "--system-prompt-file", str(prompt)], capsys
        )
        assert code == 0  # not a clash: the profile only sets the default of --system-prompt
        (args,) = handled.args
        assert args.system_prompt_file == prompt  # and chat reads the file in preference

    def test_a_profile_may_keep_its_character_in_a_file(self, tmp_path, handled, capsys):
        (tmp_path / "project").mkdir(exist_ok=True)
        (tmp_path / "project" / "mario.md").write_text("  Be Mario.\n", encoding="utf-8")
        make_profile(tmp_path, "mario", 'system_prompt_file = "mario.md"\n')
        run(["chat", "--profile", "mario"], capsys)
        assert handled.args[0].system_prompt == "Be Mario."

    def test_keys_the_command_does_not_have_are_ignored(self, tmp_path, handled, capsys):
        make_profile(
            tmp_path,
            "mario",
            'language = "it"\nsystem_prompt = "Be Mario."\nollama_model = "mistral"\n'
            'history = 3\nspeaker_name = "mario"\ntemperature = 0.2\n',
        )
        code, _, err = run(["synthesize", "Hi.", "--profile", "mario"], capsys)
        assert code == 0 and "error" not in err
        (args,) = handled.args
        assert (args.language, args.temperature) == ("it", 0.2)
        assert not hasattr(args, "system_prompt") and not hasattr(args, "ollama_model")

    def test_train_takes_the_dataset_language_and_speaker_from_the_profile(
        self, tmp_path, handled, capsys
    ):
        make_profile(
            tmp_path,
            "mario",
            'dataset = "dataset/mario"\nlanguage = "it"\nspeaker_name = "mario"\n'
            'checkpoint = "models/not_trained_yet"\ndevice = "cpu"\n',
        )
        code, _, err = run(["train", "--profile", "mario"], capsys)
        assert code == 0  # train does not check the checkpoint: it is about to create it
        (args,) = handled.args
        assert args.dataset_dir == tmp_path / "project" / "dataset" / "mario"
        assert (args.language, args.speaker_name) == ("it", "mario")
        assert not hasattr(args, "checkpoint")
        assert "Profile: mario (profiles/mario.toml)" in err

    def test_the_environment_selects_a_profile_too_and_the_option_beats_it(
        self, tmp_path, handled, monkeypatch, capsys
    ):
        make_profile(tmp_path, "mario", 'language = "it"\n')
        make_profile(tmp_path, "luigi", 'language = "es"\n')
        monkeypatch.setenv("HERALD_PROFILE", "mario")
        run(["chat"], capsys)
        run(["chat", "--profile", "luigi"], capsys)
        assert [a.language for a in handled.args] == ["it", "es"]

    def test_a_profile_can_be_given_as_a_path(self, tmp_path, handled, capsys):
        elsewhere = tmp_path / "elsewhere.toml"
        elsewhere.write_text('language = "it"\n', encoding="utf-8")
        code, _, err = run(["chat", "--profile", str(elsewhere)], capsys)
        assert code == 0
        assert handled.args[0].language == "it"
        assert "Profile: elsewhere" in err

    def test_a_reference_list_from_the_profile_is_replaced_not_extended(
        self, tmp_path, handled, capsys
    ):
        for name in ("a.wav", "b.wav", "mine.wav"):
            (tmp_path / "project").mkdir(exist_ok=True)
            (tmp_path / "project" / name).write_bytes(b"RIFF")
        make_profile(tmp_path, "mario", 'reference_wavs = ["a.wav", "b.wav"]\n')
        run(["synthesize", "Hi.", "--profile", "mario"], capsys)
        run(["synthesize", "Hi.", "--profile", "mario", "--reference-wav", "mine.wav"], capsys)
        run(["synthesize", "Hi."], capsys)
        project = tmp_path / "project"
        assert handled.args[0].reference_wav == [project / "a.wav", project / "b.wav"]
        assert handled.args[1].reference_wav == [Path("mine.wav")]  # replaced, not extended
        assert handled.args[2].reference_wav is None  # no profile: as before

    def test_the_profile_is_announced(self, tmp_path, handled, capsys):
        make_profile(tmp_path, "mario")
        _, out, err = run(["synthesize", "Hi.", "--profile", "mario"], capsys)
        assert "Profile: mario (profiles/mario.toml)\n" in err
        assert "Profile" not in out  # stdout is for results
        _, _, err = run(["synthesize", "Hi."], capsys)
        assert "Profile" not in err

    def test_the_profile_reaches_the_voice(self, dataset_dir, tmp_path, fake_load, capsys):
        voice = tmp_path / "project" / "models" / "mario.pth"
        voice.parent.mkdir(parents=True)
        voice.write_bytes(b"x")
        make_profile(
            tmp_path,
            "mario",
            f'checkpoint = "models/mario.pth"\ndataset = "{dataset_dir}"\nlanguage = "it"\n'
            "num_references = 2\n",
        )
        argv = ["synthesize", "Hi.", "--profile", "mario", "-o", str(tmp_path / "o.wav")]
        code, _, _ = run([*argv, "--checkpoint-dir", str(tmp_path / "ckpt")], capsys)
        assert code == 0
        (call,) = fake_load.load_calls
        assert call["finetuned_checkpoint"] == voice
        assert call["language"] == "it"
        assert len(call["reference_wavs"]) == 2
        assert all(str(dataset_dir) in wav for wav in call["reference_wavs"])


class TestProfileErrors:
    @pytest.mark.parametrize("command", ["synthesize", "chat", "train"])
    def test_a_missing_profile_is_a_one_line_error_before_anything_happens(
        self, command, tmp_path, handled, capsys
    ):
        make_profile(tmp_path, "frieren")
        code, out, err = run([command, "--profile", "nope"], capsys)
        assert code == 1
        assert err.count("\n") == 1 and "Profile 'nope' not found" in err
        assert "frieren" in err  # it says which profiles exist
        assert handled.args == [] and out == ""

    def test_a_broken_profile_names_its_file(self, tmp_path, handled, capsys):
        path = make_profile(tmp_path, "mario", "language = it\n")  # not TOML
        code, _, err = run(["chat", "--profile", "mario"], capsys)
        assert code == 1 and str(path) in err and "Traceback" not in err
        assert handled.args == []

    def test_a_typo_in_a_profile_is_an_error(self, tmp_path, handled, capsys):
        make_profile(tmp_path, "mario", 'languge = "it"\n')
        code, _, err = run(["chat", "--profile", "mario"], capsys)
        assert code == 1 and "languge" in err and "language" in err

    @pytest.mark.parametrize("argv", [["--help"], ["chat", "--help"], ["synthesize", "--help"]])
    def test_help_works_with_a_broken_profile(self, argv, tmp_path, monkeypatch, capsys):
        make_profile(tmp_path, "mario", "this is not toml")
        monkeypatch.setenv("HERALD_PROFILE", "mario")
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 0
        out, err = capsys.readouterr()
        assert "usage: herald" in out and err == ""

    def test_help_works_with_a_missing_profile_option(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["chat", "--profile", "nope", "--help"])
        assert exc.value.code == 0
        assert "usage: herald chat" in capsys.readouterr().out

    def test_help_shows_the_defaults_of_a_valid_profile(self, tmp_path, capsys):
        make_profile(tmp_path, "mario", 'language = "it"\nollama_model = "mistral"\n')
        with pytest.raises(SystemExit):
            cli.main(["chat", "--profile", "mario", "--help"])
        out = capsys.readouterr().out
        assert "(default: it)" in out and "(default: mistral)" in out

    def test_a_missing_value_for_the_option_is_a_usage_error(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["chat", "--profile"])
        assert exc.value.code == 2

    def test_other_commands_do_not_look_at_a_broken_profile(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERALD_PROFILE", "nope")
        code, out, err = run(["tools"], capsys)
        assert code == 0 and "set_timer" in out and "nope" not in err
        code, _, err = run(["slim", str(tmp_path / "x.pth")], capsys)
        assert "Profile" not in err  # its own error, not a profile one

    def test_empty_profile_option_means_no_profile(self, tmp_path, handled, monkeypatch, capsys):
        monkeypatch.setenv("HERALD_PROFILE", "nope")
        code, _, _ = run(["chat", "--profile", ""], capsys)
        assert code == 0


class TestProfileChecks:
    """A profile that points to files that are not there fails early, with one error each."""

    def test_missing_files_are_reported_one_per_line(self, tmp_path, handled, capsys):
        make_profile(tmp_path, "mario", 'checkpoint = "models/mario"\ndataset = "dataset/mario"\n')
        code, _, err = run(["chat", "--profile", "mario"], capsys)
        assert code == 1
        lines = [line for line in err.splitlines() if line.startswith("error:")]
        assert len(lines) == 2
        assert "profile mario: checkpoint not found" in lines[0]
        assert "profile mario: dataset directory not found" in lines[1]
        assert handled.args == []  # nothing ran

    def test_synthesize_checks_too(self, tmp_path, handled, capsys):
        make_profile(tmp_path, "mario", 'checkpoint = "models/mario"\n')
        code, _, err = run(["synthesize", "Hi.", "--profile", "mario"], capsys)
        assert code == 1 and "checkpoint not found" in err

    def test_an_explicit_checkpoint_makes_the_profile_checkpoint_moot(
        self, tmp_path, handled, capsys
    ):
        make_profile(tmp_path, "mario", 'checkpoint = "models/mario"\n')
        code, _, err = run(["chat", "--profile", "mario", "--checkpoint", "mine"], capsys)
        assert code == 0 and "error" not in err
        assert handled.args[0].checkpoint == Path("mine")

    def test_an_explicit_dataset_dir_makes_the_profile_dataset_moot(
        self, tmp_path, handled, capsys
    ):
        make_profile(tmp_path, "mario", 'dataset = "dataset/mario"\n')
        code, _, _ = run(["chat", "--profile", "mario", "--dataset-dir", str(tmp_path)], capsys)
        assert code == 0

    def test_explicit_reference_clips_make_the_profile_clips_and_dataset_moot(
        self, tmp_path, handled, capsys
    ):
        make_profile(tmp_path, "mario", 'reference_wavs = ["gone.wav"]\ndataset = "dataset/gone"\n')
        code, _, _ = run(["chat", "--profile", "mario"], capsys)
        assert code == 1  # without the option, both are missing
        code, _, err = run(["chat", "--profile", "mario", "--reference-wav", "mine.wav"], capsys)
        assert code == 0 and "error" not in err

    def test_other_problems_still_count_when_one_is_moot(self, tmp_path, handled, capsys):
        make_profile(tmp_path, "mario", 'checkpoint = "models/mario"\ndataset = "dataset/gone"\n')
        code, _, err = run(["chat", "--profile", "mario", "--checkpoint", "mine"], capsys)
        assert code == 1
        assert "dataset directory not found" in err and "checkpoint not found" not in err

    def test_a_valid_profile_passes(self, dataset_dir, tmp_path, handled, capsys):
        voice = tmp_path / "project" / "models" / "mario"
        voice.mkdir(parents=True)
        (voice / "best_model.pth").write_bytes(b"x")
        make_profile(tmp_path, "mario", f'checkpoint = "models/mario"\ndataset = "{dataset_dir}"\n')
        code, _, err = run(["chat", "--profile", "mario"], capsys)
        assert code == 0 and "error" not in err


class TestProfilesCommand:
    def test_lists_every_profile_with_its_problems(self, dataset_dir, tmp_path, capsys):
        voice = tmp_path / "project" / "models" / "ok"
        voice.mkdir(parents=True)
        (voice / "best_model.pth").write_bytes(b"x")
        make_profile(
            tmp_path,
            "ok",
            'description = "A fine voice"\nlanguage = "it"\ncheckpoint = "models/ok"\n',
        )
        make_profile(tmp_path, "plain", 'description = "The base voice"\n')
        make_profile(tmp_path, "broken", 'checkpoint = "models/gone"\n')
        make_profile(tmp_path, "_template", 'description = "not listed"\n')
        code, out, err = run(["profiles"], capsys)
        assert code == 0 and err == ""
        lines = out.splitlines()
        row = {
            name: next(line for line in lines if f" {name} " in f" {line} ")
            for name in ("ok", "plain", "broken")
        }
        assert row["ok"].startswith("  ") and "it" in row["ok"]
        assert "models/ok" in row["ok"] and "A fine voice" in row["ok"]
        assert "base voice" in row["plain"] and row["plain"].startswith("  ")
        assert row["broken"].startswith("! ")  # flagged...
        assert any("checkpoint not found" in line and "!" in line for line in lines)  # ...and why
        assert "_template" not in out and "not listed" not in out

    def test_no_profiles(self, capsys):
        code, out, _ = run(["profiles"], capsys)
        assert code == 0
        assert "No profiles in" in out and "herald new-profile NAME" in out

    def test_shows_one_profile_with_its_resolved_values(self, tmp_path, capsys):
        make_profile(
            tmp_path,
            "mario",
            'description = "Mario"\ncheckpoint = "models/mario"\nlanguage = "it"\n'
            'system_prompt = "Line one.\\nLine two."\nreference_wavs = ["a.wav", "b.wav"]\n',
        )
        code, out, _ = run(["profiles", "mario"], capsys)
        assert code == 0
        assert out.splitlines()[0] == "mario  (profiles/mario.toml)"
        assert "  checkpoint: models/mario" in out  # shown relative to the project
        assert "  language: it" in out
        assert "  reference_wavs: a.wav, b.wav" in out
        assert "  system_prompt: Line one.\n    Line two." in out
        assert "  ! checkpoint not found" in out  # the problems come with it, exit code still 0

    def test_a_profile_without_checkpoint_is_the_base_voice(self, tmp_path, capsys):
        make_profile(tmp_path, "plain")
        _, out, _ = run(["profiles", "plain"], capsys)
        assert "voice: base voice" in out

    def test_an_unknown_or_broken_profile_is_exit_1(self, tmp_path, capsys):
        make_profile(tmp_path, "mario", "nonsense")
        code, _, err = run(["profiles", "nope"], capsys)
        assert code == 1 and "Profile 'nope' not found" in err
        code, _, err = run(["profiles", "mario"], capsys)
        assert code == 1 and "invalid TOML" in err

    def test_the_directory_comes_from_the_environment(self, tmp_path, monkeypatch, capsys):
        folder = tmp_path / "mine"
        folder.mkdir()
        (folder / "x.toml").write_text('description = "X"\n', encoding="utf-8")
        monkeypatch.setenv("HERALD_PROFILES_DIR", str(folder))
        _, out, _ = run(["profiles"], capsys)
        assert " x " in f" {out} "


class TestNewProfile:
    def written(self, tmp_path, name="mario"):
        return tomllib.loads(
            (tmp_path / "project" / "profiles" / f"{name}.toml").read_text(encoding="utf-8")
        )

    def test_writes_what_was_given_with_paths_relative_to_the_project(
        self, tmp_path, monkeypatch, capsys
    ):
        project = tmp_path / "project"
        monkeypatch.chdir(tmp_path)  # not the project: paths are typed from where you are
        code, out, err = run(
            [
                "new-profile", "mario",
                "--description", "Mario, the plumber",
                "--checkpoint", "project/models/mario",
                "--dataset", str(project / "dataset" / "mario"),
                "--reference-wav", "project/clips/a.wav",
                "--reference-wav", "/somewhere/else/b.wav",
                "--language", "it",
                "--system-prompt", "You are Mario.",
                "--ollama-model", "mistral",
            ],
            capsys,
        )  # fmt: skip
        assert code == 0
        assert self.written(tmp_path) == {
            "description": "Mario, the plumber",
            "checkpoint": "models/mario",
            "dataset": "dataset/mario",
            "reference_wavs": ["clips/a.wav", "/somewhere/else/b.wav"],  # outside: as it is
            "language": "it",
            "system_prompt": "You are Mario.",
            "ollama_model": "mistral",
        }
        assert out.splitlines() == [
            "Created profiles/mario.toml",
            "Use it with: herald chat --profile mario",
        ]
        # The files it points to do not exist yet: said, but not a failure.
        assert "warning: checkpoint not found" in err

    def test_only_the_given_options_are_written(self, tmp_path, capsys):
        code, out, _ = run(["new-profile", "plain", "--language", "en"], capsys)
        assert code == 0
        assert self.written(tmp_path, "plain") == {"language": "en"}

    def test_the_new_profile_can_be_used_at_once(self, tmp_path, handled, capsys):
        run(["new-profile", "mario", "--language", "it", "--ollama-model", "mistral"], capsys)
        code, _, _ = run(["chat", "--profile", "mario"], capsys)
        assert code == 0
        assert (handled.args[0].language, handled.args[0].ollama_model) == ("it", "mistral")

    def test_the_system_prompt_can_come_from_a_file(self, tmp_path, monkeypatch, capsys):
        project = tmp_path / "project"
        project.mkdir()
        (project / "mario.md").write_text("Be Mario.", encoding="utf-8")
        monkeypatch.chdir(project)
        code, _, _ = run(["new-profile", "mario", "--system-prompt-file", "mario.md"], capsys)
        assert code == 0
        assert self.written(tmp_path) == {"system_prompt_file": "mario.md"}

    def test_a_missing_system_prompt_file_is_an_error(self, tmp_path, capsys):
        code, _, err = run(
            ["new-profile", "mario", "--system-prompt-file", str(tmp_path / "gone.md")], capsys
        )
        assert code == 1 and "gone.md" in err
        assert not (tmp_path / "project" / "profiles" / "mario.toml").exists()

    def test_the_two_ways_to_give_the_character_exclude_each_other(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["new-profile", "m", "--system-prompt", "x", "--system-prompt-file", "y"])
        assert exc.value.code == 2

    def test_it_does_not_replace_a_profile_unless_forced(self, tmp_path, capsys):
        run(["new-profile", "mario", "--language", "it"], capsys)
        code, _, err = run(["new-profile", "mario", "--language", "es"], capsys)
        assert code == 1
        assert "profiles/mario.toml already exists" in err and "--force" in err
        assert self.written(tmp_path) == {"language": "it"}  # untouched

        code, out, _ = run(["new-profile", "mario", "--language", "es", "--force"], capsys)
        assert code == 0 and out.startswith("Replaced profiles/mario.toml")
        assert self.written(tmp_path) == {"language": "es"}

    @pytest.mark.parametrize("name", ["bad name", "a/b", "a.b", "été", "x:y"])
    def test_names_are_letters_digits_dash_and_underscore(self, name, tmp_path, capsys):
        code, _, err = run(["new-profile", name], capsys)
        assert code == 1 and "Invalid profile name" in err
        assert not (tmp_path / "project" / "profiles").exists()

    def test_the_directory_comes_from_the_environment(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERALD_PROFILES_DIR", str(tmp_path / "mine"))
        code, out, _ = run(["new-profile", "mario"], capsys)
        assert code == 0
        assert (tmp_path / "mine" / "mario.toml").is_file()

    def test_a_name_is_required(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["new-profile"])
        assert exc.value.code == 2


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
