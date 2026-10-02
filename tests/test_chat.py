"""`herald chat`: the loop, the speech pipeline, the tools. No real model, audio or network.

The voice model, the LLM and the player are fakes. The speaker is a recording fake in most tests
and the real `Speaker` (with a fake engine and player) in the pipeline tests at the end.
Everything that crosses a thread is synchronised with events, never with sleeps.
"""

import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from herald import chat, cli
from herald.config import DEFAULT_OLLAMA_TIMEOUT
from herald.errors import OllamaError
from herald.tools import LoadReport, ToolContext, ToolRegistry
from herald.tools.scheduler import Scheduler

REAL_IN_BACKGROUND = chat._in_background
WAIT = 10.0  # an upper bound for something that takes milliseconds

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


# --- fixtures and fakes ------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Keep the host's HERALD_* variables out, and write outputs under tmp_path."""
    for name in list(os.environ):
        if name.startswith("HERALD_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HERALD_PROJECT_ROOT", str(tmp_path / "project"))


@pytest.fixture
def log():
    """What happened, in order, across the fakes: the thing to look at for ordering."""
    return []


@pytest.fixture(autouse=True)
def no_network_and_no_threads(monkeypatch, log):
    """The Ollama warm-up is a recording fake (never an HTTP call) and background work runs at
    once on the calling thread, so that the order of events is deterministic."""
    from herald.llm.ollama_client import OllamaClient

    monkeypatch.setattr(OllamaClient, "warm_up", lambda self: log.append("client.warm_up"))
    monkeypatch.setattr(chat, "_in_background", lambda func: func())


@pytest.fixture(autouse=True)
def scratch_root(tmp_path, monkeypatch):
    """Where tempfile puts the chat's scratch directory, so that tests can look inside."""
    root = tmp_path / "scratch"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def run(argv, capsys):
    code = cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


class FakeEngine:
    """Synthesizes silence. Enough of an XttsEngine for the Speaker and for the chat."""

    language = "en"
    sample_rate = 1000

    def __init__(self, log):
        self.log = log
        self.spoken = []

    def synthesize(self, text):
        self.spoken.append(text)
        return np.zeros(100, dtype=np.float32)

    def warm_up(self):
        self.log.append("engine.warm_up")
        return 0.0


@pytest.fixture
def fake_load(monkeypatch, log):
    """Replace engine.load_engine; records its calls and returns a FakeEngine."""
    from herald.tts import engine

    fake = FakeEngine(log)
    fake.load_calls = []

    def load_engine(checkpoint_dir, reference_wavs, **kwargs):
        log.append("engine.load")
        fake.load_calls.append({"checkpoint_dir": checkpoint_dir, "reference_wavs": reference_wavs})
        return fake

    monkeypatch.setattr(engine, "load_engine", load_engine)
    return fake


class FakePlayer:
    """Records what is played. The real one blocks until the end of the file."""

    available = True
    was_stopped = False

    def __init__(self):
        self.played = []

    def play(self, path):
        self.played.append((Path(path), Path(path).is_file()))
        return True

    def stop(self):
        pass


@pytest.fixture
def players(monkeypatch):
    """Replace Player (the real one looks for afplay/aplay on the host)."""
    made = []

    def make():
        made.append(FakePlayer())
        return made[-1]

    monkeypatch.setattr("herald.audio_playback.Player", make)
    return made


class FakeSpeaker:
    """Stands in for the Speaker and writes what it is asked to do into the shared log."""

    instances = []

    def __init__(self, engine, **kwargs):
        self.engine = engine
        self.kwargs = kwargs
        self.log = kwargs.pop("log_to")
        self.say_error = None
        self.scratch_when_closed = None
        FakeSpeaker.instances.append(self)

    def feed(self, delta):
        self.log.append(("feed", delta))

    def end_utterance(self, name=None):
        self.log.append(("end", name))

    def say(self, text, name=None):
        if self.say_error:
            raise self.say_error
        self.log.append(("say", text, name))

    def cancel(self):
        self.log.append("cancel")

    def wait(self, timeout=None):
        self.log.append("wait")
        return True

    def close(self):
        self.log.append("close")
        self.scratch_when_closed = Path(self.kwargs["scratch_dir"]).is_dir()


@pytest.fixture
def fake_speaker(monkeypatch, log, players):
    """Replace the Speaker class with a recording fake; the instances made are in ``.instances``."""
    FakeSpeaker.instances = []

    class Made(FakeSpeaker):
        def __init__(self, engine, **kwargs):
            super().__init__(engine, log_to=log, **kwargs)

    monkeypatch.setattr("herald.speaker.Speaker", Made)
    return SimpleNamespace(instances=FakeSpeaker.instances)


@pytest.fixture
def assistant(monkeypatch, log):
    """Replace Assistant with a script. A reply is streamed to ``on_text`` word by word; an
    exception is raised instead; ``(text, exception)`` streams ``text`` and then raises."""
    from herald import assistant as assistant_module

    seen = SimpleNamespace(created=[], said=[], snapshots=[])
    script = iter(["First reply.", OllamaError("Ollama is down"), "Third reply.", "Fourth."])
    seen.script = script

    class FakeAssistant:
        def __init__(self, llm, system_prompt, **kwargs):
            seen.created.append({"llm": llm, "system_prompt": system_prompt, **kwargs})

        def respond(self, user_text, on_text=None):
            seen.said.append(user_text)
            log.append(("respond", user_text))
            step = next(seen.script)
            partial, error = step if isinstance(step, tuple) else (None, step)
            text = partial if partial is not None else (step if isinstance(step, str) else "")
            for piece in re.findall(r"\S+\s*", text):
                on_text(piece)
                seen.snapshots.append(sys.stdout.getvalue())  # what capsys has so far
            if isinstance(error, Exception):
                raise error
            return text

    monkeypatch.setattr(assistant_module, "Assistant", FakeAssistant)
    return seen


@pytest.fixture
def typed(monkeypatch):
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


def chat_args(dataset_dir, tmp_path, *extra, tools=False):
    """``herald chat`` for a test dataset. Without ``tools=True`` the chat is plain (no tools)."""
    argv = ["chat", "--dataset-dir", str(dataset_dir), "--checkpoint-dir", str(tmp_path / "ckpt")]
    return [*argv, *extra] if tools else [*argv, "--no-tools", *extra]


# --- start-up -----------------------------------------------------------------------------------


class TestStartup:
    def test_both_models_are_warmed_up_in_the_background_in_the_right_order(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, log, capsys
    ):
        typed()
        run(chat_args(dataset_dir, tmp_path), capsys)
        # Ollama starts loading its model first, so that it loads while XTTS loads; the voice
        # warm-up follows the load (the first synthesis is slow, so it is done before it is needed).
        warm = [entry for entry in log if isinstance(entry, str) and "." in entry]
        assert warm == ["client.warm_up", "engine.load", "engine.warm_up"]

    def test_warm_ups_run_on_daemon_threads(self):
        done = threading.Event()
        seen = {}

        def work():
            seen["thread"] = threading.current_thread()
            done.set()

        REAL_IN_BACKGROUND(work)
        assert done.wait(WAIT)
        assert seen["thread"] is not threading.main_thread()
        assert seen["thread"].daemon  # a warm-up never keeps the program from exiting

    def test_the_ollama_warm_up_also_happens_in_a_text_only_chat(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, log, capsys
    ):
        typed()
        run(chat_args(dataset_dir, tmp_path, "--no-play"), capsys)
        assert log[0] == "client.warm_up"
        assert "engine.load" not in log

    def test_the_speaker_is_set_up_like_the_options_say(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, players, capsys
    ):
        typed()
        run(chat_args(dataset_dir, tmp_path, "--pause-ms", "50", "--max-chars", "120"), capsys)
        (speaker,) = fake_speaker.instances
        kwargs = speaker.kwargs
        assert speaker.engine is fake_load
        assert kwargs["player"] is players[0]  # it plays
        assert kwargs["save_dir"] is None
        assert (kwargs["pause_ms"], kwargs["max_chars"]) == (50, 120)
        assert kwargs["scratch_dir"].parent == tmp_path / "scratch"  # a per-session directory
        assert kwargs["scratch_dir"].name.startswith("herald-chat-")

    def test_max_chars_is_left_to_the_speaker_unless_given(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, capsys
    ):
        typed()
        run(chat_args(dataset_dir, tmp_path), capsys)
        assert fake_speaker.instances[0].kwargs["max_chars"] is None  # the limit of the language

    def test_save_dir_alone_plays_and_saves(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, players, capsys
    ):
        typed()
        run(chat_args(dataset_dir, tmp_path, "--save-dir", str(tmp_path / "wavs")), capsys)
        kwargs = fake_speaker.instances[0].kwargs
        assert kwargs["player"] is players[0]
        assert kwargs["save_dir"] == tmp_path / "wavs"

    def test_no_play_with_save_dir_only_saves(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, players, capsys
    ):
        typed()
        argv = chat_args(dataset_dir, tmp_path, "--no-play", "--save-dir", str(tmp_path / "wavs"))
        code, _, err = run(argv, capsys)
        assert code == 0
        kwargs = fake_speaker.instances[0].kwargs
        assert kwargs["player"] is None and players == []  # nothing to play with
        assert kwargs["save_dir"] == tmp_path / "wavs"
        assert "text only" not in err

    def test_no_play_alone_is_a_text_only_chat(
        self, tmp_path, fake_load, fake_speaker, assistant, typed, scratch_root, capsys
    ):
        typed("a", "b", "c")
        # No dataset and no weights: nothing is needed because nothing is spoken.
        code, out, err = run(chat_args(tmp_path / "no_dataset", tmp_path, "--no-play"), capsys)
        assert code == 0
        assert "Herald: First reply." in out and "Herald: Third reply." in out
        assert fake_load.load_calls == []  # the model is never loaded...
        assert fake_speaker.instances == []  # ...and there is no speaker
        assert err.count("chatting in text only") == 1
        assert "Audio is neither played nor saved (--no-play without --save-dir)" in err
        assert "Add --save-dir DIR to keep WAV files." in err
        assert list(scratch_root.iterdir()) == []

    def test_the_scratch_directory_lives_as_long_as_the_session(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, scratch_root, capsys
    ):
        typed("a")
        run(chat_args(dataset_dir, tmp_path), capsys)
        (speaker,) = fake_speaker.instances
        assert speaker.scratch_when_closed is True  # the speaker is closed before it is removed
        assert list(scratch_root.iterdir()) == []  # and nothing is left behind

    def test_no_dataset_is_a_helpful_error_before_anything_is_loaded(
        self, tmp_path, fake_load, fake_speaker, assistant, typed, capsys
    ):
        code, _, err = run(chat_args(tmp_path / "nope", tmp_path), capsys)
        assert code == 1 and "No dataset found" in err
        assert fake_speaker.instances == []

    def test_no_player_notice_says_where_the_replies_are_only_if_they_are_kept(self):
        plain = chat._no_player_notice(None)
        assert plain == (
            "No audio player available; use --save-dir DIR to keep the replies as WAV files."
        )
        assert chat._no_player_notice(Path("wavs")) == (
            "No audio player available; replies are saved in wavs"
        )

    def test_speech_errors_and_the_missing_player_reach_stderr(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, capsys
    ):
        typed()
        run(chat_args(dataset_dir, tmp_path), capsys)
        kwargs = fake_speaker.instances[0].kwargs
        kwargs["on_error"]("could not speak the reply: model exploded")
        kwargs["on_no_player"]()
        err = capsys.readouterr().err
        assert "error: could not speak the reply: model exploded\n" in err
        assert "No audio player available; use --save-dir DIR" in err


# --- the conversation ---------------------------------------------------------------------------


class TestConversation:
    def test_the_reply_is_printed_as_it_arrives_and_fed_to_the_speaker_in_order(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, log, capsys
    ):
        typed("Hello")
        code, out, _ = run(chat_args(dataset_dir, tmp_path), capsys)
        assert code == 0
        # Piece by piece: when the second word arrives the first one is already on the screen.
        assert assistant.snapshots[0].endswith("Herald: First ")
        assert assistant.snapshots[1].endswith("Herald: First reply.")
        assert "Herald: First reply.\n" in out
        fed = [entry[1] for entry in log if isinstance(entry, tuple) and entry[0] == "feed"]
        assert fed == ["First ", "reply."]

    def test_every_reply_ends_with_a_named_utterance(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, log, capsys
    ):
        typed("a", "b", "c")  # the second turn fails
        run(chat_args(dataset_dir, tmp_path), capsys)
        names = [entry[1] for entry in log if isinstance(entry, tuple) and entry[0] == "end"]
        assert [re.fullmatch(r"chat_\d{8}_\d{6}_(\d{3})", n).group(1) for n in names] == [
            "001",
            "003",
        ]  # numbered by turn, and the turn that failed has none

    def test_a_new_message_cuts_the_speech_off_before_it_is_answered(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, log, capsys
    ):
        typed("a", "b", "c")
        run(chat_args(dataset_dir, tmp_path), capsys)
        turns = [
            i for i, entry in enumerate(log) if isinstance(entry, tuple) and entry[0] == "respond"
        ]
        assert len(turns) == 3
        for index in turns:
            assert log[index - 1] == "cancel"  # barge-in, right before the LLM is asked

    def test_the_speech_is_not_waited_for_between_turns_but_at_the_end(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, log, capsys
    ):
        typed("a", "b", "c")
        run(chat_args(dataset_dir, tmp_path), capsys)
        assert log.count("wait") == 1  # only once: the loop goes straight back to input()
        last_end = max(i for i, e in enumerate(log) if isinstance(e, tuple) and e[0] == "end")
        assert last_end < log.index("wait") < log.index("close")

    def test_an_empty_line_ends_the_chat_and_lets_the_last_reply_finish(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, log, capsys
    ):
        typed("a", "")
        code, _, _ = run(chat_args(dataset_dir, tmp_path), capsys)
        assert code == 0
        assert "wait" in log

    def test_an_ollama_error_cancels_the_speech_and_the_chat_goes_on(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, log, capsys
    ):
        assistant.script = iter([("A part ", OllamaError("cut off")), "Whole reply."])
        typed("a", "b")
        code, out, err = run(chat_args(dataset_dir, tmp_path), capsys)
        assert code == 0
        assert "Herald: A part \n" in out  # the line is finished, whatever came before
        assert "error: cut off" in err
        assert "Herald: Whole reply.\n" in out  # the next turn is answered normally
        first_turn = log[: log.index(("respond", "b"))]
        assert ("feed", "part ") in first_turn
        assert first_turn[-2:] == ["cancel", "cancel"]  # one for the error, one before turn two
        assert [e[1] for e in log if isinstance(e, tuple) and e[0] == "end"][0].endswith("_002")

    def test_ctrl_c_cancels_the_speech_and_leaves_without_waiting(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, monkeypatch, log, capsys
    ):
        lines = iter(["a"])

        def input_then_interrupt(prompt=""):
            try:
                return next(lines)
            except StopIteration:
                raise KeyboardInterrupt from None

        monkeypatch.setattr("builtins.input", input_then_interrupt)
        code, _, _ = run(chat_args(dataset_dir, tmp_path), capsys)
        assert code == 130
        assert "wait" not in log
        assert log[-2:] == ["cancel", "close"]  # cancelled, then closed with the session

    def test_ctrl_c_while_waiting_for_the_last_reply_cancels_it(
        self,
        dataset_dir,
        tmp_path,
        fake_load,
        fake_speaker,
        assistant,
        typed,
        monkeypatch,
        log,
        capsys,
    ):
        typed("a")

        def interrupted(self, timeout=None):
            raise KeyboardInterrupt

        monkeypatch.setattr(FakeSpeaker, "wait", interrupted)
        code, _, _ = run(chat_args(dataset_dir, tmp_path), capsys)
        assert code == 130
        assert log[-2:] == ["cancel", "close"]

    def test_the_text_only_chat_prints_in_the_same_way(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, capsys
    ):
        typed("a", "b", "c")
        code, out, err = run(chat_args(dataset_dir, tmp_path, "--no-play"), capsys)
        assert code == 0
        assert assistant.snapshots[0].endswith("Herald: First ")
        assert "Herald: First reply.\n" in out and "Herald: Third reply.\n" in out
        assert "error: Ollama is down" in err  # the failed turn does not end the session

    def test_the_system_prompt_history_and_client_settings_reach_the_assistant(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, capsys
    ):
        typed()
        extra = ["--ollama-url", "http://ollama:11434", "--ollama-model", "mistral"]
        extra += ["--ollama-timeout", "9", "--system-prompt", "Be brief.", "--history", "3"]
        run(chat_args(dataset_dir, tmp_path, *extra), capsys)
        (created,) = assistant.created
        client = created["llm"]
        assert (client.base_url, client.model, client.timeout) == (
            "http://ollama:11434",
            "mistral",
            9,
        )
        assert created["system_prompt"] == "Be brief."  # the Assistant owns the system prompt
        assert created["history_turns"] == 3

    @pytest.mark.parametrize(
        ("extra", "turns"), [([], 10), (["--history", "3"], 3), (["--history", "0"], 0)]
    )
    def test_history_setting(
        self, extra, turns, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, capsys
    ):
        typed()
        run(chat_args(dataset_dir, tmp_path, *extra), capsys)
        assert assistant.created[0]["history_turns"] == turns

    def test_system_prompt_from_a_file_with_a_byte_order_mark(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, capsys
    ):
        prompt_file = tmp_path / "notepad.txt"
        prompt_file.write_bytes(b"\xef\xbb\xbf  You are a pirate.\r\n")
        typed()
        run(chat_args(dataset_dir, tmp_path, "--system-prompt-file", str(prompt_file)), capsys)
        assert assistant.created[0]["system_prompt"] == "You are a pirate."

    def test_system_prompt_file_must_be_utf8(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, capsys
    ):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_bytes(b"\xff\xfe\x00")
        code, _, err = run(
            chat_args(dataset_dir, tmp_path, "--system-prompt-file", str(prompt_file)), capsys
        )
        assert code == 1
        assert "not a UTF-8 text file" in err
        assert fake_load.load_calls == []  # fails before the model is loaded

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
        fake_speaker,
        assistant,
        typed,
        monkeypatch,
        capsys,
    ):
        if env is not None:
            monkeypatch.setenv("HERALD_OLLAMA_TIMEOUT", env)
        typed()
        extra = ["--ollama-timeout", option] if option else []
        code, _, _ = run(chat_args(dataset_dir, tmp_path, *extra), capsys)
        assert code == 0
        assert assistant.created[0]["llm"].timeout == expected

    def test_a_bad_ollama_timeout_in_the_environment_stops_chat_before_the_model_loads(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, monkeypatch, capsys
    ):
        monkeypatch.setenv("HERALD_OLLAMA_TIMEOUT", "soon")
        code, _, err = run(chat_args(dataset_dir, tmp_path), capsys)
        assert code == 1 and "HERALD_OLLAMA_TIMEOUT" in err
        assert fake_load.load_calls == []


# --- alerts (what a timer says) -----------------------------------------------------------------


class TestAlerts:
    def test_an_alert_is_printed_and_goes_through_the_speaker(self, log, capsys):
        speaker = SimpleNamespace(say=lambda text, name=None: log.append(("say", text, name)))
        announce = chat._make_announcer(speaker, "20260101_120000")
        announce("Your tea is ready.")
        announce("Again.")
        assert capsys.readouterr().out == "\a\n[Herald] Your tea is ready.\n\a\n[Herald] Again.\n"
        assert log == [
            ("say", "Your tea is ready.", "alert_20260101_120000_001"),
            ("say", "Again.", "alert_20260101_120000_002"),
        ]

    def test_a_text_only_chat_prints_the_alert_and_does_not_speak(self, capsys):
        chat._make_announcer(None, "s")("Your tea is ready.")
        assert capsys.readouterr().out == "\a\n[Herald] Your tea is ready.\n"

    def test_a_failing_speaker_never_raises_out_of_the_timer_thread(self, capsys):
        def broken(text, name=None):
            raise RuntimeError("model exploded")

        chat._make_announcer(SimpleNamespace(say=broken), "s")("Your tea is ready.")
        captured = capsys.readouterr()
        assert "[Herald] Your tea is ready." in captured.out  # the text still gets through
        assert "error: could not speak the reply: model exploded" in captured.err


# --- the tools ----------------------------------------------------------------------------------


class TestTools:
    @pytest.fixture
    def load_calls(self, monkeypatch):
        """Spy on load_tools: records (tools_dir, context) and runs the real thing."""
        import herald.tools

        calls = []
        real = herald.tools.load_tools

        def load_tools(user_dir, context, **kwargs):
            calls.append((user_dir, context))
            return real(user_dir, context, **kwargs)

        monkeypatch.setattr(herald.tools, "load_tools", load_tools)
        return calls

    @pytest.fixture
    def scheduler(self, monkeypatch, log):
        """A Scheduler that records its shutdown instead of running timers."""
        import herald.tools.scheduler

        seen = SimpleNamespace(shutdowns=0, pending=0)

        class FakeScheduler:
            def shutdown(self):
                seen.shutdowns += 1
                log.append("stop_timers")
                return seen.pending

        monkeypatch.setattr(herald.tools.scheduler, "Scheduler", FakeScheduler)
        return seen

    def tool_names(self, registry):
        return [schema["function"]["name"] for schema in registry.schemas()]

    def test_the_tools_are_offered_to_the_assistant(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, load_calls, capsys
    ):
        typed()
        code, out, _ = run(chat_args(dataset_dir, tmp_path, tools=True), capsys)
        assert code == 0
        ((user_dir, context),) = load_calls
        assert user_dir == tmp_path / "project" / "tools"  # the default tools directory
        assert isinstance(context, ToolContext) and isinstance(context.scheduler, Scheduler)
        assert self.tool_names(assistant.created[0]["tools"]) == ["set_timer"]
        assert "Tools: set_timer\n" in out

    def test_tools_dir_option_and_environment(
        self,
        dataset_dir,
        tmp_path,
        fake_load,
        fake_speaker,
        assistant,
        typed,
        load_calls,
        monkeypatch,
        capsys,
    ):
        typed()
        monkeypatch.setenv("HERALD_TOOLS_DIR", str(tmp_path / "from_env"))
        run(chat_args(dataset_dir, tmp_path, tools=True), capsys)
        run(
            chat_args(dataset_dir, tmp_path, "--tools-dir", str(tmp_path / "mine"), tools=True),
            capsys,
        )
        assert [d for d, _ in load_calls] == [tmp_path / "from_env", tmp_path / "mine"]

    def test_your_own_tool_scripts_are_loaded(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, capsys
    ):
        tools_dir = tmp_path / "mine"
        tools_dir.mkdir()
        (tools_dir / "shout.py").write_text(SHOUT_TOOL, encoding="utf-8")
        typed()
        argv = chat_args(dataset_dir, tmp_path, "--tools-dir", str(tools_dir), tools=True)
        _, out, _ = run(argv, capsys)
        assert "Tools: set_timer, shout\n" in out
        assert self.tool_names(assistant.created[0]["tools"]) == ["set_timer", "shout"]

    def test_no_tools(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, load_calls, capsys
    ):
        typed("a")
        code, out, _ = run(chat_args(dataset_dir, tmp_path), capsys)  # --no-tools
        assert code == 0
        assert load_calls == []
        assert assistant.created[0]["tools"] is None
        assert "Tools:" not in out
        assert "Herald: First reply." in out  # a plain chat

    @pytest.mark.parametrize(
        ("extra", "use_triggers"),
        [
            ([], True),  # a tool with trigger words is only offered when one of them is said
            (["--always-offer-tools"], False),
            (["--no-tools", "--always-offer-tools"], False),  # harmless without tools
        ],
    )
    def test_always_offer_tools(
        self,
        extra,
        use_triggers,
        dataset_dir,
        tmp_path,
        fake_load,
        fake_speaker,
        assistant,
        typed,
        capsys,
    ):
        typed()
        code, _, _ = run(chat_args(dataset_dir, tmp_path, *extra, tools=True), capsys)
        assert code == 0
        assert assistant.created[0]["use_triggers"] is use_triggers

    def test_no_tools_found_means_no_registry_and_no_line(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, monkeypatch, capsys
    ):
        import herald.tools

        nothing = LoadReport(ToolRegistry(), (), ())
        monkeypatch.setattr(herald.tools, "load_tools", lambda user_dir, context, **kw: nothing)
        typed()
        code, out, err = run(chat_args(dataset_dir, tmp_path, tools=True), capsys)
        assert code == 0
        assert assistant.created[0]["tools"] is None
        assert "Tools:" not in out and "warning" not in err

    def test_a_broken_tool_script_is_a_warning_not_the_end_of_the_chat(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, capsys
    ):
        tools_dir = tmp_path / "mine"
        tools_dir.mkdir()
        (tools_dir / "broken.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
        (tools_dir / "shout.py").write_text(SHOUT_TOOL, encoding="utf-8")
        typed("a")
        argv = chat_args(dataset_dir, tmp_path, "--tools-dir", str(tools_dir), tools=True)
        code, out, err = run(argv, capsys)
        assert code == 0
        assert err.count("warning: tools: ") == 1 and "boom" in err
        assert "Tools: set_timer, shout\n" in out  # the good ones still load
        assert "Herald: First reply." in out

    def test_timers_are_cancelled_when_the_chat_ends_before_the_speaker_closes(
        self,
        dataset_dir,
        tmp_path,
        fake_load,
        fake_speaker,
        assistant,
        typed,
        scheduler,
        log,
        capsys,
    ):
        scheduler.pending = 2
        typed("a", "")
        code, _, err = run(chat_args(dataset_dir, tmp_path, tools=True), capsys)
        assert code == 0
        assert scheduler.shutdowns == 1
        assert (
            "2 timer(s) were still running and have been cancelled because the chat ended." in err
        )
        assert log.index("stop_timers") < log.index(
            "close"
        )  # no timer speaks into a closed speaker

    def test_no_message_when_no_timer_was_running(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, scheduler, capsys
    ):
        typed()
        _, _, err = run(chat_args(dataset_dir, tmp_path, tools=True), capsys)
        assert scheduler.shutdowns == 1
        assert "timer(s)" not in err

    def test_timers_are_cancelled_on_ctrl_c_too(
        self,
        dataset_dir,
        tmp_path,
        fake_load,
        fake_speaker,
        assistant,
        scheduler,
        monkeypatch,
        capsys,
    ):
        scheduler.pending = 1

        def interrupted(prompt=""):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", interrupted)
        code, _, err = run(chat_args(dataset_dir, tmp_path, tools=True), capsys)
        assert code == 130
        assert scheduler.shutdowns == 1
        assert "1 timer(s) were still running" in err

    def test_no_scheduler_without_tools(
        self, dataset_dir, tmp_path, fake_load, fake_speaker, assistant, typed, scheduler, capsys
    ):
        typed()
        run(chat_args(dataset_dir, tmp_path), capsys)
        assert scheduler.shutdowns == 0


# --- the real pipeline: real Speaker, real Assistant and OllamaClient, fake engine/player/HTTP --


class FakeHTTP:
    """What ``requests.post`` returns: a JSON body, or the lines of a streamed reply."""

    def __init__(self, body=None, chunks=None):
        self.ok, self.status_code = True, 200
        self._body, self._chunks = body, chunks
        self.text = json.dumps(body if body is not None else chunks)

    def json(self):
        return self._body

    def iter_lines(self):
        return iter(json.dumps(chunk).encode() for chunk in self._chunks)

    def close(self):
        pass


def streamed(*pieces):
    message = {"role": "assistant"}
    chunks = [{"message": {**message, "content": piece}, "done": False} for piece in pieces]
    return FakeHTTP(chunks=[*chunks, {"message": {**message, "content": ""}, "done": True}])


class TestPipeline:
    def test_the_streamed_reply_is_spoken_sentence_by_sentence(
        self, dataset_dir, tmp_path, fake_load, players, typed, mocker, scratch_root, capsys
    ):
        post = mocker.patch(
            "requests.post", return_value=streamed("Hello there. ", "How are ", "you today?")
        )
        typed("Hi")
        code, out, _ = run(chat_args(dataset_dir, tmp_path), capsys)

        assert code == 0
        assert "Herald: Hello there. How are you today?\n" in out
        assert post.call_args.kwargs["stream"] is True
        # The whole reply was spoken (the end of the chat waits for it), in sentence-sized pieces.
        assert " ".join(fake_load.spoken) == "Hello there. How are you today?"
        assert len(fake_load.spoken) >= 2
        (player,) = players
        assert len(player.played) == len(fake_load.spoken)
        assert all(existed for _, existed in player.played)  # each piece was a real file...
        assert all(path.parent.parent == scratch_root for path, _ in player.played)
        assert list(scratch_root.iterdir()) == []  # ...and nothing is left behind

    def test_the_reply_is_saved_as_one_file_when_asked_to(
        self, dataset_dir, tmp_path, fake_load, players, typed, mocker, capsys
    ):
        mocker.patch("requests.post", return_value=streamed("Hello there. ", "How are you?"))
        typed("Hi")
        wavs = tmp_path / "wavs"
        run(chat_args(dataset_dir, tmp_path, "--no-play", "--save-dir", str(wavs)), capsys)
        (saved,) = wavs.glob("chat_*_001.wav")  # one file for the whole reply
        assert saved.stat().st_size > 44
        assert players == []  # --no-play: nothing was played

    def test_history_stays_clean_after_an_ollama_error(
        self, dataset_dir, tmp_path, fake_load, players, typed, mocker, capsys
    ):
        post = mocker.patch(
            "requests.post",
            side_effect=[
                streamed("First."),
                requests.ConnectionError("down"),
                streamed("Third."),
            ],
        )
        typed("one", "two", "three")
        code, out, err = run(chat_args(dataset_dir, tmp_path), capsys)

        assert code == 0
        assert "Herald: First.\n" in out and "Herald: Third.\n" in out
        assert "error: Cannot connect to Ollama" in err
        sent = [call.kwargs["json"]["messages"] for call in post.call_args_list]
        assert [m["content"] for m in sent[2]][1:] == ["one", "First.", "three"]  # no "two"

    def test_a_timer_speaks_while_the_chat_waits_for_input(
        self, dataset_dir, tmp_path, fake_load, players, mocker, monkeypatch, capsys
    ):
        """Real Assistant, tools, Scheduler and Speaker; only the HTTP call, the voice model, the
        player and the terminal are fake."""
        call = {"function": {"name": "set_timer", "arguments": {"seconds": 1, "message": "Tea."}}}
        mocker.patch(
            "requests.post",
            side_effect=[
                FakeHTTP(
                    body={"message": {"role": "assistant", "content": "", "tool_calls": [call]}}
                ),
                FakeHTTP(body={"message": {"role": "assistant", "content": "Timer set."}}),
            ],
        )
        alert_spoken = threading.Event()
        real_synthesize = fake_load.synthesize

        def synthesize(text):
            samples = real_synthesize(text)
            if text == "Tea.":
                alert_spoken.set()
            return samples

        fake_load.synthesize = synthesize
        lines = iter(["Set a one second timer for tea."])

        def input_then_wait(prompt=""):
            try:
                return next(lines)
            except StopIteration:
                # The main thread sits in input() while the timer thread fires, like a real chat.
                assert alert_spoken.wait(WAIT), "the timer never spoke"
                return ""

        monkeypatch.setattr("builtins.input", input_then_wait)

        code, out, err = run(chat_args(dataset_dir, tmp_path, tools=True), capsys)

        assert code == 0
        assert "Tools: set_timer" in out
        assert "Herald: Timer set.\n" in out
        assert "[Herald] Tea." in out  # printed by the timer thread...
        assert "Tea." in fake_load.spoken  # ...and spoken
        assert "timer(s)" not in err  # it had already fired when the chat ended
