"""The speech pipeline with a fake engine and a fake player. Nothing is synthesized or played.

Everything is synchronised with events and queues that have timeouts, never with sleeps.
"""

import queue
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from herald import audio_playback
from herald.audio_playback import Player
from herald.speaker import Speaker

WAIT = 5.0  # generous upper bound for something that should take milliseconds
SAMPLE_RATE = 1000


class FakeEngine:
    """Synthesizes ``len(text)`` samples. Can hold or fail a piece, and records everything."""

    def __init__(self, language="en"):
        self.language = language
        self.sample_rate = SAMPLE_RATE
        self.calls = []
        self.threads = set()
        self.gates = {}  # text -> Event that must be set before that synthesis returns
        self.failures = {}  # text -> exception to raise
        self._entered = {}
        self._lock = threading.Lock()

    def entered(self, text):
        """An Event that is set once the synthesis of ``text`` has started."""
        with self._lock:
            return self._entered.setdefault(text, threading.Event())

    def synthesize(self, text):
        self.calls.append(text)
        self.threads.add(threading.current_thread().name)
        self.entered(text).set()
        gate = self.gates.get(text)
        if gate is not None:
            assert gate.wait(WAIT)
        if text in self.failures:
            raise self.failures[text]
        return samples_for(text)


def samples_for(text):
    return np.array([(ord(c) % 100) / 200 for c in text], dtype=np.float32)


class FakePlayer:
    """Records what is played (reading the WAV file while it exists) and can hold a playback."""

    def __init__(self, available=True):
        self.available = available
        self.was_stopped = False
        self.frames = []  # frame count of each played file
        self.threads = set()
        self.starts = queue.Queue()  # one item per play() call, as soon as it starts
        self.stop_calls = 0
        self.hold = False  # block each play() until release() or stop()
        self.fail = False  # report a failure that is not a stop
        self.raises = None
        self._release = threading.Event()
        self._playing = False

    def play(self, path):
        with wave.open(str(path), "rb") as f:
            self.frames.append(f.getnframes())
        self.threads.add(threading.current_thread().name)
        self.was_stopped = False
        self._playing = True
        self.starts.put(self.frames[-1])
        try:
            if self.raises:
                raise self.raises
            if self.hold:
                assert self._release.wait(WAIT)
                if self.was_stopped:
                    return False
            return not self.fail
        finally:
            self._playing = False

    def stop(self):
        self.stop_calls += 1
        if self._playing:  # like the real player: idle means nothing to stop
            self.was_stopped = True
            self._release.set()

    def release(self):
        self.hold = False
        self._release.set()

    def wait_started(self):
        return self.starts.get(timeout=WAIT)


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def player():
    return FakePlayer()


@pytest.fixture
def scratch(tmp_path):
    return tmp_path / "scratch"


@pytest.fixture
def make(engine, player, scratch, tmp_path):
    """Build a Speaker (closed at the end of the test) with the fake engine and player."""
    speakers = []

    def make(**kwargs):
        kwargs.setdefault("player", player)
        kwargs.setdefault("scratch_dir", scratch)
        kwargs.setdefault("engine", engine)
        speaker = Speaker(kwargs.pop("engine"), **kwargs)
        speakers.append(speaker)
        return speaker

    yield make
    for speaker in speakers:
        speaker.close()


def read_wav(path):
    with wave.open(str(path), "rb") as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, SAMPLE_RATE)
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")


FIRST, SECOND, THIRD = "First sentence here.", "Second sentence here.", "Third sentence here."
TWO = f"{FIRST} {SECOND}"
THREE = f"{TWO} {THIRD}"
# By default the second and third sentence are merged into one piece (too short to stand alone).
# With this limit each of the three is a piece of its own.
THREE_PIECES = {"max_chars": 25}


class TestSpeaking:
    def test_pieces_are_synthesized_and_played_in_order(self, make, engine, player, scratch):
        speaker = make(**THREE_PIECES)
        speaker.say(THREE)

        assert speaker.wait(WAIT)
        assert engine.calls == [FIRST, SECOND, THIRD]
        assert player.frames == [len(FIRST), len(SECOND), len(THIRD)]
        assert list(scratch.iterdir()) == []  # every piece file is deleted right after playing

    def test_short_sentences_are_merged_into_one_piece(self, make, engine):
        speaker = make()
        speaker.say(THREE)
        assert speaker.wait(WAIT)
        assert engine.calls == [FIRST, f"{SECOND} {THIRD}"]  # the first one is not held back

    def test_synthesis_and_playback_run_on_their_own_threads(self, make, engine, player):
        speaker = make()
        speaker.say(TWO)
        assert speaker.wait(WAIT)
        assert engine.threads == {"herald-speaker-synth"}
        assert player.threads == {"herald-speaker-play"}

    def test_the_next_piece_is_synthesized_while_the_previous_one_plays(self, make, engine, player):
        """The point of the whole design: no gap between sentences."""
        player.hold = True
        speaker = make()
        speaker.say(TWO)

        assert player.wait_started() == len(FIRST)  # the first piece is playing...
        assert engine.entered(SECOND).wait(WAIT)  # ...and the second is already being made
        assert not speaker.wait(0)  # nothing has finished yet
        player.release()

        assert speaker.wait(WAIT)
        assert player.frames == [len(FIRST), len(SECOND)]

    def test_playback_starts_before_the_text_has_ended(self, make, engine, player):
        speaker = make()
        speaker.feed("First sentence here. Seco")

        assert player.wait_started() == len(FIRST)  # spoken while the model is still writing
        speaker.feed("nd sentence here.")
        speaker.end_utterance()
        assert speaker.wait(WAIT)
        assert engine.calls == [FIRST, SECOND]

    def test_streamed_deltas_give_the_same_pieces_as_one_text(self, make, engine):
        speaker = make()
        for character in TWO:
            speaker.feed(character)
        speaker.end_utterance()
        assert speaker.wait(WAIT)
        assert engine.calls == [FIRST, SECOND]

    def test_without_a_player_nothing_is_synthesized_or_played(self, make, engine, scratch):
        calls = []
        speaker = make(player=None, on_no_player=lambda: calls.append(1))
        speaker.say(TWO)
        assert speaker.wait(WAIT)
        assert engine.calls == [] and calls == []  # asking for no player is not a problem
        assert not scratch.exists()

    def test_blank_text_is_ignored(self, make, engine):
        speaker = make()
        speaker.say("")
        speaker.say("  \n ")
        speaker.feed("")
        speaker.end_utterance()  # nothing was fed
        assert speaker.wait(0)
        assert engine.calls == []

    def test_an_alert_does_not_mix_with_a_reply_being_streamed(self, make, engine):
        speaker = make()
        speaker.feed("Reply part one. Reply par")
        speaker.say("Alert!")  # e.g. from a timer thread
        speaker.feed("t two.")
        speaker.end_utterance()

        assert speaker.wait(WAIT)
        assert engine.calls == ["Reply part one.", "Alert!", "Reply part two."]

    def test_the_piece_size_follows_the_language_of_the_engine(self, make, engine):
        engine.language = "ja"  # 71 characters at most
        speaker = make()
        speaker.say(" ".join(["word"] * 40))
        assert speaker.wait(WAIT)
        assert len(engine.calls) > 1 and all(len(c) <= 71 for c in engine.calls)

    def test_an_explicit_max_chars_wins(self, make, engine):
        speaker = make(max_chars=30)
        speaker.say(" ".join(["word"] * 40))
        assert speaker.wait(WAIT)
        assert all(len(c) <= 30 for c in engine.calls)


class TestSaving:
    TEXT = "Alpha one here. Beta two here. Gamma three here."  # three pieces with max_chars=20

    def make_saving(self, make, tmp_path, **kwargs):
        return make(save_dir=tmp_path / "saved", pause_ms=10, max_chars=20, **kwargs)

    def test_the_whole_utterance_is_saved_with_pauses_between_pieces(self, make, engine, tmp_path):
        speaker = self.make_saving(make, tmp_path)
        speaker.say(self.TEXT, name="reply_001")
        assert speaker.wait(WAIT)

        assert engine.calls == ["Alpha one here.", "Beta two here.", "Gamma three here."]
        pause = np.zeros(10, dtype=np.float32)  # 10 ms at 1 kHz
        expected = np.concatenate(
            [samples_for(engine.calls[0]), pause, samples_for(engine.calls[1]), pause]
            + [samples_for(engine.calls[2])]
        )
        saved = read_wav(tmp_path / "saved" / "reply_001.wav")
        assert len(saved) == sum(len(c) for c in engine.calls) + 2 * 10
        assert np.array_equal(saved, (expected * 32767.0).astype("<i2"))

    def test_the_default_name_is_numbered(self, make, tmp_path):
        speaker = self.make_saving(make, tmp_path)
        speaker.say("First one here.")
        speaker.say("Second one here.")
        assert speaker.wait(WAIT)
        assert sorted(p.name for p in (tmp_path / "saved").iterdir()) == [
            "utterance_1.wav",
            "utterance_2.wav",
        ]

    def test_a_streamed_utterance_is_saved_when_it_ends(self, make, tmp_path):
        speaker = self.make_saving(make, tmp_path)
        speaker.feed("Alpha one here. Beta two")
        speaker.feed(" here.")
        assert not (tmp_path / "saved").exists()  # not before the end
        speaker.end_utterance("chat_001")
        assert speaker.wait(WAIT)
        assert len(read_wav(tmp_path / "saved" / "chat_001.wav")) == 15 + 10 + 14

    def test_a_name_cannot_escape_the_save_dir(self, make, tmp_path):
        speaker = self.make_saving(make, tmp_path)
        speaker.say("Alpha one here.", name="../../evil.wav")
        assert speaker.wait(WAIT)
        assert [p.name for p in (tmp_path / "saved").iterdir()] == ["evil.wav"]
        assert not (tmp_path / "evil.wav").exists()

    def test_saving_works_without_a_player(self, make, engine, tmp_path):
        speaker = self.make_saving(make, tmp_path, player=None)
        speaker.say(self.TEXT, name="silent")
        assert speaker.wait(WAIT)
        assert len(read_wav(tmp_path / "saved" / "silent.wav")) == 15 + 14 + 17 + 20

    def test_nothing_is_saved_when_save_dir_is_not_set(self, make, engine, tmp_path):
        speaker = make()
        speaker.say(self.TEXT)
        assert speaker.wait(WAIT)
        assert not (tmp_path / "saved").exists()


class TestCancel:
    def make_busy(self, make, engine, player, tmp_path, text=THREE, **kwargs):
        """Piece 1 is playing (held), piece 2 is synthesized and waiting, piece 3 is in flight."""
        player.hold = True
        engine.gates[THIRD] = threading.Event()
        speaker = make(save_dir=tmp_path / "saved", **THREE_PIECES, **kwargs)
        speaker.say(text, name="doomed")
        assert player.wait_started() == len(FIRST)
        assert engine.entered(THIRD).wait(WAIT)
        return speaker

    def test_stops_the_playback_and_drops_everything_queued(self, make, engine, player, tmp_path):
        speaker = self.make_busy(make, engine, player, tmp_path)

        speaker.cancel()

        assert player.stop_calls >= 1 and player.was_stopped
        engine.gates[THIRD].set()  # the synthesis that was in flight finishes, uselessly
        assert speaker.wait(WAIT)
        player.release()
        speaker.say("Fresh start.")  # the speaker stays usable
        assert speaker.wait(WAIT)
        assert player.frames == [len(FIRST), len("Fresh start.")]  # nothing of the old reply
        assert engine.calls == [FIRST, SECOND, THIRD, "Fresh start."]
        assert not (tmp_path / "saved" / "doomed.wav").exists()  # and nothing saved for it

    def test_wait_returns_at_once_after_a_cancel(self, make, engine, player, tmp_path):
        speaker = self.make_busy(make, engine, player, tmp_path)
        speaker.cancel()
        assert speaker.wait(0)
        engine.gates[THIRD].set()

    def test_text_that_was_fed_but_not_ended_is_forgotten(self, make, engine, player):
        speaker = make()
        speaker.feed("Half a sentence that never ends")
        speaker.cancel()
        speaker.feed("New text here.")
        speaker.end_utterance()
        assert speaker.wait(WAIT)
        assert engine.calls == ["New text here."]

    def test_cancel_is_safe_when_idle_and_when_repeated(self, make, engine, player):
        speaker = make()
        speaker.cancel()
        speaker.cancel()
        assert player.stop_calls == 0  # nothing was playing
        speaker.say("Still working.")
        assert speaker.wait(WAIT)

    def test_a_playback_cut_by_cancel_is_not_an_error(self, make, engine, player, tmp_path):
        errors, no_player = [], []
        speaker = self.make_busy(
            make,
            engine,
            player,
            tmp_path,
            on_error=errors.append,
            on_no_player=lambda: no_player.append(1),
        )
        speaker.cancel()
        engine.gates[THIRD].set()
        assert speaker.wait(WAIT)
        assert errors == [] and no_player == []

    def test_cancel_from_another_thread(self, make, engine, player, tmp_path):
        speaker = self.make_busy(make, engine, player, tmp_path)
        thread = threading.Thread(target=speaker.cancel)
        thread.start()
        thread.join(WAIT)
        assert not thread.is_alive() and player.was_stopped
        engine.gates[THIRD].set()

    def test_a_stale_synthesis_never_reaches_the_player(self, make, engine, player, tmp_path):
        speaker = self.make_busy(make, engine, player, tmp_path)
        speaker.cancel()
        player.release()
        speaker.say("Fresh start.")  # queued behind the synthesis that cannot be interrupted
        engine.gates[THIRD].set()
        assert speaker.wait(WAIT)
        assert player.frames == [len(FIRST), len("Fresh start.")]


class TestErrors:
    def test_a_failed_piece_is_reported_once_and_the_rest_is_spoken(
        self, make, engine, player, tmp_path
    ):
        engine.failures[FIRST] = RuntimeError("model exploded")
        engine.failures[SECOND] = RuntimeError("again")
        errors = []
        speaker = make(on_error=errors.append, save_dir=tmp_path / "saved", **THREE_PIECES)
        speaker.say(THREE, name="partial")

        assert speaker.wait(WAIT)
        assert errors == ["could not speak the reply: model exploded"]  # once per utterance
        assert engine.calls == [FIRST, SECOND, THIRD]
        assert player.frames == [len(THIRD)]  # the piece after the failures is still spoken
        assert not (tmp_path / "saved" / "partial.wav").exists()  # incomplete: not saved

    def test_each_utterance_reports_its_own_error(self, make, engine):
        engine.failures["Broken one here."] = RuntimeError("one")
        engine.failures["Broken two here."] = RuntimeError("two")
        errors = []
        speaker = make(on_error=errors.append)
        speaker.say("Broken one here.")
        speaker.say("Broken two here.")
        assert speaker.wait(WAIT)
        assert errors == ["could not speak the reply: one", "could not speak the reply: two"]

    def test_the_pipeline_continues_after_an_error(self, make, engine, player):
        engine.failures["Broken one here."] = RuntimeError("boom")
        speaker = make(on_error=lambda message: None)
        speaker.say("Broken one here.")
        speaker.say("Good one here.")
        assert speaker.wait(WAIT)
        assert player.frames == [len("Good one here.")]

    def test_a_playback_exception_is_reported_once_and_the_file_is_removed(
        self, make, engine, player, scratch
    ):
        player.raises = OSError("audio device vanished")
        errors = []
        speaker = make(on_error=errors.append)
        speaker.say(TWO)
        assert speaker.wait(WAIT)
        assert errors == ["could not speak the reply: audio device vanished"]
        assert list(scratch.iterdir()) == []

    def test_a_failing_callback_does_not_kill_the_workers(self, make, engine, player):
        engine.failures["Broken one here."] = RuntimeError("boom")

        def bad_callback(message):
            raise ValueError("the callback itself is broken")

        speaker = make(on_error=bad_callback)
        speaker.say("Broken one here.")
        speaker.say("Good one here.")
        assert speaker.wait(WAIT)
        assert player.frames == [len("Good one here.")]

    def test_errors_go_to_the_log_by_default(self, make, engine, caplog):
        engine.failures["Broken one here."] = RuntimeError("boom")
        speaker = make()
        speaker.say("Broken one here.")
        assert speaker.wait(WAIT)
        assert "could not speak the reply: boom" in caplog.text

    def test_a_save_error_is_reported(self, make, engine, tmp_path):
        blocked = tmp_path / "saved"
        blocked.write_text("a file where the directory should be")
        errors = []
        speaker = make(save_dir=blocked, player=None, on_error=errors.append)
        speaker.say("Alpha one here.")
        assert speaker.wait(WAIT)
        assert len(errors) == 1 and errors[0].startswith("could not save the reply")


class TestNoPlayer:
    def test_an_unavailable_player_is_reported_once(self, make, engine):
        calls = []
        speaker = make(player=FakePlayer(available=False), on_no_player=lambda: calls.append(1))
        speaker.say(TWO)
        speaker.say("Another one here.")
        assert speaker.wait(WAIT)
        assert calls == [1]
        assert engine.calls == []  # nothing would be done with the audio

    def test_an_unavailable_player_still_allows_saving(self, make, engine, tmp_path):
        calls = []
        speaker = make(
            player=FakePlayer(available=False),
            on_no_player=lambda: calls.append(1),
            save_dir=tmp_path / "saved",
        )
        speaker.say("Alpha one here.", name="kept")
        assert speaker.wait(WAIT)
        assert calls == [1] and (tmp_path / "saved" / "kept.wav").is_file()

    def test_a_playback_that_fails_is_reported_once(self, make, player):
        player.fail = True
        calls = []
        speaker = make(on_no_player=lambda: calls.append(1))
        speaker.say(TWO)
        speaker.say("Another one here.")
        assert speaker.wait(WAIT)
        assert calls == [1]

    def test_no_callback_is_fine(self, make):
        speaker = make(player=FakePlayer(available=False))
        speaker.say(TWO)
        assert speaker.wait(WAIT)


class TestWait:
    def test_returns_true_at_once_when_idle(self, make):
        assert make().wait() is True
        assert make().wait(0) is True

    def test_times_out_while_a_piece_is_still_playing(self, make, player):
        player.hold = True
        speaker = make()
        speaker.say(FIRST)
        player.wait_started()
        assert speaker.wait(0.05) is False
        player.release()
        assert speaker.wait(WAIT) is True

    def test_waits_for_the_saved_file_too(self, make, tmp_path):
        speaker = make(player=None, save_dir=tmp_path / "saved")
        speaker.say("Alpha one here.", name="done")
        assert speaker.wait(WAIT)
        assert (tmp_path / "saved" / "done.wav").is_file()

    def test_does_not_wait_for_text_that_was_never_ended(self, make, engine):
        speaker = make()
        speaker.feed("No sentence end yet")
        assert speaker.wait(0) is True
        assert engine.calls == []


class TestClose:
    def worker_threads(self):
        return [t for t in threading.enumerate() if t.name.startswith("herald-speaker-")]

    def test_stops_the_threads_and_leaves_no_files(self, make, engine, player, scratch):
        speaker = make()
        speaker.say(TWO)
        assert speaker.wait(WAIT)
        speaker.close()

        assert self.worker_threads() == []
        assert list(scratch.iterdir()) == []

    def test_close_is_idempotent(self, make):
        speaker = make()
        speaker.close()
        speaker.close()
        assert self.worker_threads() == []

    def test_close_does_not_hang_on_a_playback_in_progress(self, make, engine, player):
        player.hold = True
        speaker = make()
        speaker.say(TWO)
        player.wait_started()
        speaker.close()  # stops the player instead of waiting for it
        assert player.was_stopped and self.worker_threads() == []

    def test_a_closed_speaker_ignores_new_text(self, make, engine):
        speaker = make()
        speaker.close()
        speaker.say(FIRST)
        speaker.feed(SECOND)
        speaker.end_utterance()
        speaker.cancel()
        assert speaker.wait(0)
        assert engine.calls == []

    def test_works_as_a_context_manager(self, engine, player, scratch):
        with Speaker(engine, player=player, scratch_dir=scratch) as speaker:
            speaker.say(FIRST)
            assert speaker.wait(WAIT)
        assert player.frames == [len(FIRST)]
        assert self.worker_threads() == []

    def test_a_pending_synthesis_does_not_block_close_forever(self, make, engine, monkeypatch):
        monkeypatch.setattr("herald.speaker._JOIN_TIMEOUT", 0.05)
        engine.gates[FIRST] = threading.Event()
        speaker = make()
        speaker.say(FIRST)
        assert engine.entered(FIRST).wait(WAIT)

        speaker.close()  # gives up waiting for the synthesis after a short timeout

        engine.gates[FIRST].set()  # the abandoned worker finishes and exits by itself
        for thread in speaker._threads:
            thread.join(WAIT)
        assert self.worker_threads() == []


class TestSetup:
    def test_the_pause_must_not_be_negative(self, engine, player, scratch):
        with pytest.raises(ValueError, match="pause_ms"):
            Speaker(engine, player=player, scratch_dir=scratch, pause_ms=-1)

    def test_the_scratch_dir_is_created_when_a_player_is_used(self, make, scratch):
        make()
        assert scratch.is_dir()


class FakeProcess:
    """An audio player process (``afplay``) that plays until the test ends it or it is stopped."""

    created: queue.Queue = queue.Queue()

    def __init__(self, args, **kwargs):
        self.args = args
        self.returncode = None
        self.terminated = False
        self.scratch_file_existed = Path(args[-1]).is_file()
        self._exited = threading.Event()
        FakeProcess.created.put(self)

    def wait(self, timeout=None):
        if not self._exited.wait(timeout):
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.exit(-15)

    def kill(self):
        self.exit(-9)

    def exit(self, code=0):
        self.returncode = code
        self._exited.set()


class TestWithTheRealPlayer:
    """How ``herald chat`` builds the speaker: a real Player, here over a fake process."""

    @pytest.fixture
    def afplay(self, mocker, monkeypatch):
        FakeProcess.created = queue.Queue()
        mocker.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(audio_playback, "_ON_WINDOWS", False)
        monkeypatch.setattr(subprocess, "Popen", FakeProcess)
        return FakeProcess

    def test_plays_every_piece_with_the_system_player(self, make, afplay, scratch):
        errors, no_player = [], []
        speaker = make(
            player=Player(), on_error=errors.append, on_no_player=lambda: no_player.append(1)
        )
        speaker.say(TWO)

        for _ in range(2):
            process = afplay.created.get(timeout=WAIT)
            assert process.args[0] == "afplay" and process.scratch_file_existed
            process.exit(0)

        assert speaker.wait(WAIT)
        assert errors == [] and no_player == []
        assert list(scratch.iterdir()) == []

    def test_cancel_terminates_the_player_process_and_is_not_an_error(
        self, make, engine, afplay, scratch
    ):
        errors, no_player = [], []
        speaker = make(
            player=Player(), on_error=errors.append, on_no_player=lambda: no_player.append(1)
        )
        speaker.say(TWO)
        playing = afplay.created.get(timeout=WAIT)

        speaker.cancel()  # the next user message arrives while Herald is talking

        assert playing.terminated
        assert speaker.wait(WAIT)
        speaker.say("Fresh start.")  # and the speaker carries on with the next answer
        afplay.created.get(timeout=WAIT).exit(0)
        assert speaker.wait(WAIT)
        assert errors == [] and no_player == []
        assert list(scratch.iterdir()) == []

    def test_a_player_that_exits_with_an_error_is_reported_once(self, make, afplay):
        no_player = []
        speaker = make(player=Player(), on_no_player=lambda: no_player.append(1))
        speaker.say(TWO)
        for _ in range(2):
            afplay.created.get(timeout=WAIT).exit(1)
        assert speaker.wait(WAIT)
        assert no_player == [1]

    def test_no_system_player_at_all(self, make, engine, mocker, monkeypatch, scratch):
        mocker.patch("shutil.which", return_value=None)
        monkeypatch.setattr(audio_playback, "_ON_WINDOWS", False)
        calls = []
        speaker = make(player=Player(), on_no_player=lambda: calls.append(1))
        speaker.say(TWO)
        assert speaker.wait(WAIT)
        assert calls == [1] and engine.calls == []
