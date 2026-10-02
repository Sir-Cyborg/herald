"""Playback with fake processes and a fake ``winsound``: no sound is ever played."""

import subprocess
import sys
import threading
import types

import pytest

from herald import audio_playback
from herald.audio_playback import Player

WAIT = 3.0


@pytest.fixture(autouse=True)
def not_windows(monkeypatch):
    """Most tests are about the external players; the Windows ones opt back in."""
    monkeypatch.setattr(audio_playback, "_ON_WINDOWS", False)


class FakePopen:
    """A player process that runs until the test finishes it (or ``terminate`` does)."""

    created: list["FakePopen"] = []
    started = threading.Event()
    ignore_terminate = False
    start_error: Exception | None = None

    def __init__(self, args, **kwargs):
        if FakePopen.start_error:
            raise FakePopen.start_error
        self.args, self.kwargs = args, kwargs
        self.returncode = None
        self.terminated = self.killed = False
        self._exited = threading.Event()
        FakePopen.created.append(self)
        FakePopen.started.set()

    def wait(self, timeout=None):
        if not self._exited.wait(timeout):
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not FakePopen.ignore_terminate:
            self.exit(-15)

    def kill(self):
        self.killed = True
        self.exit(-9)

    def exit(self, code=0):
        self.returncode = code
        self._exited.set()


@pytest.fixture
def popen(monkeypatch):
    FakePopen.created = []
    FakePopen.started = threading.Event()
    FakePopen.ignore_terminate = False
    FakePopen.start_error = None
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(audio_playback, "_TERMINATE_WAIT", 0.05)
    return FakePopen


@pytest.fixture
def players(mocker):
    """Pretend every known player is installed."""
    mocker.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}")


def play_in_thread(player, path, results):
    thread = threading.Thread(target=lambda: results.append(player.play(path)))
    thread.start()
    assert FakePopen.started.wait(WAIT)
    return thread


class TestFindPlayer:
    def test_no_player_available(self, mocker):
        mocker.patch("shutil.which", return_value=None)
        assert audio_playback.find_player() is None

    def test_afplay_is_preferred(self, players):
        assert audio_playback.find_player() == ["afplay"]

    def test_falls_back_to_the_next_player(self, mocker):
        mocker.patch(
            "shutil.which", side_effect=lambda name: "/usr/bin/aplay" if name == "aplay" else None
        )
        assert audio_playback.find_player() == ["aplay", "-q"]


class TestPlayer:
    def test_plays_the_file_and_waits_for_the_process(self, players, popen, tmp_path):
        player, results = Player(), []
        wav = tmp_path / "x.wav"

        thread = play_in_thread(player, wav, results)
        assert results == []  # still playing
        (process,) = popen.created
        assert process.args == ["afplay", str(wav)]
        assert process.kwargs["stdout"] == subprocess.DEVNULL
        process.exit(0)
        thread.join(WAIT)

        assert results == [True] and not player.was_stopped

    def test_a_failing_player_is_reported_not_raised(self, players, popen, tmp_path):
        player, results = Player(), []
        thread = play_in_thread(player, tmp_path / "x.wav", results)
        popen.created[0].exit(1)
        thread.join(WAIT)
        assert results == [False] and not player.was_stopped

    @pytest.mark.parametrize("error", [FileNotFoundError("afplay"), PermissionError("afplay")])
    def test_a_player_that_cannot_start_is_reported_not_raised(
        self, players, popen, tmp_path, error
    ):
        popen.start_error = error
        assert Player().play(tmp_path / "x.wav") is False

    def test_no_player_installed(self, mocker, popen, tmp_path):
        mocker.patch("shutil.which", return_value=None)
        player = Player()
        assert player.available is False
        assert player.play(tmp_path / "x.wav") is False
        assert popen.created == [] and not player.was_stopped

    def test_available_when_a_player_exists(self, players):
        assert Player().available is True

    def test_stop_ends_the_playback_at_once(self, players, popen, tmp_path):
        player, results = Player(), []
        thread = play_in_thread(player, tmp_path / "x.wav", results)

        player.stop()
        thread.join(WAIT)

        assert results == [False]  # cut short: not played to the end...
        assert player.was_stopped  # ...and the caller can tell it was on purpose
        assert popen.created[0].terminated and not popen.created[0].killed

    def test_a_process_that_ignores_terminate_is_killed(self, players, popen, tmp_path):
        popen.ignore_terminate = True
        player, results = Player(), []
        thread = play_in_thread(player, tmp_path / "x.wav", results)

        player.stop()
        thread.join(WAIT)

        assert results == [False] and popen.created[0].killed

    def test_stop_is_idempotent_and_harmless_when_idle(self, players, popen, tmp_path):
        player = Player()
        player.stop()
        assert not player.was_stopped  # nothing was playing, so nothing was cut short

        results = []
        thread = play_in_thread(player, tmp_path / "x.wav", results)
        player.stop()
        player.stop()
        thread.join(WAIT)
        player.stop()
        assert results == [False] and player.was_stopped

    def test_was_stopped_describes_the_last_playback_only(self, players, popen, tmp_path):
        player, results = Player(), []
        thread = play_in_thread(player, tmp_path / "x.wav", results)
        player.stop()
        thread.join(WAIT)
        assert player.was_stopped

        popen.started.clear()
        thread = play_in_thread(player, tmp_path / "y.wav", results)
        popen.created[1].exit(0)
        thread.join(WAIT)
        assert results == [False, True] and not player.was_stopped

    def test_stop_from_another_thread_while_the_process_starts(self, players, popen, tmp_path):
        """Many stops racing with one playback never raise and always end it."""
        player, results = Player(), []
        thread = play_in_thread(player, tmp_path / "x.wav", results)
        stoppers = [threading.Thread(target=player.stop) for _ in range(8)]
        for stopper in stoppers:
            stopper.start()
        for stopper in stoppers:
            stopper.join(WAIT)
        thread.join(WAIT)
        assert results == [False]


class TestPlayWav:
    """The stateless helper, built on Player."""

    def test_plays_the_file(self, players, popen, tmp_path):
        results = []
        wav = tmp_path / "x.wav"
        thread = threading.Thread(target=lambda: results.append(audio_playback.play_wav(wav)))
        thread.start()
        assert popen.started.wait(WAIT)
        popen.created[0].exit(0)
        thread.join(WAIT)
        assert results == [True]
        assert popen.created[0].args == ["afplay", str(wav)]

    def test_no_player_available(self, mocker, popen, tmp_path):
        mocker.patch("shutil.which", return_value=None)
        assert audio_playback.play_wav(tmp_path / "x.wav") is False
        assert popen.created == []

    def test_failing_player_is_reported_not_raised(self, players, popen, tmp_path):
        popen.start_error = OSError("exec format error")
        assert audio_playback.play_wav(tmp_path / "x.wav") is False


class TestWindows:
    """``winsound`` only exists on Windows, so a fake module stands in for it."""

    @pytest.fixture
    def winsound(self, monkeypatch):
        monkeypatch.setattr(audio_playback, "_ON_WINDOWS", True)
        fake = types.SimpleNamespace(
            SND_FILENAME=0x20000,
            SND_NODEFAULT=0x2,
            SND_PURGE=0x40,
            calls=[],
            release=threading.Event(),
            started=threading.Event(),
            hold=False,
            error=None,
        )

        def play_sound(sound, flags):
            fake.calls.append((sound, flags))
            if sound is None:  # SND_PURGE: end whatever is playing
                fake.release.set()
                return
            if fake.error:
                raise fake.error
            fake.started.set()
            if fake.hold:
                assert fake.release.wait(WAIT)

        fake.PlaySound = play_sound
        monkeypatch.setitem(sys.modules, "winsound", fake)
        return fake

    def test_plays_the_file_without_an_external_player(self, winsound, mocker, tmp_path):
        popen = mocker.patch("subprocess.Popen")
        player = Player()
        wav = tmp_path / "x.wav"

        assert player.available is True  # nothing to install
        assert player.play(wav) is True
        # NODEFAULT: a missing file must fail, not play the system beep.
        assert winsound.calls == [(str(wav), 0x20000 | 0x2)]
        popen.assert_not_called()

    def test_failure_is_reported_not_raised(self, winsound, tmp_path):
        winsound.error = RuntimeError("Failed to play sound")
        player = Player()
        assert player.play(tmp_path / "x.wav") is False
        assert not player.was_stopped

    def test_missing_module_is_reported_not_raised(self, winsound, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "winsound", None)  # any import fails
        assert Player().play(tmp_path / "x.wav") is False

    def test_stop_purges_the_sound(self, winsound, tmp_path):
        winsound.hold = True
        player, results = Player(), []
        thread = threading.Thread(target=lambda: results.append(player.play(tmp_path / "x.wav")))
        thread.start()
        assert winsound.started.wait(WAIT)

        player.stop()
        thread.join(WAIT)

        assert (None, 0x40) in winsound.calls  # PlaySound(None, SND_PURGE)
        assert results == [False] and player.was_stopped

    def test_stop_when_idle_does_nothing(self, winsound):
        player = Player()
        player.stop()
        assert winsound.calls == [] and not player.was_stopped
