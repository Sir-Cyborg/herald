import subprocess
import sys
import types

import pytest

from herald import audio_playback


@pytest.fixture(autouse=True)
def not_windows(monkeypatch):
    """Most tests are about the external players; the Windows ones opt back in."""
    monkeypatch.setattr(audio_playback, "_ON_WINDOWS", False)


def test_no_player_available(mocker, tmp_path):
    mocker.patch("shutil.which", return_value=None)
    run = mocker.patch("subprocess.run")
    assert audio_playback.find_player() is None
    assert audio_playback.play_wav(tmp_path / "x.wav") is False
    run.assert_not_called()


def test_afplay_is_preferred(mocker, tmp_path):
    mocker.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    assert audio_playback.find_player() == ["afplay"]


def test_falls_back_to_the_next_player(mocker):
    mocker.patch(
        "shutil.which", side_effect=lambda name: "/usr/bin/aplay" if name == "aplay" else None
    )
    assert audio_playback.find_player() == ["aplay", "-q"]


def test_plays_the_file(mocker, tmp_path):
    mocker.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    run = mocker.patch("subprocess.run")
    wav = tmp_path / "x.wav"
    assert audio_playback.play_wav(wav) is True
    assert run.call_args.args[0] == ["afplay", str(wav)]
    assert run.call_args.kwargs["check"] is True


def test_failing_player_is_reported_not_raised(mocker, tmp_path):
    mocker.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    mocker.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "afplay"))
    assert audio_playback.play_wav(tmp_path / "x.wav") is False


def test_missing_binary_is_reported_not_raised(mocker, tmp_path):
    mocker.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    mocker.patch("subprocess.run", side_effect=FileNotFoundError("afplay"))
    assert audio_playback.play_wav(tmp_path / "x.wav") is False


class TestWindows:
    """``winsound`` only exists on Windows, so a fake module stands in for it."""

    @pytest.fixture
    def winsound(self, monkeypatch, mocker):
        monkeypatch.setattr(audio_playback, "_ON_WINDOWS", True)
        fake = types.SimpleNamespace(
            SND_FILENAME=0x20000, SND_NODEFAULT=0x2, PlaySound=mocker.Mock()
        )
        monkeypatch.setitem(sys.modules, "winsound", fake)
        return fake

    def test_plays_the_file_without_an_external_player(self, winsound, mocker, tmp_path):
        run = mocker.patch("subprocess.run")
        wav = tmp_path / "x.wav"
        assert audio_playback.play_wav(wav) is True
        # NODEFAULT: a missing file must fail, not play the system beep.
        winsound.PlaySound.assert_called_once_with(str(wav), 0x20000 | 0x2)
        run.assert_not_called()

    def test_failure_is_reported_not_raised(self, winsound, tmp_path):
        winsound.PlaySound.side_effect = RuntimeError("Failed to play sound")
        assert audio_playback.play_wav(tmp_path / "x.wav") is False

    def test_missing_module_is_reported_not_raised(self, winsound, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "winsound", None)  # any import fails
        assert audio_playback.play_wav(tmp_path / "x.wav") is False
