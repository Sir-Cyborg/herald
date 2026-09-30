"""Header statistics of dataset clips (tiny WAV files written with the stdlib)."""

import wave
from pathlib import Path

import pytest

from herald.dataset import audio_stats
from herald.dataset.audio_stats import (
    MAX_CLIP_SECONDS,
    MAX_WAV_SAMPLES,
    SAMPLE_RATE_TRAIN,
    inspect_audio,
)


def write_wav(path: Path, seconds: float, *, rate: int = 22_050, channels: int = 1) -> Path:
    return write_wav_frames(path, round(seconds * rate), rate=rate, channels=channels)


def write_wav_frames(path: Path, frames: int, *, rate: int = 22_050, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as clip:
        clip.setnchannels(channels)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(b"\x00\x00" * channels * frames)
    return path


@pytest.fixture
def clips(tmp_path) -> dict[str, Path]:
    """One clip of each kind."""
    return {
        "mono": write_wav(tmp_path / "mono_5s.wav", 5),
        "stereo": write_wav(tmp_path / "stereo_44k_2s.wav", 2, rate=44_100, channels=2),
        "short": write_wav(tmp_path / "short_0.3s.wav", 0.3),
        "long": write_wav(tmp_path / "long_12s.wav", 12),
    }


class TestInspectAudio:
    def test_the_limits_are_those_of_the_xtts_trainer(self):
        assert (SAMPLE_RATE_TRAIN, MAX_WAV_SAMPLES) == (22_050, 255_995)
        assert MAX_CLIP_SECONDS == pytest.approx(11.61, abs=0.01)
        assert audio_stats.MIN_CLIP_SECONDS == 0.5
        assert audio_stats.MIN_CONDITIONING_SECONDS == 3.0

    def test_summarises_each_kind_of_clip(self, clips):
        stats = inspect_audio(list(clips.values()))

        assert stats.clips_checked == 4
        assert stats.unreadable == ()
        assert stats.total_seconds == pytest.approx(5 + 2 + 0.3 + 12)
        assert stats.sample_rates == {22_050: 3, 44_100: 1}
        assert stats.channels == {1: 3, 2: 1}
        assert stats.too_short == (str(clips["short"]),)
        assert stats.too_long == (str(clips["long"]),)
        assert stats.weak_reference == 1  # the 2 s clip; the 0.3 s one is not usable at all

    def test_a_good_dataset_has_nothing_to_report(self, tmp_path):
        paths = [write_wav(tmp_path / f"{i}.wav", 4 + i) for i in range(5)]

        stats = inspect_audio(paths)

        assert stats.clips_checked == 5
        assert (stats.too_short, stats.too_long, stats.unreadable) == ((), (), ())
        assert stats.weak_reference == 0

    def test_no_clips(self):
        stats = inspect_audio([])

        assert stats == audio_stats.AudioStats(0, (), 0.0, {}, {}, (), (), 0)

    @pytest.mark.parametrize(
        ("frames", "kind"),
        [
            (11_024, "too_short"),  # just under 0.5 s at 22.05 kHz
            (11_025, "weak_reference"),  # exactly 0.5 s is usable
            (66_149, "weak_reference"),  # just under 3 s
            (66_150, None),  # exactly 3 s is a full reference
            (MAX_WAV_SAMPLES, None),  # the longest clip the trainer accepts
            (MAX_WAV_SAMPLES + 1, "too_long"),
        ],
    )
    def test_boundaries(self, tmp_path, frames, kind):
        path = write_wav_frames(tmp_path / "clip.wav", frames)

        stats = inspect_audio([path])

        assert stats.too_short == ((str(path),) if kind == "too_short" else ())
        assert stats.too_long == ((str(path),) if kind == "too_long" else ())
        assert stats.weak_reference == (1 if kind == "weak_reference" else 0)

    def test_the_sample_rate_is_taken_into_account(self, tmp_path):
        # 12 s of 44.1 kHz audio is 529,200 frames: too long, whatever the frame count says.
        path = write_wav(tmp_path / "long.wav", 12, rate=44_100)
        assert inspect_audio([path]).too_long == (str(path),)

    def test_the_limits_can_be_changed(self, clips):
        stats = inspect_audio(
            list(clips.values()), min_seconds=2.5, max_seconds=6, reference_seconds=10
        )

        assert set(stats.too_short) == {str(clips["short"]), str(clips["stereo"])}
        assert stats.too_long == (str(clips["long"]),)
        assert stats.weak_reference == 1  # the 5 s clip

    def test_paths_are_reported_in_the_order_given(self, tmp_path):
        paths = [write_wav(tmp_path / f"{name}.wav", 0.2) for name in ("b", "c", "a")]
        assert inspect_audio(paths).too_short == tuple(str(p) for p in paths)

    def test_paths_can_be_reported_relative_to_a_root(self, tmp_path):
        (tmp_path / "dataset" / "audio").mkdir(parents=True)
        (tmp_path / "elsewhere").mkdir()
        inside = write_wav(tmp_path / "dataset" / "audio" / "000001.wav", 0.2)
        outside = write_wav(tmp_path / "elsewhere" / "000002.wav", 0.2)

        stats = inspect_audio([inside, outside], root=tmp_path / "dataset")

        assert stats.too_short == ("audio/000001.wav", str(outside))

    def test_accepts_strings(self, clips):
        assert inspect_audio([str(clips["mono"])]).clips_checked == 1


class TestUnreadable:
    def test_files_that_are_not_wav(self, tmp_path):
        junk = tmp_path / "junk.wav"
        junk.write_bytes(b"RIFF")
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        mp3 = tmp_path / "voice.mp3"
        mp3.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 64)
        missing = tmp_path / "missing.wav"
        good = write_wav(tmp_path / "good.wav", 5)

        stats = inspect_audio([junk, empty, mp3, missing, good])

        assert stats.unreadable == tuple(str(p) for p in (junk, empty, mp3, missing))
        assert stats.clips_checked == 1  # the statistics only describe the readable clip
        assert stats.total_seconds == pytest.approx(5)

    def test_float_wav_is_not_inspected(self, tmp_path):
        path = write_wav(tmp_path / "float.wav", 5)
        data = bytearray(path.read_bytes())
        data[20:22] = (3).to_bytes(2, "little")  # format tag 3: IEEE float, unsupported by wave
        path.write_bytes(bytes(data))

        assert inspect_audio([path]).unreadable == (str(path),)

    def test_a_zero_sample_rate_is_unreadable(self, tmp_path):
        path = write_wav(tmp_path / "zero.wav", 5)
        data = bytearray(path.read_bytes())
        data[24:28] = (0).to_bytes(4, "little")
        path.write_bytes(bytes(data))

        stats = inspect_audio([path])

        assert stats.unreadable == (str(path),)
        assert stats.clips_checked == 0
