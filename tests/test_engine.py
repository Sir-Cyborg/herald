"""Everything in the engine module that runs without a real model: device selection,
checkpoint lookup, model loading wiring, WAV writing and synthesis with a fake model."""

import logging
import os
import sys
import threading
import time
import types
import wave

import numpy as np
import pytest

from herald.errors import CheckpointError, ConfigError, DependencyError
from herald.tts import engine

MPS_FALLBACK = "PYTORCH_ENABLE_MPS_FALLBACK"


@pytest.fixture(autouse=True)
def isolated_torch_env(monkeypatch):
    """Start every test without ``MPS_FALLBACK``, and restore the real environment after it.

    ``prepare_torch_env`` (called by ``load_engine`` and ``select_device``) sets it for the
    whole process. Setting it first makes monkeypatch remember that it was absent, which
    ``delenv(raising=False)`` alone would not.
    """
    monkeypatch.setenv(MPS_FALLBACK, "placeholder")
    monkeypatch.delenv(MPS_FALLBACK)


class TestPickDevice:
    @pytest.mark.parametrize(
        ("cuda", "mps", "expected"),
        [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
    )
    def test_auto_prefers_cuda_then_mps_then_cpu(self, cuda, mps, expected):
        assert engine.pick_device("auto", cuda=cuda, mps=mps) == expected

    @pytest.mark.parametrize("choice", ["cpu", "mps", "cuda", "cuda:1"])
    def test_explicit_choice_wins(self, choice):
        assert engine.pick_device(choice, cuda=False, mps=False) == choice

    @pytest.mark.parametrize("choice", ["gpu", "cuda:", "CPU", ""])
    def test_invalid_choice(self, choice):
        with pytest.raises(ConfigError, match="Invalid device"):
            engine.pick_device(choice, cuda=True, mps=True)


class TestSelectDevice:
    def test_explicit_device_does_not_import_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)  # any import would fail
        assert engine.select_device("cpu") == "cpu"

    def test_auto_asks_torch(self, monkeypatch):
        fake = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
        )
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert engine.select_device("auto") == "mps"

    def test_auto_without_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(DependencyError, match="not installed"):
            engine.select_device("auto")


class TestResolveFinetunedCheckpoint:
    def test_a_file_is_returned_as_is(self, tmp_path):
        ckpt = tmp_path / "anything.pth"
        ckpt.write_bytes(b"x")
        assert engine.resolve_finetuned_checkpoint(ckpt) == ckpt

    def test_best_model_wins_in_a_run_directory(self, tmp_path):
        for name in ("best_model.pth", "best_model_2925.pth", "checkpoint_5000.pth"):
            (tmp_path / name).write_bytes(b"x")
        assert engine.resolve_finetuned_checkpoint(tmp_path).name == "best_model.pth"

    def test_highest_numbered_best_model(self, tmp_path):
        for name in ("best_model_900.pth", "best_model_2925.pth", "checkpoint_5000.pth"):
            (tmp_path / name).write_bytes(b"x")
        # 2925 > 900 numerically, though "900" sorts after "2925" as text.
        assert engine.resolve_finetuned_checkpoint(tmp_path).name == "best_model_2925.pth"

    def test_falls_back_to_the_latest_checkpoint(self, tmp_path):
        for name in ("checkpoint_250.pth", "checkpoint_1000.pth", "notes.txt"):
            (tmp_path / name).write_bytes(b"x")
        assert engine.resolve_finetuned_checkpoint(tmp_path).name == "checkpoint_1000.pth"

    def test_empty_run_directory(self, tmp_path):
        with pytest.raises(CheckpointError, match="No best_model"):
            engine.resolve_finetuned_checkpoint(tmp_path)

    def test_missing_path(self, tmp_path):
        with pytest.raises(CheckpointError, match="not found"):
            engine.resolve_finetuned_checkpoint(tmp_path / "nope")


class FakeXtts:
    """Records how the model is loaded."""

    instances: list["FakeXtts"] = []

    def __init__(self):
        self.calls = []
        FakeXtts.instances.append(self)

    @classmethod
    def init_from_config(cls, config):
        inst = cls()
        inst.calls.append(("init", config.loaded))
        return inst

    def load_checkpoint(self, config, **kwargs):
        self.calls.append(("load_checkpoint", kwargs))

    def to(self, device):
        self.calls.append(("to", device))

    def eval(self):
        self.calls.append(("eval",))

    def get_conditioning_latents(self, audio_path):
        self.calls.append(("conditioning", audio_path))
        return "latent", "embedding"


class FakeConfig:
    loaded = None

    def load_json(self, path):
        FakeConfig.loaded = path


@pytest.fixture
def fake_tts(monkeypatch):
    FakeXtts.instances.clear()
    FakeConfig.loaded = None
    modules = {
        "TTS": types.ModuleType("TTS"),
        "TTS.tts": types.ModuleType("TTS.tts"),
        "TTS.tts.configs": types.ModuleType("TTS.tts.configs"),
        "TTS.tts.configs.xtts_config": types.SimpleNamespace(XttsConfig=FakeConfig),
        "TTS.tts.models": types.ModuleType("TTS.tts.models"),
        "TTS.tts.models.xtts": types.SimpleNamespace(Xtts=FakeXtts),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return FakeXtts


@pytest.fixture
def ensure(mocker):
    return mocker.patch("herald.tts.checkpoints.ensure_base_checkpoints")


class TestLoadEngine:
    def test_base_model(self, tmp_path, fake_tts, ensure):
        eng = engine.load_engine(
            tmp_path, ["a.wav", "b.wav"], device="cpu", language="it", temperature=0.5
        )

        assert ensure.call_args.kwargs["files"] == ("config.json", "vocab.json", "model.pth")
        model = fake_tts.instances[0]
        assert FakeConfig.loaded == str(tmp_path / "config.json")
        assert ("load_checkpoint", {"checkpoint_dir": str(tmp_path), "use_deepspeed": False}) in (
            model.calls
        )
        assert ("to", "cpu") in model.calls and ("eval",) in model.calls
        assert ("conditioning", ["a.wav", "b.wav"]) in model.calls
        assert (eng.gpt_cond_latent, eng.speaker_embedding) == ("latent", "embedding")
        assert (eng.language, eng.temperature) == ("it", 0.5)

    def test_device_move_comes_before_eval_and_conditioning(self, tmp_path, fake_tts, ensure):
        engine.load_engine(tmp_path, ["a.wav"], device="cpu")
        steps = [call[0] for call in fake_tts.instances[0].calls]
        # The latents are computed on the model's device, so it must have been moved by then.
        assert steps == ["init", "load_checkpoint", "to", "eval", "conditioning"]

    def test_finetuned_checkpoint_skips_the_base_model_download(self, tmp_path, fake_tts, ensure):
        ckpt = tmp_path / "run" / "best_model.pth"
        engine.load_engine(tmp_path, ["a.wav"], finetuned_checkpoint=ckpt, device="cpu")

        assert ensure.call_args.kwargs["files"] == ("config.json", "vocab.json")
        (call,) = [c for c in fake_tts.instances[0].calls if c[0] == "load_checkpoint"]
        assert call[1] == {
            "checkpoint_path": str(ckpt),
            "vocab_path": str(tmp_path / "vocab.json"),
            "use_deepspeed": False,
        }

    def test_custom_checkpoint_url_is_forwarded(self, tmp_path, fake_tts, ensure):
        engine.load_engine(tmp_path, ["a.wav"], device="cpu", checkpoint_url="http://mirror")
        assert ensure.call_args.kwargs["base_url"] == "http://mirror"

    def test_reference_wavs_are_required(self, tmp_path, fake_tts, ensure):
        with pytest.raises(ValueError, match="reference wav"):
            engine.load_engine(tmp_path, [], device="cpu")

    def test_missing_tts_stack(self, tmp_path, monkeypatch, ensure):
        monkeypatch.setitem(sys.modules, "TTS.tts.configs.xtts_config", None)
        with pytest.raises(DependencyError, match="not installed"):
            engine.load_engine(tmp_path, ["a.wav"], device="cpu")

    def test_mps_fallback_is_enabled_before_loading(self, tmp_path, fake_tts, ensure):
        engine.load_engine(tmp_path, ["a.wav"], device="cpu")
        assert os.environ[MPS_FALLBACK] == "1"

    def test_an_explicit_mps_fallback_setting_is_kept(
        self, tmp_path, fake_tts, ensure, monkeypatch
    ):
        monkeypatch.setenv(MPS_FALLBACK, "0")
        engine.load_engine(tmp_path, ["a.wav"], device="cpu")
        assert os.environ[MPS_FALLBACK] == "0"


class TestSaveWav:
    def test_round_trip(self, tmp_path):
        samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        path = engine.save_wav(tmp_path / "sub" / "out.wav", samples)
        with wave.open(str(path), "rb") as f:
            assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (
                1,
                2,
                engine.SAMPLE_RATE,
            )
            data = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
        assert data.tolist() == [0, 16383, -16383, 32767, -32767]

    def test_out_of_range_samples_are_clipped(self, tmp_path):
        path = engine.save_wav(tmp_path / "loud.wav", np.array([3.0, -3.0]))
        with wave.open(str(path), "rb") as f:
            data = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
        assert data.tolist() == [32767, -32767]

    def test_custom_sample_rate(self, tmp_path):
        path = engine.save_wav(tmp_path / "x.wav", np.zeros(4), sample_rate=16_000)
        with wave.open(str(path), "rb") as f:
            assert f.getframerate() == 16_000


class FakeModel:
    """Stands in for Xtts: records the calls, and returns a clip that depends on the text."""

    def __init__(self):
        self.calls = []

    @property
    def texts(self):
        return [call["text"] for call in self.calls]

    def inference(self, **kwargs):
        self.calls.append(kwargs)
        return {"wav": np.array([ord(c) for c in kwargs["text"]], dtype=np.float32) / 1000}


def make_engine(model=None, **kwargs):
    return engine.XttsEngine(
        model or FakeModel(), "latent", "embedding", sample_rate=1000, **kwargs
    )


class TestSynthesize:
    def test_passes_voice_and_settings(self):
        eng = make_engine(language="it", temperature=0.4)
        wav = eng.synthesize("ciao")
        (call,) = eng.model.calls
        assert call == {
            "text": "ciao",
            "language": "it",
            "gpt_cond_latent": "latent",
            "speaker_embedding": "embedding",
            "temperature": 0.4,
        }
        assert wav.shape == (4,)

    def test_torch_like_tensors_are_converted(self):
        class Tensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.ones((1, 3))

        model = FakeModel()
        model.inference = lambda **kw: {"wav": Tensor()}
        wav = make_engine(model).synthesize("x")
        assert wav.shape == (3,) and wav.dtype == np.float32

    def test_concurrent_calls_run_one_at_a_time(self):
        """XTTS keeps per-call state in the model, so two calls must never overlap."""
        first_started, release = threading.Event(), threading.Event()
        running = peak = 0

        def inference(**kwargs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            first_started.set()
            release.wait(timeout=5)
            running -= 1
            return {"wav": np.zeros(3, dtype=np.float32)}

        model = FakeModel()
        model.inference = inference
        eng = make_engine(model)
        threads = [threading.Thread(target=eng.synthesize, args=(text,)) for text in "ab"]
        threads[0].start()
        assert first_started.wait(timeout=5)
        threads[1].start()
        time.sleep(0.1)  # gives the second call every chance to (wrongly) start
        assert peak == 1
        release.set()
        for thread in threads:
            thread.join(timeout=5)
        assert peak == 1 and not any(t.is_alive() for t in threads)

    def test_speed_is_logged_for_tuning(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=engine.logger.name):
            make_engine().synthesize("Hello there.")
        assert any("real time" in r.getMessage() for r in caplog.records)


class TestSynthStream:
    TEXT = "Aaaa. Bbbb. Cccc."  # three chunks of 5 characters with max_chars=5

    def stream(self, eng, **kwargs):
        return eng.synth_stream(self.TEXT, max_chars=5, **kwargs)

    def test_yields_each_chunk_with_silence_between_but_not_after(self):
        eng = make_engine()
        items = list(self.stream(eng, pause_ms=10))  # 10 ms at 1 kHz is 10 samples

        silence = np.zeros(10, dtype=np.float32)
        expected = [eng.synthesize("Aaaa."), silence, eng.synthesize("Bbbb."), silence]
        expected.append(eng.synthesize("Cccc."))
        assert len(items) == len(expected)
        for item, want in zip(items, expected, strict=True):
            assert item.dtype == np.float32 and item.ndim == 1
            assert np.array_equal(item, want)

    def test_a_single_chunk_has_no_silence(self):
        items = list(make_engine().synth_stream("Hello.", pause_ms=500))
        assert [len(i) for i in items] == [6]

    def test_zero_pause_yields_only_audio(self):
        items = list(self.stream(make_engine(), pause_ms=0))
        assert [len(i) for i in items] == [5, 5, 5]

    def test_text_is_synthesized_lazily_one_chunk_at_a_time(self):
        model = FakeModel()
        stream = self.stream(make_engine(model), pause_ms=10)
        assert model.texts == []  # nothing happens until the first item is requested

        next(stream)
        assert model.texts == ["Aaaa."]
        next(stream)  # the pause after the first chunk needs no synthesis
        assert model.texts == ["Aaaa."]
        next(stream)
        assert model.texts == ["Aaaa.", "Bbbb."]

    def test_stopping_early_skips_the_rest(self):
        model = FakeModel()
        stream = self.stream(make_engine(model))
        next(stream)
        stream.close()
        assert model.texts == ["Aaaa."]

    def test_silence_arrays_are_independent(self):
        _, first_silence, _, second_silence, _ = self.stream(make_engine(), pause_ms=10)
        first_silence += 1  # a consumer may modify what it receives...
        assert not second_silence.any()  # ...without affecting the other silences

    def test_pieces_are_normalised_to_float32_mono(self):
        model = FakeModel()
        model.inference = lambda **kw: {"wav": np.ones((1, 4), dtype=np.float64)}
        (piece,) = make_engine(model).synth_stream("Hi.")
        assert piece.dtype == np.float32 and piece.shape == (4,)

    def test_blank_text_raises_on_the_first_iteration(self):
        model = FakeModel()
        stream = make_engine(model).synth_stream("  \n ")  # a generator: no error yet
        with pytest.raises(ValueError, match="empty"):
            next(stream)
        assert model.texts == []

    def test_negative_pause_is_rejected_before_any_synthesis(self):
        model = FakeModel()
        stream = self.stream(make_engine(model), pause_ms=-1)
        with pytest.raises(ValueError, match="pause_ms"):
            next(stream)
        assert model.texts == []

    def test_invalid_max_chars_is_rejected(self):
        with pytest.raises(ValueError, match="max_chars"):
            next(make_engine().synth_stream("Hi.", max_chars=0))


class TestLanguageChunking:
    """With ``max_chars=None`` the chunk size follows the engine's language."""

    SENTENCE = " ".join(["ab"] * 25) + "."  # 75 characters
    TEXT = " ".join([SENTENCE] * 3)  # 227: over the Italian limit (213), under the English one

    @pytest.fixture(params=["synth_long", "synth_stream"])
    def run(self, request):
        def run(eng, text, **kwargs):
            if request.param == "synth_stream":
                return list(eng.synth_stream(text, **kwargs))
            return eng.synth_long(text, **kwargs)

        return run

    def chunks_for(self, language, run, **kwargs):
        eng = make_engine(language=language)
        run(eng, self.TEXT, pause_ms=0, **kwargs)
        return eng.model.texts

    def test_the_text_sits_between_the_two_limits(self):
        assert engine.default_max_chars("it") < len(self.TEXT) <= engine.default_max_chars("en")

    def test_english_keeps_it_in_one_chunk(self, run):
        assert self.chunks_for("en", run) == [self.TEXT]

    def test_italian_splits_it(self, run):
        texts = self.chunks_for("it", run)
        assert len(texts) == 2
        assert all(len(t) <= 213 for t in texts)
        assert " ".join(texts) == self.TEXT

    def test_an_explicit_max_chars_wins(self, run):
        assert self.chunks_for("it", run, max_chars=250) == [self.TEXT]
        assert len(self.chunks_for("en", run, max_chars=100)) == 3

    def test_the_language_is_looked_up_on_every_call(self, run):
        eng = make_engine(language="it")
        run(eng, self.TEXT)
        assert len(eng.model.texts) == 2
        eng.model.calls.clear()
        eng.language = "en"
        run(eng, self.TEXT)
        assert eng.model.texts == [self.TEXT]

    def test_a_regional_language_code_is_understood(self, run):
        assert len(self.chunks_for("pt-BR", run)) == 2  # pt allows 203 characters


class TestSynthLong:
    TEXT = "Aaaa. Bbbb. Cccc. Dd."  # four chunks with max_chars=5

    @pytest.mark.parametrize(
        ("pause_ms", "pause_samples"), [(0, 0), (10, 10), (12.5, 12), (150, 150)]
    )
    def test_is_the_chunks_joined_by_silence(self, pause_ms, pause_samples):
        eng = make_engine()
        silence = np.zeros(pause_samples, dtype=np.float32)
        parts = []
        for i, chunk in enumerate(["Aaaa.", "Bbbb.", "Cccc.", "Dd."]):
            if i:
                parts.append(silence)
            parts.append(eng.synthesize(chunk))
        expected = np.concatenate(parts)

        wav = eng.synth_long(self.TEXT, pause_ms=pause_ms, max_chars=5)

        assert wav.dtype == np.float32
        assert wav.tobytes() == expected.tobytes()
        assert len(wav) == 5 * 3 + 3 + 3 * pause_samples

    def test_short_text_is_a_single_call(self):
        eng = make_engine()
        eng.synth_long("Hello.")
        assert eng.model.texts == ["Hello."]

    def test_each_chunk_is_synthesized_in_order(self):
        eng = make_engine()
        eng.synth_long(self.TEXT, pause_ms=10, max_chars=5)
        assert eng.model.texts == ["Aaaa.", "Bbbb.", "Cccc.", "Dd."]

    def test_blank_text_and_bad_arguments_raise(self):
        eng = make_engine()
        with pytest.raises(ValueError, match="empty"):
            eng.synth_long("   ")
        with pytest.raises(ValueError, match="pause_ms"):
            eng.synth_long("Hello.", pause_ms=-5)
        with pytest.raises(ValueError, match="max_chars"):
            eng.synth_long("Hello.", max_chars=0)
