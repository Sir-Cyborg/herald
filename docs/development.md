# Developing Herald

```bash
pip install -e ".[dev]"
ruff check src tests && ruff format --check src tests
pytest -m "not integration"      # unit tests: fast, no model, no network
pytest -m integration            # loads real weights; skips itself if they are missing
```

The unit tests replace torch, coqui-tts, Ollama and the network with fakes, so they run without
the heavy dependencies. CI (`.github/workflows/ci.yml`) installs only what they import, runs
ruff and the unit tests, and validates the Compose files on every push and pull request.

```
src/herald/
  cli.py                     the `herald` command: options, profiles, one handler per command
  chat.py                    the `chat` loop: streaming, barge-in, timers, tools
  voice.py                   picks the reference clips and loads the voice model
  config.py, paths.py        settings from HERALD_* variables; project paths
  profiles.py                voice profiles: reading, validating, writing TOML
  assistant.py               conversation state and the tool-calling loop
  llm/ollama_client.py       Ollama client (text, streaming, tool calls)
  speaker.py                 speech pipeline: synthesize sentence k+1 while k plays
  audio_playback.py          best-effort, stoppable playback
  tts/engine.py              model loading, synthesis (short, long, streamed), sentence splitting
  tts/train.py               fine-tuning and promotion of the best model
  tts/slim.py                drops the optimizer state from a checkpoint (`herald slim`)
  tts/checkpoints.py         base weight download
  dataset/metadata.py        metadata loading, audio lookup, reference clips, train/eval split
  dataset/audio_stats.py     checks the recordings before training (length, rate, channels)
  tools/                     the tool framework: @tool, registry, loader, scheduler
  tools/builtin/             the tools that ship with Herald (set_timer)
tools/                       your own tool scripts (not part of the package)
profiles/                    voice profiles
```

## Where it is going

Herald is meant to become a voice-controlled home assistant: an Ollama model that can *do*
things, speaking through an ESP32 while Ollama and XTTS run on a PC. Timers and the tool
framework exist today; the ESP32 side does not:

- **More tools.** The framework is in place (see [the tools guide](../tools/README.md)): a new capability is one
  function. Next candidates are listing and cancelling timers, and anything you add to `tools/`.
  A timer only lives while the chat runs; timers that survive a restart would need storage.
- **Speech.** `XttsEngine` is a long-lived object, and `synth_stream()` yields audio chunk by
  chunk, so a server can start sending audio before the whole reply is synthesized. `chat` still
  waits for the full reply.
- **Front ends.** The terminal loop in `cli.py` is thin on purpose: a future ESP32 server would
  reuse `Assistant` and the engine and replace only the input and output.
