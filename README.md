# Herald

Clone a voice with [XTTS-v2](https://huggingface.co/coqui/XTTS-v2), fine-tune it on your own
dataset, and talk to it: `herald chat` sends what you type to a local LLM
([Ollama](https://ollama.com)) and speaks the replies in the cloned voice.

```
herald synthesize             speak a text to a WAV file (base or fine-tuned voice)
herald train                  fine-tune XTTS-v2 on a dataset
herald slim                   shrink a trained voice to a third of its size
herald chat                   chat with an Ollama model and hear the answers
herald download-checkpoints   fetch the base XTTS-v2 weights (about 2 GB)
```

## Contents

- [Which setup should I use?](#which-setup-should-i-use)
- [What is not in the repository](#what-is-not-in-the-repository-first-time-setup)
- [Quick start (native)](#quick-start-native)
- [Quick start (Docker)](#quick-start-docker)
- [Project layout](#project-layout)
- [Dataset](#dataset)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development](#development)
- [Where it is going](#where-it-is-going)
- [Security note](#security-note)
- [Licenses](#licenses)

## Which setup should I use?

| You have | Use | Why |
| --- | --- | --- |
| A Mac with Apple Silicon | **Native** | Docker Desktop cannot use the Mac's GPU (MPS) or play sound |
| Linux or Windows with an NVIDIA GPU | **Docker, GPU image** (or native) | Fast synthesis and training |
| Anything else | **Docker, CPU image** | Works everywhere; synthesis is slower, training very slow |

Common requirements:

- About 2 GB of disk for the base weights, plus about 5.6 GB per fine-tuned voice.
- [Ollama](https://ollama.com) with a model pulled (default `llama3.2:3b`) for `chat`. Docker
  Compose starts one for you.
- Native only: Python 3.11 (3.12 is allowed by the package metadata but untested).
- Training runs on CUDA when there is one, otherwise on the CPU: the trainer has no Apple MPS
  support, so a Mac trains on the CPU and needs a lot of patience.

## What is not in the repository (first-time setup)

Git only holds the code. The large and personal files are git-ignored (`dataset/`, `models/`,
`runs/`, `output/`, `POC/`), so **a fresh clone has none of them**. This is what goes where:

| Folder | What goes there | How you get it |
| --- | --- | --- |
| `dataset/<name>/` | `audio/*.wav` and `metadata.csv` (see [Dataset](#dataset)) | Your own recordings. Only needed to train, or to pick reference clips automatically. **You create it.** |
| `models/xtts_v2/` | The base XTTS-v2 weights (about 2 GB) | Downloaded by `herald download-checkpoints`, or automatically the first time a command needs them |
| `models/<name>/` | A fine-tuned voice: `best_model.pth` and its `config.json` | Produced by `herald train`, **or copied there from whoever trained it** (weights are too big for git) |
| `runs/` | Training logs | Created by `herald train` |
| `output/` | Generated audio | Created the first time herald writes a file |

Natively, herald creates `models/xtts_v2/`, `runs/` and `output/` by itself. With Docker, create
the four folders before the first `docker compose` command (see the Docker quick start),
otherwise Docker creates them owned by root.

**Using a voice somebody gave you** (for example the fine-tuned Frieren model): put its files in
a folder named after the voice, then point `--checkpoint` at that folder. A voice still needs a
few seconds of reference audio, either `--reference-wav` or a dataset:

```bash
mkdir -p models/frieren
cp /where/you/got/it/best_model.pth /where/you/got/it/config.json models/frieren/
herald synthesize "Hello." --checkpoint models/frieren --reference-wav my_clip.wav
```

The folder layout you end up with:

```
dataset/frieren/audio/000000.wav ...     (optional, your data)
dataset/frieren/metadata.csv
models/xtts_v2/                          (config.json, vocab.json, model.pth, ...)
models/frieren/best_model.pth            (optional, a fine-tuned voice)
models/frieren/config.json
```

## Quick start (native)

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # drop [dev] if you only want to run it

herald download-checkpoints          # once, about 2 GB into ./models/xtts_v2
herald synthesize "It has been many years since I left my home village." \
    --reference-wav my_voice.wav --play
```

`--reference-wav` is a few seconds of the voice to clone (repeat it for several clips). Without
it Herald picks clips from a [dataset](#dataset), which is not part of the repository: put your
own in `dataset/<name>/` first.

`herald` only exists inside the virtual environment: run the `source` line again in every new
terminal (or call `.venv/bin/herald` directly). If the shell says `command not found: herald`,
that is why.

Dependencies are pinned in `pyproject.toml` (there is no `requirements.txt`: `pip install -e .`
reads it): `coqui-tts`, `torch`/`torchaudio` 2.8 (which still load audio without
`torchcodec`/`ffmpeg`), `transformers`, `numpy` and `requests`.

## Quick start (Docker)

`docker-compose.yml` runs an `ollama` service (the official image) and a `herald` service. The
dataset, models, runs and output directories are bind-mounted from the host, so the big files
never enter the image.

```bash
cp .env.example .env                                  # optional: read the comments in it
mkdir -p dataset models runs output                   # otherwise Docker creates them as root
docker compose build
docker compose run --rm herald download-checkpoints   # base weights into ./models/xtts_v2
docker compose run --rm herald synthesize "Hello." --reference-wav dataset/my_voice.wav \
    -o output/hello.wav
docker compose run --rm herald chat --no-play          # text only; needs the ollama service (see below)
```

Only `chat` needs Ollama:

```bash
docker compose up -d --wait ollama
docker compose exec ollama ollama pull llama3.2:3b
```

- **CPU image** (`docker/Dockerfile`, the default): `python:3.11-slim` with CPU-only torch;
  amd64 and arm64.
- **NVIDIA GPU image** (`docker/Dockerfile.gpu`): Linux amd64 with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  and a driver that supports CUDA 12.8. It speeds up training as well as synthesis. Add the GPU
  file to every command, or set `COMPOSE_FILE` in `.env` (see `.env.example`):

  ```bash
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm herald train
  ```

- **Files owned by the wrong user (Linux).** Files written to the bind mounts belong to the
  container user (UID/GID 1000). Match your own by setting `HERALD_UID` and `HERALD_GID` in
  `.env` (`id -u`, `id -g`), then `docker compose build`.
- **Use an Ollama that already runs on the host** instead of the container (recommended on a
  Mac, where containers get no GPU):

  ```bash
  HERALD_OLLAMA_URL=http://host.docker.internal:11434 \
    docker compose run --rm herald chat --no-play
  ```

- Ollama's port is not published, because a native Ollama usually owns 11434. Add
  `ports: ["11434:11434"]` to the `ollama` service to reach it from the host (it has no
  authentication).

> **macOS note.** Docker Desktop runs containers in a Linux VM with **no GPU passthrough and no
> audio device**. Synthesis there is CPU-only, and training is CPU-only in a VM with limited
> memory. For training and for speed, run natively. `herald chat` in a container cannot play
> sound: add `--no-play --save-dir output/chat` and open the WAV files it saves there, or run
> `chat` natively (it can still talk to the Ollama container).

## Project layout

```
dataset/<speaker>/         audio/*.wav + metadata.csv          (your data, not in git)
models/xtts_v2/            base XTTS-v2 weights                (downloaded, not in git)
models/<speaker>/          fine-tuned voice: best_model.pth    (produced by `herald train`)
runs/                      training logs and TensorBoard data
output/                    generated audio
POC/                       the original notebook and its samples (local only, not in git)
src/herald/                the package
tests/                     unit tests (no model, no network) and integration tests
docker/, docker-compose*.yml, .env.example
```

`dataset/`, `models/`, `runs/`, `output/` and `POC/` are git-ignored: they are large or personal.

## Dataset

A dataset is a directory with the audio clips and one metadata file:

```
dataset/frieren/
  audio/000000.wav ...
  metadata.csv
```

`metadata.csv` has two columns, the audio path and the transcript, with or without a header row
and separated by `|`, a tab or a comma:

```
audio/000000.wav|What's wrong?
audio/000001.wav|After all, no one knows better than I do how frightening I can be.
```

Audio paths are looked up relative to the dataset directory, then its `audio/` folder, then by
file name in `audio/`. There is no separate train/eval file: `herald train` splits
`metadata.csv` itself (see [Train](#train)). Point Herald at your dataset with `--dataset-dir`
or `HERALD_DATASET_DIR`; the default is `dataset/frieren` under the project root.

### What the recordings should look like

Herald reads audio with XTTS-v2's own loader, which accepts almost anything and skips what it
cannot use. There are few hard rules; the rest is advice for a good voice:

| | Rule | What happens otherwise |
| --- | --- | --- |
| Format | WAV is what is tested here | Other formats are read only if your audio backend can decode them |
| Sample rate, channels | Any: audio is converted to mono and resampled to 22,050 Hz on the fly. Record at 22.05 kHz or more; upsampling a poor recording adds nothing | Nothing to do |
| Clip length | **0.5 s to about 11.6 s**, ideally 3 to 10 s | Shorter or longer clips are **skipped, silently** |
| Short clips | Under 3 s still work, but teach the voice less | Many very short clips give a weaker result |
| Transcript | Exactly what is said, **one language per dataset**, up to about 200 tokens (roughly one long sentence) | Longer lines are skipped. A character the tokenizer does not know also makes the clip fail, silently |
| Content | One speaker, one microphone, no music, no noise, no overlapping voices | The model learns the noise and the room too |
| Amount | Herald enforces no minimum. More clean speech is better; tens of minutes is a sensible target | Too little and the voice generalizes poorly |

Because bad clips are dropped without a word, run `herald train --dry-run` first: it reports the
total speech, the sample rates and how many clips would be skipped.

## Usage

Everything below is also documented by `herald <command> --help`. Add `-v` **before** the
command (`herald -v chat`) for debug logs, which include the synthesis speed.

### Synthesize

```bash
herald synthesize "Magic is not something you learn in a day." -o output/hello.wav --play
herald synthesize --text-file speech.txt
echo "Read from stdin." | herald synthesize
```

By default the voice is conditioned on 3 random clips of the dataset (`--seed` makes the pick
reproducible, `--num-references` changes how many). Use `--reference-wav clip.wav` (repeatable)
to choose them yourself.

Long text is split at sentence boundaries into chunks of at most 250 characters, the limit
XTTS-v2 handles well for English, and joined with 150 ms of silence (`--max-chars`,
`--pause-ms`). Output is a 16-bit mono WAV at 24 kHz.

**Using a fine-tuned voice.** Pass a checkpoint file or a voice directory (it picks
`best_model.pth`, else the highest `best_model_<step>.pth`), or set `HERALD_CHECKPOINT` once:

```bash
herald synthesize "Hello." --checkpoint models/frieren
export HERALD_CHECKPOINT=models/frieren       # then `synthesize` and `chat` use it by default
```

A fine-tuned voice only needs the small `config.json` and `vocab.json` of the base model, so
its first run downloads a few hundred KB, not 2 GB. Unset `HERALD_CHECKPOINT` to use the base
voice again.

### Train

**What training does.** XTTS-v2 turns text plus a few seconds of a reference voice into speech.
Fine-tuning teaches its GPT part (the one that decides how the text should sound) your
speaker: the rest of the model is left alone. Each step takes a clip, converts it to 22,050 Hz
mono, tokenizes its transcript in the chosen `--language`, and cuts a random 3 to 6 second slice
of the same clip as the "voice sample" it must imitate. After every epoch the model is scored on
the held-out evaluation clips and the best one is kept. The defaults are 5 epochs, an effective
batch of 32 clips (batch 2 x 16 accumulation steps) and a small learning rate (5e-6), which
adapts the voice without wrecking what the model already knows.

**What you need.**

- A dataset as described in [Dataset](#dataset). For another language, pass `--language` (for
  example `--language it`) and use transcripts in that language.
- Time and hardware. On the original run, 51 minutes of speech (1,299 clips) took about an hour
  and a half for 5 epochs on a Mac (the trainer has no Apple MPS support there). A CUDA GPU is
  much faster.
- Free disk: about **22 GB** while it runs (the trainer keeps several 5.6 GB checkpoints at a
  time; an estimate from the trainer's code, not a measurement), and about 2 GB per voice once
  it is done.
- The base weights: `train` downloads them by itself the first time.

Check the dataset and the resolved settings first; this needs no GPU and loads no model:

```bash
herald train --dry-run
```

Then a quick sanity run (1 epoch, a few dozen samples) and the full run:

```bash
herald train --smoke
herald train
```

| Setting | `full` (default) | `--smoke` |
| --- | --- | --- |
| epochs | 5 | 1 |
| batch size | 2 | 1 |
| gradient accumulation | 16 | 4 |
| samples | all | 60 train / 20 eval |

Explicit options (`--epochs`, `--batch-size`, `--grad-accum-steps`, `--lr`, `--num-workers`,
`--save-step`, `--max-train-samples`, ...) override the preset. The speaker name and run name
default to the dataset directory name.

- **Split.** `metadata.csv` is shuffled with a fixed seed and 10% becomes the evaluation set
  (`--eval-fraction`, `--seed`). It is reproducible and the two sets never overlap. Adding rows
  to `metadata.csv` changes the split.
- **Output.** Logs and TensorBoard data go to `runs/<run name>-<date>/`. When training ends, the
  best model is **moved** to `models/<speaker>/best_model.pth` next to a `config.json`, and the
  command prints the exact `synthesize` command to use it.
- **A model in `models/` is never overwritten or deleted.** If `models/<speaker>/` already
  holds a model, it is renamed `best_model.<date>-<time>.pth` (with its config) before the new
  one moves in. These backups are big: delete the ones you do not need.
- **`--smoke` never touches your real voice.** A smoke run is a throwaway, so its model goes to
  `models/<speaker>_smoke/`.
- **The saved voice is slim, and the leftovers are deleted.** A training checkpoint weighs 5.6 GB
  because it also carries the optimizer's state, which is only needed to *resume* training (Herald
  does not do that). The model itself is about 2 GB. So `herald train` saves the slim version as
  `models/<speaker>/best_model.pth`, then deletes the trainer's leftover checkpoint files from
  `runs/<run>/` once the model is safe, and tells you how much space that freed. Pass
  `--keep-checkpoints` to keep them.
- **If you stop a run with Ctrl-C**, the trainer writes one last checkpoint and exits without
  cleaning up. Nothing is promoted; turn what it left into a slim voice with
  `herald slim runs/<run name>-<date>` and delete the rest of that folder yourself.

**Shrinking a model you already have.**

```bash
herald slim models/frieren                 # writes models/frieren/best_model.slim.pth, keeps the original
herald slim models/frieren --replace       # replaces best_model.pth with the slim one (checked first)
```

A slim file loads and sounds exactly like the full one, and is much easier to share. `herald slim`
only accepts checkpoints you trust: they are Python pickles (see the security note).

### Chat

```bash
ollama pull llama3.2:3b
herald chat
```

Type a message, get a reply, and hear it. An empty line or Ctrl-D quits. Replies are played
and then **discarded**, so a long conversation does not fill the disk. Playback is best effort
(`afplay` on macOS, `winsound` on Windows, a few common players on Linux). If a reply cannot be
spoken, the error is printed and the conversation continues.

| You want | Use |
| --- | --- |
| To hear the replies (default) | `herald chat` |
| To hear them **and** keep them | `herald chat --save-dir replies/` (one WAV per reply) |
| Only to save them, without playing (for example in Docker) | `herald chat --no-play --save-dir replies/` |
| A text-only chat, no voice model loaded | `herald chat --no-play` |

The character is set by the system prompt (`--system-prompt`, `--system-prompt-file` or
`HERALD_SYSTEM_PROMPT`); the built-in one is Frieren. `--history N` controls how many past
exchanges the model remembers (default 10, 0 for none).

### Languages

XTTS-v2 speaks 17 languages: `en es fr de it pt pl tr ru nl cs ar zh-cn hu ko ja hi`. Pick one
with `--language` (or `HERALD_LANGUAGE`); the default is English.

**A voice trained on English can speak other languages**, because the base model is
multilingual and only the voice was adapted. What you can expect: the timbre carries over, but
the accent can lean towards the training language, and the more the fine-tuning pulls the model
towards one language the more the others can suffer. It is worth listening for yourself:

```bash
herald synthesize "Sono passati molti anni da quando ho lasciato il mio villaggio." \
    --language it --checkpoint models/frieren -o output/prova_it.wav --play
```

For the best result in a language, fine-tune on recordings **in that language** of the same
speaker (`herald train --language it`). A dataset holds one language: Herald trains one
language per run.

Two practical details for non-English speech:

- **Chunk length.** Each language has a limit on how much text XTTS can speak in one go before
  the audio may be cut short (250 characters for English, 213 for Italian, 82 for Chinese).
  `--max-chars` defaults to the limit of your language.
- **Chat.** Ollama answers in the language of the system prompt, so tell it, for example
  `herald chat --language it --system-prompt "Sei Frieren, un'elfa maga. Rispondi sempre in italiano, in una o due frasi brevi."`.
  Llama 3.2 lists Italian among its supported languages; small models are weaker outside English.

## Configuration

Every setting has an environment variable; a command line option overrides it. Empty variables
count as unset. Relative paths are resolved from the directory you run herald in.

| Variable | Default | Meaning |
| --- | --- | --- |
| `HERALD_PROJECT_ROOT` | source checkout, else the current directory | Base for the defaults below |
| `HERALD_DATASET_DIR` | `<root>/dataset/frieren` | Dataset directory |
| `HERALD_MODELS_DIR` | `<root>/models` | Where fine-tuned voices are stored by `train` |
| `HERALD_CHECKPOINT_DIR` | `<models dir>/xtts_v2` | Base XTTS-v2 weights |
| `HERALD_CHECKPOINT` | none (base voice) | Fine-tuned voice for `synthesize` and `chat` |
| `HERALD_RUNS_DIR` | `<root>/runs` | Training runs |
| `HERALD_OUTPUT_DIR` | `<root>/output` | Generated audio |
| `HERALD_DEVICE` | `auto` | `auto` (cuda, then mps, then cpu), `cpu`, `mps`, `cuda`, `cuda:N` |
| `HERALD_LANGUAGE` | `en` | Language code passed to XTTS |
| `HERALD_OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `HERALD_OLLAMA_MODEL` | `llama3.2:3b` | Ollama model |
| `HERALD_OLLAMA_TIMEOUT` | `120` | Seconds to wait for a reply |
| `HERALD_SYSTEM_PROMPT` | Frieren | System prompt for `chat` |
| `HERALD_CHECKPOINT_URL` | Coqui's download gateway | Where `download-checkpoints` fetches from |

In Docker, put the ones you need in `.env` (see `.env.example`). Compose forwards the model,
voice, language, device and Ollama settings; the path variables are fixed by the bind mounts.

## Development

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
  config.py, paths.py        settings from HERALD_* variables; project paths
  dataset/metadata.py        metadata loading, audio lookup, reference clips, train/eval split
  tts/checkpoints.py         base weight download
  tts/engine.py              model loading, synthesis (short, long, streamed), WAV output
  tts/train.py               fine-tuning and promotion of the best model
  llm/ollama_client.py       Ollama chat client (text and tool calls)
  assistant.py               conversation state and the tool-calling loop
  audio_playback.py          best-effort playback
  cli.py                     the `herald` command
```

## Where it is going

Herald is meant to become a voice-controlled home assistant: an Ollama model that can *do*
things (timers first), speaking through an ESP32 while Ollama and XTTS run on a PC. Only the
foundations exist today:

- **Tools.** `herald.assistant.Assistant` already runs Ollama's tool-calling loop, but no tool is
  registered yet. Adding one is a small change:

  ```python
  from herald.assistant import Assistant, Tool, ToolRegistry

  tools = ToolRegistry()
  tools.register(Tool(
      name="set_timer",
      description="Start a countdown timer.",
      parameters={"type": "object",
                  "properties": {"seconds": {"type": "integer"}},
                  "required": ["seconds"]},
      handler=lambda seconds: f"Timer set for {seconds} seconds.",
  ))
  assistant = Assistant(client, system_prompt, tools=tools)
  ```

  A handler returns a short text for the model; errors are handed back to the model instead of
  crashing the conversation. A timer that must *speak later* needs a scheduler and a way to
  push audio; that part is not designed yet.
- **Speech.** `XttsEngine` is a long-lived object, and `synth_stream()` yields audio chunk by
  chunk, so a server can start sending audio before the whole reply is synthesized. `chat` still
  waits for the full reply.
- **Front ends.** The terminal loop in `cli.py` is thin on purpose: a future ESP32 server would
  reuse `Assistant` and the engine and replace only the input and output.

## Security note

`.pth` checkpoints are Python pickles: loading one can run code. Only load checkpoints you
made or that come from someone you trust, and be careful with models shared by others.

## Licenses

- Herald's own code is under the [MIT License](LICENSE).
- The XTTS-v2 weights are **not** covered by it: they stay under the Coqui Public Model License
  (see the [model card](https://huggingface.co/coqui/XTTS-v2)), which restricts commercial use.
  Fine-tuned checkpoints derived from them inherit these terms.
- Datasets keep their own licenses. Check that the license of your dataset allows what you
  intend to do with it and with a voice cloned from it.
