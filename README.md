# Herald

Give an AI assistant **a voice you choose**. Herald clones a voice from a short recording, and
`herald chat` lets you talk to a language model that answers out loud in that voice. It can also
do things, such as setting a timer, and you can teach it new ones by dropping a script in a folder.

Everything runs on your own computer: no account, no cloud service, no data leaves the machine.

It is built on [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) (voice cloning) and
[Ollama](https://ollama.com) (the language model).

## What you need

**You do not need a trained model or a dataset to start.** Herald clones a voice from one short
recording and downloads the base model (about 2 GB) by itself, once, the first time it needs it.

| You need | Details |
| --- | --- |
| A computer | A Mac with Apple Silicon is the tested setup. Linux and Windows should work too, ideally through [Docker](docs/docker.md), but have not been tried with the full stack |
| Disk space | About 5 GB: the Python libraries, plus the base model |
| Python 3.11 | Or Docker instead; see [docs/docker.md](docs/docker.md) |
| **A recording of the voice** | 6 to 30 seconds of one person speaking: clear, no music, no echo. A WAV file is best. A phone voice memo of your own voice works |
| Ollama | Only for `chat`. Install it from [ollama.com](https://ollama.com) |

## Quick start

### 1. Install

```bash
git clone <this repository's URL> herald
cd herald
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

Installing takes a few minutes. `herald` only exists inside that virtual environment, so run the
`source` line again in every new terminal (if the shell says `command not found: herald`, that is
why).

### 2. Make it speak

Put your recording in the project folder. The `dataset/` folder is ignored by git, so a voice kept
there never ends up on GitHub:

```bash
mkdir -p dataset/me
cp ~/Downloads/my_voice.wav dataset/me/voice.wav
```

Now say something with it:

```bash
herald synthesize "Hello! This is my cloned voice." --reference-wav dataset/me/voice.wav --play
```

The very first run downloads the base model (about 2 GB) into `models/xtts_v2/`. That takes a few
minutes and happens only once. The result is saved in `output/` and played.

If it does not sound right, try a longer or cleaner recording, or give several: repeat
`--reference-wav` for each clip.

### 3. Talk to it

Install [Ollama](https://ollama.com), start it, and download a model:

```bash
ollama pull llama3.2:3b            # about 2 GB, once
```

Then chat, using the same recording:

```bash
herald chat --reference-wav dataset/me/voice.wav
```

Type a message and hear the answer. An empty line (or Ctrl-D) quits. Loading the voice takes
15 to 20 seconds at the start. To use another language, add `--language it` and tell the model
which language to speak with `--system-prompt "Rispondi sempre in italiano."`.

### 4. Keep your settings in a profile

Typing the recording, the language and the personality every time gets old. A **profile** keeps
them in one small file:

```bash
herald new-profile me --reference-wav dataset/me/voice.wav --language en \
    --system-prompt "You are a calm assistant. Answer in one or two short sentences."

herald chat --profile me
```

Profiles live in the `profiles/` folder. [`profiles/_template.toml`](profiles/_template.toml) lists
every setting, and `herald profiles` shows what you have.

## Make the voice better

A single recording gives a recognisable voice. To get closer to the real one you can **fine-tune**
the model on more speech. That needs tens of minutes of clean recordings cut into short clips, with
a text file of what is said in each, and then an hour or more of training. The whole procedure is
in [docs/training.md](docs/training.md).

**Already have a trained voice from someone?** (for example this project's author gave you the
files of a fine-tuned model.) Copy them into a folder named after the voice, then point Herald at
it:

```bash
mkdir -p models/frieren
cp /where/you/got/it/best_model.pth /where/you/got/it/config.json models/frieren/
herald chat --checkpoint models/frieren --reference-wav dataset/me/voice.wav
```

A trained voice still needs a short reference recording, as above. Trained voices are too big for
git (about 2 GB), which is why they are shared separately.

## What the assistant can do

Besides talking, the assistant can run **tools**. One comes built in: ask for a timer ("set a timer
for 10 minutes, the pasta will be ready") and it speaks up when time is over. You can add your own
by dropping a Python script into the [`tools/`](tools/README.md) folder. `herald tools` shows what
is available.

## When something does not work

| What you see | What to do |
| --- | --- |
| `command not found: herald` | Activate the environment: `source .venv/bin/activate` |
| `No dataset found ... pass --reference-wav` | Give a recording: `--reference-wav path/to/voice.wav` |
| `Cannot connect to Ollama` | Start the Ollama app (or run `ollama serve`) and check that the model is pulled |
| No sound | Herald plays with `afplay` (macOS), `winsound` (Windows) or a common Linux player. Add `--save-dir some_folder` to keep the audio as files |
| The first start is slow | Loading the voice takes 15 to 20 seconds. Later answers are faster |
| The download of the base model fails | Run `herald download-checkpoints` again: files already finished are kept, the unfinished one starts over |
| Anything else | Run the command again with `-v` before it (`herald -v chat`) for details |

## Where to read more

| | |
| --- | --- |
| [docs/training.md](docs/training.md) | Prepare a dataset and fine-tune a voice |
| [docs/reference.md](docs/reference.md) | Every command and option, settings, languages, folder layout |
| [docs/docker.md](docs/docker.md) | Run everything in Docker (Linux, Windows, GPU) |
| [tools/README.md](tools/README.md) | Write your own tools for the assistant |
| [docs/development.md](docs/development.md) | Tests, code layout and the project's direction |

## The folders, at a glance

| Folder | What it is | In git? |
| --- | --- | --- |
| `src/herald/` | The program | yes |
| `tools/` | Your tool scripts for the assistant | yes |
| `profiles/` | Voice profiles | yes |
| `docs/`, `tests/`, `docker/` | Guides, automatic tests, Docker files | yes |
| `dataset/` | Your recordings (and training data) | no, private |
| `models/` | The base model, and trained voices (`models/<name>/`) | no, too big |
| `runs/` | Training logs | no |
| `output/` | Audio that Herald generates | no |

A fresh clone has none of the folders marked "no". Herald creates `models/`, `runs/` and `output/`
on its own; you create `dataset/` when you add a recording.

## Licenses and credits

- Herald's own code is under the [MIT License](LICENSE).
- The XTTS-v2 model is **not** covered by it: it stays under the Coqui Public Model License (see
  its [model card](https://huggingface.co/coqui/XTTS-v2)), which restricts commercial use. A voice
  fine-tuned from it inherits those terms.
- Voices are personal data. Clone only voices you have the right to use, and check the license of
  any dataset you train on.
- `.pth` model files are Python pickles: loading one can run code. Only load models you made or
  that come from someone you trust.
