# Reference

Everything here is also available from the terminal: `herald --help` and `herald <command> --help`.

## Commands

| Command | What it does |
| --- | --- |
| `herald synthesize` | Speak a text to a WAV file |
| `herald chat` | Talk to an Ollama model and hear the answers |
| `herald train` | Fine-tune XTTS-v2 on a dataset ([training guide](training.md)) |
| `herald slim` | Shrink a trained voice to about a third of its size |
| `herald profiles` | List the voice profiles, or show one |
| `herald new-profile` | Create a voice profile |
| `herald tools` | List the tools the assistant can use |
| `herald download-checkpoints` | Fetch the base XTTS-v2 weights (about 2 GB) |
| `herald doctor` | Check that this computer is ready for Herald and say what is missing |

## Synthesize

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
```

To make a voice the default instead of typing `--checkpoint` every time, use a
[profile](#profiles), or set the variable `HERALD_CHECKPOINT=models/frieren` (macOS and Linux:
`export HERALD_CHECKPOINT=models/frieren`; Windows PowerShell: `$env:HERALD_CHECKPOINT = "models/frieren"`).

A fine-tuned voice only needs the small `config.json` and `vocab.json` of the base model, so
its first run downloads a few hundred KB, not 2 GB. Unset `HERALD_CHECKPOINT` to use the base
voice again.

## Chat

```bash
ollama pull llama3.2:3b
herald chat
```

Type a message, get a reply, and hear it. An empty line or Ctrl-D quits. Replies are played
and then **discarded**, so a long conversation does not fill the disk. Playback is best effort
(`afplay` on macOS, `winsound` on Windows, a few common players on Linux). If a reply cannot be
spoken, the error is printed and the conversation continues.

**How it stays quick.** Herald does not wait for the whole reply. The text appears as the model
writes it, is cut into sentences, and each sentence is turned into speech while the previous one
is still playing. Starting up, it loads the language model and warms up the voice in the
background, so the first answer is not slower than the rest. If you type something while Herald
is still speaking, it stops and answers the new message.

What to expect: loading the voice takes 15 to 20 seconds when `chat` starts. On an Apple Silicon
Mac the first sound of an answer came after about 3 seconds in a test with a three-sentence
reply (against roughly 13 seconds if the whole reply were synthesized first). There can be a short
pause after the first clause, because the voice computes about as fast as it speaks; after that
the speech ran on without gaps. A computer without a GPU will be slower.

| You want | Use |
| --- | --- |
| To hear the replies (default) | `herald chat` |
| To hear them **and** keep them | `herald chat --save-dir replies/` (one WAV per reply) |
| Only to save them, without playing (for example in Docker) | `herald chat --no-play --save-dir replies/` |
| A text-only chat, no voice model loaded | `herald chat --no-play` |

The character is set by the system prompt (`--system-prompt`, `--system-prompt-file` or
`HERALD_SYSTEM_PROMPT`); the built-in one is Frieren. `--history N` controls how many past
exchanges the model remembers (default 10, 0 for none).

## Profiles

A profile is a small TOML file in `profiles/` that bundles a voice and its character, so that
`herald chat --profile frieren` replaces half a dozen options.

```bash
herald profiles                  # list the profiles, and flag what is missing
herald profiles frieren          # show one, with its resolved settings
herald new-profile mario --checkpoint models/mario --dataset dataset/mario --language it
herald chat --profile mario      # also works with `synthesize` and `train`
```

`herald train` creates a profile for the voice it has just trained (it never overwrites an
existing one). [`profiles/_template.toml`](../profiles/_template.toml) documents every setting:

| Key | Meaning |
| --- | --- |
| `description` | Free text, shown by `herald profiles` |
| `checkpoint` | A fine-tuned voice: a model file or a folder with `best_model.pth`. Leave out for the base voice |
| `dataset`, `num_references` | Where reference clips are picked from, and how many |
| `reference_wavs` | A list of reference clips, instead of the dataset |
| `language`, `temperature`, `device` | The language, how varied the voice is, and where it runs |
| `system_prompt` or `system_prompt_file` | The character (a text, or a file holding it) |
| `ollama_model`, `history` | The language model, and how many exchanges it remembers |
| `speaker_name` | The name `herald train` gives a voice |

- Every key is optional, and a misspelt key is an error, so typos do not pass silently.
- Relative paths start from the project folder, not from where you run Herald.
- **Precedence:** an option typed on the command line beats the profile, which beats environment
  variables (`HERALD_*`), which beat the built-in defaults. Set `HERALD_PROFILE=mario` to make a
  profile the default.

## Languages

XTTS-v2 speaks 17 languages: `en es fr de it pt pl tr ru nl cs ar zh-cn hu ko ja hi`. Pick one
with `--language` (or `HERALD_LANGUAGE`); the default is English.

**A voice trained on English can speak other languages**, because the base model is
multilingual and only the voice was adapted. What you can expect: the timbre carries over, but
the accent can lean towards the training language, and the more the fine-tuning pulls the model
towards one language the more the others can suffer. It is worth listening for yourself:

```bash
herald synthesize "Sono passati molti anni da quando ho lasciato il mio villaggio." --language it --checkpoint models/frieren -o output/prova_it.wav --play
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

## Tools

`herald chat` can let the assistant **do** things, not just talk. It calls a tool by itself
when what you say needs one. Herald ships with one, `set_timer`:

```
You: set a timer for 10 minutes, the pasta will be ready
Herald: The timer has been set.
   ... ten minutes later, even while the prompt is waiting for you ...
[Herald] The pasta will be ready
```

The alert is printed and spoken in the cloned voice, in the language of the conversation (the
model writes it). Timers live in memory: when the chat ends, pending ones are cancelled and
Herald says so. The clock does not advance while the computer sleeps.

**Your own tools.** Drop a Python script into the `tools/` folder and the assistant can use it
the next time `chat` starts: a function with `@tool` on top is all it takes, and Herald builds
the rest from its type hints and docstring. [`tools/README.md`](../tools/README.md) explains it
step by step, and `tools/_example.py` has working examples.

```bash
herald tools                     # what the assistant can use, and any script that failed to load
herald chat --no-tools           # a plain chat, without tools
herald chat --tools-dir my_dir   # another folder (or HERALD_TOOLS_DIR)
```

A tool is only offered to the model when your message contains one of the tool's trigger words
(`set_timer` knows "timer", "minutes", "sveglia", "avvisami" and similar in several languages).
Without this, small models call a tool on almost every message, even for "how are you?".
`--always-offer-tools` switches the filter off for models that handle tools well.

Two things to know:
- **Your scripts run with your privileges**, and the model picks their arguments. Read the
  safety notes in `tools/README.md` before writing anything that touches files or programs.
- **Small models are unreliable with tools.** `llama3.2:3b` sometimes misses a request, calls a
  tool nobody asked for, or sends numbers as text. Herald forgives the common slips (numbers
  written as strings, a tool call printed as plain JSON), but a larger model behaves better:
  try `--ollama-model` with one you have pulled.

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
| `HERALD_PROFILE` | none | The voice profile to use (name or path), as if `--profile` were given |
| `HERALD_PROFILES_DIR` | `<root>/profiles` | Where voice profiles live |
| `HERALD_TOOLS_DIR` | `<root>/tools` | Your tool scripts for `chat` |
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

## Folders

```
dataset/<speaker>/         audio/*.wav + metadata.csv          (your data, not in git)
models/xtts_v2/            base XTTS-v2 weights                (downloaded, not in git)
models/<speaker>/          fine-tuned voice: best_model.pth    (produced by `herald train`)
runs/                      training logs and TensorBoard data
output/                    generated audio
profiles/                  voice profiles (see Profiles above)         (in git)
tools/                     your own tool scripts for the assistant   (in git; see tools/README.md)
POC/                       the original notebook and its samples (local only, not in git)
src/herald/                the package
tests/                     unit tests (no model, no network) and integration tests
docs/                      the guides
docker/, docker-compose*.yml, .env.example
```

`dataset/`, `models/`, `runs/`, `output/` and `POC/` are git-ignored: they are large or personal.

## Security note

`.pth` checkpoints are Python pickles: loading one can run code. Only load checkpoints you
made or that come from someone you trust, and be careful with models shared by others.
