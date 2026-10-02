# Training your own voice

Herald can clone a voice in two ways:

| | Zero-shot (the [README](../README.md) quick start) | Fine-tuned (this page) |
| --- | --- | --- |
| You need | One clip of 6 to 30 seconds | Tens of minutes of clean speech, cut into short clips, with transcripts |
| Setup time | Minutes | A few hours of recording and preparation, then an hour or more of training |
| Result | Usually a recognisable likeness; it depends a lot on the clip | Usually closer to the real voice, especially for a distinctive one |

Start with zero-shot. Come back here when you want the voice to sound more like the real thing.

## Preparing the dataset

Herald does not cut or transcribe audio for you yet: you bring a folder of short clips and a text
file with what is said in each.

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
`metadata.csv` itself (see [Training](#training)). Point Herald at your dataset with `--dataset-dir`
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

## Training

**What training does.** XTTS-v2 turns text plus a few seconds of a reference voice into speech.
Fine-tuning teaches its GPT part (the one that decides how the text should sound) your
speaker: the rest of the model is left alone. Each step takes a clip, converts it to 22,050 Hz
mono, tokenizes its transcript in the chosen `--language`, and cuts a random 3 to 6 second slice
of the same clip as the "voice sample" it must imitate. After every epoch the model is scored on
the held-out evaluation clips and the best one is kept. The defaults are 5 epochs, an effective
batch of 32 clips (batch 2 x 16 accumulation steps) and a small learning rate (5e-6), which
adapts the voice without wrecking what the model already knows.

**What you need.**

- A dataset as described in [Preparing the dataset](#preparing-the-dataset). For another language, pass `--language` (for
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
