"""Command line interface.

``herald synthesize | train | slim | chat | tools | profiles | new-profile | doctor |
download-checkpoints``.

This module must stay cheap to import so that ``herald --help`` is instant: torch,
coqui-tts, numpy and requests are only imported inside the command handlers.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from herald import __version__
from herald.config import DEFAULT_OLLAMA_TIMEOUT, Settings, read_utf8
from herald.errors import ConfigError, HeraldError, ProfileError

logger = logging.getLogger(__name__)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Like ArgumentDefaultsHelpFormatter, but silent about ``None`` and ``False`` defaults."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        text = action.help or ""
        shows_default = (
            action.option_strings
            and action.default is not None
            and action.default is not False
            and action.default is not argparse.SUPPRESS
            and "%(default)" not in text
        )
        return f"{text} (default: %(default)s)" if shows_default else text


def _number(kind: type[int] | type[float], minimum: float, *, above: bool = False):
    """An argparse ``type``: a ``kind`` that is at least ``minimum`` (greater with ``above``).

    Rejecting bad values while parsing gives a usage error at once, instead of a traceback
    after the model has been loaded.
    """

    def parse(text: str):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid {kind.__name__} value: {text!r}") from None
        if not (value > minimum if above else value >= minimum):  # also rejects NaN
            raise argparse.ArgumentTypeError(
                f"must be {'greater than' if above else 'at least'} {minimum:g}, got {text}"
            )
        return value

    return parse


_POSITIVE_INT = _number(int, 1)
_NON_NEGATIVE_FLOAT = _number(float, 0)
_POSITIVE_FLOAT = _number(float, 0, above=True)


# --- parser -----------------------------------------------------------------------------------


# Where a default comes from, for the options a profile can set: the profile if there is one
# (``pd``, see ``profiles.profile_defaults``), otherwise the environment or the built-in value.
# So the precedence is: command line option > profile > environment > built-in default.


def _paths_parent(settings: Settings, pd: Mapping) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=pd.get("dataset_dir", settings.dataset_dir),
        help="Dataset directory (audio/ plus metadata.csv). Env: HERALD_DATASET_DIR.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=settings.checkpoint_dir,
        help="Base XTTS-v2 weights. Env: HERALD_CHECKPOINT_DIR.",
    )
    return p


def _tools_parent(settings: Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--tools-dir",
        type=Path,
        default=settings.tools_dir,
        help="Directory with your own tool scripts (*.py). Env: HERALD_TOOLS_DIR.",
    )
    return p


def _profile_parent(settings: Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--profile",
        metavar="NAME_OR_PATH",
        default=settings.profile,
        help="Voice profile: a name from the profiles directory, or the path of a .toml file. "
        "It sets the defaults of the options it contains; options given here win over it. "
        "Env: HERALD_PROFILE.",
    )
    return p


def _voice_parent(settings: Settings, pd: Mapping) -> argparse.ArgumentParser:
    """Options shared by the commands that speak (synthesize, chat)."""
    p = argparse.ArgumentParser(add_help=False, parents=[_paths_parent(settings, pd)])
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=pd.get("checkpoint", settings.checkpoint),
        help="Fine-tuned voice: a checkpoint file or a directory holding best_model.pth "
        "(e.g. models/frieren). Omit to use the base model. Env: HERALD_CHECKPOINT.",
    )
    p.add_argument(
        "--reference-wav",
        action="append",
        type=Path,
        metavar="WAV",
        # No default here, not even a profile's: argparse would add the clips given on the
        # command line to it instead of replacing it. main() fills it in after parsing.
        help="Reference clip for the voice (repeatable). Default: random clips from the dataset.",
    )
    p.add_argument(
        "--num-references",
        type=_POSITIVE_INT,
        default=pd.get("num_references", 3),
        help="Dataset clips to use as reference.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for picking reference clips.")
    p.add_argument(
        "--language",
        default=pd.get("language", settings.language),
        help="Language. Env: HERALD_LANGUAGE.",
    )
    p.add_argument(
        "--temperature",
        type=_POSITIVE_FLOAT,
        default=pd.get("temperature", 0.7),
        help="Sampling temperature.",
    )
    p.add_argument(
        "--device",
        default=pd.get("device", settings.device),
        help="auto, cpu, mps, cuda or cuda:N (auto: cuda > mps > cpu). Env: HERALD_DEVICE.",
    )
    p.add_argument(
        "--pause-ms", type=_NON_NEGATIVE_FLOAT, default=150, help="Silence between text chunks."
    )
    p.add_argument(
        "--max-chars",
        type=_POSITIVE_INT,
        help="Longest text chunk to synthesize. "
        "Default: the limit of --language (250 for English, 213 for Italian).",
    )
    return p


def build_parser(settings: Settings, pd: Mapping | None = None) -> argparse.ArgumentParser:
    """The argument parser. ``pd`` holds the defaults a voice profile sets, by option name."""
    pd = {} if pd is None else pd
    parser = argparse.ArgumentParser(
        prog="herald",
        description="Voice-cloning text-to-speech (XTTS-v2) with fine-tuning and LLM chat.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, help_: str, **kwargs) -> argparse.ArgumentParser:
        return sub.add_parser(
            name, help=help_, description=help_, formatter_class=_HelpFormatter, **kwargs
        )

    # synthesize
    p = add(
        "synthesize",
        "Speak a text and write it to a WAV file.",
        parents=[_voice_parent(settings, pd), _profile_parent(settings)],
    )
    p.add_argument(
        "text", nargs="?", help="Text to speak. Or use --text-file, or pipe it on stdin."
    )
    p.add_argument("--text-file", type=Path, help="Read the text from this file.")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output WAV. Default: <output dir>/herald_<timestamp>.wav.",
    )
    p.add_argument("--play", action="store_true", help="Play the result when done (best effort).")
    p.set_defaults(func=_cmd_synthesize)

    # train
    p = add(
        "train",
        "Fine-tune XTTS-v2 on a dataset (CUDA if available, otherwise the CPU).",
        parents=[_paths_parent(settings, pd), _profile_parent(settings)],
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=settings.runs_dir,
        help="Training logs and TensorBoard data. Env: HERALD_RUNS_DIR.",
    )
    p.add_argument(
        "--metadata-file",
        default="metadata.csv",
        help="Metadata file inside the dataset directory (all rows; split for training).",
    )
    p.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="Fraction of the rows held out for evaluation.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for the train/eval split.")
    p.add_argument(
        "--speaker-name",
        default=pd.get("speaker_name"),
        help="Speaker label; the voice is saved as <models dir>/<speaker>/best_model.pth "
        "(env HERALD_MODELS_DIR). Default: the dataset directory name.",
    )
    p.add_argument(
        "--language",
        default=pd.get("language", settings.language),
        help="Language. Env: HERALD_LANGUAGE.",
    )
    p.add_argument("--run-name", help="Run name. Default: <speaker>_full or <speaker>_smoke.")
    p.add_argument("--project-name", help="Project name. Default: <speaker>_xtts.")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick sanity run: 1 epoch on a few dozen samples. Other options still override it.",
    )
    p.add_argument("--epochs", type=int, help="Training epochs (full: 5, smoke: 1).")
    p.add_argument("--batch-size", type=int, help="Batch size (full: 2, smoke: 1).")
    p.add_argument("--eval-batch-size", type=int, help="Eval batch size. Default: the batch size.")
    p.add_argument(
        "--grad-accum-steps", type=int, help="Gradient accumulation (full: 16, smoke: 4)."
    )
    p.add_argument("--lr", type=float, help="Learning rate (5e-6).")
    p.add_argument("--num-workers", type=int, help="Data loader workers (2).")
    p.add_argument(
        "--save-step", type=int, help="Checkpoint every N steps (full: 250, smoke: 500)."
    )
    p.add_argument("--max-train-samples", type=int, help="Use only the first N training samples.")
    p.add_argument("--max-eval-samples", type=int, help="Use only the first N eval samples.")
    p.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep the trainer's checkpoint files (about 5.6 GB each) in the run directory "
        "instead of deleting them once the best model is safe in the models directory.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved settings and check the dataset, without training.",
    )
    p.set_defaults(func=_cmd_train)

    # slim
    p = add(
        "slim",
        "Shrink a fine-tuned checkpoint by dropping the optimizer state it carries "
        "(about 3x smaller, same voice).",
    )
    p.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint file, or a voice or run directory holding best_model.pth.",
    )
    target = p.add_mutually_exclusive_group()
    target.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the slim checkpoint here. Default: <name>.slim.pth next to the original, "
        "which is left alone.",
    )
    target.add_argument(
        "--replace",
        action="store_true",
        help="Replace the original file with the slim one instead of writing a copy.",
    )
    p.set_defaults(func=_cmd_slim)

    # chat
    p = add(
        "chat",
        "Talk to a local Ollama model and hear the replies in the cloned voice.",
        parents=[_voice_parent(settings, pd), _tools_parent(settings), _profile_parent(settings)],
    )
    p.add_argument("--ollama-url", default=settings.ollama_url, help="Env: HERALD_OLLAMA_URL.")
    p.add_argument(
        "--ollama-model",
        default=pd.get("ollama_model", settings.ollama_model),
        help="Env: HERALD_OLLAMA_MODEL.",
    )
    p.add_argument(
        "--ollama-timeout",
        type=_POSITIVE_FLOAT,
        # No default here: the environment value is only validated when chat needs it.
        help=f"Seconds. Env: HERALD_OLLAMA_TIMEOUT. Default: {DEFAULT_OLLAMA_TIMEOUT:g}.",
    )
    prompt = p.add_mutually_exclusive_group()
    prompt.add_argument(
        "--system-prompt",
        default=pd.get("system_prompt", settings.system_prompt),
        help="Env: HERALD_SYSTEM_PROMPT.",
    )
    prompt.add_argument(
        "--system-prompt-file", type=Path, help="Read the system prompt from a file."
    )
    p.add_argument(
        "--history",
        type=int,
        default=pd.get("history", 10),
        help="Past exchanges sent to the model (0: none).",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        help="Keep every reply as a WAV file in DIR. By default replies are played and then "
        "discarded, so chat does not fill the disk.",
    )
    p.add_argument(
        "--no-play",
        action="store_true",
        help="Do not play the replies. Without --save-dir either, chat in text only "
        "(the voice model is not even loaded).",
    )
    p.add_argument(
        "--no-tools",
        action="store_true",
        help="Plain chat: do not offer any tools (timers, your own scripts) to the model.",
    )
    p.add_argument(
        "--always-offer-tools",
        action="store_true",
        help="Offer every tool on every message, ignoring the tools' trigger words. "
        "Use it with models that handle tools well.",
    )
    p.set_defaults(func=_cmd_chat)

    # tools
    p = add(
        "tools",
        "List the tools the assistant can use, built-in and from your tools directory.",
        parents=[_tools_parent(settings)],
    )
    p.set_defaults(func=_cmd_tools)

    # profiles
    p = add(
        "profiles",
        "List the voice profiles, or show one with its resolved values and problems.",
    )
    p.add_argument("name", nargs="?", help="Profile to show. Without it, every profile is listed.")
    p.set_defaults(func=_cmd_profiles)

    # new-profile
    p = add(
        "new-profile",
        "Create a voice profile: a voice and its character in profiles/NAME.toml.",
    )
    p.add_argument("name", help="Profile name: letters, digits, '-' and '_'.")
    p.add_argument("--description", help="One line about the profile, shown by `herald profiles`.")
    p.add_argument("--checkpoint", type=Path, help="Fine-tuned voice. Omit for the base voice.")
    p.add_argument("--dataset", type=Path, help="Dataset directory to pick reference clips from.")
    p.add_argument(
        "--reference-wav",
        action="append",
        type=Path,
        metavar="WAV",
        help="Reference clip (repeatable), used instead of picking clips from the dataset.",
    )
    p.add_argument("--language", help="Language of the voice, for example en or it.")
    character = p.add_mutually_exclusive_group()
    character.add_argument("--system-prompt", help="The character: the text of the system prompt.")
    character.add_argument(
        "--system-prompt-file", type=Path, help="The character, read from this file."
    )
    p.add_argument("--ollama-model", help="Ollama model to chat with.")
    p.add_argument("--force", action="store_true", help="Replace the profile if it exists.")
    p.set_defaults(func=_cmd_new_profile)

    # doctor
    p = add(
        "doctor",
        "Check that this computer is ready to run Herald and say what is missing.",
        parents=[_profile_parent(settings)],
    )
    p.set_defaults(func=_cmd_doctor)

    # download-checkpoints
    p = add(
        "download-checkpoints",
        "Download the base XTTS-v2 weights (about 2 GB) into the checkpoint directory.",
        parents=[_paths_parent(settings, pd)],
    )
    p.add_argument(
        "--base-url", default=settings.checkpoint_url, help="Env: HERALD_CHECKPOINT_URL."
    )
    p.set_defaults(func=_cmd_download_checkpoints)

    return parser


# --- handlers ---------------------------------------------------------------------------------


def _cmd_download_checkpoints(args: argparse.Namespace, settings: Settings) -> int:
    from herald.tts import checkpoints

    path = checkpoints.ensure_base_checkpoints(args.checkpoint_dir, base_url=args.base_url)
    print(f"Checkpoints ready in {path}")
    return 0


def _read_text(args: argparse.Namespace) -> str:
    """The text to speak: the TEXT argument, or --text-file, or stdin when it is piped."""
    if args.text is not None and args.text_file:
        raise ConfigError("Pass either TEXT or --text-file, not both")
    if args.text_file:
        text = read_utf8(args.text_file)
    elif args.text is not None:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise ConfigError("No text given: pass TEXT, use --text-file, or pipe text on stdin")
    if not text.strip():
        raise ConfigError("The text to synthesize is empty")
    return text


def _cmd_synthesize(args: argparse.Namespace, settings: Settings) -> int:
    text = _read_text(args)
    output = args.output or settings.output_dir / f"herald_{time.strftime('%Y%m%d_%H%M%S')}.wav"

    from herald import audio_playback
    from herald.tts import engine
    from herald.voice import load_engine

    eng = load_engine(args, settings)
    wav = eng.synth_long(text, pause_ms=args.pause_ms, max_chars=args.max_chars)
    engine.save_wav(output, wav, eng.sample_rate)
    print(f"Wrote {output} ({len(wav) / eng.sample_rate:.1f}s)")
    if args.play and not audio_playback.play_wav(output):
        print("No audio player available; skipping playback.", file=sys.stderr)
    return 0


def _cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    from herald.tts import train

    params = train.make_params(
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        runs_dir=args.runs_dir,
        models_dir=settings.models_dir,
        checkpoint_url=settings.checkpoint_url,
        preset="smoke" if args.smoke else "full",
        language=args.language,
        metadata_file=args.metadata_file,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        speaker_name=args.speaker_name,
        run_name=args.run_name,
        project_name=args.project_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        num_workers=args.num_workers,
        save_step=args.save_step,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        keep_checkpoints=args.keep_checkpoints,
    )

    if args.dry_run:
        for key, value in dataclasses.asdict(params).items():
            print(f"{key}: {value}")
        report = train.validate_dataset(params)
        print(f"train rows: {report.train_rows}")
        print(f"eval rows: {report.eval_rows}")
        _print_dataset_check(report)
        if report.missing_audio:
            print(f"missing audio: {len(report.missing_audio)} (e.g. {report.missing_audio[0]})")
            return 1
        print("dataset OK")
        return 0

    result = train.run_training(params)
    print(f"Training finished. Run directory: {result.run_dir}")
    if result.model_path is None:
        print("No fine-tuned model was saved (no best_model.pth in the run directory).")
    else:
        size = (
            f" ({_human_size(result.model_path.stat().st_size)})"
            if result.model_path.is_file()
            else ""
        )
        print(f"Fine-tuned voice: {result.model_path}{size}")
        if result.freed_bytes > 0:
            print(f"Freed {_human_size(result.freed_bytes)} of checkpoints")
        print(f"Use it with:  {_synthesize_hint(args, settings, result.model_path)}")
        _create_voice_profile(params, result.model_path, settings)
    return 0


def _create_voice_profile(params, model_path: Path, settings: Settings) -> None:
    """Give a freshly trained voice a profile, unless one with that name exists (never replaced).

    The profile is named after the voice folder, so a smoke run (``<speaker>_smoke``) never
    touches the profile of the real voice.
    """
    from herald import profiles

    name = model_path.parent.name
    root = settings.project_root
    values = {
        "description": f"Created by herald train on {time.strftime('%Y-%m-%d')}",
        "checkpoint": profiles.relative_to_root(model_path.parent, root),
        "dataset": profiles.relative_to_root(params.dataset_dir, root),
        "language": params.language,
    }
    try:
        path, created = profiles.ensure_profile(settings.profiles_dir, name, values)
    except ProfileError as exc:  # the training worked; only the convenience did not
        print(f"warning: no profile was created: {exc}", file=sys.stderr)
        return
    if created:
        shown = profiles.relative_to_root(path, root)
        print(f"Profile created: {shown} (use it with: herald chat --profile {name})")


def _print_dataset_check(report) -> None:
    """What ``train --dry-run`` found in the audio and the transcripts: totals, then warnings.

    Warnings never fail the dry run; only missing audio does.
    """
    audio = report.audio
    lines = []
    if audio is not None:
        channel_names = {1: "mono", 2: "stereo"}
        lines += [
            f"speech: {audio.total_seconds / 60:.1f} min in {audio.clips_checked} clip(s)",
            "sample rates: "
            + ", ".join(f"{rate} Hz x{n}" for rate, n in sorted(audio.sample_rates.items())),
            "channels: "
            + ", ".join(
                f"{channel_names.get(ch, f'{ch} channels')} x{n}"
                for ch, n in sorted(audio.channels.items())
            ),
        ]
        if audio.too_short:
            lines.append(
                f"warning: {len(audio.too_short)} clip(s) shorter than 0.5 s will be skipped "
                f"by the trainer (e.g. {audio.too_short[0]})"
            )
        if audio.too_long:
            lines.append(
                f"warning: {len(audio.too_long)} clip(s) longer than ~11.6 s will be skipped "
                f"(e.g. {audio.too_long[0]})"
            )
    if report.long_text_rows:
        lines.append(
            f"warning: {report.long_text_rows} transcript(s) over 200 characters may be skipped"
        )
    if audio is not None:
        if audio.weak_reference:
            lines.append(
                f"warning: {audio.weak_reference} clip(s) shorter than 3 s: usable, "
                "but they teach the voice less"
            )
        if audio.unreadable:
            lines.append(
                f"warning: {len(audio.unreadable)} file(s) are not WAV and were not inspected"
            )
    if lines:
        print("Dataset check:")
        for line in lines:
            print(f"  {line}")


def _human_size(num_bytes: int) -> str:
    """``5600000000`` -> ``"5.6 GB"`` (decimal units, like the OS file managers)."""
    if num_bytes < 1000:
        return f"{num_bytes} B"
    value = float(num_bytes)
    for unit in ("KB", "MB", "GB"):
        value /= 1000
        if value < 1000:
            return f"{value:.1f} {unit}"
    return f"{value / 1000:.1f} TB"


def _cmd_slim(args: argparse.Namespace, settings: Settings) -> int:
    from herald.tts import engine, slim

    source = engine.resolve_finetuned_checkpoint(args.checkpoint)
    result = slim.slim_checkpoint(source, args.output, replace=args.replace)
    if result.already_slim:
        print(f"{result.source}: already slim ({_human_size(result.size_after)}), nothing to do")
        return 0
    change = (
        f"{_human_size(result.size_before)} -> {_human_size(result.size_after)} "
        f"(saved {_human_size(result.size_before - result.size_after)})"
    )
    if args.replace:
        print(f"{result.source}: {change}, the original was replaced")
    else:
        print(f"{result.source} -> {result.path}: {change}")
    return 0


def _shell_quote(text: str) -> str:
    """Quote ``text`` for the shell the user will paste the command into."""
    return subprocess.list2cmdline([text]) if os.name == "nt" else shlex.quote(text)


def _synthesize_hint(args: argparse.Namespace, settings: Settings, model_path: Path) -> str:
    """The ``herald synthesize`` command that speaks with a freshly trained voice.

    Options that differ from the defaults are included, so the command works as printed.
    """
    voice = model_path.parent
    try:
        voice = voice.relative_to(Path.cwd())
    except ValueError:
        pass
    parts = ["herald", "synthesize", '"Hello."', "--checkpoint", _shell_quote(str(voice))]
    for flag, value, default in (
        ("--dataset-dir", args.dataset_dir, settings.dataset_dir),
        ("--checkpoint-dir", args.checkpoint_dir, settings.checkpoint_dir),
        ("--language", args.language, settings.language),
    ):
        if value != default:
            parts += [flag, _shell_quote(str(value))]
    return " ".join(parts)


def _cmd_chat(args: argparse.Namespace, settings: Settings) -> int:
    from herald.chat import run_chat

    return run_chat(args, settings)


def _table(rows: list[tuple[str, ...]]) -> list[str]:
    """Rows of text as lines with the columns aligned (the last column is not padded)."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=False))
        + "  "
        + row[-1]
        for row in rows
    ]


def _cmd_profiles(args: argparse.Namespace, settings: Settings) -> int:
    from herald import profiles

    root = settings.project_root
    if args.name is not None:
        profile = profiles.load_profile(args.name, settings.profiles_dir, root)
        print(f"{profile.name}  ({profiles.relative_to_root(profile.path, root)})")
        for key, value in profile.values.items():
            if isinstance(value, Path):
                value = profiles.relative_to_root(value, root)
            elif isinstance(value, list):
                value = ", ".join(profiles.relative_to_root(item, root) for item in value)
            print(f"  {key}: " + str(value).replace("\n", "\n    "))
        if "checkpoint" not in profile.values:
            print("  voice: base voice (no checkpoint)")
        for problem in profiles.check_profile(profile):
            print(f"  ! {problem}")
        return 0

    summaries = profiles.list_profiles(settings.profiles_dir, root)
    if not summaries:
        print(f"No profiles in {settings.profiles_dir}. Create one with: herald new-profile NAME")
        return 0
    rows = [
        (
            summary.name,
            summary.language or "-",
            summary.checkpoint or "base voice",
            summary.description,
        )
        for summary in summaries
    ]
    for summary, line in zip(summaries, _table(rows), strict=True):
        print(("! " if summary.problems else "  ") + line.rstrip())
        for problem in summary.problems:
            print(f"    ! {problem}")
    return 0


_PROFILE_NAME = re.compile(r"[A-Za-z0-9_-]+")


def _cmd_new_profile(args: argparse.Namespace, settings: Settings) -> int:
    from herald import profiles

    if not _PROFILE_NAME.fullmatch(args.name):
        raise ProfileError(f"Invalid profile name {args.name!r}: use letters, digits, '-' and '_'")
    root = settings.project_root

    def project_path(path: Path) -> str:
        # Typed relative to the current directory; a profile's paths are relative to the root.
        return profiles.relative_to_root(Path(os.path.abspath(path.expanduser())), root)

    values: dict[str, object] = {}
    if args.description is not None:
        values["description"] = args.description
    if args.checkpoint is not None:
        values["checkpoint"] = project_path(args.checkpoint)
    if args.dataset is not None:
        values["dataset"] = project_path(args.dataset)
    if args.reference_wav:
        values["reference_wavs"] = [project_path(wav) for wav in args.reference_wav]
    if args.language is not None:
        values["language"] = args.language
    if args.system_prompt is not None:
        values["system_prompt"] = args.system_prompt
    if args.system_prompt_file is not None:
        if not args.system_prompt_file.is_file():
            raise ProfileError(f"System prompt file not found: {args.system_prompt_file}")
        values["system_prompt_file"] = project_path(args.system_prompt_file)
    if args.ollama_model is not None:
        values["ollama_model"] = args.ollama_model

    path = settings.profiles_dir / f"{args.name}.toml"
    shown = profiles.relative_to_root(path, root)
    existed = path.exists()
    if existed and not args.force:
        raise ProfileError(f"{shown} already exists: use --force to replace it")
    profiles.write_profile(path, values, overwrite=args.force)
    profile = profiles.load_profile(args.name, settings.profiles_dir, root)  # it must load again
    for problem in profiles.check_profile(profile):
        print(f"warning: {problem}", file=sys.stderr)
    print(f"{'Replaced' if existed else 'Created'} {shown}")
    print(f"Use it with: herald chat --profile {args.name}")
    return 0


def _cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    from herald.doctor import run_doctor

    return run_doctor(settings, args.profile)


def _describe_parameters(schema: dict) -> list[str]:
    """``name (type, required): description`` for each parameter of a tool schema."""
    parameters = schema.get("function", {}).get("parameters", {})
    required = set(parameters.get("required", ()))
    lines = []
    for name, spec in parameters.get("properties", {}).items():
        flags = f"{spec.get('type', 'any')}{', required' if name in required else ''}"
        description = spec.get("description")
        lines.append(f"{name} ({flags})" + (f": {description}" if description else ""))
    return lines


def _cmd_tools(args: argparse.Namespace, settings: Settings) -> int:
    from herald.tools import ToolContext, load_tools
    from herald.tools.scheduler import Scheduler

    scheduler = Scheduler()
    try:
        report = load_tools(args.tools_dir, ToolContext(say=print, scheduler=scheduler))
    finally:
        scheduler.shutdown()

    schemas = {s["function"]["name"]: s for s in report.registry.schemas()}
    for tool in report.tools:
        print(f"{tool.name}  ({tool.source})")
        for line in tool.description.splitlines():
            print(f"  {line}")
        print(
            f"  offered when the message contains: {', '.join(tool.triggers)}"
            if tool.triggers
            else "  always offered"
        )
        for line in _describe_parameters(schemas.get(tool.name, {})):
            print(f"    {line}")
        print()
    if not report.tools:
        print("No tools found.")
    for error in report.errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if report.errors else 0


# --- entry point ------------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    """Log to stderr. ``-v`` turns on debug output for herald only: at the root it would also
    switch on the very chatty debug logs of numba, matplotlib and friends."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s" if verbose else "%(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("herald").setLevel(logging.DEBUG if verbose else logging.NOTSET)


# The commands that have --profile.
_PROFILE_COMMANDS = ("synthesize", "chat", "train")


def _select_profile(argv: Sequence[str], settings: Settings):
    """The profile the command line selects (``--profile``, else HERALD_PROFILE), or None.

    It must be known before the real parser is built, because its values become the defaults of
    that parser's options, so ``--profile`` is looked for by a tiny parser of its own. Only the
    commands that have the option look at it: a broken HERALD_PROFILE must not get in the way of
    ``herald slim``. Raises ProfileError if the profile cannot be loaded.
    """
    # No global option takes a value, so the command is the first word that is not an option.
    command = next((arg for arg in argv if not arg.startswith("-")), None)
    if command not in _PROFILE_COMMANDS:
        return None
    pre = argparse.ArgumentParser(prog="herald", add_help=False, allow_abbrev=False)
    pre.add_argument("--profile", default=settings.profile)
    name = pre.parse_known_args(list(argv))[0].profile
    if not name:
        return None
    from herald import profiles

    return profiles.load_profile(name, settings.profiles_dir, settings.project_root)


def _use_profile(args: argparse.Namespace, profile, settings: Settings) -> bool:
    """Apply ``profile`` to the parsed arguments. False, after saying why, if it cannot be used.

    Its values are already the defaults of the options (see ``build_parser``), except the list of
    reference clips, which is filled in here. Then comes the check that the files it points to
    exist, so that a wrong path fails at once and not after the model has loaded. A problem is
    ignored when the command line overrides what it is about: ``--checkpoint`` makes the
    profile's checkpoint moot, ``--dataset-dir`` its dataset, and ``--reference-wav`` its clips
    and its dataset (the clips are picked from it only when none are given). ``train`` does not
    check anything: its checkpoint is what it is about to create.
    """
    from herald import profiles

    defaults = profiles.profile_defaults(profile)
    explicit_clips = getattr(args, "reference_wav", None) is not None
    if hasattr(args, "reference_wav") and not explicit_clips and "reference_wav" in defaults:
        args.reference_wav = defaults["reference_wav"]

    shown = profiles.relative_to_root(profile.path, settings.project_root)
    print(f"Profile: {profile.name} ({shown})", file=sys.stderr)
    if args.command == "train":
        return True

    moot = set()
    if args.checkpoint != defaults.get("checkpoint"):
        moot.add("checkpoint")
    if args.dataset_dir != defaults.get("dataset_dir"):
        moot.add("dataset")
    if explicit_clips:
        moot.update(("reference_wavs", "dataset"))
    usable = dataclasses.replace(
        profile, values={k: v for k, v in profile.values.items() if k not in moot}
    )
    problems = profiles.check_profile(usable)
    for problem in problems:
        print(f"error: profile {profile.name}: {problem}", file=sys.stderr)
    return not problems


def _make_output_robust() -> None:
    """Never let a character that the console cannot show crash the program.

    On Windows the console, or a pipe, often has a narrow code page (cp1252, cp850): printing a
    reply with an emoji would raise UnicodeEncodeError. With ``errors="replace"`` the character
    comes out as ``?`` instead.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (OSError, ValueError):  # a closed or unusual stream: leave it as it is
                pass


def main(argv: Sequence[str] | None = None) -> int:
    _make_output_robust()
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        settings = Settings.from_env()
    except HeraldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        profile = _select_profile(argv, settings)
    except ProfileError as exc:
        if "-h" not in argv and "--help" not in argv:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        profile = None  # the help must work even if the profile does not
    pd = {}
    if profile is not None:
        from herald import profiles

        pd = profiles.profile_defaults(profile)

    args = build_parser(settings, pd).parse_args(argv)
    _configure_logging(args.verbose)
    try:
        if profile is not None and not _use_profile(args, profile, settings):
            return 1
        return args.func(args, settings)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except (HeraldError, OSError) as exc:
        if args.verbose:
            logger.exception("Command failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1
