"""Command line interface.

``herald synthesize | train | slim | chat | tools | download-checkpoints``.

This module must stay cheap to import so that ``herald --help`` is instant: torch,
coqui-tts, numpy and requests are only imported inside the command handlers.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import itertools
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from herald import __version__
from herald.config import DEFAULT_OLLAMA_TIMEOUT, Settings
from herald.errors import ConfigError, HeraldError

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


def _paths_parent(settings: Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=settings.dataset_dir,
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


def _voice_parent(settings: Settings) -> argparse.ArgumentParser:
    """Options shared by the commands that speak (synthesize, chat)."""
    p = argparse.ArgumentParser(add_help=False, parents=[_paths_parent(settings)])
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=settings.checkpoint,
        help="Fine-tuned voice: a checkpoint file or a directory holding best_model.pth "
        "(e.g. models/frieren). Omit to use the base model. Env: HERALD_CHECKPOINT.",
    )
    p.add_argument(
        "--reference-wav",
        action="append",
        type=Path,
        metavar="WAV",
        help="Reference clip for the voice (repeatable). Default: random clips from the dataset.",
    )
    p.add_argument(
        "--num-references",
        type=_POSITIVE_INT,
        default=3,
        help="Dataset clips to use as reference.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for picking reference clips.")
    p.add_argument("--language", default=settings.language, help="Language. Env: HERALD_LANGUAGE.")
    p.add_argument("--temperature", type=_POSITIVE_FLOAT, default=0.7, help="Sampling temperature.")
    p.add_argument(
        "--device",
        default=settings.device,
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


def build_parser(settings: Settings) -> argparse.ArgumentParser:
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
        "synthesize", "Speak a text and write it to a WAV file.", parents=[_voice_parent(settings)]
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
        parents=[_paths_parent(settings)],
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
        help="Speaker label; the voice is saved as <models dir>/<speaker>/best_model.pth "
        "(env HERALD_MODELS_DIR). Default: the dataset directory name.",
    )
    p.add_argument("--language", default=settings.language, help="Language. Env: HERALD_LANGUAGE.")
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
        parents=[_voice_parent(settings), _tools_parent(settings)],
    )
    p.add_argument("--ollama-url", default=settings.ollama_url, help="Env: HERALD_OLLAMA_URL.")
    p.add_argument(
        "--ollama-model", default=settings.ollama_model, help="Env: HERALD_OLLAMA_MODEL."
    )
    p.add_argument(
        "--ollama-timeout",
        type=_POSITIVE_FLOAT,
        # No default here: the environment value is only validated when chat needs it.
        help=f"Seconds. Env: HERALD_OLLAMA_TIMEOUT. Default: {DEFAULT_OLLAMA_TIMEOUT:g}.",
    )
    prompt = p.add_mutually_exclusive_group()
    prompt.add_argument(
        "--system-prompt", default=settings.system_prompt, help="Env: HERALD_SYSTEM_PROMPT."
    )
    prompt.add_argument(
        "--system-prompt-file", type=Path, help="Read the system prompt from a file."
    )
    p.add_argument(
        "--history", type=int, default=10, help="Past exchanges sent to the model (0: none)."
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

    # download-checkpoints
    p = add(
        "download-checkpoints",
        "Download the base XTTS-v2 weights (about 2 GB) into the checkpoint directory.",
        parents=[_paths_parent(settings)],
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


def _load_engine(args: argparse.Namespace, settings: Settings):
    """Pick reference clips and load the base or fine-tuned model."""
    from herald.dataset import metadata
    from herald.tts import engine

    if args.reference_wav:
        for wav in args.reference_wav:
            if not wav.is_file():
                raise FileNotFoundError(f"Reference wav not found: {wav}")
        references = [str(w) for w in args.reference_wav]
    else:
        metadata_csv = args.dataset_dir / metadata.METADATA_FILE
        if not metadata_csv.is_file():
            raise ConfigError(
                f"No dataset found at {args.dataset_dir} (no {metadata.METADATA_FILE}): pass "
                "--reference-wav CLIP.wav (repeatable, a few seconds of the voice) "
                "or --dataset-dir DIR"
            )
        references = metadata.pick_reference_wavs(
            metadata_csv, args.dataset_dir, n=args.num_references, seed=args.seed
        )
    logger.info("Reference clips: %s", ", ".join(references))

    finetuned = engine.resolve_finetuned_checkpoint(args.checkpoint) if args.checkpoint else None
    return engine.load_engine(
        args.checkpoint_dir,
        references,
        finetuned_checkpoint=finetuned,
        device=args.device,
        language=args.language,
        temperature=args.temperature,
        checkpoint_url=settings.checkpoint_url,
    )


def _read_utf8(path: Path) -> str:
    """Read a UTF-8 text file; a BOM (Windows Notepad adds one) is dropped."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise ConfigError(f"{path} is not a UTF-8 text file") from None


def _read_text(args: argparse.Namespace) -> str:
    """The text to speak: the TEXT argument, or --text-file, or stdin when it is piped."""
    if args.text is not None and args.text_file:
        raise ConfigError("Pass either TEXT or --text-file, not both")
    if args.text_file:
        text = _read_utf8(args.text_file)
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

    eng = _load_engine(args, settings)
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
    return 0


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


def _chat_settings(args: argparse.Namespace, settings: Settings) -> tuple[str, float]:
    """The system prompt and the Ollama timeout. Read first, so that a bad file or value fails
    before the voice model is loaded."""
    system_prompt = (
        _read_utf8(args.system_prompt_file).strip()
        if args.system_prompt_file
        else args.system_prompt
    )
    timeout = args.ollama_timeout if args.ollama_timeout is not None else settings.ollama_timeout
    return system_prompt, timeout


def _build_assistant(args: argparse.Namespace, system_prompt: str, timeout: float, tools):
    """The chat brain: an Ollama client wrapped in an Assistant (system prompt, history, tools)."""
    from herald.assistant import Assistant
    from herald.llm.ollama_client import OllamaClient

    client = OllamaClient(args.ollama_url, args.ollama_model, timeout=timeout)
    return Assistant(
        client,
        system_prompt,
        tools=tools,
        history_turns=args.history,
        use_triggers=not args.always_offer_tools,
    )


def _make_speaker(args: argparse.Namespace, settings: Settings, stack: contextlib.ExitStack):
    """Load the voice and return ``speak(text, kind, number)``, which speaks one text.

    ``kind`` is ``"chat"`` for a reply or ``"alert"`` for a timer, and only matters for the
    file name. Returns None for a text-only chat (``--no-play`` without ``--save-dir``):
    nothing would be done with the audio, so the model is not loaded. Texts are kept in
    ``--save-dir`` when given; otherwise each one is written to a single scratch file,
    overwritten every time and removed with its directory when ``stack`` closes, whatever
    ends the session.

    One lock serializes all speaking: a timer alert comes from another thread and must neither
    overlap a reply nor overwrite the scratch file while the player reads it. When ``stack``
    closes, the utterance in progress is allowed to finish and later ones are dropped, so
    nothing is written into the scratch directory after it is removed.
    """
    if args.no_play and args.save_dir is None:
        print(
            "Audio is neither played nor saved (--no-play without --save-dir): chatting in "
            "text only. Add --save-dir DIR to keep WAV files.",
            file=sys.stderr,
        )
        return None

    from herald import audio_playback
    from herald.tts import engine

    eng = _load_engine(args, settings)
    folder = args.save_dir or Path(
        stack.enter_context(tempfile.TemporaryDirectory(prefix="herald-chat-"))
    )
    session = time.strftime("%Y%m%d_%H%M%S")
    lock = threading.Lock()
    closed = False
    warned_no_player = False

    def close() -> None:
        nonlocal closed
        with lock:  # waits for the utterance in progress
            closed = True

    stack.callback(close)

    def speak(text: str, kind: str, number: int) -> None:
        nonlocal warned_no_player
        with lock:
            if closed:
                return
            wav = eng.synth_long(text, pause_ms=args.pause_ms, max_chars=args.max_chars)
            # Saved texts are numbered; a scratch one reuses one name, so only one is on disk.
            name = f"{kind}_{session}_{number:03d}.wav" if args.save_dir else "speech.wav"
            path = engine.save_wav(folder / name, wav, eng.sample_rate)
            if not args.no_play and not audio_playback.play_wav(path) and not warned_no_player:
                warned_no_player = True
                print(
                    "No audio player available; "
                    + (
                        f"replies are saved in {args.save_dir}"
                        if args.save_dir
                        else "use --save-dir DIR to keep the replies as WAV files."
                    ),
                    file=sys.stderr,
                )

    return speak


def _make_announcer(speak):
    """The ``say`` function of the tools: a timer that fires prints and speaks its message.

    It runs on a timer thread while the main thread waits in ``input()``, and never raises.
    In a text-only chat (``speak`` is None) it only prints.
    """
    numbers = itertools.count(1)

    def announce(text: str) -> None:
        # One write, so the alert is not split by the main thread's own printing.
        sys.stdout.write(f"\a\n[Herald] {text}\n")
        sys.stdout.flush()
        if speak is None:
            return
        try:
            speak(text, "alert", next(numbers))
        except Exception as exc:  # a failed alert must not kill the timer thread
            logger.debug("Speech synthesis failed", exc_info=True)
            print(f"error: could not speak the reply: {exc}", file=sys.stderr)

    return announce


def _stop_timers(scheduler) -> None:
    cancelled = scheduler.shutdown()
    if cancelled:
        print(
            f"{cancelled} timer(s) were still running and have been cancelled because the "
            "chat ended.",
            file=sys.stderr,
        )


def _load_chat_tools(args: argparse.Namespace, announce, stack: contextlib.ExitStack):
    """Load the tools for the assistant: the registry, or None (``--no-tools``, none found).

    The timers are cancelled when ``stack`` closes. Load errors are warnings: a broken tool
    script must not stop the chat.
    """
    if args.no_tools:
        return None
    from herald.tools import ToolContext, load_tools
    from herald.tools.scheduler import Scheduler

    scheduler = Scheduler()
    stack.callback(_stop_timers, scheduler)
    report = load_tools(args.tools_dir, ToolContext(say=announce, scheduler=scheduler))
    for error in report.errors:
        print(f"warning: tools: {error}", file=sys.stderr)
    if report.tools:
        print("Tools: " + ", ".join(tool.name for tool in report.tools))
    return report.registry if len(report.registry) else None


def _cmd_chat(args: argparse.Namespace, settings: Settings) -> int:
    from herald.errors import OllamaError

    system_prompt, timeout = _chat_settings(args, settings)
    with contextlib.ExitStack() as stack:
        speak = _make_speaker(args, settings, stack)
        # Registered after the speaker, so the timers are cancelled before it closes.
        tools = _load_chat_tools(args, _make_announcer(speak), stack)
        assistant = _build_assistant(args, system_prompt, timeout, tools)

        print("Type a message; an empty line or Ctrl-D quits.")
        for turn in itertools.count(1):
            try:
                user_text = input("You: ").strip()
            except EOFError:
                break
            if not user_text:
                break

            try:
                reply = assistant.respond(user_text)
            except OllamaError as exc:
                print(f"error: {exc}", file=sys.stderr)
                continue
            print(f"Herald: {reply}")

            if speak is None:
                continue
            try:
                speak(reply, "chat", turn)
            except Exception as exc:  # one failed reply must not end the conversation
                logger.debug("Speech synthesis failed", exc_info=True)
                print(f"error: could not speak the reply: {exc}", file=sys.stderr)
    return 0


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


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = Settings.from_env()
    except HeraldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args = build_parser(settings).parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.func(args, settings)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except (HeraldError, OSError) as exc:
        if args.verbose:
            logger.exception("Command failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1
