"""``herald chat``: talk to a local Ollama model and hear the replies in the cloned voice.

Why a reply starts to be heard quickly: the text is not waited for as a whole. Ollama streams it,
the :class:`~herald.speaker.Speaker` cuts it into sentences, synthesizes sentence k+1 while
sentence k plays, and the next user message cuts the speech off (barge-in). At start-up the
Ollama model and the voice model are warmed up in the background, so the first message does not
pay for loading them.

This module only wires the pieces together; the conversation itself is ``herald.assistant`` and
the speech is ``herald.speaker``. A different front end (a microphone, a device with a speaker)
would replace :func:`_converse` and keep the rest.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import logging
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from herald.config import Settings, read_utf8
from herald.voice import load_engine

logger = logging.getLogger(__name__)

TEXT_ONLY_NOTICE = (
    "Audio is neither played nor saved (--no-play without --save-dir): chatting in text only. "
    "Add --save-dir DIR to keep WAV files."
)


def _write(stream, text: str) -> None:
    """One ``write`` so that output from several threads is not split in the middle of a line."""
    stream.write(text)
    stream.flush()


def _in_background(func: Callable[[], object]) -> None:
    """Run ``func`` on a daemon thread: a warm-up must never keep the program from exiting."""
    threading.Thread(target=func, daemon=True, name="herald-warm-up").start()


def _chat_settings(args: argparse.Namespace, settings: Settings) -> tuple[str, float]:
    """The system prompt and the Ollama timeout. Read first, so that a bad file or value fails
    before any model is loaded."""
    system_prompt = (
        read_utf8(args.system_prompt_file).strip()
        if args.system_prompt_file
        else args.system_prompt
    )
    timeout = args.ollama_timeout if args.ollama_timeout is not None else settings.ollama_timeout
    return system_prompt, timeout


# --- the voice --------------------------------------------------------------------------------


def _start_speaker(args: argparse.Namespace, settings: Settings, stack: contextlib.ExitStack):
    """Load the voice and return the Speaker, or None for a text-only chat.

    A chat is text only with ``--no-play`` and no ``--save-dir``: nothing would be done with the
    audio, so the model is not even loaded. Otherwise the speaker plays the replies (unless
    ``--no-play``) and, with ``--save-dir``, keeps each one there as a WAV file. The pieces it
    plays live in a scratch directory that goes away, with the speaker, when ``stack`` closes.
    """
    if args.no_play and args.save_dir is None:
        _write(sys.stderr, TEXT_ONLY_NOTICE + "\n")
        return None

    from herald.audio_playback import Player
    from herald.speaker import Speaker

    engine = load_engine(args, settings)
    _in_background(engine.warm_up)  # overlaps with the user's typing and the first LLM call

    scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="herald-chat-")))
    speaker = Speaker(
        engine,
        player=None if args.no_play else Player(),
        scratch_dir=scratch,
        save_dir=args.save_dir,
        pause_ms=args.pause_ms,
        max_chars=args.max_chars,
        on_error=lambda message: _write(sys.stderr, f"error: {message}\n"),
        on_no_player=lambda: _write(sys.stderr, _no_player_notice(args.save_dir) + "\n"),
    )
    stack.callback(speaker.close)  # registered after the scratch directory: closed before it
    return speaker


def _no_player_notice(save_dir: Path | None) -> str:
    if save_dir is not None:
        return f"No audio player available; replies are saved in {save_dir}"
    return "No audio player available; use --save-dir DIR to keep the replies as WAV files."


def _make_announcer(speaker, session: str) -> Callable[[str], None]:
    """The ``say`` function of the tools: a timer that fires prints and speaks its message.

    It runs on a timer thread while the main thread waits in ``input()``, and never raises. In a
    text-only chat (no speaker) it only prints. The speaker serializes it with the replies.
    """
    numbers = itertools.count(1)

    def announce(text: str) -> None:
        _write(sys.stdout, f"\a\n[Herald] {text}\n")
        if speaker is None:
            return
        try:
            speaker.say(text, name=f"alert_{session}_{next(numbers):03d}")
        except Exception as exc:  # a failed alert must not kill the timer thread
            logger.debug("Speech failed", exc_info=True)
            _write(sys.stderr, f"error: could not speak the reply: {exc}\n")

    return announce


# --- the tools --------------------------------------------------------------------------------


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


# --- the conversation -------------------------------------------------------------------------


class _Reply:
    """A reply on its way out: printed as it arrives and handed to the speaker piece by piece."""

    def __init__(self, speaker) -> None:
        self._speaker = speaker
        self._started = False

    def write(self, delta: str) -> None:
        if not delta:
            return
        if not self._started:
            self._started = True
            sys.stdout.write("Herald: ")
        sys.stdout.write(delta)
        sys.stdout.flush()
        if self._speaker is not None:
            self._speaker.feed(delta)

    def finish(self) -> None:
        """End the printed line, if anything was printed."""
        if self._started:
            _write(sys.stdout, "\n")


def _converse(assistant, speaker, session: str) -> None:
    """Read messages until the user quits; print and speak the replies as they stream in.

    The speech is not waited for: the loop goes back to ``input()`` while it keeps playing, and
    whatever the user types next cuts it off.
    """
    from herald.errors import OllamaError

    for turn in itertools.count(1):
        try:
            user_text = input("You: ").strip()
        except EOFError:
            return
        if not user_text:
            return

        if speaker is not None:
            speaker.cancel()  # barge-in: the user spoke, so stop talking
        reply = _Reply(speaker)
        try:
            assistant.respond(user_text, on_text=reply.write)
        except OllamaError as exc:
            if speaker is not None:
                speaker.cancel()  # what was said so far is only part of a reply
            reply.finish()
            print(f"error: {exc}", file=sys.stderr)
            continue
        reply.finish()
        if speaker is not None:
            speaker.end_utterance(f"chat_{session}_{turn:03d}")


def run_chat(args: argparse.Namespace, settings: Settings) -> int:
    from herald.assistant import Assistant
    from herald.llm.ollama_client import OllamaClient

    system_prompt, timeout = _chat_settings(args, settings)
    client = OllamaClient(args.ollama_url, args.ollama_model, timeout=timeout)
    _in_background(client.warm_up)  # Ollama loads its model while the voice model loads

    session = time.strftime("%Y%m%d_%H%M%S")
    with contextlib.ExitStack() as stack:
        speaker = _start_speaker(args, settings, stack)
        # Registered after the speaker, so the timers are cancelled before it closes.
        tools = _load_chat_tools(args, _make_announcer(speaker, session), stack)
        assistant = Assistant(
            client,
            system_prompt,
            tools=tools,
            history_turns=args.history,
            use_triggers=not args.always_offer_tools,
        )

        print("Type a message; an empty line or Ctrl-D quits.")
        try:
            _converse(assistant, speaker, session)
            if speaker is not None:
                speaker.wait()  # let the last reply finish
        except BaseException:  # Ctrl-C included: leave quickly, do not wait for the speech
            if speaker is not None:
                speaker.cancel()
            raise
    return 0
