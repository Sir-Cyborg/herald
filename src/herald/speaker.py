"""Speak text without making the listener wait: a threaded synthesize-and-play pipeline.

XTTS needs about as long to synthesize a sentence as the sentence lasts, so speaking a whole
reply at once means a long silence first. :class:`Speaker` instead takes the reply as it is
written (``feed``), cuts it into sentences (:class:`~herald.tts.engine.SentenceBuffer`),
synthesizes sentence k+1 on one thread while sentence k plays on another, and can be cut off at
any moment (``cancel``) when the user speaks again::

    with Speaker(engine, player=Player(), scratch_dir=tmp) as speaker:
        for delta in llm_stream:          # text as it arrives
            speaker.feed(delta)
        speaker.end_utterance()           # nothing more is coming for this reply
        speaker.wait()                    # until it has been spoken
    # an alert from any thread:  speaker.say("Your timer is up.")
    # the user types a new message while Herald is talking:  speaker.cancel()

Threads: ``feed``, ``say``, ``end_utterance``, ``cancel``, ``wait`` and ``close`` may be called
from any thread. The callbacks ``on_error`` and ``on_no_player`` are called from the speaker's
own worker threads, so they must be thread-safe and quick, and must not call ``wait``, ``close``
or ``cancel`` (``cancel`` would then not wait for the playback to stop).
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from herald.audio_playback import Player
from herald.tts.engine import (
    DEFAULT_PAUSE_MS,
    SentenceBuffer,
    default_max_chars,
    make_silence,
    save_wav,
)

logger = logging.getLogger(__name__)

# How long close() waits for each worker thread. A synthesis that is already running cannot be
# interrupted, so a worker can outlive close(); the threads are daemons and die with the program.
_JOIN_TIMEOUT = 2.0
# How long cancel() keeps trying to stop a playback that is just starting.
_STOP_TIMEOUT = 2.0


class _Engine(Protocol):
    """What the speaker needs from a TTS engine (``XttsEngine`` provides it)."""

    language: str
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray: ...


@dataclass
class _Utterance:
    """One reply or alert: its pieces are synthesized and played in order."""

    number: int
    arrays: list[np.ndarray] = field(default_factory=list)  # kept only to save the whole thing
    failed: bool = False  # a piece is missing, so the utterance must not be saved
    reported: bool = False  # an error was already reported for it


@dataclass
class _Piece:
    generation: int
    utterance: _Utterance
    text: str


@dataclass
class _End:
    """Marks the end of an utterance in the synthesis queue."""

    generation: int
    utterance: _Utterance
    name: str | None


@dataclass
class _Audio:
    generation: int
    utterance: _Utterance
    samples: np.ndarray


_STOP = object()  # tells a worker thread to exit


def _log_error(message: str) -> None:
    logger.error(message)


class Speaker:
    """Turns text into speech in the background. See the module docstring.

    ``engine`` needs ``synthesize(text)``, ``sample_rate`` and ``language``. ``player`` plays the
    audio; with ``None``, or a player that is not ``available``, nothing is played. Audio is only
    synthesized if something uses it: a usable player or ``save_dir``. With ``save_dir``, each
    utterance whose pieces were all synthesized is also written there as one WAV file, the pieces
    separated by ``pause_ms`` of silence. ``scratch_dir`` holds the short-lived file of the piece
    being played, deleted right after. ``max_chars`` is the longest piece (``None``: the limit
    of the engine's language).

    Failures never stop the pipeline. A piece that cannot be synthesized or played is reported
    once per utterance through ``on_error(message)`` and the next piece goes on. ``on_no_player()``
    is called at most once, the first time audio could not be played because the ``player`` is
    not available or its playback failed (a playback that is cut short by ``cancel`` is not a
    failure). It is not called when ``player`` is ``None``: that is a choice, not a problem.
    """

    def __init__(
        self,
        engine: _Engine,
        *,
        player: Player | None,
        scratch_dir: Path,
        save_dir: Path | None = None,
        pause_ms: float = DEFAULT_PAUSE_MS,
        max_chars: int | None = None,
        on_error: Callable[[str], None] = _log_error,
        on_no_player: Callable[[], None] | None = None,
    ) -> None:
        if pause_ms < 0:
            raise ValueError("pause_ms must not be negative")
        self._engine = engine
        self._player = player
        self._playable = player is not None and player.available
        self._scratch_dir = Path(scratch_dir)
        self._save_dir = Path(save_dir) if save_dir is not None else None
        self._pause_ms = pause_ms
        self._max_chars = max_chars if max_chars is not None else default_max_chars(engine.language)
        self._on_error = on_error
        self._on_no_player = on_no_player
        self._uses_audio = self._playable or self._save_dir is not None
        if self._playable:
            self._scratch_dir.mkdir(parents=True, exist_ok=True)

        # One condition (and its lock) guards all the state below.
        self._cond = threading.Condition()
        self._generation = 0  # bumped by cancel(): older work is dropped
        self._outstanding = 0  # pieces not yet played or dropped, plus unprocessed end marks
        self._closed = False
        self._playing = False  # the playback worker is handling a piece
        self._warned_no_player = False
        self._buffer = SentenceBuffer(max_chars=self._max_chars)
        self._current: _Utterance | None = None  # the utterance being streamed in by feed()
        self._utterance_numbers = itertools.count(1)
        self._file_numbers = itertools.count(1)

        self._synth_queue: queue.Queue = queue.Queue()
        self._play_queue: queue.Queue = queue.Queue()
        self._threads = [
            threading.Thread(target=self._synth_loop, name="herald-speaker-synth", daemon=True),
            threading.Thread(target=self._play_loop, name="herald-speaker-play", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def __enter__(self) -> Speaker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- public API ---------------------------------------------------------------------------

    def feed(self, text_delta: str) -> None:
        """Add streamed text. Pieces that are complete sentences are queued at once. Never blocks.

        Several ``feed`` calls up to the next ``end_utterance`` form one utterance.
        """
        with self._cond:
            if self._closed or not text_delta:
                return
            if self._current is None:
                self._current = self._new_utterance()
            self._queue_pieces(self._current, self._buffer.feed(text_delta))

    def end_utterance(self, name: str | None = None) -> None:
        """Say that the streamed text is complete: queue what is left and close the utterance.

        With ``save_dir``, the whole utterance is saved as ``<name>.wav`` (default
        ``utterance_<n>.wav``) once its last piece is synthesized. Does nothing if no text was
        fed since the last call.
        """
        with self._cond:
            if self._closed or self._current is None:
                return
            utterance, self._current = self._current, None
            self._queue_pieces(utterance, self._buffer.flush())
            self._queue(self._synth_queue, _End(self._generation, utterance, name))

    def say(self, text: str, name: str | None = None) -> None:
        """Speak ``text`` as one complete utterance: for alerts and replies that are not streamed.

        It has its own text buffer, so it is safe to call (say, from a timer thread) while
        another utterance is being streamed in with ``feed``: the two are not mixed, although the
        alert's sentences may be spoken between two sentences of the reply.
        """
        buffer = SentenceBuffer(max_chars=self._max_chars)
        pieces = buffer.feed(text) + buffer.flush()
        with self._cond:
            if self._closed or not pieces:
                return
            utterance = self._new_utterance()
            self._queue_pieces(utterance, pieces)
            self._queue(self._synth_queue, _End(self._generation, utterance, name))

    def cancel(self) -> None:
        """Drop everything queued, stop the playback now and forget the unfinished utterance.

        Nothing is saved for the dropped utterances. The speaker stays usable. A synthesis that
        is already running cannot be interrupted: its result is discarded, but the next
        utterance's first piece has to wait for it to finish.
        """
        with self._cond:
            self._generation += 1
            self._outstanding = 0
            _drain(self._synth_queue)
            _drain(self._play_queue)
            self._buffer.flush()
            self._current = None
            playing = self._playing
            self._cond.notify_all()
        if playing and self._player is not None:
            self._stop_playback()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every queued piece has been synthesized and played; False on timeout.

        Text that has been fed but not ended with ``end_utterance`` is not waited for.
        """
        with self._cond:
            return self._cond.wait_for(lambda: self._outstanding == 0, timeout)

    def close(self) -> None:
        """Cancel everything and stop the worker threads. Safe to call more than once."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
        self.cancel()
        self._synth_queue.put(_STOP)
        self._play_queue.put(_STOP)
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(_JOIN_TIMEOUT)

    # --- queueing (the caller holds self._cond) -----------------------------------------------

    def _new_utterance(self) -> _Utterance:
        return _Utterance(next(self._utterance_numbers))

    def _queue(self, target: queue.Queue, item: _Piece | _End) -> None:
        self._outstanding += 1
        target.put(item)

    def _queue_pieces(self, utterance: _Utterance, pieces: list[str]) -> None:
        for text in pieces:
            self._queue(self._synth_queue, _Piece(self._generation, utterance, text))

    # --- worker threads -----------------------------------------------------------------------

    def _synth_loop(self) -> None:
        while (item := self._synth_queue.get()) is not _STOP:
            try:
                self._synthesize(item)
            except Exception:  # a bug here must not leave the pipeline without a worker
                logger.exception("Speaker synthesis thread failed on %r", item)
                self._complete(item.generation)

    def _synthesize(self, item: _Piece | _End) -> None:
        if not self._is_current(item.generation):
            return
        if isinstance(item, _End):
            self._save(item)
            self._complete(item.generation)
            return
        if self._player is not None and not self._playable:
            self._warn_no_player()
        if not self._uses_audio:
            self._complete(item.generation)
            return
        try:
            samples = self._engine.synthesize(item.text)
        except Exception as exc:
            item.utterance.failed = True
            self._report(item, exc, "speak")
            self._complete(item.generation)
            return
        if not self._is_current(item.generation):
            return  # cancelled while it was being synthesized
        if self._save_dir is not None:
            item.utterance.arrays.append(samples)
        if self._playable:
            self._play_queue.put(_Audio(item.generation, item.utterance, samples))
        else:
            self._complete(item.generation)

    def _save(self, end: _End) -> None:
        utterance = end.utterance
        arrays, utterance.arrays = utterance.arrays, []
        if self._save_dir is None or utterance.failed or not arrays:
            return
        try:
            parts = [arrays[0]]
            for samples in arrays[1:]:
                parts.append(make_silence(self._pause_ms, self._engine.sample_rate))
                parts.append(samples)
            name = Path(end.name or f"utterance_{utterance.number}").name
            if not name.lower().endswith(".wav"):
                name += ".wav"
            save_wav(self._save_dir / name, np.concatenate(parts), self._engine.sample_rate)
        except Exception as exc:
            self._report(end, exc, "save")

    def _play_loop(self) -> None:
        while (item := self._play_queue.get()) is not _STOP:
            with self._cond:
                if item.generation != self._generation:
                    continue
                self._playing = True
            try:
                self._play(item)
            except Exception as exc:
                self._report(item, exc, "speak")
            finally:
                with self._cond:
                    self._playing = False
                    self._complete_locked(item.generation)

    def _play(self, item: _Audio) -> None:
        assert self._player is not None
        path = self._scratch_dir / f"piece_{next(self._file_numbers):05d}.wav"
        try:
            save_wav(path, item.samples, self._engine.sample_rate)
            if not self._is_current(item.generation):
                return  # cancelled while the file was being written
            played = self._player.play(path)
        finally:
            path.unlink(missing_ok=True)
        if not played and not self._player.was_stopped:
            self._warn_no_player()

    # --- helpers ------------------------------------------------------------------------------

    def _is_current(self, generation: int) -> bool:
        with self._cond:
            return generation == self._generation

    def _complete(self, generation: int) -> None:
        with self._cond:
            self._complete_locked(generation)

    def _complete_locked(self, generation: int) -> None:
        if generation == self._generation:
            self._outstanding -= 1
        self._cond.notify_all()

    def _stop_playback(self) -> None:
        """Stop the player, repeating until the playback worker is really done with the piece.

        One call is not enough when the worker has just decided to play and has not started the
        player yet: that playback would then run to its end.
        """
        assert self._player is not None
        if threading.current_thread() in self._threads:  # called from a callback: cannot wait
            self._player.stop()
            return
        deadline = time.monotonic() + _STOP_TIMEOUT
        while True:
            self._player.stop()
            with self._cond:
                remaining = deadline - time.monotonic()
                if not self._playing or remaining <= 0:
                    return
                self._cond.wait(min(0.02, remaining))

    def _report(self, item: _Piece | _End | _Audio, exc: Exception, what: str) -> None:
        """Tell ``on_error`` about a failure, once per utterance and never for dropped work."""
        with self._cond:
            if item.generation != self._generation or item.utterance.reported:
                return
            item.utterance.reported = True
        logger.debug("Speaker failed to %s", what, exc_info=exc)
        self._call(self._on_error, f"could not {what} the reply: {exc}")

    def _warn_no_player(self) -> None:
        with self._cond:
            if self._warned_no_player:
                return
            self._warned_no_player = True
        if self._on_no_player is not None:
            self._call(self._on_no_player)

    @staticmethod
    def _call(callback: Callable[..., None], *args: str) -> None:
        try:
            callback(*args)
        except Exception:  # a broken callback must not kill a worker thread
            logger.exception("Speaker callback %r failed", callback)


def _drain(target: queue.Queue) -> None:
    """Remove everything from ``target``, but leave the stop mark of a closing speaker."""
    stop = False
    try:
        while True:
            stop = target.get_nowait() is _STOP or stop
    except queue.Empty:
        pass
    if stop:
        target.put(_STOP)
