"""Best-effort audio playback. Never required: callers just get ``False`` if it fails.

:class:`Player` plays one WAV file at a time and can be cut short from another thread, which a
voice assistant needs when the user speaks again while Herald is still talking. ``play_wav`` and
``find_player`` are the simple, stateless forms.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_ON_WINDOWS = sys.platform == "win32"

# First one found on PATH wins. `afplay` ships with macOS; the rest cover common Linux setups.
# Windows needs no external player: Player uses the standard library's `winsound`.
_PLAYERS: tuple[tuple[str, ...], ...] = (
    ("afplay",),
    ("paplay",),
    ("aplay", "-q"),
    ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"),
)

# After terminate(), how long a player may take to exit before it is killed.
_TERMINATE_WAIT = 0.5


def find_player() -> list[str] | None:
    """Return the command (without the file argument) of an available player, if any."""
    for command in _PLAYERS:
        if shutil.which(command[0]):
            return list(command)
    return None


class Player:
    """Plays WAV files with whatever the system offers, and can be stopped at any moment.

    ``play`` blocks until the file has been played. ``stop`` (callable from any thread, safe to
    repeat, harmless when nothing is playing) ends the current playback at once, and then ``play``
    returns ``False``. That is not a failure: ``was_stopped`` tells the two cases apart, so a
    caller that only reacts to ``play(...) is False`` must also check ``was_stopped``.
    """

    def __init__(self) -> None:
        self._command = None if _ON_WINDOWS else find_player()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._winsound_playing = False
        self._stopped = False

    @property
    def available(self) -> bool:
        """Whether there is anything to play with (always true on Windows)."""
        return _ON_WINDOWS or self._command is not None

    @property
    def was_stopped(self) -> bool:
        """Whether the last ``play`` call was cut short by ``stop``."""
        return self._stopped

    def play(self, path: Path) -> bool:
        """Play ``path`` and wait for it to finish. Returns whether it was played to the end."""
        if _ON_WINDOWS:
            return self._play_with_winsound(path)
        if self._command is None:
            logger.debug("No audio player found; skipping playback")
            return False
        with self._lock:
            self._stopped = False
            try:
                process = subprocess.Popen(
                    [*self._command, str(path)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                logger.debug("Playback with %s failed: %s", self._command[0], exc)
                return False
            self._process = process
        try:
            returncode = process.wait()
        finally:
            with self._lock:
                self._process = None
        if self._stopped:
            return False
        if returncode != 0:
            logger.debug("Playback with %s failed: exit status %s", self._command[0], returncode)
            return False
        return True

    def stop(self) -> None:
        """Stop the playback in progress, if any. Safe from any thread and idempotent."""
        with self._lock:
            process = self._process
            winsound_playing = self._winsound_playing
            if process is not None or winsound_playing:
                self._stopped = True
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=_TERMINATE_WAIT)
            except subprocess.TimeoutExpired:
                process.kill()
        if winsound_playing:
            self._purge_winsound()

    def _play_with_winsound(self, path: Path) -> bool:
        """Blocking playback with the standard library's ``winsound`` (Windows only)."""
        try:
            import winsound

            with self._lock:
                self._stopped = False
                self._winsound_playing = True
            try:
                # NODEFAULT: fail on a missing file instead of playing the system beep.
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            finally:
                with self._lock:
                    self._winsound_playing = False
        except (ImportError, RuntimeError) as exc:
            logger.debug("Playback with winsound failed: %s", exc)
            return False
        return not self._stopped

    @staticmethod
    def _purge_winsound() -> None:
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except (ImportError, RuntimeError) as exc:
            logger.debug("Stopping winsound failed: %s", exc)


def play_wav(path: Path) -> bool:
    """Play ``path`` and wait for it to finish. Returns whether playback succeeded."""
    return Player().play(path)
