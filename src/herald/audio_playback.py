"""Best-effort audio playback. Never required: callers just get ``False`` if it fails."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ON_WINDOWS = sys.platform == "win32"

# First one found on PATH wins. `afplay` ships with macOS; the rest cover common Linux setups.
# Windows needs no external player: play_wav uses the standard library's `winsound`.
_PLAYERS: tuple[tuple[str, ...], ...] = (
    ("afplay",),
    ("paplay",),
    ("aplay", "-q"),
    ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"),
)


def find_player() -> list[str] | None:
    """Return the command (without the file argument) of an available player, if any."""
    for command in _PLAYERS:
        if shutil.which(command[0]):
            return list(command)
    return None


def _play_with_winsound(path: Path) -> bool:
    """Play ``path`` with the standard library's ``winsound`` (Windows only, blocking)."""
    try:
        import winsound

        # NODEFAULT: fail on a missing file instead of playing the system beep.
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
    except (ImportError, RuntimeError) as exc:
        logger.debug("Playback with winsound failed: %s", exc)
        return False
    return True


def play_wav(path: Path) -> bool:
    """Play ``path`` and wait for it to finish. Returns whether playback succeeded."""
    if _ON_WINDOWS:
        return _play_with_winsound(path)
    command = find_player()
    if command is None:
        logger.debug("No audio player found; skipping playback")
        return False
    try:
        subprocess.run(
            [*command, str(path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Playback with %s failed: %s", command[0], exc)
        return False
    return True
