"""``herald doctor``: check that this computer is ready to run Herald, and say what is missing.

Meant for the person who cannot get Herald to start and for whoever has to help from a distance:
the whole report is plain ASCII, so it can be pasted into a chat message as it is. Every check
prints one line, ``ok``, ``!!`` (a problem: something will not work) or ``--`` (information), and
every ``!!`` is followed by a line that says what to do about it::

    Python
      ok  Python 3.11.9 (/home/me/herald/.venv/bin/python)
      !!  coqui-tts is not installed
          -> run: pip install -e .

Nothing here loads a model or downloads anything. The only network access is a GET to the
Ollama server on this machine (``/api/tags``). A check that itself fails is reported as a
problem and never stops the others.
"""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass

from herald import __version__
from herald.config import Settings
from herald.errors import ProfileError

logger = logging.getLogger(__name__)

OK, PROBLEM, INFO = "ok", "!!", "--"
SUPPORTED_PYTHON = ((3, 11), (3, 12))  # what pyproject.toml allows
OLLAMA_TIMEOUT = 3.0  # seconds: a local server answers at once, or not at all


@dataclass(frozen=True)
class Line:
    """The result of a check: its status, what was found, and what to do about it."""

    status: str
    text: str
    fix: str | None = None


def _ok(text: str) -> Line:
    return Line(OK, text)


def _info(text: str, fix: str | None = None) -> Line:
    return Line(INFO, text, fix)


def _problem(text: str, fix: str) -> Line:
    return Line(PROBLEM, text, fix)


Check = Callable[[], list[Line]]


# --- System -----------------------------------------------------------------------------------


def _herald() -> list[Line]:
    return [_ok(f"Herald {__version__}")]


def _python() -> list[Line]:
    major, minor, micro = sys.version_info[:3]
    where = f"({sys.executable})"
    if (major, minor) in SUPPORTED_PYTHON:
        return [_ok(f"Python {major}.{minor}.{micro} {where}")]
    return [
        _problem(
            f"Python {major}.{minor} is not supported {where}",
            "install Python 3.11 (Windows: py -3.11 / winget install Python.Python.3.11)",
        )
    ]


def _platform() -> list[Line]:
    return [_info(f"Platform: {platform.platform()}")]


def _virtual_environment() -> list[Line]:
    if sys.prefix != sys.base_prefix:
        return [_ok(f"Virtual environment active ({sys.prefix})")]
    return [
        _info(
            "No virtual environment is active",
            "the `herald` command only exists inside the environment: activate it or call "
            ".venv/bin/herald (Windows: .venv\\Scripts\\herald)",
        )
    ]


# --- Libraries --------------------------------------------------------------------------------

_INSTALL = "run: pip install -e .  (inside the Herald folder, with the environment active)"


def _torch() -> list[Line]:
    try:
        import torch
    except (ImportError, OSError) as exc:  # on Windows a broken DLL is an OSError
        return [_problem(f"torch cannot be imported ({type(exc).__name__}: {exc})", _INSTALL)]
    accelerators = []
    if torch.cuda.is_available():
        accelerators.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        accelerators.append("mps")
    return [
        _ok(f"torch {torch.__version__} (accelerator: {', '.join(accelerators) or 'cpu only'})")
    ]


def _coqui_tts() -> list[Line]:
    # The version comes from the package metadata: importing TTS itself takes seconds.
    try:
        version = importlib.metadata.version("coqui-tts")
    except importlib.metadata.PackageNotFoundError:
        return [_problem("coqui-tts is not installed", _INSTALL)]
    return [_ok(f"coqui-tts {version}")]


# --- Voice files ------------------------------------------------------------------------------

_DOWNLOAD = "run: herald download-checkpoints"


def _base_voice(settings: Settings) -> list[Line]:
    folder = settings.checkpoint_dir
    lines = [_info(f"Base voice folder: {folder}")]
    for name in ("config.json", "vocab.json"):
        if (folder / name).is_file():
            lines.append(_ok(name))
        else:
            lines.append(_problem(f"{name} is missing", _DOWNLOAD))
    if (folder / "model.pth").is_file():
        lines.append(_ok("model.pth"))
    else:
        lines.append(
            _info(
                "model.pth not there yet: it is downloaded (about 2 GB) the first time the "
                "base voice is used"
            )
        )
    return lines


def _dataset(settings: Settings) -> list[Line]:
    folder = settings.dataset_dir
    if (folder / "metadata.csv").is_file():
        return [_info(f"Dataset: {folder}")]
    return [
        _info(
            f"Dataset: none at {folder}",
            "only needed to pick reference clips and to train: pass --reference-wav CLIP.wav "
            "or --dataset-dir DIR otherwise",
        )
    ]


def _profiles(settings: Settings, profile_name: str | None) -> list[Line]:
    from herald import profiles

    found = profiles.list_profiles(settings.profiles_dir, settings.project_root)
    lines = [_info(f"Profiles: {len(found)} in {settings.profiles_dir}")]
    if not profile_name:
        return lines
    try:
        profile = profiles.load_profile(profile_name, settings.profiles_dir, settings.project_root)
    except ProfileError as exc:
        return [*lines, _problem(f"profile {profile_name!r}: {exc}", "run: herald profiles")]
    problems = profiles.check_profile(profile)
    for problem in problems:
        lines.append(
            _problem(
                f"profile {profile.name}: {problem}",
                f"fix the path in {profile.path}, or give the option on the command line",
            )
        )
    if not problems:
        lines.append(_ok(f"profile {profile.name}: the files it points to are there"))
    return lines


# --- Chat -------------------------------------------------------------------------------------


def _ollama_model(settings: Settings, profile_name: str | None) -> str:
    """The model chat would use: the profile's if it names one, else the setting."""
    if profile_name:
        from herald import profiles

        try:
            profile = profiles.load_profile(
                profile_name, settings.profiles_dir, settings.project_root
            )
        except ProfileError:
            return settings.ollama_model  # _profiles reports it
        return profile.values.get("ollama_model", settings.ollama_model)
    return settings.ollama_model


def _ollama(settings: Settings, model: str) -> list[Line]:
    import requests

    base = settings.ollama_url.rstrip("/").removesuffix("/api/chat")
    try:
        response = requests.get(f"{base}/api/tags", timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        names = [entry["name"] for entry in response.json()["models"]]
    except requests.RequestException as exc:
        return [
            _problem(
                f"Cannot reach Ollama at {base} ({type(exc).__name__})",
                "start the Ollama app (or run: ollama serve)",
            )
        ]
    except (ValueError, KeyError, TypeError):
        return [
            _problem(
                f"{base} answered, but not like Ollama does",
                "check HERALD_OLLAMA_URL: it should be the address of the Ollama server",
            )
        ]
    lines = [_ok(f"Ollama at {base} ({len(names)} model(s) pulled)")]
    pulled = model in names or (":" not in model and f"{model}:latest" in names)
    if pulled:
        lines.append(_ok(f"model '{model}' is pulled"))
    else:
        lines.append(_problem(f"model '{model}' is not pulled", f"run: ollama pull {model}"))
    return lines


def _audio_player() -> list[Line]:
    if sys.platform == "win32":
        try:
            import winsound  # noqa: F401  (only to see that it is there)
        except ImportError as exc:
            return [_problem(f"winsound is not available ({exc})", _INSTALL)]
        return [_ok("Audio player: winsound (built into Windows)")]
    from herald.audio_playback import Player, find_player

    if Player().available:
        command = find_player()
        return [_ok(f"Audio player: {command[0] if command else 'available'}")]
    return [
        _problem(
            "No audio player found",
            "use --save-dir to keep the replies as files (or install ffmpeg, which has ffplay)",
        )
    ]


def _tools(settings: Settings) -> list[Line]:
    from herald.tools import ToolContext, load_tools
    from herald.tools.scheduler import Scheduler

    scheduler = Scheduler()
    try:
        report = load_tools(
            settings.tools_dir, ToolContext(say=lambda text: None, scheduler=scheduler)
        )
    finally:
        scheduler.shutdown()
    names = ", ".join(tool.name for tool in report.tools)
    lines = [_ok(f"Tools: {len(report.tools)} loaded ({names})")]
    for error in report.errors:
        lines.append(
            _problem(f"tool script: {error}", "fix or delete that script, see: herald tools")
        )
    return lines


# --- the report -------------------------------------------------------------------------------


def _sections(
    settings: Settings, profile_name: str | None
) -> list[tuple[str, list[tuple[str, Check]]]]:
    return [
        (
            "System",
            [
                ("Herald", _herald),
                ("Python", _python),
                ("Platform", _platform),
                ("Virtual environment", _virtual_environment),
            ],
        ),
        ("Libraries", [("torch", _torch), ("coqui-tts", _coqui_tts)]),
        (
            "Voice files",
            [
                ("Base voice", lambda: _base_voice(settings)),
                ("Dataset", lambda: _dataset(settings)),
                ("Profiles", lambda: _profiles(settings, profile_name)),
            ],
        ),
        (
            "Chat",
            [
                ("Ollama", lambda: _ollama(settings, _ollama_model(settings, profile_name))),
                ("Audio player", _audio_player),
                ("Tools", lambda: _tools(settings)),
            ],
        ),
    ]


def _run(name: str, check: Check) -> list[Line]:
    """Run one check. If it raises, that is a finding too, not the end of the report."""
    try:
        return check()
    except Exception as exc:
        logger.debug("Check %s failed", name, exc_info=True)
        return [
            _problem(f"{name}: {str(exc) or type(exc).__name__}", "this is a bug in herald doctor")
        ]


def run_doctor(
    settings: Settings,
    profile: str | None = None,
    *,
    out: Callable[[str], None] = print,
) -> int:
    """Print the report. Returns the exit code: 0 if nothing is wrong, 1 otherwise."""
    problems = 0
    for title, checks in _sections(settings, profile):
        out(title)
        for name, check in checks:
            for line in _run(name, check):
                out(f"  {line.status}  {line.text}")
                if line.fix:
                    out(f"      -> {line.fix}")
                if line.status == PROBLEM:
                    problems += 1
        out("")
    out("Ready." if not problems else f"{problems} problem(s) found.")
    return 1 if problems else 0
