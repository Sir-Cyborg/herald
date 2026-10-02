"""Voice profiles: one small TOML file that bundles a voice and its character.

``herald chat --profile frieren`` reads ``profiles/frieren.toml`` and uses it in place of
a handful of options: the fine-tuned voice, the reference clips, the language, the system
prompt, the Ollama model... Command line options still win over the profile, which wins
over the environment and the built-in defaults.

A profile file looks like this; every key is optional::

    description = "Frieren, the elf mage"            # free text, shown by `herald profiles`
    checkpoint = "models/frieren"                    # fine-tuned voice (file or directory);
                                                     # omit it for the base voice
    dataset = "dataset/frieren"                      # where reference clips are picked from
    reference_wavs = ["clips/a.wav", "clips/b.wav"]  # explicit reference clips (instead)
    num_references = 3                               # how many clips to pick from the dataset
    language = "en"
    temperature = 0.7
    device = "auto"
    system_prompt = "You are Frieren ..."            # the character, or:
    system_prompt_file = "profiles/frieren.md"       # a file that holds the character
    ollama_model = "llama3.2:3b"
    history = 10                                     # past exchanges sent to the model
    speaker_name = "frieren"                         # used by `herald train`

Rules:

* Relative paths are resolved against the project root (not the current directory and not
  the profile's folder); ``~`` is expanded.
* ``system_prompt`` and ``system_prompt_file`` are mutually exclusive; the file is read
  (UTF-8) when the profile is loaded.
* Unknown keys, values of the wrong type and invalid TOML are errors (:class:`ProfileError`)
  that name the profile file, so that a typo is never silently ignored.

Only the standard library is used (``tomllib`` for reading, a tiny writer for the profiles
herald creates itself).
"""

from __future__ import annotations

import difflib
import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from herald.dataset.metadata import METADATA_FILE
from herald.errors import ProfileError

# Also the order in which write_profile writes them.
PROFILE_KEYS: tuple[str, ...] = (
    "description",
    "checkpoint",
    "dataset",
    "reference_wavs",
    "num_references",
    "language",
    "temperature",
    "device",
    "system_prompt",
    "system_prompt_file",
    "ollama_model",
    "history",
    "speaker_name",
)

# TOML key -> the command line option (argparse ``dest``) it is the default of. ``description``
# and ``system_prompt_file`` are not options: the latter's text is the ``system_prompt``.
_CLI_DESTS = {
    "checkpoint": "checkpoint",
    "dataset": "dataset_dir",
    "reference_wavs": "reference_wav",
    "num_references": "num_references",
    "language": "language",
    "temperature": "temperature",
    "device": "device",
    "system_prompt": "system_prompt",
    "ollama_model": "ollama_model",
    "history": "history",
    "speaker_name": "speaker_name",
}

# What each key must be (checked by _check_value).
_PATH_KEYS = ("checkpoint", "dataset", "system_prompt_file")
_WORD_KEYS = ("language", "device", "ollama_model", "speaker_name", "system_prompt")

_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
_TYPE_NAMES = {
    str: "a string",
    bool: "a boolean",
    int: "an integer",
    float: "a number",
    list: "a list",
    dict: "a table",
}


@dataclass(frozen=True)
class Profile:
    name: str
    path: Path
    # The validated settings: paths are absolute ``Path`` objects (``reference_wavs`` is a
    # list of them) and the text of ``system_prompt_file`` is in ``system_prompt``.
    values: Mapping[str, Any]
    description: str


@dataclass(frozen=True)
class ProfileSummary:
    name: str
    path: Path
    description: str
    language: str | None
    checkpoint: str | None  # relative to the project root when inside it
    problems: tuple[str, ...]  # why the profile cannot be used; empty if it is fine


# --- reading ----------------------------------------------------------------------------


def _type_name(value: Any) -> str:
    return _TYPE_NAMES.get(type(value), type(value).__name__)


def _check_value(path: Path, key: str, value: Any) -> Any:
    """``value`` if it is valid for ``key`` (an int temperature becomes a float)."""

    def bad(expected: str) -> ProfileError:
        return ProfileError(f"{path}: {key} must be {expected}, got {_type_name(value)}")

    if key == "description":
        if not isinstance(value, str):
            raise bad("a string")
    elif key in _PATH_KEYS or key in _WORD_KEYS:
        if not isinstance(value, str):
            raise bad("a string")
        if not value.strip():
            raise ProfileError(f"{path}: {key} must not be empty")
    elif key == "reference_wavs":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise bad("a list of strings")
        if not value or not all(v.strip() for v in value):
            raise ProfileError(f"{path}: {key} must be a list of file names, none of them empty")
    elif key in ("num_references", "history"):
        minimum = 1 if key == "num_references" else 0
        if isinstance(value, bool) or not isinstance(value, int):
            raise bad("an integer")
        if value < minimum:
            raise ProfileError(f"{path}: {key} must be at least {minimum}, got {value}")
    elif key == "temperature":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise bad("a number")
        if not math.isfinite(value) or value <= 0:
            raise ProfileError(f"{path}: {key} must be a positive number, got {value}")
        value = float(value)
    return value


def _check_values(path: Path, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the keys and values of a profile; the result is in PROFILE_KEYS order."""
    unknown = [key for key in raw if key not in PROFILE_KEYS]
    if unknown:
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(str(key), PROFILE_KEYS, n=1)
            hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        plural = "keys" if len(unknown) > 1 else "key"
        raise ProfileError(
            f"{path}: unknown {plural} {', '.join(hints)}; valid keys: {', '.join(PROFILE_KEYS)}"
        )
    if "system_prompt" in raw and "system_prompt_file" in raw:
        raise ProfileError(f"{path}: use either system_prompt or system_prompt_file, not both")
    return {key: _check_value(path, key, raw[key]) for key in PROFILE_KEYS if key in raw}


def _resolve_path(path: Path, key: str, value: str, project_root: Path) -> Path:
    try:
        resolved = Path(value).expanduser()
    except RuntimeError as exc:  # "~user" of an unknown user, no home directory...
        raise ProfileError(f"{path}: {key} {value!r}: {exc}") from exc
    return resolved if resolved.is_absolute() else Path(project_root) / resolved


def _read_prompt_file(path: Path, file: Path) -> str:
    try:
        text = file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise ProfileError(f"{path}: system_prompt_file not found: {file}") from None
    except UnicodeDecodeError:
        raise ProfileError(f"{path}: {file} is not valid UTF-8 text") from None
    except OSError as exc:
        raise ProfileError(f"{path}: cannot read system_prompt_file {file}: {exc}") from exc
    if not text:
        raise ProfileError(f"{path}: system_prompt_file {file} is empty")
    return text


def _profile_files(profiles_dir: Path) -> list[Path]:
    """The profile files of a directory (not the ``_template.toml`` kind), sorted by name."""
    profiles_dir = Path(profiles_dir)
    if not profiles_dir.is_dir():
        return []
    return sorted(
        (
            p
            for p in profiles_dir.glob("*.toml")
            if p.is_file() and not p.name.startswith(("_", "."))
        ),
        key=lambda p: p.name,
    )


def _available(profiles_dir: Path) -> str:
    names = [p.stem for p in _profile_files(profiles_dir)]
    return f"Available profiles: {', '.join(names)}" if names else f"No profiles in {profiles_dir}"


def load_profile(name_or_path: str, profiles_dir: Path, project_root: Path) -> Profile:
    """Read and validate a profile.

    A bare name (letters, digits, ``-`` and ``_``) is looked up as
    ``<profiles_dir>/<name>.toml``. A value that contains a path separator or ends in
    ``.toml`` is the path of a profile file instead (relative to the current directory, as
    typed; ``.toml`` may be left out). Paths *inside* the profile are resolved against
    ``project_root``. The system prompt, inline or from a file, has its outer blank space
    removed.
    """
    value = str(name_or_path)
    if "/" in value or "\\" in value or value.endswith(".toml"):
        try:
            path = Path(value).expanduser()
        except RuntimeError as exc:
            raise ProfileError(f"Invalid profile path {value!r}: {exc}") from exc
        if not path.exists() and not path.suffix:
            path = path.with_suffix(".toml")  # `--profile profiles/frieren`
        name = path.stem
    else:
        if not _NAME_RE.fullmatch(value):
            raise ProfileError(
                f"Invalid profile name {value!r}: use letters, digits, '-' and '_'. "
                f"{_available(profiles_dir)}"
            )
        name, path = value, Path(profiles_dir) / f"{value}.toml"

    if not path.is_file():
        raise ProfileError(
            f"Profile {value!r} not found: no file {path}. {_available(profiles_dir)}"
        )
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"{path}: invalid TOML: {exc}") from exc
    except UnicodeDecodeError:
        raise ProfileError(f"{path} is not valid UTF-8 text; save it as UTF-8") from None
    except OSError as exc:
        raise ProfileError(f"{path}: cannot read the profile: {exc}") from exc

    values = _check_values(path, raw)
    for key in ("checkpoint", "dataset"):
        if key in values:
            values[key] = _resolve_path(path, key, values[key], project_root)
    if "reference_wavs" in values:
        values["reference_wavs"] = [
            _resolve_path(path, "reference_wavs", v, project_root) for v in values["reference_wavs"]
        ]
    if "system_prompt" in values:
        values["system_prompt"] = values["system_prompt"].strip()
    if "system_prompt_file" in values:
        file = _resolve_path(
            path, "system_prompt_file", values.pop("system_prompt_file"), project_root
        )
        values["system_prompt"] = _read_prompt_file(path, file)

    return Profile(
        name=name,
        path=path,
        values=MappingProxyType(values),
        description=values.get("description", ""),
    )


def profile_defaults(profile: Profile) -> dict[str, Any]:
    """The profile as defaults for the command line options, keyed by argparse ``dest``.

    Only the settings the profile makes are included. ``reference_wavs`` becomes the
    ``reference_wav`` list, ``dataset`` becomes ``dataset_dir`` and the text of
    ``system_prompt_file`` becomes ``system_prompt``; ``description`` is not an option.
    """
    defaults: dict[str, Any] = {}
    for key, dest in _CLI_DESTS.items():
        if key in profile.values:
            value = profile.values[key]
            defaults[dest] = list(value) if isinstance(value, list) else value
    return defaults


# --- checking and listing ------------------------------------------------------------------


def check_profile(profile: Profile) -> list[str]:
    """Why the profile cannot be used as it is, one human-readable line each (empty: fine).

    Looks at the files it points to: the checkpoint, the reference clips, and the dataset
    (only when there are no explicit reference clips, since that is what it is for then).
    """
    problems = []
    values = profile.values
    checkpoint = values.get("checkpoint")
    if checkpoint is not None:
        if not checkpoint.exists():
            problems.append(f"checkpoint not found: {checkpoint}")
        elif checkpoint.is_dir() and not any(checkpoint.glob("*.pth")):
            problems.append(f"no .pth model file in the checkpoint directory {checkpoint}")

    wavs = values.get("reference_wavs")
    if wavs is not None:
        problems += [f"reference clip not found: {wav}" for wav in wavs if not wav.is_file()]
    elif "dataset" in values:
        dataset = values["dataset"]
        if not dataset.is_dir():
            problems.append(f"dataset directory not found: {dataset}")
        elif not (dataset / METADATA_FILE).is_file():
            problems.append(f"no {METADATA_FILE} in the dataset directory {dataset}")
    return problems


def list_profiles(profiles_dir: Path, project_root: Path) -> list[ProfileSummary]:
    """A summary of every profile in ``profiles_dir``, sorted by name.

    Files starting with ``_`` or ``.`` (templates, hidden files) are ignored. A profile
    that cannot be loaded is listed anyway, with the reason in ``problems``; so is one
    that loads but points to files that are missing. A missing directory gives ``[]``.
    """
    summaries = []
    for path in _profile_files(profiles_dir):
        try:
            profile = load_profile(str(path), profiles_dir, project_root)
        except ProfileError as exc:
            summaries.append(ProfileSummary(path.stem, path, "", None, None, (str(exc),)))
            continue
        problems = check_profile(profile)
        if not _NAME_RE.fullmatch(path.stem):
            problems.append("the name cannot be used with --profile: use letters, digits, - and _")
        checkpoint = profile.values.get("checkpoint")
        summaries.append(
            ProfileSummary(
                name=profile.name,
                path=path,
                description=profile.description,
                language=profile.values.get("language"),
                checkpoint=None
                if checkpoint is None
                else relative_to_root(checkpoint, project_root),
                problems=tuple(problems),
            )
        )
    return summaries


# --- writing ---------------------------------------------------------------------------


def relative_to_root(path: Path, project_root: Path) -> str:
    """``path`` as a string relative to ``project_root`` (``models/frieren``) when it is inside
    it, otherwise as it is. For the profiles herald creates, so that they stay portable."""
    path, project_root = Path(path).expanduser(), Path(project_root)
    if not path.is_absolute():
        return path.as_posix()
    for candidate, root in ((path, project_root), (path.resolve(), project_root.resolve())):
        if candidate.is_relative_to(root):
            return candidate.relative_to(root).as_posix()
    return str(path)


_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _escape(text: str, *, keep: str = "") -> str:
    """Escape ``text`` for the inside of a TOML basic string (``keep``: characters left as is)."""
    out = []
    for char in text:
        if char in keep:
            out.append(char)
        elif char in _ESCAPES:
            out.append(_ESCAPES[char])
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return "".join(out)


def _quote(text: str) -> str:
    return f'"{_escape(text)}"'


def _multiline(text: str) -> str:
    """A ``\"\"\"`` block that keeps the line breaks (and the quotes) of ``text`` readable."""
    # A quote right before the closing delimiter would be ambiguous: it is escaped.
    trailing_quote = text.endswith('"')
    if trailing_quote:
        text = text[:-1]
    body = _escape(text, keep='\n\t"').replace('"""', '""\\"')  # no run of 3 quotes
    if trailing_quote:
        body += '\\"'
    return f'"""\n{body}"""'


def _format(key: str, value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _multiline(value) if "\n" in value else _quote(value)
    if isinstance(value, list):
        return "[" + ", ".join(_quote(item) for item in value) + "]"
    raise ProfileError(f"Cannot write {key}: unsupported value of type {type(value).__name__}")


def write_profile(path: Path, values: Mapping[str, Any], *, overwrite: bool = False) -> Path:
    """Write ``values`` as a profile file and return its path.

    The values are validated like a profile that is loaded (valid keys, right types), so a
    file written here always loads again. Keys are written in :data:`PROFILE_KEYS` order;
    strings with line breaks become ``\"\"\"`` blocks. Paths are written as given (pass
    project-relative strings, see :func:`relative_to_root`; ``Path`` objects are converted).
    Parent directories are created. An existing file is never replaced unless
    ``overwrite=True``.
    """
    path = Path(path)
    raw: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Path):
            value = value.as_posix()
        elif isinstance(value, list):
            value = [v.as_posix() if isinstance(v, Path) else v for v in value]
        raw[key] = value
    checked = _check_values(path, raw)

    text = "".join(f"{key} = {_format(key, value)}\n" for key, value in checked.items())
    if tomllib.loads(text) != checked:  # cannot happen; never write a file that does not load
        raise ProfileError(f"{path}: the profile could not be written as TOML")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w" if overwrite else "x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    except FileExistsError:
        raise ProfileError(f"{path} already exists; it was not overwritten") from None
    return path


def ensure_profile(profiles_dir: Path, name: str, values: Mapping[str, Any]) -> tuple[Path, bool]:
    """Create ``<profiles_dir>/<name>.toml`` from ``values`` unless it exists already.

    Returns ``(path, created)``. An existing profile is never touched.
    """
    if not _NAME_RE.fullmatch(name):
        raise ProfileError(f"Invalid profile name {name!r}: use letters, digits, '-' and '_'")
    path = Path(profiles_dir) / f"{name}.toml"
    if path.exists():
        return path, False
    write_profile(path, values)
    return path, True
