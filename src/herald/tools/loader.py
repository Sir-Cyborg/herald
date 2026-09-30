"""Find the tools: the built-in ones and the user's own scripts.

Built-in tools are the modules of the ``herald.tools.builtin`` package. User tools are plain
``.py`` files directly inside a directory (``tools/`` in the project by default). Either way, a
tool is a function decorated with ``@tool`` (or a module-level ``Tool`` instance); every such
object a module defines is registered.

Security: a user script is imported, so it runs with your privileges as soon as Herald starts,
like any program you launch. The tool arguments it later receives are written by the model, so a
handler must treat them as untrusted input: never pass them to a shell, ``eval`` or ``exec``.

Loading never raises for a bad script. A script that fails to import, a tool that cannot be
registered and a duplicate name are listed in ``LoadReport.errors`` and the rest still loads.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType

from herald.tools.context import ToolContext
from herald.tools.registry import Tool, ToolRegistry

BUILTIN_PACKAGE = "herald.tools.builtin"
_USER_MODULE_PREFIX = "herald_user_tools"


@dataclass(frozen=True)
class LoadedTool:
    name: str
    description: str
    source: str  # "builtin" or the path of the user script
    triggers: tuple[str, ...] = ()  # empty: the tool is offered on every message


@dataclass(frozen=True)
class LoadReport:
    registry: ToolRegistry
    tools: tuple[LoadedTool, ...]
    errors: tuple[str, ...]  # one readable line per problem


def load_tools(
    user_dir: Path | None, context: ToolContext | None, *, builtin: bool = True
) -> LoadReport:
    """Build a registry from the built-in tools (if ``builtin``) and the scripts in ``user_dir``.

    ``user_dir`` may be ``None`` or missing. Built-ins are loaded first, then scripts sorted by
    file name; on a duplicate name the first tool wins and the other one is reported.
    """
    registry = ToolRegistry(context)
    tools: list[LoadedTool] = []
    errors: list[str] = []
    origins: dict[str, tuple[Tool, str]] = {}  # tool name -> (tool, where it came from)

    def add_module(label: str, source: str, import_module: Callable[[], ModuleType]) -> None:
        try:
            found = list(_tools_in(import_module()))
        except (Exception, SystemExit) as exc:  # a script may even call sys.exit()
            errors.append(f"{label}: {_describe(exc)}")
            return
        for tool in found:
            first = origins.get(tool.name)
            if first is not None:
                # The same function imported into another module is not a duplicate.
                if first[0] is not tool:
                    errors.append(
                        f"{label}: tool {tool.name!r} is already defined in {first[1]}; ignored"
                    )
                continue
            try:
                registry.register(tool)
            except ValueError as exc:
                errors.append(f"{label}: {exc}")
                continue
            origins[tool.name] = (tool, label)
            tools.append(LoadedTool(tool.name, tool.description, source, tool.triggers))

    if builtin:
        for module_name in _builtin_module_names(errors):
            add_module(module_name, "builtin", partial(importlib.import_module, module_name))
    for script in _user_scripts(user_dir, errors):
        add_module(str(script), str(script), partial(_import_script, script))
    return LoadReport(registry, tuple(tools), tuple(errors))


def _builtin_module_names(errors: list[str]) -> list[str]:
    """Names of the modules in the built-in package (none if the package is absent or empty)."""
    try:
        package = importlib.import_module(BUILTIN_PACKAGE)
    except ModuleNotFoundError as exc:
        if exc.name != BUILTIN_PACKAGE:  # a dependency of the package is missing: say so
            errors.append(f"{BUILTIN_PACKAGE}: {_describe(exc)}")
        return []
    except (Exception, SystemExit) as exc:
        errors.append(f"{BUILTIN_PACKAGE}: {_describe(exc)}")
        return []
    names = [info.name for info in pkgutil.iter_modules(package.__path__)]
    return [f"{BUILTIN_PACKAGE}.{name}" for name in sorted(names) if not name.startswith("_")]


def _user_scripts(user_dir: Path | None, errors: list[str]) -> list[Path]:
    """The ``*.py`` files directly inside ``user_dir``, sorted by name, without ``_x``/``.x``."""
    if user_dir is None or not Path(user_dir).is_dir():
        return []
    directory = Path(user_dir)
    try:
        scripts = [p for p in directory.glob("*.py") if p.is_file()]
    except OSError as exc:
        errors.append(f"{directory}: {_describe(exc)}")
        return []
    return sorted((p for p in scripts if not p.name.startswith(("_", "."))), key=lambda p: p.name)


def _import_script(path: Path) -> ModuleType:
    """Import a file under a unique module name, without touching ``sys.path``."""
    name = f"{_USER_MODULE_PREFIX}.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # some libraries (dataclasses, pickle) look a module up by name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _tools_in(module: ModuleType) -> Iterator[Tool]:
    """The tools a module defines: ``@tool`` functions and module-level ``Tool`` objects."""
    for value in list(vars(module).values()):
        if isinstance(value, Tool):
            yield value
        else:
            marked = getattr(value, "__herald_tool__", None)
            if isinstance(marked, Tool):
                yield marked


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
