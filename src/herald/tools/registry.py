"""Tools the model may call, and the registry that offers them and runs them safely.

A :class:`Tool` is a name, a description, a JSON Schema for its arguments and a plain function.
You rarely build one by hand: decorate a function with ``@tool`` (see ``herald.tools``) and the
schema is generated from its type hints.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from herald.llm.ollama_client import ToolCall
from herald.tools.context import ToolContext

logger = logging.getLogger(__name__)

# Tool names are sent to the model and used as dictionary keys: keep them boring.
_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")

# Longest tool result (in characters) that is passed back to the model.
_MAX_RESULT_CHARS = 2000

# What a small model is allowed to get slightly wrong (see `_coerce_value`). ASCII digits only,
# no exponent notation, and integers stop well short of anything silly.
_INTEGER_TEXT = re.compile(r"([+-]?[0-9]{1,18})(?:\.0+)?")
_NUMBER_TEXT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")
_MAX_INTEGRAL_FLOAT = 1e18

# JSON Schema type names that `_check_arguments` knows how to check.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


@dataclass(frozen=True)
class Tool:
    """A function the model may call.

    The handler receives arguments written by the model, so treat them as untrusted input. The
    registry checks them against ``parameters`` first (required names, no unknown names unless
    ``additionalProperties`` is true, and the basic types), but not their values: a handler must
    still validate ranges and must never pass them to a shell, ``eval`` or an arbitrary file path.
    Whatever it returns is cut to 2000 characters before the model sees it.

    Small models often get an argument's JSON type slightly wrong, so the registry first fixes
    the clean cases (the handler gets the fixed value): ``"120"`` or ``120.0`` for an integer,
    ``"1.5"`` for a number, ``"true"``/``"false"`` for a boolean, ``"[1, 2]"`` for an array,
    ``"{...}"`` for an object, and ``null`` for an optional parameter (it counts as not given).
    Anything else, such as ``"2 minutes"``, ``"1e9"``, ``2.5`` for an integer or ``true`` for a
    number, is rejected. A handler should therefore expect the types its schema declares.

    ``context_param`` names a handler parameter that receives the registry's
    :class:`~herald.tools.context.ToolContext` (to speak or schedule work). It is not part of
    ``parameters`` and the model cannot set it.

    ``triggers`` are words or phrases that say when the tool is relevant, for example
    ``("timer", "sveglia")``. A tool with triggers is only offered to the model for a user message
    that contains one of them (ignoring case and accents); a tool without triggers is always
    offered. Why: a small model (a 3B one, say) tends to call any tool it is offered on almost
    every message, even for "Tell me a joke", and no wording of the description prevents that. So
    the decision is made here, deterministically, before the model sees the message. Choose the
    triggers generously, since a message that matches none of them can never reach the tool.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema: {"type": "object", "properties": {...}, ...}
    handler: Callable[..., str]  # called with the model's arguments as keywords
    context_param: str | None = None
    triggers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Accept any list or tuple, keep a tuple. A bare string would be read letter by letter.
        triggers = self.triggers
        if not isinstance(triggers, (list, tuple)) or not all(
            isinstance(t, str) and _fold(t).strip() for t in triggers
        ):
            raise TypeError(
                f"Tool {self.name!r}: triggers must be a list or tuple of non-empty strings, "
                f"got {triggers!r}"
            )
        object.__setattr__(self, "triggers", tuple(triggers))

    def schema(self) -> dict[str, Any]:
        """The Ollama (OpenAI-style) function schema sent to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """The tools an assistant offers to the model. Empty means no tools.

    ``context`` is handed to every tool that asks for one (``Tool.context_param``).
    """

    def __init__(self, context: ToolContext | None = None) -> None:
        self.context = context
        self._tools: dict[str, Tool] = {}

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def register(self, tool: Tool) -> None:
        """Add ``tool``. Raises ``ValueError`` for a bad or already used name."""
        if not _NAME_PATTERN.fullmatch(tool.name):
            raise ValueError(
                f"Invalid tool name {tool.name!r}: use 1-64 letters, digits, '_' or '-'"
            )
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def matching(self, text: str | None = None) -> list[str]:
        """Names of the tools to offer for the user message ``text`` (all of them if ``None``).

        A tool without triggers is always offered; one with triggers only if at least one of them
        occurs in ``text`` as a substring, ignoring case and accents ("Tè" matches "te"). See
        :class:`Tool` for why.
        """
        if text is None:
            return list(self._tools)
        folded = _fold(text)
        return [
            name
            for name, tool in self._tools.items()
            if not tool.triggers or any(_fold(trigger) in folded for trigger in tool.triggers)
        ]

    def schemas(self, text: str | None = None) -> list[dict[str, Any]]:
        """The function schemas of the tools :meth:`matching` ``text`` (all if ``None``)."""
        return [self._tools[name].schema() for name in self.matching(text)]

    def call(self, call: ToolCall, offered: Collection[str] | None = None) -> str:
        """Run a tool call and return its result text (at most ~2000 characters).

        Never raises: an unknown tool, invalid arguments, a missing context or a failing handler
        come back as an ``error: ...`` text, which the model can read and recover from. Arguments
        with a slightly wrong JSON type are fixed first, as described in :class:`Tool`.

        ``offered`` lists the tool names that were offered to the model for this message; a call
        to any other tool is refused exactly like a call to an unregistered one. ``None`` allows
        every registered tool.
        """
        text = self._run(call, offered)
        return text if len(text) <= _MAX_RESULT_CHARS else text[:_MAX_RESULT_CHARS] + "…"

    def _run(self, call: ToolCall, offered: Collection[str] | None) -> str:
        tool = self._tools.get(call.name)
        if tool is None or (offered is not None and call.name not in offered):
            return _tool_error(f"unknown tool {call.name!r}")
        arguments = _coerce_arguments(tool.parameters, call.arguments, call.name)
        problem = _check_arguments(tool.parameters, arguments)
        if problem:
            return _tool_error(f"invalid arguments for {call.name}: {problem}")
        kwargs = dict(arguments)
        if tool.context_param:
            if self.context is None:
                return _tool_error(f"{call.name} needs a tool context and this registry has none")
            kwargs[tool.context_param] = self.context  # always ours, whatever the model sent
        try:
            return str(tool.handler(**kwargs))
        except Exception as exc:  # any handler failure goes back to the model
            return _tool_error(f"{call.name} failed: {type(exc).__name__}: {exc}")


def _fold(text: str) -> str:
    """Lower-case ``text`` and strip its accents, for matching triggers: "Tè" -> "te"."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _tool_error(message: str) -> str:
    logger.warning("Tool call failed: %s", message)
    return f"error: {message}"


def _coerce_arguments(
    schema: dict[str, Any], arguments: dict[str, Any], tool_name: str
) -> dict[str, Any]:
    """Fix the clean, unambiguous type slips of ``arguments``; everything else is left alone."""
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    fixed: dict[str, Any] = {}
    for name, value in arguments.items():
        if value is None and name in properties and name not in required:
            logger.debug("Tool %s: dropped null for the optional argument %s", tool_name, name)
            continue
        expected = properties.get(name, {}).get("type")
        coerced = _coerce_value(value, expected) if isinstance(expected, str) else value
        if coerced is not value:
            logger.debug("Tool %s: read %s=%r as %r", tool_name, name, value, coerced)
        fixed[name] = coerced
    return fixed


def _coerce_value(value: Any, expected: str) -> Any:
    """``value`` read as the JSON type ``expected``, or ``value`` itself if that is not clean."""
    if isinstance(value, bool):  # never turn a boolean into a number, or the other way round
        return value
    if expected == "integer":
        if isinstance(value, float) and value.is_integer() and abs(value) < _MAX_INTEGRAL_FLOAT:
            return int(value)
        match = _INTEGER_TEXT.fullmatch(value.strip()) if isinstance(value, str) else None
        if match:
            return int(match[1])
    elif expected == "number":
        if isinstance(value, str) and _NUMBER_TEXT.fullmatch(value.strip()):
            number = float(value)
            return number if math.isfinite(number) else value
    elif expected == "boolean":
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
    elif expected in ("array", "object") and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, RecursionError):
            return value
        if isinstance(parsed, _JSON_TYPES[expected]):
            return parsed
    return value


def _check_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """Say what is wrong with ``arguments`` compared to the JSON ``schema``, or return None.

    A deliberately small subset of JSON Schema. Unlike the standard, unknown names are rejected
    unless the schema says ``"additionalProperties": true``.
    """
    properties = schema.get("properties", {})
    missing = [name for name in schema.get("required", []) if name not in arguments]
    if missing:
        return "missing " + ", ".join(missing)
    if not schema.get("additionalProperties", False):
        unknown = [name for name in arguments if name not in properties]
        if unknown:
            return "unknown " + ", ".join(unknown)
    for name, value in arguments.items():
        expected = properties.get(name, {}).get("type")
        if isinstance(expected, str) and expected in _JSON_TYPES:
            # bool is a subclass of int in Python, but true is not a number in JSON.
            is_bool = isinstance(value, bool)
            if not isinstance(value, _JSON_TYPES[expected]) or (is_bool and expected != "boolean"):
                return f"{name} must be of type {expected}"
    return None
