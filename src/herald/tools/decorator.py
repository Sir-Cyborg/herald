"""``@tool``: turn a plain function into a tool the model can call.

The JSON Schema Ollama needs is generated from the function's type hints and docstring::

    from herald.tools import tool

    @tool
    def add(a: int, b: int = 0) -> str:
        \"\"\"Add two numbers.

        Args:
            a: The first number.
            b: The second number.
        \"\"\"
        return str(a + b)

The decorated function is returned unchanged; its :class:`~herald.tools.registry.Tool` is stored
in its ``__herald_tool__`` attribute, which is what the loader looks for.

Give the tool ``triggers`` (``@tool(triggers=["timer", "sveglia"])``) so that it is only offered
to the model when the user's message mentions one of them. Small models call every tool they are
offered, on almost every message, so without triggers a tool is always in the way.

Supported type hints: ``str``, ``int``, ``float``, ``bool``, ``list`` and ``list[T]``, ``dict``,
``Literal["a", "b"]`` (strings only) and ``X | None`` (the ``None`` is dropped). A parameter
without a default is required. A parameter annotated ``ToolContext`` is filled in by the
application, not by the model, and is left out of the schema.
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeVar, overload

from herald.tools.context import ToolContext
from herald.tools.registry import Tool

F = TypeVar("F", bound=Callable[..., Any])

_SIMPLE_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# Docstring section headers. The description ends at the first one; only the first group
# describes parameters.
_PARAM_SECTIONS = frozenset({"Args", "Arguments", "Parameters"})
_SECTIONS = _PARAM_SECTIONS | {
    "Returns",
    "Raises",
    "Yields",
    "Example",
    "Examples",
    "Note",
    "Notes",
}
_PARAM_LINE = re.compile(r"(\*{0,2}\w+)\s*(?:\([^)]*\))?:\s*(.*)")  # "name: text", "name (T): text"


@overload
def tool(func: F, /) -> F: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    triggers: Sequence[str] | None = None,
) -> Callable[[F], F]: ...


def tool(
    func: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
    triggers: Sequence[str] | None = None,
) -> Any:
    """Mark a function as a tool: ``@tool`` or ``@tool(name=..., description=..., triggers=...)``.

    The name defaults to the function name and the description to its docstring. ``triggers`` is
    a list of words or phrases (non-empty strings): the tool is only offered to the model for a
    user message that contains one of them, ignoring case and accents. Without triggers it is
    offered every time. Use them: small models call any tool they are offered on almost every
    message, so the match is decided here, before the model sees the message. Raises
    ``TypeError`` or ``ValueError`` (naming the function) if the tool cannot be built.
    """
    if func is not None and not callable(func):
        raise TypeError("Use @tool or @tool(name=..., description=..., triggers=...)")

    def decorate(function: F) -> F:
        built = _build_tool(function, name, description, triggers)
        function.__herald_tool__ = built  # type: ignore[attr-defined]
        return function

    return decorate if func is None else decorate(func)


def _build_tool(
    func: Callable[..., Any],
    name: str | None,
    description: str | None,
    triggers: Sequence[str] | None,
) -> Tool:
    label = func.__name__
    doc_description, param_docs = _parse_docstring(inspect.getdoc(func) or "")
    description = " ".join((description or doc_description).split())
    if not description:
        raise ValueError(
            f"Tool function {label!r} needs a docstring: the model reads it to know what it does"
        )
    parameters, context_param = _build_parameters(func, param_docs)
    return Tool(
        name or label,
        description,
        parameters,
        func,
        context_param,
        () if triggers is None else triggers,  # type: ignore[arg-type]  # Tool checks the type
    )


def _build_parameters(
    func: Callable[..., Any], param_docs: dict[str, str]
) -> tuple[dict[str, Any], str | None]:
    label = func.__name__
    try:
        hints = typing.get_type_hints(func)
    except Exception as exc:  # e.g. a NameError for a type that only exists under TYPE_CHECKING
        raise TypeError(f"Tool function {label!r}: cannot resolve its type hints: {exc}") from exc

    properties: dict[str, Any] = {}
    required: list[str] = []
    context_param: str | None = None
    for name, param in inspect.signature(func).parameters.items():
        where = f"parameter {name!r} of tool function {label!r}"
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"{where}: *args and **kwargs are not supported")
        if param.kind is param.POSITIONAL_ONLY:
            raise TypeError(f"{where}: positional-only parameters are not supported")
        if name not in hints:
            raise TypeError(f"{where} needs a type hint")

        annotation = hints[name]
        if annotation is ToolContext:
            if context_param is not None:
                raise TypeError(f"Tool function {label!r} has more than one ToolContext parameter")
            context_param = name
            continue

        schema = _json_schema(annotation)
        if schema is None:
            raise TypeError(f"{where}: unsupported type hint {annotation!r}")
        if name in param_docs:
            schema["description"] = param_docs[name]
        properties[name] = schema
        if param.default is param.empty:
            required.append(name)

    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    parameters["additionalProperties"] = False
    return parameters, context_param


def _json_schema(annotation: Any) -> dict[str, Any] | None:
    """The JSON Schema of a type hint, or ``None`` if the hint is not supported."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if isinstance(annotation, type) and annotation in _SIMPLE_TYPES:
        return {"type": _SIMPLE_TYPES[annotation]}
    if origin is list:
        schema: dict[str, Any] = {"type": "array"}
        if args:
            items = _json_schema(args[0])
            if items is None:
                return None
            schema["items"] = items
        return schema
    if origin is dict:
        return {"type": "object"}
    if origin is Literal:
        return (
            {"type": "string", "enum": list(args)}
            if all(isinstance(a, str) for a in args)
            else None
        )
    if origin is typing.Union or origin is types.UnionType:
        others = [a for a in args if a is not type(None)]
        return _json_schema(others[0]) if len(others) == 1 and len(args) == 2 else None
    return None


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into its description and its ``Args:`` descriptions.

    Whitespace is normalised, so wrapped lines become one sentence.
    """
    description: list[str] = []
    params: dict[str, list[str]] = {}
    section: str | None = None
    current: str | None = None  # the parameter whose description is being read
    indent: int | None = None  # indentation of the "name: text" lines
    for line in doc.splitlines():
        text = line.strip()
        if not line[:1].isspace() and text.endswith(":") and text[:-1] in _SECTIONS:
            section, current, indent = text[:-1], None, None
        elif section is None:
            description.append(text)
        elif section in _PARAM_SECTIONS and text:
            width = len(line) - len(line.lstrip())
            indent = width if indent is None else indent
            match = _PARAM_LINE.fullmatch(text)
            if match and width <= indent:
                current = match[1].lstrip("*")
                params[current] = [match[2]]
            elif current is not None:  # a wrapped continuation line
                params[current].append(text)
    return _squash(description), {name: _squash(parts) for name, parts in params.items()}


def _squash(lines: list[str]) -> str:
    return " ".join(" ".join(lines).split())
