from __future__ import annotations

from typing import Literal, Optional, Union

import pytest

from herald.llm.ollama_client import ToolCall
from herald.tools import Tool, ToolContext, ToolRegistry, tool


def schema_of(func):
    return func.__herald_tool__.parameters


# --- the decorator itself -----------------------------------------------------------------------


def test_the_function_is_returned_unchanged_with_its_tool_attached():
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    decorated = tool(add)

    assert decorated is add
    assert add(2, 3) == "5"  # still a normal function
    assert isinstance(add.__herald_tool__, Tool)
    assert add.__herald_tool__.handler is add


def test_name_and_description_default_to_the_function_and_its_docstring():
    @tool
    def get_time() -> str:
        """Tell the time."""
        return "12:00"

    assert get_time.__herald_tool__.name == "get_time"
    assert get_time.__herald_tool__.description == "Tell the time."


def test_bare_and_called_forms_are_equivalent():
    @tool
    def one() -> str:
        """Doc."""
        return ""

    @tool()
    def two() -> str:
        """Doc."""
        return ""

    assert one.__herald_tool__.parameters == two.__herald_tool__.parameters


def test_name_and_description_can_be_overridden():
    @tool(name="clock", description="  Current   time. ")
    def get_time() -> str:
        return "12:00"  # no docstring needed when the description is given

    assert get_time.__herald_tool__.name == "clock"
    assert get_time.__herald_tool__.description == "Current time."


def test_using_the_decorator_with_a_positional_name_is_an_error():
    with pytest.raises(TypeError, match="@tool"):
        tool("clock")


def test_the_generated_schema_is_a_valid_ollama_function_schema():
    @tool
    def add(a: int, b: int = 1) -> str:
        """Add two numbers."""
        return str(a + b)

    assert add.__herald_tool__.schema() == {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
    }


def test_no_parameters_means_no_required_list():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    assert schema_of(ping) == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


# --- type hints ---------------------------------------------------------------------------------


@tool
def every_type(
    text: str,
    count: int,
    ratio: float,
    flag: bool,
    things: list,
    names: list[str],
    scores: list[float],
    nested: list[list[int]],
    options: dict,
    mapping: dict[str, int],
) -> str:
    """Use every type."""
    return ""


def test_every_supported_type_maps_to_json_schema():
    properties = schema_of(every_type)["properties"]
    assert properties == {
        "text": {"type": "string"},
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
        "things": {"type": "array"},
        "names": {"type": "array", "items": {"type": "string"}},
        "scores": {"type": "array", "items": {"type": "number"}},
        "nested": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        "options": {"type": "object"},
        "mapping": {"type": "object"},
    }
    assert schema_of(every_type)["required"] == list(properties)


# `Optional[X]` and `Union[X, None]` are the older spellings of `X | None` and must work too.
@tool
def optionals(
    a: str | None,
    b: Optional[int] = None,  # noqa: UP045
    c: Union[float, None] = 1.5,  # noqa: UP007
    d: Literal["hot", "cold"] = "hot",
    e: list[Literal["x", "y"]] | None = None,
    f: bool = False,
) -> str:
    """Optionals."""
    return ""


def test_optional_types_drop_the_none_and_defaults_make_parameters_optional():
    schema = schema_of(optionals)
    assert schema["properties"] == {
        "a": {"type": "string"},
        "b": {"type": "integer"},
        "c": {"type": "number"},
        "d": {"type": "string", "enum": ["hot", "cold"]},
        "e": {"type": "array", "items": {"type": "string", "enum": ["x", "y"]}},
        "f": {"type": "boolean"},
    }
    # `a` has no default, so it stays required even though it may be None.
    assert schema["required"] == ["a"]


def function_with_hint(annotation):
    def broken(value):
        """Doc."""
        return ""

    broken.__annotations__ = {"value": annotation}
    return broken


@pytest.mark.parametrize(
    "annotation",
    [bytes, set[str], list[bytes], str | int, Literal[1, 2], tuple[int, int], type(None), object],
    ids=["bytes", "set", "list-of-bytes", "union", "number-literal", "tuple", "none", "object"],
)
def test_unsupported_type_hints_are_a_clear_type_error(annotation):
    with pytest.raises(TypeError, match=r"parameter 'value' of tool function 'broken'"):
        tool(function_with_hint(annotation))


def test_a_missing_type_hint_is_an_error():
    def untyped(value) -> str:
        """Doc."""
        return ""

    with pytest.raises(TypeError, match=r"parameter 'value' of tool function 'untyped'.*type hint"):
        tool(untyped)


def test_star_arguments_are_rejected():
    def varargs(*values: int) -> str:
        """Doc."""
        return ""

    def varkw(**values: int) -> str:
        """Doc."""
        return ""

    for function in (varargs, varkw):
        with pytest.raises(TypeError, match=r"tool function .*\*args and \*\*kwargs"):
            tool(function)


def test_positional_only_parameters_are_rejected():
    def posonly(value: int, /) -> str:
        """Doc."""
        return ""

    with pytest.raises(TypeError, match=r"parameter 'value' of tool function 'posonly'"):
        tool(posonly)


def test_unresolvable_annotations_are_a_type_error_naming_the_function():
    def broken(value: DoesNotExist) -> str:  # noqa: F821
        """Doc."""
        return ""

    with pytest.raises(TypeError, match=r"Tool function 'broken': cannot resolve.*type hints"):
        tool(broken)


# --- docstrings ---------------------------------------------------------------------------------


def test_the_description_is_the_text_before_the_args_section():
    @tool
    def sample(a: int) -> str:
        """Do the thing.

        It may take a few
        lines and paragraphs.

        Args:
            a: A number.

        Returns:
            Text that is not part of the description.
        """
        return ""

    assert (
        sample.__herald_tool__.description
        == "Do the thing. It may take a few lines and paragraphs."
    )


def test_parameter_descriptions_come_from_the_args_section_with_wrapped_lines():
    @tool
    def sample(seconds: int, message: str = "x", flag: bool = False) -> str:
        """Start something.

        Args:
            seconds: How long to wait, in seconds
                (1 to 86400). Convert minutes
                and hours to seconds.
            message (str): What to say: in the language
                of the conversation.
            flag: Whether to do it.
        """
        return ""

    properties = schema_of(sample)["properties"]
    assert properties["seconds"]["description"] == (
        "How long to wait, in seconds (1 to 86400). Convert minutes and hours to seconds."
    )
    assert properties["message"]["description"] == (
        "What to say: in the language of the conversation."
    )
    assert properties["flag"]["description"] == "Whether to do it."


def test_parameters_missing_from_the_docstring_have_no_description():
    @tool
    def sample(a: int, b: int) -> str:
        """Doc.

        Args:
            a: Documented.
            ghost: Not a parameter of the function, ignored.
        """
        return ""

    properties = schema_of(sample)["properties"]
    assert properties["a"] == {"type": "integer", "description": "Documented."}
    assert properties["b"] == {"type": "integer"}
    assert "ghost" not in properties


def test_a_docstring_with_only_a_summary_line():
    @tool
    def sample(a: int) -> str:
        """Just one line."""
        return ""

    assert sample.__herald_tool__.description == "Just one line."
    assert schema_of(sample)["properties"]["a"] == {"type": "integer"}


def test_a_function_without_a_docstring_is_an_error():
    def silent(a: int) -> str:
        return ""

    with pytest.raises(ValueError, match=r"'silent' needs a docstring"):
        tool(silent)


def test_a_docstring_with_only_sections_has_no_description():
    def sample(a: int) -> str:
        """Args:
        a: A number.
        """
        return ""

    with pytest.raises(ValueError, match="needs a docstring"):
        tool(sample)


# --- the context parameter ------------------------------------------------------------------------


def test_a_context_parameter_is_hidden_from_the_schema_and_recorded():
    @tool
    def remind(seconds: int, *, ctx: ToolContext) -> str:
        """Remind later.

        Args:
            seconds: Delay.
            ctx: The context (ignored in the schema).
        """
        return ""

    built = remind.__herald_tool__
    assert built.context_param == "ctx"
    assert list(built.parameters["properties"]) == ["seconds"]
    assert built.parameters["required"] == ["seconds"]
    assert "ctx" not in str(built.schema())


def test_a_context_parameter_does_not_have_to_be_keyword_only():
    @tool
    def remind(ctx: ToolContext, seconds: int) -> str:
        """Remind later."""
        return ""

    assert remind.__herald_tool__.context_param == "ctx"
    assert list(remind.__herald_tool__.parameters["properties"]) == ["seconds"]


def test_tools_without_a_context_have_no_context_param():
    @tool
    def plain() -> str:
        """Plain."""
        return ""

    assert plain.__herald_tool__.context_param is None


def test_two_context_parameters_are_an_error():
    def twice(a: ToolContext, b: ToolContext) -> str:
        """Doc."""
        return ""

    with pytest.raises(TypeError, match="more than one ToolContext"):
        tool(twice)


def test_the_context_is_injected_when_the_tool_is_called_through_a_registry():
    context = ToolContext(say=lambda text: None, scheduler=None)

    @tool
    def remind(seconds: int, *, ctx: ToolContext) -> str:
        """Remind later."""
        return f"{seconds} {ctx is context}"

    registry = ToolRegistry(context)
    registry.register(remind.__herald_tool__)

    assert registry.call(ToolCall("remind", {"seconds": 5})) == "5 True"
    assert registry.call(ToolCall("remind", {"seconds": 5, "ctx": "x"})).startswith("error:")


def test_a_decorated_function_works_end_to_end_through_the_registry():
    @tool
    def add(a: int, b: int = 10) -> str:
        """Add two numbers."""
        return str(a + b)

    registry = ToolRegistry()
    registry.register(add.__herald_tool__)

    assert registry.call(ToolCall("add", {"a": 1})) == "11"
    assert registry.call(ToolCall("add", {"a": 1, "b": 2})) == "3"
    assert registry.call(ToolCall("add", {"a": "5"})) == "15"  # a numeric string is fine
    assert registry.call(ToolCall("add", {"a": "one"})).startswith("error: invalid arguments")
    assert registry.call(ToolCall("add", {})).startswith("error: invalid arguments")


# --- triggers -----------------------------------------------------------------------------------


def test_triggers_are_stored_as_a_tuple():
    @tool(triggers=["timer", "sveglia"])
    def set_timer() -> str:
        """Start a timer."""
        return ""

    assert set_timer.__herald_tool__.triggers == ("timer", "sveglia")


def test_triggers_can_be_given_as_a_tuple_with_other_options():
    @tool(name="alarm", description="Ring later.", triggers=("alarm",))
    def ring() -> str:
        return ""

    built = ring.__herald_tool__
    assert (built.name, built.description, built.triggers) == ("alarm", "Ring later.", ("alarm",))


def test_a_tool_without_triggers_is_always_offered():
    @tool
    def bare() -> str:
        """Bare."""
        return ""

    @tool()
    def called() -> str:
        """Called."""
        return ""

    @tool(triggers=None)
    def explicit_none() -> str:
        """None."""
        return ""

    @tool(triggers=[])
    def empty() -> str:
        """Empty."""
        return ""

    for function in (bare, called, explicit_none, empty):
        assert function.__herald_tool__.triggers == ()


@pytest.mark.parametrize("triggers", ["timer", "", [""], ["ok", "  "], [1], 5, [["a"]]])
def test_invalid_triggers_are_a_type_error(triggers):
    with pytest.raises(TypeError, match="triggers"):

        @tool(triggers=triggers)
        def broken() -> str:
            """Broken."""
            return ""


def test_the_triggers_are_not_part_of_the_schema():
    @tool(triggers=["sveglia"])
    def set_timer(seconds: int) -> str:
        """Start a countdown."""
        return ""

    schema = str(set_timer.__herald_tool__.schema())
    assert "sveglia" not in schema
    assert "triggers" not in schema
