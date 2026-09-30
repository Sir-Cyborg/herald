import logging

import pytest

from herald.llm.ollama_client import ToolCall
from herald.tools import Tool, ToolContext, ToolRegistry


@pytest.fixture
def timers():
    """A registry with one `set_timer` tool."""

    def set_timer(seconds: int) -> str:
        return f"timer set for {seconds} seconds"

    registry = ToolRegistry()
    registry.register(
        Tool(
            "set_timer",
            "Start a countdown timer.",
            {
                "type": "object",
                "properties": {"seconds": {"type": "integer"}},
                "required": ["seconds"],
            },
            set_timer,
        )
    )
    return registry


def test_tool_schema_is_an_ollama_function_schema():
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}}
    tool = Tool("weather", "Look up the weather.", parameters, lambda city: "sunny")
    assert tool.schema() == {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Look up the weather.",
            "parameters": parameters,
        },
    }


def test_registry_lists_schemas_and_counts_tools(timers):
    assert len(ToolRegistry()) == 0
    assert not ToolRegistry()
    assert len(timers) == 1
    assert timers
    assert [s["function"]["name"] for s in timers.schemas()] == ["set_timer"]


def test_registering_a_duplicate_name_is_an_error(timers):
    duplicate = Tool("set_timer", "Another.", {"type": "object"}, lambda: "x")
    with pytest.raises(ValueError, match="set_timer"):
        timers.register(duplicate)


def test_registry_call_runs_the_handler(timers):
    assert timers.call(ToolCall("set_timer", {"seconds": 3})) == "timer set for 3 seconds"


def test_registry_call_turns_the_result_into_text():
    registry = ToolRegistry()
    registry.register(Tool("answer", "The answer.", {"type": "object"}, lambda: 42))
    assert registry.call(ToolCall("answer", {})) == "42"


def make_registry(handler):
    registry = ToolRegistry()
    registry.register(
        Tool(
            "add",
            "Add two numbers.",
            {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a"],
            },
            handler,
        )
    )
    return registry


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (ToolCall("nope", {}), "unknown tool 'nope'"),
        (ToolCall("add", {}), "invalid arguments for add: missing a"),
        (ToolCall("add", {"a": 1, "loud": True}), "invalid arguments for add: unknown loud"),
        (ToolCall("add", {"a": "soon"}), "invalid arguments for add: a must be of type number"),
        (ToolCall("add", {"a": 1, "b": 0}), "add failed: ZeroDivisionError"),  # inside the handler
    ],
    ids=["unknown-tool", "missing", "unknown-argument", "wrong-type", "handler-error"],
)
def test_registry_call_never_raises(caplog, call, expected):
    registry = make_registry(lambda a, b=1: a / b)
    with caplog.at_level(logging.WARNING, logger="herald.tools.registry"):
        result = registry.call(call)

    assert result.startswith("error: ")
    assert expected in result
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_invalid_arguments_never_reach_the_handler():
    calls = []
    registry = make_registry(lambda **kwargs: calls.append(kwargs) or "ok")
    for arguments in ({}, {"a": "x"}, {"a": 1, "extra": 2}):
        assert registry.call(ToolCall("add", arguments)).startswith("error:")
    assert calls == []
    assert registry.call(ToolCall("add", {"a": 1, "b": 2.5})) == "ok"
    assert calls == [{"a": 1, "b": 2.5}]


def check(schema, arguments):
    """Run `arguments` through a tool with `schema`; True if the handler was reached."""
    registry = ToolRegistry()
    registry.register(Tool("t", "Test.", schema, lambda **kwargs: "ok"))
    return registry.call(ToolCall("t", arguments)) == "ok"


def typed(json_type):
    """A schema with one required argument `x` of the given type."""
    return {"type": "object", "properties": {"x": {"type": json_type}}, "required": ["x"]}


@pytest.mark.parametrize(
    ("json_type", "good", "bad"),
    [
        ("string", ["", "a", "120"], [1, None, True, ["a"], {"k": 1}]),
        ("integer", [0, 5, -3], [1.5, True, False, None, ["1"], {}]),
        ("number", [0, 1.5, -2], [True, False, None, ["1"], {}]),
        ("boolean", [True, False], [0, 1, None, ["true"]]),
        ("array", [[], [1, "a"]], [{}, "a", None, 5]),
        ("object", [{}, {"k": 1}], [[], "a", None, 5]),
    ],
)
def test_argument_types_are_checked(json_type, good, bad):
    schema = typed(json_type)
    assert all(check(schema, {"x": value}) for value in good)
    assert not any(check(schema, {"x": value}) for value in bad)


# Values a small model sends that are close enough to the declared type: (type, given, received).
COERCIBLE = [
    ("integer", "120", 120),
    ("integer", "+5", 5),
    ("integer", "-3", -3),
    ("integer", "  7 ", 7),
    ("integer", "120.0", 120),
    ("integer", 120.0, 120),
    ("integer", 1e9, 1_000_000_000),
    ("number", "1", 1.0),
    ("number", "1.5", 1.5),
    ("number", "-.5", -0.5),
    ("number", "5.", 5.0),
    ("boolean", "true", True),
    ("boolean", "False", False),
    ("boolean", " TRUE ", True),
    ("array", "[1, 2]", [1, 2]),
    ("array", "[]", []),
    ("object", '{"k": [1]}', {"k": [1]}),
]

# Readings that would be guesses: these stay errors.
NOT_COERCIBLE = [
    ("integer", "2 minutes"),
    ("integer", "1e9"),
    ("integer", "1.5"),
    ("integer", 1.5),
    ("integer", "twelve"),
    ("integer", ""),
    ("integer", "1_000"),
    ("integer", "\u0661\u0662"),  # Arabic-Indic digits
    ("integer", "1" * 19),
    ("integer", 1e30),
    ("integer", float("inf")),
    ("integer", float("nan")),
    ("integer", True),
    ("integer", False),
    ("number", "1e3"),
    ("number", "1,5"),
    ("number", "abc"),
    ("number", "inf"),
    ("number", "nan"),
    ("number", ""),
    ("number", "."),
    ("number", True),
    ("boolean", "yes"),
    ("boolean", "1"),
    ("boolean", "truthy"),
    ("boolean", ""),
    ("boolean", 1),
    ("boolean", 0),
    ("array", "[1, 2"),
    ("array", '{"k": 1}'),
    ("array", "a, b"),
    ("array", "[" * 100_000),  # deeply nested: must not crash the parser
    ("object", "[1]"),
    ("object", "{bad"),
    ("object", "5"),
]


@pytest.mark.parametrize(("json_type", "given", "received"), COERCIBLE)
def test_clean_type_slips_are_fixed_before_the_handler_sees_them(json_type, given, received):
    seen = []
    registry = ToolRegistry()
    registry.register(Tool("t", "Test.", typed(json_type), lambda x: seen.append(x) or "ok"))

    assert registry.call(ToolCall("t", {"x": given})) == "ok"

    assert seen == [received]
    assert type(seen[0]) is type(received)  # 5 is not 5.0, and True is not 1


@pytest.mark.parametrize(("json_type", "given"), NOT_COERCIBLE)
def test_guesses_are_not_made(json_type, given):
    registry = ToolRegistry()
    registry.register(Tool("t", "Test.", typed(json_type), lambda x: "ok"))
    result = registry.call(ToolCall("t", {"x": given}))
    assert result == f"error: invalid arguments for t: x must be of type {json_type}"


def test_a_string_parameter_is_never_reinterpreted():
    seen = []
    registry = ToolRegistry()
    registry.register(Tool("t", "Test.", typed("string"), lambda x: seen.append(x) or "ok"))
    for text in ("120", "true", "[1]", '{"k": 1}', "null"):
        registry.call(ToolCall("t", {"x": text}))
    assert seen == ["120", "true", "[1]", '{"k": 1}', "null"]


def test_coercion_is_logged_at_debug_level(caplog):
    registry = ToolRegistry()
    registry.register(Tool("t", "Test.", typed("integer"), lambda x: "ok"))
    with caplog.at_level(logging.DEBUG, logger="herald.tools.registry"):
        registry.call(ToolCall("t", {"x": "120"}))
    (record,) = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert record.getMessage() == "Tool t: read x='120' as 120"


def test_a_value_that_needs_no_fixing_is_not_logged(caplog):
    registry = ToolRegistry()
    registry.register(Tool("t", "Test.", typed("integer"), lambda x: "ok"))
    with caplog.at_level(logging.DEBUG, logger="herald.tools.registry"):
        registry.call(ToolCall("t", {"x": 120}))
    assert caplog.records == []


def optional_tool(seen):
    schema = {
        "type": "object",
        "properties": {"need": {"type": "integer"}, "maybe": {"type": "string"}, "free": {}},
        "required": ["need"],
    }
    return Tool("t", "Test.", schema, lambda **kwargs: seen.append(kwargs) or "ok")


def test_null_for_an_optional_parameter_means_not_provided():
    seen = []
    registry = ToolRegistry()
    registry.register(optional_tool(seen))

    assert registry.call(ToolCall("t", {"need": 1, "maybe": None, "free": None})) == "ok"

    assert seen == [{"need": 1}]  # the handler falls back to its own defaults


def test_null_for_a_required_parameter_is_still_an_error():
    seen = []
    registry = ToolRegistry()
    registry.register(optional_tool(seen))

    result = registry.call(ToolCall("t", {"need": None}))

    assert result == "error: invalid arguments for t: need must be of type integer"
    assert seen == []


def test_null_does_not_hide_an_unknown_argument():
    seen = []
    registry = ToolRegistry()
    registry.register(optional_tool(seen))
    result = registry.call(ToolCall("t", {"need": 1, "bogus": None}))
    assert result == "error: invalid arguments for t: unknown bogus"


def test_unknown_arguments_are_still_rejected_after_coercion():
    seen = []
    registry = ToolRegistry()
    registry.register(optional_tool(seen))
    assert registry.call(ToolCall("t", {"need": "5", "bogus": "1"})).startswith("error:")
    assert seen == []


def test_coercion_happens_in_tools_that_take_a_context():
    context = ToolContext(say=lambda text: None, scheduler=None)
    seen = []
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    registry = ToolRegistry(context)
    registry.register(
        Tool(
            "t", "Test.", schema, lambda n, ctx: seen.append((n, ctx)) or "ok", context_param="ctx"
        )
    )
    assert registry.call(ToolCall("t", {"n": "3"})) == "ok"
    assert seen == [(3, context)]


def test_optional_and_untyped_arguments():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}, "y": {}, "z": {"type": ["a"]}},
    }
    assert check(schema, {})  # nothing is required
    assert check(schema, {"y": [1, 2], "z": 3})  # no type, or a type we do not check


def test_extra_arguments_are_only_allowed_when_the_schema_says_so():
    strict = {"type": "object", "properties": {"x": {"type": "string"}}}
    assert not check(strict, {"x": "a", "other": 1})
    assert not check({"type": "object"}, {"other": 1})
    assert not check({**strict, "additionalProperties": False}, {"other": 1})
    assert check({**strict, "additionalProperties": True}, {"x": "a", "other": 1})
    assert check({"type": "object", "additionalProperties": True}, {"other": 1})


def test_a_long_result_is_cut_before_it_reaches_the_model():
    registry = ToolRegistry()
    registry.register(Tool("dump", "Lots of text.", {"type": "object"}, lambda: "x" * 5000))
    result = registry.call(ToolCall("dump", {}))
    assert result == "x" * 2000 + "…"


def test_a_result_at_the_limit_is_left_alone():
    registry = ToolRegistry()
    registry.register(Tool("dump", "Text.", {"type": "object"}, lambda: "x" * 2000))
    assert registry.call(ToolCall("dump", {})) == "x" * 2000


def test_long_error_texts_are_cut_too():
    def broken() -> str:
        raise RuntimeError("z" * 5000)

    registry = ToolRegistry()
    registry.register(Tool("boom", "Fails.", {"type": "object"}, broken))
    assert len(registry.call(ToolCall("boom", {}))) == 2001


# --- tool names ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["set_timer", "a", "Tool-2", "x" * 64, "UPPER_lower-123"])
def test_good_tool_names_are_accepted(name):
    ToolRegistry().register(Tool(name, "Test.", {"type": "object"}, lambda: "ok"))


@pytest.mark.parametrize(
    "name",
    ["", "x" * 65, "has space", "dot.name", "naïve", "new\nline", "trailing\n", "a/b", "set timer"],
)
def test_bad_tool_names_are_rejected(name):
    with pytest.raises(ValueError, match="Invalid tool name"):
        ToolRegistry().register(Tool(name, "Test.", {"type": "object"}, lambda: "ok"))


# --- tool context -------------------------------------------------------------------------------


def context_tool(handler, *, parameters=None):
    return Tool(
        "ctx_tool",
        "Uses the context.",
        parameters or {"type": "object", "properties": {"x": {"type": "integer"}}},
        handler,
        context_param="ctx",
    )


def test_the_context_is_injected_into_the_handler():
    context = ToolContext(say=lambda text: None, scheduler=None)
    seen = {}

    def handler(x, ctx):
        seen.update(x=x, ctx=ctx)
        return "ok"

    registry = ToolRegistry(context)
    registry.register(context_tool(handler))

    assert registry.call(ToolCall("ctx_tool", {"x": 4})) == "ok"
    assert seen == {"x": 4, "ctx": context}
    assert registry.context is context


def test_the_context_parameter_is_not_in_the_schema():
    tool = context_tool(lambda x, ctx: "ok")
    assert "ctx" not in tool.schema()["function"]["parameters"]["properties"]


def test_the_model_cannot_supply_the_context():
    context = ToolContext(say=lambda text: None, scheduler=None)
    seen = []
    registry = ToolRegistry(context)
    registry.register(context_tool(lambda x, ctx: seen.append(ctx) or "ok"))

    result = registry.call(ToolCall("ctx_tool", {"x": 1, "ctx": "evil"}))

    assert result.startswith("error: invalid arguments") and "unknown ctx" in result
    assert seen == []


def test_the_model_cannot_override_the_context_even_with_additional_properties():
    context = ToolContext(say=lambda text: None, scheduler=None)
    seen = []
    schema = {"type": "object", "properties": {}, "additionalProperties": True}
    registry = ToolRegistry(context)
    registry.register(context_tool(lambda ctx, **kw: seen.append(ctx) or "ok", parameters=schema))

    assert registry.call(ToolCall("ctx_tool", {"ctx": "evil"})) == "ok"
    assert seen == [context]


def test_a_tool_that_needs_a_context_fails_politely_without_one(caplog):
    called = []
    registry = ToolRegistry()  # no context
    registry.register(context_tool(lambda x, ctx: called.append(x) or "ok"))

    with caplog.at_level(logging.WARNING, logger="herald.tools.registry"):
        result = registry.call(ToolCall("ctx_tool", {"x": 1}))

    assert result.startswith("error: ") and "context" in result
    assert called == []
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_tools_without_a_context_param_work_without_a_context(timers):
    assert timers.context is None
    assert timers.call(ToolCall("set_timer", {"seconds": 2})) == "timer set for 2 seconds"


# --- triggers: which tools are offered for a message ----------------------------------------------


def trigger_registry(seen=None):
    """`clock` has no triggers; `set_timer` and `weather` do."""
    seen = [] if seen is None else seen
    registry = ToolRegistry()
    registry.register(Tool("clock", "The time.", {"type": "object"}, lambda: "12:00"))
    registry.register(
        Tool(
            "set_timer",
            "Start a timer.",
            {"type": "object"},
            lambda: seen.append("timer") or "ok",
            triggers=("timer", "sveglia", "caffè"),
        )
    )
    registry.register(
        Tool("weather", "The weather.", {"type": "object"}, lambda: "sunny", triggers=["meteo"])
    )
    return registry


def test_none_means_every_tool():
    assert trigger_registry().matching(None) == ["clock", "set_timer", "weather"]
    assert trigger_registry().matching() == ["clock", "set_timer", "weather"]
    assert len(trigger_registry().schemas()) == 3
    assert len(trigger_registry().schemas(None)) == 3


def test_a_tool_without_triggers_is_always_offered():
    registry = trigger_registry()
    for text in ("", "What is the capital of France?", "timer", "   "):
        assert "clock" in registry.matching(text)


def test_a_message_without_any_trigger_offers_only_the_tools_without_triggers():
    registry = trigger_registry()
    assert registry.matching("What is the capital of France?") == ["clock"]
    assert registry.matching("Tell me a joke.") == ["clock"]
    assert registry.matching("Come stai oggi?") == ["clock"]
    assert registry.matching("") == ["clock"]


def test_a_trigger_in_the_message_offers_its_tool_in_registration_order():
    registry = trigger_registry()
    assert registry.matching("Un CAFFE, per favore") == ["clock", "set_timer"]
    assert registry.matching("Set a timer for 5 minutes") == ["clock", "set_timer"]
    assert registry.matching("timer and meteo") == ["clock", "set_timer", "weather"]
    assert registry.matching("che meteo fa?") == ["clock", "weather"]


@pytest.mark.parametrize(
    ("trigger", "text"),
    [
        ("timer", "Set a TIMER for five minutes"),  # case
        ("TIMER", "set a timer"),
        ("Sveglia", "metti una sveglia alle 7"),  # case, the other way round
        ("timer", "Can you set timers?"),  # substring
        ("timer", "retimer"),
        ("tè", "vorrei un te caldo"),  # accents on the trigger
        ("te caldo", "un tè caldo"),  # ... and on the message
        ("café", "a CAFE please"),
        ("cafe", "Un CAFFÈ, anzi un CAFÉ"),
        ("cafe", "café"),  # decomposed e + combining accent
        ("café", "café"),  # precomposed vs decomposed
        ("Straße", "STRASSE"),  # casefold, not just lower
        ("timer", "ＴＩＭＥＲ"),  # full-width letters (NFKD)
        ("set a timer", "Please set a timer now"),  # a phrase
    ],
)
def test_triggers_ignore_case_and_accents_on_both_sides(trigger, text):
    registry = ToolRegistry()
    registry.register(Tool("t", "Test.", {"type": "object"}, lambda: "ok", triggers=(trigger,)))
    assert registry.matching(text) == ["t"]


@pytest.mark.parametrize(
    ("trigger", "text"),
    [
        ("timer", "What is the capital of France?"),
        ("timer", "tim er"),
        ("timer", "time"),
        ("set a timer", "set the timer"),  # a phrase is matched as a whole
        ("tè", "ciao"),
    ],
)
def test_a_trigger_that_is_not_in_the_message_does_not_match(trigger, text):
    registry = ToolRegistry()
    registry.register(Tool("t", "Test.", {"type": "object"}, lambda: "ok", triggers=(trigger,)))
    assert registry.matching(text) == []


def test_schemas_only_include_the_matching_tools():
    registry = trigger_registry()
    names = [s["function"]["name"] for s in registry.schemas("Come stai oggi?")]
    assert names == ["clock"]
    names = [s["function"]["name"] for s in registry.schemas("metti una sveglia")]
    assert names == ["clock", "set_timer"]
    assert registry.schemas("metti una sveglia")[1] == registry._tools["set_timer"].schema()


def test_a_registry_without_tools_offers_nothing():
    assert ToolRegistry().matching("anything") == []
    assert ToolRegistry().schemas("anything") == []


def test_calls_to_tools_that_were_not_offered_are_refused_like_unknown_ones():
    seen = []
    registry = trigger_registry(seen)
    call = ToolCall("set_timer", {})

    refused = registry.call(call, offered=["clock"])

    assert refused == "error: unknown tool 'set_timer'"
    assert refused == registry.call(ToolCall("nope", {}), offered=["clock"]).replace(
        "nope", "set_timer"
    )
    assert seen == []


def test_calls_to_offered_tools_run():
    seen = []
    registry = trigger_registry(seen)
    assert registry.call(ToolCall("set_timer", {}), offered={"clock", "set_timer"}) == "ok"
    assert seen == ["timer"]


def test_without_an_offered_list_every_registered_tool_can_run():
    seen = []
    registry = trigger_registry(seen)
    assert registry.call(ToolCall("set_timer", {})) == "ok"
    assert registry.call(ToolCall("set_timer", {}), offered=None) == "ok"
    assert seen == ["timer", "timer"]


def test_an_empty_offered_list_refuses_everything():
    registry = trigger_registry()
    assert registry.call(ToolCall("clock", {}), offered=[]).startswith("error: unknown tool")


def test_tools_have_no_triggers_by_default():
    assert Tool("t", "Test.", {"type": "object"}, lambda: "ok").triggers == ()


def test_triggers_given_as_a_list_are_kept_as_a_tuple():
    tool = Tool("t", "Test.", {"type": "object"}, lambda: "ok", triggers=["a", "b"])
    assert tool.triggers == ("a", "b")


@pytest.mark.parametrize(
    "triggers",
    ["timer", "", [""], ["ok", ""], ["   "], [1], ["a", None], None, {"a"}, 5, [["a"]]],
    ids=[
        "bare-string",
        "empty-string",
        "empty-item",
        "one-empty-item",
        "blank-item",
        "number-item",
        "none-item",
        "none",
        "set",
        "number",
        "nested-list",
    ],
)
def test_bad_triggers_are_a_type_error(triggers):
    with pytest.raises(TypeError, match=r"Tool 't': triggers must be a list or tuple"):
        Tool("t", "Test.", {"type": "object"}, lambda: "ok", triggers=triggers)
