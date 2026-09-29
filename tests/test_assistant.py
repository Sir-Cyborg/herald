import logging
from types import SimpleNamespace

import pytest

from herald.assistant import Assistant, Tool, ToolRegistry
from herald.errors import OllamaError
from herald.llm.ollama_client import ChatReply, OllamaClient, ToolCall

SYSTEM = {"role": "system", "content": "Be brief."}


def say(text):
    return ChatReply(text)


def call_tools(*calls, content=""):
    return ChatReply(content, tuple(ToolCall(name, args) for name, args in calls))


class ScriptedLLM:
    """A ChatBackend that plays back canned replies (or raises the exceptions in the script)."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []  # (messages, tools) of every request

    def chat_messages(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def user(text):
    return {"role": "user", "content": text}


def assistant_text(text):
    return {"role": "assistant", "content": text}


def http_reply(message):
    """A fake `requests` response carrying one Ollama chat message."""
    return SimpleNamespace(ok=True, text="", json=lambda: {"message": message})


@pytest.fixture
def started():
    """The `seconds` of every timer the `timers` tool has started."""
    return []


@pytest.fixture
def timers(started):
    """A registry with one `set_timer` tool."""

    def set_timer(seconds: int) -> str:
        started.append(seconds)
        return f"timer set for {seconds} seconds"

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="set_timer",
            description="Start a countdown timer.",
            parameters={
                "type": "object",
                "properties": {"seconds": {"type": "integer"}},
                "required": ["seconds"],
            },
            handler=set_timer,
        )
    )
    return registry


# --- plain conversation (no tools) ------------------------------------------------------------


def test_respond_returns_the_reply_and_sends_system_then_user():
    llm = ScriptedLLM(say("Hello."))
    assert Assistant(llm, "Be brief.").respond("Hi") == "Hello."
    assert llm.calls == [([SYSTEM, user("Hi")], None)]


def test_without_tools_none_is_sent_to_the_model():
    llm = ScriptedLLM(say("a"), say("b"))
    assistant = Assistant(llm, "Be brief.", tools=ToolRegistry())
    assistant.respond("one")
    assistant.respond("two")
    assert [tools for _, tools in llm.calls] == [None, None]


def test_history_is_sent_with_the_next_message():
    llm = ScriptedLLM(say("first"), say("second"))
    assistant = Assistant(llm, "Be brief.")
    assistant.respond("one")
    assistant.respond("two")
    assert llm.calls[1][0] == [
        SYSTEM,
        user("one"),
        assistant_text("first"),
        user("two"),
    ]
    assert assistant.history == [
        user("one"),
        assistant_text("first"),
        user("two"),
        assistant_text("second"),
    ]


def test_history_keeps_only_the_last_turns():
    llm = ScriptedLLM(*(say(f"a{i}") for i in range(4)))
    assistant = Assistant(llm, "Be brief.", history_turns=2)
    for i in range(4):
        assistant.respond(f"q{i}")
    assert assistant.history == [user("q2"), assistant_text("a2"), user("q3"), assistant_text("a3")]
    # The model never gets more than 2 past exchanges either.
    assert len(llm.calls[3][0]) == 1 + 2 * 2 + 1


def test_zero_history_turns_means_no_memory():
    llm = ScriptedLLM(say("a"), say("b"))
    assistant = Assistant(llm, "Be brief.", history_turns=0)
    assistant.respond("one")
    assistant.respond("two")
    assert assistant.history == []
    assert llm.calls[1][0] == [SYSTEM, user("two")]


def test_reset_forgets_the_conversation():
    llm = ScriptedLLM(say("a"), say("b"))
    assistant = Assistant(llm, "Be brief.")
    assistant.respond("one")
    assistant.reset()
    assert assistant.history == []
    assistant.respond("two")
    assert llm.calls[1][0] == [SYSTEM, user("two")]


def test_history_property_is_a_copy():
    assistant = Assistant(ScriptedLLM(say("a")), "Be brief.")
    assistant.respond("one")
    assistant.history.clear()
    assert len(assistant.history) == 2


def test_an_llm_error_propagates_and_is_not_remembered():
    llm = ScriptedLLM(say("first"), OllamaError("boom"), say("third"))
    assistant = Assistant(llm, "Be brief.")
    assistant.respond("one")

    with pytest.raises(OllamaError, match="boom"):
        assistant.respond("two")
    assert assistant.history == [user("one"), assistant_text("first")]

    assert assistant.respond("three") == "third"
    assert llm.calls[2][0] == [SYSTEM, user("one"), assistant_text("first"), user("three")]


# --- tool calling -----------------------------------------------------------------------------


def test_tool_call_then_final_answer(timers, started):
    llm = ScriptedLLM(
        call_tools(("set_timer", {"seconds": 60})),
        say("Your timer is running."),
    )
    assistant = Assistant(llm, "Be brief.", tools=timers)

    assert assistant.respond("Timer for a minute") == "Your timer is running."
    assert started == [60]

    (first_messages, first_tools), (second_messages, second_tools) = llm.calls
    assert first_messages == [SYSTEM, user("Timer for a minute")]
    assert first_tools == second_tools == timers.schemas()
    assert second_messages == [
        SYSTEM,
        user("Timer for a minute"),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "set_timer", "arguments": {"seconds": 60}}}],
        },
        {"role": "tool", "tool_name": "set_timer", "content": "timer set for 60 seconds"},
    ]


def test_tool_round_trips_are_not_kept_in_the_history(timers):
    llm = ScriptedLLM(call_tools(("set_timer", {"seconds": 5})), say("Done."), say("Sure."))
    assistant = Assistant(llm, "Be brief.", tools=timers)
    assistant.respond("Timer please")

    assert assistant.history == [user("Timer please"), assistant_text("Done.")]
    assistant.respond("Thanks")
    assert llm.calls[2][0] == [
        SYSTEM,
        user("Timer please"),
        assistant_text("Done."),
        user("Thanks"),
    ]


def test_several_tool_calls_in_one_reply_each_get_a_result(timers):
    llm = ScriptedLLM(
        call_tools(("set_timer", {"seconds": 1}), ("set_timer", {"seconds": 2})),
        say("Two timers."),
    )
    assert Assistant(llm, "Be brief.", tools=timers).respond("Two timers") == "Two timers."

    tool_messages = [m for m in llm.calls[1][0] if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages] == [
        "timer set for 1 seconds",
        "timer set for 2 seconds",
    ]
    assistant_message = llm.calls[1][0][-3]
    assert len(assistant_message["tool_calls"]) == 2


def test_the_model_can_chain_tool_calls(timers, started):
    llm = ScriptedLLM(
        call_tools(("set_timer", {"seconds": 1})),
        call_tools(("set_timer", {"seconds": 2})),
        say("Both set."),
    )
    assert Assistant(llm, "Be brief.", tools=timers).respond("Go") == "Both set."
    assert started == [1, 2]
    assert len(llm.calls) == 3


def test_an_unknown_tool_is_reported_to_the_model(timers):
    llm = ScriptedLLM(call_tools(("launch_rocket", {})), say("I cannot do that."))
    assistant = Assistant(llm, "Be brief.", tools=timers)

    assert assistant.respond("Launch") == "I cannot do that."
    (tool_message,) = [m for m in llm.calls[1][0] if m["role"] == "tool"]
    assert tool_message["tool_name"] == "launch_rocket"
    assert tool_message["content"].startswith("error: unknown tool")


def test_a_failing_handler_is_reported_to_the_model():
    def broken() -> str:
        raise RuntimeError("no clock")

    registry = ToolRegistry()
    registry.register(Tool("get_time", "The time.", {"type": "object"}, broken))
    llm = ScriptedLLM(call_tools(("get_time", {})), say("The clock is broken."))

    assert Assistant(llm, "Be brief.", tools=registry).respond("Time?") == "The clock is broken."
    (tool_message,) = [m for m in llm.calls[1][0] if m["role"] == "tool"]
    assert tool_message["content"].startswith("error:")
    assert "no clock" in tool_message["content"]


def test_the_step_limit_stops_a_model_that_never_answers(timers, started):
    endless = call_tools(("set_timer", {"seconds": 1}))
    llm = ScriptedLLM(*[endless] * 10)
    assistant = Assistant(llm, "Be brief.", tools=timers, max_tool_steps=3)

    text = assistant.respond("Loop")

    assert text == "Sorry, I could not finish that."
    assert started == [1, 1, 1]  # 3 rounds run, the 4th request is dropped
    assert len(llm.calls) == 4
    # The tools did run, so the turn is kept: asking again would start more timers.
    assert assistant.history == [user("Loop"), assistant_text(text)]


def test_the_step_limit_prefers_text_the_model_already_produced(timers):
    llm = ScriptedLLM(
        call_tools(("set_timer", {"seconds": 1}), content="Setting a timer."),
        call_tools(("set_timer", {"seconds": 1})),
    )
    assistant = Assistant(llm, "Be brief.", tools=timers, max_tool_steps=1)
    assert assistant.respond("Go") == "Setting a timer."


def test_zero_tool_steps_never_runs_a_tool(timers, started):
    llm = ScriptedLLM(call_tools(("set_timer", {"seconds": 1})))
    text = Assistant(llm, "Be brief.", tools=timers, max_tool_steps=0).respond("Go")
    assert text == "Sorry, I could not finish that."
    assert started == []
    assert len(llm.calls) == 1


def test_silence_after_a_tool_ran_is_answered_and_remembered(timers, started):
    llm = ScriptedLLM(call_tools(("set_timer", {"seconds": 30})), say(""))
    assistant = Assistant(llm, "Be brief.", tools=timers)

    assert assistant.respond("Timer for 30 seconds") == "Done."
    assert started == [30]
    assert assistant.history == [user("Timer for 30 seconds"), assistant_text("Done.")]


def test_silence_after_a_tool_ran_prefers_text_from_before_the_tool(timers):
    llm = ScriptedLLM(call_tools(("set_timer", {"seconds": 1}), content="Timer started."), say(""))
    assert Assistant(llm, "Be brief.", tools=timers).respond("Go") == "Timer started."


def test_silence_with_no_tool_is_an_error_and_not_remembered():
    llm = ScriptedLLM(say("first"), say(""), say("third"))
    assistant = Assistant(llm, "Be brief.")
    assistant.respond("one")

    with pytest.raises(OllamaError, match="empty reply"):
        assistant.respond("two")
    assert assistant.history == [user("one"), assistant_text("first")]
    assert assistant.respond("three") == "third"


def test_silence_after_a_tool_over_http_does_not_raise_or_lose_the_turn(mocker, timers, started):
    """The review's scenario: the real client and an empty final answer after a tool ran."""
    timer_call = {"function": {"name": "set_timer", "arguments": {"seconds": 9}}}
    mocker.patch(
        "requests.post",
        side_effect=[
            http_reply({"role": "assistant", "content": "", "tool_calls": [timer_call]}),
            http_reply({"role": "assistant", "content": ""}),
        ],
    )
    assistant = Assistant(OllamaClient(model="m"), "Be brief.", tools=timers)

    assert assistant.respond("Nine seconds please") == "Done."
    assert started == [9]
    assert assistant.history == [user("Nine seconds please"), assistant_text("Done.")]


def test_silence_over_http_with_no_tool_raises_a_clean_error(mocker):
    mocker.patch("requests.post", return_value=http_reply({"role": "assistant", "content": ""}))
    assistant = Assistant(OllamaClient(model="m"), "Be brief.")
    with pytest.raises(OllamaError, match="empty reply"):
        assistant.respond("Hello?")
    assert assistant.history == []


def test_an_error_during_the_tool_loop_is_not_remembered(timers):
    llm = ScriptedLLM(call_tools(("set_timer", {"seconds": 1})), OllamaError("boom"))
    assistant = Assistant(llm, "Be brief.", tools=timers)
    with pytest.raises(OllamaError):
        assistant.respond("Go")
    assert assistant.history == []


def test_tools_registered_after_construction_are_offered():
    registry = ToolRegistry()
    llm = ScriptedLLM(say("ok"), say("ok"))
    assistant = Assistant(llm, "Be brief.", tools=registry)
    assistant.respond("one")
    registry.register(Tool("set_timer", "Timer.", {"type": "object"}, lambda seconds: "ok"))
    assistant.respond("two")
    assert llm.calls[0][1] is None
    assert [t["function"]["name"] for t in llm.calls[1][1]] == ["set_timer"]


def test_assistant_works_with_the_real_client_over_http(mocker, timers):
    """The tool round-trip must be in the shape Ollama expects on the wire."""
    timer_call = {"function": {"name": "set_timer", "arguments": {"seconds": 9}}}
    responses = [
        http_reply({"role": "assistant", "content": "", "tool_calls": [timer_call]}),
        http_reply({"role": "assistant", "content": "Nine seconds."}),
    ]
    post = mocker.patch("requests.post", side_effect=responses)
    assistant = Assistant(OllamaClient(model="m"), "Be brief.", tools=timers)

    assert assistant.respond("Nine seconds please") == "Nine seconds."

    first, second = (call.kwargs["json"] for call in post.call_args_list)
    assert first["tools"] == timers.schemas()
    assert second["messages"][-2:] == [
        {"role": "assistant", "content": "", "tool_calls": [timer_call]},
        {"role": "tool", "tool_name": "set_timer", "content": "timer set for 9 seconds"},
    ]


# --- Tool and ToolRegistry --------------------------------------------------------------------


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
        (ToolCall("add", {"a": "1"}), "invalid arguments for add: a must be of type number"),
        (ToolCall("add", {"a": 1, "b": 0}), "add failed: ZeroDivisionError"),  # inside the handler
    ],
    ids=["unknown-tool", "missing", "unknown-argument", "wrong-type", "handler-error"],
)
def test_registry_call_never_raises(caplog, call, expected):
    registry = make_registry(lambda a, b=1: a / b)
    with caplog.at_level(logging.WARNING, logger="herald.assistant"):
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
    return {"type": "object", "properties": {"x": {"type": json_type}}}


@pytest.mark.parametrize(
    ("json_type", "good", "bad"),
    [
        ("string", ["", "a"], [1, None, True, ["a"]]),
        ("integer", [0, 5, -3], [1.5, "1", True, False, None]),
        ("number", [0, 1.5, -2], ["1", True, None]),
        ("boolean", [True, False], [0, 1, "true", None]),
        ("array", [[], [1, "a"]], [{}, "a", None]),
        ("object", [{}, {"k": 1}], [[], "a", None]),
    ],
)
def test_argument_types_are_checked(json_type, good, bad):
    schema = typed(json_type)
    assert all(check(schema, {"x": value}) for value in good)
    assert not any(check(schema, {"x": value}) for value in bad)


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


def test_the_model_gets_the_cut_result():
    registry = ToolRegistry()
    registry.register(Tool("dump", "Lots of text.", {"type": "object"}, lambda: "y" * 5000))
    llm = ScriptedLLM(call_tools(("dump", {})), say("ok"))
    Assistant(llm, "Be brief.", tools=registry).respond("Dump")
    (tool_message,) = [m for m in llm.calls[1][0] if m["role"] == "tool"]
    assert tool_message["content"] == "y" * 2000 + "…"


def test_long_error_texts_are_cut_too():
    def broken() -> str:
        raise RuntimeError("z" * 5000)

    registry = ToolRegistry()
    registry.register(Tool("boom", "Fails.", {"type": "object"}, broken))
    assert len(registry.call(ToolCall("boom", {}))) == 2001
