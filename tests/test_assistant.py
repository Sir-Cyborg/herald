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


def timer_tool(started, **options):
    """A `set_timer` tool that records the seconds it was asked for."""

    def set_timer(seconds: int, message: str = "") -> str:
        started.append(seconds)
        return f"timer set for {seconds} seconds"

    return Tool(
        name="set_timer",
        description="Start a countdown timer.",
        parameters={
            "type": "object",
            "properties": {"seconds": {"type": "integer"}, "message": {"type": "string"}},
            "required": ["seconds"],
        },
        handler=set_timer,
        **options,
    )


@pytest.fixture
def timers(started):
    """A registry with one `set_timer` tool, offered on every message."""
    registry = ToolRegistry()
    registry.register(timer_tool(started))
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


def test_the_model_gets_the_cut_result():
    registry = ToolRegistry()
    registry.register(Tool("dump", "Lots of text.", {"type": "object"}, lambda: "y" * 5000))
    llm = ScriptedLLM(call_tools(("dump", {})), say("ok"))
    Assistant(llm, "Be brief.", tools=registry).respond("Dump")
    (tool_message,) = [m for m in llm.calls[1][0] if m["role"] == "tool"]
    assert tool_message["content"] == "y" * 2000 + "…"


# --- tool calls written as text -----------------------------------------------------------------

SET_TIMER_TEXT = '{"name": "set_timer", "parameters": {"seconds": 60}}'


def test_a_tool_call_written_as_text_is_run_like_a_real_one(timers, started):
    llm = ScriptedLLM(say(SET_TIMER_TEXT), say("Your timer is running."))
    assistant = Assistant(llm, "Be brief.", tools=timers)

    assert assistant.respond("Timer for a minute") == "Your timer is running."

    assert started == [60]
    assert llm.calls[1][0][2:] == [
        {
            "role": "assistant",
            "content": "",  # the JSON text is gone from the transcript
            "tool_calls": [{"function": {"name": "set_timer", "arguments": {"seconds": 60}}}],
        },
        {"role": "tool", "tool_name": "set_timer", "content": "timer set for 60 seconds"},
    ]
    assert assistant.history == [
        user("Timer for a minute"),
        assistant_text("Your timer is running."),
    ]


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "set_timer", "arguments": {"seconds": 60}}',  # "arguments" instead
        '{"type": "function", "name": "set_timer", "parameters": {"seconds": 60}}',
        '  \n{"name": "set_timer", "parameters": {"seconds": 60}}\n  ',
        '```json\n{"name": "set_timer", "parameters": {"seconds": 60}}\n```',
        '```JSON {"name": "set_timer", "parameters": {"seconds": 60}} ```',
        '```\n{"name": "set_timer", "parameters": {"seconds": 60}}\n```',
        '\n```json\n{"name": "set_timer", "parameters": {"seconds": 60}}\n```\n',
    ],
    ids=[
        "arguments-key",
        "with-type",
        "whitespace",
        "json-fence",
        "inline-fence",
        "bare-fence",
        "padded",
    ],
)
def test_the_usual_ways_of_writing_a_call_as_text_are_understood(timers, started, text):
    llm = ScriptedLLM(say(text), say("Done, one minute."))
    assert Assistant(llm, "Be brief.", tools=timers).respond("Timer") == "Done, one minute."
    assert started == [60]


def test_a_call_to_an_unknown_tool_is_never_spoken(timers, started, caplog):
    invented = '{"name": "wikipedia", "parameters": {"query": "elves"}}'
    llm = ScriptedLLM(say(invented))
    assistant = Assistant(llm, "Be brief.", tools=timers)

    with caplog.at_level(logging.WARNING, logger="herald.assistant"):
        text = assistant.respond("Tell me about elves")

    assert text == "Sorry, I could not do that."
    assert started == []
    assert len(llm.calls) == 1
    assert assistant.history == [user("Tell me about elves"), assistant_text(text)]
    assert "wikipedia" in caplog.text


def test_an_unknown_tool_after_a_tool_ran_is_answered_with_done(timers, started):
    llm = ScriptedLLM(
        call_tools(("set_timer", {"seconds": 5})),
        say('{"name": "wikipedia", "parameters": {}}'),
    )
    assistant = Assistant(llm, "Be brief.", tools=timers)

    assert assistant.respond("Timer") == "Done."
    assert started == [5]
    assert assistant.history == [user("Timer"), assistant_text("Done.")]


def test_text_the_model_wrote_before_the_tool_is_still_preferred(timers):
    llm = ScriptedLLM(
        call_tools(("set_timer", {"seconds": 5}), content="Timer started."),
        say('{"name": "wikipedia", "parameters": {}}'),
    )
    assert Assistant(llm, "Be brief.", tools=timers).respond("Timer") == "Timer started."


def test_a_recovered_call_with_bad_arguments_gets_the_error_back_like_any_other(timers, started):
    """The first real-world failure: the model forgot the required `seconds`."""
    llm = ScriptedLLM(
        say('{"name": "set_timer", "parameters": {"message": "Tea is ready"}}'),
        say('{"name": "set_timer", "parameters": {"seconds": 120, "message": "Tea is ready"}}'),
        say("Timer set."),
    )
    assistant = Assistant(llm, "Be brief.", tools=timers)

    assert assistant.respond("Tea timer, 2 minutes") == "Timer set."

    (first_error,) = [m for m in llm.calls[1][0] if m["role"] == "tool"]
    assert first_error["content"] == "error: invalid arguments for set_timer: missing seconds"
    assert started == [120]


def test_a_numeric_string_from_the_model_reaches_the_handler_as_a_number(timers, started):
    """The other real-world failure: `"seconds": "120"`."""
    llm = ScriptedLLM(call_tools(("set_timer", {"seconds": "120"})), say("Two minutes, set."))
    assert Assistant(llm, "Be brief.", tools=timers).respond("Timer") == "Two minutes, set."
    assert started == [120]


@pytest.mark.parametrize(
    "text",
    [
        f"Sure, this is the call: {SET_TIMER_TEXT} and that is all.",
        f"{SET_TIMER_TEXT} Hope that helps!",
        f"Here you go:\n```json\n{SET_TIMER_TEXT}\n```\nAnything else?",
        f"```json\n{SET_TIMER_TEXT}\n```\n```json\n{SET_TIMER_TEXT}\n```",
    ],
    ids=["in-the-middle", "trailing-prose", "fence-inside-prose", "two-fences"],
)
def test_json_inside_normal_prose_is_left_alone(timers, started, text):
    llm = ScriptedLLM(say(text))
    assistant = Assistant(llm, "Be brief.", tools=timers)

    assert assistant.respond("Show me the JSON") == text
    assert started == []
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "set_timer"}',  # no parameters
        '{"name": "set_timer", "parameters": "60"}',  # parameters is not an object
        '{"name": "set_timer", "parameters": [60]}',
        '{"name": 7, "parameters": {}}',  # name is not a string
        '{"parameters": {"seconds": 60}}',  # no name
        '{"name": "set_timer", "parameters": {"seconds": 60}',  # not valid JSON
        '[{"name": "set_timer", "parameters": {"seconds": 60}}]',  # a list, not an object
        '{"city": "Paris", "temperature": 21}',  # JSON, but not a call
        '"just a string"',
        "{}",
        "{",
    ],
)
def test_json_that_is_not_a_tool_call_is_left_alone(timers, started, text):
    llm = ScriptedLLM(say(text))
    assert Assistant(llm, "Be brief.", tools=timers).respond("Hi") == text
    assert started == []


@pytest.mark.parametrize("registry", [None, ToolRegistry()], ids=["no-registry", "empty-registry"])
def test_without_tools_a_json_answer_is_left_alone(registry):
    """In plain chat a user may legitimately get JSON back."""
    text = '{"name": "set_timer", "parameters": {"seconds": 60}}'
    llm = ScriptedLLM(say(text))
    assistant = Assistant(llm, "Be brief.", tools=registry)

    assert assistant.respond("Give me JSON") == text
    assert assistant.history[-1] == assistant_text(text)


def test_the_repair_applies_to_later_replies_too(timers, started):
    llm = ScriptedLLM(
        call_tools(("set_timer", {"seconds": 1})),
        say('{"name": "set_timer", "parameters": {"seconds": 2}}'),
        say("Two timers."),
    )
    assert Assistant(llm, "Be brief.", tools=timers).respond("Two timers") == "Two timers."
    assert started == [1, 2]


def test_recovered_calls_count_toward_the_step_limit(timers, started):
    llm = ScriptedLLM(*[say(SET_TIMER_TEXT)] * 10)
    assistant = Assistant(llm, "Be brief.", tools=timers, max_tool_steps=2)

    text = assistant.respond("Loop")

    assert text == "Sorry, I could not finish that."
    assert started == [60, 60]  # two rounds ran; the third call is not executed
    assert len(llm.calls) == 3
    assert assistant.history == [user("Loop"), assistant_text(text)]


def test_a_call_written_as_text_over_http_reaches_ollama_as_a_proper_tool_call(
    mocker, timers, started
):
    post = mocker.patch(
        "requests.post",
        side_effect=[
            http_reply({"role": "assistant", "content": "```json\n" + SET_TIMER_TEXT + "\n```"}),
            http_reply({"role": "assistant", "content": "Sixty seconds."}),
        ],
    )
    assistant = Assistant(OllamaClient(model="m"), "Be brief.", tools=timers)

    assert assistant.respond("A minute please") == "Sixty seconds."

    second = post.call_args_list[1].kwargs["json"]
    assert second["messages"][-2:] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "set_timer", "arguments": {"seconds": 60}}}],
        },
        {"role": "tool", "tool_name": "set_timer", "content": "timer set for 60 seconds"},
    ]
    assert started == [60]


# --- triggers: which tools the model is offered -------------------------------------------------


@pytest.fixture
def gated(started):
    """`set_timer` only for messages mentioning a timer, next to an always available `clock`."""
    registry = ToolRegistry()
    registry.register(Tool("clock", "The time.", {"type": "object"}, lambda: "12:00"))
    registry.register(timer_tool(started, triggers=("timer", "sveglia")))
    return registry


def offered(llm, call=0):
    """The tool names the model was offered in request number `call`."""
    tools = llm.calls[call][1]
    return None if tools is None else [t["function"]["name"] for t in tools]


@pytest.mark.parametrize(
    "message", ["What is the capital of France?", "Tell me a joke.", "Come stai oggi?"]
)
def test_a_message_without_triggers_is_plain_chat(started, message):
    registry = ToolRegistry()
    registry.register(timer_tool(started, triggers=("timer", "sveglia")))
    llm = ScriptedLLM(say("Paris."))
    assistant = Assistant(llm, "Be brief.", tools=registry)

    assert assistant.respond(message) == "Paris."

    assert llm.calls == [([SYSTEM, user(message)], None)]  # no tools at all
    assert assistant.history == [user(message), assistant_text("Paris.")]


def test_a_message_with_a_trigger_offers_the_tool(started):
    registry = ToolRegistry()
    registry.register(timer_tool(started, triggers=("timer", "sveglia")))
    llm = ScriptedLLM(call_tools(("set_timer", {"seconds": 300})), say("Five minutes, set."))
    assistant = Assistant(llm, "Be brief.", tools=registry)

    assert assistant.respond("Set a TIMER for five minutes") == "Five minutes, set."

    assert llm.calls[0][1] == registry.schemas()
    assert started == [300]


def test_tools_without_triggers_are_offered_with_every_message(gated):
    llm = ScriptedLLM(say("a"), say("b"))
    assistant = Assistant(llm, "Be brief.", tools=gated)
    assistant.respond("Tell me a joke")
    assistant.respond("Metti una sveglia")
    assert offered(llm, 0) == ["clock"]
    assert offered(llm, 1) == ["clock", "set_timer"]


def test_the_offered_tools_are_decided_again_for_each_message(gated):
    """Only the current message counts, not the history (documented limit of triggers)."""
    llm = ScriptedLLM(say("How long?"), say("Okay."))
    assistant = Assistant(llm, "Be brief.", tools=gated)
    assistant.respond("Set a timer")
    assistant.respond("Five minutes")
    assert offered(llm, 0) == ["clock", "set_timer"]
    assert offered(llm, 1) == ["clock"]


def test_a_multi_step_turn_keeps_the_same_offered_tools(gated, started):
    llm = ScriptedLLM(
        call_tools(("clock", {})),
        call_tools(("set_timer", {"seconds": 5})),
        say("All done."),
    )
    assistant = Assistant(llm, "Be brief.", tools=gated)

    assert assistant.respond("Check the time, then set a timer") == "All done."

    assert [offered(llm, i) for i in range(3)] == [["clock", "set_timer"]] * 3
    assert started == [5]


def test_a_multi_step_turn_without_a_trigger_keeps_the_tool_out(gated, started):
    llm = ScriptedLLM(call_tools(("clock", {})), say("It is noon."))
    assert Assistant(llm, "Be brief.", tools=gated).respond("What time is it?") == "It is noon."
    # The same set is offered on the second step: the turn keeps its decision.
    assert [offered(llm, i) for i in range(2)] == [["clock"], ["clock"]]
    assert started == []


def test_a_call_to_a_tool_that_was_not_offered_is_refused(gated, started):
    llm = ScriptedLLM(call_tools(("set_timer", {"seconds": 5})), say("I cannot set timers."))
    assistant = Assistant(llm, "Be brief.", tools=gated)

    assert assistant.respond("Tell me a joke") == "I cannot set timers."

    assert started == []
    assert llm.calls[1][0][-2:] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "set_timer", "arguments": {"seconds": 5}}}],
        },
        {"role": "tool", "tool_name": "set_timer", "content": "error: unknown tool 'set_timer'"},
    ]


def test_with_nothing_offered_a_tool_call_is_refused_too(started):
    registry = ToolRegistry()
    registry.register(timer_tool(started, triggers=("timer",)))
    llm = ScriptedLLM(call_tools(("set_timer", {"seconds": 5})), say("Sorry, no."))

    assert Assistant(llm, "Be brief.", tools=registry).respond("Hello") == "Sorry, no."

    assert [llm.calls[0][1], llm.calls[1][1]] == [None, None]
    assert started == []


def test_use_triggers_false_offers_every_tool_every_time(gated, started):
    llm = ScriptedLLM(call_tools(("set_timer", {"seconds": 5})), say("Timer set."))
    assistant = Assistant(llm, "Be brief.", tools=gated, use_triggers=False)

    assert assistant.respond("Tell me a joke") == "Timer set."

    assert llm.calls[0][1] == gated.schemas()
    assert offered(llm, 0) == ["clock", "set_timer"]
    assert started == [5]


def test_use_triggers_is_on_by_default(gated):
    llm = ScriptedLLM(say("ok"))
    Assistant(llm, "Be brief.", tools=gated).respond("Hello")
    assert offered(llm) == ["clock"]


def test_inline_json_naming_a_tool_that_was_not_offered_is_never_spoken(started):
    registry = ToolRegistry()
    registry.register(timer_tool(started, triggers=("timer",)))
    llm = ScriptedLLM(say('{"name": "set_timer", "parameters": {"seconds": 60}}'))
    assistant = Assistant(llm, "Be brief.", tools=registry)

    text = assistant.respond("What is the capital of France?")

    assert text == "Sorry, I could not do that."
    assert started == []
    assert len(llm.calls) == 1
    assert assistant.history == [user("What is the capital of France?"), assistant_text(text)]


def test_inline_json_for_a_tool_that_was_not_offered_after_a_tool_ran_is_answered_with_done(
    gated, started
):
    llm = ScriptedLLM(
        call_tools(("clock", {})),
        say('{"name": "set_timer", "parameters": {"seconds": 60}}'),
    )
    assert Assistant(llm, "Be brief.", tools=gated).respond("What time is it?") == "Done."
    assert started == []


def test_inline_json_for_an_offered_tool_is_still_run(gated, started):
    llm = ScriptedLLM(
        say('{"name": "set_timer", "parameters": {"seconds": 60}}'),
        say("One minute."),
    )
    assert Assistant(llm, "Be brief.", tools=gated).respond("A timer please") == "One minute."
    assert started == [60]


def test_the_triggers_work_with_the_real_client_over_http(mocker, started):
    registry = ToolRegistry()
    registry.register(timer_tool(started, triggers=("timer",)))
    post = mocker.patch(
        "requests.post",
        side_effect=[
            http_reply({"role": "assistant", "content": "Paris."}),
            http_reply(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "set_timer", "arguments": {"seconds": 9}}}
                    ],
                }
            ),
            http_reply({"role": "assistant", "content": "Nine seconds."}),
        ],
    )
    assistant = Assistant(OllamaClient(model="m"), "Be brief.", tools=registry)

    assistant.respond("What is the capital of France?")
    assistant.respond("Set a timer for nine seconds")

    bodies = [call.kwargs["json"] for call in post.call_args_list]
    assert "tools" not in bodies[0]  # plain chat: nothing was offered
    assert [t["function"]["name"] for t in bodies[1]["tools"]] == ["set_timer"]
    assert [t["function"]["name"] for t in bodies[2]["tools"]] == ["set_timer"]
    assert started == [9]
