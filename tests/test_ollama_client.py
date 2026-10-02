import logging
import threading

import pytest
import requests

from herald.errors import OllamaError
from herald.llm.ollama_client import ChatReply, OllamaClient, ToolCall
from ollama_fakes import FakeResponse, finished, line, piece, stream_reply


def reply(content="Hello."):
    return FakeResponse(200, {"message": {"role": "assistant", "content": content}})


HI = [{"role": "user", "content": "Hi"}]


@pytest.fixture
def post(mocker):
    return mocker.patch("requests.post")


def test_success_returns_the_reply_text(post):
    post.return_value = reply("  It has been a long time.\n")
    assert OllamaClient().chat_messages(HI) == ChatReply("It has been a long time.")


def test_request_payload(post):
    post.return_value = reply()
    client = OllamaClient("http://ollama:11434/", "mistral", timeout=9)
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Before"},
        {"role": "assistant", "content": "Earlier"},
        {"role": "user", "content": "Now"},
    ]
    client.chat_messages(messages)

    (url,), kwargs = post.call_args
    assert url == "http://ollama:11434/api/chat"
    assert kwargs["timeout"] == 9
    # The messages go out exactly as given: nothing is added, not even a system prompt.
    assert kwargs["json"] == {
        "model": "mistral",
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
    }


def test_messages_are_not_modified(post):
    post.return_value = reply()
    messages = [{"role": "user", "content": "Before"}]
    OllamaClient().chat_messages(messages)
    assert messages == [{"role": "user", "content": "Before"}]


def test_defaults_come_from_the_config(post):
    post.return_value = reply()
    client = OllamaClient()
    client.chat_messages(HI)
    assert post.call_args.args[0] == client.chat_url == "http://localhost:11434/api/chat"
    assert post.call_args.kwargs["json"]["model"] == client.model


@pytest.mark.parametrize(
    "given",
    ["http://localhost:11434", "http://localhost:11434/", "http://localhost:11434/api/chat"],
)
def test_url_forms_all_reach_the_chat_endpoint(post, given):
    post.return_value = reply()
    OllamaClient(base_url=given).chat_messages(HI)
    assert post.call_args.args[0] == "http://localhost:11434/api/chat"


def test_http_error_includes_the_server_message(post):
    post.return_value = FakeResponse(500, {"error": "llama runner crashed"})
    with pytest.raises(OllamaError, match=r"HTTP 500: llama runner crashed"):
        OllamaClient().chat_messages(HI)


def test_missing_model_suggests_pulling_it(post):
    post.return_value = FakeResponse(404, {"error": "model 'nope' not found"})
    with pytest.raises(OllamaError, match=r"not found.*ollama pull nope"):
        OllamaClient(model="nope").chat_messages(HI)


def test_http_error_with_a_non_json_body(post):
    post.return_value = FakeResponse(502, payload=None, text="Bad Gateway")
    with pytest.raises(OllamaError, match=r"HTTP 502: Bad Gateway"):
        OllamaClient().chat_messages(HI)


def test_connection_error(post):
    post.side_effect = requests.ConnectionError("refused")
    with pytest.raises(OllamaError, match=r"Cannot connect to Ollama at http://localhost:11434"):
        OllamaClient().chat_messages(HI)


def test_timeout(post):
    post.side_effect = requests.ReadTimeout("slow")
    with pytest.raises(OllamaError, match=r"did not answer within 5s"):
        OllamaClient(timeout=5).chat_messages(HI)


def test_connect_timeout_is_reported_as_a_timeout(post):
    post.side_effect = requests.ConnectTimeout("slow")
    with pytest.raises(OllamaError, match="did not answer"):
        OllamaClient().chat_messages(HI)


def test_other_request_errors(post):
    post.side_effect = requests.exceptions.InvalidURL("bad url")
    with pytest.raises(OllamaError, match="failed"):
        OllamaClient().chat_messages(HI)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(200, payload=None, text="<html>"),  # not JSON
        FakeResponse(200, {"unexpected": True}),  # no message
        FakeResponse(200, {"message": "just a string"}),  # wrong shape
    ],
    ids=["not-json", "no-message", "wrong-shape"],
)
def test_unexpected_response_shape(post, response):
    post.return_value = response
    with pytest.raises(OllamaError, match="Unexpected response"):
        OllamaClient().chat_messages(HI)


@pytest.mark.parametrize("content", ["", "   ", None], ids=["empty", "blank", "null"])
def test_an_empty_reply_is_returned_not_raised(post, content):
    """Whether silence is an error is the caller's call (it depends on whether tools ran)."""
    post.return_value = FakeResponse(200, {"message": {"role": "assistant", "content": content}})
    assert OllamaClient().chat_messages(HI) == ChatReply("")  # never the word "None"


# --- chat_messages: tools ---------------------------------------------------------------------

TIMER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_timer",
        "description": "Start a timer.",
        "parameters": {"type": "object", "properties": {"seconds": {"type": "integer"}}},
    },
}


def tool_reply(*calls, content=""):
    message = {"role": "assistant", "content": content, "tool_calls": list(calls)}
    return FakeResponse(200, {"message": message})


def function_call(name, arguments):
    return {"function": {"name": name, "arguments": arguments}}


def test_chat_messages_sends_the_tool_schemas(post):
    post.return_value = reply()
    OllamaClient().chat_messages(HI, tools=[TIMER_SCHEMA])
    assert post.call_args.kwargs["json"]["tools"] == [TIMER_SCHEMA]


@pytest.mark.parametrize("tools", [None, []])
def test_chat_messages_without_tools_omits_the_field(post, tools):
    post.return_value = reply()
    OllamaClient().chat_messages(HI, tools=tools)
    assert "tools" not in post.call_args.kwargs["json"]


def test_chat_messages_parses_tool_calls(post):
    post.return_value = tool_reply(
        function_call("set_timer", {"seconds": 60}), function_call("get_time", {})
    )
    result = OllamaClient().chat_messages(HI, tools=[TIMER_SCHEMA])
    assert result == ChatReply(
        "", (ToolCall("set_timer", {"seconds": 60}), ToolCall("get_time", {}))
    )


def test_chat_messages_keeps_text_next_to_tool_calls(post):
    post.return_value = tool_reply(function_call("set_timer", {"seconds": 1}), content=" Sure. ")
    result = OllamaClient().chat_messages(HI)
    assert result.content == "Sure."
    assert result.tool_calls == (ToolCall("set_timer", {"seconds": 1}),)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [('{"seconds": 5}', {"seconds": 5}), ("", {}), (None, {})],
    ids=["json-string", "empty-string", "null"],
)
def test_chat_messages_tolerates_unusual_tool_arguments(post, arguments, expected):
    post.return_value = tool_reply(function_call("set_timer", arguments))
    result = OllamaClient().chat_messages(HI)
    assert result.tool_calls == (ToolCall("set_timer", expected),)


def test_chat_messages_tool_call_without_content_field(post):
    payload = {"message": {"role": "assistant", "tool_calls": [function_call("get_time", {})]}}
    post.return_value = FakeResponse(200, payload)
    result = OllamaClient().chat_messages(HI)
    assert result == ChatReply("", (ToolCall("get_time", {}),))


@pytest.mark.parametrize(
    "tool_calls",
    [
        ["get_time"],  # not an object
        [{"name": "get_time"}],  # missing "function"
        [{"function": {"arguments": {}}}],  # missing name
        [{"function": {"name": "t", "arguments": "{not json"}}],
        [{"function": {"name": "t", "arguments": [1, 2]}}],  # arguments must be an object
        [{"function": {"name": 7, "arguments": {}}}],
    ],
    ids=["not-object", "no-function", "no-name", "bad-json", "list-arguments", "numeric-name"],
)
def test_chat_messages_malformed_tool_calls(post, tool_calls):
    payload = {"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}
    post.return_value = FakeResponse(200, payload)
    with pytest.raises(OllamaError, match="Unexpected response"):
        OllamaClient().chat_messages(HI)


# --- keep_alive -----------------------------------------------------------------------------------


def test_keep_alive_is_sent_with_every_request_by_default(post):
    post.return_value = reply()
    client = OllamaClient()
    assert client.keep_alive == "30m"
    client.chat_messages(HI)
    assert post.call_args.kwargs["json"]["keep_alive"] == "30m"


def test_keep_alive_can_be_changed_or_left_to_ollama(post):
    post.return_value = reply()
    OllamaClient(keep_alive="1h").chat_messages(HI)
    assert post.call_args.kwargs["json"]["keep_alive"] == "1h"

    OllamaClient(keep_alive=None).chat_messages(HI)
    assert "keep_alive" not in post.call_args.kwargs["json"]


def test_the_constructor_stays_compatible_with_the_positional_form(post):
    post.return_value = reply()
    client = OllamaClient("http://ollama:11434", "mistral", 9)
    assert (client.base_url, client.model, client.timeout) == (
        "http://ollama:11434",
        "mistral",
        9,
    )


def test_a_non_streamed_request_asks_for_the_whole_reply_at_once(post):
    post.return_value = reply()
    OllamaClient().chat_messages(HI)
    assert post.call_args.kwargs["json"]["stream"] is False
    assert post.call_args.kwargs["stream"] is False


def test_the_response_is_closed_after_a_normal_reply(post):
    response = reply()
    post.return_value = response
    OllamaClient().chat_messages(HI)
    assert response.closed


def test_the_response_is_closed_after_a_bad_reply(post):
    response = FakeResponse(200, {"unexpected": True})
    post.return_value = response
    with pytest.raises(OllamaError):
        OllamaClient().chat_messages(HI)
    assert response.closed


def test_the_response_is_closed_after_an_http_error(post):
    response = FakeResponse(500, {"error": "boom"})
    post.return_value = response
    with pytest.raises(OllamaError, match="HTTP 500"):
        OllamaClient().chat_messages(HI)
    assert response.closed


# --- warm_up --------------------------------------------------------------------------------------


def test_warm_up_asks_ollama_to_load_the_model_without_generating(post):
    response = FakeResponse(200, {"done": True, "done_reason": "load"})
    post.return_value = response

    OllamaClient("http://ollama:11434", "mistral", timeout=7).warm_up()

    (url,), kwargs = post.call_args
    assert url == "http://ollama:11434/api/chat"
    assert kwargs["json"] == {"model": "mistral", "messages": [], "keep_alive": "30m"}
    assert kwargs["timeout"] == 7
    assert response.closed


def test_warm_up_without_keep_alive_leaves_it_out(post):
    post.return_value = FakeResponse(200, {"done": True})
    OllamaClient(model="m", keep_alive=None).warm_up()
    assert post.call_args.kwargs["json"] == {"model": "m", "messages": []}


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("refused"),
        requests.ReadTimeout("slow"),
        requests.exceptions.InvalidURL("bad url"),
        RuntimeError("anything at all"),
        ValueError("even this"),
    ],
    ids=["connection", "timeout", "invalid-url", "runtime", "value"],
)
def test_warm_up_never_raises(post, caplog, failure):
    post.side_effect = failure
    with caplog.at_level(logging.DEBUG, logger="herald.llm.ollama_client"):
        OllamaClient().warm_up()
    assert [r.levelno for r in caplog.records] == [logging.DEBUG]


def test_warm_up_survives_an_http_error_and_logs_it_quietly(post, caplog):
    response = FakeResponse(404, {"error": "model 'nope' not found"})
    post.return_value = response
    with caplog.at_level(logging.DEBUG, logger="herald.llm.ollama_client"):
        OllamaClient(model="nope").warm_up()
    assert [r.levelno for r in caplog.records] == [logging.DEBUG]
    assert "ollama pull nope" in caplog.text
    assert response.closed


# --- streaming ------------------------------------------------------------------------------------


def stream(post, *lines):
    response = stream_reply(list(lines))
    post.return_value = response
    return response


def collect(post_lines, post, **client_options):
    """Run a streamed chat over `post_lines`; return (reply, pieces, response)."""
    response = stream(post, *post_lines)
    pieces = []
    result = OllamaClient(**client_options).chat_messages(HI, on_delta=pieces.append)
    return result, pieces, response


def test_a_streamed_request_asks_for_a_stream_and_keeps_the_connection_open(post):
    collect([finished()], post)
    assert post.call_args.kwargs["json"]["stream"] is True
    assert post.call_args.kwargs["stream"] is True
    assert post.call_args.kwargs["timeout"] == 120.0  # applies between chunks


def test_a_streamed_request_has_the_same_body_as_a_plain_one_but_for_stream(post):
    post.return_value = reply()
    OllamaClient().chat_messages(HI, tools=[TIMER_SCHEMA])
    plain = post.call_args.kwargs["json"]

    collect([finished()], post)
    streamed = {**post.call_args.kwargs["json"]}
    # (the streamed call above had no tools; compare the common part)
    assert streamed == {**{k: v for k, v in plain.items() if k != "tools"}, "stream": True}

    stream(post, finished())
    OllamaClient().chat_messages(HI, tools=[TIMER_SCHEMA], on_delta=lambda piece: None)
    assert post.call_args.kwargs["json"] == {**plain, "stream": True}


def test_the_pieces_are_delivered_in_order_and_the_reply_is_the_whole_text(post):
    result, pieces, response = collect(
        [piece("It has"), piece(" been"), piece(" a long"), piece(" time."), finished()], post
    )
    assert pieces == ["It has", " been", " a long", " time."]
    assert result == ChatReply("It has been a long time.")
    assert "".join(pieces) == result.content
    assert response.closed


def test_a_piece_is_delivered_before_the_next_one_is_read(post):
    """The point of streaming: the caller can act while the reply is still being generated."""
    events = []

    def lines():
        for text in ("One", " two", " three"):
            events.append(f"read {text!r}")
            yield piece(text)
        events.append("read done")
        yield finished()

    post.return_value = stream_reply(lines())
    result = OllamaClient().chat_messages(
        HI, on_delta=lambda text: events.append(f"delta {text!r}")
    )
    events.append("returned")

    assert events == [
        "read 'One'",
        "delta 'One'",
        "read ' two'",
        "delta ' two'",
        "read ' three'",
        "delta ' three'",
        "read done",
        "returned",
    ]
    assert result.content == "One two three"


def test_pieces_are_delivered_on_the_calling_thread(post):
    stream(post, piece("Hi"), finished())
    threads = []

    def remember_thread(text):
        threads.append(threading.current_thread())

    OllamaClient().chat_messages(HI, on_delta=remember_thread)

    assert threads == [threading.current_thread()]


def test_whitespace_around_the_whole_text_is_dropped_but_not_inside_it(post):
    lines = [
        piece("\n"),
        piece("  Hello"),
        piece(" "),
        piece("big"),
        piece(" world!"),
        piece("\n"),
        piece(" \n"),
        finished(),
    ]
    result, pieces, _ = collect(lines, post)

    assert result.content == "Hello big world!"
    assert pieces == ["Hello", " big", " world!"]
    assert "".join(pieces) == result.content


def test_whitespace_that_is_followed_by_more_text_is_delivered_with_it(post):
    result, pieces, _ = collect([piece("a"), piece("\n\n"), piece("b"), finished()], post)
    assert pieces == ["a", "\n\nb"]
    assert result.content == "a\n\nb"


def test_empty_pieces_and_blank_lines_are_skipped(post):
    result, pieces, _ = collect([b"", piece(""), piece("Hi"), b"", b"  ", finished()], post)
    assert pieces == ["Hi"]
    assert result.content == "Hi"


def test_a_stream_with_no_text_is_an_empty_reply_for_the_caller_to_judge(post):
    result, pieces, _ = collect([finished()], post)
    assert result == ChatReply("")
    assert pieces == []


def test_text_in_the_final_chunk_is_not_lost(post):
    last = line(message={"role": "assistant", "content": "end"}, done=True)
    result, pieces, _ = collect([piece("the "), last], post)
    assert pieces == ["the", " end"]  # the space waits for the text that follows it
    assert result.content == "the end"


def test_the_final_chunk_may_have_no_message_at_all(post):
    result, pieces, _ = collect([piece("Hi"), line(done=True)], post)
    assert (result.content, pieces) == ("Hi", ["Hi"])


def test_lines_after_the_done_chunk_are_not_read(post):
    result, pieces, _ = collect([piece("Hi"), finished(), piece("ignored")], post)
    assert result.content == "Hi"


def test_tool_calls_in_any_chunk_are_collected(post):
    call_one = line(
        message={"role": "assistant", "content": "", "tool_calls": [function_call("a", {"x": 1})]},
        done=False,
    )
    call_two = line(
        message={"role": "assistant", "content": "", "tool_calls": [function_call("b", {})]},
        done=False,
    )
    result, pieces, _ = collect(
        [piece("Let me"), call_one, piece(" see."), call_two, finished()], post
    )

    assert result == ChatReply("Let me see.", (ToolCall("a", {"x": 1}), ToolCall("b", {})))
    assert pieces == ["Let me", " see."]


def test_a_tool_call_only_stream_delivers_no_text(post):
    chunk = line(
        message={"role": "assistant", "content": "", "tool_calls": [function_call("a", "{}")]},
        done=False,
    )
    result, pieces, _ = collect([chunk, finished()], post)
    assert result == ChatReply("", (ToolCall("a", {}),))
    assert pieces == []


def test_an_error_line_becomes_an_ollama_error_with_its_text(post):
    response = stream(post, piece("Par"), line(error="model runner has unexpectedly stopped"))
    pieces = []
    with pytest.raises(OllamaError, match=r"returned an error: model runner has unexpectedly"):
        OllamaClient().chat_messages(HI, on_delta=pieces.append)
    assert pieces == ["Par"]  # what arrived before the error was delivered
    assert response.closed


def test_a_stream_that_ends_without_done_is_cut_off(post):
    response = stream(post, piece("Hello"), piece(" wor"))
    pieces = []
    with pytest.raises(OllamaError, match="The reply from Ollama was cut off"):
        OllamaClient().chat_messages(HI, on_delta=pieces.append)
    assert pieces == ["Hello", " wor"]
    assert response.closed


def test_an_empty_stream_is_cut_off_too(post):
    stream(post)
    with pytest.raises(OllamaError, match="cut off"):
        OllamaClient().chat_messages(HI, on_delta=lambda text: None)


@pytest.mark.parametrize(
    "bad_line",
    [
        b"<html>Bad gateway</html>",  # not JSON
        b'{"message": {"content": "x"',  # truncated JSON
        b"[1, 2]",  # JSON, but not an object
        b'"just a string"',
        line(message="just a string", done=False),
        line(unexpected=True),  # no message and not done
        line(
            message={"content": "", "tool_calls": ["not-an-object"]},
            done=False,
        ),
        line(
            message={"content": "", "tool_calls": [{"function": {"name": "t", "arguments": "{x"}}]},
            done=False,
        ),
    ],
    ids=[
        "html",
        "truncated-json",
        "list",
        "string",
        "string-message",
        "no-message",
        "bad-tool-call",
        "bad-arguments",
    ],
)
def test_a_malformed_line_is_an_unexpected_response(post, bad_line):
    response = stream(post, piece("ok"), bad_line, finished())
    with pytest.raises(OllamaError, match="Unexpected response from Ollama"):
        OllamaClient().chat_messages(HI, on_delta=lambda text: None)
    assert response.closed


def test_http_errors_are_reported_the_same_way_when_streaming(post):
    response = FakeResponse(404, {"error": "model 'nope' not found"})
    post.return_value = response
    with pytest.raises(OllamaError, match=r"not found.*ollama pull nope"):
        OllamaClient(model="nope").chat_messages(HI, on_delta=lambda text: None)
    assert response.closed


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (
            requests.ConnectionError("refused"),
            r"Cannot connect to Ollama at http://localhost:11434",
        ),
        (requests.ReadTimeout("slow"), r"did not answer within 5s"),
        (requests.ConnectTimeout("slow"), r"did not answer within 5s"),
        (requests.exceptions.InvalidURL("bad url"), r"failed"),
    ],
    ids=["connection", "read-timeout", "connect-timeout", "other"],
)
def test_connection_problems_are_reported_the_same_way_when_streaming(post, failure, message):
    post.side_effect = failure
    with pytest.raises(OllamaError, match=message):
        OllamaClient(timeout=5).chat_messages(HI, on_delta=lambda text: None)


def failing_after(first_lines, failure):
    def lines():
        yield from first_lines
        raise failure

    return lines()


def test_a_timeout_between_chunks_is_reported_as_a_timeout(post):
    response = stream_reply(failing_after([piece("Hel")], requests.ReadTimeout("slow")))
    post.return_value = response
    pieces = []
    with pytest.raises(OllamaError, match=r"did not answer within 5s"):
        OllamaClient(timeout=5).chat_messages(HI, on_delta=pieces.append)
    assert pieces == ["Hel"]
    assert response.closed


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("Connection broken: IncompleteRead"),
        requests.exceptions.ChunkedEncodingError("Connection broken"),
        requests.exceptions.ContentDecodingError("bad gzip"),
    ],
    ids=["connection", "chunked", "decoding"],
)
def test_a_connection_that_drops_mid_stream_is_cut_off(post, failure):
    response = stream_reply(failing_after([piece("Hel")], failure))
    post.return_value = response
    with pytest.raises(OllamaError, match="The reply from Ollama was cut off"):
        OllamaClient().chat_messages(HI, on_delta=lambda text: None)
    assert response.closed


def test_an_exception_from_the_callback_propagates_unchanged_and_closes_the_response(post):
    response = stream(post, piece("Hi"), piece(" there"), finished())
    seen = []

    def boom(text):
        seen.append(text)
        raise RuntimeError("the speaker broke")

    with pytest.raises(RuntimeError, match="the speaker broke"):
        OllamaClient().chat_messages(HI, on_delta=boom)
    assert seen == ["Hi"]  # nothing is read after the failure
    assert response.closed


def test_streaming_with_keep_alive_none_omits_it(post):
    collect([finished()], post, keep_alive=None)
    assert "keep_alive" not in post.call_args.kwargs["json"]
