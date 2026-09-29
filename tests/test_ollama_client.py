import json

import pytest
import requests

from herald.errors import OllamaError
from herald.llm.ollama_client import ChatReply, OllamaClient, ToolCall


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


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
    assert kwargs["json"] == {"model": "mistral", "messages": messages, "stream": False}


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
