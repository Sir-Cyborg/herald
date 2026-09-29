"""Minimal Ollama chat client with explicit timeout and error handling."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import requests

from herald.config import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_TIMEOUT, DEFAULT_OLLAMA_URL
from herald.errors import OllamaError

# One chat message in Ollama's format. Usually {"role": ..., "content": ...}; assistant messages
# may also carry "tool_calls" and tool results use {"role": "tool", "tool_name": ..., ...}.
Message = dict[str, Any]

_CHAT_PATH = "/api/chat"


@dataclass(frozen=True)
class ToolCall:
    """The model asks for the function ``name`` to be run with ``arguments``."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatReply:
    """One model answer: text to speak and/or tools to run first."""

    content: str  # may be "": the model only wants to call tools, or (rarely) said nothing
    tool_calls: tuple[ToolCall, ...] = ()


class OllamaClient:
    """Talks to ``POST <base_url>/api/chat`` without streaming.

    It only moves messages: the caller (see ``herald.assistant``) owns the system prompt, the
    history and the decision of what an empty answer means.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
    ) -> None:
        base = base_url.rstrip("/")
        # Accept the full endpoint URL too, as older scripts used it.
        self.base_url = base.removesuffix(_CHAT_PATH)
        self.model = model
        self.timeout = timeout

    @property
    def chat_url(self) -> str:
        return self.base_url + _CHAT_PATH

    def chat_messages(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None
    ) -> ChatReply:
        """Send ``messages`` as they are and parse the answer.

        ``tools`` is an optional list of Ollama function schemas. The reply may hold text, tool
        calls, both or neither; only HTTP and protocol problems raise ``OllamaError``.
        """
        payload: dict[str, Any] = {"model": self.model, "messages": list(messages), "stream": False}
        if tools:
            payload["tools"] = list(tools)
        resp = self._post(payload)

        try:
            return _parse_reply(resp.json())
        except (ValueError, KeyError, TypeError):
            raise OllamaError(
                f"Unexpected response from Ollama (not a chat reply): {resp.text[:200]!r}"
            ) from None

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        try:
            resp = requests.post(self.chat_url, json=payload, timeout=self.timeout)
        except requests.Timeout:
            raise OllamaError(
                f"Ollama at {self.base_url} did not answer within {self.timeout:g}s "
                "(the model may still be loading; raise HERALD_OLLAMA_TIMEOUT)"
            ) from None
        except requests.ConnectionError as exc:
            raise OllamaError(
                f"Cannot connect to Ollama at {self.base_url}: is it running (`ollama serve`)? "
                f"({exc})"
            ) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Request to Ollama at {self.base_url} failed: {exc}") from exc

        if not resp.ok:
            raise OllamaError(self._http_error_message(resp))
        return resp

    def _http_error_message(self, resp: requests.Response) -> str:
        try:
            detail = resp.json().get("error", "")
        except (ValueError, AttributeError):
            detail = ""
        detail = detail or resp.text[:200]
        message = f"Ollama returned HTTP {resp.status_code}: {detail}"
        if resp.status_code == 404:
            message += f" (try `ollama pull {self.model}`)"
        return message


def _parse_reply(data: Any) -> ChatReply:
    """Read a ``/api/chat`` body. Raises ``ValueError``/``KeyError``/``TypeError`` if malformed."""
    message = data["message"]
    if not isinstance(message, dict):
        raise ValueError("'message' is not an object")
    # `content` can be missing or null when the model only calls tools; never str(None).
    content = str(message.get("content") or "").strip()
    tool_calls = tuple(_parse_tool_call(raw) for raw in message.get("tool_calls") or ())
    return ChatReply(content, tool_calls)


def _parse_tool_call(raw: Any) -> ToolCall:
    """Read ``{"function": {"name": ..., "arguments": {...}}}``; JSON-text arguments also work."""
    function = raw["function"]
    name = function["name"]
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):  # some Ollama-compatible servers send JSON text
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError("malformed tool call")
    return ToolCall(name, arguments)
