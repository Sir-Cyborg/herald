"""Minimal Ollama chat client with explicit timeout and error handling."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import requests

from herald.config import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_TIMEOUT, DEFAULT_OLLAMA_URL
from herald.errors import OllamaError

# One chat message in Ollama's format. Usually {"role": ..., "content": ...}; assistant messages
# may also carry "tool_calls" and tool results use {"role": "tool", "tool_name": ..., ...}.
Message = dict[str, Any]

logger = logging.getLogger(__name__)

_CHAT_PATH = "/api/chat"
DEFAULT_KEEP_ALIVE = "30m"  # Ollama itself unloads an idle model after 5 minutes


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
    """Talks to ``POST <base_url>/api/chat``, optionally streaming the reply.

    It only moves messages: the caller (see ``herald.assistant``) owns the system prompt, the
    history and the decision of what an empty answer means.

    ``keep_alive`` (an Ollama duration such as ``"30m"``, or ``None`` for Ollama's own default) is
    sent with every request so that the model stays loaded between messages.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
        keep_alive: str | None = DEFAULT_KEEP_ALIVE,
    ) -> None:
        base = base_url.rstrip("/")
        # Accept the full endpoint URL too, as older scripts used it.
        self.base_url = base.removesuffix(_CHAT_PATH)
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive

    @property
    def chat_url(self) -> str:
        return self.base_url + _CHAT_PATH

    def chat_messages(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        on_delta: Callable[[str], None] | None = None,
    ) -> ChatReply:
        """Send ``messages`` as they are and parse the answer.

        ``tools`` is an optional list of Ollama function schemas. The reply may hold text, tool
        calls, both or neither; only HTTP and protocol problems raise ``OllamaError``.

        With ``on_delta`` the reply is streamed: the function is called, on the calling thread,
        with each piece of text as soon as it arrives, and the complete ``ChatReply`` is still
        returned at the end. The pieces add up to ``reply.content`` exactly (whitespace around
        the whole text is dropped). If an error is raised half way, some pieces were already
        delivered. Without ``on_delta`` the whole reply is requested at once.
        """
        stream = on_delta is not None
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": stream,
        }
        if tools:
            payload["tools"] = list(tools)
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        resp = self._post(payload, stream=stream)
        try:
            if on_delta is not None:
                return self._read_stream(resp, on_delta)
            try:
                return _parse_reply(resp.json())
            except (ValueError, KeyError, TypeError):
                raise _unexpected(resp.text) from None
        finally:
            resp.close()

    def warm_up(self) -> None:
        """Ask Ollama to load the model now, so that the first real message does not wait for it.

        Meant for a background thread at start-up. Never raises: a failure only means the first
        message is slower, and it is logged at debug level.
        """
        payload: dict[str, Any] = {"model": self.model, "messages": []}
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        try:
            resp = requests.post(self.chat_url, json=payload, timeout=self.timeout)
            try:
                if not resp.ok:
                    logger.debug("Ollama warm-up: %s", self._http_error_message(resp))
            finally:
                resp.close()
        except Exception as exc:  # a warm-up is only an optimisation
            logger.debug("Ollama warm-up failed: %s", exc)

    def _post(self, payload: dict[str, Any], *, stream: bool) -> requests.Response:
        try:
            resp = requests.post(self.chat_url, json=payload, timeout=self.timeout, stream=stream)
        except requests.Timeout:
            raise OllamaError(self._timeout_message()) from None
        except requests.ConnectionError as exc:
            raise OllamaError(
                f"Cannot connect to Ollama at {self.base_url}: is it running (`ollama serve`)? "
                f"({exc})"
            ) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Request to Ollama at {self.base_url} failed: {exc}") from exc

        if not resp.ok:
            try:
                raise OllamaError(self._http_error_message(resp))
            finally:
                resp.close()
        return resp

    def _read_stream(self, resp: requests.Response, on_delta: Callable[[str], None]) -> ChatReply:
        """Read the newline-delimited JSON chunks of a streamed reply until the ``done`` one."""
        trimmed = _TrimmedDeltas(on_delta)
        pieces: list[str] = []
        tool_calls: list[ToolCall] = []
        for chunk in self._chunks(resp):
            if not isinstance(chunk, dict):
                raise _unexpected(chunk)
            if chunk.get("error"):
                raise OllamaError(f"Ollama returned an error: {chunk['error']}")
            done = bool(chunk.get("done"))
            if "message" in chunk or not done:
                try:
                    piece, calls = _parse_message(chunk)
                except (ValueError, KeyError, TypeError):
                    raise _unexpected(chunk) from None
                if piece:
                    pieces.append(piece)
                    trimmed.feed(piece)
                tool_calls.extend(calls)
            if done:
                return ChatReply("".join(pieces).strip(), tuple(tool_calls))
        raise OllamaError("The reply from Ollama was cut off")

    def _chunks(self, resp: requests.Response) -> Iterator[Any]:
        """The decoded JSON lines of a streamed body, with transport errors as ``OllamaError``."""
        try:
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    text = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
                    raise _unexpected(text) from None
        except requests.Timeout:
            raise OllamaError(self._timeout_message()) from None
        except requests.RequestException as exc:
            raise OllamaError(f"The reply from Ollama was cut off: {exc}") from exc

    def _timeout_message(self) -> str:
        return (
            f"Ollama at {self.base_url} did not answer within {self.timeout:g}s "
            "(the model may still be loading; raise HERALD_OLLAMA_TIMEOUT)"
        )

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
    content, tool_calls = _parse_message(data)
    return ChatReply(content.strip(), tool_calls)


def _parse_message(data: Any) -> tuple[str, tuple[ToolCall, ...]]:
    """The raw text and the tool calls of one ``/api/chat`` body or stream chunk."""
    message = data["message"]
    if not isinstance(message, dict):
        raise ValueError("'message' is not an object")
    # `content` can be missing or null when the model only calls tools; never str(None).
    content = str(message.get("content") or "")
    tool_calls = tuple(_parse_tool_call(raw) for raw in message.get("tool_calls") or ())
    return content, tool_calls


def _unexpected(what: Any) -> OllamaError:
    return OllamaError(f"Unexpected response from Ollama (not a chat reply): {str(what)[:200]!r}")


class _TrimmedDeltas:
    """Forwards pieces of text, except the whitespace at the start and end of the whole text.

    Whitespace between two pieces is kept (a space may arrive on its own), so the forwarded
    pieces add up to the final, stripped content.
    """

    def __init__(self, on_delta: Callable[[str], None]) -> None:
        self._on_delta = on_delta
        self._pending = ""  # trailing whitespace, forwarded only if more text follows
        self._started = False

    def feed(self, piece: str) -> None:
        text = self._pending + piece
        if not self._started:
            text = text.lstrip()
        body = text.rstrip()
        self._pending = text[len(body) :]
        if body:
            self._started = True
            self._on_delta(body)


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
