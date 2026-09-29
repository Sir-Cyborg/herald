"""Conversation logic: history, the system prompt and the tool-calling loop.

This module knows nothing about the terminal or about speech. The CLI chat loop (and, later, any
other front end) feeds it the user's text and speaks whatever ``Assistant.respond`` returns.

To add a tool, build a :class:`Tool` (a name, a description, a JSON Schema for its arguments and
a plain function that returns a short text for the model) and register it::

    def get_time() -> str:
        return time.strftime("%H:%M")

    registry = ToolRegistry()
    registry.register(
        Tool("get_time", "Current local time.", {"type": "object", "properties": {}}, get_time)
    )
    assistant = Assistant(client, system_prompt, tools=registry)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from herald.errors import OllamaError
from herald.llm.ollama_client import ChatReply, Message, ToolCall

logger = logging.getLogger(__name__)

# Spoken when the model keeps asking for tools and never produces any text.
_GIVE_UP_REPLY = "Sorry, I could not finish that."
# Spoken when tools ran but the model then said nothing: the action happened, so say so.
_DONE_REPLY = "Done."

# Longest tool result (in characters) that is passed back to the model.
_MAX_RESULT_CHARS = 2000

# JSON Schema type names that `_check_arguments` knows how to check.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ChatBackend(Protocol):
    """What :class:`Assistant` needs from an LLM client (``OllamaClient`` provides it)."""

    def chat_messages(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None
    ) -> ChatReply: ...


@dataclass(frozen=True)
class Tool:
    """A function the model may call.

    The handler receives arguments written by the model, so treat them as untrusted input. The
    registry checks them against ``parameters`` first (required names, no unknown names unless
    ``additionalProperties`` is true, and the basic types), but not their values: a handler must
    still validate ranges and must never pass them to a shell, ``eval`` or an arbitrary file path.
    Whatever it returns is cut to 2000 characters before the model sees it.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema: {"type": "object", "properties": {...}, ...}
    handler: Callable[..., str]  # called with the model's arguments as keywords

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
    """The tools an :class:`Assistant` offers to the model. Empty means no tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def __len__(self) -> int:
        return len(self._tools)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def call(self, call: ToolCall) -> str:
        """Run a tool call and return its result text (at most ~2000 characters).

        Never raises: an unknown tool, invalid arguments or a failing handler come back as an
        ``error: ...`` text, which the model can read and recover from.
        """
        text = self._run(call)
        return text if len(text) <= _MAX_RESULT_CHARS else text[:_MAX_RESULT_CHARS] + "…"

    def _run(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return _tool_error(f"unknown tool {call.name!r}")
        problem = _check_arguments(tool.parameters, call.arguments)
        if problem:
            return _tool_error(f"invalid arguments for {call.name}: {problem}")
        try:
            return str(tool.handler(**call.arguments))
        except Exception as exc:  # any handler failure goes back to the model
            return _tool_error(f"{call.name} failed: {type(exc).__name__}: {exc}")


def _tool_error(message: str) -> str:
    logger.warning("Tool call failed: %s", message)
    return f"error: {message}"


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


class Assistant:
    """Keeps the conversation and turns a user message into the text to speak.

    ``history_turns`` is how many past exchanges (user message + final answer) are sent to the
    model with each message; 0 means no memory. ``max_tool_steps`` is how many rounds of tool
    calls are run for one message before giving up, so a confused model cannot loop forever.

    Not thread-safe: use one ``Assistant`` per conversation (a server would need one per client).
    """

    def __init__(
        self,
        llm: ChatBackend,
        system_prompt: str,
        *,
        tools: ToolRegistry | None = None,
        history_turns: int = 10,
        max_tool_steps: int = 4,
    ) -> None:
        self._llm = llm
        self._system: Message = {"role": "system", "content": system_prompt}
        # Not `tools or ToolRegistry()`: an empty registry the caller fills later must be kept.
        self._tools = ToolRegistry() if tools is None else tools
        self._history_turns = history_turns
        self._max_tool_steps = max_tool_steps
        self._history: list[Message] = []

    @property
    def history(self) -> list[Message]:
        """A copy of the remembered messages: user and final assistant texts only."""
        return list(self._history)

    def reset(self) -> None:
        """Forget the conversation."""
        self._history = []

    def respond(self, user_text: str) -> str:
        """Answer ``user_text``, running any tools the model asks for; return the text to speak.

        Exceptions from the LLM client (``OllamaError``) propagate, and the message is then not
        remembered, so the caller can report the error and carry on. The same goes for an empty
        answer when no tool ran. If tools did run, the turn is always kept and answered (with
        "Done." if the model says nothing), because asking again would repeat their effects.
        """
        user: Message = {"role": "user", "content": user_text}
        messages: list[Message] = [self._system, *self._history, user]
        schemas = self._tools.schemas() or None

        reply = self._llm.chat_messages(messages, tools=schemas)
        spoken = reply.content
        ran_tools = False
        for _ in range(self._max_tool_steps):
            if not reply.tool_calls:
                break
            messages = self._run_tools(messages, reply)
            ran_tools = True
            reply = self._llm.chat_messages(messages, tools=schemas)
            spoken = reply.content or spoken
        if reply.tool_calls:
            logger.warning("Giving up after %d rounds of tool calls", self._max_tool_steps)

        if spoken:
            text = spoken
        elif reply.tool_calls:
            text = _GIVE_UP_REPLY
        elif ran_tools:
            text = _DONE_REPLY
        else:
            raise OllamaError("The model returned an empty reply")
        self._remember(user, {"role": "assistant", "content": text})
        return text

    def _run_tools(self, messages: list[Message], reply: ChatReply) -> list[Message]:
        """Return ``messages`` plus the model's tool request and the result of each call.

        These extra messages only live for the current ``respond`` call, not in the history.
        """
        request: Message = {
            "role": "assistant",
            "content": reply.content,
            "tool_calls": [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in reply.tool_calls
            ],
        }
        results: list[Message] = [
            {"role": "tool", "tool_name": call.name, "content": self._tools.call(call)}
            for call in reply.tool_calls
        ]
        return [*messages, request, *results]

    def _remember(self, *messages: Message) -> None:
        turns = self._history_turns
        self._history = [*self._history, *messages][-2 * turns :] if turns > 0 else []
