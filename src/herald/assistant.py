"""Conversation logic: history, the system prompt and the tool-calling loop.

This module knows nothing about the terminal or about speech. The CLI chat loop (and, later, any
other front end) feeds it the user's text and speaks whatever ``Assistant.respond`` returns, or,
to start speaking before the model has finished, whatever it passes to ``on_text`` meanwhile.

Tools are given as a :class:`~herald.tools.registry.ToolRegistry`, usually built by
``herald.tools.load_tools`` from functions marked with ``@tool``. To add one, see ``herald.tools``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from herald.errors import OllamaError
from herald.llm.ollama_client import ChatReply, Message, ToolCall
from herald.tools.registry import Tool, ToolRegistry

__all__ = ["Assistant", "ChatBackend", "Tool", "ToolRegistry"]  # Tool, ToolRegistry: re-exported

logger = logging.getLogger(__name__)

# Spoken when the model keeps asking for tools and never produces any text.
_GIVE_UP_REPLY = "Sorry, I could not finish that."
# Spoken when tools ran but the model then said nothing: the action happened, so say so.
_DONE_REPLY = "Done."
# Spoken when the model wrote a call to a tool that does not exist (see `_inline_tool_call`).
_CANNOT_REPLY = "Sorry, I could not do that."

# A reply that is nothing but a fenced block: ```json ... ```
_FENCED = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class ChatBackend(Protocol):
    """What :class:`Assistant` needs from an LLM client (``OllamaClient`` provides it)."""

    def chat_messages(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        on_delta: Callable[[str], None] | None = None,
    ) -> ChatReply: ...


class Assistant:
    """Keeps the conversation and turns a user message into the text to speak.

    ``history_turns`` is how many past exchanges (user message + final answer) are sent to the
    model with each message; 0 means no memory. ``max_tool_steps`` is how many rounds of tool
    calls are run for one message before giving up, so a confused model cannot loop forever.

    ``use_triggers`` decides which tools the model is offered for a message. A tool with
    ``triggers`` (see :class:`~herald.tools.registry.Tool`) is only offered when the user's message
    contains one of them, and a message that matches none is plain chat, with no tools at all.
    This is deterministic on purpose: small models call any tool they are offered on almost every
    message. The tools offered are fixed once per message and refused afterwards if the model
    calls another one. With ``use_triggers=False`` every tool is offered every time.

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
        use_triggers: bool = True,
    ) -> None:
        self._llm = llm
        self._system: Message = {"role": "system", "content": system_prompt}
        # Not `tools or ToolRegistry()`: an empty registry the caller fills later must be kept.
        self._tools = ToolRegistry() if tools is None else tools
        self._history_turns = history_turns
        self._max_tool_steps = max_tool_steps
        self._use_triggers = use_triggers
        self._history: list[Message] = []

    @property
    def history(self) -> list[Message]:
        """A copy of the remembered messages: user and final assistant texts only."""
        return list(self._history)

    def reset(self) -> None:
        """Forget the conversation."""
        self._history = []

    def respond(self, user_text: str, on_text: Callable[[str], None] | None = None) -> str:
        """Answer ``user_text``, running any tools the model asks for; return the text to speak.

        ``on_text`` receives the answer as it becomes known, in one or several pieces whose
        concatenation is exactly the returned text. For a plain chat message (no tool offered)
        the pieces arrive while the model is still generating, which lets the caller start
        speaking early. When tools are offered the model may answer with a tool call, or with
        text that only looks like one, so nothing is passed on until the final answer is known;
        then ``on_text`` gets it whole, once. A reply that starts like JSON (``{``) or a code
        fence is held back the same way when tools are registered, so a tool call written as text
        is never passed on. ``on_text`` is called on the calling thread. If an error is raised
        half way (``OllamaError``), ``on_text`` may already have received part of the answer:
        the caller should stop whatever it started (cancel the speech). An exception from
        ``on_text`` itself propagates the same way; once the exchange is complete it is
        remembered even if delivering the last piece fails.

        Exceptions from the LLM client (``OllamaError``) propagate, and the message is then not
        remembered, so the caller can report the error and carry on. The same goes for an empty
        answer when no tool ran. If tools did run, the turn is always kept and answered (with
        "Done." if the model says nothing), because asking again would repeat their effects.

        Small models sometimes write a tool call as plain text, e.g.
        ``{"name": "set_timer", "parameters": {...}}``, instead of calling the tool properly. When
        tools are registered such a reply is run like a real call, or, if the tool does not exist
        or was not offered for this message, dropped instead of being read aloud (answering
        "Sorry, I could not do that.").
        """
        user: Message = {"role": "user", "content": user_text}
        messages: list[Message] = [self._system, *self._history, user]
        # Decided once, from the user's message, and kept for every step of this turn.
        trigger_text = user_text if self._use_triggers else None
        offered = frozenset(self._tools.matching(trigger_text))
        schemas = self._tools.schemas(trigger_text) or None

        # Stream only a plain chat turn: with tools on offer the reply may be a call, not speech.
        sink = _TextSink(on_text, hold_json=bool(self._tools)) if on_text is not None else None
        on_delta = sink.feed if sink is not None and not offered else None

        reply, dropped = self._ask(messages, schemas, offered, on_delta)
        spoken = reply.content
        ran_tools = False
        for _ in range(self._max_tool_steps):
            if not reply.tool_calls:
                break
            messages = self._run_tools(messages, reply, offered)
            ran_tools = True
            reply, dropped = self._ask(messages, schemas, offered, on_delta)
            spoken = reply.content or spoken
        if reply.tool_calls:
            logger.warning("Giving up after %d rounds of tool calls", self._max_tool_steps)

        if spoken:
            text = spoken
        elif reply.tool_calls:
            text = _GIVE_UP_REPLY
        elif ran_tools:
            text = _DONE_REPLY
        elif dropped:
            text = _CANNOT_REPLY
        else:
            raise OllamaError("The model returned an empty reply")
        self._remember(user, {"role": "assistant", "content": text})
        if sink is not None:
            sink.finish(text)  # whatever the caller has not received yet, usually all of it or none
        return text

    def _ask(
        self,
        messages: list[Message],
        schemas: list[dict[str, Any]] | None,
        offered: frozenset[str],
        on_delta: Callable[[str], None] | None,
    ) -> tuple[ChatReply, bool]:
        """Ask the model and repair a tool call it wrote as text; see :meth:`respond`.

        Returns the reply and whether a call to an unknown or not offered tool was dropped from it
        (the reply then has no content). Only done when tools are registered: without any, JSON
        is an answer.
        """
        if on_delta is None:  # plain call: backends that cannot stream need not know the keyword
            reply = self._llm.chat_messages(messages, tools=schemas)
        else:
            reply = self._llm.chat_messages(messages, tools=schemas, on_delta=on_delta)
        if reply.tool_calls or not self._tools:
            return reply, False
        call = _inline_tool_call(reply.content)
        if call is None:
            return reply, False
        if call.name in offered:
            logger.debug("Read a tool call for %s written as text", call.name)
            return ChatReply("", (call,)), False
        logger.warning("Dropped a call to the unavailable tool %r written as text", call.name)
        return ChatReply(""), True

    def _run_tools(
        self, messages: list[Message], reply: ChatReply, offered: frozenset[str]
    ) -> list[Message]:
        """Return ``messages`` plus the model's tool request and the result of each call.

        A call to a tool that was not offered for this message is refused like an unknown one.

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
            {"role": "tool", "tool_name": call.name, "content": self._tools.call(call, offered)}
            for call in reply.tool_calls
        ]
        return [*messages, request, *results]

    def _remember(self, *messages: Message) -> None:
        turns = self._history_turns
        self._history = [*self._history, *messages][-2 * turns :] if turns > 0 else []


class _TextSink:
    """Hands text to the caller's ``on_text`` and remembers how much it has handed over.

    ``feed`` takes the pieces of a streamed reply. With ``hold_json`` it keeps back a reply whose
    first visible character is ``{`` or a backtick (a tool call written as text, which must not
    be spoken, see :func:`_inline_tool_call`) and lets any other reply through at once.
    ``finish`` then sends what the caller has not got yet: nothing for a reply that was streamed,
    the whole text for one that was not.
    """

    def __init__(self, on_text: Callable[[str], None], *, hold_json: bool) -> None:
        self._on_text = on_text
        self._hold_json = hold_json
        self._seen = ""  # everything fed before the first visible character
        self._holding: bool | None = None  # None until the first visible character is known
        self.delivered = ""

    def feed(self, piece: str) -> None:
        if self._holding is None:
            self._seen += piece
            visible = self._seen.lstrip()
            if not visible:
                return
            self._holding = self._hold_json and visible[0] in "{`"
            piece, self._seen = self._seen, ""
        if not self._holding:
            self._send(piece)

    def finish(self, text: str) -> None:
        if not text.startswith(self.delivered):
            logger.warning("The streamed text differs from the final answer; not sending it again")
            return
        if len(text) > len(self.delivered):
            self._send(text[len(self.delivered) :])

    def _send(self, piece: str) -> None:
        self._on_text(piece)
        self.delivered += piece


def _inline_tool_call(content: str) -> ToolCall | None:
    """Read a reply that is only ``{"name": ..., "parameters": {...}}`` (optionally in a fence).

    Anything else, such as prose that merely contains JSON, is not a tool call: return ``None``.
    """
    text = content.strip()
    fenced = _FENCED.fullmatch(text)
    if fenced:
        text = fenced[1]
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None
    for key in ("parameters", "arguments"):
        if isinstance(data.get(key), dict):
            return ToolCall(data["name"], data[key])
    return None
