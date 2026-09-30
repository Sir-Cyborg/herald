"""Example tools. Herald ignores every file whose name starts with an underscore.

To switch these on, copy this file to a name without the underscore (for example
`tools/basics.py`), then run `herald tools` to check that they load. The next `herald chat`
offers them to the assistant.

A tool is a plain function with `@tool` on top. Herald builds what the model sees from three
things you already write anyway:
  - the type hints  -> which arguments exist and what kind of value each one takes
  - the docstring   -> what the tool does (the model reads this to decide when to call it)
  - the "Args:" part of the docstring -> what each argument means

The string you return is handed back to the model, which then answers the user with it.
Raise ValueError for bad input: the model receives "error: ..." and can correct itself.

`triggers` lists words that must appear in the user's message for the tool to be offered to the
model at all. Small models call any tool they are offered, even for "how are you?", so give
every tool the words someone would use when they really want it (short stems work, because
matching is by substring and ignores case and accents). Leave `triggers` out to always offer it.
"""

import random
from datetime import datetime

from herald.tools import tool


@tool(triggers=["time", "date", "what day", "che ore", "che giorno", "quale giorno"])
def current_time() -> str:
    """Tell the current local date and time. Use it when the user asks what time or day it is."""
    return datetime.now().strftime("%A %d %B %Y, %H:%M")


@tool(triggers=["dice", "roll a", "dado", "tira un"])
def roll_dice(sides: int = 6) -> str:
    """Roll one die and report the result.

    Args:
        sides: How many faces the die has (2 to 1000). Six if the user does not say.
    """
    if not 2 <= sides <= 1000:
        raise ValueError("sides must be between 2 and 1000")
    return f"The die landed on {random.randint(1, sides)} (out of {sides})."


# A tool that has to act LATER, or speak on its own, asks for the context. Put a keyword-only
# parameter annotated with ToolContext at the end; Herald fills it in and hides it from the model:
#
#     from herald.tools import tool, ToolContext
#
#     @tool
#     def remind_me(minutes: int, what: str, *, ctx: ToolContext) -> str:
#         """Remind the user of something after a number of minutes.
#
#         Args:
#             minutes: How many minutes to wait.
#             what: What to remind the user of, in the language of the conversation.
#         """
#         ctx.scheduler.schedule(minutes * 60, lambda: ctx.say(what), label=what)
#         return f"I will remind you in {minutes} minutes."
#
# `ctx.say(text)` shows and speaks a message at any moment; `ctx.scheduler` runs things later.
