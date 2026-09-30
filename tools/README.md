# Your tools

Put a Python script in this folder and the assistant can use it. Herald loads every `*.py` file
here when `herald chat` starts; the model decides by itself when to call a tool, fills in its
arguments, and answers the user with the result.

The tools that ship with Herald (today: `set_timer`) live in `src/herald/tools/builtin/`. This
folder is for yours.

## Write a tool in five lines

```python
# tools/time.py
from datetime import datetime
from herald.tools import tool

@tool
def current_time() -> str:
    """Tell the current local date and time. Use it when the user asks what time it is."""
    return datetime.now().strftime("%A %d %B %Y, %H:%M")
```

Then check that it loads, and chat:

```bash
herald tools          # lists every tool Herald can see, and any script that failed to load
herald chat
```

[`_example.py`](_example.py) has two more, plus a template for a tool that acts later. Files
whose name starts with `_` (or `.`) are ignored, so that one does nothing until you copy it.

## What Herald builds from your function

| You write | The model sees |
| --- | --- |
| The function name | The tool name (letters, digits, `_` and `-`) |
| The docstring, up to the `Args:` part | What the tool does. **Write it for the model**: say *when* to use it |
| The type hints (`str`, `int`, `float`, `bool`, `list[str]`, `Literal["a", "b"]`, `X \| None`) | The arguments and their types |
| A default value | The argument is optional |
| The `Args:` lines (`name: meaning`) | What each argument means |
| `@tool(triggers=[...])` | Words that must appear in the user's message for the tool to be offered (see below) |
| The returned string | The result, which the model puts into its answer |

Raise `ValueError("...")` for input you cannot accept: the model gets `error: ...` and can try
again. Keep the returned text short, because it becomes part of what is spoken.

## Offer a tool only when it is needed: `triggers`

**Small models call any tool they are offered.** With `llama3.2:3b`, a timer tool got called for
"what is the capital of France?" and "how are you?" almost every time, and rewording the
description did not help. So Herald can filter *before* the model sees anything:

```python
@tool(triggers=["timer", "alarm", "remind", "minut", "second"])
def set_timer(seconds: int, ...): ...
```

The tool is offered only on messages that contain at least one of those words. Matching is by
substring and ignores case and accents, so short stems catch inflections (`minut` matches
"minutes", "minuti", "minuten"). If nothing matches, the model gets a plain chat, exactly as
without tools.

- A tool **without** `triggers` is offered on every message. Fine for a large model; with a small
  one, expect it to be called at random.
- Pick words someone uses when they really want the tool, in the languages you speak to it, and
  avoid ones that appear in almost every sentence.
- Only the **latest** message is checked. If the assistant asks "how long?", the answer must
  contain a trigger too ("five minutes" does, "five" does not), so include unit words.
- A larger model does not need this: `herald chat --always-offer-tools` ignores the triggers.
- `herald tools` shows the triggers of every tool.

## Acting later, or speaking on your own

Add a keyword-only parameter annotated `ToolContext`. Herald fills it in and hides it from the
model:

```python
from herald.tools import tool, ToolContext

@tool
def remind_me(minutes: int, what: str, *, ctx: ToolContext) -> str:
    """Remind the user of something after a number of minutes."""
    ctx.scheduler.schedule(minutes * 60, lambda: ctx.say(what), label=what)
    return f"I will remind you in {minutes} minutes."
```

- `ctx.say(text)` shows a message and speaks it, from any thread, at any moment.
- `ctx.scheduler.schedule(seconds, function)` runs `function` later on a background thread.

Timers live in memory: when the chat ends, pending ones are cancelled, and Herald tells you.

## Things to keep in mind

- **Your scripts run with your privileges**, once when Herald loads them and again every time
  the model calls them. Only put code you wrote or trust in this folder.
- **The model chooses the arguments, and it can be wrong or be talked into things.** Treat every
  argument as untrusted input: never pass one to a shell (`os.system`, `subprocess` with
  `shell=True`), to `eval`, or use it as a file path without checking it. Prefer tools that do
  something small, harmless and easy to undo.
- **Fewer tools work better.** Small local models get confused when offered many tools, and a
  vague description makes them call the wrong one. Give each tool a clear docstring and its
  `triggers`.
- **A broken script never stops Herald.** It is reported as a warning at startup (and by
  `herald tools`), and the other tools still load.
- **In Docker**, the tools run inside the container, which cannot reach host programs or files
  that are not mounted. This folder is mounted read-only.
- `herald chat --no-tools` starts a plain chat, `--tools-dir DIR` (or `HERALD_TOOLS_DIR`) points
  Herald at another folder.
