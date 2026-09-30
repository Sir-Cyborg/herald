"""The ``set_timer`` tool: a countdown that speaks a message when it ends.

This is also the reference example of a tool that uses the :class:`ToolContext`. A parameter
annotated with ``ToolContext`` is filled in by the registry and is invisible to the model. The
tool does its work later through ``ctx.scheduler`` and speaks through ``ctx.say``, so nothing
here knows about the terminal, the TTS engine or the loudspeaker.

The model supplies the message itself, so that the timer speaks in the language and in the
character of the conversation, with no extra LLM call when it rings.

The text returned to the model is phrased as an instruction and does not repeat the message.
Measured on llama3.2:3b, a result such as "Timer set. It will say: X" made the model say X at
once, as if the timer had already rung; telling it to confirm briefly works better.

There is deliberately no tool to list or cancel timers: small models choose worse when offered
many tools. ``Scheduler.pending()`` and ``Scheduler.cancel()`` make both easy to add later.
"""

from __future__ import annotations

from herald.tools import ToolContext, tool

MIN_SECONDS = 1
MAX_SECONDS = 24 * 60 * 60
MAX_MESSAGE_CHARS = 200
DEFAULT_MESSAGE = "Your timer is up."

# The tool is only offered for a message that contains one of these (see ``Tool.triggers``): a 3B
# model asked "What is the capital of France?" otherwise calls it anyway. Matching is a substring
# test that ignores case and accents, so these are stems that inflections still contain.
# Everyday words are avoided ("ora", "hour" in languages where "what time is it" says "hour"):
# a missed trigger only costs a rephrasing, while a common one brings the false calls back.
TRIGGERS = (
    # English ("timer" and "alarm" also cover most other languages)
    "timer",
    "alarm",
    "remind",
    "countdown",
    "wake me",
    "hour",
    "seconds",
    # Minutes: minute(s), minuto/minuti/minutos, minuten, minuta, minuteur
    "minut",
    # Italian. No "ore": "Che ore sono?" asks the time. "secondi", since "secondo me" is everyday.
    "sveglia",  # also svegliami
    "allarm",
    "ricordam",  # ricordami, ricordamelo
    "ricordarm",
    "promemoria",
    "avvis",  # avvisami
    "secondi",
    # Spanish and Portuguese. No "hora(s)": "¿Qué hora es?", "Que horas são?" ask the time.
    "temporizador",
    "recuerdam",  # recuérdame
    "recordatorio",
    "despiert",  # despiértame
    "despert",  # also despertador
    "lembre",  # lembre-me, lembrete
    "acorde",  # me acorde
    "segundos",
    # French. No "heure": "Quelle heure est-il ?" asks the time.
    "rappel",  # rappelle-moi
    "réveill",  # réveille-moi
    "seconde",  # also Dutch seconde(n)
    # German
    "wecker",
    "weck mich",
    "wecke mich",
    "erinnerung",
    "stunde",
    "sekund",  # also Polish sekunda
    # Dutch
    "wekker",
    "herinnering",
    "herinner me",
    "minuut",
    # Polish
    "budzik",
    "przypomnij",
    # Russian
    "таймер",
    "будильник",
    "напомни",
    "минут",
    "секунд",
    # Turkish
    "zamanlay",  # zamanlayıcı
    "dakika",
    "saniye",
)


@tool(triggers=TRIGGERS)
def set_timer(seconds: int, message: str = DEFAULT_MESSAGE, *, ctx: ToolContext) -> str:
    """Start a countdown timer. When it ends, Herald says the message out loud.

    Args:
        seconds: How long to wait, in seconds (1 to 86400). Convert minutes and hours to seconds.
        message: What to say when the timer ends, in the language of the conversation.
    """
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise ValueError(f"seconds must be between {MIN_SECONDS} and {MAX_SECONDS}, got {seconds}")
    message = message.strip()[:MAX_MESSAGE_CHARS].rstrip() or DEFAULT_MESSAGE

    def ring() -> None:
        ctx.say(message)

    ctx.scheduler.schedule(seconds, ring, label=message)
    return (
        f"Done: the timer is running for {_human_duration(seconds)} and will announce itself "
        "when it ends. Now confirm to the user, in one short sentence, that it is set."
    )


def _human_duration(seconds: int) -> str:
    """Spell out a duration for the model: 45 -> "45 seconds", 5400 -> "1 hour 30 minutes"."""
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [
        f"{count} {unit}{'' if count == 1 else 's'}"
        for count, unit in ((hours, "hour"), (minutes, "minute"), (secs, "second"))
        if count
    ]
    return " ".join(parts)
