"""The ``set_timer`` tool, called directly and through a registry. Real timers only where needed."""

import threading

import pytest

from herald.llm.ollama_client import ToolCall
from herald.tools import ToolContext, ToolRegistry, load_tools
from herald.tools.builtin.timer import DEFAULT_MESSAGE, TRIGGERS, _human_duration, set_timer
from herald.tools.scheduler import Scheduler

WAIT = 2.0


class FakeScheduler:
    """Records what the tool schedules, without starting any timer."""

    def __init__(self):
        self.calls = []

    def schedule(self, delay_seconds, callback, *, label=""):
        self.calls.append((delay_seconds, callback, label))


class QuickScheduler(Scheduler):
    """A real scheduler whose timers all fire after 0.05 s, however long they were asked for."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requested = []

    def schedule(self, delay_seconds, callback, *, label=""):
        self.requested.append(delay_seconds)
        return super().schedule(0.05, callback, label=label)


@pytest.fixture
def said():
    return []


@pytest.fixture
def fake(said):
    scheduler = FakeScheduler()
    return ToolContext(say=said.append, scheduler=scheduler)


class TestSetTimer:
    def test_schedules_the_message_and_confirms(self, fake, said):
        result = set_timer(300, "The tea is ready.", ctx=fake)

        ((delay, callback, label),) = fake.scheduler.calls
        assert (delay, label) == (300, "The tea is ready.")
        assert result == (
            "Done: the timer is running for 5 minutes and will announce itself when it ends. "
            "Now confirm to the user, in one short sentence, that it is set."
        )
        assert said == []  # nothing is spoken until the timer rings
        callback()
        assert said == ["The tea is ready."]

    def test_the_message_is_optional(self, fake, said):
        result = set_timer(10, ctx=fake)

        ((_, callback, label),) = fake.scheduler.calls
        callback()
        assert label == DEFAULT_MESSAGE and said == [DEFAULT_MESSAGE]
        assert "10 seconds" in result

    @pytest.mark.parametrize("message", ["", "   ", "\n\t "])
    def test_a_blank_message_means_the_default(self, fake, said, message):
        set_timer(10, message, ctx=fake)
        fake.scheduler.calls[0][1]()
        assert said == [DEFAULT_MESSAGE]

    def test_the_message_is_stripped(self, fake):
        set_timer(10, "  Pasta!\n", ctx=fake)
        assert fake.scheduler.calls[0][2] == "Pasta!"

    def test_a_long_message_is_cut_to_200_characters(self, fake, said):
        set_timer(10, "word " * 100, ctx=fake)
        fake.scheduler.calls[0][1]()
        (spoken,) = said
        assert len(spoken) <= 200
        assert spoken == ("word " * 100)[:200].rstrip()

    def test_the_result_tells_the_model_what_to_do_without_the_message(self, fake):
        result = set_timer(5400, "Pasta is ready!", ctx=fake)

        assert "1 hour 30 minutes" in result
        assert "confirm" in result
        # Repeating the alert text made small models announce it at once, as if it had rung.
        assert "Pasta" not in result

    def test_the_result_stays_short_whatever_the_message(self, fake):
        assert len(set_timer(3600, "x" * 500, ctx=fake)) < 200

    @pytest.mark.parametrize("seconds", [1, 86_400])
    def test_the_limits_are_inclusive(self, fake, seconds):
        set_timer(seconds, ctx=fake)
        assert fake.scheduler.calls[0][0] == seconds

    @pytest.mark.parametrize("seconds", [0, -1, -3600, 86_401, 10**9])
    def test_seconds_outside_the_limits_are_rejected(self, fake, seconds):
        with pytest.raises(ValueError, match=r"between 1 and 86400"):
            set_timer(seconds, ctx=fake)
        assert fake.scheduler.calls == []

    def test_rings_through_a_real_scheduler(self, said):
        scheduler = QuickScheduler()
        rang = threading.Event()
        ctx = ToolContext(say=lambda text: (said.append(text), rang.set()), scheduler=scheduler)
        try:
            set_timer(300, "Time to stretch.", ctx=ctx)

            assert scheduler.requested == [300]
            assert rang.wait(WAIT)
            assert said == ["Time to stretch."]
            assert scheduler.pending() == []
        finally:
            scheduler.shutdown()

    def test_too_many_timers_is_an_error(self, said):
        scheduler = Scheduler(max_tasks=1)
        ctx = ToolContext(say=said.append, scheduler=scheduler)
        try:
            set_timer(600, ctx=ctx)
            with pytest.raises(RuntimeError, match=r"too many pending timers \(max 1\)"):
                set_timer(600, ctx=ctx)
            assert len(scheduler.pending()) == 1
        finally:
            scheduler.shutdown()


class TestHumanDuration:
    @pytest.mark.parametrize(
        ("seconds", "text"),
        [
            (1, "1 second"),
            (45, "45 seconds"),
            (60, "1 minute"),
            (90, "1 minute 30 seconds"),
            (300, "5 minutes"),
            (3600, "1 hour"),
            (5400, "1 hour 30 minutes"),
            (7200, "2 hours"),
            (3661, "1 hour 1 minute 1 second"),
            (86_400, "24 hours"),
        ],
    )
    def test_durations_are_spelled_out(self, seconds, text):
        assert _human_duration(seconds) == text


class TestAsAModelTool:
    def test_the_schema_hides_the_context_and_requires_only_seconds(self):
        spec = set_timer.__herald_tool__
        assert spec.name == "set_timer"
        assert spec.description.startswith("Start a countdown timer.")
        assert spec.parameters["required"] == ["seconds"]
        assert set(spec.parameters["properties"]) == {"seconds", "message"}  # no "ctx"
        assert spec.parameters["properties"]["seconds"]["type"] == "integer"
        assert spec.parameters["properties"]["message"]["type"] == "string"
        assert spec.context_param == "ctx"

    def test_the_parameter_descriptions_come_from_the_docstring(self):
        properties = set_timer.__herald_tool__.parameters["properties"]
        assert "seconds" in properties["seconds"]["description"]
        assert "language" in properties["message"]["description"]

    def test_the_registry_runs_it_with_its_own_context(self, fake, said):
        registry = ToolRegistry(fake)
        registry.register(set_timer.__herald_tool__)

        result = registry.call(ToolCall("set_timer", {"seconds": 90, "message": "Dinner!"}))

        assert "1 minute 30 seconds" in result and "Dinner" not in result
        fake.scheduler.calls[0][1]()
        assert said == ["Dinner!"]

    def test_the_registry_reports_problems_to_the_model(self, fake):
        registry = ToolRegistry(fake)
        registry.register(set_timer.__herald_tool__)

        too_long = registry.call(ToolCall("set_timer", {"seconds": 100_000}))
        no_seconds = registry.call(ToolCall("set_timer", {"message": "hi"}))
        forged = registry.call(ToolCall("set_timer", {"seconds": 5, "ctx": "evil"}))

        assert too_long.startswith("error:") and "between 1 and 86400" in too_long
        assert no_seconds.startswith("error:") and "seconds" in no_seconds
        assert forged.startswith("error:")
        assert fake.scheduler.calls == []

    def test_a_full_scheduler_comes_back_as_an_error_text(self, said):
        scheduler = Scheduler(max_tasks=1)
        registry = ToolRegistry(ToolContext(say=said.append, scheduler=scheduler))
        registry.register(set_timer.__herald_tool__)
        try:
            registry.call(ToolCall("set_timer", {"seconds": 600}))
            result = registry.call(ToolCall("set_timer", {"seconds": 600}))
            assert result.startswith("error:") and "too many pending timers" in result
        finally:
            scheduler.shutdown()

    def test_it_is_a_builtin_tool(self, fake):
        report = load_tools(None, fake)

        assert report.errors == ()
        assert "set_timer" in [loaded.name for loaded in report.tools]
        assert ("set_timer", "builtin") in [(t.name, t.source) for t in report.tools]
        assert "set_timer" in [s["function"]["name"] for s in report.registry.schemas()]


class TestTriggers:
    """The tool is offered only for messages that look like a timer request."""

    @pytest.fixture
    def registry(self, fake):
        registry = ToolRegistry(fake)
        registry.register(set_timer.__herald_tool__)
        return registry

    @pytest.mark.parametrize(
        "message",
        [
            # English
            "Set a timer for 10 seconds",
            "Wake me up in 2 minutes",
            "Remind me to call mom in an hour",
            "Start a countdown of five minutes",
            "Set an alarm for tomorrow",
            # Italian
            "Imposta un timer di 10 secondi",
            "Avvisami tra 1 minuto",
            "Ricordami di spegnere il forno tra 20 minuti",
            "Svegliami tra dieci minuti",
            "Metti la sveglia",
            # Spanish
            "Pon un temporizador de 5 minutos",
            "Recuérdame en 10 segundos que apague el horno",
            "Despiértame en una hora",
            # French
            "Mets un minuteur de 3 minutes",
            "Rappelle-moi dans 10 secondes",
            "Réveille-moi dans une heure",
            # German
            "Stell einen Wecker auf 10 Minuten",
            "Erinnere mich in einer Stunde an die Pizza",
            "Timer auf 30 Sekunden",
            # Portuguese
            "Me lembre em 5 minutos de tirar o bolo",
            "Acorde-me em 10 segundos",
            # Dutch
            "Zet een wekker over 10 minuten",
            "Herinner me over een uur aan de pizza",
            # Polish
            "Ustaw budzik na 5 minut",
            "Przypomnij mi za 10 sekund",
            # Russian
            "Поставь таймер на 5 минут",
            "Напомни мне через час",
            # Turkish
            "5 dakika için zamanlayıcı kur",
            "10 saniye sonra bana haber ver",
            # Case does not matter
            "SET A TIMER FOR TEN SECONDS",
        ],
    )
    def test_timer_requests_reach_the_tool(self, registry, message):
        assert registry.matching(message) == ["set_timer"]

    @pytest.mark.parametrize(
        "message",
        [
            "What is the capital of France?",
            "Tell me a joke.",
            "What time is it?",
            "Come stai oggi?",
            "Quanto fa 12 per 12?",
            "Secondo te, qual è il film migliore?",
            "Che ore sono?",
            "Bonjour, ça va ?",
            "Quelle heure est-il ?",
            "¿Cómo estás?",
            "¿Qué hora es?",
            "Wie geht es dir heute?",
            "Como você está?",
            "Que horas são?",
            "Hoe gaat het met je?",
            "Jak się masz?",
            "Как дела?",
            "Nasılsın?",
            "",
        ],
    )
    def test_ordinary_messages_do_not(self, registry, message):
        assert registry.matching(message) == []

    def test_no_message_means_every_tool(self, registry):
        assert registry.matching(None) == ["set_timer"]

    def test_the_trigger_list_is_clean(self):
        assert set_timer.__herald_tool__.triggers == TRIGGERS
        assert len(TRIGGERS) == len(set(TRIGGERS))
        assert all(trigger == trigger.strip() and len(trigger) >= 3 for trigger in TRIGGERS)
