import sys
import textwrap
from pathlib import Path

import pytest

import herald.tools
from herald.assistant import Assistant
from herald.llm.ollama_client import ChatReply, ToolCall
from herald.tools import LoadedTool, LoadReport, ToolContext, ToolRegistry, load_tools, loader

GOOD = textwrap.dedent(
    '''
    from herald.tools import tool

    @tool
    def greet(name: str) -> str:
        """Greet someone.

        Args:
            name: Who to greet.
        """
        return f"Hello, {name}!"
    '''
)


def script(directory: Path, name: str, source: str = GOOD) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def names(report: LoadReport) -> list[str]:
    return [t.name for t in report.tools]


@pytest.fixture(autouse=True)
def forget_imported_scripts():
    """Loaded scripts and fake packages must not leak from one test into the next."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith(("herald_user_tools", "fake_builtin")):
            del sys.modules[name]


@pytest.fixture
def tools_dir(tmp_path):
    directory = tmp_path / "tools"
    directory.mkdir()
    return directory


# --- user scripts ---------------------------------------------------------------------------------


def test_a_good_script_is_loaded_and_works(tools_dir):
    path = script(tools_dir, "greeter.py")

    report = load_tools(tools_dir, None, builtin=False)

    assert report.errors == ()
    assert report.tools == (LoadedTool("greet", "Greet someone.", str(path)),)
    assert isinstance(report.registry, ToolRegistry)
    assert report.registry.call(ToolCall("greet", {"name": "Frieren"})) == "Hello, Frieren!"
    assert report.registry.schemas()[0]["function"]["parameters"]["properties"] == {
        "name": {"type": "string", "description": "Who to greet."}
    }


def test_scripts_load_in_file_name_order_and_one_script_may_hold_several_tools(tools_dir):
    script(
        tools_dir,
        "b_two.py",
        '''
        from herald.tools import tool

        @tool
        def second() -> str:
            """Second."""
            return "2"

        @tool
        def third() -> str:
            """Third."""
            return "3"
        ''',
    )
    script(tools_dir, "a_one.py", GOOD)

    assert names(load_tools(tools_dir, None, builtin=False)) == ["greet", "second", "third"]


def test_a_module_level_tool_object_is_found(tools_dir):
    script(
        tools_dir,
        "manual.py",
        """
        from herald.tools import Tool

        shout = Tool(
            "shout",
            "Shout a word.",
            {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]},
            lambda word: word.upper(),
        )
        """,
    )
    report = load_tools(tools_dir, None, builtin=False)
    assert report.errors == ()
    assert names(report) == ["shout"]


def test_the_same_tool_reachable_twice_in_a_module_is_registered_once(tools_dir):
    script(
        tools_dir,
        "aliases.py",
        GOOD
        + textwrap.dedent(
            """
            alias = greet
            same_tool = greet.__herald_tool__
            """
        ),
    )
    report = load_tools(tools_dir, None, builtin=False)
    assert names(report) == ["greet"]
    assert report.errors == ()


def test_only_plain_py_files_directly_in_the_directory_are_loaded(tools_dir):
    script(tools_dir, "good.py")
    script(tools_dir, "_helper.py", GOOD.replace("greet", "from_underscore"))
    script(tools_dir, ".hidden.py", GOOD.replace("greet", "from_hidden"))
    script(tools_dir, "notes.txt", GOOD.replace("greet", "from_txt"))
    script(tools_dir, "tool.pyc", GOOD.replace("greet", "from_pyc"))
    (tools_dir / "folder.py").mkdir()  # a directory that looks like a script
    nested = tools_dir / "nested"
    nested.mkdir()
    script(nested, "deep.py", GOOD.replace("greet", "from_nested"))

    report = load_tools(tools_dir, None, builtin=False)

    assert names(report) == ["greet"]
    assert report.errors == ()


@pytest.mark.parametrize("which", ["none", "missing", "file"])
def test_a_missing_tools_directory_is_not_an_error(tmp_path, which):
    not_a_directory = tmp_path / "file.txt"
    not_a_directory.write_text("x")
    user_dir = {"none": None, "missing": tmp_path / "nope", "file": not_a_directory}[which]

    report = load_tools(user_dir, None, builtin=False)

    assert report.tools == ()
    assert report.errors == ()
    assert len(report.registry) == 0


def test_loading_does_not_touch_sys_path(tools_dir):
    script(tools_dir, "greeter.py")
    before = list(sys.path)
    load_tools(tools_dir, None, builtin=False)
    assert sys.path == before


def test_scripts_get_unique_module_names(tools_dir):
    script(tools_dir, "greeter.py")
    load_tools(tools_dir, None, builtin=False)
    assert "herald_user_tools.greeter" in sys.modules


# --- broken scripts -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def broken(:\n    pass\n", "SyntaxError"),
        ("raise RuntimeError('boom at import')\n", "RuntimeError: boom at import"),
        ("import module_that_does_not_exist_anywhere\n", "ModuleNotFoundError"),
        ("import sys\nsys.exit(3)\n", "SystemExit: 3"),
        ("raise SystemExit\n", "SystemExit"),
        ("1 / 0\n", "ZeroDivisionError"),
        (
            "from herald.tools import tool\n\n@tool\ndef silent() -> str:\n    return ''\n",
            "ValueError: Tool function 'silent' needs a docstring",
        ),
        (
            "from herald.tools import tool\n\n@tool\ndef untyped(x):\n    '''Doc.'''\n",
            "TypeError: parameter 'x' of tool function 'untyped' needs a type hint",
        ),
    ],
    ids=[
        "syntax-error",
        "raises",
        "missing-import",
        "sys-exit",
        "system-exit",
        "zero-division",
        "no-docstring",
        "no-type-hint",
    ],
)
def test_a_broken_script_is_reported_and_the_others_still_load(tools_dir, source, expected):
    broken = tools_dir / "a_broken.py"
    broken.write_text(source, encoding="utf-8")
    script(tools_dir, "b_good.py")

    report = load_tools(tools_dir, None, builtin=False)

    assert names(report) == ["greet"]
    (error,) = report.errors
    assert error.startswith(f"{broken}: ")
    assert expected in error
    assert "herald_user_tools.a_broken" not in sys.modules


def test_a_script_that_fails_halfway_registers_nothing_from_itself(tools_dir):
    script(tools_dir, "half.py", GOOD + "\nraise RuntimeError('too late')\n")
    report = load_tools(tools_dir, None, builtin=False)
    assert report.tools == ()
    assert len(report.errors) == 1


def test_ctrl_c_during_an_import_is_not_swallowed(tools_dir):
    script(tools_dir, "stuck.py", "raise KeyboardInterrupt\n")
    with pytest.raises(KeyboardInterrupt):
        load_tools(tools_dir, None, builtin=False)


def test_an_invalid_tool_name_is_reported_with_the_file(tools_dir):
    path = script(tools_dir, "badname.py", GOOD.replace("@tool", "@tool(name='not valid!')"))
    script(tools_dir, "good.py", GOOD.replace("greet", "fine"))

    report = load_tools(tools_dir, None, builtin=False)

    assert names(report) == ["fine"]
    (error,) = report.errors
    assert error.startswith(f"{path}: ") and "Invalid tool name" in error


def test_a_duplicate_name_keeps_the_first_and_names_both_files(tools_dir):
    first = script(tools_dir, "a_first.py", GOOD)
    second = script(tools_dir, "b_second.py", GOOD.replace("Hello", "Hi"))

    report = load_tools(tools_dir, None, builtin=False)

    assert report.tools == (LoadedTool("greet", "Greet someone.", str(first)),)
    assert report.registry.call(ToolCall("greet", {"name": "x"})) == "Hello, x!"
    (error,) = report.errors
    assert str(second) in error and str(first) in error and "'greet'" in error


# --- the context ----------------------------------------------------------------------------------


def test_the_context_reaches_tools_that_ask_for_it(tools_dir):
    script(
        tools_dir,
        "speaker.py",
        '''
        from herald.tools import ToolContext, tool

        @tool
        def announce(text: str, *, ctx: ToolContext) -> str:
            """Say something out loud."""
            ctx.say(text)
            return "said"
        ''',
    )
    said = []
    context = ToolContext(say=said.append, scheduler=None)

    report = load_tools(tools_dir, context, builtin=False)

    assert report.registry.context is context
    assert report.registry.call(ToolCall("announce", {"text": "hello"})) == "said"
    assert said == ["hello"]
    assert "ctx" not in str(report.registry.schemas())


def test_without_a_context_such_a_tool_is_loaded_but_fails_politely(tools_dir):
    script(
        tools_dir,
        "speaker.py",
        '''
        from herald.tools import ToolContext, tool

        @tool
        def announce(text: str, *, ctx: ToolContext) -> str:
            """Say something out loud."""
            return "said"
        ''',
    )
    report = load_tools(tools_dir, None, builtin=False)
    assert names(report) == ["announce"]
    assert report.registry.call(ToolCall("announce", {"text": "hi"})).startswith("error: ")


# --- built-in tools -------------------------------------------------------------------------------


@pytest.fixture
def fake_builtin(tmp_path, monkeypatch):
    """Make `load_tools` read its built-ins from a throwaway package: `make({"mod": source})`."""
    counter = [0]

    def make(modules: dict[str, str], init: str = "") -> str:
        counter[0] += 1
        package = f"fake_builtin_{counter[0]}"
        directory = tmp_path / "site" / package
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text(init, encoding="utf-8")
        for name, source in modules.items():
            script(directory, f"{name}.py", source)
        monkeypatch.syspath_prepend(str(tmp_path / "site"))
        monkeypatch.setattr(loader, "BUILTIN_PACKAGE", package)
        return package

    return make


def test_builtin_modules_are_loaded_first_and_marked_builtin(tools_dir, fake_builtin):
    fake_builtin({"hello": GOOD, "_private": GOOD.replace("greet", "hidden")})
    script(tools_dir, "mine.py", GOOD.replace("greet", "mine"))

    report = load_tools(tools_dir, None)

    assert report.errors == ()
    assert [(t.name, t.source) for t in report.tools] == [
        ("greet", "builtin"),
        ("mine", str(tools_dir / "mine.py")),
    ]


def test_builtin_false_skips_the_builtin_tools(tools_dir, fake_builtin):
    fake_builtin({"hello": GOOD})
    assert load_tools(tools_dir, None, builtin=False).tools == ()


def test_a_broken_builtin_module_is_reported_by_module_name(fake_builtin):
    package = fake_builtin({"a_bad": "raise RuntimeError('oops')\n", "b_good": GOOD})

    report = load_tools(None, None)

    assert names(report) == ["greet"]
    (error,) = report.errors
    assert error == f"{package}.a_bad: RuntimeError: oops"


def test_a_user_script_cannot_replace_a_builtin_tool(tools_dir, fake_builtin):
    package = fake_builtin({"hello": GOOD})
    script(tools_dir, "evil.py", GOOD.replace("Hello", "Gotcha"))

    report = load_tools(tools_dir, None)

    assert [t.source for t in report.tools] == ["builtin"]
    assert report.registry.call(ToolCall("greet", {"name": "x"})) == "Hello, x!"
    (error,) = report.errors
    assert f"{package}.hello" in error and "evil.py" in error


def test_a_builtin_package_without_modules_gives_no_tools(fake_builtin):
    fake_builtin({})
    report = load_tools(None, None)
    assert report.tools == ()
    assert report.errors == ()


def test_a_missing_builtin_package_gives_no_tools(monkeypatch):
    monkeypatch.setattr(loader, "BUILTIN_PACKAGE", "fake_builtin_does_not_exist")
    report = load_tools(None, None)
    assert report.tools == ()
    assert report.errors == ()


def test_a_builtin_package_with_a_missing_dependency_is_reported(fake_builtin):
    fake_builtin({}, init="import a_dependency_nobody_installed\n")
    report = load_tools(None, None)
    assert report.tools == ()
    (error,) = report.errors
    assert "ModuleNotFoundError" in error and "a_dependency_nobody_installed" in error


def test_the_shipped_builtin_tools_load_cleanly():
    report = load_tools(None, None)
    assert report.errors == ()
    assert "set_timer" in names(report)
    assert {t.source for t in report.tools} == {"builtin"}


# --- with the assistant ---------------------------------------------------------------------------


class ScriptedLLM:
    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def chat_messages(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        return self.script.pop(0)


def test_an_assistant_can_use_a_registry_built_by_the_loader(tools_dir):
    script(tools_dir, "greeter.py")
    report = load_tools(tools_dir, None, builtin=False)
    llm = ScriptedLLM(
        ChatReply("", (ToolCall("greet", {"name": "Fern"}),)),
        ChatReply("I greeted Fern."),
    )

    assistant = Assistant(llm, "Be brief.", tools=report.registry)

    assert assistant.respond("Say hi to Fern") == "I greeted Fern."
    assert llm.calls[0][1] == report.registry.schemas()
    tool_message = llm.calls[1][0][-1]
    assert tool_message == {"role": "tool", "tool_name": "greet", "content": "Hello, Fern!"}


# --- package exports ------------------------------------------------------------------------------


def test_the_public_names_are_exported():
    for name in herald.tools.__all__:
        assert hasattr(herald.tools, name), name
    assert set(herald.tools.__all__) >= {
        "tool",
        "Tool",
        "ToolRegistry",
        "ToolContext",
        "load_tools",
        "LoadReport",
        "LoadedTool",
    }


def test_assistant_still_re_exports_the_tool_classes():
    from herald import assistant

    assert assistant.Tool is herald.tools.Tool
    assert assistant.ToolRegistry is herald.tools.ToolRegistry


def test_scheduler_is_available_lazily():
    scheduler_module = pytest.importorskip("herald.tools.scheduler")
    assert herald.tools.Scheduler is scheduler_module.Scheduler
    with pytest.raises(AttributeError):
        herald.tools.DoesNotExist  # noqa: B018
