"""`herald doctor`: every check with something that is fine and something that is not.

Nothing here depends on the machine running the tests: the interpreter, the libraries, the
network and the audio player are all replaced by fakes.
"""

import builtins
import importlib.metadata
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import requests

from herald import __version__, cli, doctor
from herald.config import Settings
from herald.doctor import INFO, OK, PROBLEM, Line

SHOUT_TOOL = """\
from herald.tools import tool


@tool
def shout(text: str) -> str:
    \"\"\"Repeat the text in capitals.\"\"\"
    return text.upper()
"""


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Keep the host's HERALD_* variables out, and use a project root under tmp_path."""
    for name in list(os.environ):
        if name.startswith("HERALD_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HERALD_PROJECT_ROOT", str(tmp_path / "project"))


@pytest.fixture
def settings():
    return Settings.from_env()


def statuses(lines):
    return [line.status for line in lines]


def texts(lines):
    return "\n".join(line.text for line in lines)


# --- System -------------------------------------------------------------------------------------


class TestSystem:
    def test_herald_version(self):
        (line,) = doctor._herald()
        assert line == Line(OK, f"Herald {__version__}")

    @pytest.mark.parametrize("version", [(3, 11, 9), (3, 12, 0)])
    def test_supported_python(self, version, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (*version, "final", 0))
        monkeypatch.setattr(sys, "executable", "/the/python")
        (line,) = doctor._python()
        assert line.status == OK
        assert line.text == "Python {}.{}.{} (/the/python)".format(*version)

    @pytest.mark.parametrize("version", [(3, 10, 14), (3, 13, 1), (2, 7, 18)])
    def test_unsupported_python_says_what_to_install(self, version, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (*version, "final", 0))
        (line,) = doctor._python()
        assert line.status == PROBLEM
        assert f"Python {version[0]}.{version[1]} is not supported" in line.text
        assert "install Python 3.11" in line.fix
        assert "py -3.11" in line.fix and "winget install Python.Python.3.11" in line.fix

    def test_platform_is_information(self):
        import platform

        (line,) = doctor._platform()
        assert line.status == INFO and platform.platform() in line.text

    def test_a_virtual_environment_is_fine(self, monkeypatch):
        monkeypatch.setattr(sys, "prefix", "/project/.venv")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        (line,) = doctor._virtual_environment()
        assert line.status == OK and "/project/.venv" in line.text

    def test_without_one_the_herald_command_may_be_missing(self, monkeypatch):
        monkeypatch.setattr(sys, "prefix", "/usr")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        (line,) = doctor._virtual_environment()
        assert line.status == INFO  # information, not an error: it can be on purpose
        assert "only exists inside the environment" in line.fix
        assert ".venv/bin/herald" in line.fix and ".venv\\Scripts\\herald" in line.fix


# --- Libraries ----------------------------------------------------------------------------------


def fake_torch(*, cuda=False, mps=False, version="2.8.0"):
    torch = ModuleType("torch")
    torch.__version__ = version
    torch.cuda = SimpleNamespace(is_available=lambda: cuda)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    return torch


class TestLibraries:
    @pytest.mark.parametrize(
        ("cuda", "mps", "expected"),
        [
            (True, False, "cuda"),
            (False, True, "mps"),
            (True, True, "cuda, mps"),
            (False, False, "cpu only"),
        ],
    )
    def test_torch_and_its_accelerators(self, cuda, mps, expected, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=cuda, mps=mps))
        (line,) = doctor._torch()
        assert line == Line(OK, f"torch 2.8.0 (accelerator: {expected})")

    def test_a_torch_without_mps_support_does_not_crash(self, monkeypatch):
        torch = fake_torch()
        torch.backends = SimpleNamespace()  # an old build: no mps at all
        monkeypatch.setitem(sys.modules, "torch", torch)
        (line,) = doctor._torch()
        assert line.status == OK and "cpu only" in line.text

    def test_torch_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` fail
        (line,) = doctor._torch()
        assert line.status == PROBLEM
        assert "torch cannot be imported" in line.text and "import of torch" in line.text
        assert "pip install -e ." in line.fix

    def test_torch_that_fails_to_load_is_reported_with_the_reason(self, monkeypatch):
        real_import = builtins.__import__

        def broken_import(name, *args, **kwargs):
            if name == "torch":
                raise OSError("DLL load failed while importing _C")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(builtins, "__import__", broken_import)
        (line,) = doctor._torch()
        assert line.status == PROBLEM and "DLL load failed" in line.text

    def test_coqui_tts_version_comes_from_the_metadata_without_importing_it(self, monkeypatch):
        seen = []

        def version(name):
            seen.append(name)
            return "0.27.5"

        monkeypatch.setattr(importlib.metadata, "version", version)
        monkeypatch.setitem(sys.modules, "TTS", None)  # importing it would fail: it must not be
        (line,) = doctor._coqui_tts()
        assert line == Line(OK, "coqui-tts 0.27.5")
        assert seen == ["coqui-tts"]

    def test_coqui_tts_missing(self, monkeypatch):
        def version(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", version)
        (line,) = doctor._coqui_tts()
        assert line.status == PROBLEM and "coqui-tts is not installed" in line.text
        assert "pip install -e ." in line.fix


# --- Voice files --------------------------------------------------------------------------------


class TestVoiceFiles:
    @pytest.fixture
    def folder(self, tmp_path, monkeypatch):
        folder = tmp_path / "voice"
        folder.mkdir()
        monkeypatch.setenv("HERALD_CHECKPOINT_DIR", str(folder))
        return folder

    def test_everything_is_there(self, folder):
        for name in ("config.json", "vocab.json", "model.pth"):
            (folder / name).write_bytes(b"x")
        lines = doctor._base_voice(Settings.from_env())
        assert statuses(lines) == [INFO, OK, OK, OK]
        assert str(folder) in lines[0].text

    @pytest.mark.parametrize("missing", ["config.json", "vocab.json"])
    def test_a_missing_small_file_is_a_problem_with_the_fix(self, folder, missing):
        for name in ("config.json", "vocab.json", "model.pth"):
            if name != missing:
                (folder / name).write_bytes(b"x")
        lines = doctor._base_voice(Settings.from_env())
        (problem,) = [line for line in lines if line.status == PROBLEM]
        assert problem.text == f"{missing} is missing"
        assert problem.fix == "run: herald download-checkpoints"

    def test_the_model_is_downloaded_on_first_use_so_missing_is_only_information(self, folder):
        for name in ("config.json", "vocab.json"):
            (folder / name).write_bytes(b"x")
        lines = doctor._base_voice(Settings.from_env())
        assert PROBLEM not in statuses(lines)
        assert any(line.status == INFO and "about 2 GB" in line.text for line in lines)

    def test_an_empty_folder_has_two_problems(self, folder):
        lines = doctor._base_voice(Settings.from_env())
        assert statuses(lines).count(PROBLEM) == 2

    def test_the_dataset_is_information_either_way(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        monkeypatch.setenv("HERALD_DATASET_DIR", str(data))
        (line,) = doctor._dataset(Settings.from_env())
        assert line.status == INFO and "none at" in line.text and "--reference-wav" in line.fix
        data.mkdir()
        (data / "metadata.csv").write_text("a.wav|Hi.\n", encoding="utf-8")
        (line,) = doctor._dataset(Settings.from_env())
        assert line == Line(INFO, f"Dataset: {data}")


class TestProfiles:
    def make(self, tmp_path, name, text):
        folder = tmp_path / "project" / "profiles"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.toml").write_text(text, encoding="utf-8")

    def test_counts_the_profiles(self, tmp_path, settings):
        self.make(tmp_path, "a", "")
        self.make(tmp_path, "b", "")
        lines = doctor._profiles(settings, None)
        assert len(lines) == 1 and lines[0].status == INFO
        assert lines[0].text.startswith("Profiles: 2 in ")

    def test_no_profiles_directory_is_not_a_problem(self, settings):
        (line,) = doctor._profiles(settings, None)
        assert line.status == INFO and line.text.startswith("Profiles: 0 in ")

    def test_the_selected_profile_is_checked(self, tmp_path, settings):
        voice = tmp_path / "project" / "models" / "mario"
        voice.mkdir(parents=True)
        (voice / "best_model.pth").write_bytes(b"x")
        self.make(tmp_path, "mario", 'checkpoint = "models/mario"\n')
        lines = doctor._profiles(settings, "mario")
        assert statuses(lines) == [INFO, OK]
        assert "profile mario" in lines[1].text

    def test_a_profile_that_points_to_missing_files(self, tmp_path, settings):
        self.make(tmp_path, "mario", 'checkpoint = "models/mario"\ndataset = "dataset/mario"\n')
        lines = doctor._profiles(settings, "mario")
        problems = [line for line in lines if line.status == PROBLEM]
        assert len(problems) == 2
        assert "checkpoint not found" in problems[0].text
        assert "profile mario" in problems[0].text and "mario.toml" in problems[0].fix

    def test_a_profile_that_cannot_be_loaded(self, settings):
        lines = doctor._profiles(settings, "nope")
        (problem,) = [line for line in lines if line.status == PROBLEM]
        assert "profile 'nope'" in problem.text and "not found" in problem.text
        assert problem.fix == "run: herald profiles"


# --- Chat ---------------------------------------------------------------------------------------


class FakeTags:
    def __init__(self, body=None, error=None):
        self.body, self.error = body, error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def tags(*names):
    return FakeTags({"models": [{"name": name} for name in names]})


@pytest.fixture
def get(monkeypatch):
    """Replace requests.get with a recorder that answers what the test says."""
    calls = SimpleNamespace(args=[], answer=tags())

    def fake_get(url, **kwargs):
        calls.args.append((url, kwargs))
        if isinstance(calls.answer, Exception):
            raise calls.answer
        return calls.answer

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


class TestOllama:
    def test_reachable_and_the_model_is_pulled(self, settings, get):
        get.answer = tags("llama3.2:3b", "mistral:latest")
        lines = doctor._ollama(settings, "llama3.2:3b")
        assert lines == [
            Line(OK, "Ollama at http://localhost:11434 (2 model(s) pulled)"),
            Line(OK, "model 'llama3.2:3b' is pulled"),
        ]
        assert get.args == [("http://localhost:11434/api/tags", {"timeout": 3.0})]

    def test_a_model_name_without_a_tag_means_latest(self, settings, get):
        get.answer = tags("mistral:latest")
        assert doctor._ollama(settings, "mistral")[1].status == OK

    def test_the_model_is_not_pulled(self, settings, get):
        get.answer = tags("mistral:latest")
        lines = doctor._ollama(settings, "llama3.2:3b")
        assert lines[0].status == OK
        assert lines[1] == Line(
            PROBLEM, "model 'llama3.2:3b' is not pulled", "run: ollama pull llama3.2:3b"
        )

    @pytest.mark.parametrize(
        "error",
        [
            requests.ConnectionError("refused"),
            requests.Timeout("slow"),
            requests.HTTPError("500 Server Error"),
        ],
    )
    def test_ollama_cannot_be_reached(self, settings, get, error):
        get.answer = error if not isinstance(error, requests.HTTPError) else FakeTags(error=error)
        (line,) = doctor._ollama(settings, "llama3.2:3b")  # no model check without a server
        assert line.status == PROBLEM
        assert (
            line.text == f"Cannot reach Ollama at http://localhost:11434 ({type(error).__name__})"
        )
        assert line.fix == "start the Ollama app (or run: ollama serve)"

    @pytest.mark.parametrize("body", [{"oops": 1}, ValueError("not json"), [1, 2], None])
    def test_a_server_that_is_not_ollama(self, settings, get, body):
        get.answer = FakeTags(body)
        (line,) = doctor._ollama(settings, "llama3.2:3b")
        assert line.status == PROBLEM and "not like Ollama does" in line.text
        assert "HERALD_OLLAMA_URL" in line.fix

    @pytest.mark.parametrize(
        "url", ["http://box:1234", "http://box:1234/", "http://box:1234/api/chat"]
    )
    def test_the_url_may_be_given_in_the_forms_chat_accepts(self, url, monkeypatch, get):
        monkeypatch.setenv("HERALD_OLLAMA_URL", url)
        get.answer = tags("m:1")
        doctor._ollama(Settings.from_env(), "m:1")
        assert get.args[0][0] == "http://box:1234/api/tags"

    def test_the_model_of_the_profile_is_the_one_checked(self, tmp_path, settings):
        folder = tmp_path / "project" / "profiles"
        folder.mkdir(parents=True)
        (folder / "mario.toml").write_text('ollama_model = "mistral"\n', encoding="utf-8")
        assert doctor._ollama_model(settings, "mario") == "mistral"
        assert doctor._ollama_model(settings, None) == settings.ollama_model
        assert doctor._ollama_model(settings, "nope") == settings.ollama_model  # reported elsewhere


class TestAudioPlayer:
    @pytest.fixture(autouse=True)
    def not_windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")

    def fake_player(self, monkeypatch, *, command):
        import herald.audio_playback as audio

        class FakePlayer:
            available = command is not None

        monkeypatch.setattr(audio, "Player", FakePlayer)
        monkeypatch.setattr(audio, "find_player", lambda: command)

    def test_a_player_was_found(self, monkeypatch):
        self.fake_player(monkeypatch, command=["aplay", "-q"])
        assert doctor._audio_player() == [Line(OK, "Audio player: aplay")]

    def test_no_player(self, monkeypatch):
        self.fake_player(monkeypatch, command=None)
        (line,) = doctor._audio_player()
        assert line.status == PROBLEM and line.text == "No audio player found"
        assert "--save-dir" in line.fix

    def test_windows_needs_no_external_player(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setitem(sys.modules, "winsound", ModuleType("winsound"))
        (line,) = doctor._audio_player()
        assert line.status == OK and "winsound" in line.text

    def test_windows_without_winsound(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setitem(sys.modules, "winsound", None)
        (line,) = doctor._audio_player()
        assert line.status == PROBLEM and "winsound" in line.text


class TestTools:
    def test_the_built_in_tools_load(self, settings):
        (line,) = doctor._tools(settings)
        assert line.status == OK
        assert line.text.startswith("Tools: 1 loaded (") and "set_timer" in line.text

    def test_your_scripts_and_their_errors(self, tmp_path, monkeypatch):
        folder = tmp_path / "mine"
        folder.mkdir()
        (folder / "shout.py").write_text(SHOUT_TOOL, encoding="utf-8")
        (folder / "broken.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
        monkeypatch.setenv("HERALD_TOOLS_DIR", str(folder))
        lines = doctor._tools(Settings.from_env())
        assert lines[0].status == OK and "shout" in lines[0].text and "set_timer" in lines[0].text
        (problem,) = lines[1:]
        assert problem.status == PROBLEM and "boom" in problem.text
        assert "herald tools" in problem.fix

    def test_the_scheduler_is_always_shut_down(self, settings, monkeypatch):
        import herald.tools
        import herald.tools.scheduler

        stopped = []

        class FakeScheduler:
            def shutdown(self):
                stopped.append(True)
                return 0

        def failing(user_dir, context, **kwargs):
            raise RuntimeError("cannot load")

        monkeypatch.setattr(herald.tools.scheduler, "Scheduler", FakeScheduler)
        monkeypatch.setattr(herald.tools, "load_tools", failing)
        with pytest.raises(RuntimeError):
            doctor._tools(settings)
        assert stopped == [True]


# --- the report ---------------------------------------------------------------------------------


def run_with(monkeypatch, sections, settings, profile=None):
    """Run the report with the given sections instead of the real ones; return (code, lines)."""
    monkeypatch.setattr(doctor, "_sections", lambda settings, profile: sections)
    printed = []
    code = doctor.run_doctor(settings, profile, out=printed.append)
    return code, printed


class TestReport:
    def test_all_fine(self, settings, monkeypatch):
        sections = [("First", [("a", lambda: [doctor._ok("fine"), doctor._info("fyi")])])]
        code, printed = run_with(monkeypatch, sections, settings)
        assert code == 0
        assert printed == ["First", "  ok  fine", "  --  fyi", "", "Ready."]

    def test_problems_come_with_their_fix_and_are_counted(self, settings, monkeypatch):
        sections = [
            ("First", [("a", lambda: [doctor._problem("it is broken", "do this")])]),
            (
                "Second",
                [
                    ("b", lambda: [doctor._ok("fine")]),
                    ("c", lambda: [doctor._problem("also broken", "do that")]),
                ],
            ),
        ]
        code, printed = run_with(monkeypatch, sections, settings)
        assert code == 1
        assert printed == [
            "First",
            "  !!  it is broken",
            "      -> do this",
            "",
            "Second",
            "  ok  fine",
            "  !!  also broken",
            "      -> do that",
            "",
            "2 problem(s) found.",
        ]

    def test_information_does_not_make_the_computer_not_ready(self, settings, monkeypatch):
        sections = [("S", [("a", lambda: [doctor._info("note", "a tip")])])]
        code, printed = run_with(monkeypatch, sections, settings)
        assert code == 0 and printed[-1] == "Ready."
        assert "      -> a tip" in printed  # a hint may come with information too

    def test_a_check_that_raises_is_reported_and_the_others_still_run(self, settings, monkeypatch):
        def boom():
            raise RuntimeError("kaboom")

        def silent_boom():
            raise ValueError

        sections = [
            (
                "S",
                [
                    ("first", boom),
                    ("second", lambda: [doctor._ok("still here")]),
                    ("third", silent_boom),
                ],
            )
        ]
        code, printed = run_with(monkeypatch, sections, settings)
        assert code == 1
        assert "  !!  first: kaboom" in printed
        assert "  ok  still here" in printed  # the report went on
        assert "  !!  third: ValueError" in printed  # no message: at least the kind of error
        assert printed[-1] == "2 problem(s) found."

    def test_ctrl_c_inside_a_check_is_not_swallowed(self, settings, monkeypatch):
        """Only errors are findings: Ctrl-C must still stop the command."""

        def interrupted():
            raise KeyboardInterrupt

        sections = [("S", [("a", interrupted)])]
        with pytest.raises(KeyboardInterrupt):
            run_with(monkeypatch, sections, settings)

    def test_the_selected_profile_reaches_the_sections(self, settings, monkeypatch):
        seen = []

        def sections(settings, profile):
            seen.append(profile)
            return []

        monkeypatch.setattr(doctor, "_sections", sections)
        doctor.run_doctor(settings, "mario", out=lambda line: None)
        assert seen == ["mario"]


class TestRealReport:
    """The real checks, with the outside world faked: what a user would see."""

    @pytest.fixture
    def world(self, tmp_path, monkeypatch, get):
        """A computer where everything is fine."""
        import herald.audio_playback as audio

        monkeypatch.setattr(sys, "version_info", (3, 11, 9, "final", 0))
        monkeypatch.setattr(sys, "prefix", "/env")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=True))
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.27.5")
        voice = tmp_path / "voice"
        voice.mkdir()
        for name in ("config.json", "vocab.json", "model.pth"):
            (voice / name).write_bytes(b"x")
        monkeypatch.setenv("HERALD_CHECKPOINT_DIR", str(voice))
        get.answer = tags("llama3.2:3b")
        monkeypatch.setattr(audio, "Player", lambda: SimpleNamespace(available=True))
        monkeypatch.setattr(audio, "find_player", lambda: ["afplay"])
        return get

    def report(self, settings=None, profile=None):
        printed = []
        code = doctor.run_doctor(settings or Settings.from_env(), profile, out=printed.append)
        return code, printed

    def test_a_ready_computer(self, world):
        code, printed = self.report()
        assert code == 0
        assert printed[-1] == "Ready."
        assert [
            line for line in printed if line in ("System", "Libraries", "Voice files", "Chat")
        ] == [
            "System",
            "Libraries",
            "Voice files",
            "Chat",
        ]
        assert "!!" not in "\n".join(printed)

    def test_the_report_is_plain_ascii_in_a_fixed_shape(self, world):
        _, printed = self.report()
        assert "\n".join(printed).isascii()
        for line in printed:
            assert (
                line == ""
                or not line.startswith(" ")  # a section title, or the last line
                or line[:6] in ("  ok  ", "  !!  ", "  --  ")
                or line.startswith("      -> ")
            ), line

    def test_what_goes_wrong_on_a_fresh_windows_machine(self, world, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "version_info", (3, 13, 0, "final", 0))
        monkeypatch.setattr(sys, "base_prefix", sys.prefix)  # no environment
        monkeypatch.setitem(sys.modules, "torch", None)

        def not_installed(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", not_installed)
        world.answer = requests.ConnectionError("refused")
        (tmp_path / "voice" / "config.json").unlink()
        code, printed = self.report()
        out = "\n".join(printed)
        assert code == 1
        assert "Python 3.13 is not supported" in out
        assert "torch cannot be imported" in out and "coqui-tts is not installed" in out
        assert "config.json is missing" in out and "Cannot reach Ollama" in out
        assert "No virtual environment is active" in out  # information: not counted
        assert printed[-1] == "5 problem(s) found."
        # every problem is followed by what to do about it
        for index, line in enumerate(printed):
            if line.startswith("  !!  "):
                assert printed[index + 1].startswith("      -> "), line

    def test_a_profile_given_to_doctor_is_checked(self, world, tmp_path):
        folder = tmp_path / "project" / "profiles"
        folder.mkdir(parents=True)
        (folder / "mario.toml").write_text('checkpoint = "models/gone"\n', encoding="utf-8")
        code, printed = self.report(profile="mario")
        assert code == 1
        assert any("profile mario: checkpoint not found" in line for line in printed)

    def test_nothing_is_loaded_or_downloaded(self, world, monkeypatch):
        """No model, no TTS import, no download: only a GET to Ollama."""
        monkeypatch.setitem(sys.modules, "TTS", None)  # an import of TTS would fail the checks
        from herald.tts import checkpoints, engine

        def forbidden(*args, **kwargs):
            raise AssertionError("doctor must not do this")

        monkeypatch.setattr(checkpoints, "ensure_base_checkpoints", forbidden)
        monkeypatch.setattr(engine, "load_engine", forbidden)
        code, printed = self.report()
        assert code == 0, printed
        assert len(world.args) == 1  # the single request: /api/tags


# --- through the command line -------------------------------------------------------------------


class TestCommand:
    def test_the_command_runs_the_report_and_returns_its_exit_code(
        self, tmp_path, monkeypatch, capsys
    ):
        calls = []

        def fake_run(settings, profile=None, **kwargs):
            calls.append((settings, profile))
            return 1

        monkeypatch.setattr(doctor, "run_doctor", fake_run)
        assert cli.main(["doctor"]) == 1
        assert calls[0][0].project_root == (tmp_path / "project").resolve()
        assert calls[0][1] is None

    def test_profile_option_and_environment(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            doctor, "run_doctor", lambda settings, profile=None: calls.append(profile) or 0
        )
        assert cli.main(["doctor", "--profile", "mario"]) == 0
        monkeypatch.setenv("HERALD_PROFILE", "luigi")
        cli.main(["doctor"])
        cli.main(["doctor", "--profile", "mario"])
        assert calls == ["mario", "luigi", "mario"]

    def test_a_broken_profile_does_not_stop_the_report(self, monkeypatch, capsys):
        """doctor loads the profile itself, so a bad one is a finding, not a crash."""
        monkeypatch.setitem(sys.modules, "torch", fake_torch())
        code = cli.main(["doctor", "--profile", "nope"])
        out = capsys.readouterr().out
        assert code == 1 and "profile 'nope'" in out
        assert out.rstrip().splitlines()[-1].endswith("problem(s) found.")

    def test_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["doctor", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "Check that this computer is ready to run Herald" in out
        assert "--profile" in out
