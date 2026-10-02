"""Voice profiles: reading, validating, listing, checking and writing TOML profile files."""

import tomllib
from pathlib import Path

import pytest

from herald import profiles
from herald.errors import HeraldError, ProfileError
from herald.profiles import (
    PROFILE_KEYS,
    Profile,
    check_profile,
    ensure_profile,
    list_profiles,
    load_profile,
    profile_defaults,
    relative_to_root,
    write_profile,
)

FULL_PROFILE = """\
description = "Frieren, the elf mage"
checkpoint = "models/frieren"
dataset = "dataset/frieren"
reference_wavs = ["clips/a.wav", "clips/b.wav"]
num_references = 4
language = "en"
temperature = 0.8
device = "cpu"
system_prompt = "You are Frieren."
ollama_model = "llama3.2:3b"
history = 6
speaker_name = "frieren"
"""


@pytest.fixture
def root(tmp_path) -> Path:
    path = tmp_path / "project"
    (path / "profiles").mkdir(parents=True)
    return path


@pytest.fixture
def profiles_dir(root) -> Path:
    return root / "profiles"


def put(profiles_dir: Path, name: str, text: str) -> Path:
    path = profiles_dir / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def load(profiles_dir: Path, root: Path, text: str, name: str = "voice") -> Profile:
    put(profiles_dir, name, text)
    return load_profile(name, profiles_dir, root)


class TestLoadProfile:
    def test_a_full_profile(self, root, profiles_dir):
        profile = load(profiles_dir, root, FULL_PROFILE)

        assert profile.name == "voice"
        assert profile.path == profiles_dir / "voice.toml"
        assert profile.description == "Frieren, the elf mage"
        values = profile.values
        assert values["checkpoint"] == root / "models" / "frieren"
        assert values["dataset"] == root / "dataset" / "frieren"
        assert values["reference_wavs"] == [root / "clips" / "a.wav", root / "clips" / "b.wav"]
        assert (values["num_references"], values["history"]) == (4, 6)
        assert (values["language"], values["device"], values["speaker_name"]) == (
            "en",
            "cpu",
            "frieren",
        )
        assert (values["temperature"], values["ollama_model"]) == (0.8, "llama3.2:3b")
        assert values["system_prompt"] == "You are Frieren."
        assert set(values) == set(PROFILE_KEYS) - {"system_prompt_file"}

    def test_every_key_is_optional(self, root, profiles_dir):
        profile = load(profiles_dir, root, "# nothing here\n")
        assert dict(profile.values) == {}
        assert profile.description == ""

    def test_the_values_cannot_be_changed(self, root, profiles_dir):
        profile = load(profiles_dir, root, FULL_PROFILE)
        with pytest.raises(TypeError):
            profile.values["language"] = "it"  # type: ignore[index]

    def test_an_integer_temperature_is_a_float(self, root, profiles_dir):
        profile = load(profiles_dir, root, "temperature = 1\n")
        assert profile.values["temperature"] == 1.0
        assert isinstance(profile.values["temperature"], float)

    def test_relative_paths_are_relative_to_the_project_root(
        self, root, profiles_dir, tmp_path, monkeypatch
    ):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        sub = profiles_dir / "nested"
        sub.mkdir()
        # Neither the current directory nor the profile's own folder counts.
        put(sub, "voice", 'checkpoint = "models/x"\nreference_wavs = ["a.wav"]\n')

        profile = load_profile(str(sub / "voice.toml"), profiles_dir, root)

        assert profile.values["checkpoint"] == root / "models" / "x"
        assert profile.values["reference_wavs"] == [root / "a.wav"]

    def test_absolute_paths_are_kept(self, root, profiles_dir, tmp_path):
        absolute = tmp_path / "somewhere" / "voice.pth"
        profile = load(profiles_dir, root, f'checkpoint = "{absolute.as_posix()}"\n')
        assert profile.values["checkpoint"] == absolute

    def test_the_tilde_is_expanded(self, root, profiles_dir, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        text = 'checkpoint = "~/voices/frieren"\nreference_wavs = ["~/a.wav"]\n'

        profile = load(profiles_dir, root, text)

        assert profile.values["checkpoint"] == home / "voices" / "frieren"
        assert profile.values["reference_wavs"] == [home / "a.wav"]

    def test_the_system_prompt_file_is_read(self, root, profiles_dir):
        (root / "prompts").mkdir()
        (root / "prompts" / "frieren.md").write_text(
            "\nYou are Frieren.\nShort answers: élfa, 魔法.\n\n", encoding="utf-8"
        )

        profile = load(profiles_dir, root, 'system_prompt_file = "prompts/frieren.md"\n')

        assert profile.values["system_prompt"] == "You are Frieren.\nShort answers: élfa, 魔法."
        assert "system_prompt_file" not in profile.values

    def test_loading_by_the_path_of_the_file(self, root, profiles_dir, tmp_path):
        path = put(profiles_dir, "frieren", 'language = "it"\n')
        by_name = load_profile("frieren", profiles_dir, root)

        assert load_profile(str(path), profiles_dir, root).values == by_name.values
        assert load_profile(str(path), profiles_dir, root).name == "frieren"
        other = tmp_path / "other"
        other.mkdir()
        (other / "mine.toml").write_text('language = "fr"\n')
        assert load_profile(str(other / "mine.toml"), profiles_dir, root).values["language"] == "fr"

    def test_a_relative_file_path_is_relative_to_the_current_directory(
        self, root, profiles_dir, monkeypatch
    ):
        put(profiles_dir, "frieren", 'language = "it"\n')
        monkeypatch.chdir(root)
        assert load_profile("profiles/frieren", profiles_dir, root).values["language"] == "it"
        assert load_profile("profiles/frieren.toml", profiles_dir, root).name == "frieren"

    def test_errors_are_herald_errors(self):
        assert issubclass(ProfileError, HeraldError)


class TestLoadProfileErrors:
    def error(self, root, profiles_dir, text, match):
        path = put(profiles_dir, "voice", text)
        with pytest.raises(ProfileError, match=match) as excinfo:
            load_profile("voice", profiles_dir, root)
        assert str(path) in str(excinfo.value)  # every message names the profile file
        return str(excinfo.value)

    def test_an_unknown_key_names_it_and_lists_the_valid_ones(self, root, profiles_dir):
        message = self.error(root, profiles_dir, 'colour = "red"\n', "unknown key 'colour'")
        assert all(key in message for key in PROFILE_KEYS)

    def test_a_typo_gets_a_suggestion(self, root, profiles_dir):
        self.error(root, profiles_dir, 'system_promt = "x"\n', "did you mean 'system_prompt'")

    def test_several_unknown_keys(self, root, profiles_dir):
        self.error(root, profiles_dir, "a = 1\nb = 2\n", "unknown keys 'a', 'b'")

    def test_a_table_is_an_unknown_key(self, root, profiles_dir):
        self.error(root, profiles_dir, '[voice]\nlanguage = "en"\n', "unknown key 'voice'")

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("description = 3", "description must be a string, got an integer"),
            ("checkpoint = 3", "checkpoint must be a string, got an integer"),
            ("checkpoint = ['a']", "checkpoint must be a string, got a list"),
            ("dataset = true", "dataset must be a string, got a boolean"),
            ("language = 1.5", "language must be a string, got a number"),
            ("reference_wavs = 'a.wav'", "reference_wavs must be a list of strings, got a string"),
            ("reference_wavs = ['a.wav', 3]", "reference_wavs must be a list of strings"),
            ("num_references = '3'", "num_references must be an integer, got a string"),
            ("num_references = 2.5", "num_references must be an integer, got a number"),
            ("history = true", "history must be an integer, got a boolean"),
            ("temperature = 'hot'", "temperature must be a number, got a string"),
            ("temperature = true", "temperature must be a number, got a boolean"),
            ("system_prompt = 3", "system_prompt must be a string, got an integer"),
            ("system_prompt_file = []", "system_prompt_file must be a string, got a list"),
            ("ollama_model = 3", "ollama_model must be a string"),
            ("speaker_name = 3", "speaker_name must be a string"),
            ("device = 3", "device must be a string"),
        ],
    )
    def test_wrong_types_name_the_key_and_the_expected_type(self, root, profiles_dir, text, match):
        self.error(root, profiles_dir, text + "\n", match)

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("checkpoint = ''", "checkpoint must not be empty"),
            ("language = '  '", "language must not be empty"),
            ("system_prompt = ''", "system_prompt must not be empty"),
            ("reference_wavs = []", "reference_wavs must be a list of file names"),
            ("reference_wavs = ['a.wav', '']", "reference_wavs must be a list of file names"),
            ("num_references = 0", "num_references must be at least 1, got 0"),
            ("history = -1", "history must be at least 0, got -1"),
            ("temperature = 0", "temperature must be a positive number, got 0"),
            ("temperature = -0.5", "temperature must be a positive number"),
            ("temperature = inf", "temperature must be a positive number"),
            ("temperature = nan", "temperature must be a positive number"),
        ],
    )
    def test_invalid_values(self, root, profiles_dir, text, match):
        self.error(root, profiles_dir, text + "\n", match)

    def test_a_tilde_that_cannot_be_expanded(self, root, profiles_dir):
        self.error(root, profiles_dir, 'checkpoint = "~no_such_user_xyz/voice"\n', "checkpoint")

    def test_a_prompt_and_a_prompt_file_are_exclusive(self, root, profiles_dir):
        text = 'system_prompt = "x"\nsystem_prompt_file = "y.md"\n'
        self.error(root, profiles_dir, text, "either system_prompt or system_prompt_file")

    def test_a_missing_prompt_file(self, root, profiles_dir):
        self.error(root, profiles_dir, 'system_prompt_file = "nope.md"\n', "not found.*nope.md")

    def test_an_empty_prompt_file(self, root, profiles_dir):
        (root / "empty.md").write_text("  \n")
        self.error(root, profiles_dir, 'system_prompt_file = "empty.md"\n', "empty.md is empty")

    def test_a_prompt_file_that_is_not_utf8(self, root, profiles_dir):
        (root / "latin.md").write_bytes("café".encode("latin-1"))
        self.error(root, profiles_dir, 'system_prompt_file = "latin.md"\n', "not valid UTF-8")

    def test_invalid_toml_reports_the_parser_message(self, root, profiles_dir):
        message = self.error(root, profiles_dir, 'language = "en"\nhistory = \n', "invalid TOML")
        assert "line 2" in message

    def test_a_profile_that_is_not_utf8(self, root, profiles_dir):
        path = profiles_dir / "voice.toml"
        path.write_bytes('description = "café"\n'.encode("latin-1"))
        with pytest.raises(ProfileError, match="not valid UTF-8"):
            load_profile("voice", profiles_dir, root)

    @pytest.mark.parametrize("name", ["", "a b", "a.b", "..", "a:b"])
    def test_invalid_names(self, root, profiles_dir, name):
        with pytest.raises(ProfileError, match="Invalid profile name"):
            load_profile(name, profiles_dir, root)

    def test_a_missing_profile_lists_the_available_ones(self, root, profiles_dir):
        put(profiles_dir, "frieren", "")
        put(profiles_dir, "fern", "")
        put(profiles_dir, "_template", "")

        with pytest.raises(
            ProfileError, match="'stark' not found.*Available profiles: fern, frieren"
        ):
            load_profile("stark", profiles_dir, root)

    def test_a_missing_profile_without_any_profiles(self, root, tmp_path):
        with pytest.raises(ProfileError, match="No profiles in"):
            load_profile("stark", tmp_path / "no_such_dir", root)

    def test_a_directory_is_not_a_profile(self, root, profiles_dir):
        (profiles_dir / "voice.toml").mkdir()
        with pytest.raises(ProfileError, match="not found"):
            load_profile("voice", profiles_dir, root)

    def test_a_missing_profile_file_path(self, root, profiles_dir, tmp_path):
        with pytest.raises(ProfileError, match="not found"):
            load_profile(str(tmp_path / "x" / "voice.toml"), profiles_dir, root)


class TestProfileDefaults:
    def test_maps_the_toml_keys_to_the_cli_options(self, root, profiles_dir):
        profile = load(profiles_dir, root, FULL_PROFILE)

        assert profile_defaults(profile) == {
            "checkpoint": root / "models" / "frieren",
            "dataset_dir": root / "dataset" / "frieren",
            "reference_wav": [root / "clips" / "a.wav", root / "clips" / "b.wav"],
            "num_references": 4,
            "language": "en",
            "temperature": 0.8,
            "device": "cpu",
            "system_prompt": "You are Frieren.",
            "ollama_model": "llama3.2:3b",
            "history": 6,
            "speaker_name": "frieren",
        }

    def test_only_what_the_profile_sets(self, root, profiles_dir):
        profile = load(profiles_dir, root, 'description = "Just a note"\nlanguage = "it"\n')
        assert profile_defaults(profile) == {"language": "it"}

    def test_the_text_of_the_prompt_file_is_the_system_prompt(self, root, profiles_dir):
        (root / "p.md").write_text("Be brief.")
        profile = load(profiles_dir, root, 'system_prompt_file = "p.md"\n')
        assert profile_defaults(profile) == {"system_prompt": "Be brief."}

    def test_the_reference_list_is_a_copy(self, root, profiles_dir):
        profile = load(profiles_dir, root, 'reference_wavs = ["a.wav"]\n')
        profile_defaults(profile)["reference_wav"].append(Path("x.wav"))
        assert profile.values["reference_wavs"] == [root / "a.wav"]

    def test_every_key_but_the_description_and_prompt_file_is_an_option(self):
        assert set(profiles._CLI_DESTS) == set(PROFILE_KEYS) - {"description", "system_prompt_file"}


class TestListProfiles:
    def test_a_missing_directory(self, root):
        assert list_profiles(root / "nope", root) == []

    def test_an_empty_directory(self, profiles_dir, root):
        assert list_profiles(profiles_dir, root) == []

    def test_sorted_summaries(self, root, profiles_dir):
        (root / "models" / "frieren").mkdir(parents=True)
        (root / "models" / "frieren" / "best_model.pth").write_bytes(b"x")
        put(profiles_dir, "zed", 'description = "Last"\n')
        put(
            profiles_dir,
            "frieren",
            'description = "The elf"\nlanguage = "en"\ncheckpoint = "models/frieren"\n',
        )

        summaries = list_profiles(profiles_dir, root)

        assert [s.name for s in summaries] == ["frieren", "zed"]
        frieren, zed = summaries
        assert frieren.path == profiles_dir / "frieren.toml"
        assert (frieren.description, frieren.language) == ("The elf", "en")
        assert frieren.checkpoint == "models/frieren"
        assert frieren.problems == ()
        assert (zed.description, zed.language, zed.checkpoint) == ("Last", None, None)

    def test_templates_and_hidden_files_are_ignored(self, root, profiles_dir):
        put(profiles_dir, "_template", 'description = "A template"\n')
        put(profiles_dir, ".hidden", "")
        put(profiles_dir, "real", "")
        (profiles_dir / "notes.md").write_text("not a profile")
        (profiles_dir / "dir.toml").mkdir()

        assert [s.name for s in list_profiles(profiles_dir, root)] == ["real"]

    def test_a_broken_profile_is_listed_with_its_error(self, root, profiles_dir):
        put(profiles_dir, "good", 'description = "Fine"\n')
        broken = put(profiles_dir, "broken", 'colour = "red"\n')
        invalid = put(profiles_dir, "invalid", "history = \n")

        summaries = {s.name: s for s in list_profiles(profiles_dir, root)}

        assert summaries["good"].problems == ()
        for name, path in (("broken", broken), ("invalid", invalid)):
            (problem,) = summaries[name].problems
            assert str(path) in problem
            assert (summaries[name].description, summaries[name].checkpoint) == ("", None)
        assert "unknown key 'colour'" in summaries["broken"].problems[0]

    def test_missing_files_are_reported_as_problems(self, root, profiles_dir):
        put(profiles_dir, "voice", 'checkpoint = "models/gone"\n')

        (summary,) = list_profiles(profiles_dir, root)

        assert summary.problems == (f"checkpoint not found: {root / 'models' / 'gone'}",)
        assert summary.checkpoint == "models/gone"

    def test_a_name_that_cannot_be_used_with_the_option(self, root, profiles_dir):
        put(profiles_dir, "my voice", "")
        (summary,) = list_profiles(profiles_dir, root)
        assert summary.name == "my voice"
        assert "cannot be used with --profile" in summary.problems[0]


class TestCheckProfile:
    @pytest.fixture
    def files(self, root) -> Path:
        (root / "models" / "frieren").mkdir(parents=True)
        (root / "models" / "frieren" / "best_model.pth").write_bytes(b"x")
        (root / "dataset" / "frieren").mkdir(parents=True)
        (root / "dataset" / "frieren" / "metadata.csv").write_text("a.wav|Hi\n")
        (root / "clips").mkdir()
        (root / "clips" / "a.wav").write_bytes(b"x")
        return root

    def check(self, profiles_dir, root, text) -> list[str]:
        return check_profile(load(profiles_dir, root, text))

    def test_a_usable_profile(self, files, profiles_dir):
        text = 'checkpoint = "models/frieren"\ndataset = "dataset/frieren"\n'
        assert self.check(profiles_dir, files, text) == []

    def test_the_base_voice_needs_no_files(self, files, profiles_dir):
        assert self.check(profiles_dir, files, 'language = "en"\n') == []

    def test_a_checkpoint_file_is_fine(self, files, profiles_dir):
        text = 'checkpoint = "models/frieren/best_model.pth"\n'
        assert self.check(profiles_dir, files, text) == []

    def test_a_missing_checkpoint(self, files, profiles_dir):
        problems = self.check(profiles_dir, files, 'checkpoint = "models/gone"\n')
        assert problems == [f"checkpoint not found: {files / 'models' / 'gone'}"]

    def test_a_checkpoint_directory_without_a_model(self, files, profiles_dir):
        (files / "models" / "empty").mkdir()
        (problem,) = self.check(profiles_dir, files, 'checkpoint = "models/empty"\n')
        assert "no .pth model file" in problem

    def test_a_missing_dataset(self, files, profiles_dir):
        (problem,) = self.check(profiles_dir, files, 'dataset = "dataset/gone"\n')
        assert problem == f"dataset directory not found: {files / 'dataset' / 'gone'}"

    def test_a_dataset_without_metadata(self, files, profiles_dir):
        (files / "dataset" / "bare").mkdir()
        (problem,) = self.check(profiles_dir, files, 'dataset = "dataset/bare"\n')
        assert "no metadata.csv" in problem

    def test_a_missing_reference_clip(self, files, profiles_dir):
        text = 'reference_wavs = ["clips/a.wav", "clips/b.wav", "clips/c.wav"]\n'
        assert self.check(profiles_dir, files, text) == [
            f"reference clip not found: {files / 'clips' / 'b.wav'}",
            f"reference clip not found: {files / 'clips' / 'c.wav'}",
        ]

    def test_the_dataset_is_not_needed_when_there_are_reference_clips(self, files, profiles_dir):
        text = 'reference_wavs = ["clips/a.wav"]\ndataset = "dataset/gone"\n'
        assert self.check(profiles_dir, files, text) == []

    def test_all_problems_are_listed(self, files, profiles_dir):
        text = 'checkpoint = "models/gone"\ndataset = "dataset/gone"\n'
        assert len(self.check(profiles_dir, files, text)) == 2


class TestWriteProfile:
    def roundtrip(self, tmp_path, values) -> dict:
        path = write_profile(tmp_path / "out.toml", values)
        return tomllib.loads(path.read_text(encoding="utf-8"))

    def test_every_kind_of_value_round_trips(self, tmp_path):
        values = {
            "description": "Frieren, the elf mage",
            "checkpoint": "models/frieren",
            "reference_wavs": ["clips/a.wav", "clips/b.wav"],
            "num_references": 3,
            "temperature": 0.7,
            "history": 0,
        }
        assert self.roundtrip(tmp_path, values) == values

    @pytest.mark.parametrize(
        "text",
        [
            'He said "hello"',
            "back\\slash and C:\\path\\to\\file",
            "tab\there",
            "caf\u00e9, \u9b54\u6cd5, \U0001f9dd",
            "line one\nline two\n",
            "\nstarts with a newline",
            'ends with a quote"',
            'multi\nline with "quotes", """triple""" and \\ backslash\n',
            'a run of quotes """" """""',
            '"""',
            "\r\nwindows\r\nnewlines\r\n",
            "control \x01 \x1f \x7f characters",
            "single line with \\n and \\u00e9 written out",
            "  leading and trailing space  ",
            "#not a comment = true",
        ],
    )
    def test_strings_round_trip(self, tmp_path, text):
        values = {"description": text, "system_prompt": text, "reference_wavs": [text]}
        assert self.roundtrip(tmp_path, values) == values

    def test_odd_numbers_round_trip(self, tmp_path):
        for temperature in (5e-06, 1.0, 1e16, 0.1 + 0.2, 3):
            assert self.roundtrip(tmp_path / str(temperature), {"temperature": temperature}) == {
                "temperature": temperature
            }

    def test_multiline_strings_are_blocks(self, tmp_path):
        path = write_profile(tmp_path / "p.toml", {"system_prompt": "You are Frieren.\nBe brief."})

        text = path.read_text(encoding="utf-8")

        assert text == 'system_prompt = """\nYou are Frieren.\nBe brief."""\n'

    def test_keys_are_written_in_a_stable_order(self, tmp_path):
        values = {key: "x" for key in reversed(PROFILE_KEYS)}
        for key in ("num_references", "history"):
            values[key] = 1
        values["temperature"] = 0.5
        values["reference_wavs"] = ["a.wav"]
        del values["system_prompt_file"]  # exclusive with system_prompt

        text = write_profile(tmp_path / "p.toml", values).read_text(encoding="utf-8")

        written = [line.split(" = ")[0] for line in text.splitlines()]
        assert written == [k for k in PROFILE_KEYS if k in values]

    def test_unicode_is_written_as_utf8(self, tmp_path):
        path = write_profile(tmp_path / "p.toml", {"description": "魔法使い"})
        assert "魔法使い" in path.read_bytes().decode("utf-8")

    def test_it_refuses_to_overwrite(self, tmp_path):
        path = tmp_path / "p.toml"
        path.write_text('language = "it"\n')

        with pytest.raises(ProfileError, match="already exists"):
            write_profile(path, {"language": "en"})

        assert path.read_text() == 'language = "it"\n'

    def test_overwrite_replaces_the_file(self, tmp_path):
        path = tmp_path / "p.toml"
        path.write_text('language = "it"\nhistory = 3\n')

        write_profile(path, {"language": "en"}, overwrite=True)

        assert tomllib.loads(path.read_text()) == {"language": "en"}

    def test_parent_directories_are_created(self, tmp_path):
        path = write_profile(tmp_path / "a" / "b" / "p.toml", {"language": "en"})
        assert path.is_file()

    def test_invalid_values_are_refused_and_nothing_is_written(self, tmp_path):
        for values, match in [
            ({"colour": "red"}, "unknown key 'colour'"),
            ({"history": "ten"}, "history must be an integer"),
            ({"history": True}, "history must be an integer"),
            ({"temperature": 0}, "temperature must be a positive number"),
            ({"system_prompt": "a", "system_prompt_file": "b"}, "either"),
        ]:
            with pytest.raises(ProfileError, match=match):
                write_profile(tmp_path / "bad.toml", values)
        assert not (tmp_path / "bad.toml").exists()

    def test_path_objects_are_converted(self, tmp_path):
        values = {"checkpoint": Path("models/frieren"), "reference_wavs": [Path("clips/a.wav")]}
        assert self.roundtrip(tmp_path, values) == {
            "checkpoint": "models/frieren",
            "reference_wavs": ["clips/a.wav"],
        }

    def test_the_file_loads_as_a_profile(self, root, profiles_dir):
        values = {
            "description": 'The "elf" mage',
            "checkpoint": "models/frieren",
            "system_prompt": "You are Frieren.\nShort answers.\n",
            "temperature": 0.7,
            "history": 10,
        }
        write_profile(profiles_dir / "frieren.toml", values)

        profile = load_profile("frieren", profiles_dir, root)

        assert profile.description == 'The "elf" mage'
        assert profile.values["checkpoint"] == root / "models" / "frieren"
        assert profile.values["system_prompt"] == "You are Frieren.\nShort answers."  # stripped


class TestEnsureProfile:
    def test_creates_a_missing_profile(self, profiles_dir):
        path, created = ensure_profile(profiles_dir, "frieren", {"language": "en"})

        assert (path, created) == (profiles_dir / "frieren.toml", True)
        assert tomllib.loads(path.read_text()) == {"language": "en"}

    def test_never_overwrites(self, profiles_dir):
        path = put(profiles_dir, "frieren", 'language = "it"  # my own profile\n')

        result = ensure_profile(profiles_dir, "frieren", {"language": "en"})

        assert result == (path, False)
        assert path.read_text() == 'language = "it"  # my own profile\n'

    def test_creates_the_profiles_directory(self, tmp_path):
        path, created = ensure_profile(tmp_path / "new" / "profiles", "x", {"language": "en"})
        assert created and path.is_file()

    @pytest.mark.parametrize("name", ["", "a b", "../x", "a/b", "a.toml"])
    def test_the_name_is_validated(self, profiles_dir, name):
        with pytest.raises(ProfileError, match="Invalid profile name"):
            ensure_profile(profiles_dir, name, {})
        assert list(profiles_dir.iterdir()) == []

    def test_invalid_values_leave_no_file(self, profiles_dir):
        with pytest.raises(ProfileError):
            ensure_profile(profiles_dir, "bad", {"colour": "red"})
        assert list(profiles_dir.iterdir()) == []


class TestRelativeToRoot:
    def test_a_path_inside_the_root(self, root):
        assert relative_to_root(root / "models" / "frieren", root) == "models/frieren"

    def test_the_root_itself(self, root):
        assert relative_to_root(root, root) == "."

    def test_a_path_outside_the_root_stays_absolute(self, root, tmp_path):
        outside = tmp_path / "elsewhere" / "voice.pth"
        assert relative_to_root(outside, root) == str(outside)

    def test_a_sibling_with_the_same_prefix_is_outside(self, root):
        sibling = root.parent / (root.name + "-other") / "x"
        assert relative_to_root(sibling, root) == str(sibling)

    def test_a_relative_path_is_taken_as_project_relative(self, root):
        assert relative_to_root(Path("models/frieren"), root) == "models/frieren"

    def test_a_symlinked_root(self, root, tmp_path):
        link = tmp_path / "link"
        link.symlink_to(root, target_is_directory=True)
        (root / "models").mkdir()
        assert relative_to_root(root / "models", link) == "models"
        assert relative_to_root(link / "models", root) == "models"
