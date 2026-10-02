from pathlib import Path

import pytest

from herald import paths
from herald.config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_SYSTEM_PROMPT,
    Settings,
)
from herald.errors import ConfigError


def test_defaults_live_under_the_project_root(tmp_path):
    s = Settings.from_env({"HERALD_PROJECT_ROOT": str(tmp_path)})
    root = tmp_path.resolve()
    assert s.project_root == root
    assert s.dataset_dir == root / "dataset" / "frieren"
    assert s.models_dir == root / "models"
    assert s.checkpoint_dir == root / "models" / "xtts_v2"
    assert s.checkpoint is None
    assert s.runs_dir == root / "runs"
    assert s.output_dir == root / "output"
    assert s.tools_dir == root / "tools"
    assert s.profiles_dir == root / "profiles"
    assert s.profile is None
    assert s.device == "auto"
    assert s.language == "en"
    assert s.ollama_model == DEFAULT_OLLAMA_MODEL
    assert s.system_prompt == DEFAULT_SYSTEM_PROMPT


def test_environment_overrides(tmp_path):
    env = {
        "HERALD_PROJECT_ROOT": str(tmp_path),
        "HERALD_DATASET_DIR": "/data/voice",
        "HERALD_MODELS_DIR": "/data/models",
        "HERALD_CHECKPOINT_DIR": "/data/ckpt",
        "HERALD_CHECKPOINT": "/data/models/frieren",
        "HERALD_RUNS_DIR": "/data/runs",
        "HERALD_OUTPUT_DIR": "/data/out",
        "HERALD_TOOLS_DIR": "/data/tools",
        "HERALD_PROFILES_DIR": "/data/profiles",
        "HERALD_PROFILE": "frieren",
        "HERALD_DEVICE": "cpu",
        "HERALD_LANGUAGE": "it",
        "HERALD_OLLAMA_URL": "http://ollama:11434",
        "HERALD_OLLAMA_MODEL": "mistral",
        "HERALD_OLLAMA_TIMEOUT": "7.5",
        "HERALD_SYSTEM_PROMPT": "Be brief.",
        "HERALD_CHECKPOINT_URL": "http://mirror/xtts",
    }
    s = Settings.from_env(env)
    assert s.dataset_dir == Path("/data/voice")
    assert s.models_dir == Path("/data/models")
    assert s.checkpoint_dir == Path("/data/ckpt")
    assert s.checkpoint == Path("/data/models/frieren")
    assert s.runs_dir == Path("/data/runs")
    assert s.output_dir == Path("/data/out")
    assert s.tools_dir == Path("/data/tools")
    assert s.profiles_dir == Path("/data/profiles")
    assert s.profile == "frieren"
    assert (s.device, s.language) == ("cpu", "it")
    assert (s.ollama_url, s.ollama_model, s.ollama_timeout) == (
        "http://ollama:11434",
        "mistral",
        7.5,
    )
    assert s.system_prompt == "Be brief."
    assert s.checkpoint_url == "http://mirror/xtts"


def test_base_weights_follow_the_models_directory(tmp_path):
    env = {"HERALD_PROJECT_ROOT": str(tmp_path), "HERALD_MODELS_DIR": "/data/models"}
    assert Settings.from_env(env).checkpoint_dir == Path("/data/models/xtts_v2")


def test_checkpoint_dir_wins_over_the_models_directory(tmp_path):
    env = {
        "HERALD_PROJECT_ROOT": str(tmp_path),
        "HERALD_MODELS_DIR": "/data/models",
        "HERALD_CHECKPOINT_DIR": "/data/base",
    }
    s = Settings.from_env(env)
    assert (s.models_dir, s.checkpoint_dir) == (Path("/data/models"), Path("/data/base"))


def test_empty_variable_counts_as_unset(tmp_path):
    env = {
        "HERALD_PROJECT_ROOT": str(tmp_path),
        "HERALD_DEVICE": "",
        "HERALD_MODELS_DIR": "",
        "HERALD_CHECKPOINT_DIR": "",
        "HERALD_CHECKPOINT": "",
        "HERALD_TOOLS_DIR": "",
        "HERALD_PROFILES_DIR": "",
        "HERALD_PROFILE": "",
    }
    s = Settings.from_env(env)
    root = tmp_path.resolve()
    assert s.device == "auto"
    assert s.tools_dir == root / "tools"
    assert s.profiles_dir == root / "profiles"
    assert s.profile is None
    assert s.checkpoint_dir == root / "models" / "xtts_v2"
    assert s.checkpoint is None


def test_home_is_expanded(tmp_path):
    env = {
        "HERALD_PROJECT_ROOT": str(tmp_path),
        "HERALD_DATASET_DIR": "~/voices",
        "HERALD_MODELS_DIR": "~/models",
        "HERALD_CHECKPOINT": "~/models/frieren",
        "HERALD_TOOLS_DIR": "~/my-tools",
        "HERALD_PROFILES_DIR": "~/my-profiles",
    }
    s = Settings.from_env(env)
    assert s.tools_dir == Path("~/my-tools").expanduser()
    assert s.profiles_dir == Path("~/my-profiles").expanduser()
    assert s.dataset_dir == Path("~/voices").expanduser()
    assert s.models_dir == Path("~/models").expanduser()
    assert s.checkpoint == Path("~/models/frieren").expanduser()


@pytest.mark.parametrize("value", ["soon", "0", "-3"])
def test_invalid_timeout_is_reported_when_it_is_used(tmp_path, value):
    env = {"HERALD_PROJECT_ROOT": str(tmp_path), "HERALD_OLLAMA_TIMEOUT": value}
    settings = Settings.from_env(env)  # a bad timeout must not block commands that ignore it
    assert settings.dataset_dir  # the rest of the settings is usable
    with pytest.raises(ConfigError, match="HERALD_OLLAMA_TIMEOUT"):
        _ = settings.ollama_timeout


def test_timeout_defaults_when_unset_or_empty(tmp_path):
    for env in ({}, {"HERALD_OLLAMA_TIMEOUT": ""}):
        s = Settings.from_env({"HERALD_PROJECT_ROOT": str(tmp_path), **env})
        assert s.ollama_timeout == DEFAULT_OLLAMA_TIMEOUT


def test_relative_paths_are_relative_to_the_working_directory(tmp_path):
    """The project root only anchors the defaults; explicit relative paths are left as they are."""
    env = {
        "HERALD_PROJECT_ROOT": str(tmp_path),
        "HERALD_MODELS_DIR": "voices",
        "HERALD_CHECKPOINT": "voices/frieren",
        "HERALD_TOOLS_DIR": "my-tools",
        "HERALD_PROFILES_DIR": "my-profiles",
    }
    s = Settings.from_env(env)
    assert s.tools_dir == Path("my-tools")
    assert s.profiles_dir == Path("my-profiles")
    assert s.models_dir == Path("voices")
    assert s.checkpoint == Path("voices/frieren")
    assert s.checkpoint_dir == Path("voices") / "xtts_v2"
    assert s.dataset_dir == tmp_path.resolve() / "dataset" / "frieren"


class TestProjectRoot:
    def test_explicit_variable_wins(self, tmp_path):
        assert paths.project_root({"HERALD_PROJECT_ROOT": str(tmp_path)}) == tmp_path.resolve()

    def test_source_checkout_is_used_before_the_working_directory(self, tmp_path):
        root = paths.project_root({}, cwd=tmp_path)
        # The test suite runs from a source checkout, so that is what we get.
        assert (root / "pyproject.toml").is_file()

    def test_falls_back_to_the_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "_source_checkout_root", lambda: None)
        assert paths.project_root({}, cwd=tmp_path) == tmp_path.resolve()
