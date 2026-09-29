import logging

import pytest
import requests

from herald.errors import CheckpointError
from herald.tts import checkpoints


class FakeDownload:
    """A streaming ``requests`` response."""

    def __init__(self, body=b"", *, headers=None, fail_after=None, status_error=None):
        self.body = body
        self.headers = {"Content-Length": str(len(body))} if headers is None else headers
        self.fail_after = fail_after
        self.status_error = status_error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def iter_content(self, chunk_size):
        half = len(self.body) // 2
        yield self.body[:half]
        if self.fail_after:
            raise requests.ConnectionError("connection dropped")
        yield self.body[half:]


@pytest.fixture
def get(mocker):
    return mocker.patch("requests.get")


def test_downloads_missing_files(tmp_path, get):
    get.side_effect = lambda url, **kw: FakeDownload(url.rsplit("/", 1)[1].encode())
    dest = tmp_path / "ckpt"

    result = checkpoints.ensure_base_checkpoints(
        dest, files=["config.json", "vocab.json"], base_url="http://host/xtts/"
    )

    assert result == dest
    assert (dest / "config.json").read_bytes() == b"config.json"
    assert (dest / "vocab.json").read_bytes() == b"vocab.json"
    assert [c.args[0] for c in get.call_args_list] == [
        "http://host/xtts/config.json",
        "http://host/xtts/vocab.json",
    ]
    assert get.call_args.kwargs["stream"] is True
    assert not list(dest.glob("*.part"))


def test_request_uses_the_timeout_and_follows_redirects(tmp_path, get):
    get.return_value = FakeDownload(b"{}")
    checkpoints.ensure_base_checkpoints(tmp_path, files=["config.json"], timeout=7.5)
    assert get.call_args.kwargs["timeout"] == 7.5
    assert get.call_args.kwargs["allow_redirects"] is True


def test_stale_part_file_from_a_killed_run_is_replaced(tmp_path, get):
    (tmp_path / "model.pth.part").write_bytes(b"leftover from a killed download" * 10)
    get.return_value = FakeDownload(b"fresh")
    checkpoints.ensure_base_checkpoints(tmp_path, files=["model.pth"])
    assert (tmp_path / "model.pth").read_bytes() == b"fresh"
    assert not list(tmp_path.glob("*.part"))


def test_progress_is_logged(tmp_path, get, monkeypatch, caplog):
    monkeypatch.setattr(checkpoints, "_LOG_EVERY_BYTES", 1)
    get.return_value = FakeDownload(b"x" * (2 << 20))
    with caplog.at_level(logging.INFO, logger=checkpoints.logger.name):
        checkpoints.ensure_base_checkpoints(tmp_path, files=["model.pth"])
    assert any("model.pth: 1 MiB of 2 MiB" in r.getMessage() for r in caplog.records)


def test_existing_files_are_not_downloaded_again(tmp_path, get):
    (tmp_path / "config.json").write_text("{}")
    checkpoints.ensure_base_checkpoints(tmp_path, files=["config.json"])
    get.assert_not_called()


def test_empty_files_are_downloaded_again(tmp_path, get):
    (tmp_path / "config.json").write_bytes(b"")
    get.return_value = FakeDownload(b"{}")
    checkpoints.ensure_base_checkpoints(tmp_path, files=["config.json"])
    assert (tmp_path / "config.json").read_bytes() == b"{}"


def test_interrupted_download_leaves_no_file_behind(tmp_path, get):
    get.return_value = FakeDownload(b"x" * 100, fail_after=True)
    with pytest.raises(CheckpointError, match="Failed to download"):
        checkpoints.ensure_base_checkpoints(tmp_path, files=["model.pth"])
    assert list(tmp_path.iterdir()) == []


def test_truncated_download_is_detected(tmp_path, get):
    get.return_value = FakeDownload(b"x" * 10, headers={"Content-Length": "100"})
    with pytest.raises(CheckpointError, match="truncated"):
        checkpoints.ensure_base_checkpoints(tmp_path, files=["model.pth"])
    assert list(tmp_path.iterdir()) == []


def test_content_encoding_disables_the_size_check(tmp_path, get):
    get.return_value = FakeDownload(
        b"x" * 10, headers={"Content-Length": "4", "Content-Encoding": "gzip"}
    )
    checkpoints.ensure_base_checkpoints(tmp_path, files=["config.json"])
    assert (tmp_path / "config.json").read_bytes() == b"x" * 10


def test_http_error(tmp_path, get):
    get.return_value = FakeDownload(status_error=requests.HTTPError("404 Not Found"))
    with pytest.raises(CheckpointError, match="404"):
        checkpoints.ensure_base_checkpoints(tmp_path, files=["model.pth"])
    assert list(tmp_path.iterdir()) == []


def test_default_file_lists():
    assert set(checkpoints.BASE_FILES) == {
        "vocab.json",
        "config.json",
        "model.pth",
        "dvae.pth",
        "mel_stats.pth",
        "speakers_xtts.pth",
    }
    # Fine-tuned inference must not pull the 1.9 GB base model.
    assert "model.pth" not in checkpoints.FINETUNED_INFERENCE_FILES
