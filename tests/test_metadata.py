from pathlib import Path

import pytest

from herald.dataset.metadata import (
    load_metadata,
    pick_reference_wavs,
    resolve_audio_path,
    sniff_delimiter,
    split_items,
)
from herald.errors import ConfigError, MetadataError


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadMetadata:
    def test_pipe_without_header(self, tmp_path):
        csv_path = write(
            tmp_path / "m.csv", "audio/0.wav|What's wrong?\naudio/1.wav|Fine, thanks.\n"
        )
        assert load_metadata(csv_path) == [
            {"audio_file": "audio/0.wav", "text": "What's wrong?"},
            {"audio_file": "audio/1.wav", "text": "Fine, thanks."},
        ]

    def test_pipe_with_header(self, tmp_path):
        csv_path = write(tmp_path / "m.csv", "audio_file|text\naudio/0.wav|Hello\n")
        assert load_metadata(csv_path) == [{"audio_file": "audio/0.wav", "text": "Hello"}]

    def test_tab_delimiter(self, tmp_path):
        csv_path = write(tmp_path / "m.csv", "audio/0.wav\tHello, world\n")
        assert load_metadata(csv_path) == [{"audio_file": "audio/0.wav", "text": "Hello, world"}]

    def test_comma_delimiter_with_header_and_quotes(self, tmp_path):
        csv_path = write(tmp_path / "m.csv", 'file_name,transcript_text\nclip.wav,"Well, well."\n')
        assert load_metadata(csv_path) == [{"audio_file": "clip.wav", "text": "Well, well."}]

    def test_header_columns_in_any_order(self, tmp_path):
        csv_path = write(tmp_path / "m.csv", "text|path\nHello|audio/0.wav\n")
        assert load_metadata(csv_path) == [{"audio_file": "audio/0.wav", "text": "Hello"}]

    def test_utf8_bom_is_ignored(self, tmp_path):
        csv_path = tmp_path / "m.csv"
        csv_path.write_bytes(b"\xef\xbb\xbfaudio_file|text\naudio/0.wav|Hello\n")
        assert load_metadata(csv_path) == [{"audio_file": "audio/0.wav", "text": "Hello"}]

    def test_skips_short_and_empty_rows(self, tmp_path):
        csv_path = write(
            tmp_path / "m.csv",
            "audio/0.wav|Kept\naudio/1.wav\naudio/2.wav|\n\n|no audio\naudio/3.wav|Also kept\n",
        )
        assert [i["audio_file"] for i in load_metadata(csv_path)] == ["audio/0.wav", "audio/3.wav"]

    def test_empty_file_gives_no_rows(self, tmp_path):
        assert load_metadata(write(tmp_path / "m.csv", "")) == []

    def test_header_without_needed_columns_is_an_error(self, tmp_path):
        csv_path = write(tmp_path / "m.csv", "id|speaker\n1|a\n")
        with pytest.raises(MetadataError, match="audio column"):
            load_metadata(csv_path)

    def test_doubled_quotes_inside_a_quoted_field(self, tmp_path):
        csv_path = write(
            tmp_path / "m.csv",
            'audio/0.wav|"Monsters cried, ""Help me."""\naudio/1.wav|Next\n',
        )
        assert load_metadata(csv_path) == [
            {"audio_file": "audio/0.wav", "text": 'Monsters cried, "Help me."'},
            {"audio_file": "audio/1.wav", "text": "Next"},
        ]

    def test_an_unclosed_quote_is_an_error_instead_of_swallowing_lines(self, tmp_path):
        csv_path = write(
            tmp_path / "m.csv",
            'audio/0.wav|Fine\naudio/1.wav|"Never closed\naudio/2.wav|Swallowed\n',
        )
        with pytest.raises(MetadataError, match=r"line 2 .*2-3.*quote") as excinfo:
            load_metadata(csv_path)
        assert "m.csv" in str(excinfo.value)

    def test_a_quote_that_is_not_closed_on_the_last_line_only_affects_that_row(self, tmp_path):
        csv_path = write(tmp_path / "m.csv", 'audio/0.wav|Fine\naudio/1.wav|"Never closed\n')
        assert [i["audio_file"] for i in load_metadata(csv_path)] == ["audio/0.wav", "audio/1.wav"]

    def test_a_file_that_is_not_utf8_is_an_error(self, tmp_path):
        csv_path = tmp_path / "m.csv"
        csv_path.write_bytes(b"audio/0.wav|Fine\naudio/1.wav|caf\xe9\n")  # Latin-1
        with pytest.raises(MetadataError, match="not valid UTF-8.*save it as UTF-8"):
            load_metadata(csv_path)

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(MetadataError, match="not found"):
            load_metadata(tmp_path / "nope.csv")


def test_sniff_delimiter_prefers_pipe_over_comma(tmp_path):
    csv_path = write(tmp_path / "m.csv", "audio/0.wav|Hello, world\n")
    assert sniff_delimiter(csv_path) == "|"


def test_sniff_delimiter_rejects_a_file_that_is_not_utf8(tmp_path):
    csv_path = tmp_path / "m.csv"
    csv_path.write_bytes(b"audio/0.wav|caf\xe9\n")
    with pytest.raises(MetadataError, match="not valid UTF-8"):
        sniff_delimiter(csv_path)


def test_sniff_delimiter_defaults_to_comma(tmp_path):
    csv_path = write(tmp_path / "m.csv", "single_column\n")
    assert sniff_delimiter(csv_path) == ","


class TestResolveAudioPath:
    def test_relative_to_dataset_dir(self, dataset_dir):
        assert (
            resolve_audio_path("audio/000001.wav", dataset_dir)
            == (dataset_dir / "audio" / "000001.wav").resolve()
        )

    def test_relative_to_audio_subdir(self, dataset_dir):
        assert resolve_audio_path("000002.wav", dataset_dir).name == "000002.wav"

    def test_only_the_file_name_is_used_as_last_resort(self, dataset_dir):
        assert resolve_audio_path("elsewhere/000003.wav", dataset_dir).name == "000003.wav"

    def test_absolute_path(self, dataset_dir, tmp_path):
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"RIFF")
        assert resolve_audio_path(str(outside), dataset_dir) == outside.resolve()

    def test_working_directory_does_not_shadow_the_dataset(
        self, dataset_dir, tmp_path, monkeypatch
    ):
        cwd = tmp_path / "cwd"
        (cwd / "audio").mkdir(parents=True)
        (cwd / "audio" / "000000.wav").write_bytes(b"decoy")
        monkeypatch.chdir(cwd)
        assert resolve_audio_path("audio/000000.wav", dataset_dir).is_relative_to(
            dataset_dir.resolve()
        )

    def test_missing_file(self, dataset_dir):
        with pytest.raises(FileNotFoundError, match="ghost.wav"):
            resolve_audio_path("audio/ghost.wav", dataset_dir)

    def test_a_directory_is_not_an_audio_file(self, dataset_dir):
        with pytest.raises(FileNotFoundError):
            resolve_audio_path("audio", dataset_dir)


class TestPickReferenceWavs:
    def test_returns_n_existing_absolute_paths(self, dataset_dir):
        picked = pick_reference_wavs(dataset_dir / "metadata.csv", dataset_dir, n=3)
        assert len(picked) == len(set(picked)) == 3
        assert all(Path(p).is_absolute() and Path(p).is_file() for p in picked)

    def test_same_seed_same_clips(self, dataset_dir):
        a = pick_reference_wavs(dataset_dir / "metadata.csv", dataset_dir, n=3, seed=7)
        b = pick_reference_wavs(dataset_dir / "metadata.csv", dataset_dir, n=3, seed=7)
        assert a == b

    def test_different_seeds_can_differ(self, dataset_dir):
        picks = {
            tuple(pick_reference_wavs(dataset_dir / "metadata.csv", dataset_dir, n=3, seed=s))
            for s in range(10)
        }
        assert len(picks) > 1

    def test_fewer_rows_than_requested(self, dataset_dir):
        assert len(pick_reference_wavs(dataset_dir / "metadata.csv", dataset_dir, n=50)) == 5

    def test_empty_metadata_is_an_error(self, tmp_path):
        csv_path = write(tmp_path / "m.csv", "")
        with pytest.raises(MetadataError, match="no usable rows"):
            pick_reference_wavs(csv_path, tmp_path)


class TestSplitItems:
    ITEMS = list(range(20))

    def test_same_seed_same_split(self):
        assert split_items(self.ITEMS, 0.25, seed=3) == split_items(self.ITEMS, 0.25, seed=3)

    def test_seed_defaults_to_42(self):
        assert split_items(self.ITEMS, 0.25) == split_items(self.ITEMS, 0.25, seed=42)

    def test_different_seeds_can_differ(self):
        evals = {tuple(split_items(self.ITEMS, 0.25, seed=s)[1]) for s in range(10)}
        assert len(evals) > 1

    def test_train_and_eval_are_disjoint_and_cover_the_input(self):
        train, eval_ = split_items(self.ITEMS, 0.3, seed=1)
        assert set(train).isdisjoint(eval_)
        assert sorted(train + eval_) == self.ITEMS

    def test_the_input_is_left_untouched(self):
        items = list(self.ITEMS)
        split_items(items, 0.3)
        assert items == self.ITEMS

    def test_the_items_are_shuffled(self):
        train, eval_ = split_items(self.ITEMS, 0.5, seed=1)
        assert train + eval_ != self.ITEMS

    def test_works_on_metadata_rows(self, tmp_path):
        csv_path = write(
            tmp_path / "m.csv", "".join(f"audio/{i}.wav|Text {i}\n" for i in range(10))
        )
        rows = load_metadata(csv_path)
        train, eval_ = split_items(rows, 0.2)
        assert (len(train), len(eval_)) == (8, 2)
        assert {r["audio_file"] for r in train}.isdisjoint(r["audio_file"] for r in eval_)

    @pytest.mark.parametrize(
        ("n", "fraction", "n_eval"),
        [
            (100, 0.1, 10),
            (20, 0.5, 10),
            (10, 0.25, 2),  # 2.5 rounds to the even 2
            (5, 0.1, 1),  # rounds to 0 but a positive fraction keeps at least one
            (2, 0.01, 1),
            (2, 0.9, 1),  # rounds to 2 but the train set is never empty
            (1, 0.5, 0),  # a single item stays in the train set
            (100, 0, 0),
            (0, 0.1, 0),
        ],
    )
    def test_eval_size(self, n, fraction, n_eval):
        train, eval_ = split_items(list(range(n)), fraction)
        assert len(eval_) == n_eval
        assert len(train) == n - n_eval

    def test_train_is_never_empty_for_a_non_empty_input(self):
        for n in range(1, 8):
            for fraction in (0, 0.1, 0.5, 0.99):
                assert split_items(list(range(n)), fraction)[0]

    def test_zero_fraction_gives_an_empty_eval_set(self):
        train, eval_ = split_items(self.ITEMS, 0)
        assert eval_ == []
        assert sorted(train) == self.ITEMS

    def test_empty_input(self):
        assert split_items([], 0.1) == ([], [])

    @pytest.mark.parametrize("fraction", [-0.1, 1, 1.5, float("nan")])
    def test_fraction_must_be_in_zero_one(self, fraction):
        with pytest.raises(ConfigError, match="eval_fraction"):
            split_items(self.ITEMS, fraction)
