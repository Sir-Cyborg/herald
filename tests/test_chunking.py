"""Text splitting and the per-language chunk limits: pure functions, no model involved."""

import ast
import importlib.util
from pathlib import Path

import pytest

from herald.tts.engine import (
    DEFAULT_MAX_CHARS,
    LANGUAGE_CHAR_LIMITS,
    default_max_chars,
    split_text,
)


class TestSplitText:
    def test_short_text_is_one_chunk(self):
        assert split_text("Hello there.") == ["Hello there."]

    def test_blank_text_gives_no_chunks(self):
        assert split_text("") == []
        assert split_text("  \n\t ") == []

    def test_whitespace_is_normalised(self):
        assert split_text("Hello   there,\n  friend.") == ["Hello there, friend."]

    def test_exactly_at_the_limit_is_not_split(self):
        text = "a" * 20
        assert split_text(text, max_chars=20) == [text]

    def test_long_text_is_cut_at_sentence_boundaries(self):
        sentences = [f"This is sentence number {i}." for i in range(30)]
        chunks = split_text(" ".join(sentences), max_chars=100)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)
        assert all(c.endswith(".") for c in chunks)
        assert " ".join(chunks) == " ".join(sentences)

    def test_sentences_are_packed_greedily(self):
        chunks = split_text("One two. Three four. Five six. Seven eight.", max_chars=22)
        assert chunks == ["One two. Three four.", "Five six. Seven eight."]

    def test_oversized_sentence_is_wrapped_at_words(self):
        sentence = " ".join(["word"] * 40)  # 199 characters, no sentence break
        chunks = split_text(sentence, max_chars=50)
        assert len(chunks) > 1
        assert all(len(c) <= 50 for c in chunks)
        assert " ".join(chunks) == sentence

    def test_oversized_sentence_between_normal_ones(self):
        long_sentence = " ".join(["word"] * 30) + "."
        chunks = split_text(f"Short one. {long_sentence} Short two.", max_chars=60)
        assert chunks[0] == "Short one."
        assert chunks[-1] == "Short two."
        assert all(len(c) <= 60 for c in chunks)

    @pytest.mark.parametrize("mark", ["!", "?", "…", "。"])
    def test_other_sentence_terminators(self, mark):
        text = f"First part{mark} Second part{mark}"
        assert split_text(text, max_chars=20) == [f"First part{mark}", f"Second part{mark}"]

    def test_default_limit_matches_the_xtts_limit(self):
        assert DEFAULT_MAX_CHARS == 250
        chunks = split_text("Word. " * 100)
        assert all(len(c) <= 250 for c in chunks)

    def test_limit_must_be_positive(self):
        with pytest.raises(ValueError):
            split_text("text", max_chars=0)


class TestSplitTextDialogue:
    """Closing quotes and brackets stay with their sentence, and do not hide its end."""

    @pytest.mark.parametrize("end", ['."', '?"', '!"', "!'", ".)", ".”", ".’", ".»", '…"'])
    def test_sentence_ends_followed_by_a_closing_mark(self, end):
        text = f"First part{end} Second part{end}"
        # Without the fix this would be one 26-character "sentence" wrapped at a word.
        assert split_text(text, max_chars=20) == [f"First part{end}", f"Second part{end}"]

    def test_dialogue_is_cut_between_the_lines(self):
        text = 'He said "Stop." Then he left. She asked "Why?" He did not answer.'
        assert split_text(text, max_chars=20) == [
            'He said "Stop."',
            "Then he left.",
            'She asked "Why?"',
            "He did not answer.",
        ]

    def test_several_closing_marks_in_a_row(self):
        text = '(He said "no.") Then he left.'
        assert split_text(text, max_chars=20) == ['(He said "no.")', "Then he left."]

    def test_an_opening_quote_starts_the_next_sentence(self):
        assert split_text('"One." "Two."', max_chars=8) == ['"One."', '"Two."']

    def test_a_closing_mark_not_followed_by_a_space_is_not_a_boundary(self):
        text = '"Stop.", he said. Then he left.'
        assert split_text(text, max_chars=20) == ['"Stop.", he said.', "Then he left."]

    def test_a_long_quoted_line_is_still_wrapped_to_the_limit(self):
        text = '"' + " ".join(["word"] * 30) + '." Short.'
        chunks = split_text(text, max_chars=40)
        assert all(len(c) <= 40 for c in chunks)
        assert chunks[-1] == "Short."
        assert " ".join(chunks) == text


class TestDefaultMaxChars:
    @pytest.mark.parametrize(
        ("language", "limit"),
        [("en", 250), ("es", 239), ("it", 213), ("pt", 203), ("ru", 182), ("hi", 150)]
        + [("ko", 95), ("zh", 82), ("ja", 71)],
    )
    def test_each_language_has_its_own_limit(self, language, limit):
        assert default_max_chars(language) == limit

    @pytest.mark.parametrize("language", ["fr", "de", "nl"])
    def test_never_more_than_the_default_even_if_the_tokenizer_allows_it(self, language):
        assert LANGUAGE_CHAR_LIMITS[language] > DEFAULT_MAX_CHARS
        assert default_max_chars(language) == DEFAULT_MAX_CHARS

    def test_never_above_the_default_for_any_language(self):
        assert all(default_max_chars(lang) <= DEFAULT_MAX_CHARS for lang in LANGUAGE_CHAR_LIMITS)

    @pytest.mark.parametrize("language", ["IT", "It", " it ", "it-IT", "it_it"])
    def test_case_and_region_are_ignored(self, language):
        assert default_max_chars(language) == 213

    @pytest.mark.parametrize("language", ["zh-cn", "zh_CN", "ZH-CN", "zh-tw"])
    def test_chinese_variants_are_chinese(self, language):
        assert default_max_chars(language) == 82

    @pytest.mark.parametrize("language", ["pt-br", "pt_BR", "PT-pt"])
    def test_portuguese_variants_are_portuguese(self, language):
        assert default_max_chars(language) == 203

    @pytest.mark.parametrize("language", ["xx", "klingon", "", "-"])
    def test_unknown_languages_get_the_default(self, language):
        assert default_max_chars(language) == DEFAULT_MAX_CHARS

    def test_table_matches_the_xtts_tokenizer(self):
        """Drift guard: compare with ``char_limits`` in the installed coqui-tts, if any.

        The source is parsed rather than imported: importing TTS takes several seconds.
        """
        spec = importlib.util.find_spec("TTS")  # imports nothing for a top-level name
        if spec is None or not spec.submodule_search_locations:
            pytest.skip("coqui-tts is not installed")
        source = Path(spec.submodule_search_locations[0]) / "tts/layers/xtts/tokenizer.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        tables = [
            ast.literal_eval(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "char_limits" for t in node.targets)
        ]
        assert tables, f"no char_limits table found in {source}: update this test"
        assert LANGUAGE_CHAR_LIMITS == tables[0]
