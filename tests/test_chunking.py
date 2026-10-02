"""Text splitting and the per-language chunk limits: pure functions, no model involved."""

import ast
import importlib.util
import random
from pathlib import Path

import pytest

from herald.tts.engine import (
    DEFAULT_MAX_CHARS,
    LANGUAGE_CHAR_LIMITS,
    SentenceBuffer,
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


def stream(deltas, **kwargs):
    """Feed ``deltas`` to a new SentenceBuffer, flush, and return every piece in order."""
    buffer = SentenceBuffer(**kwargs)
    pieces = []
    for delta in deltas:
        pieces += buffer.feed(delta)
    return pieces + buffer.flush()


def cut_randomly(text, rng):
    """``text`` as deltas cut at random places (also inside words and inside whitespace)."""
    cuts = sorted(rng.sample(range(1, len(text)), rng.randint(0, min(25, len(text) - 1))))
    return [text[i:j] for i, j in zip([0, *cuts], [*cuts, len(text)], strict=True)]


class TestSentenceBuffer:
    def test_a_sentence_is_released_only_once_whitespace_follows_it(self):
        buffer = SentenceBuffer()
        assert buffer.feed("Hello there, my friend.") == []  # "3." might become "3.14"
        assert buffer.feed(" ") == ["Hello there, my friend."]

    def test_flush_releases_a_last_sentence_without_whitespace(self):
        buffer = SentenceBuffer()
        assert buffer.feed("Hello there, my friend.") == []
        assert buffer.flush() == ["Hello there, my friend."]

    def test_a_decimal_point_is_not_a_sentence_end(self):
        assert stream(["The number is 3", ".", "14 and more."]) == ["The number is 3.14 and more."]

    def test_the_first_sentence_is_released_alone_as_soon_as_it_is_long_enough(self):
        buffer = SentenceBuffer(first_chars=12)
        assert buffer.feed("Hello there! How") == ["Hello there!"]  # 12 characters

    def test_a_short_first_sentence_waits_for_the_next_one(self):
        buffer = SentenceBuffer(first_chars=12)
        assert buffer.feed("Hi. How are you today? Fine") == ["Hi. How are you today?"]

    def test_later_short_sentences_are_merged_until_min_chars(self):
        text = "First sentence here. A. B. C. This one is long enough to stand alone. Last one."
        assert stream([text], min_chars=40) == [
            "First sentence here.",
            "A. B. C. This one is long enough to stand alone.",
            "Last one.",
        ]

    def test_without_a_minimum_every_sentence_is_a_piece(self):
        assert stream(["Aa. Bb. Cc."], min_chars=0, first_chars=0) == ["Aa.", "Bb.", "Cc."]

    def test_merging_never_exceeds_max_chars(self):
        text = "Ten chars. " + "x" * 30 + ". " + "y" * 30 + ". " + "z" * 30 + "."
        pieces = stream([text], max_chars=50, min_chars=40, first_chars=5)
        assert all(len(p) <= 50 for p in pieces)
        assert " ".join(pieces) == text

    def test_a_long_sentence_is_wrapped_at_word_boundaries(self):
        sentence = " ".join(["word"] * 40)  # 199 characters
        pieces = stream([sentence + ". Next one."], max_chars=50)
        assert len(pieces) > 2
        assert all(len(p) <= 50 for p in pieces)
        assert " ".join(pieces) == sentence + ". Next one."

    def test_a_word_longer_than_the_limit_is_broken_but_nothing_is_lost(self):
        pieces = stream(["a" * 120 + " and more."], max_chars=50)
        assert all(len(p) <= 50 for p in pieces)
        assert "".join(pieces).replace(" ", "") == "a" * 120 + "andmore."

    def test_closing_quotes_stay_with_their_sentence(self):
        text = 'He said "Stop." Then he left. She asked "Why?" He did not answer.'
        assert stream([text], min_chars=0, first_chars=0) == [
            'He said "Stop."',
            "Then he left.",
            'She asked "Why?"',
            "He did not answer.",
        ]

    def test_a_closing_quote_arriving_late_is_still_attached(self):
        buffer = SentenceBuffer(first_chars=0)
        assert buffer.feed('He said "Stop.') == []
        assert buffer.feed('"') == []
        assert buffer.feed(" Then") == ['He said "Stop."']

    def test_whitespace_is_normalised(self):
        deltas = ["Hello \n\n  there,\t friend.   ", "  Next   one."]
        assert stream(deltas) == ["Hello there, friend.", "Next one."]
        assert stream(deltas, first_chars=100) == ["Hello there, friend. Next one."]

    def test_blank_input_gives_nothing(self):
        buffer = SentenceBuffer()
        assert buffer.feed("") == [] and buffer.feed("  \n ") == []
        assert buffer.flush() == []

    def test_flush_starts_a_new_text(self):
        buffer = SentenceBuffer(first_chars=12)
        assert buffer.feed("A. ") == [] and buffer.flush() == ["A."]
        # The first piece of the new text follows the quick first-piece rule again.
        assert buffer.feed("Hello there! More") == ["Hello there!"]
        assert buffer.flush() == ["More"]
        assert buffer.flush() == []

    @pytest.mark.parametrize(
        "kwargs",
        [{"max_chars": 0}, {"min_chars": -1}, {"first_chars": -1}, {"first_clause_chars": -1}],
    )
    def test_invalid_sizes_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            SentenceBuffer(**kwargs)

    @pytest.mark.parametrize("first_clause_chars", [None, 0, 1, 24])
    def test_valid_clause_sizes_are_accepted(self, first_clause_chars):
        SentenceBuffer(first_clause_chars=first_clause_chars)

    SEA = "The sea is vast and old, and it remembers everything the shore forgets."

    def test_the_first_piece_is_released_at_a_comma(self):
        assert stream([self.SEA]) == [
            "The sea is vast and old,",  # 24 characters
            "and it remembers everything the shore forgets.",
        ]

    def test_the_first_clause_is_released_as_soon_as_whitespace_follows_the_comma(self):
        buffer = SentenceBuffer()
        assert buffer.feed("The sea is vast and old,") == []  # "1,5" might be a number
        assert buffer.feed(" and") == ["The sea is vast and old,"]
        assert buffer.flush() == ["and"]

    def test_a_comma_that_comes_too_early_is_not_a_break(self):
        text = "Well, this is a short opening sentence."  # "Well," is only 5 characters
        buffer = SentenceBuffer()
        assert buffer.feed(text + " ") == [text]  # released at the sentence end, as before

    def test_the_text_is_collected_up_to_a_comma_that_is_late_enough(self):
        text = "Well, this is a longer opening sentence, and it goes on."
        assert stream([text]) == ["Well, this is a longer opening sentence,", "and it goes on."]

    def test_one_character_less_than_the_minimum_is_not_enough(self):
        assert stream([self.SEA], first_clause_chars=25) == [
            "The sea is vast and old, and it remembers everything the shore forgets."
        ]

    def test_only_the_first_piece_is_cut_at_a_clause(self):
        text = "The sea is vast and old, and it remembers everything, the shore forgets, always."
        assert stream([text]) == [
            "The sea is vast and old,",
            "and it remembers everything, the shore forgets, always.",
        ]

    def test_later_pieces_are_not_cut_at_clauses(self):
        text = "First sentence here. The sea is vast and old, and it remembers everything it saw."
        assert stream([text]) == [
            "First sentence here.",
            "The sea is vast and old, and it remembers everything it saw.",
        ]

    def test_the_rule_applies_again_after_a_flush(self):
        buffer = SentenceBuffer()
        assert buffer.feed(self.SEA) == ["The sea is vast and old,"]
        assert buffer.flush() == ["and it remembers everything the shore forgets."]
        assert buffer.feed(self.SEA) == ["The sea is vast and old,"]  # a new text

    @pytest.mark.parametrize("disabled", [0, None])
    def test_the_clause_rule_can_be_turned_off(self, disabled):
        assert stream([self.SEA], first_clause_chars=disabled) == [self.SEA]

    def test_a_text_without_a_comma_is_released_at_its_sentence_end(self):
        buffer = SentenceBuffer()
        assert buffer.feed("The sea is vast and old and it remembers everything. Th") == [
            "The sea is vast and old and it remembers everything."
        ]

    def test_semicolons_and_colons_are_clause_breaks_too(self):
        assert stream(["The sea is vast and old; and it remembers."])[0] == (
            "The sea is vast and old;"
        )
        assert stream(["The sea is vast and old: it remembers."])[0] == "The sea is vast and old:"

    def test_a_clause_break_is_not_a_time_or_a_decimal(self):
        text = "It will be there at around 12:30 sharp not 3,5 minutes later than that."
        assert stream([text]) == [text]  # neither ":" nor "," is followed by a space

    def test_a_closing_quote_stays_with_its_clause(self):
        assert stream(['"The sea is vast and old," he said, remembering it all.']) == [
            '"The sea is vast and old,"',
            "he said, remembering it all.",
        ]

    def test_short_sentences_before_the_clause_are_part_of_the_first_piece(self):
        # "Hi." is too short to be released alone, so it waits for the clause that follows.
        text = "Hi. Well, that is a long story, indeed."
        assert stream([text]) == ["Hi. Well, that is a long story,", "indeed."]

    def test_a_clause_longer_than_max_chars_is_not_a_piece(self):
        text = "a" * 20 + " " + "b" * 20 + ", more words follow here."
        pieces = stream([text], max_chars=30)
        assert all(len(p) <= 30 for p in pieces)
        assert " ".join(pieces) == text

    def test_the_rest_of_the_sentence_follows_the_normal_rules(self):
        # After the first piece the minimum length applies again, so a short rest is held.
        text = "The sea is vast and old, and it waits. Then it sleeps all through the long night."
        assert stream([text]) == [
            "The sea is vast and old,",
            "and it waits. Then it sleeps all through the long night.",
        ]

    def test_the_result_does_not_depend_on_how_the_text_was_cut(self):
        """Property test: random texts, random cuts, always the same pieces."""
        rng = random.Random(1234)
        words = ["a", "bb", "ccc", "dog", "so", "I", "3.14", "Dr.", "ok", "no", "yes"]
        words += ["and,", "so,", "well;", "note:", 'said,"', "ok),", "12:30"]  # clause breaks
        ends = [".", "!", "?", "...", '."', '?"', ".)", ".”", "…", ",", ";", ""]
        spaces = [" ", " ", " ", "  ", "\n", " \t "]
        for _ in range(300):
            text = ""
            for _ in range(rng.randint(1, 12)):
                sentence = " ".join(rng.choices(words, k=rng.randint(1, 15)))
                text += sentence + rng.choice(ends) + rng.choice(spaces)
            text = text.rstrip() if rng.random() < 0.5 else text
            limits = {
                "max_chars": rng.choice([20, 45, 80, 250]),
                "min_chars": rng.choice([0, 15, 40]),
                "first_chars": rng.choice([0, 12]),
                "first_clause_chars": rng.choice([None, 0, 8, 24]),
            }
            reference = stream([text], **limits)

            assert stream(cut_randomly(text, rng), **limits) == reference
            assert stream(list(text), **limits) == reference  # one character at a time
            assert all(0 < len(piece) <= limits["max_chars"] for piece in reference)
            assert " ".join(reference) == " ".join(text.split())  # nothing lost or repeated

    def test_a_buffer_can_be_reused_for_many_texts(self):
        rng = random.Random(7)
        buffer = SentenceBuffer(max_chars=40)
        for _ in range(50):
            text = " ".join(rng.choices(["one.", "two", "three!", "four", "five?"], k=12))
            pieces = []
            for delta in cut_randomly(text, rng):
                pieces += buffer.feed(delta)
            pieces += buffer.flush()
            assert pieces == stream([text], max_chars=40)
