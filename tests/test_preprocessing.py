"""
EduHinglish — Tests: EnglishPreprocessor
Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from preprocessing import EnglishPreprocessor


@pytest.fixture(scope="module")
def proc():
    return EnglishPreprocessor()


BIOLOGY_SENTENCE = (
    "All living organisms are made up of cells, "
    "which are the fundamental structural and functional units of life."
)


class TestTokenization:
    def test_returns_list(self, proc):
        tokens = proc.tokenize(BIOLOGY_SENTENCE)
        assert isinstance(tokens, list)

    def test_non_empty(self, proc):
        tokens = proc.tokenize(BIOLOGY_SENTENCE)
        assert len(tokens) > 0

    def test_biology_term_present(self, proc):
        tokens = proc.tokenize(BIOLOGY_SENTENCE)
        lower = [t.lower() for t in tokens]
        assert "cells" in lower


class TestStopWordRemoval:
    def test_returns_tuple(self, proc):
        tokens = proc.tokenize(BIOLOGY_SENTENCE)
        result = proc.remove_stop_words(tokens)
        assert isinstance(result, tuple) and len(result) == 2

    def test_reduces_tokens(self, proc):
        tokens = proc.tokenize(BIOLOGY_SENTENCE)
        kept, _ = proc.remove_stop_words(tokens)
        assert len(kept) < len(tokens)

    def test_common_stop_words_removed(self, proc):
        tokens = ["The", "cell", "is", "a", "unit"]
        kept, removed = proc.remove_stop_words(tokens)
        removed_lower = [r.lower() for r in removed]
        assert "the" in removed_lower
        assert "is" in removed_lower

    def test_protected_words_kept(self, proc):
        tokens = ["not", "all", "cells", "are", "equal"]
        kept, _ = proc.remove_stop_words(tokens)
        kept_lower = [k.lower() for k in kept]
        assert "not" in kept_lower
        assert "all" in kept_lower


class TestStemming:
    def test_returns_two_lists(self, proc):
        tokens = ["cells", "permeable", "structural"]
        porter, lancaster = proc.stem_tokens(tokens)
        assert len(porter) == len(tokens)
        assert len(lancaster) == len(tokens)

    def test_cells_stemmed(self, proc):
        porter, _ = proc.stem_tokens(["cells"])
        # Porter should reduce "cells" to "cell"
        assert porter[0] == "cell"

    def test_running_stemmed(self, proc):
        porter, _ = proc.stem_tokens(["running"])
        assert porter[0] == "run"


class TestLemmatization:
    def test_returns_list(self, proc):
        result = proc.lemmatize_tokens(["cells", "organisms"])
        assert isinstance(result, list)

    def test_plural_to_singular(self, proc):
        result = proc.lemmatize_tokens(["cells"])
        assert result[0] == "cell"

    def test_verb_form(self, proc):
        result = proc.lemmatize_tokens(["running"])
        assert result[0] == "run"


class TestPOSTagging:
    def test_returns_tuple(self, proc):
        result = proc.pos_tag(["cell", "contains", "nucleus"])
        assert isinstance(result, tuple) and len(result) == 2

    def test_categories_keys(self, proc):
        _, cats = proc.pos_tag(["cell", "contains", "fundamental"])
        assert "nouns" in cats
        assert "verbs" in cats
        assert "adjectives" in cats


class TestSentenceSegmentation:
    def test_returns_list(self, proc):
        text = (
            "The cell is the fundamental unit of life. "
            "Cells were first discovered by Robert Hooke."
        )
        result = proc.segment_sentences(text)
        assert isinstance(result, list)

    def test_two_sentences_detected(self, proc):
        text = (
            "The cell is the fundamental unit of life. "
            "Cells were first discovered by Robert Hooke."
        )
        result = proc.segment_sentences(text)
        assert len(result) >= 2