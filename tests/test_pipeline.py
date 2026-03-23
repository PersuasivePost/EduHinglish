"""
EduHinglish — Tests: Unified Pipeline
Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from pipeline import EduHinglishPipeline


@pytest.fixture(scope="module")
def pipeline():
    return EduHinglishPipeline()


ENGLISH_SENT  = "The cell membrane controls the movement of substances."
HINGLISH_SENT = "Cell membrane ek selectively permeable membrane hoti hai."
STUDENT_QUERY = "Sir, endoplasmic reticulum ka cell mein kya function hota hai?"


class TestAutoRouting:
    def test_english_routes_to_english_mode(self, pipeline):
        result = pipeline.process(ENGLISH_SENT)
        assert result["mode"] == "english"

    def test_hinglish_routes_to_hinglish_mode(self, pipeline):
        result = pipeline.process(HINGLISH_SENT)
        assert result["mode"] == "hinglish"

    def test_student_query_routes_to_hinglish(self, pipeline):
        result = pipeline.process(STUDENT_QUERY)
        assert result["mode"] == "hinglish"


class TestEnglishPipeline:
    def test_all_steps_present(self, pipeline):
        result = pipeline.process_english(ENGLISH_SENT)
        steps = result["steps"]
        assert "tokenization" in steps
        assert "stop_word_removal" in steps
        assert "stemming" in steps
        assert "lemmatization" in steps
        assert "pos_tagging" in steps

    def test_tokenization_non_empty(self, pipeline):
        result = pipeline.process_english(ENGLISH_SENT)
        assert len(result["steps"]["tokenization"]) > 0

    def test_stop_word_removal_reduces_tokens(self, pipeline):
        result = pipeline.process_english(ENGLISH_SENT)
        tokens = result["steps"]["tokenization"]
        kept   = result["steps"]["stop_word_removal"]["kept"]
        assert len(kept) <= len(tokens)


class TestHinglishPipeline:
    def test_all_steps_present(self, pipeline):
        result = pipeline.process_hinglish(HINGLISH_SENT)
        steps  = result["steps"]
        assert "script_detection"       in steps
        assert "normalization"          in steps
        assert "language_identification" in steps
        assert "tokenization"           in steps
        assert "stop_word_removal"      in steps
        assert "stemming_lemmatization" in steps

    def test_cmi_computed(self, pipeline):
        result = pipeline.process_hinglish(HINGLISH_SENT)
        cmi    = result["steps"]["language_identification"]["cmi"]
        assert isinstance(cmi, float)
        assert 0 <= cmi <= 100

    def test_normalization_output_is_string(self, pipeline):
        result = pipeline.process_hinglish(HINGLISH_SENT)
        norm   = result["steps"]["normalization"]["output"]
        assert isinstance(norm, str) and len(norm) > 0

    def test_spelling_normalization_applied(self, pipeline):
        messy  = "Sir osmosis kia hota h plz smjhao"
        result = pipeline.process_hinglish(messy)
        norm   = result["steps"]["normalization"]["output"].lower()
        # 'kia' should be normalized to 'kya'
        assert "kya" in norm

    def test_hindi_words_not_stemmed(self, pipeline):
        result = pipeline.process_hinglish(HINGLISH_SENT)
        for item in result["steps"]["stemming_lemmatization"]:
            if item["language"] == "HI":
                # Stem and lemma should equal original token for Hindi
                assert item["stem"]  == item["token"]
                assert item["lemma"] == item["token"]


class TestComparison:
    def test_compare_returns_both_keys(self, pipeline):
        result = pipeline.compare(ENGLISH_SENT, HINGLISH_SENT)
        assert "english"  in result
        assert "hinglish" in result

    def test_english_mode_in_comparison(self, pipeline):
        result = pipeline.compare(ENGLISH_SENT, HINGLISH_SENT)
        assert result["english"]["mode"]  == "english"
        assert result["hinglish"]["mode"] == "hinglish"