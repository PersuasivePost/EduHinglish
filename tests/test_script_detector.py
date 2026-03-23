"""
EduHinglish — Tests: ScriptDetector, HinglishNormalizer, WordLevelLID
Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from script_detector import ScriptDetector, HinglishNormalizer, WordLevelLID


# ─────────────────────────────────────────────────────────────
# ScriptDetector
# ─────────────────────────────────────────────────────────────

class TestScriptDetector:
    @pytest.fixture
    def det(self):
        return ScriptDetector()

    def test_pure_english_detected_as_roman(self, det):
        result = det.detect("All living organisms are made up of cells.")
        assert result["overall_script"] == "ROMAN"

    def test_devanagari_text_detected(self, det):
        result = det.detect("यह एक वाक्य है।")
        assert result["overall_script"] == "DEVANAGARI"

    def test_mixed_script_detected(self, det):
        result = det.detect("Cell membrane एक selectively permeable membrane है।")
        assert result["overall_script"] == "MIXED"

    def test_roman_hinglish_detected_as_roman(self, det):
        result = det.detect("Sabhi living organisms cells se bane hote hain.")
        assert result["overall_script"] == "ROMAN"

    def test_returns_distribution(self, det):
        result = det.detect("Test sentence.")
        assert "distribution" in result
        assert "roman_pct" in result["distribution"]

    def test_word_scripts_populated(self, det):
        result = det.detect("Cell membrane hoti hai.")
        assert isinstance(result["word_scripts"], dict)
        assert len(result["word_scripts"]) > 0


# ─────────────────────────────────────────────────────────────
# HinglishNormalizer
# ─────────────────────────────────────────────────────────────

class TestHinglishNormalizer:
    @pytest.fixture
    def norm(self):
        return HinglishNormalizer()

    def test_kia_normalized_to_kya(self, norm):
        result, changes = norm.normalize_spelling("osmosis kia hota hai")
        assert "kya" in result.lower()
        assert len(changes) >= 1

    def test_h_normalized_to_hai(self, norm):
        result, changes = norm.normalize_spelling("cell kya h")
        assert "hai" in result.lower()

    def test_plz_expanded(self, norm):
        result, changes = norm.normalize_spelling("plz batao")
        assert "please" in result.lower()

    def test_no_changes_on_clean_input(self, norm):
        result, changes = norm.normalize_spelling("cell membrane hoti hai")
        assert len(changes) == 0

    def test_normalize_sentence_strips_extra_whitespace(self, norm):
        result = norm.normalize_sentence("cell   membrane   hoti  hai")
        assert "  " not in result

    def test_repeated_punctuation_collapsed(self, norm):
        result = norm.normalize_sentence("kya hai???")
        assert result.count("?") == 1


# ─────────────────────────────────────────────────────────────
# WordLevelLID
# ─────────────────────────────────────────────────────────────

class TestWordLevelLID:
    @pytest.fixture
    def lid(self):
        return WordLevelLID()

    # ── Single word tests ────────────────────────────────────
    def test_cells_is_english(self, lid):
        assert lid.identify_word("cells") == "EN"

    def test_mein_is_hindi(self, lid):
        assert lid.identify_word("mein") == "HI"

    def test_sir_is_universal(self, lid):
        assert lid.identify_word("Sir") == "UNIV"

    def test_number_is_universal(self, lid):
        assert lid.identify_word("1665") == "UNIV"

    def test_mitochondria_is_english(self, lid):
        assert lid.identify_word("mitochondria") == "EN"

    def test_aur_is_hindi(self, lid):
        assert lid.identify_word("aur") == "HI"

    def test_devanagari_word_is_hindi(self, lid):
        assert lid.identify_word("होता") == "HI"

    # ── Sentence-level tests ─────────────────────────────────
    def test_sentence_returns_words_list(self, lid):
        result = lid.identify_sentence("Cell mein nucleus hota hai.")
        assert isinstance(result["words"], list)
        assert len(result["words"]) > 0

    def test_cmi_in_range(self, lid):
        result = lid.identify_sentence("Cell mein nucleus hota hai.")
        assert 0 <= result["cmi"] <= 100

    def test_pure_english_has_low_cmi(self, lid):
        result = lid.identify_sentence("All living organisms are made up of cells.")
        # Nearly all English → CMI should be very low
        assert result["cmi"] < 20

    def test_hinglish_has_moderate_cmi(self, lid):
        result = lid.identify_sentence(
            "Cell membrane ek selectively permeable membrane hoti hai."
        )
        # Mixed → CMI should be in Hinglish range
        assert result["cmi"] > 10

    def test_distribution_keys_present(self, lid):
        result = lid.identify_sentence("Cell mein hai.")
        assert "language_distribution" in result
        dist = result["language_distribution"]
        # Should have at least EN and HI
        assert "EN" in dist or "HI" in dist