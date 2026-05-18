"""
EduHinglish — Unified Preprocessing Pipeline
=============================================
Authors : Ashvatth & Jatin
Module  : M1 — Input Processing (complete)
Purpose : Combine all preprocessing modules into a single pipeline.
          Handles both English and Hinglish input automatically.
          Provides side-by-side comparison for mentor presentation.

Usage:
    python pipeline.py                # runs all demos
    python pipeline.py --sentence "Your Hinglish sentence here"
    python pipeline.py --demo 1       # run a specific demo (1-4)
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

# ── Import our modules ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing  import EnglishPreprocessor
from script_detector import ScriptDetector, HinglishNormalizer, WordLevelLID


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class EduHinglishPipeline:
    """
    End-to-end preprocessing pipeline for EduHinglish.

    Auto-routing:
        Input → script detection
            ↓
        ROMAN-only + no Hindi words  →  English pipeline
        Devanagari / Mixed / Hindi words detected  →  Hinglish pipeline

    English pipeline steps:
        Tokenization → Stop Word Removal → Stemming →
        Lemmatization → POS Tagging

    Hinglish pipeline steps:
        Script Detection → Normalization → Language ID →
        Tokenization → Language-Aware Stop Word Removal →
        Stemming+Lemmatization (English segments only) → POS Tagging
    """

    # Hindi stop words (short grammatical words OK to strip)
    HINDI_STOP_WORDS = {
        "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein",
        "ek", "yeh", "woh", "bhi", "hi", "toh", "ne", "par", "pe",
    }

    def __init__(self):
        print("  EduHinglish — Preprocessing Pipeline  v1.0")

        self.en_proc   = EnglishPreprocessor()
        self.detector  = ScriptDetector()
        self.normalizer = HinglishNormalizer()
        self.lid        = WordLevelLID()

        print("\n[OK] All modules loaded. Pipeline ready.")

    # ── Routing ───────────────────────────────────────────────────────────────

    def _is_hinglish(self, sentence: str) -> bool:
        """Return True if sentence contains Devanagari or Hindi Roman words."""
        script_info = self.detector.detect(sentence)
        if script_info["overall_script"] in ("DEVANAGARI", "MIXED"):
            return True
        # Check Roman Hindi words
        words      = sentence.split()
        hindi_count = sum(
            1 for w in words
            if self.lid.identify_word(w) == "HI"
        )
        return hindi_count > 0

    # ── English pipeline ──────────────────────────────────────────────────────

    def process_english(self, sentence: str) -> dict:
        """Run standard NLP pipeline on a monolingual English sentence."""
        print(f"\n[PIPELINE MODE] ENGLISH")
        print(f"  Input: \"{sentence[:65]}{'...' if len(sentence)>65 else ''}\"")

        steps: dict = {}

        tokens                       = self.en_proc.tokenize(sentence)
        steps["tokenization"]        = tokens

        kept, removed                = self.en_proc.remove_stop_words(tokens)
        steps["stop_word_removal"]   = {"kept": kept, "removed": removed}

        porter, lancaster            = self.en_proc.stem_tokens(kept)
        steps["stemming"]            = {"porter": porter, "lancaster": lancaster}

        lemmas                       = self.en_proc.lemmatize_tokens(kept)
        steps["lemmatization"]       = lemmas

        pos_tags, categories         = self.en_proc.pos_tag(kept)
        steps["pos_tagging"]         = {
            "tags"      : [(w, t) for w, t in pos_tags],
            "categories": categories,
        }

        return {"mode": "english", "input": sentence, "steps": steps}

    # ── Hinglish pipeline ─────────────────────────────────────────────────────

    def process_hinglish(self, sentence: str) -> dict:
        """Run Hinglish-aware pipeline on a code-mixed sentence."""
        print(f"\n[PIPELINE MODE] HINGLISH")
        print(f"  Input: \"{sentence[:65]}{'...' if len(sentence)>65 else ''}\"")

        steps: dict = {}

        # 1 — Script detection
        print("\n--- HINGLISH STEP 1 / 6 : Script Detection ---")
        script_result           = self.detector.detect(sentence)
        steps["script_detection"] = script_result

        # 2 — Normalization
        print("\n--- HINGLISH STEP 2 / 6 : Normalization ---")
        normalized              = self.normalizer.normalize_sentence(sentence)
        steps["normalization"]  = {"input": sentence, "output": normalized}

        # 3 — Word-level Language ID
        print("\n--- HINGLISH STEP 3 / 6 : Language Identification ---")
        lid_result                     = self.lid.identify_sentence(normalized)
        steps["language_identification"] = lid_result

        # 4 — Tokenization
        print("\n--- HINGLISH STEP 4 / 6 : Tokenization ---")
        tokens                   = self.en_proc.tokenize(normalized)
        steps["tokenization"]    = tokens

        # 5 — Language-aware stop word removal
        print("\n--- HINGLISH STEP 5 / 6 : Stop Word Removal (language-aware) ---")
        print(f"\n  {'Token':<25} {'Lang':<8} Action")
        print(f"  {'-'*25} {'-'*8} ------")

        kept, removed = [], []
        for tok in tokens:
            lower     = tok.lower()
            word_lang = self.lid.identify_word(tok)

            # Always discard lone punctuation
            if not tok.isalnum():
                removed.append((tok, "PUNCT"))
                print(f"  {tok:<25} {'PUNCT':<8} removed")
                continue

            # English stop words
            if word_lang == "EN" and lower in self.en_proc.stop_words and lower not in self.en_proc.protected_words:
                removed.append((tok, "EN_STOP"))
                print(f"  {tok:<25} {'EN_STOP':<8} removed")
                continue

            # Hindi stop words
            if word_lang == "HI" and lower in self.HINDI_STOP_WORDS:
                removed.append((tok, "HI_STOP"))
                print(f"  {tok:<25} {'HI_STOP':<8} removed")
                continue

            kept.append(tok)
            print(f"  {tok:<25} {word_lang:<8} kept")

        steps["stop_word_removal"] = {"kept": kept, "removed": removed}

        # 6 — Stemming & Lemmatization (English words only)
        print("\n--- HINGLISH STEP 6 / 6 : Stemming & Lemmatization (EN words only) ---")
        print(f"\n  {'Token':<22} {'Lang':<7} {'Stem':<22} {'Lemma':<22}")
        print(f"  {'-'*22} {'-'*7} {'-'*22} {'-'*22}")

        stem_lemma_results = []
        for tok in kept:
            lang  = self.lid.identify_word(tok)
            if lang == "EN":
                stem  = self.en_proc.stemmer_porter.stem(tok)
                lemma = self.en_proc.lemmatizer.lemmatize(tok.lower())
            else:
                stem  = tok   # Don't stem Hindi words
                lemma = tok

            marker = " ←" if stem != tok.lower() or lemma != tok.lower() else ""
            print(f"  {tok:<22} {lang:<7} {stem:<22} {lemma:<22}{marker}")

            stem_lemma_results.append({
                "token": tok, "language": lang,
                "stem": stem, "lemma": lemma,
            })

        steps["stemming_lemmatization"] = stem_lemma_results

        return {"mode": "hinglish", "input": sentence, "steps": steps}

    # ── Auto-routing ──────────────────────────────────────────────────────────

    def process(self, sentence: str) -> dict:
        """Auto-detect language type and route to correct pipeline."""
        if self._is_hinglish(sentence):
            return self.process_hinglish(sentence)
        else:
            return self.process_english(sentence)

    # ── Comparison ────────────────────────────────────────────────────────────

    def compare(self, english: str, hinglish: str) -> dict:
        """
        Process both versions and show a side-by-side summary.
        This is the PRIMARY output for the mentor presentation.
        """
        print(f"\n[COMPARISON — English vs Hinglish")
        print(f"\n  English : \"{english}\"")
        print(f"  Hinglish: \"{hinglish}\"")

        en_result = self.process_english(english)
        hi_result = self.process_hinglish(hinglish)

        # ── Summary table ─────────────────────────────────────
        print("COMPARISON SUMMARY")

        en_tok   = en_result["steps"]["tokenization"]
        hi_tok   = hi_result["steps"]["tokenization"]
        en_kept  = en_result["steps"]["stop_word_removal"]["kept"]
        hi_kept  = hi_result["steps"]["stop_word_removal"]["kept"]
        cmi      = hi_result["steps"]["language_identification"]["cmi"]
        script   = hi_result["steps"]["script_detection"]["overall_script"]
        en_stems = en_result["steps"]["stemming"]["porter"]
        en_lem   = en_result["steps"]["lemmatization"]

        rows = [
            ("Aspect",                    "English",                        "Hinglish"),
            ("-" * 30,                    "-" * 30,                         "-" * 30),
            ("Total tokens",              str(len(en_tok)),                 str(len(hi_tok))),
            ("After stop word removal",   str(len(en_kept)),                str(len(hi_kept))),
            ("Script",                    "ROMAN (monolingual)",            script),
            ("Code-Mixing Index (CMI)",   "N/A",                            f"{cmi}%"),
            ("Porter stems (EN words)",   str(en_stems),                    "EN words only"),
            ("Lemmas (EN words)",         str(en_lem),                      "EN words only"),
        ]
        for row in rows:
            print(f"  {row[0]:<32} {row[1]:<32} {row[2]}")

        return {"english": en_result, "hinglish": hi_result}

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, data, output_path: str):
        """Save pipeline output as JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        serializable = json.loads(json.dumps(data, default=str, ensure_ascii=False))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

DEMOS = {
    1: {
        "title": "English vs Hinglish comparison (cell membrane)",
        "english" : "The cell membrane is a selectively permeable membrane that controls the movement of substances into and out of the cell.",
        "hinglish": "Cell membrane ek selectively permeable membrane hoti hai jo substances ke movement ko cell ke andar aur bahar control karti hai.",
    },
    2: {
        "title": "Student's Hinglish doubt (endoplasmic reticulum)",
        "hinglish": "Sir, endoplasmic reticulum ka cell mein kya function hota hai?",
    },
    3: {
        "title": "Spelling-variant / abbreviation normalization",
        "hinglish": "Sir osmosis kia hota h? plz smjhao",
    },
    4: {
        "title": "Mixed-script input (Devanagari + Roman)",
        "hinglish": "Mitochondria को cell ka powerhouse kaha jaata hai kyunki yeh ATP ke form mein energy produce karte hain.",
    },
}


def run_demo(pipeline: EduHinglishPipeline, demo_id: int) -> dict:
    d = DEMOS[demo_id]
    print(f"\n[DEMO {demo_id} — {d['title']}")

    if "english" in d and "hinglish" in d:
        return pipeline.compare(d["english"], d["hinglish"])
    else:
        return pipeline.process(d["hinglish"])


# ─────────────────────────────────────────────────────────────────────────────
# CLI / direct run
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EduHinglish Preprocessing Pipeline")
    parser.add_argument("--sentence", type=str, default=None,
                        help="Single sentence to process")
    parser.add_argument("--demo", type=int, default=None,
                        choices=[1, 2, 3, 4],
                        help="Run a specific demo (1-4)")
    args = parser.parse_args()

    pipeline = EduHinglishPipeline()
    results  = {}

    if args.sentence:
        results["custom"] = pipeline.process(args.sentence)

    elif args.demo:
        results[f"demo_{args.demo}"] = run_demo(pipeline, args.demo)

    else:
        # Run ALL demos
        for demo_id in DEMOS:
            results[f"demo_{demo_id}"] = run_demo(pipeline, demo_id)

    # Save
    out_path = (
        Path(__file__).parent.parent
        / "outputs" / "pipeline_results"
        / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    print("ALL DEMOS COMPLETE")


if __name__ == "__main__":
    main()