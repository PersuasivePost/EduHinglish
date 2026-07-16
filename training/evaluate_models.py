"""
EduHinglish — Phase 3 Model Evaluation
======================================
Loads LID, NER, and Intent models and runs hardcoded test cases.

Usage:
  python training/evaluate_models.py
"""

from __future__ import annotations

from pathlib import Path
import spacy

PROJECT_ROOT = Path(__file__).parent.parent

LID_MODEL = PROJECT_ROOT / "models" / "lid_v2"
NER_MODEL = PROJECT_ROOT / "models" / "ner_v2"
INTENT_MODEL = PROJECT_ROOT / "models" / "intent_v2"

LID_TESTS = [
    "Mitochondria ko cell ka powerhouse kehte hain.",
    "Sir, photosynthesis kaise hoti hai?",
    "DNA replication mein enzyme kya role play karta hai?",
    "Mendel ne pea plants par experiments kiye the.",
    "Yeh process anaerobic respiration kehlata hai.",
]

INTENT_TESTS = [
    ("Sir, nucleus ka kaam kya hota hai?", "explain_concept"),
    ("Mitochondria aur chloroplast mein kya fark hai?", "compare_concepts"),
    ("Ek example do osmosis ka?", "give_example"),
]


def _thick():
    print("═" * 58)


def eval_lid(nlp: spacy.Language) -> None:
    print("\n── LID Model (models/lid_v2/) ──")
    for sent in LID_TESTS:
        print(f"Input: \"{sent}\"")
        doc = nlp(sent)
        for tok in doc:
            if tok.is_space:
                continue
            label = tok.tag_ or "?"
            print(f"  {tok.text:<12} → {label}")
        print()


def eval_ner(nlp: spacy.Language) -> None:
    print("\n── NER Model (models/ner_v2/) ──")
    for sent in LID_TESTS:
        print(f"Input: \"{sent}\"")
        doc = nlp(sent)
        if not doc.ents:
            print("  (no entities)")
        else:
            for ent in doc.ents:
                print(f"  [{ent.text}] {ent.label_}")
        print()


def eval_intent(nlp: spacy.Language) -> None:
    print("\n── Intent Model (models/intent_v2/) ──")
    for sent, expected in INTENT_TESTS:
        doc = nlp(sent)
        pred, score = max(doc.cats.items(), key=lambda x: x[1])
        marker = "✓" if pred == expected else "✗"
        print(f"Input: \"{sent}\"")
        print(f"  Predicted: {pred} (conf: {score:.2f})  {marker}")
        print()


def main() -> None:
    _thick()
    print("EduHinglish — Phase 3 Model Evaluation")
    _thick()

    # Load models
    lid_nlp = spacy.load(LID_MODEL)
    ner_nlp = spacy.load(NER_MODEL)
    intent_nlp = spacy.load(INTENT_MODEL)

    eval_lid(lid_nlp)
    eval_ner(ner_nlp)
    eval_intent(intent_nlp)


if __name__ == "__main__":
    main()
