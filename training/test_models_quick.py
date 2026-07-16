"""
EduHinglish — Quick Model Tester (Interactive)
================================================
Test any of the 3 spaCy models interactively from the command line.

Usage:
  python training/test_models_quick.py --model lid      # Test LID model
  python training/test_models_quick.py --model ner      # Test NER model
  python training/test_models_quick.py --model intent   # Test Intent model
  python training/test_models_quick.py --model all      # Test all 3 models on same input
  python training/test_models_quick.py --model lid --text "Mitochondria ko powerhouse kehte hain"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import spacy

PROJECT_ROOT = Path(__file__).parent.parent

MODELS = {
    "lid": PROJECT_ROOT / "models" / "lid_v2",
    "ner": PROJECT_ROOT / "models" / "ner_v2",
    "intent": PROJECT_ROOT / "models" / "intent_v2",
}


def test_lid(nlp, text: str) -> None:
    print("\n── LID (Language ID) ──")
    doc = nlp(text)
    print(f"  {'Token':<20} {'Label':<8}")
    print(f"  {'─'*20} {'─'*8}")
    for tok in doc:
        if tok.is_space:
            continue
        label = tok.tag_ or "?"
        print(f"  {tok.text:<20} {label:<8}")


def test_ner(nlp, text: str) -> None:
    print("\n── NER (Named Entities) ──")
    doc = nlp(text)
    if not doc.ents:
        print("  (no entities detected)")
    else:
        print(f"  {'Entity':<25} {'Label':<15}")
        print(f"  {'─'*25} {'─'*15}")
        for ent in doc.ents:
            print(f"  {ent.text:<25} {ent.label_:<15}")


def test_intent(nlp, text: str) -> None:
    print("\n── Intent Classification ──")
    doc = nlp(text)
    # Sort by confidence descending
    sorted_cats = sorted(doc.cats.items(), key=lambda x: x[1], reverse=True)
    pred, score = sorted_cats[0]
    print(f"  Predicted: {pred} (confidence: {score:.2f})")
    print(f"\n  All scores:")
    for cat, s in sorted_cats:
        bar = "█" * int(s * 30)
        marker = " ◀" if cat == pred else ""
        print(f"    {cat:<20} {s:.3f} {bar}{marker}")


def interactive_mode(model_name: str) -> None:
    print("═" * 55)
    print(f"  EduHinglish — Interactive Model Tester ({model_name.upper()})")
    print("═" * 55)
    print("Type a Hinglish sentence and press Enter. Type 'q' to quit.\n")

    # Load models
    if model_name == "all":
        nlps = {name: spacy.load(path) for name, path in MODELS.items()}
        print(f"  Loaded: LID, NER, Intent\n")
    else:
        nlps = {model_name: spacy.load(MODELS[model_name])}
        print(f"  Loaded: {model_name.upper()}\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not text or text.lower() == "q":
            print("Bye!")
            break

        print(f'\nInput: "{text}"')

        if "lid" in nlps:
            test_lid(nlps["lid"], text)
        if "ner" in nlps:
            test_ner(nlps["ner"], text)
        if "intent" in nlps:
            test_intent(nlps["intent"], text)
        print()


def single_test(model_name: str, text: str) -> None:
    print("═" * 55)
    print(f"  EduHinglish — Model Test ({model_name.upper()})")
    print("═" * 55)
    print(f'\nInput: "{text}"')

    if model_name == "all":
        for name, path in MODELS.items():
            nlp = spacy.load(path)
            if name == "lid":
                test_lid(nlp, text)
            elif name == "ner":
                test_ner(nlp, text)
            elif name == "intent":
                test_intent(nlp, text)
    else:
        nlp = spacy.load(MODELS[model_name])
        if model_name == "lid":
            test_lid(nlp, text)
        elif model_name == "ner":
            test_ner(nlp, text)
        elif model_name == "intent":
            test_intent(nlp, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick model tester for EduHinglish spaCy models")
    parser.add_argument("--model", choices=["lid", "ner", "intent", "all"], default="all",
                        help="Which model to test (default: all)")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to test (if omitted, enters interactive mode)")
    parser.add_argument("text_positional", nargs="*", default=None,
                        help="Text to test (positional, alternative to --text)")
    args = parser.parse_args()

    # Support both: --text "..." and just "..." as positional
    text = args.text or (" ".join(args.text_positional) if args.text_positional else None)

    if text:
        single_test(args.model, text)
    else:
        interactive_mode(args.model)


if __name__ == "__main__":
    main()
