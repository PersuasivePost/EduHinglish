"""
EduHinglish — Phase 3, Task 4: Train Intent Model
===================================================
Trains a spaCy textcat pipeline for student query intent.

Labels: explain_concept, compare_concepts, give_example,
        formula_request, definition

Usage:
  python training/train_intent.py
  python training/train_intent.py --epochs 50 --output models/intent_v2/
  python training/train_intent.py --arch cnn
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import compounding, minibatch

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_DATA = Path("training/data/intent_train.spacy")
DEV_DATA   = Path("training/data/intent_dev.spacy")
OUTPUT_DIR = Path("models/intent_v1")

INTENT_LABELS = [
    "explain_concept",
    "compare_concepts",
    "give_example",
    "formula_request",
    "definition",
]

SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _thick():
    print("  " + "═" * 50)

def _divider():
    print("  " + "─" * 50)


def load_examples(path: Path, nlp: spacy.Language) -> list[Example]:
    """Load .spacy file and convert to spaCy Example objects."""
    if not path.exists():
        print(f"  [ERROR] Data file not found: {path}")
        print("  Run prepare_spacy_data.py first.")
        sys.exit(1)

    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    examples = []
    for doc in docs:
        predicted = nlp.make_doc(doc.text)
        example = Example(predicted, doc)
        examples.append(example)
    return examples


def count_intents(docs: list[spacy.tokens.Doc]) -> Counter:
    counts: Counter = Counter()
    for doc in docs:
        for label, val in doc.cats.items():
            if val == 1:
                counts[label] += 1
    return counts


def pick_label(cats: dict) -> str | None:
    if not cats:
        return None
    return max(cats.items(), key=lambda item: item[1])[0]


def evaluate_textcat(nlp: spacy.Language, examples: list[Example]) -> tuple[float, dict]:
    """
    Compute macro-averaged accuracy across intents and return confusion matrix.
    """
    # confusion[true][pred] = count
    confusion = {t: {p: 0 for p in INTENT_LABELS} for t in INTENT_LABELS}
    per_label_totals = Counter()
    per_label_correct = Counter()

    for example in examples:
        ref_doc = example.reference
        pred_doc = nlp(ref_doc.text)

        true_label = pick_label(ref_doc.cats)
        pred_label = pick_label(pred_doc.cats)
        if true_label is None or pred_label is None:
            continue

        confusion[true_label][pred_label] += 1
        per_label_totals[true_label] += 1
        if true_label == pred_label:
            per_label_correct[true_label] += 1

    # Macro accuracy: average of per-label accuracies
    accuracies = []
    for lbl in INTENT_LABELS:
        total = per_label_totals[lbl]
        if total == 0:
            continue
        accuracies.append(per_label_correct[lbl] / total)

    macro_acc = (sum(accuracies) / len(accuracies) * 100) if accuracies else 0.0
    return macro_acc, confusion


def create_pipeline(arch: str) -> spacy.Language:
    nlp = spacy.blank("en")

    if arch == "cnn":
        textcat_config = {
            "model": {
                "@architectures": "spacy.TextCatCNN.v2",
                "exclusive_classes": True,
                "tok2vec": {
                    "@architectures": "spacy.Tok2Vec.v2",
                    "embed": {
                        "@architectures": "spacy.MultiHashEmbed.v2",
                        "width": 64,
                        "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
                        "rows": [2000, 1000, 1000, 1000],
                        "include_static_vectors": False,
                    },
                    "encode": {
                        "@architectures": "spacy.MaxoutWindowEncoder.v2",
                        "width": 64,
                        "depth": 2,
                        "window_size": 1,
                        "maxout_pieces": 2,
                    },
                },
            },
        }
    else:
        textcat_config = {
            "model": {
                "@architectures": "spacy.TextCatBOW.v2",
                "exclusive_classes": True,
                "ngram_size": 1,
                "no_output_layer": False,
            },
        }

    nlp.add_pipe("textcat", config=textcat_config)
    return nlp


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(max_epochs: int = 50, output_dir: Path = OUTPUT_DIR, arch: str = "bow") -> None:
    print()
    _thick()
    print("  EduHinglish — Phase 3: Train Intent Model")
    _thick()
    print()

    # ── 1. Create pipeline ────────────────────────────────────────────────
    print(f"  Creating pipeline: textcat ({arch})")
    nlp = create_pipeline(arch)

    # ── 2. Load data ───────────────────────────────────────────────────────
    ref_nlp = spacy.blank("en")
    train_doc_bin = DocBin().from_disk(TRAIN_DATA)
    dev_doc_bin = DocBin().from_disk(DEV_DATA)
    train_docs = list(train_doc_bin.get_docs(ref_nlp.vocab))
    dev_docs = list(dev_doc_bin.get_docs(ref_nlp.vocab))

    print(f"  Train docs: {len(train_docs)}")
    print(f"  Dev docs  : {len(dev_docs)}")

    total_intents = len(train_docs) + len(dev_docs)
    if total_intents < 60:
        print("\n  WARNING: Only {} intent entries found.".format(total_intents))
        print("  Training with BOW architecture (recommended for <200 examples).")
        print("  Consider adding more student queries to improve accuracy.")
    elif arch == "bow":
        print("\n  Training with BOW architecture (recommended for <200 examples).")
    else:
        print("\n  Training with CNN architecture (forced).")

    # ── 3. Initialize pipeline ─────────────────────────────────────────────
    train_examples = [Example(nlp.make_doc(doc.text), doc) for doc in train_docs]
    dev_examples = [Example(nlp.make_doc(doc.text), doc) for doc in dev_docs]

    textcat = nlp.get_pipe("textcat")
    for label in INTENT_LABELS:
        textcat.add_label(label)

    nlp.initialize(lambda: train_examples)

    # ── 4. Training loop ───────────────────────────────────────────────────
    _divider()
    print("  Training started")
    _divider()
    print()
    print(f"  {'Epoch':>8s} | {'Loss':>10s} | {'Dev Acc':>10s} | {'Best':>10s} |")
    print(f"  {'─'*8:>8s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┤")

    best_acc = 0.0
    best_epoch = 0

    best_model_dir = Path(tempfile.mkdtemp()) / "best_model"
    optimizer = nlp.create_optimizer()

    for epoch in range(1, max_epochs + 1):
        random.seed(SEED + epoch)
        random.shuffle(train_examples)

        losses = {}
        batches = minibatch(train_examples, size=compounding(1.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses, drop=0.2)

        dev_acc, _ = evaluate_textcat(nlp, dev_examples)
        epoch_loss = losses.get("textcat", 0.0)

        marker = ""
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_epoch = epoch
            nlp.to_disk(best_model_dir)
            marker = "✓"

        print(f"  {epoch:>8d} | {epoch_loss:>10.3f} | {dev_acc:>9.2f}% | {best_acc:>8.2f}% {marker}")

    # ── 5. Save best model ─────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    # Re-load best model if saved
    if best_model_dir.exists():
        best_nlp = spacy.load(best_model_dir)
        best_nlp.to_disk(output_dir)
    else:
        nlp.to_disk(output_dir)

    # ── 6. Confusion matrix on dev ─────────────────────────────────────────
    best_model = spacy.load(output_dir)
    dev_acc, confusion = evaluate_textcat(best_model, dev_examples)

    short = {
        "explain_concept": "explain",
        "compare_concepts": "compare",
        "give_example": "example",
        "formula_request": "formula",
        "definition": "definition",
    }

    print("\n  Confusion matrix (dev):")
    header = "Predicted →  " + "  ".join(f"{short[lbl]:<9s}" for lbl in INTENT_LABELS)
    print(f"  {header}")
    for true_lbl in INTENT_LABELS:
        row = [confusion[true_lbl][pred] for pred in INTENT_LABELS]
        row_str = "  " + f"{short[true_lbl]:<9s} " + "  ".join(f"{n:<9d}" for n in row)
        print(row_str)

    print("\n  ══════════════════════════════════")
    print("  Intent Model Training Complete")
    print(f"  Best dev accuracy: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"  Saved to: {output_dir}")
    print("  ══════════════════════════════════")


def main() -> None:
    parser = argparse.ArgumentParser(description="EduHinglish — Train Intent Model")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (default: 50)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--arch", type=str, default="bow", choices=["bow", "cnn"], help="Architecture: bow or cnn")
    args = parser.parse_args()

    train(max_epochs=args.epochs, output_dir=Path(args.output), arch=args.arch)


if __name__ == "__main__":
    main()
