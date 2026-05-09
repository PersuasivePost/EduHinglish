"""
EduHinglish — Phase 3, Task 2: Train LID (Language ID) Model
==============================================================
Trains a spaCy tok2vec + tagger pipeline for word-level
language identification on Hinglish text.

Labels: HI, EN, NE, UNIV, MIX

Usage:
  python training/train_lid.py
  python training/train_lid.py --epochs 50 --output models/lid_v2/
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import compounding, minibatch

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_DATA = Path("training/data/lid_train.spacy")
DEV_DATA   = Path("training/data/lid_dev.spacy")
OUTPUT_DIR = Path("models/lid_v1")

LID_LABELS = ["HI", "EN", "NE", "UNIV", "MIX"]
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
        # The gold doc has tag_ set; create Example(predicted, reference)
        predicted = nlp.make_doc(doc.text)
        example = Example(predicted, doc)
        examples.append(example)
    return examples


def evaluate_tagger(nlp: spacy.Language, examples: list[Example]) -> dict:
    """
    Evaluate tagger on dev examples.
    Returns overall accuracy and per-label precision/recall/F1.
    """
    correct = 0
    total = 0
    # Per-label tracking
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    for example in examples:
        ref_doc = example.reference
        pred_doc = nlp(ref_doc.text)

        # Align tokens — since we use whitespace tokenisation, they should match
        ref_tags = [tok.tag_ for tok in ref_doc]
        pred_tags = [tok.tag_ for tok in pred_doc]

        min_len = min(len(ref_tags), len(pred_tags))
        for i in range(min_len):
            total += 1
            ref_t = ref_tags[i]
            pred_t = pred_tags[i]
            if ref_t == pred_t:
                correct += 1
                tp[ref_t] += 1
            else:
                fp[pred_t] += 1
                fn[ref_t] += 1

    accuracy = (correct / total * 100) if total > 0 else 0.0

    # Compute per-label metrics
    label_metrics = {}
    for lbl in LID_LABELS:
        p_denom = tp[lbl] + fp[lbl]
        r_denom = tp[lbl] + fn[lbl]
        precision = (tp[lbl] / p_denom * 100) if p_denom > 0 else 0.0
        recall = (tp[lbl] / r_denom * 100) if r_denom > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        label_metrics[lbl] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp[lbl] + fn[lbl],
        }

    return {"accuracy": accuracy, "labels": label_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def create_pipeline() -> spacy.Language:
    """Create a fresh spaCy pipeline with tok2vec + tagger."""
    nlp = spacy.blank("en")

    # Add tok2vec
    tok2vec_config = {
        "model": {
            "@architectures": "spacy.Tok2Vec.v2",
            "embed": {
                "@architectures": "spacy.MultiHashEmbed.v2",
                "width": 96,
                "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
                "rows": [5000, 2500, 2500, 2500],
                "include_static_vectors": False,
            },
            "encode": {
                "@architectures": "spacy.MaxoutWindowEncoder.v2",
                "width": 96,
                "depth": 4,
                "window_size": 1,
                "maxout_pieces": 3,
            },
        }
    }
    nlp.add_pipe("tok2vec", config=tok2vec_config)

    # Add tagger
    tagger_config = {
        "model": {
            "@architectures": "spacy.Tagger.v2",
            "nO": None,
            "tok2vec": {
                "@architectures": "spacy.Tok2VecListener.v1",
                "width": 96,
            },
        }
    }
    nlp.add_pipe("tagger", config=tagger_config)

    return nlp


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(
    max_epochs: int = 30,
    patience: int = 5,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Main training function."""
    print()
    _thick()
    print("  EduHinglish — Phase 3: Train LID Model")
    _thick()
    print()

    # ── 1. Create pipeline ─────────────────────────────────────────────────
    print("  Creating pipeline: tok2vec + tagger")
    nlp = create_pipeline()

    # ── 2. Load data ───────────────────────────────────────────────────────
    print(f"  Loading train data: {TRAIN_DATA}")
    # We need a reference nlp (with vocab) to load DocBin
    ref_nlp = spacy.blank("en")
    train_doc_bin = DocBin().from_disk(TRAIN_DATA)
    dev_doc_bin = DocBin().from_disk(DEV_DATA)

    train_docs = list(train_doc_bin.get_docs(ref_nlp.vocab))
    dev_docs = list(dev_doc_bin.get_docs(ref_nlp.vocab))

    print(f"  Train docs: {len(train_docs)}")
    print(f"  Dev docs  : {len(dev_docs)}")
    print(f"  Loading dev data  : {DEV_DATA}")

    # ── 3. Initialize the pipeline ─────────────────────────────────────────
    print("\n  Initializing pipeline...")

    # Create Example objects for initialization (spaCy needs these to infer label set)
    train_examples = []
    for doc in train_docs:
        pred = nlp.make_doc(doc.text)
        example = Example(pred, doc)
        train_examples.append(example)

    dev_examples = []
    for doc in dev_docs:
        pred = nlp.make_doc(doc.text)
        example = Example(pred, doc)
        dev_examples.append(example)

    nlp.initialize(lambda: train_examples)

    # Check labels learned
    tagger = nlp.get_pipe("tagger")
    print(f"  Labels: {tagger.labels}")
    print()

    # ── 4. Training loop ───────────────────────────────────────────────────
    _divider()
    print("  Training started")
    _divider()
    print()
    print(f"  {'Epoch':>8s} | {'Loss':>10s} | {'Dev Acc':>10s} | {'Best':>10s} |")
    print(f"  {'─'*8:>8s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┤")

    best_acc = 0.0
    epochs_without_improvement = 0
    best_epoch = 0

    # Use a temp dir to save best model, then copy to final output
    best_model_dir = Path(tempfile.mkdtemp()) / "best_model"

    optimizer = nlp.create_optimizer()

    for epoch in range(1, max_epochs + 1):
        random.seed(SEED + epoch)
        random.shuffle(train_examples)

        # Accumulate losses
        losses = {}
        batches = minibatch(train_examples, size=compounding(1.0, 32.0, 1.001))

        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses, drop=0.2)

        # Evaluate on dev set
        eval_result = evaluate_tagger(nlp, dev_examples)
        dev_acc = eval_result["accuracy"]
        epoch_loss = losses.get("tagger", 0.0)

        # Check improvement
        marker = ""
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            marker = " ✓ new best"
            # Save best model
            if best_model_dir.exists():
                import shutil
                shutil.rmtree(best_model_dir)
            nlp.to_disk(best_model_dir)
        else:
            epochs_without_improvement += 1

        print(
            f"  {epoch:>5d}/{max_epochs:<2d} | "
            f"{epoch_loss:>10.4f} | "
            f"{dev_acc:>9.1f}% | "
            f"{best_acc:>9.1f}% |{marker}"
        )

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n  ⏹ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ── 5. Save best model ─────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy best model to output
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(best_model_dir, output_dir)

    # Clean up temp
    shutil.rmtree(best_model_dir.parent, ignore_errors=True)

    # ── 6. Final evaluation on best model ──────────────────────────────────
    print()
    _divider()
    print("  Running final evaluation on best model...")
    _divider()

    best_nlp = spacy.load(output_dir)
    final_eval = evaluate_tagger(best_nlp, dev_examples)

    # Per-label table
    print()
    print(f"  {'Label':>6s} | {'Precision':>10s} | {'Recall':>10s} | {'F1':>10s} | {'Support':>8s}")
    print(f"  {'─'*6:>6s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┼─{'─'*8:>8s}")
    for lbl in LID_LABELS:
        m = final_eval["labels"][lbl]
        print(
            f"  {lbl:>6s} | "
            f"{m['precision']:>9.1f}% | "
            f"{m['recall']:>9.1f}% | "
            f"{m['f1']:>9.1f}% | "
            f"{m['support']:>8d}"
        )

    # ── 7. Summary ─────────────────────────────────────────────────────────
    baseline = 75.0  # rule-based baseline from Phase 2
    improvement = best_acc - baseline

    print()
    _thick()
    print("  LID Model Training Complete")
    _thick()
    print(f"  Best dev accuracy    : {best_acc:.1f}%")
    print(f"  Best epoch           : {best_epoch}")
    print(f"  Saved to             : {output_dir}/")
    print(f"  Baseline (rule-based): ~{baseline:.0f}%")
    print(f"  Improvement          : {'+' if improvement >= 0 else ''}{improvement:.1f}pp")
    _thick()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train_lid",
        description="Train spaCy LID (Language ID) tagger on EduHinglish data.",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=30,
        help="Maximum training epochs (default: 30).",
    )
    parser.add_argument(
        "--patience", "-p",
        type=int,
        default=5,
        help="Early stopping patience (default: 5).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for saved model (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    train(
        max_epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
