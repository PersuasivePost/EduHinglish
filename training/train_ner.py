"""
EduHinglish — Phase 3, Task 3: Train NER (Science Entity) Model
=================================================================
Trains a spaCy tok2vec + ner pipeline for science named-entity
recognition on Hinglish text.

Labels: SCIENTIST, ORGANELLE, PROCESS, CONCEPT, INSTRUMENT

Usage:
  python training/train_ner.py
  python training/train_ner.py --epochs 60 --output models/ner_v2/
"""

from __future__ import annotations

import argparse
import random
import shutil
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

TRAIN_DATA = Path("training/data/ner_train.spacy")
DEV_DATA   = Path("training/data/ner_dev.spacy")
OUTPUT_DIR = Path("models/ner_v1")

NER_LABELS = ["SCIENTIST", "ORGANELLE", "PROCESS", "CONCEPT", "INSTRUMENT"]
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


def evaluate_ner(nlp: spacy.Language, examples: list[Example]) -> dict:
    """
    Evaluate NER on dev examples.
    Returns overall F1 and per-label precision/recall/F1.
    """
    # Per-label tracking
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    for example in examples:
        ref_doc = example.reference
        pred_doc = nlp(ref_doc.text)

        # Build sets of (start_char, end_char, label) for comparison
        ref_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in ref_doc.ents}
        pred_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents}

        # True positives: in both ref and pred
        for ent in ref_ents & pred_ents:
            tp[ent[2]] += 1
        # False positives: in pred but not ref
        for ent in pred_ents - ref_ents:
            fp[ent[2]] += 1
        # False negatives: in ref but not pred
        for ent in ref_ents - pred_ents:
            fn[ent[2]] += 1

    # Compute per-label metrics
    label_metrics = {}
    total_tp = total_fp = total_fn = 0

    for lbl in NER_LABELS:
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
        total_tp += tp[lbl]
        total_fp += fp[lbl]
        total_fn += fn[lbl]

    # Micro-averaged F1
    micro_p = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
    micro_r = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    return {
        "f1": micro_f1,
        "precision": micro_p,
        "recall": micro_r,
        "labels": label_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def create_pipeline() -> spacy.Language:
    """Create a fresh spaCy pipeline with tok2vec + ner."""
    nlp = spacy.blank("en")

    # Add tok2vec — same architecture as LID (shared design, separate weights)
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

    # Add NER
    ner_config = {
        "model": {
            "@architectures": "spacy.TransitionBasedParser.v2",
            "state_type": "ner",
            "extra_state_tokens": False,
            "hidden_width": 64,
            "maxout_pieces": 2,
            "use_upper": True,
            "nO": None,
            "tok2vec": {
                "@architectures": "spacy.Tok2VecListener.v1",
                "width": 96,
            },
        }
    }
    nlp.add_pipe("ner", config=ner_config)

    return nlp


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(
    max_epochs: int = 40,
    patience: int = 5,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Main training function."""
    print()
    _thick()
    print("  EduHinglish — Phase 3: Train NER Model")
    _thick()
    print()

    # ── 1. Create pipeline ─────────────────────────────────────────────────
    print("  Creating pipeline: tok2vec + ner")
    nlp = create_pipeline()

    # ── 2. Load data ───────────────────────────────────────────────────────
    print(f"  Loading train data: {TRAIN_DATA}")
    ref_nlp = spacy.blank("en")
    train_doc_bin = DocBin().from_disk(TRAIN_DATA)
    dev_doc_bin = DocBin().from_disk(DEV_DATA)

    train_docs = list(train_doc_bin.get_docs(ref_nlp.vocab))
    dev_docs = list(dev_doc_bin.get_docs(ref_nlp.vocab))

    print(f"  Train docs: {len(train_docs)}")
    print(f"  Dev docs  : {len(dev_docs)}")
    print(f"  Loading dev data  : {DEV_DATA}")

    # Count entity spans in training data
    train_ent_counts: Counter = Counter()
    for doc in train_docs:
        for ent in doc.ents:
            train_ent_counts[ent.label_] += 1
    print()
    print("  Entity spans in training data:")
    for lbl in NER_LABELS:
        print(f"    {lbl:14s}: {train_ent_counts[lbl]}")
    total_train_ents = sum(train_ent_counts.values())
    print(f"    {'TOTAL':14s}: {total_train_ents}")

    if total_train_ents < 50:
        print()
        print("  ⚠ WARNING: Very few entity spans in training data.")
        print("  NER model may not generalise well. Consider adding more data.")

    # ── 3. Initialize the pipeline ─────────────────────────────────────────
    print("\n  Initializing pipeline...")

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
    ner = nlp.get_pipe("ner")
    print(f"  Labels: {ner.labels}")
    print()

    # ── 4. Training loop ───────────────────────────────────────────────────
    _divider()
    print("  Training started")
    _divider()
    print()
    print(f"  {'Epoch':>8s} | {'Loss':>10s} | {'Dev F1':>10s} | {'Best F1':>10s} |")
    print(f"  {'─'*8:>8s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┤")

    best_f1 = 0.0
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
        eval_result = evaluate_ner(nlp, dev_examples)
        dev_f1 = eval_result["f1"]
        epoch_loss = losses.get("ner", 0.0)

        # Check improvement
        marker = ""
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            marker = " ✓ new best"
            # Save best model
            if best_model_dir.exists():
                shutil.rmtree(best_model_dir)
            nlp.to_disk(best_model_dir)
        else:
            epochs_without_improvement += 1

        print(
            f"  {epoch:>5d}/{max_epochs:<2d} | "
            f"{epoch_loss:>10.4f} | "
            f"{dev_f1:>9.1f}% | "
            f"{best_f1:>9.1f}% |{marker}"
        )

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n  ⏹ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ── 5. Save best model ─────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    final_eval = evaluate_ner(best_nlp, dev_examples)

    # Per-label table
    print()
    print(f"  {'Entity':>14s} | {'Precision':>10s} | {'Recall':>10s} | {'F1':>10s} | {'Support':>8s}")
    print(f"  {'─'*14:>14s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┼─{'─'*8:>8s}")
    for lbl in NER_LABELS:
        m = final_eval["labels"][lbl]
        print(
            f"  {lbl:>14s} | "
            f"{m['precision']:>9.1f}% | "
            f"{m['recall']:>9.1f}% | "
            f"{m['f1']:>9.1f}% | "
            f"{m['support']:>8d}"
        )

    # Micro-averaged row
    print(f"  {'─'*14:>14s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┼─{'─'*10:>10s}─┼─{'─'*8:>8s}")
    print(
        f"  {'MICRO AVG':>14s} | "
        f"{final_eval['precision']:>9.1f}% | "
        f"{final_eval['recall']:>9.1f}% | "
        f"{final_eval['f1']:>9.1f}% | "
    )

    # ── 7. Summary ─────────────────────────────────────────────────────────
    print()
    _thick()
    print("  NER Model Training Complete")
    _thick()
    print(f"  Best dev F1          : {best_f1:.1f}%")
    print(f"  Best epoch           : {best_epoch}")
    print(f"  Saved to             : {output_dir}/")
    print(f"  Baseline (no NER)    : 0%")
    print(f"  Improvement          : +{best_f1:.1f}pp")
    _thick()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train_ner",
        description="Train spaCy NER model for science entities on EduHinglish data.",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=40,
        help="Maximum training epochs (default: 40).",
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
