"""
EduHinglish — Phase 4, Task 1: Prepare HuggingFace Data
=========================================================
Authors : Ashvatth & Jatin
Module  : Phase 4 — MuRIL Fine-tuning (Data Preparation)
Purpose : Convert data/unified_biology_dataset.json into HuggingFace
          Dataset format for MuRIL fine-tuning on two tasks:

  1. Token Classification (LID) — word-level language identification
  2. Sequence Classification (Intent) — student query intent

CRITICAL: MuRIL uses WordPiece subword tokenization. Each original
word may split into multiple subword tokens. The label for the FIRST
subword comes from word_level_labels; continuation subwords get -100
(ignored in loss computation). This is the standard HuggingFace
token classification alignment strategy.

Usage:
  python training/prepare_hf_data.py                   # all tasks
  python training/prepare_hf_data.py --task lid         # LID only
  python training/prepare_hf_data.py --task intent      # Intent only
  python training/prepare_hf_data.py --dataset PATH     # custom dataset
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

# Ensure stdout can handle UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

try:
    from transformers import AutoTokenizer
except ImportError:
    print("[ERROR] transformers not installed.")
    print("  Run: pip install transformers")
    sys.exit(1)

try:
    from datasets import Dataset, DatasetDict
except ImportError:
    print("[ERROR] datasets not installed.")
    print("  Run: pip install datasets")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATASET   = Path("data/unified_biology_dataset.json")
LID_OUTPUT_DIR    = Path("training/data/muril_lid_dataset")
INTENT_OUTPUT_DIR = Path("training/data/muril_intent_dataset")

MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 128
SEED       = 42

# ── LID label mappings ───────────────────────────────────────────────────────
LID_LABEL2ID = {"HI": 0, "EN": 1, "NE": 2, "UNIV": 3, "MIX": 4}
LID_ID2LABEL = {v: k for k, v in LID_LABEL2ID.items()}
LID_LABELS   = list(LID_LABEL2ID.keys())

# ── Intent label mappings ────────────────────────────────────────────────────
INTENT_LABEL2ID = {
    "explain_concept":  0,
    "compare_concepts": 1,
    "give_example":     2,
    "formula_request":  3,
    "definition":       4,
}
INTENT_ID2LABEL = {v: k for k, v in INTENT_LABEL2ID.items()}
INTENT_LABELS   = list(INTENT_LABEL2ID.keys())


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _thick():
    print("  " + "=" * 58)

def _divider():
    print("  " + "-" * 58)


def load_dataset_json(path: Path) -> list[dict]:
    """Load the unified JSON dataset."""
    if not path.exists():
        print(f"  [ERROR] Dataset not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entries from {path}")
    return data


_PUNCT_CHARS = frozenset(list(".,?!;:()-/") + ['"', "'", "\u2013", "\u2014", "\u2026"])
_PUNCT_STRIP = "".join(_PUNCT_CHARS)


def strip_punct(word: str) -> str:
    """Strip sentence-boundary punctuation from a token
    (mirrors prepare_spacy_data.strip_punct)."""
    return word.strip(_PUNCT_STRIP)


def is_punct_token(text: str) -> bool:
    """Check if a token is purely punctuation."""
    return len(text) > 0 and all(ch in _PUNCT_CHARS for ch in text)


def split_train_dev(items: list, dev_ratio: float = 0.2):
    """Shuffle and split into (train, dev). Same seed as Phase 3."""
    random.seed(SEED)
    shuffled = list(items)
    random.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * (1 - dev_ratio)))
    return shuffled[:cut], shuffled[cut:]


def lookup_label(word: str, labels_dict: dict) -> str:
    """
    Look up the LID label for a word from the word_level_labels dict.

    Matching strategy (same priority as prepare_spacy_data.py):
      1. Exact match
      2. Lowercase / capitalize variants
      3. Strip punctuation, then retry 1–2
      4. Fuzzy: any key matches after stripping + lowering
      5. Pure punctuation → UNIV
      6. Default fallback → EN

    Returns: label string (HI / EN / NE / UNIV / MIX)
    """
    cleaned = strip_punct(word)

    # 1–2: Exact and case variants on raw word
    lbl = (
        labels_dict.get(word)
        or labels_dict.get(word.lower())
        or labels_dict.get(word.capitalize())
    )
    if lbl:
        return lbl

    # 3: Strip punctuation, then retry
    if cleaned != word:
        lbl = (
            labels_dict.get(cleaned)
            or labels_dict.get(cleaned.lower())
            or labels_dict.get(cleaned.capitalize())
        )
        if lbl:
            return lbl

    # 4: Fuzzy match (strip + lowercase on both sides)
    if cleaned:
        cleaned_lower = cleaned.lower()
        for k, v in labels_dict.items():
            if strip_punct(k).lower() == cleaned_lower:
                return v

    # 5: Pure punctuation → UNIV
    if is_punct_token(word) or not cleaned:
        return "UNIV"

    # 6: Default fallback (NCERT text is English-heavy)
    return "EN"


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1a — LID DATA  (Token Classification)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_lid(dataset: list[dict], tokenizer) -> None:
    """
    Convert word_level_labels into MuRIL-aligned token classification data.

    Strategy:
      1. Split hinglish_roman by whitespace → original words
      2. Look up each word's LID label from word_level_labels
      3. Tokenize with is_split_into_words=True (WordPiece on each word)
      4. Align labels using word_ids():
         - word_id is None   → -100 (special token: [CLS], [SEP])
         - first subword     → label ID for that word
         - continuation      → -100 (ignored in loss)
      5. 80/20 train/dev split
      6. Save as HuggingFace DatasetDict
    """
    print()
    _thick()
    print("  Preparing LID (Token Classification) data for MuRIL")
    _thick()

    all_examples: list[dict] = []
    skipped      = 0
    total_subwords = 0
    label_counter: Counter = Counter()

    for entry in dataset:
        hinglish    = entry.get("hinglish_roman", "")
        labels_dict = entry.get("word_level_labels", {})
        if not hinglish or not labels_dict:
            skipped += 1
            continue

        # Step 1: Split into words
        words = hinglish.split()

        # Step 2: Look up label for each word
        word_labels: list[str] = []
        for w in words:
            lbl = lookup_label(w, labels_dict)
            if lbl not in LID_LABEL2ID:
                lbl = "EN"  # safety fallback
            word_labels.append(lbl)

        # Step 3: Tokenize with word-level alignment
        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,       # pad during training, not here
        )

        # Step 4: Align labels to subword tokens
        word_ids = tokenized.word_ids()   # None for special tokens
        aligned_labels: list[int] = []
        previous_word_id = None

        for wid in word_ids:
            if wid is None:
                # Special token ([CLS], [SEP])
                aligned_labels.append(-100)
            elif wid != previous_word_id:
                # First subword of a new word → real label
                lbl_str = word_labels[wid]
                aligned_labels.append(LID_LABEL2ID[lbl_str])
                label_counter[lbl_str] += 1
            else:
                # Continuation subword → ignore in loss
                aligned_labels.append(-100)
            previous_word_id = wid

        total_subwords += len(tokenized["input_ids"])

        all_examples.append({
            "input_ids":      tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels":         aligned_labels,
            "word_ids":       [w if w is not None else -1 for w in word_ids],
        })

    # ── Split ────────────────────────────────────────────────────────────────
    train_examples, dev_examples = split_train_dev(all_examples)

    # ── Save as HuggingFace DatasetDict ──────────────────────────────────────
    def _to_dataset(examples: list[dict]) -> Dataset:
        return Dataset.from_dict({
            "input_ids":      [e["input_ids"]      for e in examples],
            "attention_mask": [e["attention_mask"]  for e in examples],
            "labels":         [e["labels"]          for e in examples],
            "word_ids":       [e["word_ids"]        for e in examples],
        })

    ds = DatasetDict({
        "train": _to_dataset(train_examples),
        "dev":   _to_dataset(dev_examples),
    })

    LID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(LID_OUTPUT_DIR))

    # ── Save label maps as JSON (used by training scripts) ───────────────────
    maps = {
        "label2id":   LID_LABEL2ID,
        "id2label":   {str(k): v for k, v in LID_ID2LABEL.items()},
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
    }
    maps_path = LID_OUTPUT_DIR / "label_maps.json"
    with open(maps_path, "w", encoding="utf-8") as f:
        json.dump(maps, f, indent=2)

    # ── Print stats ──────────────────────────────────────────────────────────
    total_labels = sum(label_counter.values())
    avg_subwords = total_subwords / len(all_examples) if all_examples else 0

    print(f"\n  MuRIL LID Dataset:")
    print(f"    Total examples : {len(all_examples)}")
    print(f"    Skipped        : {skipped}")
    print(f"    Train          : {len(train_examples)}")
    print(f"    Dev            : {len(dev_examples)}")
    print(f"    Avg subword tokens per sentence: {avg_subwords:.1f}")
    print()
    print(f"    Label distribution (first-subword only):")
    for lbl in LID_LABELS:
        cnt = label_counter[lbl]
        pct = cnt * 100 / max(total_labels, 1)
        bar = "#" * int(pct / 2)
        print(f"      {lbl:6s} {cnt:6d} ({pct:5.1f}%) {bar}")
    print()
    print(f"  [OK] Saved dataset : {LID_OUTPUT_DIR}/")
    print(f"  [OK] Saved label maps: {maps_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1b — INTENT DATA  (Sequence Classification)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_intent(dataset: list[dict], tokenizer) -> None:
    """
    Filter student queries with valid intent labels and create
    sequence classification data for MuRIL fine-tuning.

    Also computes class weights (inverse-frequency) to handle
    the skewed distribution (53 explain vs 5 formula).
    """
    print()
    _thick()
    print("  Preparing Intent (Sequence Classification) data for MuRIL")
    _thick()

    # ── Filter entries with valid intent ─────────────────────────────────────
    queries = [
        e for e in dataset
        if e.get("is_student_query") and e.get("intent") in INTENT_LABELS
    ]

    if len(queries) < 60:
        print(f"\n  WARNING: Only {len(queries)} intent entries found.")
        print("  Consider adding more student queries to improve accuracy.")

    all_examples: list[dict] = []
    intent_counter: Counter  = Counter()

    for entry in queries:
        hinglish = entry.get("hinglish_roman", "")
        intent   = entry["intent"]
        if not hinglish:
            continue

        tokenized = tokenizer(
            hinglish,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )

        all_examples.append({
            "input_ids":      tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels":         INTENT_LABEL2ID[intent],
        })
        intent_counter[intent] += 1

    # ── Split ────────────────────────────────────────────────────────────────
    train_examples, dev_examples = split_train_dev(all_examples)

    # ── Save as HuggingFace DatasetDict ──────────────────────────────────────
    def _to_dataset(examples: list[dict]) -> Dataset:
        return Dataset.from_dict({
            "input_ids":      [e["input_ids"]      for e in examples],
            "attention_mask": [e["attention_mask"]  for e in examples],
            "labels":         [e["labels"]          for e in examples],
        })

    ds = DatasetDict({
        "train": _to_dataset(train_examples),
        "dev":   _to_dataset(dev_examples),
    })

    INTENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(INTENT_OUTPUT_DIR))

    # ── Compute class weights (inverse frequency) ────────────────────────────
    total    = sum(intent_counter.values())
    n_classes = len(INTENT_LABELS)
    class_weights: dict[str, float] = {}
    for intent in INTENT_LABELS:
        count = intent_counter.get(intent, 1)  # avoid division by zero
        class_weights[intent] = round(total / (n_classes * count), 4)

    # ── Save label maps + class weights ──────────────────────────────────────
    maps = {
        "intent2id":     INTENT_LABEL2ID,
        "id2intent":     {str(k): v for k, v in INTENT_ID2LABEL.items()},
        "class_weights": class_weights,
        "model_name":    MODEL_NAME,
        "max_length":    MAX_LENGTH,
    }
    maps_path = INTENT_OUTPUT_DIR / "label_maps.json"
    with open(maps_path, "w", encoding="utf-8") as f:
        json.dump(maps, f, indent=2)

    # ── Print stats ──────────────────────────────────────────────────────────
    print(f"\n  MuRIL Intent Dataset:")
    print(f"    Total queries  : {len(all_examples)}")
    print(f"    Train          : {len(train_examples)}")
    print(f"    Dev            : {len(dev_examples)}")
    print()
    print(f"    Intent distribution:")
    for intent in INTENT_LABELS:
        cnt    = intent_counter[intent]
        weight = class_weights[intent]
        bar    = "#" * cnt
        print(f"      {intent:20s} {cnt:4d}  (weight: {weight:.2f})  {bar}")
    print()
    print(f"  [OK] Saved dataset    : {INTENT_OUTPUT_DIR}/")
    print(f"  [OK] Saved label maps : {maps_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prepare_hf_data",
        description="Convert unified dataset → HuggingFace format for MuRIL fine-tuning.",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to unified JSON dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--task", "-t",
        choices=["lid", "intent", "all"],
        default="all",
        help="Which task to prepare data for (default: all).",
    )
    args = parser.parse_args()

    print()
    _thick()
    print("  EduHinglish — Phase 4: Prepare HuggingFace Data")
    _thick()

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset = load_dataset_json(args.dataset)

    # ── Load MuRIL tokenizer ─────────────────────────────────────────────────
    print(f"\n  Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  ✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # ── Prepare requested task(s) ────────────────────────────────────────────
    if args.task in ("lid", "all"):
        prepare_lid(dataset, tokenizer)

    if args.task in ("intent", "all"):
        prepare_intent(dataset, tokenizer)

    # ── Done ─────────────────────────────────────────────────────────────────
    print()
    _thick()
    print("  All done! Datasets saved to training/data/")
    _thick()
    print()


if __name__ == "__main__":
    main()
