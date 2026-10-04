"""
EduHinglish — Phase 6, Prompt 1: Prepare Seq2Seq Data
=======================================================
Author  : Ashvatth
Module  : Phase 6 — IndicBART Hinglish Generator Fine-tuning (Data Preparation)
Purpose : Convert data/unified_biology_dataset_v2.json into (source, target) pairs
          for IndicBART seq2seq fine-tuning (English → Hinglish).

Three pair types are built:

  TYPE A — Direct Translation Pairs (all filtered entries):
      source : "Translate to Hinglish: {original_english}"
      target : "{hinglish_roman}"

  TYPE B — Query-Grounded Pairs (is_student_query=True entries only):
      source : "Explain in Hinglish: {original_english} | Student asks: {hinglish_roman}"
      target : "{hinglish_roman}"

  TYPE C — Topic-Conditioned Pairs (all filtered entries):
      source : "Topic: {topic} | Explain in Hinglish: {original_english}"
      target : "{hinglish_roman}"

Output:
  training/data/indicbart_seq2seq_dataset/   — HuggingFace DatasetDict
  training/data/indicbart_seq2seq_dataset/config.json

Usage:
  python training/prepare_seq2seq_data.py                    # all types (A+B+C)
  python training/prepare_seq2seq_data.py --types A          # type A only
  python training/prepare_seq2seq_data.py --types A C        # types A and C
  python training/prepare_seq2seq_data.py --dataset PATH     # custom dataset path
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure stdout can handle UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

try:
    from transformers import AutoTokenizer
except ImportError:
    print("  [ERROR] transformers not installed.")
    print("    Run: pip install transformers>=4.40.0")
    sys.exit(1)

try:
    from datasets import Dataset, DatasetDict
except ImportError:
    print("  [ERROR] datasets not installed.")
    print("    Run: pip install datasets>=2.19.0")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATASET = Path("data/unified_biology_dataset_v2.json")
OUTPUT_DIR      = Path("training/data/indicbart_seq2seq_dataset")

MODEL_NAME         = "ai4bharat/IndicBART"
MAX_SOURCE_LENGTH  = 256
MAX_TARGET_LENGTH  = 256
SEED               = 42
MIN_WORDS          = 5          # filter threshold — both fields must have ≥ 5 words

VALID_TYPES = ("A", "B", "C")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _thick() -> None:
    print("  " + "=" * 58)


def _divider() -> None:
    print("  " + "-" * 58)


def load_dataset_json(path: Path) -> list[dict]:
    """Load the unified JSON dataset from disk."""
    if not path.exists():
        print(f"  [ERROR] Dataset not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entries from {path}")
    return data


def _has_enough_words(text: str, min_words: int = MIN_WORDS) -> bool:
    """Return True if *text* contains at least *min_words* whitespace tokens."""
    return isinstance(text, str) and len(text.split()) >= min_words


def filter_entries(dataset: list[dict]) -> list[dict]:
    """
    Keep only entries where BOTH original_english AND hinglish_roman are
    non-empty strings with at least MIN_WORDS words each.
    """
    kept    = []
    dropped = 0
    for entry in dataset:
        eng  = entry.get("original_english", "")
        hin  = entry.get("hinglish_roman", "")
        if _has_enough_words(eng) and _has_enough_words(hin):
            kept.append(entry)
        else:
            dropped += 1
    print(f"  Filter ({MIN_WORDS}+ words): kept {len(kept)}, dropped {dropped}")
    return kept


def split_train_dev(items: list, dev_ratio: float = 0.2):
    """Shuffle and split into (train, dev). Same seed as Phase 3/4."""
    random.seed(SEED)
    shuffled = list(items)
    random.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * (1 - dev_ratio)))
    return shuffled[:cut], shuffled[cut:]


# ─────────────────────────────────────────────────────────────────────────────
# PAIR BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_type_a(entries: list[dict]) -> list[dict]:
    """
    TYPE A — Direct Translation Pairs (all filtered entries).
      source : "Translate to Hinglish: {original_english}"
      target : "{hinglish_roman}"
    """
    pairs = []
    for entry in entries:
        pairs.append({
            "source":   f"Translate to Hinglish: {entry['original_english']}",
            "target":   entry["hinglish_roman"],
            "pair_type": "A",
        })
    return pairs


def build_type_b(entries: list[dict]) -> list[dict]:
    """
    TYPE B — Query-Grounded Pairs (is_student_query=True entries only).
      source : "Explain in Hinglish: {original_english} | Student asks: {hinglish_roman}"
      target : "{hinglish_roman}"

    Teaches the model to respond to student-style conversational queries.
    """
    pairs = []
    for entry in entries:
        if not entry.get("is_student_query"):
            continue
        pairs.append({
            "source": (
                f"Explain in Hinglish: {entry['original_english']} "
                f"| Student asks: {entry['hinglish_roman']}"
            ),
            "target":    entry["hinglish_roman"],
            "pair_type": "B",
        })
    return pairs


def build_type_c(entries: list[dict]) -> list[dict]:
    """
    TYPE C — Topic-Conditioned Pairs (all filtered entries).
      source : "Topic: {topic} | Explain in Hinglish: {original_english}"
      target : "{hinglish_roman}"

    Teaches the model topic awareness during generation.
    """
    pairs = []
    for entry in entries:
        topic = entry.get("topic", "").strip() or "General"
        pairs.append({
            "source": (
                f"Topic: {topic} "
                f"| Explain in Hinglish: {entry['original_english']}"
            ),
            "target":    entry["hinglish_roman"],
            "pair_type": "C",
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# TOKENISATION
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_pairs(pairs: list[dict], tokenizer) -> list[dict]:
    """
    Tokenize every (source, target) pair using the IndicBART tokenizer.

    Stores:
      input_ids      — tokenised source
      attention_mask — source attention mask
      labels         — tokenised target (target input_ids)
    """
    tokenized = []
    for pair in pairs:
        src = tokenizer(
            pair["source"],
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
            padding=False,
        )
        tgt = tokenizer(
            pair["target"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )
        tokenized.append({
            "input_ids":      src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels":         tgt["input_ids"],   # labels = target token ids
            # retain for stats / examples (not saved to dataset)
            "_source_text":   pair["source"],
            "_target_text":   pair["target"],
            "_pair_type":     pair["pair_type"],
        })
    return tokenized


# ─────────────────────────────────────────────────────────────────────────────
# STATS + SAMPLE PRINT
# ─────────────────────────────────────────────────────────────────────────────

def _print_stats(
    all_examples:   list[dict],
    train_examples: list[dict],
    dev_examples:   list[dict],
    type_counts:    dict[str, int],
) -> None:
    """Print detailed statistics about the prepared dataset."""
    total = len(all_examples)
    if total == 0:
        print("  [WARN] No examples — nothing to report.")
        return

    avg_src = sum(len(e["input_ids"]) for e in all_examples) / total
    avg_tgt = sum(len(e["labels"])    for e in all_examples) / total

    print(f"\n  IndicBART Seq2Seq Dataset Stats:")
    _divider()
    print(f"    Total pairs    : {total}")
    for t in VALID_TYPES:
        cnt = type_counts.get(t, 0)
        bar = "#" * min(cnt // 5, 40)
        print(f"      Type {t}      : {cnt:>5d}  {bar}")
    _divider()
    print(f"    Train          : {len(train_examples)}")
    print(f"    Dev            : {len(dev_examples)}")
    _divider()
    print(f"    Avg source tokens : {avg_src:.1f}")
    print(f"    Avg target tokens : {avg_tgt:.1f}")

    # ── Print 3 sample pairs ─────────────────────────────────────────────────
    print(f"\n  Sample pairs (3 examples):")
    samples = all_examples[:3]
    for i, ex in enumerate(samples, 1):
        _divider()
        print(f"  Example {i}  [Type {ex['_pair_type']}]")
        print(f"    SOURCE : {ex['_source_text'][:120]}")
        print(f"    TARGET : {ex['_target_text'][:120]}")
        src_tok = len(ex["input_ids"])
        tgt_tok = len(ex["labels"])
        print(f"    Tokens : src={src_tok}  tgt={tgt_tok}")
    _divider()


# ─────────────────────────────────────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def prepare_seq2seq(
    dataset:   list[dict],
    tokenizer,
    types:     list[str],
) -> None:
    """
    Full pipeline:
      filter → build pairs → tokenize → split → save DatasetDict + config.json
    """
    print()
    _thick()
    print("  Preparing Seq2Seq Data for IndicBART Fine-tuning")
    _thick()
    print(f"  Selected pair types: {', '.join(sorted(types))}")

    # ── 1. Filter ─────────────────────────────────────────────────────────────
    entries = filter_entries(dataset)

    # ── 2. Build pairs ────────────────────────────────────────────────────────
    all_pairs: list[dict] = []
    type_counts: dict[str, int] = {}

    if "A" in types:
        a_pairs = build_type_a(entries)
        type_counts["A"] = len(a_pairs)
        all_pairs.extend(a_pairs)
        print(f"  Type A pairs built : {len(a_pairs)}")

    if "B" in types:
        b_pairs = build_type_b(entries)
        type_counts["B"] = len(b_pairs)
        all_pairs.extend(b_pairs)
        print(f"  Type B pairs built : {len(b_pairs)}  (student query entries)")

    if "C" in types:
        c_pairs = build_type_c(entries)
        type_counts["C"] = len(c_pairs)
        all_pairs.extend(c_pairs)
        print(f"  Type C pairs built : {len(c_pairs)}")

    total_pairs = len(all_pairs)
    print(f"\n  Total pairs        : {total_pairs}")

    if total_pairs == 0:
        print("  [ERROR] No pairs were built — check dataset and type selection.")
        sys.exit(1)

    # ── 3. Tokenize ───────────────────────────────────────────────────────────
    print(f"\n  Tokenizing {total_pairs} pairs …")
    all_examples = tokenize_pairs(all_pairs, tokenizer)
    print(f"  Tokenization complete.")

    # ── 4. Split 80/20 ───────────────────────────────────────────────────────
    train_examples, dev_examples = split_train_dev(all_examples)

    # ── 5. Save as HuggingFace DatasetDict ───────────────────────────────────
    HF_COLS = ("input_ids", "attention_mask", "labels")

    def _to_dataset(examples: list[dict]) -> Dataset:
        return Dataset.from_dict({
            col: [e[col] for e in examples]
            for col in HF_COLS
        })

    ds = DatasetDict({
        "train": _to_dataset(train_examples),
        "dev":   _to_dataset(dev_examples),
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_DIR))

    # ── 6. Save config.json ───────────────────────────────────────────────────
    config = {
        "model_name":        MODEL_NAME,
        "max_source_length": MAX_SOURCE_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "total_pairs":       total_pairs,
        "type_a_count":      type_counts.get("A", 0),
        "type_b_count":      type_counts.get("B", 0),
        "type_c_count":      type_counts.get("C", 0),
        "train_count":       len(train_examples),
        "dev_count":         len(dev_examples),
    }
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ── 7. Print stats ────────────────────────────────────────────────────────
    _print_stats(all_examples, train_examples, dev_examples, type_counts)

    print()
    print(f"  [OK] Saved dataset  : {OUTPUT_DIR}/")
    print(f"  [OK] Saved config   : {config_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prepare_seq2seq_data",
        description=(
            "Convert unified dataset → IndicBART seq2seq format "
            "(English → Hinglish pair types A / B / C)."
        ),
    )
    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to unified JSON dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--types", "-t",
        nargs="+",
        choices=list(VALID_TYPES),
        default=list(VALID_TYPES),
        metavar="TYPE",
        help=(
            "Which pair types to build: A (direct), B (query-grounded), "
            "C (topic-conditioned). Default: A B C. "
            "Example: --types A C"
        ),
    )
    args = parser.parse_args()

    # Deduplicate and uppercase the types list
    selected_types = sorted(set(t.upper() for t in args.types))
    invalid = [t for t in selected_types if t not in VALID_TYPES]
    if invalid:
        print(f"  [ERROR] Unknown pair type(s): {invalid}. Choose from A, B, C.")
        sys.exit(1)

    print()
    _thick()
    print("  EduHinglish — Phase 6: Prepare IndicBART Seq2Seq Data")
    _thick()

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = load_dataset_json(args.dataset)

    # ── Load IndicBART tokenizer ──────────────────────────────────────────────
    print(f"\n  Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  [OK] Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # ── Run preparation ───────────────────────────────────────────────────────
    prepare_seq2seq(dataset, tokenizer, selected_types)

    # ── Done ──────────────────────────────────────────────────────────────────
    print()
    _thick()
    print("  All done! Dataset saved to training/data/indicbart_seq2seq_dataset/")
    _thick()
    print()


if __name__ == "__main__":
    main()
