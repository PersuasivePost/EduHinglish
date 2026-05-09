"""
EduHinglish — Chapter Dataset Builder  (Phase 2)
=================================================
Author  : Jatin
Purpose : Two modes
          1. Interactive guided annotation for a single chapter
             (reads from data/processed/<folder>/dataset.json,
              uses annotation_helper logic, saves back)
          2. --merge  : merge all chapter datasets into
             data/unified_biology_dataset.json

Usage
-----
  # Annotate a chapter interactively
  python scripts/chapter_dataset_builder.py --chapter class10/ch06

  # Merge all chapters into unified JSON
  python scripts/chapter_dataset_builder.py --merge

  # List available chapter codes
  python scripts/chapter_dataset_builder.py --list-chapters
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

# ── Import shared helpers from annotation_helper ──────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from annotation_helper import (
        HINDI_WORDS, UNIVERSAL_WORDS, VALID_LABELS,
        predict_label, tokenise_hinglish,
        load_existing, save_dataset, annotate_sentence,
    )
except ImportError:
    print("  [ERROR] Could not import annotation_helper.py. Run from project root.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER MAP  — maps CLI code → (folder_name, chapter_title, class_num)
# folder_name is the subdirectory under  data/processed/
# ─────────────────────────────────────────────────────────────────────────────

CHAPTER_MAP: dict[str, tuple[str, str, str]] = {
    # code              folder              title                                          class
    "class9/ch01":  ("class9_ch01",  "Chapter 1: Matter in Our Surroundings",            "9"),
    "class9/ch02":  ("class9_ch02",  "Chapter 2: Is Matter Around Us Pure",              "9"),
    "class9/ch03":  ("class9_ch03",  "Chapter 3: Atoms and Molecules",                   "9"),
    "class9/ch05":  ("class9_ch05",  "Chapter 5: The Fundamental Unit of Life",          "9"),
    "class9/ch06":  ("class9_ch06",  "Chapter 6: Tissues",                               "9"),
    "class9/ch07":  ("class9_ch07",  "Chapter 7: Diversity in Living Organisms",         "9"),
    "class9/ch11":  ("class9_ch11",  "Chapter 11: Work and Energy",                      "9"),
    "class9/ch12":  ("class9_ch12",  "Chapter 12: Sound",                                "9"),
    "class9/ch13":  ("class9_ch13",  "Chapter 13: Why Do We Fall Ill",                   "9"),
    "class9/ch14":  ("class9_ch14",  "Chapter 14: Natural Resources",                    "9"),
    "class9/ch15":  ("class9_ch15",  "Chapter 15: Improvement in Food Resources",        "9"),
    "class10/ch05": ("class10_ch05", "Chapter 5: Periodic Classification of Elements",   "10"),
    "class10/ch06": ("class10_ch06", "Chapter 6: Life Processes",                        "10"),
    "class10/ch07": ("class10_ch07", "Chapter 7: Control and Coordination",              "10"),
    "class10/ch08": ("class10_ch08", "Chapter 8: How do Organisms Reproduce?",           "10"),
    "class10/ch09": ("class10_ch09", "Chapter 9: Heredity and Evolution",                "10"),
    "class10/ch13": ("class10_ch13", "Chapter 13: Our Environment",                      "10"),
    "class10/ch15": ("class10_ch15", "Chapter 15: Our Environment",                      "10"),
    "class10/ch16": ("class10_ch16", "Chapter 16: Management of Natural Resources",      "10"),
}

# Root paths (relative to project root)
PROCESSED_DIR = Path("data/processed")
UNIFIED_PATH  = Path("data/unified_biology_dataset.json")

TARGET_PER_CHAPTER = 50   # default target


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _thin():
    print("  " + "─" * 50)

def _thick():
    print("  " + "═" * 50)

def _progress(current: int, target: int) -> str:
    filled = min(current * 20 // max(target, 1), 20)
    bar = "█" * filled + "░" * (20 - filled)
    pct = current * 100 // max(target, 1)
    return f"[{bar}] {current}/{target} ({pct}%)"

def _stats(dataset: list[dict]) -> str:
    statements = sum(1 for e in dataset
                     if not e.get("is_student_query") and not e.get("is_gec_sample"))
    queries    = sum(1 for e in dataset if e.get("is_student_query"))
    gecs       = sum(1 for e in dataset if e.get("is_gec_sample"))
    return f"statements={statements}  queries={queries}  gec={gecs}"


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE EXTRACTION  (from cleaned_text.txt if present)
# ─────────────────────────────────────────────────────────────────────────────

BIO_TERMS = [
    "cell", "tissue", "organ", "plant", "animal", "bacteria", "membrane",
    "nucleus", "muscle", "nerve", "disease", "health", "reproduction",
    "energy", "blood", "epithelial", "connective", "meristematic", "permanent",
    "chromosome", "DNA", "gene", "enzyme", "photosynthesis", "respiration",
    "digestion", "excretion", "hormone", "neuron", "reflex", "evolution",
    "heredity", "ecosystem", "food", "protein", "glucose", "oxygen",
]

def extract_candidates(cleaned_path: Path) -> list[str]:
    """Extract good candidate sentences from cleaned_text.txt."""
    if not cleaned_path.exists():
        return []
    text = cleaned_path.read_text(encoding="utf-8")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    candidates: list[str] = []
    for s in sentences:
        s = s.strip().replace("\n", " ")
        if len(s) > 40 and any(t in s.lower() for t in BIO_TERMS):
            candidates.append(s)
    # Fallback: any sentence > 40 chars
    if len(candidates) < 10:
        for s in sentences:
            s = s.strip().replace("\n", " ")
            if len(s) > 40 and s not in candidates:
                candidates.append(s)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE ANNOTATION MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_chapter(chapter_code: str, target: int = TARGET_PER_CHAPTER) -> None:
    """Guided annotation loop for one chapter."""

    if chapter_code not in CHAPTER_MAP:
        print(f"\n  [ERROR] Unknown chapter code '{chapter_code}'.")
        print("  Run with --list-chapters to see all valid codes.")
        sys.exit(1)

    folder, chapter_title, class_num = CHAPTER_MAP[chapter_code]
    chapter_dir  = PROCESSED_DIR / folder
    out_path     = chapter_dir / "dataset.json"
    cleaned_path = chapter_dir / "cleaned_text.txt"

    # Create folder if new chapter
    chapter_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_existing(out_path)

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    _thick()
    print(f"  EduHinglish — Chapter Dataset Builder")
    _thick()
    print(f"  Chapter : {chapter_title}")
    print(f"  Class   : {class_num}")
    print(f"  Output  : {out_path}")
    print(f"  Existing: {len(dataset)} entries  (target: {target})")
    print(f"  Stats   : {_stats(dataset)}")
    _thick()
    print()
    print("  Controls during annotation:")
    print("    Enter        → accept predicted label")
    print("    HI/EN/NE/UNIV/MIX → override label")
    print("    s            → skip this sentence")
    print("    q            → quit and save")
    print()

    # ── Load candidates ───────────────────────────────────────────────────────
    candidates = extract_candidates(cleaned_path)
    if candidates:
        print(f"  Found {len(candidates)} candidate sentences from cleaned_text.txt")
    else:
        print(f"  [INFO] No cleaned_text.txt found — you will enter sentences manually.")
        candidates = []

    total = len(dataset)

    # ── Main loop ─────────────────────────────────────────────────────────────
    cand_idx = 0
    while True:
        print()
        _thin()
        print(f"  Progress: {_progress(total, target)}")
        print(f"  Stats   : {_stats(dataset)}")
        _thin()

        if total >= target:
            print(f"\n  ✓ Target of {target} entries reached!")
            again = input("  Continue adding more? [y/n]: ").strip().lower()
            if again != "y":
                break

        # Pre-fill English from candidates or let user type
        if cand_idx < len(candidates):
            prefilled = candidates[cand_idx]
            cand_idx += 1
            wrapped = textwrap.fill(
                f'"{prefilled}"', width=76, subsequent_indent="    "
            )
            print(f"\n  NCERT: {wrapped}")
        else:
            prefilled = None

        entry = annotate_sentence(
            chapter_title=chapter_title,
            class_num=class_num,
            entry_number=total + 1,
            prefilled_english=prefilled,
        )

        if entry == "QUIT" or entry is None:
            if entry == "QUIT":
                print("\n  [QUIT] Exiting builder.")
            break

        if isinstance(entry, dict):
            # Ensure id uses the correct chapter prefix
            ch_prefix = chapter_code.replace("class", "").replace("/ch", "_").lstrip("0")
            # e.g. class10/ch06 → "10_06", class9/ch05 → "9_05"
            entry["id"] = f"{ch_prefix}_{str(total + 1).zfill(3)}"
            entry["class"] = class_num
            entry["chapter"] = chapter_title

            dataset.append(entry)
            save_dataset(out_path, dataset)
            total = len(dataset)
            print(f"\n  [SAVED] Entry #{total} → {out_path}")

    # ── Session summary ───────────────────────────────────────────────────────
    print()
    _thick()
    print(f"  Session complete.")
    print(f"  File now has {len(dataset)} entries : {out_path}")
    print(f"  Stats: {_stats(dataset)}")
    _thick()


# ─────────────────────────────────────────────────────────────────────────────
# MERGE / UNIFIED DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_unified() -> None:
    """
    Merge all chapter datasets from data/processed/ into
    data/unified_biology_dataset.json with a metadata header.
    """
    from collections import Counter

    print()
    _thick()
    print("  EduHinglish — Unified Dataset Builder")
    _thick()

    unified: list[dict] = []
    chapter_counts: dict[str, int] = {}
    missing: list[str] = []

    for code, (folder, title, cls) in sorted(CHAPTER_MAP.items()):
        path = PROCESSED_DIR / folder / "dataset.json"
        if not path.exists():
            missing.append(f"{code} ({folder})")
            print(f"  [SKIP] {code:20s} — dataset.json not found")
            continue
        entries = load_existing(path)
        if not entries:
            missing.append(f"{code} ({folder}) — empty")
            print(f"  [SKIP] {code:20s} — empty dataset")
            continue
        chapter_counts[code] = len(entries)
        unified.extend(entries)
        print(f"  [OK]   {code:20s} — {len(entries):3d} entries  ({title})")

    if not unified:
        print("\n  [ERROR] No data found. Run annotation first.")
        sys.exit(1)

    # ── Global stats ──────────────────────────────────────────────────────────
    label_counter: Counter = Counter()
    for entry in unified:
        label_counter.update(entry.get("word_level_labels", {}).values())
    total_tokens = sum(label_counter.values())

    print()
    _thin()
    print(f"  Total entries  : {len(unified)}")
    print(f"  Total tokens   : {total_tokens}")
    print(f"  Chapters merged: {len(chapter_counts)}")
    if missing:
        print(f"  Chapters skipped: {len(missing)}")
    print()
    print("  Label distribution:")
    for lbl in ["HI", "EN", "NE", "UNIV", "MIX"]:
        cnt = label_counter[lbl]
        pct = cnt * 100 // max(total_tokens, 1)
        bar = "█" * (pct // 3)
        print(f"    {lbl:6s} {cnt:6d} ({pct:3d}%)  {bar}")
    _thin()

    # ── Save ──────────────────────────────────────────────────────────────────
    UNIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(UNIFIED_PATH, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Saved {len(unified)} entries → {UNIFIED_PATH}")
    _thick()


# ─────────────────────────────────────────────────────────────────────────────
# LIST CHAPTERS
# ─────────────────────────────────────────────────────────────────────────────

def list_chapters() -> None:
    print("\n  Available chapter codes:\n")
    print(f"  {'Code':<20}  {'Class':<7}  {'Entries':>7}  Title")
    print("  " + "─" * 70)
    for code, (folder, title, cls) in sorted(CHAPTER_MAP.items()):
        path = PROCESSED_DIR / folder / "dataset.json"
        count = len(load_existing(path)) if path.exists() else 0
        marker = "✓" if count > 0 else " "
        print(f"  {code:<20}  Class {cls}   {count:>5} {marker}  {title}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chapter_dataset_builder",
        description="EduHinglish Phase 2 — build/merge chapter datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/chapter_dataset_builder.py --chapter class10/ch06\n"
            "  python scripts/chapter_dataset_builder.py --merge\n"
            "  python scripts/chapter_dataset_builder.py --list-chapters\n"
        ),
    )
    parser.add_argument(
        "--chapter", "-c",
        metavar="CODE",
        help="Chapter code to annotate (e.g. class10/ch06).",
    )
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=TARGET_PER_CHAPTER,
        metavar="N",
        help=f"Target number of entries per chapter (default: {TARGET_PER_CHAPTER}).",
    )
    parser.add_argument(
        "--merge", "-m",
        action="store_true",
        help=f"Merge all chapter datasets → {UNIFIED_PATH}",
    )
    parser.add_argument(
        "--list-chapters", "-l",
        action="store_true",
        help="List all known chapter codes with current entry counts.",
    )

    args = parser.parse_args()

    if args.list_chapters:
        list_chapters()
    elif args.merge:
        build_unified()
    elif args.chapter:
        run_chapter(args.chapter, target=args.target)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
