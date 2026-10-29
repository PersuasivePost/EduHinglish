"""
EduHinglish — Build Unified Training Dataset (v3)
==================================================
Merges two kinds of data into data/unified_dataset_v3.json, the single input
for training/prepare_spacy_data.py and training/prepare_hf_data.py:

  1. Legacy annotated sentences (data/processed/*/dataset.json)
     → gold word_level_labels (LID), plus student queries with intents.
  2. QA datasets (QA_SOURCES, e.g. data/bio_9_10.json, data/ss9.json)
     → every question_hinglish becomes a student query with its intent.
       These have no word-level labels; pass --silver-lid to auto-tag them
       with the existing spaCy LID model (models/lid_v2). Silver-labelled
       entries are only ever used for training, never for the dev split.

To add a new subject/class later, append its JSON to QA_SOURCES (or pass
--sources) and re-run this script followed by the prepare_* scripts.

Usage:
  python scripts/build_unified_dataset.py
  python scripts/build_unified_dataset.py --silver-lid
  python scripts/build_unified_dataset.py --sources data/bio_9_10.json data/ss9.json data/ss10.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from label_config import DATA_VERSION, INTENT_LABELS, normalize_intent  # noqa: E402

# ── Sources ──────────────────────────────────────────────────────────────────
# ss10.json is still being built — add it here once it is ready.
QA_SOURCES = [
    "data/bio_9_10.json",
    "data/ss_9_10.json",
]
LEGACY_GLOB    = "data/processed/*/dataset.json"
OUTPUT_PATH    = PROJECT_ROOT / "data" / f"unified_dataset_{DATA_VERSION}.json"
SILVER_LID_MODEL = PROJECT_ROOT / "models" / "lid_v2"


def load_legacy() -> list[dict]:
    """Annotated sentences with gold word_level_labels."""
    entries = []
    for path in sorted(PROJECT_ROOT.glob(LEGACY_GLOB)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for e in data:
            if not e.get("hinglish_roman"):
                continue
            entries.append({
                **e,
                "intent":           normalize_intent(e.get("intent")),
                "subject":          "Biology",
                "lid_label_source": "gold" if e.get("word_level_labels") else "none",
                "source_file":      str(path.relative_to(PROJECT_ROOT)),
            })
        print(f"  [legacy] {path.relative_to(PROJECT_ROOT)}: {len(data)}")
    return entries


def load_qa(sources: list[str]) -> list[dict]:
    """QA entries → one student-query entry per question."""
    entries = []
    for src in sources:
        path = PROJECT_ROOT / src
        if not path.exists():
            print(f"  [WARN] {src} not found — skipping")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        kept = english_only = 0
        for e in data:
            # Fall back to the English question when the Hinglish one is missing
            # (students ask in English too, so it is still a valid intent example)
            question = (e.get("question_hinglish") or "").strip()
            if not question:
                question = (e.get("question_english") or "").strip()
                english_only += 1
            if not question:
                continue
            entries.append({
                "id":                e["id"],
                "class":             e.get("class"),
                "subject":           e.get("subject"),
                "sub_subject":       e.get("sub_subject"),
                "chapter":           e.get("chapter"),
                "chapter_num":       e.get("chapter_num"),
                "topic":             e.get("topic"),
                "original_english":  e.get("question_english", ""),
                "hinglish_roman":    question,
                "word_level_labels": {},
                "lid_label_source":  "none",
                "is_student_query":  True,
                "intent":            normalize_intent(e.get("intent")),
                "source_file":       src,
            })
            kept += 1
        print(f"  [qa]     {src}: {kept}/{len(data)}  ({english_only} without question_hinglish)")
    return entries


def add_silver_lid(entries: list[dict]) -> int:
    """Tag entries without word-level labels using the spaCy LID model."""
    import spacy

    if not SILVER_LID_MODEL.exists():
        print(f"  [WARN] {SILVER_LID_MODEL} not found — skipping silver LID")
        return 0
    nlp = spacy.load(SILVER_LID_MODEL)
    todo = [e for e in entries if e["lid_label_source"] == "none"]
    for e, doc in zip(todo, nlp.pipe(e["hinglish_roman"] for e in todo)):
        e["word_level_labels"] = {tok.text: tok.tag_ for tok in doc if tok.tag_}
        e["lid_label_source"] = "silver"
    return len(todo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified training dataset.")
    parser.add_argument("--sources", nargs="+", default=QA_SOURCES,
                        help="QA dataset JSON files (relative to project root).")
    parser.add_argument("--silver-lid", action="store_true",
                        help="Auto-tag QA questions for LID with models/lid_v2.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print("\n  Loading sources")
    entries = load_legacy() + load_qa(args.sources)

    # Dedupe by id (first occurrence wins)
    seen, unique = set(), []
    for e in entries:
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        unique.append(e)
    entries = unique

    if args.silver_lid:
        n = add_silver_lid(entries)
        print(f"\n  Silver LID labels added to {n} entries")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    lid_src = Counter(e["lid_label_source"] for e in entries)
    intents = Counter(e["intent"] for e in entries if e.get("is_student_query") and e["intent"])
    print(f"\n  Total entries      : {len(entries)}")
    print(f"  LID labels         : {dict(lid_src)}")
    print(f"  Intent queries     : {sum(intents.values())}")
    for lbl in INTENT_LABELS:
        print(f"    {lbl:22s} {intents[lbl]}")
    print(f"\n  [OK] Saved: {args.output.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
