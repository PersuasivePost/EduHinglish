"""
EduHinglish — Dataset Augmentation Helper
========================================
Author  : Ashvatth
Module  : M1 — Dataset Expansion (Ch5)
Purpose : Generate candidate Hinglish variants from labeled sentences.

Usage:
    python scripts/augment_dataset.py
    python scripts/augment_dataset.py --merge
"""

import sys
import json
import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "biology" / "class9" / "ch05"
CANDIDATES_PATH = DATA_DIR / "augmented_candidates.json"
OUTPUT_PATH = DATA_DIR / "dataset_v2.json"

# Allow importing the seed dataset if JSON is not present
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Basic word maps for augmentation
HINDI_EQUIV = {
    "cell": "koshika",
    "cells": "koshikaye",
    "membrane": "jhilli",
    "nucleus": "kendr",
    "powerhouse": "urja-kendra",
    "energy": "urja",
}

CONNECTOR_SWAP = {
    "aur": "tatha",
    "tatha": "aur",
    "lekin": "parantu",
    "parantu": "lekin",
    "kyunki": "isliye",
    "isliye": "kyunki",
}

VERB_TOKENS = {"hota", "hoti", "hote", "hai", "hain"}


def load_seed_dataset() -> list:
    """Load dataset from JSON if present, else import from src."""
    json_path = DATA_DIR / "dataset_v2.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    try:
        from hinglish_dataset_creator import DATASET
        return DATASET
    except Exception:
        return []


def split_tokens(text: str) -> list:
    return text.split()


def strip_punct(token: str) -> tuple:
    prefix = re.match(r"^\W+", token)
    suffix = re.search(r"\W+$", token)
    pre = prefix.group(0) if prefix else ""
    suf = suffix.group(0) if suffix else ""
    core = token[len(pre):len(token) - len(suf) if suf else len(token)]
    return pre, core, suf


def rebuild_labels(tokens: list, base_labels: dict) -> dict:
    """Rebuild labels using base labels with a light heuristic fallback."""
    labels = {}
    for i, tok in enumerate(tokens):
        _, core, _ = strip_punct(tok)
        if not core:
            continue
        if core in base_labels:
            labels[core] = base_labels[core]
            continue

        lower = core.lower()
        if lower == "sir":
            labels[core] = "UNIV"
        elif lower in {"nahi", "mat"}:
            labels[core] = "HI"
        elif lower in CONNECTOR_SWAP or lower in {"ka", "ki", "ke", "ko", "se", "mein"}:
            labels[core] = "HI"
        elif lower in HINDI_EQUIV.values():
            labels[core] = "HI"
        elif lower.isdigit():
            labels[core] = "UNIV"
        elif i > 0 and core[0].isupper():
            labels[core] = "NE"
        else:
            labels[core] = "EN"
    return labels


def apply_hindi_density_shift(entry: dict) -> dict | None:
    tokens = split_tokens(entry["hinglish_roman"])
    changed = False
    new_tokens = []
    for tok in tokens:
        pre, core, suf = strip_punct(tok)
        repl = HINDI_EQUIV.get(core.lower())
        if repl:
            new_tokens.append(pre + repl + suf)
            changed = True
        else:
            new_tokens.append(tok)
    if not changed:
        return None

    new_text = " ".join(new_tokens)
    labels = rebuild_labels(new_tokens, entry.get("word_level_labels", {}))
    return build_variant(entry, new_text, labels, "hindi_density")


def apply_connector_swap(entry: dict) -> dict | None:
    tokens = split_tokens(entry["hinglish_roman"])
    changed = False
    new_tokens = []
    for tok in tokens:
        pre, core, suf = strip_punct(tok)
        repl = CONNECTOR_SWAP.get(core.lower())
        if repl:
            new_tokens.append(pre + repl + suf)
            changed = True
        else:
            new_tokens.append(tok)
    if not changed:
        return None

    new_text = " ".join(new_tokens)
    labels = rebuild_labels(new_tokens, entry.get("word_level_labels", {}))
    return build_variant(entry, new_text, labels, "connector_swap")


def apply_formality_shift(entry: dict) -> dict | None:
    base = entry["hinglish_roman"].strip()
    if base.lower().startswith("sir"):
        return None

    new_text = f"Sir, {base} batao."
    tokens = split_tokens(new_text)
    labels = rebuild_labels(tokens, entry.get("word_level_labels", {}))
    return build_variant(entry, new_text, labels, "formality_shift")


def apply_question_form(entry: dict) -> dict | None:
    base = entry["hinglish_roman"].strip().rstrip(".?")
    if base.lower().startswith("sir"):
        return None

    new_text = f"Sir, {base} samjhao?"
    tokens = split_tokens(new_text)
    labels = rebuild_labels(tokens, entry.get("word_level_labels", {}))
    variant = build_variant(entry, new_text, labels, "question_form")
    variant["is_student_query"] = True
    variant["intent"] = "explain_concept"
    return variant


def apply_negation(entry: dict) -> dict | None:
    tokens = split_tokens(entry["hinglish_roman"])
    if "nahi" in [strip_punct(t)[1].lower() for t in tokens]:
        return None

    new_tokens = []
    changed = False
    for tok in tokens:
        pre, core, suf = strip_punct(tok)
        lower = core.lower()
        if not changed and lower in VERB_TOKENS:
            new_tokens.append("nahi")
            new_tokens.append(tok)
            changed = True
        else:
            new_tokens.append(tok)

    if not changed:
        return None

    new_text = " ".join(new_tokens)
    labels = rebuild_labels(new_tokens, entry.get("word_level_labels", {}))
    return build_variant(entry, new_text, labels, "negation")


def build_variant(entry: dict, text: str, labels: dict, aug_type: str) -> dict:
    variant = dict(entry)
    variant["hinglish_roman"] = text
    variant["word_level_labels"] = labels
    variant["augmented_from"] = entry.get("id")
    variant["augmentation_type"] = aug_type
    return variant


def generate_candidates(dataset: list) -> list:
    candidates = []
    for entry in dataset:
        # Skip existing student queries for question-form variants
        variants = []
        for fn in (
            apply_hindi_density_shift,
            apply_connector_swap,
            apply_formality_shift,
            apply_question_form,
            apply_negation,
        ):
            v = fn(entry)
            if v:
                variants.append(v)

        # Keep 2-3 variants max
        variants = variants[:3]
        # Apply unique id suffixes
        suffixes = ["a", "b", "c"]
        for idx, v in enumerate(variants):
            base_id = str(entry.get("id"))
            v["id"] = f"{base_id}{suffixes[idx]}"
            candidates.append(v)

    return candidates


def save_candidates(candidates: list):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)


def load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def merge_candidates():
    candidates = load_json_list(CANDIDATES_PATH)
    if not candidates:
        print("[ERROR] No candidates found. Run without --merge first.")
        return

    approved = []
    for i, cand in enumerate(candidates, 1):
        print("-" * 70)
        print(f"[{i}/{len(candidates)}] {cand.get('hinglish_roman')}")
        print(f"  From: {cand.get('augmented_from')}  Type: {cand.get('augmentation_type')}")
        choice = input("Keep? (y/n/e to edit): ").strip().lower()
        if choice == "y":
            approved.append(cand)
        elif choice == "e":
            new_text = input("  New hinglish_roman: ").strip()
            if new_text:
                tokens = split_tokens(new_text)
                cand["hinglish_roman"] = new_text
                cand["word_level_labels"] = rebuild_labels(tokens, cand.get("word_level_labels", {}))
                approved.append(cand)
        else:
            continue

    existing = load_json_list(OUTPUT_PATH)
    merged = existing + approved

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] Approved entries: {len(approved)}")
    print(f"[SAVED] Output file: {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="EduHinglish — Dataset Augmentation (Ch5)")
    parser.add_argument("--merge", action="store_true", help="Review and merge candidates into dataset_v2.json")
    args = parser.parse_args()

    if args.merge:
        merge_candidates()
        return

    base = load_seed_dataset()
    if not base:
        print("[ERROR] No source dataset found.")
        return

    candidates = generate_candidates(base)
    save_candidates(candidates)

    print(f"Original sentences: {len(base)}")
    print(f"Candidates generated: {len(candidates)}")
    print(f"Review them in: {CANDIDATES_PATH}")
    print("Run merge command after review: python scripts/augment_dataset.py --merge")


if __name__ == "__main__":
    main()
