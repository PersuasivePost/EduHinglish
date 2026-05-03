"""
EduHinglish — Dataset Validator
================================
Author  : Ashvatth
Module  : M1 — Dataset Quality Checks
Purpose : Validate dataset.json files across chapters and report issues.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --chapter class9/ch05
    python scripts/validate_dataset.py --fix
"""

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "biology"

REQUIRED_FIELDS = {
    "id",
    "original_english",
    "hinglish_roman",
    "word_level_labels",
    "topic",
    "chapter",
    "class",
}

VALID_LABELS = {"HI", "EN", "NE", "UNIV", "MIX"}


def load_json_list(path: Path) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_json_list(path: Path, data: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def tokenize(text: str) -> list:
    return text.split()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def compute_cmi(labels: dict) -> float | None:
    if not isinstance(labels, dict):
        return None

    hi = sum(1 for v in labels.values() if v == "HI")
    en = sum(1 for v in labels.values() if v == "EN")
    total = hi + en
    if total == 0:
        return None

    cmi = (1 - max(hi, en) / total) * 100
    return round(cmi, 1)


def validate_entry(entry: dict) -> tuple[list, float | None]:
    issues = []

    # 1. Required fields
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        issues.append(f"Missing fields: {', '.join(missing)}")

    # 2. word_level_labels keys match tokens
    hinglish = entry.get("hinglish_roman", "")
    tokens = tokenize(hinglish) if isinstance(hinglish, str) else []
    label_keys = list(entry.get("word_level_labels", {}).keys())
    if tokens and label_keys:
        token_set = set(tokens)
        label_set = set(label_keys)
        if token_set != label_set:
            issues.append("word_level_labels keys do not match tokens")

    # 3. valid label values
    labels = entry.get("word_level_labels", {})
    if isinstance(labels, dict):
        invalid = {v for v in labels.values() if v not in VALID_LABELS}
        if invalid:
            issues.append(f"Invalid label values: {', '.join(sorted(invalid))}")

    # 4. student query intent
    if entry.get("is_student_query") is True and not entry.get("intent"):
        issues.append("Student query missing intent")

    # 5. GEC fields
    if entry.get("is_gec_sample") is True:
        if not entry.get("hinglish_with_error") or not entry.get("error_description"):
            issues.append("GEC sample missing hinglish_with_error or error_description")

    # 6. hinglish_roman not identical to original_english
    original = entry.get("original_english", "")
    if isinstance(original, str) and isinstance(hinglish, str):
        if normalize_text(original).lower() == normalize_text(hinglish).lower():
            issues.append("hinglish_roman identical to original_english")

    cmi = compute_cmi(labels)
    if cmi is None:
        issues.append("CMI could not be computed")
    else:
        if not (10 <= cmi <= 70):
            issues.append(f"CMI out of range: {cmi}")

    return issues, cmi


def auto_fix_entry(entry: dict) -> dict:
    # Trim whitespace in key string fields
    for key in ("original_english", "hinglish_roman", "topic", "chapter", "class"):
        if isinstance(entry.get(key), str):
            entry[key] = normalize_text(entry[key])

    # Normalize label values to uppercase
    labels = entry.get("word_level_labels")
    if isinstance(labels, dict):
        entry["word_level_labels"] = {k: str(v).upper() for k, v in labels.items()}

    return entry


def find_datasets(chapter: str | None = None) -> list[Path]:
    if chapter:
        class_part, ch_part = chapter.split("/")
        target = DATA_ROOT / class_part / ch_part
        return sorted(target.glob("dataset*.json"))

    return sorted(DATA_ROOT.glob("class*/ch*/dataset*.json"))


def print_dataset_summary(label: str, entries: list, cmi_values: list, issues_count: int):
    total = len(entries)
    student_queries = sum(1 for e in entries if e.get("is_student_query") is True)
    gec_samples = sum(1 for e in entries if e.get("is_gec_sample") is True)
    statements = total - student_queries - gec_samples
    avg_cmi = round(sum(cmi_values) / max(len(cmi_values), 1), 1) if cmi_values else 0.0

    print(f"Chapter: {label}")
    print(f"Total entries:      {total}")
    print(f"Statements:         {statements}")
    print(f"Student queries:    {student_queries}")
    print(f"GEC samples:        {gec_samples}")
    print(f"Avg CMI:            {avg_cmi}%")
    print(f"Validation issues:  {issues_count}")
    print("-" * 22)


def main():
    parser = argparse.ArgumentParser(description="EduHinglish — Dataset Validator")
    parser.add_argument("--chapter", type=str, default=None,
                        help="Validate a single chapter, e.g. class9/ch05")
    parser.add_argument("--fix", action="store_true", help="Auto-fix trivial issues")
    args = parser.parse_args()

    datasets = find_datasets(args.chapter)
    if not datasets:
        print("[ERROR] No dataset.json files found.")
        return

    total_entries = 0
    total_issues = 0
    chapters_complete = 0
    chapters_incomplete = 0

    for path in datasets:
        entries = load_json_list(path)
        if args.fix:
            entries = [auto_fix_entry(e) for e in entries]

        seen_ids = set()
        dup_ids = 0
        issue_count = 0
        cmi_values = []

        for entry in entries:
            entry_id = entry.get("id")
            if entry_id in seen_ids:
                dup_ids += 1
            seen_ids.add(entry_id)

            issues, cmi = validate_entry(entry)
            if issues:
                issue_count += len(issues)
            if cmi is not None:
                cmi_values.append(cmi)

        if dup_ids:
            issue_count += dup_ids

        label = path.parent.parent.name + "/" + path.parent.name
        print_dataset_summary(label, entries, cmi_values, issue_count)

        total_entries += len(entries)
        total_issues += issue_count
        if len(entries) >= 50:
            chapters_complete += 1
        else:
            chapters_incomplete += 1

        if args.fix:
            save_json_list(path, entries)

    print("Cross-dataset stats:")
    print(f"Total entries across all chapters: {total_entries}")
    print(f"Chapters complete (>=50 entries):  {chapters_complete} / {chapters_complete + chapters_incomplete}")
    print(f"Chapters incomplete (<50 entries): {chapters_incomplete}")
    print(f"Total validation issues:           {total_issues}")


if __name__ == "__main__":
    main()
