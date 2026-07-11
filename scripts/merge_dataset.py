"""
Merge all data sources into unified_science_dataset_v2.json and data/unified_biology_dataset.json:
  - data/processed/class9_ch02/dataset.json
  - data/processed/class9_ch03/dataset.json
  - data/processed/class9_ch11/dataset.json
  - data/processed/class9_ch12/dataset.json
  - data/processed/class10_ch05/dataset.json
  - data/processed/class10_ch06/dataset.json
  - data/processed/class10_ch07/dataset.json
  - data/processed/class10_ch08/dataset.json
  - data/processed/class10_ch13/dataset.json
  - data/processed/general1/dataset.json
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

FILES = [
    "data/processed/class9_ch02/dataset.json",
    "data/processed/class9_ch03/dataset.json",
    "data/processed/class9_ch11/dataset.json",
    "data/processed/class9_ch12/dataset.json",
    "data/processed/class10_ch05/dataset.json",
    "data/processed/class10_ch06/dataset.json",
    "data/processed/class10_ch07/dataset.json",
    "data/processed/class10_ch08/dataset.json",
    "data/processed/class10_ch13/dataset.json",
    "data/processed/general1/dataset.json"
]

def main():
    all_entries = []
    for fpath in FILES:
        full_path = ROOT / fpath
        if not full_path.exists():
            print(f"WARNING: File {fpath} not found - skipping")
            continue
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries from {fpath}")
        all_entries.extend(data)

    # Filter invalid entries
    valid_entries = [e for e in all_entries if e.get("original_english") and e.get("hinglish_roman")]
    print(f"Total valid entries: {len(valid_entries)} (removed {len(all_entries) - len(valid_entries)} nulls)")

    # Sort all by ID alphabetically
    valid_entries.sort(key=lambda x: x["id"])

    # Save to the targets
    targets = [
        "data/unified_biology_dataset.json",
        "data/unified_biology_dataset_v2.json",
        "data/unified_science_dataset_v2.json"
    ]

    for target in targets:
        out_path = ROOT / target
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(valid_entries, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(valid_entries)} sorted entries → {target}")

if __name__ == "__main__":
    main()
