"""
EduHinglish — Chapter Dataset Builder
=====================================
A guided batch entry tool for rapidly annotating many sentences from a chapter.
Reads candidate sentences from cleaned_text.txt, shows them as prompts,
and records Hinglish annotations using annotation_helper logic.
"""

import argparse
import sys
import re
import textwrap
from pathlib import Path

try:
    from annotation_helper import (
        CHAPTER_MAP, load_existing, save_dataset, annotate_sentence,
    )
except ImportError:
    print("  [ERROR] Could not import annotation_helper.py. Run from project root.")
    sys.exit(1)

def _divider_thin():
    print("  " + "─" * 43)

def _divider_thick():
    print("  " + "═" * 43)

def extract_candidates(cleaned_text_path: Path) -> list[str]:
    if not cleaned_text_path.exists():
        return []
    
    with open(cleaned_text_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Basic sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    bio_terms = ["cell", "tissue", "organ", "plant", "animal", "bacteria", 
                 "membrane", "nucleus", "muscle", "nerve", "disease", 
                 "health", "reproduction", "energy", "blood", "epithelial", 
                 "connective", "meristematic", "permanent"]
    
    candidates = []
    for s in sentences:
        s = s.strip().replace("\n", " ")
        if len(s) > 50:
            if any(term in s.lower() for term in bio_terms):
                candidates.append(s)
                
    # Fallback to length if few biological matches
    if len(candidates) < 10:
        for s in sentences:
            s = s.strip().replace("\n", " ")
            if len(s) > 50 and s not in candidates:
                candidates.append(s)
                
    return candidates

def build_progress_bar(current: int, total: int) -> str:
    filled = min(current * 10 // total, 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"[{bar}] {current}/{total}"

def main():
    parser = argparse.ArgumentParser(description="Guided batch entry tool for EduHinglish")
    parser.add_argument("--chapter", required=True, help="Chapter code (e.g. class9/ch06)")
    args = parser.parse_args()

    chapter_code = args.chapter
    if chapter_code not in CHAPTER_MAP:
        print(f"  [ERROR] Unknown chapter code '{chapter_code}'.")
        print("  Available codes:")
        for code in CHAPTER_MAP:
            print(f"    {code}")
        sys.exit(1)
        
    chapter_title, class_num = CHAPTER_MAP[chapter_code]
    
    out_dir = Path("data/biology") / chapter_code
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset.json"
    cleaned_text_path = out_dir / "cleaned_text.txt"
    
    dataset = load_existing(out_path)
    
    # Track metrics
    statements = sum(1 for e in dataset if not e.get("is_student_query") and not e.get("is_gec_sample"))
    queries = sum(1 for e in dataset if e.get("is_student_query"))
    gecs = sum(1 for e in dataset if e.get("is_gec_sample"))
    total_entries = len(dataset)
    target_total = 50
    
    print()
    _divider_thick()
    print(f"  {chapter_title}")
    print(f"  Target: 30 statements + 15 queries + 5 GEC = {target_total} entries")
    print(f"  Output: {out_path}")
    _divider_thick()
    
    candidates = extract_candidates(cleaned_text_path)
    if not candidates:
        print(f"  [WARN] No candidates found. (Is {cleaned_text_path} present?)")
        print("  Generating dummy candidates from Chapter 6 for demonstration...")
        candidates = [
            "A group of cells that are similar in structure and/or work together to achieve a particular function forms a tissue.",
            "Plant tissues are of two main types – meristematic and permanent.",
            "Different types of animal tissues are epithelial, connective, muscular and nervous tissue.",
            "Blood is a type of connective tissue, and muscle is a type of muscular tissue.",
            "The cells of connective tissue are loosely spaced and embedded in an intercellular matrix."
        ]
        
    chunk_size = 5
    for i in range(0, len(candidates), chunk_size):
        chunk = candidates[i:i+chunk_size]
        
        print("\n  REFERENCE SENTENCES FROM NCERT TEXT:")
        for idx, sentence in enumerate(chunk, 1):
            wrapped = textwrap.fill(f"[{idx}] \"{sentence}\"", width=80, subsequent_indent="       ")
            print(f"  {wrapped}")
            
        for idx, english_sent in enumerate(chunk, 1):
            print()
            print(f"  Progress: {build_progress_bar(total_entries, target_total)}")
            print(f"  (Stats: {statements} statements, {queries} queries, {gecs} GEC)")
            
            entry = annotate_sentence(
                chapter_title=chapter_title,
                class_num=class_num,
                entry_number=total_entries + 1,
                prefilled_english=english_sent
            )
            
            if entry == "QUIT":
                print("\n  [QUIT] Exiting batch builder.")
                sys.exit(0)
                
            if isinstance(entry, dict):
                if entry.get("is_student_query"):
                    queries += 1
                elif entry.get("is_gec_sample"):
                    gecs += 1
                else:
                    statements += 1
                    
                dataset.append(entry)
                save_dataset(out_path, dataset)
                total_entries = len(dataset)
                
                print(f"  [SAVED] Entry #{total_entries} added to dataset.")

    print("\n  [DONE] Reached end of candidate sentences.")

if __name__ == "__main__":
    main()
