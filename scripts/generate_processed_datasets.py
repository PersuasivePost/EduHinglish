"""
EduHinglish — Processed Text Dataset Generator
==============================================
Author  : Ashvatth
Module  : M1 — Dataset Creation
Purpose : Generate datasets strictly from data/processed/*/cleaned_text.txt.

Generates per chapter:
  - 75 statements
  - 15 student queries
  - 5 GEC samples

Outputs to:
  data/processed/<chapter>/dataset.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STATEMENTS_COUNT = 75
QUERIES_COUNT = 15
GEC_COUNT = 5
TOTAL_COUNT = STATEMENTS_COUNT + QUERIES_COUNT + GEC_COUNT

HINDI_WORDS = {
    "main", "hum", "tum", "woh", "yeh", "uska", "uski", "iska", "iski",
    "mera", "meri", "tera", "teri", "unka", "unki", "hamara", "tumhara",
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "hota", "hoti", "hote",
    "karta", "karti", "karte", "kiya", "karo", "karna", "karke", "hona",
    "raha", "rahi", "rahe", "gaya", "gayi", "aata", "aati", "jaata", "jaati",
    "deta", "deti", "lete", "bana", "bante", "samjhao", "batao", "dekho",
    "kehte", "kehta", "kehti", "kaha", "dijiye", "hokar", "jinmein", "inmein",
    "ka", "ki", "ke", "ko", "se", "mein", "par", "tak", "pe", "ne", "me",
    "aur", "ya", "lekin", "kyunki", "isliye", "jabki", "phir", "toh", "bhi",
    "hi", "sirf", "bas", "kya", "kaise", "kyun", "kahan", "kab", "kaun",
    "kitna", "bahut", "thoda", "zyada", "kam", "achha", "bada", "bade", "badi",
    "chhota", "naya", "nayi", "pehle", "baad", "andar", "bahar", "upar",
    "neeche", "yahan", "wahan", "abhi", "tab", "jab", "nahi", "nhi", "na",
    "mat", "ek", "do", "teen", "sabhi", "sab", "kuch", "koi", "wala", "wale",
    "wali", "jaise", "taraf", "beech", "kaam", "saath", "jo", "tarah", "matlab",
}

UNIVERSAL_WORDS = {"sir", "madam", "ok", "okay", "hello", "hi", "bye", "please", "thanks", "sorry"}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_sentences(text: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [normalize_space(p) for p in parts if len(normalize_space(p)) > 30]
    return sentences


def predict_label(word: str, position: int) -> str:
    clean = word.strip(".,?!;:\"'()")
    if not clean:
        return "EN"

    if any("\u0900" <= c <= "\u097F" for c in clean):
        return "HI"

    lower = clean.lower()
    if lower in UNIVERSAL_WORDS:
        return "UNIV"
    if lower in HINDI_WORDS:
        return "HI"
    if clean.isdigit():
        return "UNIV"
    if position > 0 and clean and clean[0].isupper():
        return "NE"
    return "EN"


def label_sentence(sentence: str) -> dict:
    tokens = sentence.split()
    labels = {}
    for idx, tok in enumerate(tokens):
        labels[tok.strip(".,?!;:\"'()") or tok] = predict_label(tok, idx)
    return labels


def make_hinglish_statement(english: str) -> str:
    base = english.strip().rstrip(".?!")
    if re.search(r"\b(are|were)\b", base, re.IGNORECASE):
        suffix = " hain"
    else:
        suffix = " hai"
    return f"{base}{suffix}."


def make_query(english: str, suffix: str) -> str:
    words = english.strip().rstrip(".?!").split()
    short = " ".join(words[:12]) if words else english.strip().rstrip(".?!")
    return f"Sir, {short} {suffix}".strip()


def make_gec(hinglish: str) -> tuple[str, str, dict]:
    correct = hinglish.strip()
    if " hai" in correct:
        wrong = correct.replace(" hai", " hain", 1)
        error_word, correct_word = "hain", "hai"
    elif " hain" in correct:
        wrong = correct.replace(" hain", " hai", 1)
        error_word, correct_word = "hai", "hain"
    else:
        correct = correct.rstrip(".") + " hai."
        wrong = correct.replace(" hai", " hain", 1)
        error_word, correct_word = "hain", "hai"

    error_desc = {
        "error_word": error_word,
        "correct_word": correct_word,
        "error_type": "verb agreement",
        "explanation": "Incorrect verb agreement in Hinglish sentence",
    }
    return correct, wrong, error_desc


def extract_chapter_title(text: str, fallback_num: str) -> str:
    first_line = normalize_space(text.split("\n", 1)[0])
    if not first_line:
        return f"Chapter {fallback_num}"
    if "chapter" in first_line.lower():
        return first_line
    return f"Chapter {fallback_num}: {first_line}"


def build_entries(folder: Path, chapter_num: str, class_num: str) -> list[dict]:
    clean_path = folder / "cleaned_text.txt"
    text = clean_path.read_text(encoding="utf-8")
    chapter_title = extract_chapter_title(text, chapter_num)

    sentences = split_sentences(text)
    if not sentences:
        return []

    # Ensure we have enough by cycling if needed
    if len(sentences) < TOTAL_COUNT:
        repeats = (TOTAL_COUNT // len(sentences)) + 1
        sentences = (sentences * repeats)[:TOTAL_COUNT]

    entries: list[dict] = []
    idx = 1

    # Statements
    for s in sentences[:STATEMENTS_COUNT]:
        hinglish = make_hinglish_statement(s)
        entries.append({
            "id": f"{class_num}_{chapter_num}_{idx:03d}",
            "original_english": s,
            "hinglish_roman": hinglish,
            "word_level_labels": label_sentence(hinglish),
            "topic": "General",
            "chapter": chapter_title,
            "class": class_num,
            "code_mixing_type": "intra-sentential",
        })
        idx += 1

    # Queries
    query_suffixes = ["kya?", "kaise?", "kyun?", "kab?", "kahan?"]
    for i, s in enumerate(sentences[STATEMENTS_COUNT:STATEMENTS_COUNT + QUERIES_COUNT]):
        hinglish = make_query(s, query_suffixes[i % len(query_suffixes)])
        entries.append({
            "id": f"{class_num}_{chapter_num}_{idx:03d}",
            "original_english": s,
            "hinglish_roman": hinglish,
            "word_level_labels": label_sentence(hinglish),
            "topic": "General",
            "chapter": chapter_title,
            "class": class_num,
            "code_mixing_type": "intra-sentential",
            "is_student_query": True,
            "intent": "explain_concept",
        })
        idx += 1

    # GEC samples
    for s in sentences[STATEMENTS_COUNT + QUERIES_COUNT:STATEMENTS_COUNT + QUERIES_COUNT + GEC_COUNT]:
        correct = make_hinglish_statement(s)
        correct, wrong, error_desc = make_gec(correct)
        entries.append({
            "id": f"{class_num}_{chapter_num}_{idx:03d}",
            "original_english": s,
            "hinglish_roman": correct,
            "word_level_labels": label_sentence(correct),
            "topic": "General",
            "chapter": chapter_title,
            "class": class_num,
            "code_mixing_type": "intra-sentential",
            "is_gec_sample": True,
            "hinglish_with_error": wrong,
            "error_description": error_desc,
        })
        idx += 1

    return entries


def main() -> None:
    folders = sorted([p for p in PROCESSED_DIR.iterdir() if p.is_dir() and re.match(r"class\d+_ch\d+", p.name)])
    if not folders:
        print("[ERROR] No processed chapter folders found.")
        return

    for folder in folders:
        match = re.match(r"class(\d+)_ch(\d+)", folder.name)
        if not match:
            continue
        class_num = match.group(1)
        chapter_num = match.group(2).zfill(2)
        entries = build_entries(folder, chapter_num, class_num)
        if not entries:
            print(f"[WARN] No sentences found in {folder}/cleaned_text.txt")
            continue

        out_path = folder / "dataset.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

        print(f"[OK] {folder.name}: wrote {len(entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()
