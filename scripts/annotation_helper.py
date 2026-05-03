"""
EduHinglish — Hinglish Annotation Helper
=========================================
Author  : Jatin
Module  : M1 — Dataset Creation Tooling
Purpose : Interactive CLI that makes manual Hinglish sentence annotation fast.
          Both Jatin and Ashvatth use this tool to annotate sentences for
          all chapter datasets.

Workflow per sentence
---------------------
1. User enters the original English sentence
2. User enters the Hinglish (Roman script) version
3. Tool auto-splits Hinglish into words and predicts HI/EN/NE/UNIV label for each
4. Shows prediction; user hits Enter to accept or types a new label to override
5. Tool prompts for: topic, is_student_query, intent (if query), is_gec_sample,
   code_mixing_type, notes
6. Builds complete entry dict and appends to the output JSON array
7. Prints running count after each save

CLI usage
---------
  python scripts/annotation_helper.py --output data/biology/class9/ch05/dataset_v2.json
  python scripts/annotation_helper.py \\
      --chapter class9/ch01 \\
      --output  data/biology/class9/ch01/dataset.json

  --chapter  : auto-fills "chapter" and "class" fields using built-in mapping
  --output   : path to the JSON file (created if missing, appended if present)
  --dry-run  : print entry dict without saving (useful for testing)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# WORD LISTS  (exact as specified in project design doc)
# ─────────────────────────────────────────────────────────────────────────────

HINDI_WORDS: set[str] = {
    # Pronouns
    "main", "hum", "tum", "woh", "yeh", "uska", "uski", "iska", "iski",
    "mera", "meri", "tera", "teri", "unka", "unki", "hamara", "tumhara",
    "iske", "uske", "unke", "inhe", "unhe", "apna", "apni", "apne",
    # Auxiliaries / copula
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "hota", "hoti", "hote",
    # Verbs — karna (to do) forms
    "kar", "karta", "karti", "karte", "kiya", "karo", "karna", "karke", "karne",
    "kari", "karein", "karega", "karegi",
    # Verbs — hona (to be/become) forms
    "hona", "hokar", "hoke", "honi", "hone",
    # Verbs — progressive / past
    "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    # Verbs — motion
    "aata", "aati", "aate", "jaata", "jaati", "jaate", "aana", "jaana",
    # Verbs — giving / taking
    "deta", "deti", "dete", "leta", "leti", "lete", "dena", "lena",
    # Verbs — banana (to make) forms
    "bana", "bani", "bane", "banta", "banti", "bante",
    "banata", "banati", "banaye", "banana",
    # Verbs — knowing / understanding
    "jaanta", "jaanti", "jaante", "samjha", "samjhi", "samjhe",
    "samjhao", "samjho", "samajh",
    # Verbs — telling / seeing / saying
    "batao", "batata", "batati", "batana",
    "dekho", "dekhna", "dekhta", "dekhti", "dekha",
    "kehte", "kehta", "kehti", "kaha", "kehna",
    # Verbs — milna / rehna / dikhna
    "milta", "milti", "milte", "milna",
    "rehta", "rehti", "rehte", "rehna",
    "dikhta", "dikhti", "dikhte", "dikhna", "dikhai",
    # Verbs — paana / dalna / rakhna
    "paaya", "paayi", "paaye", "paana", "pata",
    "daalta", "daalti", "daalte", "daalna",
    "rakhta", "rakhti", "rakhte", "rakhna", "rakha",
    # Verbs — honorific imperatives
    "dijiye", "kijiye", "lijiye",
    # Postpositions
    "ka", "ki", "ke", "ko", "se", "mein", "par", "tak", "pe", "ne", "me",
    # Conjunctions / connectors
    "aur", "ya", "lekin", "kyunki", "isliye", "jabki", "phir", "toh", "bhi",
    "hi", "sirf", "bas", "tatha", "parantu", "magar",
    # Question words
    "kya", "kaise", "kyun", "kahan", "kab", "kaun",
    "kitna", "kitni", "kitne",
    # Adjectives / adverbs
    "bahut", "thoda", "zyada", "kam", "achha", "bura",
    "bada", "bade", "badi", "chhota", "chhoti", "chhote",
    "naya", "nayi", "naye", "alag", "zaruri", "pura", "puri",
    "pehle", "pehla", "pehli", "baad",
    "dusra", "dusri", "dusre",
    # Spatial / temporal
    "andar", "bahar", "upar", "neeche", "yahan", "wahan",
    "abhi", "tab", "jab", "hamesha", "kabhi",
    # Negation
    "nahi", "nhi", "na", "mat", "bina",
    # Numbers (Hindi)
    "ek", "do", "teen", "chaar", "paanch", "dono",
    # Quantifiers / pronouns
    "sabhi", "sab", "kuch", "koi", "wala", "wale", "wali",
    # Others
    "jaise", "jaisa", "jaisi", "taraf", "beech", "kaam", "saath",
    "jinmein", "inmein", "jo", "tarah", "matlab", "yaani",
    "wajah", "cheez", "jagah", "tarika", "prakar",
}

UNIVERSAL_WORDS: set[str] = {
    "sir", "madam", "ok", "okay", "hello", "hi", "bye",
    "please", "thanks", "sorry",
}

VALID_LABELS: tuple[str, ...] = ("HI", "EN", "NE", "UNIV", "MIX")

# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER CODE → TITLE MAPPING  (matches actual PDF filenames on disk)
# ─────────────────────────────────────────────────────────────────────────────

CHAPTER_MAP: dict[str, tuple[str, str]] = {
    # (chapter_title, class_number)
    # Class 9
    "class9/ch05":  ("Chapter 5: The Fundamental Unit of Life", "9"),
    "class9/ch06":  ("Chapter 6: Tissues", "9"),
    "class9/ch07":  ("Chapter 7: Diversity in Living Organisms", "9"),
    "class9/ch13":  ("Chapter 13: Why Do We Fall Ill", "9"),
    "class9/ch14":  ("Chapter 14: Natural Resources", "9"),
    "class9/ch15":  ("Chapter 15: Improvement in Food Resources", "9"),
    # Class 10
    "class10/ch06": ("Chapter 6: Life Processes", "10"),
    "class10/ch08": ("Chapter 8: How do Organisms Reproduce?", "10"),
    "class10/ch09": ("Chapter 9: Heredity and Evolution", "10"),
    "class10/ch15": ("Chapter 15: Our Environment", "10"),
    "class10/ch16": ("Chapter 16: Management of Natural Resources", "10"),
}

INTENT_VALUES: tuple[str, ...] = (
    "explain_concept",
    "compare_concepts",
    "give_example",
    "formula_request",
    "definition",
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _divider() -> None:
    print("  " + "─" * 43)


def _contains_devanagari(word: str) -> bool:
    """Return True if word contains any Devanagari Unicode character."""
    return any(unicodedata.category(ch) == "Lo" and "\u0900" <= ch <= "\u097F"
               for ch in word)


def _is_number(word: str) -> bool:
    """Return True for purely numeric tokens (integers, decimals, years)."""
    return bool(re.fullmatch(r"[\d,.\-]+", word))


def predict_label(word: str, position: int) -> str:
    """
    Predict HI / EN / NE / UNIV for a single word using rule-based logic.

    Rules (in priority order):
      1. Contains Devanagari → HI
      2. Lowercase form in UNIVERSAL_WORDS → UNIV
      3. Lowercase form in HINDI_WORDS → HI
      4. Is a number → UNIV
      5. All-uppercase token (e.g. ATP, DNA, RNA) → EN (science abbreviation)
      6. Starts with capital AND not the first word → NE (candidate)
      7. Otherwise → EN
    """
    clean = word.strip(".,?!;:\"'()")

    if _contains_devanagari(clean):
        return "HI"

    lower = clean.lower()

    if lower in UNIVERSAL_WORDS:
        return "UNIV"

    if lower in HINDI_WORDS:
        return "HI"

    if _is_number(clean):
        return "UNIV"

    # All-uppercase (len > 1) → science abbreviation, tag EN (not NE)
    if clean.isupper() and len(clean) > 1:
        return "EN"

    # Capital-initial AND not the very first token → Named Entity candidate
    if position > 0 and clean and clean[0].isupper():
        return "NE"

    return "EN"


def tokenise_hinglish(hinglish: str) -> list[str]:
    """
    Split Hinglish sentence into individual word tokens.
    Preserves hyphenated compounds as single tokens (well-defined).
    Strips trailing punctuation from each token.
    """
    raw_tokens = hinglish.split()
    tokens: list[str] = []
    for tok in raw_tokens:
        # Strip only sentence-boundary punctuation, keep hyphens inside
        cleaned = tok.strip(".,?!;:\"'()")
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _prompt(message: str, default: str = "") -> str:
    """Show a prompt and return stripped user input."""
    try:
        val = input(message)
    except EOFError:
        return default
    return val.strip() if val.strip() else default


# ─────────────────────────────────────────────────────────────────────────────
# CORE ANNOTATION SESSION
# ─────────────────────────────────────────────────────────────────────────────

def annotate_sentence(
    chapter_title: str | None,
    class_num: str | None,
    entry_number: int,
) -> dict | None:
    """
    Run one full annotation session for a single sentence.
    Returns the completed entry dict, or None if user aborts.
    """
    print()
    _divider()
    print(f"  SENTENCE #{entry_number}")
    _divider()

    # ── English source ────────────────────────────────────────────────────────
    original_english = _prompt("  Original English (leave blank to quit): ")
    if not original_english:
        return None

    # ── Hinglish version ──────────────────────────────────────────────────────
    hinglish_roman = _prompt("  Hinglish (Roman script): ")
    if not hinglish_roman:
        print("  [SKIP] Empty Hinglish — skipping this sentence.")
        return None

    # ── Optional Devanagari ───────────────────────────────────────────────────
    hinglish_devanagari = _prompt(
        "  Hinglish (Devanagari, optional — press Enter to skip): "
    )

    print()

    # ── Word-level annotation ─────────────────────────────────────────────────
    tokens = tokenise_hinglish(hinglish_roman)
    word_level_labels: dict[str, str] = {}

    for idx, word in enumerate(tokens):
        prediction = predict_label(word, idx)
        _divider()
        total = len(tokens)
        print(f"  Word {idx + 1}/{total}: \"{word}\"")
        print(f"  Prediction: {prediction}  ← accept? "
              f"[Enter=yes / type label to change ({'/'.join(VALID_LABELS)})]: ",
              end="")
        try:
            user_input = input().strip().upper()
        except EOFError:
            user_input = ""

        if user_input == "":
            label = prediction
        elif user_input in VALID_LABELS:
            label = user_input
        else:
            print(f"  [WARN] '{user_input}' is not a valid label. "
                  f"Keeping prediction '{prediction}'.")
            label = prediction

        word_level_labels[word] = label

    # ── Summary + metadata ────────────────────────────────────────────────────
    _divider()
    print(f"  Labels: {word_level_labels}")
    _divider()

    topic = _prompt("  Topic: ")

    # Chapter / class auto-filled or prompted
    if chapter_title and class_num:
        chapter = chapter_title
        klass = class_num
        print(f"  Chapter: {chapter}  (auto-filled)")
        print(f"  Class:   {klass}    (auto-filled)")
    else:
        chapter = _prompt("  Chapter (e.g. 'Chapter 5: The Fundamental Unit of Life'): ")
        klass   = _prompt("  Class (e.g. 9 or 10): ")

    # Code mixing type
    cmt_raw = _prompt("  Code mixing type [intra/inter] (default: intra): ", "intra")
    code_mixing_type = "intra-sentential" if cmt_raw.lower().startswith("intr") \
                       else "inter-sentential"

    # Student query?
    is_student_query_raw = _prompt("  Is student query? [y/n]: ", "n").lower()
    is_student_query = is_student_query_raw == "y"

    intent: str | None = None
    if is_student_query:
        print(f"  Intent options: {', '.join(INTENT_VALUES)}")
        intent_raw = _prompt("  Intent: ", "explain_concept").lower().strip()
        # Fuzzy match
        matched = next((v for v in INTENT_VALUES if intent_raw == v or
                        intent_raw in v), "explain_concept")
        intent = matched

    # GEC sample?
    is_gec_raw = _prompt("  Is GEC sample? [y/n]: ", "n").lower()
    is_gec_sample = is_gec_raw == "y"

    hinglish_with_error: str | None = None
    error_description: dict | None = None
    if is_gec_sample:
        hinglish_with_error = _prompt(
            "  Hinglish WITH error (erroneous version): "
        )
        error_word = _prompt("  Error word / phrase: ")
        correct_word = _prompt("  Correct word / phrase: ")
        error_type = _prompt("  Error type (e.g. 'gender agreement'): ")
        explanation = _prompt("  Explanation: ")
        error_description = {
            "error_word":   error_word,
            "correct_word": correct_word,
            "error_type":   error_type,
            "explanation":  explanation,
        }

    notes = _prompt("  Notes (optional): ")

    # ── Build entry dict ──────────────────────────────────────────────────────
    entry: dict = {
        "id":                entry_number,
        "original_english":  original_english,
        "hinglish_roman":    hinglish_roman,
        "word_level_labels": word_level_labels,
        "topic":             topic,
        "chapter":           chapter,
        "class":             klass,
        "code_mixing_type":  code_mixing_type,
    }

    if hinglish_devanagari:
        entry["hinglish_devanagari"] = hinglish_devanagari

    if is_student_query:
        entry["is_student_query"] = True
        entry["intent"] = intent

    if is_gec_sample:
        entry["is_gec_sample"] = True
        if hinglish_with_error:
            entry["hinglish_with_error"] = hinglish_with_error
        if error_description:
            entry["error_description"] = error_description

    if notes:
        entry["notes"] = notes

    return entry


# ─────────────────────────────────────────────────────────────────────────────
# JSON I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_existing(path: Path) -> list[dict]:
    """Load existing JSON array from file, or return empty list."""
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            print(f"  [WARN] {path} does not contain a JSON array — starting fresh.")
        except json.JSONDecodeError as exc:
            print(f"  [WARN] Could not parse {path}: {exc} — starting fresh.")
    return []


def save_dataset(path: Path, dataset: list[dict]) -> None:
    """Write dataset to JSON with pretty-printing and UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _label_stats(dataset: list[dict]) -> str:
    """Return a one-line label distribution summary."""
    from collections import Counter
    counter: Counter = Counter()
    for entry in dataset:
        counter.update(entry.get("word_level_labels", {}).values())
    total = sum(counter.values()) or 1
    parts = [f"{lbl}={counter[lbl]}({counter[lbl]*100//total}%)"
             for lbl in VALID_LABELS if counter[lbl]]
    return "  " + " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SESSION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_session(
    output_path: Path,
    chapter_code: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    Main annotation loop.
    Continues until user enters 'q' at the "Next sentence?" prompt.
    """
    # Resolve chapter metadata
    chapter_title: str | None = None
    class_num: str | None = None
    if chapter_code:
        if chapter_code not in CHAPTER_MAP:
            print(f"\n  [ERROR] Unknown chapter code '{chapter_code}'.")
            print("  Available codes:")
            for code in CHAPTER_MAP:
                title, cls = CHAPTER_MAP[code]
                print(f"    {code}  →  {title}  (Class {cls})")
            sys.exit(1)
        chapter_title, class_num = CHAPTER_MAP[chapter_code]

    # Load existing data
    dataset: list[dict] = [] if dry_run else load_existing(output_path)
    next_id = len(dataset) + 1

    # Header
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║     EduHinglish — Annotation Helper          ║")
    print("  ╚══════════════════════════════════════════════╝")
    if chapter_title:
        print(f"  Chapter : {chapter_title}  (Class {class_num})")
    print(f"  Output  : {output_path}")
    print(f"  Existing: {len(dataset)} entries")
    if dry_run:
        print("  Mode    : DRY RUN (nothing will be saved)")
    print()
    print("  Tip — valid labels: HI | EN | NE | UNIV | MIX")
    print("        Leave label prompt blank to accept the prediction.")
    print("        Enter blank English sentence at any time to quit.")

    session_count = 0

    while True:
        entry = annotate_sentence(chapter_title, class_num, next_id)

        if entry is None:
            # User left English sentence blank → quit
            break

        if dry_run:
            _divider()
            print("  [DRY-RUN] Entry would be saved as:")
            print("  " + json.dumps(entry, ensure_ascii=False, indent=4)
                  .replace("\n", "\n  "))
        else:
            dataset.append(entry)
            save_dataset(output_path, dataset)
            _divider()
            print(f"  [SAVED] Entry #{next_id} → {output_path}")
            print(f"  Total entries in file: {len(dataset)}")
            print(_label_stats(dataset))

        session_count += 1
        next_id += 1

        _divider()
        cont = _prompt("  Next sentence? [Enter=continue / q=quit]: ", "")
        if cont.lower() == "q":
            break

    # Session summary
    print()
    _divider()
    print(f"  Session complete — {session_count} sentence(s) annotated this session.")
    if not dry_run:
        print(f"  File now has {len(dataset)} total entries: {output_path}")
    _divider()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotation_helper",
        description=(
            "EduHinglish interactive Hinglish annotation tool.\n"
            "Annotate sentences word-by-word with HI/EN/NE/UNIV/MIX labels\n"
            "and save to a JSON dataset file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        required=False,
        default=None,
        metavar="PATH",
        help="Path to output JSON file (appended to if it exists).",
    )
    parser.add_argument(
        "--chapter", "-c",
        metavar="CODE",
        help=(
            "Chapter code to auto-fill chapter title and class. "
            f"Options: {', '.join(CHAPTER_MAP.keys())}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resulting entry dict without saving to file.",
    )
    parser.add_argument(
        "--list-chapters",
        action="store_true",
        help="List all known chapter codes and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_chapters:
        print("\n  Known chapter codes:\n")
        for code, (title, cls) in CHAPTER_MAP.items():
            print(f"    {code:<20}  Class {cls}  →  {title}")
        print()
        sys.exit(0)

    if not args.output:
        parser.error("--output / -o is required unless --list-chapters is used.")

    output_path = Path(args.output)

    run_session(
        output_path=output_path,
        chapter_code=args.chapter,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
