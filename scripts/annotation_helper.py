"""
EduHinglish — Annotation Helper CLI
=====================================
Author  : Ashvatth
Module  : M1 — Dataset Creation Tool
Purpose : Fast interactive CLI for creating labeled Hinglish dataset entries
          one by one. Covers factual statements, student queries, and GEC samples.
          Output JSON format matches src/hinglish_dataset_creator.py exactly.

Usage:
    python scripts/annotation_helper.py --chapter class9/ch06
    python scripts/annotation_helper.py --chapter class10/ch06
    python scripts/annotation_helper.py --chapter class9/ch05    # starts IDs at 021
    python scripts/annotation_helper.py --chapter class9/ch06 --output data/hinglish/custom.json
    python scripts/annotation_helper.py --chapter class9/ch06 --target 50
"""

# ── Force UTF-8 output on Windows ──────────────────────────────────────────
import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import re
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# WORD SETS FOR AUTO-PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

HINDI_WORDS = {
    # Pronouns
    "main", "hum", "tum", "woh", "yeh", "uska", "uski", "iska", "iski",
    "mera", "meri", "tera", "teri", "unka", "unki", "hamara", "tumhara",
    # Auxiliaries / verbs
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "hota", "hoti", "hote",
    "karta", "karti", "karte", "kiya", "karo", "karna", "karke", "hona",
    "raha", "rahi", "rahe", "gaya", "gayi", "aata", "aati", "jaata", "jaati",
    "deta", "deti", "lete", "bana", "bante", "samjhao", "batao", "dekho",
    "kehte", "kehta", "kehti", "kaha", "dijiye", "hokar", "jinmein", "inmein",
    # Postpositions
    "ka", "ki", "ke", "ko", "se", "mein", "par", "tak", "pe", "ne", "me",
    # Conjunctions
    "aur", "ya", "lekin", "kyunki", "isliye", "jabki", "phir", "toh", "bhi",
    "hi", "sirf", "bas",
    # Question words
    "kya", "kaise", "kyun", "kahan", "kab", "kaun", "kitna",
    # Adjectives / adverbs
    "bahut", "thoda", "zyada", "kam", "achha", "bada", "bade", "badi",
    "chhota", "naya", "nayi", "pehle", "baad", "andar", "bahar", "upar",
    "neeche", "yahan", "wahan", "abhi", "tab", "jab",
    # Negation
    "nahi", "nhi", "na", "mat",
    # Numbers
    "ek", "do", "teen",
    # Others
    "sabhi", "sab", "kuch", "koi", "wala", "wale", "wali", "jaise", "taraf",
    "beech", "kaam", "saath", "jo", "tarah", "matlab",
}

UNIVERSAL_WORDS = {
    "sir", "madam", "ok", "okay", "hello", "hi", "bye", "please", "thanks", "sorry",
}

VALID_LABELS = {"HI", "EN", "NE", "UNIV", "MIX"}

INTENT_MAP = {
    "1": "explain_concept",
    "2": "compare_concepts",
    "3": "give_example",
    "4": "formula_request",
    "5": "definition",
    "explain":  "explain_concept",
    "compare":  "compare_concepts",
    "example":  "give_example",
    "formula":  "formula_request",
    "definition": "definition",
}

# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER METADATA
# ─────────────────────────────────────────────────────────────────────────────

CHAPTER_TITLES = {
    "class9/ch01": {
        "title": "Chapter 1: Matter in Our Surroundings",
        "class": "9", "text_key": "class9_ch01",
        "id_prefix": "9_01", "start_from": 1,
    },
    "class9/ch02": {
        "title": "Chapter 2: The Fundamental Unit of Life",
        "class": "9", "text_key": "class9_ch02",
        "id_prefix": "9_02", "start_from": 1,
    },
    "class9/ch03": {
        "title": "Chapter 3: Tissues",
        "class": "9", "text_key": "class9_ch03",
        "id_prefix": "9_03", "start_from": 1,
    },
    "class9/ch11": {
        "title": "Chapter 11: Work and Energy",
        "class": "9", "text_key": "class9_ch11",
        "id_prefix": "9_11", "start_from": 1,
    },
    "class9/ch12": {
        "title": "Chapter 12: Sound",
        "class": "9", "text_key": "class9_ch12",
        "id_prefix": "9_12", "start_from": 1,
    },
    "class10/ch05": {
        "title": "Chapter 5: Life Processes",
        "class": "10", "text_key": "class10_ch05",
        "id_prefix": "10_05", "start_from": 1,
    },
    "class10/ch06": {
        "title": "Chapter 6: Control and Coordination",
        "class": "10", "text_key": "class10_ch06",
        "id_prefix": "10_06", "start_from": 1,
    },
    "class10/ch07": {
        "title": "Chapter 7: How do Organisms Reproduce?",
        "class": "10", "text_key": "class10_ch07",
        "id_prefix": "10_07", "start_from": 1,
    },
    "class10/ch08": {
        "title": "Chapter 8: Heredity",
        "class": "10", "text_key": "class10_ch08",
        "id_prefix": "10_08", "start_from": 1,
    },
    "class10/ch13": {
        "title": "Chapter 13: Our Environment",
        "class": "10", "text_key": "class10_ch13",
        "id_prefix": "10_13", "start_from": 1,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def predict_label(word: str, is_first_word: bool = False) -> str:
    """
    Predict HI / EN / NE / UNIV / MIX for a single word.

    Priority order:
      1. Contains Devanagari chars → HI
      2. In UNIVERSAL_WORDS → UNIV
      3. In HINDI_WORDS → HI
      4. Is purely numeric → UNIV
      5. Starts with capital (and not first word) → NE (candidate)
      6. Otherwise → EN
    """
    clean = re.sub(r"[^\w]", "", word)
    if not clean:
        return "EN"

    # Rule 1: Devanagari
    if any("\u0900" <= c <= "\u097F" for c in clean):
        roman_chars = sum(1 for c in clean if c.isascii() and c.isalpha())
        devanagari_chars = sum(1 for c in clean if "\u0900" <= c <= "\u097F")
        if roman_chars > 0 and devanagari_chars > 0:
            return "MIX"
        return "HI"

    lower = clean.lower()

    # Rule 2: Universal words
    if lower in UNIVERSAL_WORDS:
        return "UNIV"

    # Rule 3: Hindi word list
    if lower in HINDI_WORDS:
        return "HI"

    # Rule 4: Pure number
    if clean.isdigit():
        return "UNIV"

    # Rule 5: Capitalized non-first word → Named Entity candidate
    if not is_first_word and clean[0].isupper():
        return "NE"

    # Default
    return "EN"


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE SENTENCES FROM CLEANED TEXT
# ─────────────────────────────────────────────────────────────────────────────

def load_reference_sentences(text_key: str, n: int = 5) -> list:
    """
    Load n diverse reference sentences from the chapter's cleaned_text.txt.
    Returns list of sentence strings. Returns [] if file not found.
    """
    text_path = PROJECT_ROOT / "data" / "processed" / text_key / "cleaned_text.txt"

    if not text_path.exists():
        return []

    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(text_path, "r", encoding="latin-1") as f:
            text = f.read()

    # Split into sentences on . or \n
    raw = re.split(r"(?<=[.!?])\s+|\n", text)
    sentences = [s.strip() for s in raw if len(s.strip()) > 55]

    if not sentences:
        return []

    # Pick evenly spaced sentences for variety
    step = max(1, len(sentences) // n)
    selected = [sentences[i * step] for i in range(min(n, len(sentences)))]

    # Trim to max 120 chars for display
    return [s[:120] + ("..." if len(s) > 120 else "") for s in selected]


# ─────────────────────────────────────────────────────────────────────────────
# DATASET FILE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(output_path: Path) -> list:
    """Load existing entries from the output JSON file."""
    if not output_path.exists():
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, Exception):
        return []


def save_dataset(output_path: Path, entries: list):
    """Save all entries to the output JSON file with UTF-8 encoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def next_entry_id(entries: list, id_prefix: str, start_from: int) -> str:
    """
    Compute the next ID string.
    Format: {id_prefix}_{number:03d}  e.g. "9_06_001"

    Finds the highest existing numeric suffix for this prefix and adds 1.
    If no entries yet, uses start_from.
    """
    existing_nums = []
    pattern = re.compile(rf"^{re.escape(id_prefix)}_(\d+)$")
    for entry in entries:
        entry_id = str(entry.get("id", ""))
        m = pattern.match(entry_id)
        if m:
            existing_nums.append(int(m.group(1)))

    if existing_nums:
        next_num = max(existing_nums) + 1
    else:
        next_num = start_from

    return f"{id_prefix}_{next_num:03d}"


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

SEP = "-" * 55

def hr():
    print(SEP)

def prompt(msg: str, default: str = "") -> str:
    """Input with flush to ensure it shows on Windows."""
    sys.stdout.flush()
    val = input(msg)
    return val.strip() if val.strip() else default

def show_banner(chapter_info: dict, output_path: Path, target: int, entry_count: int):
    print()
    print("=" * 55)
    print("  EduHinglish -- Annotation Helper")
    print("=" * 55)
    print(f"  Chapter : {chapter_info['title']}")
    print(f"  Class   : {chapter_info['class']}")
    print(f"  Output  : {output_path.relative_to(PROJECT_ROOT)}")
    print(f"  Saved   : {entry_count} / {target} entries")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION FLOW — ONE ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def annotate_one_entry(
    chapter_info: dict,
    ref_sentences: list,
    entries: list,
    output_path: Path,
    target: int,
) -> bool:
    """
    Run the interactive annotation flow for ONE entry.

    Returns True to continue, False to quit.
    """

    # ── Show reference sentences ──────────────────────────────────────────────
    print()
    hr()
    print("  REFERENCE SENTENCES (from NCERT cleaned text):")
    hr()
    if ref_sentences:
        for i, s in enumerate(ref_sentences, 1):
            print(f"  [{i}] {s}")
    else:
        print("  (No cleaned text found for this chapter yet.)")
        print("  Run batch_pdf_extractor.py first to extract chapter text.")
    hr()

    # ── English sentence ──────────────────────────────────────────────────────
    print()
    print("  STEP 1 / 7 -- English sentence")
    print("  (Type the NCERT English sentence you want to convert, or 'q' to quit)")
    english = prompt("  English > ")
    if english.lower() == "q":
        return False
    if not english:
        print("  [SKIP] Empty input, skipping.")
        return True

    # ── Hinglish version ──────────────────────────────────────────────────────
    print()
    print("  STEP 2 / 7 -- Your Hinglish version of the sentence above")
    hinglish = prompt("  Hinglish > ")
    if not hinglish:
        print("  [SKIP] Empty input, skipping.")
        return True

    # ── Word-level labeling ───────────────────────────────────────────────────
    print()
    print("  STEP 3 / 7 -- Word-level labeling")
    print(f"  Sentence: \"{hinglish}\"")
    hr()
    print("  For each word: press Enter to accept prediction, OR type a label")
    print("  Valid labels: HI  EN  NE  UNIV  MIX")
    hr()

    words = hinglish.split()
    labels: dict = {}

    for i, word in enumerate(words):
        clean_word = re.sub(r"[^\w]", "", word)
        if not clean_word:
            labels[word] = "EN"
            continue

        prediction = predict_label(word, is_first_word=(i == 0))
        raw = prompt(
            f"  Word {i+1}/{len(words)}: \"{word}\"  "
            f"[predicted: {prediction}]  Accept? (Enter) or type label: "
        )

        if raw == "":
            labels[word] = prediction
        else:
            override = raw.strip().upper()
            if override in VALID_LABELS:
                labels[word] = override
            else:
                print(f"  [WARN] '{override}' is not a valid label. Using prediction '{prediction}'.")
                labels[word] = prediction

    print()
    hr()
    print("  Labels assigned:")
    for w, lbl in labels.items():
        print(f"    {w:<28} {lbl}")
    hr()

    # ── Devanagari version (optional) ─────────────────────────────────────────
    print()
    print("  STEP 4 / 7 -- Devanagari version (optional, press Enter to skip)")
    devanagari = prompt("  Devanagari > ")

    # ── Topic ─────────────────────────────────────────────────────────────────
    print()
    print("  STEP 5 / 7 -- Topic")
    print("  Examples: Cell Membrane, Nucleus, Osmosis, Tissues, Life Processes")
    topic = prompt("  Topic > ")
    if not topic:
        topic = chapter_info["title"]

    # ── Student query? ────────────────────────────────────────────────────────
    print()
    print("  STEP 6 / 7 -- Entry type")
    is_student_query = False
    intent = None
    q_flag = prompt("  Is this a student query? (y/n) [default: n] > ", default="n").lower()
    if q_flag in ("y", "yes"):
        is_student_query = True
        print("  Intent options:")
        print("    1 / explain   -> explain_concept")
        print("    2 / compare   -> compare_concepts")
        print("    3 / example   -> give_example")
        print("    4 / formula   -> formula_request")
        print("    5 / definition -> definition")
        intent_raw = prompt("  Intent (1-5 or name) [default: 1] > ", default="1")
        intent = INTENT_MAP.get(intent_raw.lower(), INTENT_MAP.get(intent_raw, "explain_concept"))
        print(f"  Intent set to: {intent}")

    # ── GEC sample? ───────────────────────────────────────────────────────────
    is_gec_sample = False
    hinglish_with_error = None
    error_description = None

    gec_flag = prompt("  Is this a GEC sample? (y/n) [default: n] > ", default="n").lower()
    if gec_flag in ("y", "yes"):
        is_gec_sample = True
        print("  Paste the version WITH the grammar error:")
        hinglish_with_error = prompt("  Hinglish with error > ")
        print("  Error word (the wrong word):")
        error_word = prompt("    Error word > ")
        print("  Correct word (what it should be):")
        correct_word = prompt("    Correct word > ")
        print("  Error type (e.g. 'gender agreement', 'verb tense', 'subject-verb agreement'):")
        error_type = prompt("    Error type > ")
        print("  Brief explanation:")
        explanation = prompt("    Explanation > ")
        error_description = {
            "error_word"  : error_word,
            "correct_word": correct_word,
            "error_type"  : error_type,
            "explanation" : explanation,
        }

    # ── Code mixing type ──────────────────────────────────────────────────────
    code_mix_raw = prompt(
        "  Code mixing type: intra/inter [default: intra] > ", default="intra"
    ).lower()
    code_mix = "inter-sentential" if "inter" in code_mix_raw else "intra-sentential"

    # ── Notes ─────────────────────────────────────────────────────────────────
    print()
    print("  STEP 7 / 7 -- Notes (optional, press Enter to skip)")
    notes = prompt("  Notes > ")

    # ── Build entry dict ───────────────────────────────────────────────────────
    entry_id = next_entry_id(entries, chapter_info["id_prefix"], chapter_info["start_from"])

    entry = {
        "id"                : entry_id,
        "original_english"  : english,
        "hinglish_roman"    : hinglish,
        "word_level_labels" : labels,
        "topic"             : topic,
        "chapter"           : chapter_info["title"],
        "class"             : chapter_info["class"],
        "code_mixing_type"  : code_mix,
    }

    if devanagari:
        entry["hinglish_devanagari"] = devanagari

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

    # ── Save ──────────────────────────────────────────────────────────────────
    entries.append(entry)
    save_dataset(output_path, entries)

    total_now = len(entries)
    print()
    print("=" * 55)
    print(f"  [SAVED] Entry {entry_id}")
    print(f"  Total in file : {total_now}")
    print(f"  Target        : {target}")
    remaining = max(0, target - total_now)
    bar_done  = min(20, int(20 * total_now / target))
    bar       = "#" * bar_done + "." * (20 - bar_done)
    print(f"  Progress      : [{bar}] {total_now}/{target}")
    if remaining > 0:
        print(f"  Still needed  : {remaining} more entries")
    else:
        print(f"  TARGET REACHED! All {target} entries complete.")
    print("=" * 55)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EduHinglish Annotation Helper -- fast CLI for building Hinglish datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/annotation_helper.py --chapter class9/ch06
  python scripts/annotation_helper.py --chapter class10/ch06
  python scripts/annotation_helper.py --chapter class9/ch05   # IDs start at 9_05_021
  python scripts/annotation_helper.py --chapter class9/ch06 --target 30
        """,
    )
    parser.add_argument(
        "--chapter",
        type=str,
        required=True,
        help='Chapter key, e.g. class9/ch06 or class10/ch08',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output JSON file path (default: data/hinglish/<chapter_key>_dataset.json)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="Target number of entries for this chapter (default: 50)",
    )

    args = parser.parse_args()

    # ── Validate chapter ──────────────────────────────────────────────────────
    # Normalize: replace backslash, strip trailing slashes
    chapter_arg = args.chapter.replace("\\", "/").strip("/")

    if chapter_arg not in CHAPTER_TITLES:
        print(f"\n[ERROR] Unknown chapter: '{chapter_arg}'")
        print("  Valid chapter keys:")
        for k in CHAPTER_TITLES:
            print(f"    {k}")
        sys.exit(1)

    chapter_info = CHAPTER_TITLES[chapter_arg]

    # ── Resolve output path ───────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        # e.g. class9/ch06 -> class9_ch06_dataset.json
        file_key = chapter_arg.replace("/", "_")
        output_path = PROJECT_ROOT / "data" / "hinglish" / f"{file_key}_dataset.json"

    # ── Load existing data ────────────────────────────────────────────────────
    entries = load_dataset(output_path)

    # ── Load reference sentences ──────────────────────────────────────────────
    ref_sentences = load_reference_sentences(chapter_info["text_key"])

    # ── Show banner ───────────────────────────────────────────────────────────
    show_banner(chapter_info, output_path, args.target, len(entries))

    if not ref_sentences:
        print(f"\n  [NOTE] No cleaned text found for '{chapter_info['text_key']}'.")
        print(f"         Run batch_pdf_extractor.py first, or add a PDF for this chapter.")
        print(f"         You can still annotate without reference sentences.\n")

    print(f"\n  Annotation session started. Type 'q' at any prompt to quit.\n")

    # ── Main annotation loop ──────────────────────────────────────────────────
    while True:
        try:
            keep_going = annotate_one_entry(
                chapter_info=chapter_info,
                ref_sentences=ref_sentences,
                entries=entries,
                output_path=output_path,
                target=args.target,
            )

            if not keep_going:
                break

            print()
            next_flag = prompt("  Next entry? (Enter = yes, q = quit) > ", default="y").lower()
            if next_flag in ("q", "quit", "exit"):
                break

        except (KeyboardInterrupt, EOFError):
            print("\n\n  [EXIT] Session ended by user.")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  SESSION SUMMARY")
    print("=" * 55)
    print(f"  Chapter : {chapter_info['title']}")
    print(f"  Entries : {len(entries)} / {args.target}")
    print(f"  Saved to: {output_path}")
    print("=" * 55)
    print()


if __name__ == "__main__":
    main()
