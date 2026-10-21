import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIVERSAL_WORDS = {"sir", "madam", "ok", "okay", "hello", "hi", "bye", "please", "thanks", "sorry"}
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

def main():
    datasets = [
        "data/bio_9_10.json",
        "data/ss9.json",
        "data/ss10.json"
    ]

    all_entries = []

    for path_str in datasets:
        path = PROJECT_ROOT / path_str
        if not path.exists():
            print(f"File not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded {len(data)} entries from {path_str}")

        for entry in data:
            if "hinglish_roman" not in entry:
                hinglish = entry.get("question_hinglish", "")
                if hinglish:
                    entry["hinglish_roman"] = hinglish
            
            if "word_level_labels" not in entry and "hinglish_roman" in entry:
                entry["word_level_labels"] = label_sentence(entry["hinglish_roman"])
            
            all_entries.append(entry)

    # Sort all by ID alphabetically
    all_entries.sort(key=lambda x: x.get("id", ""))

    out_path = PROJECT_ROOT / "data/unified_master_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_entries)} merged and enriched entries to {out_path.name}")

if __name__ == "__main__":
    main()
