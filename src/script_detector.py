"""
EduHinglish — Script Detection, Normalization & Language Identification
=======================================================================
Author  : Jatin
Module  : M1 — Input Processing (Steps 1.1, 1.2, 1.3)
Purpose : 
    1. ScriptDetector       — identify if input is Roman, Devanagari, or Mixed
    2. HinglishNormalizer   — correct spelling variants and abbreviations
    3. WordLevelLID         — tag every word as HI / EN / NE / UNIV / MIX

NOTE: This is a RULE-BASED prototype.
      Phase 2 will replace WordLevelLID with a fine-tuned MuRIL model.

Usage:
    python script_detector.py
"""

import re
from collections import Counter
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. SCRIPT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class ScriptDetector:
    """
    Detect whether input text is written in Roman script, Devanagari script,
    or a mixture of both — at both word and sentence level.

    Unicode ranges used:
        Devanagari : U+0900 – U+097F
        Latin      : U+0041 – U+005A (upper) | U+0061 – U+007A (lower)
    """

    DEVANAGARI_START = 0x0900
    DEVANAGARI_END   = 0x097F

    def _char_script(self, char: str) -> str:
        cp = ord(char)
        if self.DEVANAGARI_START <= cp <= self.DEVANAGARI_END:
            return "DEVANAGARI"
        if char.isascii() and char.isalpha():
            return "ROMAN"
        if char.isdigit():
            return "NUMERIC"
        if char.isspace():
            return "SPACE"
        return "OTHER"

    def word_script(self, word: str) -> str:
        """Return dominant script of a single word."""
        counts = Counter(self._char_script(c) for c in word
                         if self._char_script(c) not in ("SPACE", "OTHER", "NUMERIC"))
        if not counts:
            return "OTHER"
        has_dev = counts.get("DEVANAGARI", 0) > 0
        has_rom = counts.get("ROMAN", 0) > 0
        if has_dev and has_rom:
            return "MIXED"
        if has_dev:
            return "DEVANAGARI"
        if has_rom:
            return "ROMAN"
        return "OTHER"

    def detect(self, text: str) -> dict:
        """
        Detect overall script of a sentence/text.

        Returns:
            {
                "overall_script": "ROMAN" | "DEVANAGARI" | "MIXED",
                "word_scripts"  : { word: script, ... },
                "distribution"  : { roman_pct, devanagari_pct, mixed_pct }
            }
        """
        print("SCRIPT DETECTION")
        print(f"  Input : \"{text}\"")

        words         = text.split()
        word_scripts  = {}

        for w in words:
            clean = re.sub(r'[^\w]', '', w)
            if clean:
                word_scripts[w] = self.word_script(clean)

        counts    = Counter(word_scripts.values())
        total     = len(word_scripts) or 1
        roman_pct = counts.get("ROMAN", 0) / total * 100
        dev_pct   = counts.get("DEVANAGARI", 0) / total * 100
        mix_pct   = counts.get("MIXED", 0) / total * 100

        if dev_pct == 0 and mix_pct == 0:
            overall = "ROMAN"
        elif roman_pct == 0 and mix_pct == 0:
            overall = "DEVANAGARI"
        else:
            overall = "MIXED"

        # Pretty print
        print(f"\n  {'Word':<28} Script")
        print(f"  {'-'*28} ------")
        for w, s in word_scripts.items():
            print(f"  {w:<28} {s}")

        print(f"\n  Distribution:")
        print(f"    Roman      : {counts.get('ROMAN',0):>3} words  ({roman_pct:.1f}%)")
        print(f"    Devanagari : {counts.get('DEVANAGARI',0):>3} words  ({dev_pct:.1f}%)")
        print(f"    Mixed      : {counts.get('MIXED',0):>3} words  ({mix_pct:.1f}%)")
        print(f"\n  → Overall script : {overall}")

        return {
            "overall_script": overall,
            "word_scripts"  : word_scripts,
            "distribution"  : {
                "roman_pct"     : round(roman_pct, 1),
                "devanagari_pct": round(dev_pct, 1),
                "mixed_pct"     : round(mix_pct, 1),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. HINGLISH NORMALIZER
# ─────────────────────────────────────────────────────────────────────────────

class HinglishNormalizer:
    """
    Normalize Hinglish Roman-script text:
        1. Spelling variants  → standard form
        2. SMS/chat shorthand → full form
        3. Whitespace & punctuation cleanup

    The spelling map is the authoritative lookup; extend as needed.
    """

    # ── Canonical form → list of known variants ──────────────
    SPELLING_MAP = {
        "kya"     : ["kia", "kyaa", "keya"],
        "hai"     : ["he", "hae", "hay", "h"],
        "hain"    : ["hein", "haen", "hn"],
        "nahi"    : ["nhi", "nahin", "nai", "ni"],
        "mein"    : ["me", "mei", "mn"],
        "aur"     : ["or", "our", "ar", "aor"],
        "yeh"     : ["ye", "yah", "y"],
        "kaise"   : ["kese", "kse"],
        "kyun"    : ["kyu", "kyon", "q"],
        "samjhao" : ["smjhao", "samjao", "smjao"],
        "batao"   : ["btao", "btaao"],
        "karo"    : ["kro", "krdo"],
        "hota"    : ["hta"],
        "karta"   : ["krta"],
        "achha"   : ["acha", "accha", "achchha"],
        "bahut"   : ["bhut", "bahot", "bohot", "boht"],
        "lekin"   : ["lekn", "lkin"],
        "dijiye"  : ["dijie"],
        "sabhi"   : ["sbhi"],
        "wala"    : ["waala"],
        "wale"    : ["waale"],
    }

    # ── Chat abbreviations ────────────────────────────────────
    ABBREVIATIONS = {
        "pls" : "please",
        "plz" : "please",
        "thx" : "thanks",
        "bcz" : "because",
        "bcoz": "because",
        "diff": "difference",
        "hw"  : "how",
        "wht" : "what",
        "abt" : "about",
        "bio" : "biology",
        "chem": "chemistry",
        "phy" : "physics",
        "r"   : "are",
        "u"   : "you",
        "ur"  : "your",
        "d"   : "the",
    }

    def __init__(self):
        # Build reverse lookup: variant → canonical
        self._reverse: dict = {}
        for canonical, variants in self.SPELLING_MAP.items():
            for v in variants:
                self._reverse[v.lower()] = canonical
        print("[OK] HinglishNormalizer initialized")

    # ── Internal helpers ──────────────────────────────────────

    def _strip_punct(self, word: str) -> tuple:
        """Separate leading/trailing punctuation from word core."""
        prefix, suffix, core = "", "", word
        while core and not core[0].isalnum():
            prefix += core[0]; core = core[1:]
        while core and not core[-1].isalnum():
            suffix = core[-1] + suffix; core = core[:-1]
        return prefix, core, suffix

    # ── Public API ────────────────────────────────────────────

    def normalize_spelling(self, text: str) -> tuple:
        """
        Normalize spelling variants in a Hinglish sentence.
        Returns: (normalized_text, list_of_changes)
        """
        print("SPELLING NORMALIZATION")
        print(f"  Input: \"{text}\"")

        words, changes = text.split(), []
        out = []

        for word in words:
            pre, core, suf = self._strip_punct(word)
            lower = core.lower()
            if lower in self._reverse:
                norm = self._reverse[lower]
                changes.append(f"'{core}' → '{norm}'")
                out.append(pre + norm + suf)
            elif lower in self.ABBREVIATIONS:
                norm = self.ABBREVIATIONS[lower]
                changes.append(f"'{core}' → '{norm}' (abbrev)")
                out.append(pre + norm + suf)
            else:
                out.append(word)

        result = " ".join(out)
        if changes:
            print(f"  Changes made:")
            for c in changes:
                print(f"    {c}")
        else:
            print(f"  No changes needed")
        print(f"  Output: \"{result}\"")
        return result, changes

    def normalize_sentence(self, text: str) -> str:
        """
        Full pipeline normalization:
            1. Collapse whitespace
            2. Fix punctuation spacing
            3. Collapse repeated punctuation
            4. Spelling normalization
        """
        print("SENTENCE NORMALIZATION (complete)")
        print(f"  Input: \"{text}\"")

        # 1 – Whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # 2 – Punctuation spacing (remove space before . , ! ?)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # 3 – Repeated punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)

        # 4 – Spelling
        text, _ = self.normalize_spelling(text)

        print(f"  Final : \"{text}\"")
        return text


# ─────────────────────────────────────────────────────────────────────────────
# 3. WORD-LEVEL LANGUAGE IDENTIFICATION (LID)
# ─────────────────────────────────────────────────────────────────────────────

class WordLevelLID:
    """
    Rule-based word-level language identification for Hinglish.

    Labels:
        HI   — Hindi word in Roman script
        EN   — English word
        NE   — Named Entity (person, place, org)
        UNIV — Language-independent (numbers, universal greetings)
        MIX  — Mixed-script word (rare)
        PUNCT— Punctuation

    NOTE:
        This lexicon approach is the Phase 1 baseline.
        Phase 2 will replace this with a fine-tuned MuRIL token-classifier
        trained on LINCE & SAIL 2017 data for production-grade accuracy.
    """

    # ── Hindi words most commonly appearing in Hinglish Roman text ──
    HINDI_WORDS = {
        # Pronouns
        "main","hum","tum","woh","yeh","uska","uski","iska","iski",
        "mera","meri","tera","teri","unka","unki","hamara","tumhara",
        # Auxiliaries / verbs
        "hai","hain","tha","thi","the","hoga","hogi","hota","hoti","hote",
        "karta","karti","karte","kiya","karo","karna","karke","hona",
        "raha","rahi","rahe","gaya","gayi","aata","aati","jaata","jaati",
        "deta","deti","lete","bana","bante","samjhao","batao","dekho",
        "kehte","kehta","kehti","kaha","dijiye","hokar","jinmein","inmein",
        # Postpositions
        "ka","ki","ke","ko","se","mein","par","tak","pe","ne","me",
        # Conjunctions
        "aur","ya","lekin","kyunki","isliye","jabki","phir","toh","bhi",
        "hi","sirf","bas",
        # Question words
        "kya","kaise","kyun","kahan","kab","kaun","kitna","kitni","kitne",
        # Adjectives / adverbs
        "bahut","thoda","zyada","kam","achha","bura","bada","bade","badi",
        "chhota","naya","nayi","pehle","baad","andar","bahar","upar",
        "neeche","yahan","wahan","abhi","tab","jab",
        # Negation
        "nahi","nhi","na","mat",
        # Numbers (Hindi)
        "ek","do","teen","chaar","paanch",
        # Others
        "sabhi","sab","kuch","koi","wala","wale","wali","jaise","tarah",
        "matlab","yaani","saath","taraf","beech","kaam","cheez","jagah",
        "tarika",
    }

    # ── Science/biology English terms — always EN ──────────────
    SCIENCE_TERMS = {
        "cell","cells","membrane","membranes","nucleus","chromosome",
        "chromosomes","gene","genes","dna","rna","protein","proteins",
        "mitochondria","ribosome","ribosomes","lysosome","lysosomes",
        "endoplasmic","reticulum","golgi","apparatus","vacuole","vacuoles",
        "plastid","plastids","chloroplast","chloroplasts","chlorophyll",
        "osmosis","diffusion","permeable","selectively","prokaryotic",
        "eukaryotic","organism","organisms","tissue","tissues","organ",
        "organs","photosynthesis","respiration","atp","microscope","cork",
        "structural","functional","fundamental","nuclear","cytoplasm",
        "protoplasm","turgidity","rigidity","inheritance","offspring",
        "concentration","solution","energy","power","force","discovery",
        "discover","observe","package","dispatch","transport","suicide",
        "pigment","pigments","wall","plant","animal","living","life","unit",
        "function","structure","process","movement","transfer","control",
        "region","form","type","produce","provide","contain","contains",
        "present","absent","large","small","high","low","slice","bags",
        "powerhouse","observation",
    }

    # ── Universal / language-independent ────────────────────────
    UNIVERSAL_WORDS = {
        "sir","madam","ok","okay","hello","hi","bye",
        "please","thank","thanks","sorry","zero",
    }

    # ── English suffixes heuristic ──────────────────────────────
    EN_SUFFIXES = (
        "tion","sion","ness","ment","able","ible","ous","ive",
        "ful","less","ing","ed","er","est","ly","al","ity",
        "ence","ance","ism","ist","ize","ise",
    )

    def __init__(self):
        print("[OK] WordLevelLID initialized")

    def identify_word(self, word: str) -> str:
        """Identify language label of a single word."""
        clean = re.sub(r'[^\w]', '', word)
        if not clean:
            return "PUNCT"

        # Mixed-script word
        has_dev = any('\u0900' <= c <= '\u097F' for c in clean)
        has_rom = any(c.isascii() and c.isalpha() for c in clean)
        if has_dev and has_rom:
            return "MIX"
        if has_dev:
            return "HI"

        lower = clean.lower()

        if lower.isdigit():
            return "UNIV"
        if lower in self.UNIVERSAL_WORDS:
            return "UNIV"
        if lower in self.SCIENCE_TERMS:
            return "EN"
        if lower in self.HINDI_WORDS:
            return "HI"

        # Heuristic: English suffix
        if len(lower) > 4:
            for sfx in self.EN_SUFFIXES:
                if lower.endswith(sfx):
                    return "EN"

        # Default — lean English (NCERT text is English-heavy)
        return "EN"

    def identify_sentence(self, sentence: str) -> dict:
        """
        Tag every word in a Hinglish sentence and compute CMI.

        Returns:
            {
                "words": [ {"word": ..., "language": ...}, ... ],
                "cmi"  : float,
                "language_distribution": { "HI": n, "EN": n, ... }
            }
        """
        print("WORD-LEVEL LANGUAGE IDENTIFICATION (LID)")
        print(f"  Input: \"{sentence}\"\n")

        ICONS = {"HI": "🟠", "EN": "🔵", "NE": "🟢",
                 "UNIV": "⚪", "MIX": "🟡", "PUNCT": "⚫"}

        words   = sentence.split()
        results = []

        print(f"  {'Word':<28} Lang   Icon")
        print(f"  {'-'*28} -----  ----")

        for w in words:
            lang = self.identify_word(w)
            results.append({"word": w, "language": lang})
            print(f"  {w:<28} {lang:<6} {ICONS.get(lang,'?')}")

        # ── Code-Mixing Index (CMI) ──────────────────────────
        counts   = Counter(r["language"] for r in results)
        hi       = counts.get("HI", 0)
        en       = counts.get("EN", 0)
        denom    = hi + en
        cmi      = round((1 - max(hi, en) / denom) * 100, 1) if denom > 0 else 0.0

        print(f"\n  Language Distribution:")
        print(f"    Hindi (HI)  : {hi:>3} words")
        print(f"    English (EN): {en:>3} words")
        print(f"    Universal   : {counts.get('UNIV',0):>3} words")
        print(f"    Named Entity: {counts.get('NE',0):>3} words")
        print(f"    Mixed script: {counts.get('MIX',0):>3} words")
        print(f"\n  Code-Mixing Index (CMI) : {cmi}%")

        if cmi == 0:
            label = "monolingual"
        elif cmi < 30:
            label = "low code-mixing"
        elif cmi < 50:
            label = "moderate code-mixing (natural Hinglish)"
        else:
            label = "high code-mixing"
        print(f"  Interpretation          : {label}")

        return {
            "words"                : results,
            "cmi"                  : cmi,
            "language_distribution": dict(counts),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Direct run — demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    detector   = ScriptDetector()
    normalizer = HinglishNormalizer()
    lid        = WordLevelLID()

    # ── Script Detection ─────────────────────────────────────
    print("# DEMO 1 — SCRIPT DETECTION")

    script_tests = [
        "All living organisms are made up of cells.",                          # Pure English
        "Sabhi living organisms cells se bane hote hain.",                     # Roman Hinglish
        "सभी living organisms cells से बने होते हैं।",                       # Devanagari mixed
        "Mitochondria को cell ka powerhouse kaha jaata hai.",                  # Mixed script
    ]
    for t in script_tests:
        detector.detect(t)

    # ── Normalization ─────────────────────────────────────────
    print("# DEMO 2 — NORMALIZATION")

    norm_tests = [
        "Sir osmosis kia hota h? plz smjhao",
        "cell membrane ki function btao",
        "mitochondria ko powerhouse kyu kehte he?",
    ]
    for t in norm_tests:
        normalizer.normalize_sentence(t)

    # ── Language Identification ───────────────────────────────
    print("# DEMO 3 — WORD-LEVEL LANGUAGE IDENTIFICATION")

    lid_tests = [
        "All living organisms are made up of cells.",
        "Sabhi living organisms cells se bane hote hain.",
        "Sir, endoplasmic reticulum ka cell mein kya function hota hai?",
        "Mitochondria ko cell ka powerhouse kaha jaata hai kyunki yeh ATP ke form mein energy produce karte hain.",
    ]
    for t in lid_tests:
        lid.identify_sentence(t)