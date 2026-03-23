"""
EduHinglish — Hinglish Labeled Dataset Creator
===============================================
Author  : Jatin
Module  : M1 — Dataset Creation (Gap 4 of 7)
Purpose : Create a hand-crafted, labeled dataset of Hinglish sentences
          sourced from NCERT Class 9 Biology Chapter 5:
          "The Fundamental Unit of Life"

This dataset serves THREE purposes:
    1. Test input for the preprocessing pipeline
    2. Reference output for pipeline validation (word_level_labels)
    3. Seed data for future fine-tuning / evaluation

Each entry contains:
    - original_english     — the source NCERT English sentence
    - hinglish_roman       — natural Hinglish in Roman script
    - hinglish_devanagari  — same in Devanagari (partial)
    - word_level_labels    — manual HI/EN/NE/UNIV/MIX tags per word
    - topic, chapter, class
    - flags: is_student_query, is_gec_sample, intent

Usage:
    python hinglish_dataset_creator.py
"""

import json
from pathlib import Path
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

# Complete Biology Chapter 5 dataset — 20 entries
DATASET = [

    # ── BASIC CELL THEORY ────────────────────────────────────────────────────

    {
        "id": 1,
        "original_english": "All living organisms are made up of cells.",
        "hinglish_roman": "Sabhi living organisms cells se bane hote hain.",
        "hinglish_devanagari": "सभी living organisms cells से बने होते हैं।",
        "word_level_labels": {
            "Sabhi": "HI", "living": "EN", "organisms": "EN",
            "cells": "EN", "se": "HI", "bane": "HI",
            "hote": "HI", "hain": "HI",
        },
        "topic": "Introduction to Cell",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "inter-sentential",
        "notes": "Basic cell theory; Hindi provides grammatical structure",
    },

    {
        "id": 2,
        "original_english": "The cell is the fundamental structural and functional unit of life.",
        "hinglish_roman": "Cell life ki fundamental structural aur functional unit hai.",
        "hinglish_devanagari": "Cell life की fundamental structural और functional unit है।",
        "word_level_labels": {
            "Cell": "EN", "life": "EN", "ki": "HI", "fundamental": "EN",
            "structural": "EN", "aur": "HI", "functional": "EN",
            "unit": "EN", "hai": "HI",
        },
        "topic": "Cell Theory",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "notes": "Most words are English technical terms; Hindi = grammatical glue",
    },

    # ── DISCOVERY ────────────────────────────────────────────────────────────

    {
        "id": 3,
        "original_english": "Robert Hooke discovered cells in 1665 by observing a thin slice of cork under a microscope.",
        "hinglish_roman": "Robert Hooke ne 1665 mein cork ki thin slice ko microscope se observe karke cells discover kiya tha.",
        "hinglish_devanagari": "Robert Hooke ने 1665 में cork की thin slice को microscope से observe करके cells discover किया था।",
        "word_level_labels": {
            "Robert": "NE", "Hooke": "NE", "ne": "HI", "1665": "UNIV",
            "mein": "HI", "cork": "EN", "ki": "HI", "thin": "EN",
            "slice": "EN", "ko": "HI", "microscope": "EN", "se": "HI",
            "observe": "EN", "karke": "HI", "cells": "EN",
            "discover": "EN", "kiya": "HI", "tha": "HI",
        },
        "topic": "Discovery of Cell",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "notes": "Contains Named Entity (Robert Hooke) and numeral",
    },

    # ── CELL MEMBRANE ────────────────────────────────────────────────────────

    {
        "id": 4,
        "original_english": "The cell membrane is a selectively permeable membrane that controls the movement of substances into and out of the cell.",
        "hinglish_roman": "Cell membrane ek selectively permeable membrane hoti hai jo substances ke movement ko cell ke andar aur bahar control karti hai.",
        "hinglish_devanagari": "Cell membrane एक selectively permeable membrane होती है जो substances के movement को cell के अंदर और बाहर control करती है।",
        "word_level_labels": {
            "Cell": "EN", "membrane": "EN", "ek": "HI", "selectively": "EN",
            "permeable": "EN", "hoti": "HI", "hai": "HI", "jo": "HI",
            "substances": "EN", "ke": "HI", "movement": "EN", "ko": "HI",
            "andar": "HI", "aur": "HI", "bahar": "HI", "control": "EN",
            "karti": "HI",
        },
        "topic": "Cell Membrane / Plasma Membrane",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "notes": "Dense code-mixing; spatial Hindi words (andar/bahar)",
    },

    # ── NUCLEUS ──────────────────────────────────────────────────────────────

    {
        "id": 5,
        "original_english": "The nucleus contains chromosomes which carry genes and help in inheritance.",
        "hinglish_roman": "Nucleus mein chromosomes hote hain jinmein genes hote hain aur yeh inheritance mein help karte hain.",
        "hinglish_devanagari": "Nucleus में chromosomes होते हैं जिनमें genes होते हैं और यह inheritance में help करते हैं।",
        "word_level_labels": {
            "Nucleus": "EN", "mein": "HI", "chromosomes": "EN", "hote": "HI",
            "hain": "HI", "jinmein": "HI", "genes": "EN", "aur": "HI",
            "yeh": "HI", "inheritance": "EN", "help": "EN", "karte": "HI",
        },
        "topic": "Nucleus",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "notes": "Biology-heavy vocabulary",
    },

    # ── OSMOSIS ──────────────────────────────────────────────────────────────

    {
        "id": 6,
        "original_english": "Osmosis is the movement of water molecules through a selectively permeable membrane from a region of high water concentration to a region of low water concentration.",
        "hinglish_roman": "Osmosis mein water molecules selectively permeable membrane se hokar high concentration wale region se low concentration wale region ki taraf move karte hain.",
        "hinglish_devanagari": "Osmosis में water molecules selectively permeable membrane से होकर high concentration वाले region से low concentration वाले region की तरफ move करते हैं।",
        "word_level_labels": {
            "Osmosis": "EN", "mein": "HI", "water": "EN", "molecules": "EN",
            "selectively": "EN", "permeable": "EN", "membrane": "EN",
            "se": "HI", "hokar": "HI", "high": "EN", "concentration": "EN",
            "wale": "HI", "region": "EN", "low": "EN", "ki": "HI",
            "taraf": "HI", "move": "EN", "karte": "HI", "hain": "HI",
        },
        "topic": "Osmosis",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "notes": "Scientific process definition in Hinglish",
    },

    # ── MITOCHONDRIA ─────────────────────────────────────────────────────────

    {
        "id": 7,
        "original_english": "Mitochondria are known as the powerhouse of the cell because they produce energy in the form of ATP.",
        "hinglish_roman": "Mitochondria ko cell ka powerhouse kaha jaata hai kyunki yeh ATP ke form mein energy produce karte hain.",
        "hinglish_devanagari": "Mitochondria को cell का powerhouse कहा जाता है क्योंकि यह ATP के form में energy produce करते हैं।",
        "word_level_labels": {
            "Mitochondria": "EN", "ko": "HI", "cell": "EN", "ka": "HI",
            "powerhouse": "EN", "kaha": "HI", "jaata": "HI", "hai": "HI",
            "kyunki": "HI", "yeh": "HI", "ATP": "EN", "ke": "HI",
            "form": "EN", "mein": "HI", "energy": "EN", "produce": "EN",
            "karte": "HI", "hain": "HI",
        },
        "topic": "Mitochondria",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "notes": "Most commonly asked biology fact",
    },

    # ── CELL WALL ────────────────────────────────────────────────────────────

    {
        "id": 8,
        "original_english": "The cell wall is present in plant cells but not in animal cells.",
        "hinglish_roman": "Cell wall plant cells mein hoti hai lekin animal cells mein nahi hoti.",
        "hinglish_devanagari": "Cell wall plant cells में होती है लेकिन animal cells में नहीं होती।",
        "word_level_labels": {
            "Cell": "EN", "wall": "EN", "plant": "EN", "cells": "EN",
            "mein": "HI", "hoti": "HI", "hai": "HI", "lekin": "HI",
            "animal": "EN", "nahi": "HI",
        },
        "topic": "Cell Wall",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "notes": "Comparison fact; negation (nahi) preserved",
    },

    # ── PROKARYOTIC vs EUKARYOTIC ─────────────────────────────────────────────

    {
        "id": 9,
        "original_english": "Prokaryotic cells do not have a well-defined nucleus, while eukaryotic cells have a well-organised nucleus with a nuclear membrane.",
        "hinglish_roman": "Prokaryotic cells mein well-defined nucleus nahi hota, jabki eukaryotic cells mein nuclear membrane ke saath ek well-organised nucleus hota hai.",
        "hinglish_devanagari": "Prokaryotic cells में well-defined nucleus नहीं होता, जबकि eukaryotic cells में nuclear membrane के साथ एक well-organised nucleus होता है।",
        "word_level_labels": {
            "Prokaryotic": "EN", "cells": "EN", "mein": "HI", "well-defined": "EN",
            "nucleus": "EN", "nahi": "HI", "hota": "HI", "jabki": "HI",
            "eukaryotic": "EN", "nuclear": "EN", "membrane": "EN", "ke": "HI",
            "saath": "HI", "ek": "HI", "well-organised": "EN", "hai": "HI",
        },
        "topic": "Prokaryotic and Eukaryotic Cells",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "notes": "Contrast sentence using jabki (while/whereas)",
    },

    # ── PLASTIDS ─────────────────────────────────────────────────────────────

    {
        "id": 10,
        "original_english": "Plastids are present only in plant cells and they contain pigments like chlorophyll.",
        "hinglish_roman": "Plastids sirf plant cells mein hote hain aur inmein chlorophyll jaise pigments hote hain.",
        "hinglish_devanagari": "Plastids सिर्फ plant cells में होते हैं और इनमें chlorophyll जैसे pigments होते हैं।",
        "word_level_labels": {
            "Plastids": "EN", "sirf": "HI", "plant": "EN", "cells": "EN",
            "mein": "HI", "hote": "HI", "hain": "HI", "aur": "HI",
            "inmein": "HI", "chlorophyll": "EN", "jaise": "HI", "pigments": "EN",
        },
        "topic": "Plastids / Chloroplasts",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "notes": "jaise = 'like/such as' — common Hinglish connector",
    },

    # ── GOLGI APPARATUS ──────────────────────────────────────────────────────

    {
        "id": 11,
        "original_english": "The Golgi apparatus packages and dispatches materials within the cell.",
        "hinglish_roman": "Golgi apparatus cell ke andar materials ko package aur dispatch karne ka kaam karta hai.",
        "hinglish_devanagari": "Golgi apparatus cell के अंदर materials को package और dispatch करने का काम करता है।",
        "word_level_labels": {
            "Golgi": "EN", "apparatus": "EN", "cell": "EN", "ke": "HI",
            "andar": "HI", "materials": "EN", "ko": "HI", "package": "EN",
            "aur": "HI", "dispatch": "EN", "karne": "HI", "ka": "HI",
            "kaam": "HI", "karta": "HI", "hai": "HI",
        },
        "topic": "Golgi Apparatus",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "notes": "Function description; kaam = work",
    },

    # ── VACUOLES ─────────────────────────────────────────────────────────────

    {
        "id": 12,
        "original_english": "Vacuoles in plant cells are large and provide turgidity and rigidity to the cell.",
        "hinglish_roman": "Plant cells mein vacuoles bade hote hain aur yeh cell ko turgidity aur rigidity provide karte hain.",
        "hinglish_devanagari": "Plant cells में vacuoles बड़े होते हैं और यह cell को turgidity और rigidity provide करते हैं।",
        "word_level_labels": {
            "Plant": "EN", "cells": "EN", "mein": "HI", "vacuoles": "EN",
            "bade": "HI", "hote": "HI", "hain": "HI", "aur": "HI",
            "yeh": "HI", "cell": "EN", "ko": "HI", "turgidity": "EN",
            "rigidity": "EN", "provide": "EN", "karte": "HI",
        },
        "topic": "Vacuoles",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "notes": "bade = large — adjective code-switch",
    },

    # ── STUDENT QUERIES (Question style) ─────────────────────────────────────

    {
        "id": 13,
        "original_english": "What is the function of the endoplasmic reticulum in the cell?",
        "hinglish_roman": "Sir, endoplasmic reticulum ka cell mein kya function hota hai?",
        "hinglish_devanagari": "Sir, endoplasmic reticulum का cell में क्या function होता है?",
        "word_level_labels": {
            "Sir": "UNIV", "endoplasmic": "EN", "reticulum": "EN",
            "ka": "HI", "cell": "EN", "mein": "HI", "kya": "HI",
            "function": "EN", "hota": "HI", "hai": "HI",
        },
        "topic": "Endoplasmic Reticulum",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "is_student_query": True,
        "intent": "explain_concept",
        "notes": "Natural student doubt — primary input type for M1",
    },

    {
        "id": 14,
        "original_english": "Why are lysosomes called suicide bags?",
        "hinglish_roman": "Lysosomes ko suicide bags kyun kehte hain?",
        "hinglish_devanagari": "Lysosomes को suicide bags क्यों कहते हैं?",
        "word_level_labels": {
            "Lysosomes": "EN", "ko": "HI", "suicide": "EN",
            "bags": "EN", "kyun": "HI", "kehte": "HI", "hain": "HI",
        },
        "topic": "Lysosomes",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "is_student_query": True,
        "intent": "explain_concept",
        "notes": "Short, direct student question — very common doubt",
    },

    {
        "id": 15,
        "original_english": "Can you explain the difference between diffusion and osmosis?",
        "hinglish_roman": "Sir diffusion aur osmosis mein kya difference hai, samjha dijiye?",
        "hinglish_devanagari": "Sir diffusion और osmosis में क्या difference है, समझा दीजिये?",
        "word_level_labels": {
            "Sir": "UNIV", "diffusion": "EN", "aur": "HI", "osmosis": "EN",
            "mein": "HI", "kya": "HI", "difference": "EN", "hai": "HI",
            "samjha": "HI", "dijiye": "HI",
        },
        "topic": "Diffusion and Osmosis",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "is_student_query": True,
        "intent": "compare_concepts",
        "notes": "Comparison query type",
    },

    {
        "id": 16,
        "original_english": "What is the formula for calculating osmotic pressure?",
        "hinglish_roman": "Osmotic pressure ka formula kya hai batao?",
        "hinglish_devanagari": "Osmotic pressure का formula क्या है बताओ?",
        "word_level_labels": {
            "Osmotic": "EN", "pressure": "EN", "ka": "HI",
            "formula": "EN", "kya": "HI", "hai": "HI", "batao": "HI",
        },
        "topic": "Osmotic Pressure",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "is_student_query": True,
        "intent": "formula_request",
        "notes": "Formula-request intent",
    },

    {
        "id": 17,
        "original_english": "Give an example of osmosis in daily life.",
        "hinglish_roman": "Osmosis ka ek daily life example do?",
        "hinglish_devanagari": "Osmosis का एक daily life example दो?",
        "word_level_labels": {
            "Osmosis": "EN", "ka": "HI", "ek": "HI", "daily": "EN",
            "life": "EN", "example": "EN", "do": "HI",
        },
        "topic": "Osmosis",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "is_student_query": True,
        "intent": "give_example",
        "notes": "Example-request intent",
    },

    # ── GEC SAMPLES (for Module 4 testing) ───────────────────────────────────

    {
        "id": 18,
        "original_english": "The cell wall is present in plant cells but not in animal cells.",
        "hinglish_roman": "Cell wall plant cells mein hoti hai lekin animal cells mein nahi hoti.",
        "hinglish_with_error": "Cell wall plant cells mein hota hai lekin animal cells mein nahi hote.",
        "error_description": {
            "error_word"  : "hote",
            "correct_word": "hoti",
            "error_type"  : "gender agreement",
            "explanation" : "'cell wall' is grammatically feminine in Hinglish context → 'hoti', not 'hote'",
        },
        "word_level_labels": {
            "Cell": "EN", "wall": "EN", "plant": "EN", "cells": "EN",
            "mein": "HI", "hoti": "HI", "hai": "HI", "lekin": "HI",
            "animal": "EN", "nahi": "HI",
        },
        "topic": "Cell Wall",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "is_gec_sample": True,
        "notes": "GEC test — gender agreement error in Hindi auxiliary",
    },

    {
        "id": 19,
        "original_english": "The ball was moving fast because of gravity.",
        "hinglish_roman": "Ball bahut fast move kar rahi thi gravity ki wajah se.",
        "hinglish_with_error": "Ball bahut fast move karna tha gravity ki wajah se.",
        "error_description": {
            "error_phrase" : "move karna tha",
            "correct_phrase": "move kar rahi thi",
            "error_type"   : "verb tense (past continuous)",
            "explanation"  : "Ongoing past action needs past continuous 'kar rahi thi', not infinitive 'karna tha'",
        },
        "word_level_labels": {
            "Ball": "EN", "bahut": "HI", "fast": "EN", "move": "EN",
            "kar": "HI", "rahi": "HI", "thi": "HI", "gravity": "EN",
            "ki": "HI", "wajah": "HI", "se": "HI",
        },
        "topic": "Force and Motion (Physics cross-topic)",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "is_gec_sample": True,
        "notes": "GEC test — verb tense error in Hindi segment",
    },

    {
        "id": 20,
        "original_english": "She doesn't know the answer and she is confused.",
        "hinglish_roman": "Woh answer nahi jaanti aur woh confused hai.",
        "hinglish_with_error": "Woh answer nahi jaanta aur woh confused hai.",
        "error_description": {
            "error_word"  : "jaanta",
            "correct_word": "jaanti",
            "error_type"  : "gender agreement (feminine subject)",
            "explanation" : "Subject 'woh' refers to a female → verb must be 'jaanti', not 'jaanta'",
        },
        "word_level_labels": {
            "Woh": "HI", "answer": "EN", "nahi": "HI", "jaanti": "HI",
            "aur": "HI", "confused": "EN", "hai": "HI",
        },
        "topic": "GEC Demo Sentence",
        "chapter": "Chapter 5: The Fundamental Unit of Life",
        "class": "9",
        "code_mixing_type": "intra-sentential",
        "is_gec_sample": True,
        "notes": "GEC test — gender agreement; code-switch preserved",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CREATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class HinglishDatasetCreator:
    """Manage, validate, and export the Hinglish labeled dataset."""

    def __init__(self):
        self.dataset = DATASET
        print(f"[OK] Dataset loaded — {len(self.dataset)} entries")

    # ── Validation ────────────────────────────────────────────

    def validate(self) -> list:
        """Check all entries have required fields. Returns list of issues."""
        required = {"id", "original_english", "hinglish_roman", "word_level_labels", "topic"}
        issues   = []
        ids      = set()

        for entry in self.dataset:
            missing = required - entry.keys()
            if missing:
                issues.append(f"  Entry {entry.get('id','?')}: missing fields {missing}")
            if entry.get("id") in ids:
                issues.append(f"  Duplicate id: {entry['id']}")
            ids.add(entry.get("id"))

        if issues:
            print(f"\n[WARN] Validation issues:")
            for issue in issues:
                print(issue)
        else:
            print("[OK] All entries valid")

        return issues

    # ── Statistics ────────────────────────────────────────────

    def statistics(self) -> dict:
        """Compute and print dataset statistics."""
        total      = len(self.dataset)
        queries    = sum(1 for d in self.dataset if d.get("is_student_query", False))
        gec        = sum(1 for d in self.dataset if d.get("is_gec_sample", False))
        statements = total - queries - gec

        # Per-intent count
        intents = Counter(
            d.get("intent", "statement")
            for d in self.dataset
            if d.get("is_student_query", False)
        )

        # Average CMI
        cmis = []
        for entry in self.dataset:
            labels = entry.get("word_level_labels", {})
            hi  = sum(1 for v in labels.values() if v == "HI")
            en  = sum(1 for v in labels.values() if v == "EN")
            tot = hi + en
            if tot > 0:
                cmis.append((1 - max(hi, en) / tot) * 100)

        topics = sorted(set(d["topic"] for d in self.dataset))

        stats = {
            "total"                : total,
            "statements"           : statements,
            "student_queries"      : queries,
            "gec_samples"          : gec,
            "average_cmi"          : round(sum(cmis) / len(cmis), 2) if cmis else 0,
            "intent_distribution"  : dict(intents),
            "topics_covered"       : topics,
            "num_topics"           : len(topics),
        }

        print(f"\n{'='*55}")
        print("DATASET STATISTICS")
        print(f"{'='*55}")
        for k, v in stats.items():
            if isinstance(v, list):
                print(f"  {k}:")
                for item in v:
                    print(f"    • {item}")
            else:
                print(f"  {k:<30}: {v}")
        print(f"{'='*55}")

        return stats

    # ── Filters ───────────────────────────────────────────────

    def get_queries(self) -> list:
        """Return only student query entries."""
        return [d for d in self.dataset if d.get("is_student_query", False)]

    def get_gec_samples(self) -> list:
        """Return only GEC test entries."""
        return [d for d in self.dataset if d.get("is_gec_sample", False)]

    def get_by_topic(self, topic: str) -> list:
        """Return entries matching a topic keyword."""
        kw = topic.lower()
        return [d for d in self.dataset if kw in d["topic"].lower()]

    # ── Export ────────────────────────────────────────────────

    def save(self, output_path: str):
        """Save full dataset as pretty-printed JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Dataset → {output_path}  ({len(self.dataset)} entries)")

    def save_queries_only(self, output_path: str):
        """Save only student query entries."""
        q = self.get_queries()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(q, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Queries → {output_path}  ({len(q)} entries)")

    # ── Preview ───────────────────────────────────────────────

    def preview(self, n: int = 3):
        """Print the first n entries as a quick check."""
        print(f"\n{'='*70}")
        print(f"DATASET PREVIEW — first {n} entries")
        print(f"{'='*70}")
        for entry in self.dataset[:n]:
            print(f"\n  ID       : {entry['id']}")
            print(f"  English  : {entry['original_english']}")
            print(f"  Hinglish : {entry['hinglish_roman']}")
            print(f"  Labels   : {entry['word_level_labels']}")
            print(f"  Topic    : {entry['topic']}")
            if entry.get("is_student_query"):
                print(f"  Intent   : {entry.get('intent','—')}")
            if entry.get("is_gec_sample"):
                print(f"  [GEC]    : {entry.get('error_description','—')}")


# ─────────────────────────────────────────────────────────────────────────────
# Direct run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    creator = HinglishDatasetCreator()

    # Validate
    creator.validate()

    # Stats
    creator.statistics()

    # Preview
    creator.preview(n=3)

    # Save
    base = Path(__file__).parent.parent / "data" / "hinglish"
    creator.save(str(base / "hinglish_bio_ch5.json"))
    creator.save_queries_only(str(base / "student_queries.json"))

    print("\n[DONE] Dataset ready for pipeline testing.")