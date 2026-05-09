"""
EduHinglish — Phase 3, Task 1: Prepare spaCy Training Data
============================================================
Converts data/unified_biology_dataset.json into .spacy binary
files for three training tasks:

  1. LID  — token-level language identification (HI/EN/NE/UNIV/MIX)
  2. NER  — science named-entity recognition
  3. Intent — student-query intent classification

Usage:
  python training/prepare_spacy_data.py                   # all three
  python training/prepare_spacy_data.py --task lid         # LID only
  python training/prepare_spacy_data.py --task ner
  python training/prepare_spacy_data.py --task intent
  python training/prepare_spacy_data.py --dataset PATH     # custom dataset
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATASET = Path("data/unified_biology_dataset.json")
OUTPUT_DIR      = Path("training/data")

LID_LABELS = ["HI", "EN", "NE", "UNIV", "MIX"]

INTENT_LABELS = [
    "explain_concept",
    "compare_concepts",
    "give_example",
    "formula_request",
    "definition",
]

# ── NER entity lists ─────────────────────────────────────────────────────────

SCIENTISTS = {
    "mendel", "darwin", "hooke", "lamarck", "linnaeus",
    "watson", "crick", "pasteur", "fleming", "lister",
    "robert hooke", "gregor mendel", "charles darwin",
    "louis pasteur", "alexander fleming",
}

ORGANELLES = {
    "nucleus", "mitochondria", "mitochondrion", "ribosome", "ribosomes",
    "chloroplast", "chloroplasts", "vacuole", "vacuoles",
    "golgi", "golgi body", "golgi apparatus",
    "endoplasmic reticulum", "smooth endoplasmic reticulum",
    "rough endoplasmic reticulum", "SER", "RER",
    "centrosome", "lysosome", "lysosomes",
    "cell membrane", "cell wall", "plasma membrane",
    "nuclear membrane", "cytoplasm", "protoplasm",
}

PROCESSES = {
    "photosynthesis", "respiration", "digestion", "excretion",
    "reproduction", "transpiration", "fertilisation", "fertilization",
    "pollination", "osmosis", "diffusion", "plasmolysis",
    "anaerobic respiration", "aerobic respiration",
    "cell division", "mitosis", "meiosis",
    "dna replication", "protein synthesis",
    "nitrogen fixation", "decomposition",
    "budding", "fragmentation", "regeneration",
    "binary fission", "vegetative propagation",
    "natural selection", "speciation", "evolution",
}

CONCEPTS = {
    "food chain", "food web", "nitrogen cycle", "carbon cycle",
    "water cycle", "oxygen cycle", "biogeochemical cycle",
    "blood group", "gene expression", "genetic drift",
    "greenhouse effect", "ozone layer", "ozone depletion",
    "trophic level", "ecological pyramid", "biodiversity",
    "ecosystem", "biosphere", "habitat",
    "dominant trait", "recessive trait", "phenotype", "genotype",
    "homologous organs", "analogous organs", "vestigial organs",
    "fossil", "fossils",
}

INSTRUMENTS = {
    "microscope", "stethoscope", "thermometer",
    "test tube", "petri dish",
}

NER_CATEGORIES = {
    "SCIENTIST":  SCIENTISTS,
    "ORGANELLE":  ORGANELLES,
    "PROCESS":    PROCESSES,
    "CONCEPT":    CONCEPTS,
    "INSTRUMENT": INSTRUMENTS,
}

SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _divider():
    print("  " + "─" * 50)

def _thick():
    print("  " + "═" * 50)


def load_dataset(path: Path) -> list[dict]:
    """Load the unified JSON dataset."""
    if not path.exists():
        print(f"  [ERROR] Dataset not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entries from {path}")
    return data


def strip_punct(word: str) -> str:
    """Strip sentence-boundary punctuation from a token (same as annotation_helper)."""
    return word.strip(".,?!;:\"'()")


def tokenise_roman(text: str) -> list[str]:
    """
    Whitespace-split and strip punctuation — mirrors annotation_helper.tokenise_hinglish.
    Returns cleaned tokens (no empty strings).
    """
    return [t for t in (strip_punct(w) for w in text.split()) if t]


def split_train_dev(items: list, dev_ratio: float = 0.2):
    """Shuffle and split into (train, dev)."""
    random.seed(SEED)
    shuffled = list(items)
    random.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * (1 - dev_ratio)))
    return shuffled[:cut], shuffled[cut:]


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1a — LID DATA
# ─────────────────────────────────────────────────────────────────────────────

def prepare_lid(dataset: list[dict], nlp: spacy.Language) -> None:
    """
    Convert word_level_labels into token-level tagged Docs.
    Uses spaCy's 'tag' attribute to store HI/EN/NE/UNIV/MIX.
    """
    print()
    _thick()
    print("  Preparing LID (token classification) data")
    _thick()

    docs: list[Doc] = []
    skipped = 0
    label_counter: Counter = Counter()

    for entry in dataset:
        hinglish = entry.get("hinglish_roman", "")
        labels_dict = entry.get("word_level_labels", {})
        if not hinglish or not labels_dict:
            skipped += 1
            continue

        tokens = tokenise_roman(hinglish)
        # Build label list aligned with tokens
        tags: list[str] = []
        valid = True
        for tok in tokens:
            # Try exact match first, then case-insensitive
            lbl = labels_dict.get(tok) or labels_dict.get(tok.lower()) or labels_dict.get(tok.capitalize())
            if lbl is None:
                # Fuzzy: check if any key matches after stripping punct
                for k, v in labels_dict.items():
                    if strip_punct(k).lower() == tok.lower():
                        lbl = v
                        break
            if lbl is None:
                lbl = "EN"  # default fallback
            if lbl not in LID_LABELS:
                lbl = "EN"
            tags.append(lbl)
            label_counter[lbl] += 1

        # Create spaCy Doc with custom tokenization
        doc = Doc(nlp.vocab, words=tokens)
        for i, tag in enumerate(tags):
            doc[i].tag_ = tag
        docs.append(doc)

    # Split
    train_docs, dev_docs = split_train_dev(docs)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_db = DocBin(attrs=["TAG"], store_user_data=False)
    for d in train_docs:
        train_db.add(d)
    train_db.to_disk(OUTPUT_DIR / "lid_train.spacy")

    dev_db = DocBin(attrs=["TAG"], store_user_data=False)
    for d in dev_docs:
        dev_db.add(d)
    dev_db.to_disk(OUTPUT_DIR / "lid_dev.spacy")

    # Stats
    total_tokens = sum(label_counter.values())
    print(f"\n  Docs created   : {len(docs)}")
    print(f"  Docs skipped   : {skipped}")
    print(f"  Total tokens   : {total_tokens}")
    print(f"  Train docs     : {len(train_docs)}")
    print(f"  Dev docs       : {len(dev_docs)}")
    print()
    print("  Label distribution:")
    for lbl in LID_LABELS:
        cnt = label_counter[lbl]
        pct = cnt * 100 // max(total_tokens, 1)
        print(f"    {lbl:6s} {cnt:6d} ({pct:3d}%)")
    print()
    print(f"  ✓ Saved: {OUTPUT_DIR / 'lid_train.spacy'}")
    print(f"  ✓ Saved: {OUTPUT_DIR / 'lid_dev.spacy'}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1b — NER DATA
# ─────────────────────────────────────────────────────────────────────────────

def prepare_ner(dataset: list[dict], nlp: spacy.Language) -> None:
    """
    Use phrase matching against NER_CATEGORIES to create entity-annotated Docs.
    """
    print()
    _thick()
    print("  Preparing NER (science entities) data")
    _thick()

    # Build phrase lists per category (sorted longest-first for greedy matching)
    category_phrases: dict[str, list[str]] = {}
    for cat, terms in NER_CATEGORIES.items():
        phrases = sorted(terms, key=len, reverse=True)
        category_phrases[cat] = phrases

    docs: list[Doc] = []
    span_counter: Counter = Counter()
    skipped = 0

    for entry in dataset:
        hinglish = entry.get("hinglish_roman", "")
        if not hinglish:
            skipped += 1
            continue

        tokens = tokenise_roman(hinglish)
        text_lower = " ".join(tokens).lower()

        # Find entity spans: (start_token, end_token, label)
        spans: list[tuple[int, int, str]] = []

        for cat, phrases in category_phrases.items():
            for phrase in phrases:
                phrase_tokens = phrase.lower().split()
                phrase_len = len(phrase_tokens)
                tokens_lower = [t.lower() for t in tokens]

                # Slide window to find matches
                for i in range(len(tokens_lower) - phrase_len + 1):
                    if tokens_lower[i:i + phrase_len] == phrase_tokens:
                        # Check no overlap with existing spans
                        new_start, new_end = i, i + phrase_len
                        overlaps = False
                        for s_start, s_end, _ in spans:
                            if not (new_end <= s_start or new_start >= s_end):
                                overlaps = True
                                break
                        if not overlaps:
                            spans.append((new_start, new_end, cat))
                            span_counter[cat] += 1

        if not spans:
            continue

        # Create Doc with entities
        doc = Doc(nlp.vocab, words=tokens)
        ents = []
        for start, end, label in spans:
            span = doc.char_span(
                sum(len(tokens[j]) + 1 for j in range(start)),
                sum(len(tokens[j]) + 1 for j in range(end)) - 1,
                label=label,
            )
            # char_span can fail, fall back to token-based Span
            if span is None:
                from spacy.tokens import Span
                span = Span(doc, start, end, label=label)
            ents.append(span)

        try:
            doc.ents = ents
            docs.append(doc)
        except ValueError:
            # Overlapping entities — skip this doc
            skipped += 1

    # Split
    train_docs, dev_docs = split_train_dev(docs)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_db = DocBin(store_user_data=False)
    for d in train_docs:
        train_db.add(d)
    train_db.to_disk(OUTPUT_DIR / "ner_train.spacy")

    dev_db = DocBin(store_user_data=False)
    for d in dev_docs:
        dev_db.add(d)
    dev_db.to_disk(OUTPUT_DIR / "ner_dev.spacy")

    # Stats
    total_spans = sum(span_counter.values())
    print(f"\n  Docs with entities: {len(docs)}")
    print(f"  Docs skipped      : {skipped}")
    print(f"  Total spans       : {total_spans}")
    print(f"  Train docs        : {len(train_docs)}")
    print(f"  Dev docs          : {len(dev_docs)}")
    print()
    print("  NER spans by label:")
    for cat in ["SCIENTIST", "ORGANELLE", "PROCESS", "CONCEPT", "INSTRUMENT"]:
        cnt = span_counter[cat]
        print(f"    {cat:14s}: {cnt}")
    print()
    print(f"  ✓ Saved: {OUTPUT_DIR / 'ner_train.spacy'}")
    print(f"  ✓ Saved: {OUTPUT_DIR / 'ner_dev.spacy'}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1c — INTENT DATA
# ─────────────────────────────────────────────────────────────────────────────

def prepare_intent(dataset: list[dict], nlp: spacy.Language) -> None:
    """
    Filter student queries with intent labels, create text-cat Docs.
    """
    print()
    _thick()
    print("  Preparing Intent (text classification) data")
    _thick()

    # Filter entries with is_student_query and a valid intent
    queries = [
        e for e in dataset
        if e.get("is_student_query") and e.get("intent") in INTENT_LABELS
    ]

    if len(queries) < 60:
        print(f"\n  ⚠ WARNING: Only {len(queries)} intent entries found.")
        print("  Consider adding more student queries to improve accuracy.")
        print("  Proceeding with available data.\n")

    intent_counter: Counter = Counter()
    docs: list[Doc] = []

    for entry in queries:
        hinglish = entry.get("hinglish_roman", "")
        intent = entry["intent"]
        if not hinglish:
            continue

        tokens = tokenise_roman(hinglish)
        doc = Doc(nlp.vocab, words=tokens)
        # Set cats: one-hot encoding
        doc.cats = {lbl: 1.0 if lbl == intent else 0.0 for lbl in INTENT_LABELS}
        docs.append(doc)
        intent_counter[intent] += 1

    # Split
    train_docs, dev_docs = split_train_dev(docs)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_db = DocBin(store_user_data=True)  # store cats via user_data
    for d in train_docs:
        train_db.add(d)
    train_db.to_disk(OUTPUT_DIR / "intent_train.spacy")

    dev_db = DocBin(store_user_data=True)
    for d in dev_docs:
        dev_db.add(d)
    dev_db.to_disk(OUTPUT_DIR / "intent_dev.spacy")

    # Stats
    print(f"\n  Total intent entries: {len(docs)}")
    print(f"  Train docs          : {len(train_docs)}")
    print(f"  Dev docs            : {len(dev_docs)}")
    print()
    print("  Intent distribution:")
    for lbl in INTENT_LABELS:
        cnt = intent_counter[lbl]
        print(f"    {lbl:20s}: {cnt}")
    print()
    print(f"  ✓ Saved: {OUTPUT_DIR / 'intent_train.spacy'}")
    print(f"  ✓ Saved: {OUTPUT_DIR / 'intent_dev.spacy'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prepare_spacy_data",
        description="Convert unified dataset → .spacy training files.",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to unified JSON dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--task", "-t",
        choices=["lid", "ner", "intent", "all"],
        default="all",
        help="Which task to prepare data for (default: all).",
    )
    args = parser.parse_args()

    print()
    _thick()
    print("  EduHinglish — Phase 3: Prepare spaCy Data")
    _thick()

    dataset = load_dataset(args.dataset)

    # Create a blank spaCy nlp object (just for Vocab)
    nlp = spacy.blank("en")

    if args.task in ("lid", "all"):
        prepare_lid(dataset, nlp)
    if args.task in ("ner", "all"):
        prepare_ner(dataset, nlp)
    if args.task in ("intent", "all"):
        prepare_intent(dataset, nlp)

    print()
    _thick()
    print("  All done! Files saved to training/data/")
    _thick()
    print()


if __name__ == "__main__":
    main()
