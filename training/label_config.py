"""
EduHinglish — Shared Label Config
==================================
Single source of truth for label sets and the dataset/model version used by
the M1 training scripts (spaCy + MuRIL). Scripts in training/ import this
directly (`from label_config import ...`); scripts/ add training/ to sys.path.

INTENT_LABELS mirror the `intent` field of the QA datasets (bio_9_10.json,
ss9.json, ...). The intent classifier's prediction is passed to Gemma as
"Intent Type: <intent>", so both must use the same taxonomy.
"""

DATA_VERSION = "v3"

# ── Word-level language ID ───────────────────────────────────────────────────
LID_LABELS = ["HI", "EN", "NE", "UNIV", "MIX"]

# ── Query intent (matches the QA dataset taxonomy) ──────────────────────────
INTENT_LABELS = [
    "explain_concept",
    "definition",
    "compare_concepts",
    "give_example",
    "application_reallife",
    "list_enumerate",
    "describe_function",
    "analytical",
    "short_answer",
    "numerical_problem",
    "mcq",
    "true_false",
    "fill_blank",
    "one_word",
    "assertion_reason",
]

# Legacy (v2 annotated sentences) intent names → current taxonomy.
# Anything not in INTENT_LABELS after aliasing is dropped (e.g. formula_request).
INTENT_ALIASES = {
    "define_term":   "definition",
    "apply_concept": "application_reallife",
}

# ── NER entity types ─────────────────────────────────────────────────────────
NER_LABELS = [
    # Biology
    "SCIENTIST", "ORGANELLE", "PROCESS", "CONCEPT", "INSTRUMENT",
    # Social Science
    "PERSON", "PLACE", "CIVILISATION", "INSTITUTION",
]


def normalize_intent(intent: str | None) -> str | None:
    """Map an intent to the current taxonomy; None if it has no place in it."""
    if not intent:
        return None
    intent = INTENT_ALIASES.get(intent, intent)
    return intent if intent in INTENT_LABELS else None
