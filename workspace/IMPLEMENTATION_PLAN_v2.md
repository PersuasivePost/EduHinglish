# EduHinglish — Implementation Plan v2.0

> **Goal:** Upgrade EduHinglish from a biology-only, 5-intent, broken translation system into a
> full CBSE Class 9–10 multi-subject Hinglish LLM using Gemma-3-4B-it for QA and IndicBART for
> bidirectional Hinglish↔English translation.

---

## Background & Current State

| Component | Current State | Problem |
|---|---|---|
| Dataset | `unified_biology_dataset_v2.json` — 950 biology sentence pairs | Too small, wrong format, biology-only |
| LID Model | MuRIL, 5 tags (HI/EN/NE/UNIV/MIX) | Missing NUM, SYM — breaks on numericals & chemistry |
| Intent Model | MuRIL, 5 intents (biology only) | No solve_numerical, chemical_reaction, derivation etc. |
| Generator | IndicBART fine-tuned for translation | Hallucination, repetition — wrong task framing |
| Vector DB | 10 biology chapters (Class 9 & 10) | Needs all subjects |
| Pipeline | M1→M2→M3 wired | M3 broken; no translator module |

---

## New Architecture (Post-Implementation)

```
Student Query (Hinglish / Hindi / English)
         │
         ▼
[M1] MuRIL Input Processor  ──────────────────────────────────────
  │  Word-level LID: HI / EN / NE / UNIV / MIX / NUM / SYM (7 tags)
  │  Intent: 12 classes
  │  Subject + Class detected from query / UI selector
  └─ Output: normalized_query, intent, subject, class
         │
         ▼
[M2] ChromaDB RAG Retriever  ─────────────────────────────────────
  │  Embed normalized query (multilingual-MiniLM)
  │  Filter by subject + class
  │  Return top-3 NCERT English chunks
  └─ Output: [context_chunk_1, context_chunk_2, context_chunk_3]
         │
         ▼
[M3-A] IndicBART Translator (Hinglish → English)  ────────────────
  │  Input: student's Hinglish query
  │  Output: clean English query
  └─ e.g. "Newton ka 3rd law samjhao" → "Explain Newton's 3rd law"
         │
         ▼
[M3-B] Gemma-3-4B-it (English QA)  ──────────────────────────────
  │  Input: context (RAG) + English query + intent
  │  Generates: detailed English answer (3–6 sentences)
  └─ Adapts response style per intent (solve / explain / list etc.)
         │
         ▼
[M3-C] IndicBART Translator (English → Hinglish)  ────────────────
  │  Input: Gemma's English answer
  │  Output: Natural Hinglish answer
  └─ e.g. "Newton's 3rd law states..." → "Newton ka 3rd law kehta hai..."
         │
         ▼
Final Hinglish Answer → M5 UI
```

---

## Data We Have (Excel Files)

### Class 9 — `NCERT_Class 9/`
| File | Subject |
|---|---|
| `Class 9 Biology.xlsx` | Biology |
| `Claas 9 Chemistry.xlsx` | Chemistry |
| `Class 9 Physics.xlsx` | Physics |
| `Class 9 Maths.xlsx` | Mathematics |
| `Class 9 HIstory.xlsx` | History (SST) |
| `Class 9 Civics.xlsx` | Civics (SST) |
| `Class 9 Geography and Economics.xlsx` | Geography & Economics |
| `Class 9 Hindi.xlsx` | Hindi |
| `class 9 English.xlsx` | English |
| `Class 9 agricultural science.xlsx` | Agricultural Science |

### Class 10 — `NCERT_Class 10/`
| File | Subject |
|---|---|
| `10-bio.xlsx` | Biology |
| `10-chem.xlsx` | Chemistry |
| `10-phy.xlsx` | Physics |
| `10-maths.xlsx` | Mathematics |
| `10-civicshistory.xlsx` | Civics & History |
| `10-geoeco.xlsx` | Geography & Economics |
| `10-env-sci.xlsx` | Environmental Science |
| `10-eng.xlsx` | English |
| `10-agri-sci.xlsx` | Agricultural Science |

**Total: 19 Excel files. All columns: `question`, `answer`, `context`, `class`**
**All data is in English — Hinglish versions will be auto-generated using IndicBART.**

---

## Phase 1 — Data Pipeline (Excel → Unified JSON)

### 1.1 — Convert Excel → `cbse_qa_dataset.json`

**File to create:** `training/convert_excel_to_json.py` *(NEW)*

Each Excel row → one JSON record:
```json
{
  "id": "c9_bio_0042",
  "class": "9",
  "subject": "Biology",
  "source_file": "Class 9 Biology.xlsx",
  "question_english": "What is the function of the cell membrane?",
  "answer_english": "The cell membrane controls the movement of substances in and out of the cell...",
  "context_english": "The cell membrane is a thin, flexible barrier surrounding the cell...",
  "question_hinglish": null,
  "answer_hinglish": null,
  "intent": null,
  "word_level_labels": null
}
```

*`question_hinglish`, `answer_hinglish`, `intent`, `word_level_labels` start as `null` — filled in later phases.*

**Output:** `data/cbse_qa_dataset.json`

---

### 1.2 — Auto-generate Hinglish using IndicBART

**File to create:** `training/generate_hinglish_pairs.py` *(NEW)*

Run **base** IndicBART (before fine-tuning) to translate each English question + answer into Hinglish:
```python
# For each record in cbse_qa_dataset.json:
question_hinglish = indicbart_translate(question_english, direction="en→hi")
answer_hinglish   = indicbart_translate(answer_english,   direction="en→hi")
```

> **Note:** These are **approximate** Hinglish translations. Quality will improve after IndicBART is
> fine-tuned on the translation dataset in Phase 4. For now, they give enough signal to train
> MuRIL LID and Intent models.

**Output:** `data/cbse_qa_dataset_with_hinglish.json` (same records, `null` fields now populated)

---

### 1.3 — Auto-assign Intent Labels

**File to create:** `training/auto_label_intents.py` *(NEW)*

Rule-based intent labeling from question text (English):
```python
INTENT_RULES = [
    (["what is", "define", "meaning of"],                 "definition"),
    (["explain", "how does", "why does", "describe"],     "explain_concept"),
    (["difference between", "compare", "vs", "versus"],   "compare_concepts"),
    (["example", "give an example", "real life"],         "give_example"),
    (["formula", "equation for", "expression for"],       "formula_request"),
    (["calculate", "find the value", "solve", "numerical"],"solve_numerical"),
    (["reaction", "balanced equation", "products of"],    "chemical_reaction"),
    (["draw", "diagram", "label", "parts of"],            "diagram_description"),
    (["prove", "derive", "derivation of"],                "proof_derivation"),
    (["application", "used in", "where is it used"],      "application_reallife"),
    (["list", "types of", "how many", "name the"],        "list_enumerate"),
    (["summary", "revision", "important points", "recap"],"revision_summary"),
]
```

**Output:** `data/cbse_qa_dataset_labeled.json` ← **main unified dataset going forward**

---

## Phase 2 — LID Tag Expansion (5 → 7 Tags)

### Current Tags (keep as-is)
```python
{"HI": 0, "EN": 1, "NE": 2, "UNIV": 3, "MIX": 4}
```

### New Tags
| Tag | ID | Meaning | Examples |
|---|---|---|---|
| `NUM` | 5 | Numeric value / expression | `9.8`, `6.022×10²³`, `3rd`, `2H₂`, `₂` |
| `SYM` | 6 | Scientific symbol / formula | `CO₂`, `H₂SO₄`, `F=ma`, `→`, `α`, `λ`, `∆H` |

### Detection Logic
```python
import re

def detect_extra_label(token: str) -> str | None:
    # SYM: chemical formula pattern or math operators
    if re.match(r'^[A-Z][a-z]?\d*[₀-₉]*(\([A-Z][a-z]?\d*\))*$', token):
        return "SYM"   # e.g. CO2, H2SO4, Ca(OH)2
    if any(c in token for c in ['→', '⇌', '∑', '∆', 'α', 'β', 'λ', 'μ', '=', '±']):
        return "SYM"
    # NUM: numeric token
    if re.match(r'^\d+[\.,]?\d*([×x]\d+)?$', token):
        return "NUM"   # e.g. 9.8, 6.022, 100
    if re.match(r'^\d+(st|nd|rd|th)$', token):
        return "NUM"   # e.g. 3rd, 1st
    return None
```

### Final LID Tag Set (7 tags)
```python
LABEL2ID = {
    "HI":   0,   # Hindi word (Roman script)
    "EN":   1,   # English word
    "NE":   2,   # Named entity (person, place, org)
    "UNIV": 3,   # Universal (punctuation, language-neutral)
    "MIX":  4,   # Code-mixed hybrid token
    "NUM":  5,   # Numeric / mathematical number  [NEW]
    "SYM":  6,   # Scientific symbol / chemical formula / math operator  [NEW]
}
```

### Files to Modify
- `training/train_muril_lid.py` — change `LABEL2ID` dict, update `NUM_LABELS = 7`
- `training/prepare_spacy_data.py` — add `NUM`/`SYM` detection before existing label logic
- `training/train_lid.py` — same `LABEL2ID` update (spaCy baseline)

**Retrained model saved to:** `models/muril_lid_v3/`

---

## Phase 3 — Intent Class Expansion (5 → 12 Classes)

### Full Intent Set
```python
INTENT2ID = {
    "explain_concept":     0,   # kept — same ID
    "definition":          1,   # kept — same ID
    "compare_concepts":    2,   # kept — same ID
    "give_example":        3,   # kept — same ID
    "formula_request":     4,   # kept — same ID
    "solve_numerical":     5,   # NEW — Physics/Math/Chemistry numericals
    "chemical_reaction":   6,   # NEW — Chemistry balanced equations
    "diagram_description": 7,   # NEW — Biology/Physics diagrams
    "proof_derivation":    8,   # NEW — Math/Physics derivations
    "application_reallife":9,   # NEW — real-world use cases
    "list_enumerate":     10,   # NEW — types/characteristics/names
    "revision_summary":   11,   # NEW — chapter summaries
}
```

### Intent → Trigger Phrases (for auto-labeling + student UX)
| Intent | Common Hinglish Triggers |
|---|---|
| `explain_concept` | "samjhao", "kaise hota hai", "explain karo" |
| `definition` | "kya hota hai", "define karo", "matlab kya hai" |
| `compare_concepts` | "fark kya hai", "difference", "vs", "aur mein" |
| `give_example` | "example do", "example batao", "jaise ki" |
| `formula_request` | "formula kya hai", "equation do", "formula batao" |
| `solve_numerical` | "solve karo", "calculate", "answer nikalo", "value kya hai" |
| `chemical_reaction` | "reaction kya hogi", "balanced equation", "products kya hain" |
| `diagram_description` | "diagram samjhao", "parts batao", "label karo" |
| `proof_derivation` | "prove karo", "derive karo", "derivation batao" |
| `application_reallife` | "real life mein", "daily life mein kahan use hota" |
| `list_enumerate` | "kitne types hain", "list karo", "naam batao" |
| `revision_summary` | "chapter summary", "important points", "quick revision" |

### Training Data for Intent Model
- Primary: `data/cbse_qa_dataset_labeled.json` — `question_hinglish` as input text
- Supplementary: `data/hinglish/student_queries.json` — 5 manually annotated queries (existing)

### Files to Modify
- `training/train_muril_intent.py` — expand `INTENT2ID` to 12 classes, `NUM_INTENTS = 12`
- `training/prepare_hf_data.py` — update intent label list
- `training/train_intent.py` — spaCy baseline update (same dict)

**Retrained model saved to:** `models/muril_intent_v3/`

---

## Phase 4 — IndicBART: Retrain as Bidirectional Translator

IndicBART's new role is **translation only** — not QA generation. Fine-tuned on Hinglish ↔ English
sentence pairs derived from the Excel dataset.

### Why This Role Change?
IndicBART was pre-trained on the **Samanantar** corpus — millions of Hindi↔English parallel
sentence pairs. Translation is exactly what it was built for. QA generation requires instruction
following, which it was never trained for → hallucination and repetition result.

### Training Data Format

**File to create:** `training/prepare_translation_data.py` *(NEW)*

From `cbse_qa_dataset_labeled.json`, extract parallel pairs in **both directions**:

```python
# Direction 1: Hinglish → English  (used at query time: M3-A)
{
    "source": "Newton ka 3rd law kya hota hai?",
    "target": "What is Newton's 3rd law?",
    "direction": "hi2en"
}

# Direction 2: English → Hinglish  (used at answer time: M3-C)
{
    "source": "Newton's 3rd law states that every action has an equal and opposite reaction.",
    "target": "Newton ka 3rd law kehta hai ki har action ka ek equal aur opposite reaction hota hai.",
    "direction": "en2hi"
}
```

Both questions AND answers generate pairs:
- ~21,000 QA pairs × 2 (Q + A) × 2 (directions) = **~84,000 translation pairs**

### IndicBART Tokenizer Configuration
```python
# For EN → HI translation (M3-C):
tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "hi_IN"
forced_bos_token_id = tokenizer.lang_code_to_id["hi_IN"]

# For HI → EN translation (M3-A):
tokenizer.src_lang = "hi_IN"
tokenizer.tgt_lang = "en_XX"
forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]
```

**File to create:** `training/train_indicbart_translator.py` *(NEW)*

Key differences vs old `training/train_indicbart.py`:
- No Type A/B/C pairs — single clean `(source, target, direction)` format
- Bidirectional training in same dataset
- `no_repeat_ngram_size=3` and `repetition_penalty=1.3` added to generation config
- `forced_bos_token_id` set per direction

**Model saved to:** `models/indicbart_translator_v1/`

---

## Phase 5 — Gemma-3-4B-it: Fine-tune for NCERT QA

### Model
`google/gemma-3-4b-it` — 4 billion parameter instruction-tuned model. Fine-tuned with QLoRA on
Colab A100.

### Training Data Format

**File to create:** `training/prepare_gemma_data.py` *(NEW)*

Uses Gemma's native chat template format:
```
<start_of_turn>user
You are an expert NCERT tutor for Indian students (Class {class}).

Subject: {subject}
Chapter context:
{context_english}

Student's question: {question_english}
Query type: {intent}
<end_of_turn>
<start_of_turn>model
{answer_english}
<end_of_turn>
```

### Intent → Response Style Mapping
Each intent injects a style instruction into the system prompt so Gemma formats its answer
appropriately:

| Intent | Response Style Instruction |
|---|---|
| `explain_concept` | Explain clearly in 3-4 sentences with the reasoning. |
| `definition` | Give a precise definition followed by one key property. |
| `compare_concepts` | List differences in a structured way (A vs B format). |
| `solve_numerical` | Show step-by-step working with units at each step. |
| `chemical_reaction` | Write the balanced equation, then explain each reactant and product. |
| `proof_derivation` | Show the derivation step by step with each mathematical step explained. |
| `formula_request` | State the formula, define each variable, and give the SI unit. |
| `diagram_description` | Describe the structure in a spatial, top-to-bottom manner. |
| `list_enumerate` | List all items clearly numbered with a brief note on each. |
| `give_example` | Give 2 real-life examples with brief explanation of each. |
| `application_reallife` | Explain 2-3 real-world applications with context. |
| `revision_summary` | Give 5-7 bullet points covering the most important points. |

### Training Configuration

**File to create:** `training/train_gemma_ncert.py` *(NEW)*

```python
MODEL_NAME     = "google/gemma-3-4b-it"
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]   # Gemma attention layers
BATCH_SIZE     = 4       # per device, Colab A100
GRAD_ACCUM     = 4       # effective batch size = 16
EPOCHS         = 3
LR             = 2e-4
MAX_SEQ_LEN    = 1024    # context + question + answer fits comfortably
QUANTIZATION   = "4bit"  # QLoRA: bitsandbytes NF4
```

**Model saved to:** `models/gemma_ncert_v1/`

---

## Phase 6 — Pipeline Wiring: M3 Refactor

### New Module

**File to create:** `src/translator.py` *(NEW)*

```python
class HinglishTranslator:
    """
    IndicBART-based bidirectional translator.

    Methods:
        translate(text, direction="hi2en") → str   (Hinglish → English)
        translate(text, direction="en2hi") → str   (English → Hinglish)
    """
```

### `src/generator.py` — Modified

**Old behaviour:** Loads IndicBART, generates Hinglish directly from English NCERT context.

**New behaviour:** Loads Gemma-3-4B-it, generates a detailed English answer given:
- Retrieved English context (from M2)
- English query (translated by M3-A)
- Intent label (from M1)

### `src/pipeline.py` — Modified (M3 section)

```python
# ── Old M3 (one step) ─────────────────────────────────────────────
answer_hinglish = generator.generate(context + query_hinglish)

# ── New M3 (three steps) ──────────────────────────────────────────
query_english   = translator.translate(query_hinglish, direction="hi2en")   # M3-A
answer_english  = gemma.generate(context, query_english, intent)             # M3-B
answer_hinglish = translator.translate(answer_english, direction="en2hi")    # M3-C
```

---

## Phase 7 — Vector DB Expansion (All Subjects)

Currently only 10 biology chapters are indexed in ChromaDB. Expand to all Class 9 & 10 subjects.

### New Chunk Metadata Structure
```python
{
    "chunk_id": "c9_phy_ch09_chunk_003",
    "text": "Newton's third law states that for every action...",
    "metadata": {
        "class": "9",
        "subject": "Physics",
        "chapter_num": "ch09",
        "chapter_title": "Force and Laws of Motion"
    }
}
```

The `subject` and `class` metadata fields enable **filtered retrieval** — a Physics query only
searches Physics chunks, preventing irrelevant Biology results from contaminating the context.

### Files to Modify
- `src/chunker.py` — expand `CHAPTER_MAP` dict to include all Class 9 & 10 chapters across all
  subjects
- `src/retriever.py` — add `subject=None, class_num=None` filter parameters to the `retrieve()`
  method; pass as ChromaDB `where` clause
- `src/embedder.py` — re-run full indexing pipeline after chunker is updated

---

## Execution Order

Each phase is sequential — its output is consumed by the next.

```
Phase 1a │ training/convert_excel_to_json.py      → data/cbse_qa_dataset.json
Phase 1b │ training/generate_hinglish_pairs.py    → data/cbse_qa_dataset_with_hinglish.json
Phase 1c │ training/auto_label_intents.py         → data/cbse_qa_dataset_labeled.json  ★ MAIN
         │
Phase 2  │ [MODIFY] train_muril_lid.py            → models/muril_lid_v3/        (7 tags)
Phase 3  │ [MODIFY] train_muril_intent.py         → models/muril_intent_v3/     (12 intents)
         │
Phase 4  │ training/prepare_translation_data.py   →
         │ training/train_indicbart_translator.py → models/indicbart_translator_v1/
         │
Phase 5  │ training/prepare_gemma_data.py         →
         │ training/train_gemma_ncert.py          → models/gemma_ncert_v1/
         │
Phase 6  │ [NEW]    src/translator.py             → HinglishTranslator class
         │ [MODIFY] src/generator.py              → Gemma QA engine
         │ [MODIFY] src/pipeline.py               → M1→M2→M3-A→M3-B→M3-C wired
         │
Phase 7  │ [MODIFY] src/chunker.py + embedder.py → ChromaDB re-indexed (all subjects)
```

---

## Complete File Changelist

### New Files (8)
| File | Phase | Purpose |
|---|---|---|
| `training/convert_excel_to_json.py` | 1a | Merge all 19 Excel files → unified JSON |
| `training/generate_hinglish_pairs.py` | 1b | Auto-generate Hinglish Q&A via IndicBART |
| `training/auto_label_intents.py` | 1c | Rule-based intent assignment |
| `training/prepare_translation_data.py` | 4 | Build 84k bidirectional translation pairs |
| `training/train_indicbart_translator.py` | 4 | Fine-tune IndicBART as translator |
| `training/prepare_gemma_data.py` | 5 | Build Gemma chat-format QA dataset |
| `training/train_gemma_ncert.py` | 5 | Fine-tune Gemma with QLoRA |
| `src/translator.py` | 6 | HinglishTranslator inference class |

### Modified Files (8)
| File | Change | Phase |
|---|---|---|
| `training/train_muril_lid.py` | Add NUM(5), SYM(6); `NUM_LABELS=7` | 2 |
| `training/prepare_spacy_data.py` | Add NUM/SYM regex detection logic | 2 |
| `training/train_muril_intent.py` | Expand `INTENT2ID` to 12; `NUM_INTENTS=12` | 3 |
| `training/prepare_hf_data.py` | Update intent label list to 12 classes | 3 |
| `src/generator.py` | Replace IndicBART QA → Gemma-3-4B-it QA | 6 |
| `src/pipeline.py` | Wire M3-A → M3-B → M3-C | 6 |
| `src/chunker.py` | Expand `CHAPTER_MAP` to all subjects | 7 |
| `src/retriever.py` | Add subject/class filter to `retrieve()` | 7 |

### Superseded Files (keep but archive — do not delete)
| File | Replaced By |
|---|---|
| `data/unified_biology_dataset_v2.json` | `data/cbse_qa_dataset_labeled.json` |
| `training/train_indicbart.py` | `training/train_indicbart_translator.py` |
| `training/prepare_seq2seq_data.py` | `training/prepare_translation_data.py` |

---

## Hardware Requirements

| Phase | Where to Run | Minimum GPU |
|---|---|---|
| Phase 1 — data scripts | Local Mac | None (pure Python) |
| Phase 2 — MuRIL LID retrain | Google Colab (free T4) | T4 16GB |
| Phase 3 — MuRIL Intent retrain | Google Colab (free T4) | T4 16GB |
| Phase 4 — IndicBART translator | Google Colab (free T4) | T4 16GB |
| Phase 5 — Gemma-3-4B-it QLoRA | Google Colab Pro (A100) | A100 40GB |
| Phase 6–7 — code changes | Local Mac | None |

> **Gemma inference after training:** A T4 with 4-bit quantization (bitsandbytes) is sufficient.
> Mac CPU inference is possible but slow (~30s per answer).

---

## Notes

- **Hindi & English literature Excels** (`Class 9 Hindi.xlsx`, `class 9 English.xlsx`,
  `10-eng.xlsx`) — these are passage-based comprehension QA sets, structurally different from
  factual NCERT questions. Include or exclude based on project scope.

- **Phase 1b quality caveat** — The Hinglish generated by base (untuned) IndicBART will be
  Devanagari-heavy and somewhat stiff. This is acceptable for bootstrapping LID/Intent training.
  After Phase 4 (translator fine-tuning), re-running Phase 1b will produce much better Roman
  Hinglish — optionally refresh the dataset at that point.

- **Intent label noise** — Rule-based labeling (Phase 1c) will mislabel edge cases
  (e.g., "What is the formula for..." could be `definition` or `formula_request`). The MuRIL
  intent model will learn to handle these from context rather than exact keyword matches, so
  some noise in labels is acceptable.
