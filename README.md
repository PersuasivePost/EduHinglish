# EduHinglish 🎓

### AI-Powered Learning Companion for Hinglish-Speaking Indian Students

> **Final Year Engineering Project** | K.J. Somaiya College of Engineering, Mumbai
> **Team:** Ashvatth & Jatin | **Batch:** 2027

---

## The Problem

350–400 million people in India communicate in **Hinglish** (Hindi-English code-mix) as their primary informal language. Yet every NCERT textbook, every digital EdTech platform, and every AI tutoring tool forces students to choose — formal English OR formal Hindi.

A student who naturally asks _"Sir, Newton ka third law samjhao please"_ has no tool that understands that query and returns a curriculum-grounded Hinglish explanation.

**EduHinglish bridges that gap.**

---

## What We're Building

A 5-module integrated AI system:

| Module | What It Does | Status |
| --- | --- | --- |
| **M1: Input Processing** | Script detection → Normalization → Word-level LID → Intent extraction | ✅ Complete |
| **M2: NCERT Retriever** | RAG pipeline over NCERT textbook chunks | 📋 Planned |
| **M3: Hinglish Generator** | Fine-tuned IndicBART seq2seq Hinglish explanations | 📋 Planned |
| **M4: GEC Engine** | Code-mix-aware grammar error correction | 📋 Planned |
| **M5: Web Interface** | Flutter UI with FastAPI backend | 📋 Planned |

---

## Current Phase: Phase 2 — Multi-Chapter Dataset Pipeline ✅

**Scope:** 10 chapters across NCERT Class 9 & Class 10 Science — **950 labeled sentences** with word-level HI/EN/NE/UNIV/MIX annotations.

### Dataset Coverage

| Chapter Code | Chapter | Class | Entries |
| --- | --- | --- | --- |
| class9/ch01 | Matter in Our Surroundings | 9 | 95 |
| class9/ch02 | Is Matter Around Us Pure | 9 | 95 |
| class9/ch03 | Atoms and Molecules | 9 | 95 |
| class9/ch11 | Work and Energy | 9 | 95 |
| class9/ch12 | Sound | 9 | 95 |
| class10/ch05 | Periodic Classification of Elements | 10 | 95 |
| class10/ch06 | Life Processes | 10 | 95 |
| class10/ch07 | Control and Coordination | 10 | 95 |
| class10/ch08 | How do Organisms Reproduce? | 10 | 95 |
| class10/ch13 | Our Environment | 10 | 95 |
| **Total** | | | **950** |

### Label Distribution (unified dataset)

```
HI     9,558 tokens  (60%)  ████████████████████
EN     5,956 tokens  (37%)  ████████████
NE       140 tokens  ( 0%)
UNIV      53 tokens  ( 0%)
MIX        6 tokens  ( 0%)
```

### Entry Schema

Each entry in the dataset follows this structure:

```json
{
  "id": "10_06_001",
  "original_english": "The food we eat provides energy for all life processes.",
  "hinglish_roman": "Jo khana hum khate hain woh saare life processes ke liye energy deta hai.",
  "word_level_labels": {
    "Jo": "HI",
    "khana": "HI",
    "hum": "HI",
    "khate": "HI",
    "hain": "HI",
    "woh": "HI",
    "saare": "HI",
    "life": "EN",
    "processes": "EN",
    "ke": "HI",
    "liye": "HI",
    "energy": "EN",
    "deta": "HI",
    "hai": "HI"
  },
  "topic": "Life Processes - Introduction",
  "chapter": "Chapter 6: Life Processes",
  "class": "10",
  "code_mixing_type": "intra-sentential"
}
```

Optional fields: `is_student_query`, `intent`, `is_gec_sample`, `gec_data`, `hinglish_devanagari`, `notes`.

---

## Dataset Scripts

### annotation_helper.py — interactive word-level annotation

```bash
# Annotate sentences one by one for any chapter
python scripts/annotation_helper.py --chapter class10/ch06 \
    --output data/processed/class10_ch06/dataset.json

# List all available chapter codes
python scripts/annotation_helper.py --list-chapters

# Dry-run (print entry without saving)
python scripts/annotation_helper.py --dry-run
```

### chapter_dataset_builder.py — guided batch annotation + unified merge

```bash
# Guided annotation for a chapter (reads cleaned_text.txt as prompts)
python scripts/chapter_dataset_builder.py --chapter class10/ch06

# Set a custom entry target (default: 50)
python scripts/chapter_dataset_builder.py --chapter class9/ch05 --target 75

# Merge all ready chapters into the unified dataset
python scripts/chapter_dataset_builder.py --merge

# Show all chapters with current entry counts
python scripts/chapter_dataset_builder.py --list-chapters
```

### Other scripts

```bash
# Validate label consistency and compute CMI stats
python scripts/validate_dataset.py

# Augment existing sentences with rule-based variants
python scripts/augment_dataset.py

# Extract and clean text from NCERT PDFs in batch
python scripts/batch_pdf_extractor.py
```

---

## Pipeline (Phase 1 — M1 Input Processing)

```
Input (English / Hinglish / Mixed Script)
    │
    ▼
Script Detection (Roman / Devanagari / Mixed)
    │
    ▼
Normalization (spelling variants, abbreviations)
    │
    ▼
Word-Level Language Identification (HI / EN / NE / UNIV)
    │
    ▼
Sentence Segmentation
    │
    ▼
Tokenization
    │
    ▼
Stop Word Removal (language-aware)
    │
    ▼
Stemming (Porter / Lancaster)
    │
    ▼
Lemmatization (WordNet)
    │
    ▼
POS Tagging (Subject / Object / Verb analysis)
    │
    ▼
Structured Output (JSON with step-by-step trace)
```

---

## Project Structure

```
EduHinglish/
│
├── data/
│   ├── raw/                              # NCERT PDFs (not committed to git)
│   ├── processed/                        # Per-chapter datasets
│   │   ├── class9_ch01/dataset.json      # 95 entries each
│   │   ├── class9_ch02/dataset.json
│   │   ├── class9_ch03/dataset.json
│   │   ├── class9_ch11/dataset.json
│   │   ├── class9_ch12/dataset.json
│   │   ├── class10_ch05/dataset.json
│   │   ├── class10_ch06/dataset.json
│   │   ├── class10_ch07/dataset.json
│   │   ├── class10_ch08/dataset.json
│   │   └── class10_ch13/dataset.json
│   ├── hinglish/                         # Legacy / misc labeled data
│   └── unified_biology_dataset.json      # All chapters merged (950 entries)
│
├── scripts/
│   ├── annotation_helper.py              # Interactive word-level annotation CLI
│   ├── chapter_dataset_builder.py        # Batch annotation + --merge tool
│   ├── augment_dataset.py                # Rule-based sentence augmentation
│   ├── batch_pdf_extractor.py            # Process all NCERT PDFs at once
│   ├── validate_dataset.py               # Label consistency + CMI stats
│   └── download_nltk_data.py             # One-time NLTK data setup
│
├── src/
│   ├── pdf_extractor.py                  # NCERT PDF → clean text (Ashvatth)
│   ├── preprocessing.py                  # English NLP pipeline (Ashvatth)
│   ├── script_detector.py                # Script detection + LID (Jatin)
│   ├── normalizer.py                     # Hinglish normalization (Jatin)
│   ├── hinglish_dataset_creator.py       # Legacy dataset creation (Jatin)
│   └── pipeline.py                       # Unified pipeline (Both)
│
├── outputs/
│   └── pipeline_results/                 # JSON pipeline outputs
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_script_detector.py
│   └── test_pipeline.py
│
├── workspace/
│   └── phases.md                         # Full 8-phase project plan
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python **3.10.x** (recommended: 3.10.11)
- Git

### Installation

```bash
# Clone the repo
git clone https://github.com/PersuasivePost/EduHinglish.git
cd EduHinglish

# Create virtual environment with Python 3.10
python3.10 -m venv eduhinglish_env        # Mac/Linux
# py -3.10 -m venv eduhinglish_env        # Windows

# Activate
source eduhinglish_env/bin/activate       # Mac/Linux
# eduhinglish_env\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python scripts/download_nltk_data.py

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Get NCERT PDFs

Download Class 9 & Class 10 Science from [ncert.nic.in](https://ncert.nic.in/textbook.php)
Place them at: `data/raw/`

---

## Running the Pipeline

```bash
# Run the full unified pipeline
python src/pipeline.py

# Test individual modules
python src/script_detector.py
python src/preprocessing.py
```

---

## Key Research Gaps This Project Addresses

1. **Language-Curriculum Disconnect** — No tool accepts Hinglish queries and returns NCERT-grounded answers
2. **No Script-Agnostic Processing** — No tool handles Roman + Devanagari + Mixed script Hinglish
3. **No Code-Mix-Aware GEC** — All GEC systems assume monolingual input
4. **No Hinglish Educational Dataset** — Creating a novel labeled dataset as part of this project
5. **LLM Hallucination** — RAG architecture grounds all answers in NCERT
6. **No Pedagogical Scaffolding** — LLM prompted to follow Indian teaching style
7. **No Evaluation Framework** — Defining custom metrics (CMI, comprehension tests)

---

## Literature Foundation

11 papers reviewed covering:

- Code-switching NLP (Çetinoğlu 2016, Sitaram 2019, Winata 2022)
- Educational AI (Lan 2023)
- Efficient CS models (Udawatta 2024)
- Hinglish generation (Pandey 2025, Dash 2025)
- Indic language NLP (Maddu & Sanapala 2024)
- Semantic understanding (Blanco & Moldovan 2013, Roy & Zeng 2013, O'Shea 2013)

---

## Tech Stack

| Module | Component | Technology | Rationale |
| --- | --- | --- | --- |
| M1 | Script Detection | Unicode range analysis (Python) | Zero dependency; handles all three script types reliably |
| M1 | Transliteration | IndicXlit (AI4Bharat) | Best-in-class Devanagari→Roman for Indic scripts |
| M1 | LID — Phase 1 | Rule-based lexicon (WordLevelLID) | 120+ Hindi + 80+ science terms; fast baseline |
| M1 | LID — Phase 2 | MuRIL + token classifier (HuggingFace) | Best Hinglish LID accuracy (Winata et al. 2022) |
| M1 | Intent Classifier | spaCy textcat / MuRIL [CLS] | Lightweight; 6-class intent classification |
| M2 | PDF Extraction | pdfplumber | Superior NCERT table/multi-column handling vs PyPDF2 |
| M2 | Embeddings | BGE-base-en-v1.5 (BAAI) | Top MTEB-ranked embeddings; 768D; runs on CPU |
| M2 | Vector DB | ChromaDB (dev) / Pinecone (production) | Fully local in development; Pinecone provides managed vector search in production with no infrastructure overhead |
| M2 | Re-ranker | cross-encoder/ms-marco-MiniLM-L-12-v2 | Lightweight cross-encoder; improves precision@3 significantly |
| M3 | Generator | IndicBART (AI4Bharat) | 244M parameter seq2seq model pre-trained on 11 Indic languages + English; handles Hinglish natively |
| M3 | Fine-tuning | LoRA via PEFT (HuggingFace) | Fits easily on Colab T4 GPU; fast training time for seq2seq task |
| M4 | English GEC | GECToR (BEA-2019 + Lang-8) | Best English GEC F1; token edit-based, not seq2seq |
| M4 | Hindi GEC | Rule-based (gender/verb agreement) | Handles 3 most common Hinglish grammar error types |
| M5 | UI Prototype | Flutter | Rapid iteration; minimal JS; good for demos |
| M5 | UI Production | Flutter + FastAPI + Pydantic | Cross-platform (Android/iOS/Web) from single codebase; Pydantic enforces API schema and validates LLM structured outputs |
| All | Runtime | Python 3.10, PyTorch 2.1 | Standard ML stack; all dependencies open-source |

---

## Team

| Name | Role |
| --- | --- |
| Ashvatth | PDF extraction, English preprocessing pipeline, unified pipeline class |
| Jatin | Hinglish dataset creation, script detection, normalization, LID module, annotation tooling |

---

## Roadmap

- [x] Literature survey (11 papers)
- [x] Market analysis & gap identification
- [x] System architecture design
- [x] **Phase 1 — M1 Preprocessing pipeline**
- [x] **Phase 2 — Multi-chapter dataset pipeline (950 entries, 10 chapters)**
- [ ] Phase 3 — spaCy model training (LID, NER, Intent)
- [ ] Phase 4 — MuRIL fine-tuning (Colab T4)
- [ ] Phase 5 — NCERT ChromaDB knowledge base
- [ ] Phase 6 — IndicBART Hinglish generator fine-tuning
- [ ] Phase 7 — GEC engine
- [ ] Phase 8 — Full pipeline integration + Flutter/FastAPI UI
