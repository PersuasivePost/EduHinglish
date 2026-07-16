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

| Module                     | What It Does                                                           | Status         |
| -------------------------- | ---------------------------------------------------------------------- | -------------- |
| **M1: Input Processing**   | Script detection → Normalization → Multi-layer LID → Intent extraction | ✅ Complete    |
| **M2: NCERT Retriever**    | RAG pipeline over NCERT textbook chunks using ChromaDB                 | ✅ Complete    |
| **M3: Hinglish Generator** | Fine-tuned IndicBART seq2seq Hinglish explanations                     | 📋 In Progress |
| **M4: GEC Engine**         | Code-mix-aware grammar error correction                                | 📋 Planned     |
| **M5: Web Interface**      | Streamlit / Flutter UI with FastAPI backend                            | 📋 Planned     |

---

## Progress Roadmap

### Phase 1 & 2 — Dataset Foundation ✅

- **Scope:** 10 chapters across NCERT Class 9 & Class 10 Science.
- **Output:** `data/unified_biology_dataset_v2.json` (950 labeled sentences).
- **Quality:** Word-level HI/EN/NE/UNIV/MIX annotations verified via `validate_dataset.py`.

### Phase 3 & 4 — Intelligence Layer (LID & Intent) ✅

We've moved from rule-based baselines to production-grade Transformers.

| Model                | Tech Stack                      | Metric   | Baseline (spaCy) | MuRIL (Final) |
| -------------------- | ------------------------------- | -------- | ---------------- | ------------- |
| **Language ID**      | MuRIL + LoRA                    | Accuracy | 98.1%            | **99.2%**     |
| **Intent Detection** | MuRIL + Sequence Classification | Accuracy | 70.0%            | **82.5%**     |

- **Models Saved:** `models/muril_lid_v2/`, `models/muril_intent_v2/`.
- **Infrastructure:** Fine-tuned on Colab T4, optimized for local inference.

### Phase 5 — Knowledge Base & RAG ✅

The system can now "read" and retrieve relevant sections from NCERT textbooks.

- **Vector Database:** ChromaDB (Local file-based).
- **Embedding Model:** `paraphrase-multilingual-MiniLM-L12-v2` (Multi-lingual support).
- **Pipeline:** `chunker.py` → `embedder.py` → `retriever.py`.
- **Coverage:** All 10 Science chapters processed and indexed.

---

## Current Status: Phase 6 — Hinglish Answer Generation 🚀

We are currently working on **M3**, where the retrieved NCERT English context is transformed into natural Hinglish explanations using fine-tuned **IndicBART**.

---

### Label Distribution (unified dataset)

```
HI     9,558 tokens  (60%)  ████████████████████
EN     5,956 tokens  (37%)  ████████████
NE       140 tokens  ( 0%)
UNIV      53 tokens  ( 0%)
MIX        6 tokens  ( 0%)
```

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

| Module | Component         | Technology                             | Rationale                                                                                                                |
| ------ | ----------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| M1     | Script Detection  | Unicode range analysis (Python)        | Zero dependency; handles all three script types reliably                                                                 |
| M1     | Transliteration   | IndicXlit (AI4Bharat)                  | Best-in-class Devanagari→Roman for Indic scripts                                                                         |
| M1     | LID — Phase 1     | Rule-based lexicon (WordLevelLID)      | 120+ Hindi + 80+ science terms; fast baseline                                                                            |
| M1     | LID — Phase 2     | MuRIL + token classifier (HuggingFace) | Best Hinglish LID accuracy (Winata et al. 2022)                                                                          |
| M1     | Intent Classifier | spaCy textcat / MuRIL [CLS]            | Lightweight; 6-class intent classification                                                                               |
| M2     | PDF Extraction    | pdfplumber                             | Superior NCERT table/multi-column handling vs PyPDF2                                                                     |
| M2     | Embeddings        | BGE-base-en-v1.5 (BAAI)                | Top MTEB-ranked embeddings; 768D; runs on CPU                                                                            |
| M2     | Vector DB         | ChromaDB (dev) / Pinecone (production) | Fully local in development; Pinecone provides managed vector search in production with no infrastructure overhead        |
| M2     | Re-ranker         | cross-encoder/ms-marco-MiniLM-L-12-v2  | Lightweight cross-encoder; improves precision@3 significantly                                                            |
| M3     | Generator         | IndicBART (AI4Bharat)                  | 244M parameter seq2seq model pre-trained on 11 Indic languages + English; handles Hinglish natively                      |
| M3     | Fine-tuning       | LoRA via PEFT (HuggingFace)            | Fits easily on Colab T4 GPU; fast training time for seq2seq task                                                         |
| M4     | English GEC       | GECToR (BEA-2019 + Lang-8)             | Best English GEC F1; token edit-based, not seq2seq                                                                       |
| M4     | Hindi GEC         | Rule-based (gender/verb agreement)     | Handles 3 most common Hinglish grammar error types                                                                       |
| M5     | UI Prototype      | Flutter                                | Rapid iteration; minimal JS; good for demos                                                                              |
| M5     | UI Production     | Flutter + FastAPI + Pydantic           | Cross-platform (Android/iOS/Web) from single codebase; Pydantic enforces API schema and validates LLM structured outputs |
| All    | Runtime           | Python 3.10, PyTorch 2.1               | Standard ML stack; all dependencies open-source                                                                          |

---

## Team

| Name     | Role                                                                                       |
| -------- | ------------------------------------------------------------------------------------------ |
| Ashvatth | PDF extraction, English preprocessing pipeline, unified pipeline class                     |
| Jatin    | Hinglish dataset creation, script detection, normalization, LID module, annotation tooling |

---

## Roadmap

- [x] Literature survey (11 papers)
- [x] Market analysis & gap identification
- [x] System architecture design
- [x] **Phase 1 — M1 Preprocessing pipeline**
- [x] **Phase 2 — Multi-chapter dataset pipeline (950 entries, 10 chapters)**
- [x] Phase 3 — spaCy model training (LID, NER, Intent)
- [x] Phase 4 — MuRIL fine-tuning (Colab T4)
- [x] Phase 5 — NCERT ChromaDB knowledge base
- [ ] Phase 6 — IndicBART Hinglish generator fine-tuning
- [ ] Phase 7 — GEC engine
- [ ] Phase 8 — Full pipeline integration + Flutter/FastAPI UI
