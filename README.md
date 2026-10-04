# EduHinglish 🎓

### AI-Powered Learning Companion for Hinglish-Speaking Indian Students

> **BTech Project** | K.J. Somaiya College of Engineering, Mumbai
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
| **M3: Hinglish Generator** | Fine-tuned IndicBART / Gemma-3-4b-it seq2seq Hinglish explanations     | 📋 In Progress |
| **M4: GEC Engine**         | Code-mix-aware grammar error correction                                | 📋 Planned     |
| **M5: Web Interface**      | Streamlit / Flutter UI with FastAPI backend                            | 📋 Planned     |

---

## Progress Roadmap

### Phase 1 & 2 — Dataset Foundation (Scaled Up) 🚀

- **Scope Expansion:** Initially 10 chapters, now scaling to **Class 6-12 all subjects** (starting with Class 9 & 10 Social Science and Biology).
- **Automated Generation Pipeline:**
  - Extracts raw passages from NCERT PDFs.
  - Automatically generates structured English QA pairs using heuristic intent templates.
  - Asynchronously enriches the dataset with natural Hinglish translations (`question_hinglish` and `answer_hinglish`) using the **Gemini API** with robust rate-limit handling and checkpointing.
- **Output:** Large-scale, high-quality JSON datasets (e.g., `data/ss9.json` with 4,200+ entries) for comprehensive model fine-tuning.

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
- **Coverage:** Indexed Science and Social Science chapters.

---

## Current Status: Phase 6 — Hinglish Answer Generation 🚀

We are currently working on **M3**, where the retrieved NCERT English context is transformed into natural Hinglish explanations. 
**New Plan:** We are leveraging our massively scaled QA dataset to fine-tune **IndicBART** and **Gemma-3-4b-it** using PEFT/LoRA to achieve state-of-the-art Hinglish generation capabilities.

---

## Pipeline (Phase 1 — M1 Input Processing)

```text
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
4. **No Hinglish Educational Dataset** — Creating a massive novel labeled dataset as part of this project
5. **LLM Hallucination** — RAG architecture grounds all answers in NCERT
6. **No Pedagogical Scaffolding** — LLM prompted to follow Indian teaching style
7. **No Evaluation Framework** — Defining custom metrics (CMI, comprehension tests)

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
| M2     | Vector DB         | ChromaDB (dev) / Pinecone (production) | Fully local in development; Pinecone provides managed vector search in production                                        |
| M2     | Re-ranker         | cross-encoder/ms-marco-MiniLM-L-12-v2  | Lightweight cross-encoder; improves precision@3 significantly                                                            |
| M3     | Data Enrichment   | Gemini API                             | Asynchronous LLM generation for scaling Hinglish translation datasets.                                                   |
| M3     | Generator         | IndicBART / Gemma-3-4b-it              | Fine-tuning advanced LLMs for state-of-the-art native Hinglish output                                                    |
| M3     | Fine-tuning       | LoRA via PEFT (HuggingFace)            | Efficient fine-tuning on Colab T4 GPU                                                                                    |
| M4     | English GEC       | GECToR (BEA-2019 + Lang-8)             | Best English GEC F1; token edit-based, not seq2seq                                                                       |
| M4     | Hindi GEC         | Rule-based (gender/verb agreement)     | Handles 3 most common Hinglish grammar error types                                                                       |
| M5     | UI Prototype      | Flutter                                | Rapid iteration; minimal JS; good for demos                                                                              |
| M5     | UI Production     | Flutter + FastAPI + Pydantic           | Cross-platform (Android/iOS/Web) from single codebase; Pydantic enforces API schema and validates LLM structured outputs |
| All    | Runtime           | Python 3.10, PyTorch 2.1               | Standard ML stack; all dependencies open-source                                                                          |

---

## Roadmap

- [x] Literature survey (11 papers)
- [x] Market analysis & gap identification
- [x] System architecture design
- [x] Phase 1 — M1 Preprocessing pipeline
- [x] Phase 2 — Multi-chapter dataset pipeline (Scaled up to automated pipeline, targeting Class 6-12)
- [x] Phase 3 — spaCy model training (LID, NER, Intent)
- [x] Phase 4 — MuRIL fine-tuning (Colab T4)
- [x] Phase 5 — NCERT ChromaDB knowledge base
- [ ] Phase 6 — IndicBART & Gemma-3-4b-it Hinglish generator fine-tuning
- [ ] Phase 7 — GEC engine
- [ ] Phase 8 — Full pipeline integration + Flutter/FastAPI UI
