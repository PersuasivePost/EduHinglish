# EduHinglish 🎓

### AI-Powered Learning Companion for Hinglish-Speaking Indian Students

> **Final Year Engineering Project** | K.J. Somaiya College of Engineering, Mumbai  
> **Team:** Ashvatth & Jatin | **Batch:** 2027 | **Honours:** Mobile & Software Application Development

---

## The Problem

350–400 million people in India communicate in **Hinglish** (Hindi-English code-mix) as their primary informal language. Yet every NCERT textbook, every digital EdTech platform, and every AI tutoring tool forces students to choose — formal English OR formal Hindi.

A student who naturally asks _"Sir, Newton ka third law samjhao please"_ has no tool that understands that query and returns a curriculum-grounded Hinglish explanation.

**EduHinglish bridges that gap.**

---

## What We're Building

A 5-module integrated AI system:

| Module                     | What It Does                                                          | Status         |
| -------------------------- | --------------------------------------------------------------------- | -------------- |
| **M1: Input Processing**   | Script detection → Normalization → Word-level LID → Intent extraction | 🔨 In Progress |
| **M2: NCERT Retriever**    | RAG pipeline over NCERT textbook chunks                               | 📋 Planned     |
| **M3: Hinglish Generator** | LLM-powered Hinglish explanations (LLaMA 3.1 8B)                      | 📋 Planned     |
| **M4: GEC Engine**         | Code-mix-aware grammar error correction                               | 📋 Planned     |
| **M5: Web Interface**      | Streamlit chat UI                                                     | 📋 Planned     |

---

## Current Phase: Preprocessing Pipeline

**Scope:** NCERT Class 9 Science — Chapter 5: _The Fundamental Unit of Life_ (Biology)

### Pipeline Steps

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
│   ├── raw/                          # NCERT PDFs (not committed to git)
│   ├── processed/                    # Extracted & cleaned text
│   └── hinglish/                     # Labeled Hinglish dataset
│
├── src/
│   ├── pdf_extractor.py              # NCERT PDF → clean text (Ashvatth)
│   ├── preprocessing.py              # English NLP pipeline (Ashvatth)
│   ├── script_detector.py            # Script detection + LID (Jatin)
│   ├── normalizer.py                 # Hinglish normalization (Jatin)
│   ├── hinglish_dataset_creator.py   # Labeled dataset creation (Jatin)
│   └── pipeline.py                   # Unified pipeline (Both)
│
├── notebooks/
│   ├── 01_pdf_extraction.ipynb
│   ├── 02_preprocessing_english.ipynb
│   ├── 03_hinglish_dataset.ipynb
│   ├── 04_script_and_lid.ipynb
│   └── 05_full_pipeline_demo.ipynb
│
├── outputs/
│   └── pipeline_results/             # All JSON outputs saved here
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_script_detector.py
│   └── test_pipeline.py
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
git clone https://github.com/YOUR_USERNAME/EduHinglish.git
cd EduHinglish

# Create virtual environment with Python 3.10
py -3.10 -m venv eduhinglish_env          # Windows
# python3.10 -m venv eduhinglish_env      # Mac/Linux

# Activate
eduhinglish_env\Scripts\activate           # Windows
# source eduhinglish_env/bin/activate      # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python scripts/download_nltk_data.py

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Get the NCERT PDF

Download NCERT Class 9 Science from [ncert.nic.in](https://ncert.nic.in/textbook.php)  
Place it at: `data/raw/ncert_class9_science.pdf`

---

## Running the Pipeline

```bash
cd src/

# Step 1: Extract text from NCERT PDF (Ashvatth)
python pdf_extractor.py

# Step 2: Create Hinglish dataset (Jatin)
python hinglish_dataset_creator.py

# Step 3: Test individual modules
python preprocessing.py
python script_detector.py

# Step 4: Run the full unified pipeline
python pipeline.py
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

| Layer           | Technology                            |
| --------------- | ------------------------------------- |
| NLP             | NLTK, spaCy, HuggingFace Transformers |
| LID Model       | MuRIL (Google) + Token Classification |
| Transliteration | IndicXlit (AI4Bharat)                 |
| Vector DB       | ChromaDB                              |
| Embeddings      | BGE-base-en-v1.5                      |
| LLM (later)     | LLaMA 3.1 8B (QLoRA fine-tuned)       |
| Fine-tuning     | PEFT + QLoRA (4-bit)                  |
| UI (later)      | Streamlit → React + FastAPI           |
| Language        | Python 3.10                           |

---

## Team

| Name     | Role                                                                   |
| -------- | ---------------------------------------------------------------------- |
| Ashvatth | PDF extraction, English preprocessing pipeline, unified pipeline class |
| Jatin    | Hinglish dataset creation, script detection, normalization, LID module |

---

## Roadmap

- [x] Literature survey (11 papers)
- [x] Market analysis & gap identification
- [x] System architecture design
- [ ] **Preprocessing pipeline (current)**
- [ ] NCERT vector database (RAG)
- [ ] Hinglish explanation generator
- [ ] GEC module
- [ ] Web interface
- [ ] Evaluation & human study
