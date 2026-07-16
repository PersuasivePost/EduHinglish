Let me re-read the ly.md objectives and current implementation carefully before giving the revised plan.

Looking at your full picture:

**Current state:** 15 manually labeled sentences for Ch5, working preprocessing pipeline (M1 nearly done), rule-based LID.

**Goals from ly.md:** 5 modules — M1 preprocessing, M2 NCERT Retriever (RAG), M3 Hinglish Generator, M4 GEC Engine, M5 Streamlit UI. No external APIs, everything local/Colab.

**Hardware reality:** Colab/Kaggle free tier gives you T4 (16GB VRAM) — enough for MuRIL, IndicBART, spaCy, sentence-transformers. Not enough for LLaMA 7B full fine-tune, but LoRA/QLoRA works.

---

## Revised Complete Plan — 100% Local, No External APIs

---

### Phase 1 — Dataset Expansion (Week 1–2)
**Goal:** 75+ labeled sentences for Ch5, establish the template for all future chapters.

**Strategy — same as your current 15 sentences, but with augmentation assist:**

Three types of entries you need more of:
- Factual statements (like your current sentences 1–12)
- Student queries with intent tags (like 13–17)
- GEC samples with error annotations (like 18–20)

**Files to build:**
```
scripts/
├── augment_dataset.py        ← rule-based variants from existing sentences
├── annotation_helper.py      ← CLI tool to speed up manual labeling
└── validate_dataset.py       ← check label consistency, CMI stats
```

**augment_dataset.py** — takes an existing labeled sentence and generates variants by:
- Swapping Hindi connectors (aur↔lekin, hai↔hota hai)
- Changing question form (statement → student query)
- Adding/removing code-mixing density (more Hindi ↔ more English)
- Producing ~3-4 variants per original sentence → 15 sentences × 4 = ~60 variants to verify

**annotation_helper.py** — prints each word and asks you to type HI/EN/NE/UNIV, saves to JSON automatically. Much faster than editing JSON by hand.

**Target output:** `data/biology/class9/ch05/dataset_v2.json` with 75+ entries.

---

### Phase 2 — Multi-Chapter Dataset Pipeline (Week 2–4)
**Goal:** 50–75 sentences per chapter, all 11 biology chapters.

**Class 9 Biology:** Ch5 *(done)*, Ch6 Tissues, Ch7 Diversity in Living Organisms, Ch13 Why Do We Fall Ill, Ch14 Natural Resources, Ch15 Improvement in Food Resources

**Class 10 Biology:** Ch6 Life Processes, Ch8 How do Organisms Reproduce, Ch9 Heredity and Evolution, Ch15 Our Environment, Ch16 Management of Natural Resources

**Files to build:**
```
scripts/
├── batch_pdf_extractor.py     ← process all chapter PDFs at once
├── sentence_sampler.py        ← extract key sentences from cleaned text
│                                 (rule-based: longest sentences, sentences
│                                  with key biology terms, question sentences)
└── chapter_dataset_builder.py ← guided CLI: shows English sentence,
                                  you type Hinglish version, auto-labels
                                  common Hindi words, you verify the rest
data/
└── biology/
    ├── class9/
    │   ├── ch05/  ch06/  ch07/  ch13/  ch14/  ch15/
    └── class10/
        ├── ch06/  ch08/  ch09/  ch15/  ch16/
```

**chapter_dataset_builder.py workflow:**
```
[NCERT English] → you type Hinglish → auto-labels EN/HI with your 
current WordLevelLID rules → shows predicted labels → you correct 
wrong ones → saves to JSON
```

This is the fastest path to 550–750 labeled sentences (11 chapters × 50–75) without any LLM API.

**Unified dataset:** `data/unified_biology_dataset_v2.json` — all chapters merged with metadata.

---

### Phase 3 — spaCy Model Training (Week 4–5)
**Goal:** Replace rule-based LID/intent with trained models. Runs entirely on CPU or Colab T4.

**Three spaCy models:**

**3a. Token Classifier for LID**
- Architecture: `tok2vec` + token classifier head (spaCy's built-in)
- Input: tokenized Hinglish sentence
- Output: HI / EN / NE / UNIV per token
- Training data: `word_level_labels` from your entire dataset
- Estimated training time: ~10–15 min on Colab T4 with 500+ sentences
- Expected accuracy: 85–90% (rule-based baseline is ~75%)

**3b. Named Entity Recognizer (Science NER)**
- Labels: ORGANELLE, PROCESS, SCIENTIST, INSTRUMENT, CONCEPT
- Training data: annotate science terms from your dataset
- Helps M2 retriever understand what the student is asking about
- Architecture: spaCy `ner` pipeline component

**3c. Text Classifier for Intent**
- Labels: explain_concept, compare_concepts, give_example, formula_request, definition, other
- Training data: your `intent` field from student query entries (~60–80 queries across all chapters)
- Architecture: spaCy `textcat` (BOW for small data, CNN if 200+ examples)

**Files to build:**
```
training/
├── prepare_spacy_data.py      ← JSON dataset → .spacy binary format
├── configs/
│   ├── lid_config.cfg         ← spaCy config for token classifier
│   ├── ner_config.cfg
│   └── intent_config.cfg
├── train_lid.py               ← training script with eval loop
├── train_ner.py
├── train_intent.py
└── evaluate_models.py         ← accuracy, F1, confusion matrix
models/
├── lid_v2/                    ← saved spaCy pipeline
├── ner_v2/
└── intent_v2/
```

**Integration:** Replace `WordLevelLID` class in `script_detector.py` with spaCy model call. Add intent detection to `pipeline.py`.

---

### Phase 4 — MuRIL Fine-tuning for LID (Week 5–6)
**Goal:** Get production-grade LID accuracy using Google's MuRIL, fine-tuned on your dataset. Runs on Colab T4.

**Why MuRIL over spaCy tok2vec:**
- Pre-trained on 17 Indian languages including Hindi in both Roman and Devanagari
- Understands Hinglish natively without rule-based shortcuts
- Fine-tuning needs only 500+ labeled tokens to outperform rules significantly
- Model size: 236M params, fits easily in T4 16GB

**Training approach:**
- HuggingFace `transformers` + `datasets`
- Token classification head on top of `google/muril-base-cased`
- LoRA fine-tuning (PEFT) — reduces trainable params from 236M to ~2M
- Batch size 16, ~5 epochs, ~20 min on T4
- Save to `models/muril_lid_v2/`

**Files to build:**
```
training/
├── prepare_hf_data.py         ← dataset → HuggingFace Dataset format
├── train_muril_lid.py         ← PEFT/LoRA fine-tuning on T4
├── train_muril_intent.py      ← intent classifier on MuRIL [CLS] token
└── colab_notebooks/
    ├── 01_train_muril_lid.ipynb    ← ready to run on Colab
    └── 02_train_muril_intent.ipynb
```

**Output:** A model that correctly tags "mein" (HI), "mitochondria" (EN), "Robert Hooke" (NE) with 92%+ accuracy.

---

### Phase 5 — NCERT Knowledge Base / RAG (Week 6–7)
**Goal:** Store all NCERT content as searchable vectors so the system retrieves the right paragraph for any student query. No internet needed at runtime.

**Embedding model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 118M params, handles Hindi+English mixed sentences natively
- Runs fast on CPU, faster on T4
- Free, local, no API

**Vector DB:** ChromaDB — runs fully local as a file on disk.

**Chunking strategy:**
```
Full chapter text
    → paragraph-level chunks (~150–200 words)
    → each chunk tagged with: subject, class, chapter_num, chapter_title, topic
    → embed each chunk
    → store in ChromaDB collection "ncert_biology"
```

**Files to build:**
```
src/
├── chunker.py                 ← paragraph splitter with metadata tagging
├── embedder.py                ← sentence-transformer wrapper
├── retriever.py               ← ChromaDB interface, top-k search
└── knowledge_base/            ← ChromaDB persisted on disk
scripts/
└── build_knowledge_base.py    ← one-time script: all PDFs → ChromaDB
```

**Query flow:** Student asks "Sir mitochondria ka kaam kya hai?" → M1 pipeline preprocesses → MuRIL extracts key terms (mitochondria, kaam/function) → retriever finds top-3 NCERT chunks about mitochondria → passes to generator.

---

### Phase 6 — Hinglish Answer Generator Fine-tuning (Week 7–9)
**Goal:** Fine-tune a small seq2seq model to generate natural Hinglish explanations from NCERT English text. Runs on Colab T4.

**Model: IndicBART** (`ai4bharat/IndicBART`)
- 244M params, seq2seq, trained on 11 Indic languages + English
- Handles Hindi and English natively, understands Hinglish
- Fine-tune on T4 in ~30–45 min per chapter set
- Input: NCERT English paragraph + student query
- Output: Hinglish explanation

**Training data construction:**
- Source: `original_english` field from your dataset
- Target: `hinglish_roman` field from your dataset
- Augmented with: 3–4 reference Hinglish explanations per chapter topic you write manually
- ~500–700 (source, target) pairs across all chapters

**Alternative if IndicBART quality is insufficient:**
- mT5-small (300M params) — multilingual T5, same approach
- Both fit in T4, both are completely local

**Files to build:**
```
training/
├── prepare_seq2seq_data.py    ← (english_text, hinglish_output) pairs
├── train_indicbart.py         ← seq2seq fine-tuning with HuggingFace Trainer
└── colab_notebooks/
    └── 03_train_indicbart.ipynb
src/
└── generator.py               ← loads fine-tuned IndicBART, generates answer
                                   given retrieved NCERT chunk + processed query
```

---

### Phase 7 — GEC Engine (Week 9–10)
**Goal:** Detect and correct grammar errors in Hinglish student input (M4).

**Approach — two-stage:**

**Stage 1: Error Detector** (spaCy classifier)
- Uses your existing GEC samples (entries 18–20 and more you'll add)
- Detects: gender agreement errors, tense errors, subject-verb agreement
- Binary classifier per token: CORRECT / ERROR

**Stage 2: Error Corrector** (rule-based + IndicBART)
- For common error patterns: rule-based corrections (gender agreement table)
- For complex errors: IndicBART seq2seq correction (fine-tuned on your error→correct pairs)

**Files to build:**
```
src/gec/
├── error_detector.py          ← spaCy token classifier
└── error_corrector.py         ← rule-based + seq2seq correction
training/
└── train_gec.py
```

---

### Phase 8 — Full Pipeline Integration + Streamlit (Week 10–11)
**Goal:** Working end-to-end demo.

**Complete query flow:**
```
Student types: "Sir mitochondria ka kaam kya hai?"
    ↓ M1: Script detection → Normalization → MuRIL LID
    ↓ M1: Intent = explain_concept | Key term = mitochondria
    ↓ M4: GEC check (no errors)
    ↓ M2: ChromaDB retrieves top-3 NCERT chunks on mitochondria
    ↓ M3: IndicBART generates Hinglish explanation
    ↓ M5: Displayed in Streamlit chat with step-by-step trace
```

**Files to build:**
```
src/
└── answer_pipeline.py         ← orchestrates M1→M4→M2→M3
app/
└── streamlit_app.py           ← chat UI with pipeline trace sidebar
```

---

### Revised Week-by-Week Timeline

```
Week 1:   Phase 1  — Expand Ch5 to 75 sentences + augmentation scripts
Week 2:   Phase 2a — Process Ch6, Ch7 (Class 9) — 50 sentences each
Week 3:   Phase 2b — Process Ch13, Ch14, Ch15 (Class 9)
Week 4:   Phase 2c — Process Class 10 chapters (Ch6, Ch8, Ch9, Ch15, Ch16)
Week 5:   Phase 3  — Train all 3 spaCy models (LID, NER, Intent)
Week 6:   Phase 4  — Fine-tune MuRIL LID on Colab
Week 7:   Phase 5  — Build ChromaDB knowledge base
Week 8:   Phase 6  — Fine-tune IndicBART on Colab
Week 9:   Phase 7  — GEC engine
Week 10:  Phase 8a — Full pipeline integration
Week 11:  Phase 8b — Streamlit UI + presentation prep
```

---

### Model Stack Summary (100% Local, No APIs)

| Task | Model | Size | Runs On |
|------|-------|------|---------|
| LID (fast) | spaCy tok2vec | ~15MB | CPU |
| LID (accurate) | MuRIL + LoRA | 236M | Colab T4 |
| NER | spaCy ner | ~20MB | CPU |
| Intent | spaCy textcat | ~10MB | CPU |
| Embeddings | multilingual-MiniLM | 118M | CPU/T4 |
| Generation | IndicBART fine-tuned | 244M | Colab T4 |
| GEC | spaCy + rules | ~15MB | CPU |
| Vector DB | ChromaDB (local) | on-disk | CPU |

---

When you say go, tell me which phase to start with and I'll give you the exact Claude Code prompts for each file.