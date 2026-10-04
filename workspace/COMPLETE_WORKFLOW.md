# EduHinglish — Complete End-to-End Development & Training Workflow

> **Overview:** This document provides the complete, step-by-step workflow for the EduHinglish system: from raw dataset collection to preprocessing, spaCy baseline training, MuRIL transformer fine-tuning, RAG vector indexing, IndicBART bidirectional translator fine-tuning, Gemma-3-4B-it QLoRA QA fine-tuning, and full end-to-end pipeline integration.

---

## Architecture Flowchart

```
Student Query (Hinglish / Hindi / English)
         │
         ▼
[M1] MuRIL Input Processor
  ├── Word-level LID (7 tags: HI, EN, NE, UNIV, MIX, NUM, SYM)
  └── Query Intent Classifier (12 classes)
         │
         ▼
[M2] ChromaDB RAG Retriever
  ├── Multilingual MiniLM Embedder
  └── Filtered retrieval by Subject & Class
         │
         ▼
[M3-A] IndicBART Translator (Hinglish → English)
  └── Converts student query to clean English
         │
         ▼
[M3-B] Gemma-3-4B-it QA Engine (NCERT-Grounded English QA)
  └── Generates grounded, intent-styled English response
         │
         ▼
[M3-C] IndicBART Translator (English → Hinglish)
  └── Translates English response into natural Roman Hinglish
         │
         ▼
Final Hinglish Response (Sent to M5 UI)
```

---

## Phase 1 — Dataset Creation & Auto-Labeling

### 1.1 Objective
Convert raw NCERT Excel QA sheets (Class 9 & 10 across Biology, History, Civics, Geography, Physics, Chemistry) into a unified, clean JSON dataset enriched with intent labels, chapter metadata, and ground-truth Hinglish translations.

### 1.2 Unified Dataset Schema (`data/cbse_qa_dataset_labeled.json`)
```json
{
  "id": "c09_hist_ch01_001",
  "metadata": {
    "class": "9",
    "subject": "History",
    "chapter_num": "ch01",
    "chapter_title": "The French Revolution"
  },
  "intent": "explain_concept",
  "question_english": "What were the main causes of the French Revolution?",
  "question_hinglish": "French Revolution ke main causes kya the?",
  "context_english": "In 1789, France was divided into three estates...",
  "answer_english": "The main causes of the French Revolution were financial crisis, heavy taxation on the Third Estate, and food shortages.",
  "answer_hinglish": "French Revolution ke main causes financial crisis, Third Estate par heavy taxation, aur food shortages the."
}
```

### 1.3 Execution Steps
1. Place raw Excel files in `data/raw/`.
2. Run data conversion script:
   ```bash
   python training/convert_excel_to_json.py \
     --input_dir data/raw/ \
     --output_file data/cbse_qa_dataset.json
   ```
3. Run automated intent classification script:
   ```bash
   python training/auto_label_intents.py \
     --input_file data/cbse_qa_dataset.json \
     --output_file data/cbse_qa_dataset_labeled.json
   ```

---

## Phase 2 — Data Preprocessing & Annotation Formatting

### 2.1 Objective
Format the raw dataset into training formats for spaCy (DocBin), HuggingFace Datasets (PyTorch), seq2seq translation, and Gemma instruction tuning.

### 2.2 LID Tagset & Intent Class Taxonomies

#### Word-Level LID Tags (7 classes)
| Tag | Meaning | Example |
|---|---|---|
| `HI` | Hindi word (Roman script) | *hai, hota, samjhao* |
| `EN` | English word | *photosynthesis, cell, causes* |
| `NE` | Named Entity / Proper Noun | *Newton, France, India* |
| `UNIV` | Universal (Numbers, symbols) | *3rd, 1789, %, =* |
| `MIX` | Mixed Hindi-English morph | *testing-waala* |
| `NUM` | Numerical value | *25, 3.14* |
| `SYM` | Mathematical/Scientific Symbol | *+, ->, H2O* |

#### Query Intent Taxonomies (12 classes)
`explain_concept`, `definition`, `compare_concepts`, `solve_numerical`, `chemical_reaction`, `proof_derivation`, `formula_request`, `diagram_description`, `list_enumerate`, `give_example`, `application_reallife`, `revision_summary`

### 2.3 Execution Scripts
- **spaCy DocBin converter:**
  ```bash
  python training/prepare_spacy_data.py \
    --input_file data/cbse_qa_dataset_labeled.json \
    --output_dir data/processed/spacy/
  ```
- **HuggingFace Dataset converter:**
  ```bash
  python training/prepare_hf_data.py \
    --input_file data/cbse_qa_dataset_labeled.json \
    --output_dir data/processed/hf/
  ```

---

## Phase 3 — spaCy Model Training (Fast Baseline)

### 3.1 Objective
Train CPU-lightweight spaCy pipeline components for fast local testing and fallback classification.

### 3.2 Scripts & Commands
1. **Train spaCy Language Identification (LID):**
   ```bash
   python training/train_lid.py \
     --train_path data/processed/spacy/train_lid.spacy \
     --dev_path data/processed/spacy/dev_lid.spacy \
     --output_dir models/spacy_lid_v1
   ```
2. **Train spaCy Intent Classifier:**
   ```bash
   python training/train_intent.py \
     --train_path data/processed/spacy/train_intent.spacy \
     --dev_path data/processed/spacy/dev_intent.spacy \
     --output_dir models/spacy_intent_v1
   ```
3. **Train spaCy Named Entity Recognizer (NER):**
   ```bash
   python training/train_ner.py \
     --train_path data/processed/spacy/train_ner.spacy \
     --dev_path data/processed/spacy/dev_ner.spacy \
     --output_dir models/spacy_ner_v1
   ```

---

## Phase 4 — MuRIL Transformer Fine-Tuning

### 4.1 Objective
Fine-tune Google's `google/muril-base-cased` for high-accuracy token-level Language Identification (LID) and 12-class Query Intent Classification using Low-Rank Adaptation (LoRA / PEFT).

### 4.2 Fine-Tuning Specifications
- **Base Model:** `google/muril-base-cased` (236M parameters)
- **PEFT/LoRA Config:** `r=16, alpha=32, target_modules=["query", "value"]`
- **Hardware:** Google Colab T4 (Free tier, 16GB VRAM)

### 4.3 Execution Steps
1. **Fine-tune MuRIL Token-Level LID:**
   ```bash
   python training/train_muril_lid.py \
     --data_dir data/processed/hf/lid/ \
     --output_dir models/muril_lid_v3 \
     --epochs 5 \
     --batch_size 16 \
     --lr 3e-4
   ```
2. **Fine-tune MuRIL Intent Classifier:**
   ```bash
   python training/train_muril_intent.py \
     --data_dir data/processed/hf/intent/ \
     --output_dir models/muril_intent_v3 \
     --epochs 5 \
     --batch_size 16 \
     --lr 3e-4
   ```

---

## Phase 5 — RAG Vector Database Indexing

### 5.1 Objective
Index NCERT textbook chapters into ChromaDB vector database with multi-field metadata filtering (`subject`, `class`).

### 5.2 Chunking & Indexing Pipeline
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (118M params)
- **Chunk Size:** 300 words with 50-word overlap

### 5.3 Execution Commands
```bash
# Process raw text into chunked JSON
python src/chunker.py --input_dir NCERT_Class\ 9/ --output_file data/processed/chunks_class9.json

# Embed and index into ChromaDB
python src/embedder.py --chunks_file data/processed/chunks_class9.json --db_path data/chroma_db/
```

---

## Phase 6 — IndicBART Bidirectional Translator Fine-Tuning

### 6.1 Objective
Fine-tune `ai4bharat/IndicBART` (244M params) for bidirectional translation between Roman Hinglish and formal English (`Hinglish → English` and `English → Hinglish`).

### 6.2 Execution Steps
1. **Prepare Seq2Seq Translation Dataset:**
   ```bash
   python training/prepare_translation_data.py \
     --input_file data/cbse_qa_dataset_labeled.json \
     --output_dir data/processed/translation/
   ```
2. **Fine-tune IndicBART Translator:**
   ```bash
   python training/train_indicbart_translator.py \
     --data_dir data/processed/translation/ \
     --output_dir models/indicbart_translator_v1/ \
     --epochs 6 \
     --batch_size 8 \
     --lr 5e-5
   ```

---

## Phase 7 — Gemma-3-4B-it QLoRA Fine-Tuning for NCERT QA

### 7.1 Objective
Fine-tune `google/gemma-3-4b-it` (4 billion parameter model) with 4-bit QLoRA to generate grounded, high-quality NCERT answers in English given query intent and RAG context.

### 7.2 Gemma Native Chat Template Format
```
<start_of_turn>user
You are an expert NCERT tutor for Indian students (Class {class}).

Subject: {subject}
Chapter context:
{context_english}

Student's question: {question_english}
Query type: {intent}
Style instruction: {style_instruction}
<end_of_turn>
<start_of_turn>model
{answer_english}
<end_of_turn>
```

### 7.3 Training Configuration (`training/train_gemma_ncert.py`)
```python
MODEL_NAME     = "google/gemma-3-4b-it"
QUANTIZATION   = "4bit"  # bitsandbytes NF4
LORA_R         = 16
LORA_ALPHA     = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
BATCH_SIZE     = 4       # Per GPU (Colab A100 / T4)
GRAD_ACCUM     = 4       # Effective batch size = 16
MAX_SEQ_LEN    = 1024
```

### 7.4 Execution Commands
1. **Prepare Gemma Chat Dataset:**
   ```bash
   python training/prepare_gemma_data.py \
     --input_file data/cbse_qa_dataset_labeled.json \
     --output_file data/processed/gemma_chat_dataset.json
   ```
2. **Train Gemma-3-4B-it QLoRA (Run on Google Colab A100):**
   ```bash
   python training/train_gemma_ncert.py \
     --dataset_path data/processed/gemma_chat_dataset.json \
     --output_dir models/gemma_ncert_v1/ \
     --epochs 3
   ```

---

## Phase 8 — Pipeline Integration & Inference

### 8.1 Module Summary
- **`M1` (`src/muril_processor.py`)**: Predicts word LID and 12-class query intent.
- **`M2` (`src/retriever.py`)**: Retrieves top-3 NCERT chunks from ChromaDB.
- **`M3-A` (`src/translator.py`)**: Translates Hinglish query → English query.
- **`M3-B` (`src/generator.py`)**: Runs Gemma-3-4B-it QA engine on context + English query.
- **`M3-C` (`src/translator.py`)**: Translates English answer → Roman Hinglish.

### 8.2 Testing Full Pipeline Inference
```bash
python src/pipeline.py --query "Newton ka 3rd law samjhao" --subject "Physics" --class_num "9"
```

---

## Summary Checklist & Hardware Mapping

| Phase | Task | Model | Recommended GPU |
|---|---|---|---|
| **Phase 1** | Data collection & auto-labeling | Script / LLM Prompts | CPU / Local |
| **Phase 2** | Preprocessing & formatting | Python | CPU / Local |
| **Phase 3** | spaCy baseline models | spaCy | CPU / Local |
| **Phase 4** | MuRIL LID & Intent | `google/muril-base-cased` | Colab T4 (16GB VRAM) |
| **Phase 5** | Vector DB creation | `paraphrase-multilingual-MiniLM-L12-v2` | CPU / Local |
| **Phase 6** | IndicBART Translator | `ai4bharat/IndicBART` | Colab T4 (16GB VRAM) |
| **Phase 7** | Gemma NCERT QA Fine-tuning | `google/gemma-3-4b-it` | Colab Pro A100 (40GB VRAM) |
| **Phase 8** | Full Pipeline Wiring | All models integrated | Colab / T4 GPU |
