# EduHinglish Test Commands Guide (Phases 1-5)

This document contains all the commands necessary to test, evaluate, and interact with the completed modules of the EduHinglish project (Phases 1 through 5). Make sure your virtual environment (`eduHinglish_env`) is activated before running these commands.

---

## Phase 1 & 2: Preprocessing, Script Detection, and Dataset
These commands test the core NLP text processing pipeline and validate the structural integrity of your JSON datasets.

### Automated Unit Tests
```bash
pytest tests/test_preprocessing.py
pytest tests/test_script_detector.py
pytest tests/test_pipeline.py
```

### Manual Pipeline Test
Run the main pipeline module to see a live demonstration of script detection, transliteration, normalization, and baseline LID:
```bash
python src/pipeline.py
```

### Dataset Validation
Verify that your expanded 950-entry JSON dataset is structurally valid and check the Code-Mixing Index (CMI) stats:
```bash
python scripts/validate_dataset.py
```

---

## Phase 3: spaCy Model Baselines
These commands evaluate the lightweight, CPU-friendly spaCy models for LID, NER, and Intent classification.

### Full Evaluation
Run the automated evaluation suite against your test datasets:
```bash
python training/evaluate_models.py
```

### Interactive REPL Mode
Type custom sentences and see the spaCy models process them live:
```bash
python training/test_models_quick.py --model all      # Test LID, NER, and Intent together
python training/test_models_quick.py --model lid      # Test only LID
python training/test_models_quick.py --model ner      # Test only NER
python training/test_models_quick.py --model intent   # Test only Intent
```

### Quick One-Shot CLI Test
```bash
python training/test_models_quick.py --model all --text "Sir, chloroplast ka function kya hai?"
```

### Direct Python Execution (Importing Baseline Model)
```bash
python -c "
import spacy
nlp = spacy.load('models/lid_v1')
doc = nlp('Nucleus cell ka control centre hai')
for tok in doc:
    print(f'{tok.text:15} → {tok.tag_}')
"
```

---

## Phase 4: MuRIL Fine-Tuned Models
These commands evaluate the high-accuracy transformer (MuRIL) models adapted via LoRA for token classification (LID) and sequence classification (Intent).

### Full Evaluation
Compare the fine-tuned MuRIL performance against the Phase 3 spaCy baselines:
```bash
python training/evaluate_muril.py
```

### Targeted Evaluation
```bash
python training/evaluate_muril.py --lid-only
python training/evaluate_muril.py --intent-only
```

---

## Phase 5: NCERT Knowledge Base & Retriever (M2)
These commands test the Retrieval-Augmented Generation (RAG) vector database and the semantic similarity models.

### Automated Unit Tests
Test the ChromaDB retrieval logic and chunking functions:
```bash
pytest tests/test_retriever.py
```

### Embedder Similarity Test
Run the embedder's built-in test to verify that `paraphrase-multilingual-MiniLM-L12-v2` successfully maps Hinglish and English semantic equivalents to the same vector space:
```bash
python src/embedder.py
```

### Retriever Query Test
Run the retriever's built-in test to query the actual ChromaDB `ncert_biology` collection and see the top-3 retrieved NCERT chunks for hardcoded student queries:
```bash
python src/retriever.py
```

### Rebuild Knowledge Base
*(Only run this if you have modified the NCERT PDFs or chunking logic)*
Delete the existing ChromaDB collection and rebuild it from scratch:
```bash
python scripts/build_knowledge_base.py --force
```
