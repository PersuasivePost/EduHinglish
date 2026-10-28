# EduHinglish — Training Runbook (v3: Biology 9/10 + Social Science 9)

How to retrain every component on the current datasets, and where each piece
runs. Social Science class 10 (`data/ss10.json`) is **not** included yet — see
[Adding a new dataset](#adding-a-new-dataset-eg-ss10).

## How the components connect

```
data/bio_9_10.json, data/ss9.json            (QA datasets: question/answer/intent/context)
data/processed/*/dataset.json                (legacy annotated sentences: gold word-level LID)
        │
        │  scripts/build_unified_dataset.py
        ▼
data/unified_dataset_v3.json ───┬── training/prepare_spacy_data.py ──► training/data/*_v3.spacy ──► train_{lid,intent,ner}.py ──► models/{lid,intent,ner}_v3   (local, CPU)
                                └── training/prepare_hf_data.py ────► training/data/muril_*_dataset_v3 ──► Colab 01/02 ──► models/muril_{lid,intent}_v3

data/processed/<chapter>/cleaned_text.txt
        │  scripts/build_knowledge_base.py  (src/chunker.py → src/embedder.py → src/retriever.py)
        ▼
src/knowledge_base/  (ChromaDB, collection "ncert_textbooks")                                                                (local, CPU)

data/bio_9_10.json, data/ss9.json ──► Colab 04 (Gemma-3-4B-it QLoRA) ──► gemma_eduhinglish_v2/adapter
```

At inference the pieces line up like this:

1. **M1:** MuRIL LID tags words. MuRIL intent predicts one of the 15 intents.
2. **M2:** the ChromaDB retriever returns NCERT chunks.
3. **M3:** Gemma receives `Context` (the chunks), `Question`, and `Intent Type: <predicted intent>`.

The intent labels must match the `intent` field in the QA datasets, which is
what Gemma was trained on. All label sets live in `training/label_config.py`.

> Current state of `src/pipeline.py`:
> - It still uses rule-based LID (`script_detector.py`) and the IndicBART generator.
> - It does not load the trained spaCy/MuRIL models or Gemma yet.
> - Connecting them is a separate task.

## 0. Local setup (once per machine)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough locally
```

## 1. Build the unified dataset

```bash
python scripts/build_unified_dataset.py --silver-lid
```

- The **intent** examples come from every QA question: `question_hinglish`, or `question_english` when the Hinglish one is empty. That gives about 5,500 queries.
- The **LID** gold labels come from the 1,700 legacy annotated sentences.
  - `--silver-lid` also auto-tags the QA questions with `models/lid_v2`.
  - Silver data is used **only for training**, so dev accuracy is still measured on gold labels.
  - Drop the flag to train on gold labels only, and compare the two runs.

## 2. spaCy models (local, CPU)

```bash
python training/prepare_spacy_data.py
python training/train_lid.py          # → models/lid_v3
python training/train_intent.py       # → models/intent_v3
python training/train_ner.py          # → models/ner_v3   (now includes PERSON/PLACE/CIVILISATION/INSTITUTION)
python training/evaluate_models.py
```

## 3. MuRIL models (prepare locally → train on Colab)

```bash
python training/prepare_hf_data.py    # → training/data/muril_{lid,intent}_dataset_v3/
```

1. Upload both folders to Drive (layout below).
2. Run `training/colab_notebooks/01_train_muril_lid.ipynb` and `02_train_muril_intent.ipynb` on a T4.
3. Download `models/muril_lid_v3` and `models/muril_intent_v3` from Drive into the local `models/` folder.
4. Run `python training/evaluate_muril.py`.

The notebooks load the prepared dataset and `label_maps.json` directly, so they
contain no label lists of their own.

## 4. RAG knowledge base (local, CPU, ~1 min)

```bash
python scripts/build_knowledge_base.py --force
python src/retriever.py               # sample Hinglish queries
```

`src/knowledge_base/` is gitignored, so **every machine has to build it**.
Only the chapters listed in `NCERTChunker.CHAPTER_MAP` (`src/chunker.py`) are
indexed. That is currently 9 biology chapters and 9 ss9 chapters, giving 828 chunks.

## 5. Gemma (Colab)

1. Upload `data/bio_9_10.json` and `data/ss9.json` to `MyDrive/EduHinglish/data/`.
2. Run `training/colab_notebooks/04_gemma_finetune.ipynb`. It needs the `HF_TOKEN` secret, and you must accept the Gemma licence on Hugging Face.
3. The adapter is saved to `MyDrive/EduHinglish/models/gemma_eduhinglish_v2/adapter`.

## Google Drive layout

```
MyDrive/EduHinglish/
├── data/                              ← Gemma input (upload from local data/)
│   ├── bio_9_10.json
│   └── ss9.json
├── training_data/                     ← MuRIL input (upload from local training/data/)
│   ├── muril_lid_dataset_v3/
│   └── muril_intent_dataset_v3/
└── models/
    ├── muril_lid_v2/                  ← optional: previous model, used as warm start if present
    ├── muril_intent_v2/               ← optional: previous model, used as warm start if present
    ├── muril_lid_v3/                  ← written by notebook 01
    ├── muril_intent_v3/               ← written by notebook 02
    └── gemma_eduhinglish_v2/
        ├── checkpoints/               ← written by notebook 04 during training
        └── adapter/                   ← final LoRA adapter
```

If the old MuRIL models are still on Drive under `muril_models/muril_*_v2`,
move them to `models/` to use them as a warm start. Otherwise the notebooks
start from `google/muril-base-cased`.

## Adding a new dataset (e.g. ss10)

1. Add `"data/ss10.json"` to `QA_SOURCES` in `scripts/build_unified_dataset.py`.
2. Add any new intent names to `INTENT_LABELS` in `training/label_config.py`.
3. Add the `ss10_*` folders to `CHAPTER_MAP` in `src/chunker.py`.
4. Add `"ss10.json"` to `DATA_FILES` in notebook 04.
5. Optionally, add class 10 terms to the NER gazetteers in `training/prepare_spacy_data.py`.
6. Re-run steps 1–5.
