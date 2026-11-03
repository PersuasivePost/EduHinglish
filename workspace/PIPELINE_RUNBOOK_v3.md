# EduHinglish — Training Runbook (all subjects: Bio, Chem, Phys, Maths, SS — Class 9/10)

How to retrain every component on the current datasets, publish them, and run
the demo. The QA datasets are one file per subject, built by
`scripts/build_dataset.py`:

`data/bio_9_10.json` · `data/chem_9_10.json` · `data/phys_9_10.json` · `data/maths_9_10.json` · `data/ss_9_10.json` (3,604 records)

To add another subject, see [Adding a new dataset](#adding-a-new-dataset).

> As of 2026-11-03:
> - The **Gemma v4** run (notebook 05) and the **RAG index** cover all five subjects.
>   The v3 adapter covers only bio + ss.
> - The spaCy and MuRIL `*_v3` models were trained on bio + old ss9 data only.
>   `build_unified_dataset.py` now reads all five files, so re-run steps 1–3 to update them.
>   The output names still say `v3`.

## How the components connect

```
data/{bio,chem,phys,maths,ss}_9_10.json     (QA datasets: question/answer/intent/context)
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

data/{bio,chem,phys,maths,ss}_9_10.json ──► Colab 05 (Gemma-3-4B-it QLoRA, resumable) ──► gemma_eduhinglish_v4/adapter ──► HF Hub
                                         └─► scripts/publish_kaggle_dataset.py ──► Kaggle dataset
```

At inference the pieces line up like this:

1. **M1:** MuRIL LID tags words. MuRIL intent predicts one of the 15 intents.
2. **M2:** the ChromaDB retriever returns NCERT chunks.
3. **M3:** Gemma receives `Context` (the chunks), `Question`, and `Intent Type: <predicted intent>`.

The intent labels must match the `intent` field in the QA datasets, which is
what Gemma was trained on. All label sets live in `training/label_config.py`.

> Current state of `src/pipeline.py` (`EduHinglishPipeline(mode="full")`, `answer_query()`):
> - M1 uses rule-based script detection, normalisation and LID (`script_detector.py`), plus the MuRIL intent classifier (`src/intent_classifier.py`).
>   If the intent model can't be loaded, Gemma gets no intent line.
> - M2 is the ChromaDB retriever, optionally filtered by `subject` / `class_num`.
> - M3 is `src/generator.py`. It loads the adapter from `$EDUHINGLISH_ADAPTER` (a local folder or a Hub repo id), else `models/gemma_eduhinglish_v4/adapter`, else `v3`.
> - The intent is passed only when its confidence is ≥ 0.6.
> - MuRIL LID is not used at inference yet.

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
indexed. That is currently 83 chapters: 9 biology, 7 chemistry, 8 physics, 28 maths and 31 social science.

## 5. Gemma v4 (Colab, resumable)

1. Upload the five `data/*_9_10.json` files to `MyDrive/EduHinglish/data/`.
   Alternatively, set `DATA_SOURCE = "kaggle"` in the notebook to read them from the Kaggle dataset (step 6).
2. Open `training/colab_notebooks/05_gemma_finetune_v4_resumable.ipynb`. Set `HF_REPO_ID` in the settings cell, then **Run all**.
   - It needs the `HF_TOKEN` secret. Use a **write** token so the notebook can push to the Hub.
   - You must accept the Gemma licence on Hugging Face first.
3. **Resuming:** checkpoints go to `models/gemma_eduhinglish_v4/checkpoints/` on Drive every 50 steps.
   - After a disconnect, reconnect and **Run all** again. Training continues from the last checkpoint: same step, optimizer, scheduler and data order.
   - Once training has finished, `TRAINING_DONE` is written. Re-running then loads the final adapter and skips training.
4. The first v4 run warm-starts from the v3 adapter if it is on Drive. Set `WARM_START_ADAPTER = None` to train from the base model.
5. 2 epochs ≈ 860 steps ≈ 1.5–2 h on a T4. This can be spread over several sessions.
6. At the end, the adapter is saved to `models/gemma_eduhinglish_v4/adapter` and pushed to `HF_REPO_ID` (private).

Notebook 04 is the old v3 (bio + ss) run, kept for reference.

## 6. Publish: Hugging Face (models) and Kaggle (dataset)

```bash
pip install huggingface_hub kaggle
huggingface-cli login                       # write token

# Gemma adapter (only needed if you didn't push from notebook 05, e.g. after downloading it from Drive)
python scripts/push_to_hub.py gemma  models/gemma_eduhinglish_v4/adapter  <hf-user>/eduhinglish-gemma-3-4b-it-lora
# MuRIL intent model (the demo notebook downloads it from here)
python scripts/push_to_hub.py intent models/muril_intent_v3               <hf-user>/eduhinglish-muril-intent

# Dataset → Kaggle (needs ~/.kaggle/kaggle.json). Creates it the first time and adds a new version after that.
python scripts/publish_kaggle_dataset.py <kaggle-user>/eduhinglish-ncert-hinglish-qa --with-text
python scripts/publish_kaggle_dataset.py <kaggle-user>/eduhinglish-ncert-hinglish-qa -m "what changed"
```

Repos and datasets are **private** by default. Add `--public` to make them public.

## 7. Demo / inference on Colab (no training)

Open `training/colab_notebooks/06_inference_demo.ipynb` and set `HF_ADAPTER_REPO` / `HF_INTENT_REPO`. Then **Run all**. The notebook:

1. Clones the repo (`develop`). Add a `GITHUB_TOKEN` secret if the repo is private.
2. Restores the ChromaDB knowledge base from `MyDrive/EduHinglish/knowledge_base_v4`. On the first run it builds the knowledge base and caches it there.
3. Loads the full pipeline with the adapter from the Hub.
4. Prints a public `https://….gradio.live` link. It works from any device while the notebook runs.
   The page's "Use via API" section shows how to call it with `gradio_client`.

The link dies when the Colab session ends. For an always-on URL, the same code can run on a GPU Hugging Face Space later.

Locally / on any GPU machine:

```bash
python src/pipeline.py --query "Ohm ka law kya hai?" --subject physics --class 10
EDUHINGLISH_ADAPTER=<hf-user>/eduhinglish-gemma-3-4b-it-lora python src/pipeline.py --query "..."
```

## Google Drive layout

```
MyDrive/EduHinglish/
├── data/                              ← Gemma input (upload from local data/)
│   ├── bio_9_10.json
│   ├── chem_9_10.json
│   ├── phys_9_10.json
│   ├── maths_9_10.json
│   └── ss_9_10.json
├── knowledge_base_v4/                 ← ChromaDB cache written by notebook 06
├── training_data/                     ← MuRIL input (upload from local training/data/)
│   ├── muril_lid_dataset_v3/
│   └── muril_intent_dataset_v3/
└── models/
    ├── muril_lid_v2/                  ← optional: previous model, used as warm start if present
    ├── muril_intent_v2/               ← optional: previous model, used as warm start if present
    ├── muril_lid_v3/                  ← written by notebook 01
    ├── muril_intent_v3/               ← written by notebook 02
    ├── gemma_eduhinglish_v3/          ← old bio+ss run (notebook 04); warm start for v4
    │   └── adapter/
    └── gemma_eduhinglish_v4/
        ├── checkpoints/               ← written by notebook 05 every 50 steps (resume point)
        ├── adapter/                   ← final LoRA adapter (+ eval_metrics.json)
        └── TRAINING_DONE              ← marker: re-running notebook 05 skips training
```

If the old MuRIL models are still on Drive under `muril_models/muril_*_v2`,
move them to `models/` to use them as a warm start. Otherwise the notebooks
start from `google/muril-base-cased`.

## Adding a new dataset

1. Add the new file (e.g. `"data/sci_9_10.json"`) to `QA_SOURCES` in `scripts/build_unified_dataset.py`.
2. Add any new intent names to `INTENT_LABELS` in `training/label_config.py`.
3. Add its `data/processed/` chapter folders to `CHAPTER_MAP` in `src/chunker.py`.
4. Add the file name to `DATA_FILES` in notebook 05 and in `scripts/publish_kaggle_dataset.py`.
5. Optionally, add the new subject's terms to the NER gazetteers in `training/prepare_spacy_data.py`.
6. Re-run steps 1–6. Set `REBUILD_KB = True` once in notebook 06.
