# Phase 4 — MuRIL Fine-tuning for LID + Intent

> **Phase:** 4 of 8 | **Status:** Ready to start
> **Input:** `data/unified_biology_dataset_v2.json` (950 entries, 10 chapters)
> **Output:** Two fine-tuned MuRIL models — LID (token classification) + Intent (sequence classification)
> **Runs on:** Google Colab T4 (free tier) — will NOT run on CPU/Mac
> **Prerequisite:** Phase 3 complete (spaCy models trained as baseline)

---

## What Phase 4 Builds

Two transformer models fine-tuned on your Hinglish dataset using Google's MuRIL:

| Model | Task | Architecture | Input | Output |
| --- | --- | --- | --- | --- |
| `muril_lid_v2` | Word-level Language ID | MuRIL + LoRA token classifier | Hinglish sentence | HI/EN/NE/UNIV per subword token |
| `muril_intent_v2` | Student Query Intent | MuRIL + LoRA sequence classifier | Student query text | explain_concept/compare/definition/etc. |

### Why MuRIL?

- **Pre-trained on 17 Indian languages** including Hindi in both Devanagari and Roman script
- **Understands Hinglish natively** — no rule-based shortcuts needed
- **236M params** — fits easily in Colab T4 (16GB VRAM)
- **LoRA fine-tuning** — only ~2M trainable params, trains in ~20 min
- **Expected improvement:** LID 98% → 99%+, Intent 70% → 80%+

---

## PROMPT — Jatin (paste into Claude Code)

Copy everything between the triple backtick fences and paste into Claude Code:

---

```
I am Jatin, working on EduHinglish — a final-year AI project for
Hinglish-speaking Indian students. I have completed Phase 3
(spaCy model training). I now need to build Phase 4:
fine-tuning Google's MuRIL transformer on my labeled dataset
for production-grade LID and Intent classification.

═══════════════════════════════════════════════════════
PROJECT STATE (what already exists)
═══════════════════════════════════════════════════════

Unified dataset: data/unified_biology_dataset_v2.json
  - 950 entries across 10 chapters (Class 9 & 10 Science)
  - Each entry has: id, original_english, hinglish_roman,
    word_level_labels (dict: word → HI/EN/NE/UNIV/MIX),
    topic, chapter, class, code_mixing_type
  - Some entries have: is_student_query (bool), intent (str)

Example entry:
  {
    "id": "10_05_001",
    "hinglish_roman": "Living organisms bahut well-organised structures hote hain...",
    "word_level_labels": {
      "Living": "EN", "organisms": "EN", "bahut": "HI",
      "well-organised": "EN", "structures": "EN",
      "hote": "HI", "hain": "HI", ...
    },
    "is_student_query": false
  }

Label tags (for LID):
  HI   = Hindi word in Roman script
  EN   = English/science word
  NE   = Named Entity (scientist name, place)
  UNIV = Language-independent (numbers, Sir, OK, punctuation)
  MIX  = Mixed-script word (rare)

Intent values (for student queries, ~120 entries):
  explain_concept, compare_concepts, give_example,
  formula_request, definition

Phase 3 baseline results (spaCy models):
  LID:    98.1% accuracy (tok2vec + tagger)
  Intent: 70.0% accuracy (BOW textcat, 120 examples)

Phase 4 must BEAT these baselines using MuRIL.

No external APIs. Everything runs on Colab T4 (free tier).
Python 3.10.

═══════════════════════════════════════════════════════
PHASE 4 SCOPE — FILES TO BUILD
═══════════════════════════════════════════════════════

training/
├── prepare_hf_data.py           ← TASK 1
├── train_muril_lid.py           ← TASK 2
├── train_muril_intent.py        ← TASK 3
├── evaluate_muril.py            ← TASK 4
└── colab_notebooks/
    ├── 01_train_muril_lid.ipynb  ← TASK 5a
    └── 02_train_muril_intent.ipynb ← TASK 5b

models/
├── muril_lid_v2/                ← saved after training
└── muril_intent_v2/             ← saved after training

═══════════════════════════════════════════════════════
TASK 1 — training/prepare_hf_data.py
═══════════════════════════════════════════════════════

This script converts data/unified_biology_dataset_v2.json into
HuggingFace Dataset format for both LID and Intent tasks.

── 1a. Token Classification data (for MuRIL LID) ──

For each entry in the dataset:
  - Tokenise hinglish_roman using MuRIL's tokenizer
    (AutoTokenizer.from_pretrained("google/muril-base-cased"))
  - CRITICAL: MuRIL uses subword tokenization (WordPiece).
    Each original word may split into multiple subword tokens.
    The label for the FIRST subword of each word comes from
    word_level_labels. Subsequent subwords get label -100
    (ignored in loss computation). This is standard HuggingFace
    token classification alignment.
  - Punctuation tokens get label UNIV (index for UNIV)
  - Map label strings to integers:
    label2id = {"HI": 0, "EN": 1, "NE": 2, "UNIV": 3, "MIX": 4}
    id2label = {0: "HI", 1: "EN", 2: "NE", 3: "UNIV", 4: "MIX"}
  - Split 80/20 train/dev (same random seed=42 as Phase 3)

Output format: HuggingFace DatasetDict saved to disk
  training/data/muril_lid_dataset_v2/
    ├── train/
    └── dev/

Each example has:
  - input_ids: tokenized by MuRIL
  - attention_mask: standard
  - labels: aligned label IDs (with -100 for subword continuations)
  - word_ids: for debugging alignment

── 1b. Sequence Classification data (for MuRIL Intent) ──

Filter entries where is_student_query == True and intent is valid.
For each:
  - Tokenize hinglish_roman using MuRIL tokenizer
  - Map intent to integer:
    intent2id = {
      "explain_concept": 0, "compare_concepts": 1,
      "give_example": 2, "formula_request": 3, "definition": 4
    }
  - Split 80/20 train/dev

Output format: HuggingFace DatasetDict
  training/data/muril_intent_dataset_v2/
    ├── train/
    └── dev/

Each example has:
  - input_ids, attention_mask
  - labels: single integer (intent ID)

── Print stats at end ──

  MuRIL LID Dataset:
    Train: 760 examples
    Dev:   190 examples
    Label distribution: HI=56%, EN=33%, NE=1%, UNIV=9%, MIX=0%
    Avg subword tokens per sentence: 28.3

  MuRIL Intent Dataset:
    Train: 96 examples
    Dev:   24 examples
    Intent distribution: explain_concept=53, compare=29, ...

── CLI ──

  python training/prepare_hf_data.py
  python training/prepare_hf_data.py --task lid
  python training/prepare_hf_data.py --task intent
  python training/prepare_hf_data.py --dataset data/unified_biology_dataset_v2.json

═══════════════════════════════════════════════════════
TASK 2 — training/train_muril_lid.py
═══════════════════════════════════════════════════════

Fine-tune MuRIL for token-level Language ID using LoRA (PEFT).

── Dependencies (install in script/notebook) ──

  pip install transformers datasets peft accelerate
  pip install torch  # already available on Colab

── Model setup ──

  Base model: google/muril-base-cased
  Task: TokenClassification (AutoModelForTokenClassification)
  Num labels: 5 (HI, EN, NE, UNIV, MIX)
  id2label and label2id mappings (same as prepare_hf_data.py)

── LoRA configuration ──

  from peft import LoraConfig, get_peft_model, TaskType

  lora_config = LoraConfig(
      task_type=TaskType.TOKEN_CL,
      r=16,               # rank — 16 is good for small datasets
      lora_alpha=32,       # scaling factor
      lora_dropout=0.1,
      bias="none",
      target_modules=["query", "key", "value", "dense"],
  )

  This reduces trainable params from 236M to ~2M.

── Training configuration ──

  from transformers import TrainingArguments

  training_args = TrainingArguments(
      output_dir="./muril_lid_checkpoints",
      num_train_epochs=10,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=32,
      learning_rate=2e-4,       # higher LR for LoRA
      weight_decay=0.01,
      eval_strategy="epoch",
      save_strategy="epoch",
      load_best_model_at_end=True,
      metric_for_best_model="accuracy",
      greater_is_better=True,
      logging_steps=10,
      fp16=True,               # use mixed precision on T4
      report_to="none",        # no wandb
      seed=42,
  )

── Metrics function ──

  Compute token-level accuracy (ignoring -100 labels).
  Also compute per-label precision/recall/F1.
  Use seqeval or manual computation.

── Training flow ──

  1. Load HuggingFace dataset from training/data/muril_lid_dataset_v2/
  2. Load MuRIL model + LoRA adapter
  3. Create Trainer with compute_metrics
  4. Train (should take ~15-20 min on T4)
  5. Save best model to models/muril_lid_v2/
     - Save the PEFT adapter: model.save_pretrained()
     - Save the tokenizer: tokenizer.save_pretrained()
     - Also save merged model (adapter merged into base) for
       easy inference: model.merge_and_unload()

── Output at end ──

  ══════════════════════════════════════
  MuRIL LID Training Complete
  ══════════════════════════════════════
  Best dev accuracy: 99.2%
  Phase 3 baseline:  98.1% (spaCy tok2vec)
  Improvement:       +1.1pp
  Trainable params:  2.1M / 236M (0.9%)
  Training time:     18 min on T4
  Saved to:          models/muril_lid_v2/
  ══════════════════════════════════════

  Per-label F1:
    Label  | Precision | Recall | F1
    HI     |   99.3%   | 99.5%  | 99.4%
    EN     |   99.1%   | 98.8%  | 99.0%
    NE     |   95.2%   | 88.4%  | 91.7%
    UNIV   |   99.8%   | 99.9%  | 99.8%

── CLI ──

  python training/train_muril_lid.py
  python training/train_muril_lid.py --epochs 15 --lr 3e-4
  python training/train_muril_lid.py --output models/muril_lid_v2/

NOTE: This script should detect if CUDA is available.
If no GPU, print a warning and suggest using the Colab notebook.
Still allow CPU training but warn it will be very slow (~2 hours).

═══════════════════════════════════════════════════════
TASK 3 — training/train_muril_intent.py
═══════════════════════════════════════════════════════

Fine-tune MuRIL for sequence-level Intent Classification.

── Model setup ──

  Base model: google/muril-base-cased
  Task: SequenceClassification (AutoModelForSequenceClassification)
  Num labels: 5
  intent2id / id2label mappings

── LoRA configuration ──

  Same as LID but with TaskType.SEQ_CL

  lora_config = LoraConfig(
      task_type=TaskType.SEQ_CL,
      r=8,                # smaller rank — tiny dataset (120 examples)
      lora_alpha=16,
      lora_dropout=0.15,  # slightly higher dropout for small data
      bias="none",
      target_modules=["query", "key", "value", "dense"],
  )

── Training configuration ──

  training_args = TrainingArguments(
      output_dir="./muril_intent_checkpoints",
      num_train_epochs=20,        # more epochs for small dataset
      per_device_train_batch_size=8,  # smaller batch for small data
      per_device_eval_batch_size=16,
      learning_rate=2e-4,
      weight_decay=0.01,
      eval_strategy="epoch",
      save_strategy="epoch",
      load_best_model_at_end=True,
      metric_for_best_model="f1_macro",
      greater_is_better=True,
      logging_steps=5,
      fp16=True,
      report_to="none",
      seed=42,
  )

── Metrics function ──

  Compute accuracy, macro-F1, and per-class precision/recall/F1.
  Print confusion matrix at the end.

── Training flow ──

  Same as LID but for sequence classification.
  Save to models/muril_intent_v2/

── Output at end ──

  ══════════════════════════════════════
  MuRIL Intent Training Complete
  ══════════════════════════════════════
  Best dev F1 (macro): 82.5%
  Phase 3 baseline:    70.0% (spaCy BOW)
  Improvement:         +12.5pp
  Saved to:            models/muril_intent_v2/
  ══════════════════════════════════════

  Confusion matrix:
    Predicted →  explain  compare  example  formula  definition
    explain  [   9        0        1        0         0      ]
    compare  [   0        4        0        0         0      ]
    ...

── CLI ──

  python training/train_muril_intent.py
  python training/train_muril_intent.py --epochs 30

═══════════════════════════════════════════════════════
TASK 4 — training/evaluate_muril.py
═══════════════════════════════════════════════════════

Unified evaluation script that loads both MuRIL models and
runs them on the same test sentences used in Phase 3's
evaluate_models.py, enabling direct comparison.

── Test sentences (same as Phase 3) ──

LID test cases:
  "Mitochondria ko cell ka powerhouse kehte hain."
  "Sir, photosynthesis kaise hoti hai?"
  "DNA replication mein enzyme kya role play karta hai?"
  "Mendel ne pea plants par experiments kiye the."
  "Yeh process anaerobic respiration kehlata hai."

Intent test cases:
  "Sir, nucleus ka kaam kya hota hai?" → explain_concept
  "Mitochondria aur chloroplast mein kya fark hai?" → compare_concepts
  "Ek example do osmosis ka?" → give_example

── Output format ──

  ══════════════════════════════════════════════════════
  EduHinglish — Phase 4 MuRIL Model Evaluation
  ══════════════════════════════════════════════════════

  ── MuRIL LID (models/muril_lid_v2/) ──
  Input: "Mitochondria ko cell ka powerhouse kehte hain."
    Mitochondria → EN  ✓
    ko           → HI  ✓
    cell         → EN  ✓
    ka           → HI  ✓
    powerhouse   → EN  ✓
    kehte        → HI  ✓
    hain         → HI  ✓
    .            → UNIV ✓

  ── Comparison: spaCy LID vs MuRIL LID ──
  Metric     | spaCy (Phase 3) | MuRIL (Phase 4)
  Accuracy   |     98.1%       |     99.2%
  HI F1      |     98.7%       |     99.4%
  EN F1      |     97.4%       |     99.0%
  NE F1      |     76.7%       |     91.7%

  ── MuRIL Intent (models/muril_intent_v2/) ──
  Input: "Mitochondria aur chloroplast mein kya fark hai?"
    Predicted: compare_concepts (conf: 0.94)  ✓

  ── Comparison: spaCy Intent vs MuRIL Intent ──
  Metric     | spaCy (Phase 3) | MuRIL (Phase 4)
  Accuracy   |     70.0%       |     82.5%

── IMPORTANT: Subword token handling for LID ──

MuRIL tokenizes "Mitochondria" as ["Mit", "##och", "##ond", "##ria"].
Your evaluation script must:
  1. Run MuRIL inference to get per-subword predictions
  2. Aggregate subwords back to word-level labels
     (use the FIRST subword's prediction for each word)
  3. Display word-level labels (not subword-level)

This aggregation logic is critical for user-facing output.
Put it in a reusable function:
  def predict_lid(text, model, tokenizer) -> list[tuple[str, str]]:
      '''Returns [(word, label), ...] with subword aggregation.'''

── CLI ──

  python training/evaluate_muril.py
  python training/evaluate_muril.py --lid-only
  python training/evaluate_muril.py --intent-only

═══════════════════════════════════════════════════════
TASK 5 — Colab Notebooks
═══════════════════════════════════════════════════════

Create two self-contained Jupyter notebooks ready to run
on Google Colab with T4 GPU.

── 5a. training/colab_notebooks/01_train_muril_lid.ipynb ──

Cell 1: Setup
  !pip install transformers datasets peft accelerate -q
  # Check GPU
  !nvidia-smi

Cell 2: Mount Drive + Upload dataset
  from google.colab import drive
  drive.mount('/content/drive')
  # Option A: Upload unified_biology_dataset_v2.json manually
  # Option B: Clone repo from GitHub
  #   !git clone https://github.com/<your-repo>/EduHinglish.git

Cell 3: Prepare HuggingFace data
  (Embed the prepare_hf_data.py logic directly)

Cell 4: Load MuRIL + Configure LoRA
  (Embed train_muril_lid.py logic)

Cell 5: Train
  trainer.train()

Cell 6: Evaluate
  (Print accuracy, per-label F1)

Cell 7: Save model to Drive
  model.save_pretrained("/content/drive/MyDrive/EduHinglish/models/muril_lid_v2")
  tokenizer.save_pretrained("/content/drive/MyDrive/EduHinglish/models/muril_lid_v2")

Cell 8: Test on example sentences
  (Run the test cases from evaluate_muril.py)

── 5b. training/colab_notebooks/02_train_muril_intent.ipynb ──

Same structure as 01 but for intent classification.

═══════════════════════════════════════════════════════
GIT WORKFLOW
═══════════════════════════════════════════════════════

Branch: feature/jatin-phase4-muril
Base:   develop

Commit after each task:
  git add . && git commit -m "feat(jatin): [description]"
  git push origin feature/jatin-phase4-muril

Commit message examples:
  feat(jatin): prepare_hf_data - MuRIL tokenizer alignment
  feat(jatin): train_muril_lid.py LoRA fine-tuning script
  feat(jatin): train_muril_intent.py sequence classification
  feat(jatin): evaluate_muril.py with spaCy comparison
  feat(jatin): colab notebook 01 - MuRIL LID training
  feat(jatin): colab notebook 02 - MuRIL intent training
  model(jatin): muril_lid_v2 trained 99.2% dev accuracy
  model(jatin): muril_intent_v2 trained 82.5% dev F1

After all models trained, open PR → develop.

═══════════════════════════════════════════════════════
START HERE — EXACT ORDER TO EXECUTE
═══════════════════════════════════════════════════════

Step 1:  Create training/colab_notebooks/ folder
Step 2:  Write training/prepare_hf_data.py
Step 3:  Run prepare_hf_data.py locally — verify datasets created
Step 4:  Write training/train_muril_lid.py
Step 5:  Write training/train_muril_intent.py
Step 6:  Write training/evaluate_muril.py
Step 7:  Create 01_train_muril_lid.ipynb (self-contained)
Step 8:  Create 02_train_muril_intent.ipynb (self-contained)
Step 9:  Upload dataset to Colab, run notebook 01 — verify LID model
Step 10: Run notebook 02 — verify intent model
Step 11: Download trained models to models/muril_lid_v2/ and models/muril_intent_v2/
Step 12: Run evaluate_muril.py locally — verify comparison with Phase 3
Step 13: Final commit + push + PR → develop

Build all scripts (Steps 2-8) locally first, then run
notebooks on Colab for actual training.

Do not hallucinate model accuracy numbers — print actual
computed metrics from the trained model.
```

---

## Key Technical Notes

### MuRIL tokenizer behavior

MuRIL uses WordPiece tokenization. This means:
- `"Mitochondria"` → `["Mit", "##och", "##ond", "##ria"]`
- `"ko"` → `["ko"]`
- `"photosynthesis"` → `["photos", "##yn", "##thesis"]`

Your code MUST handle this subword alignment properly:
- First subword of each word gets the word's LID label
- Subsequent subwords (`##` prefix) get label -100 (ignored by loss)
- Special tokens ([CLS], [SEP], [PAD]) also get -100

### LoRA vs full fine-tuning

| Approach | Trainable Params | Training Time | GPU RAM |
| --- | --- | --- | --- |
| Full fine-tune | 236M (100%) | ~1 hour | ~12GB |
| LoRA (r=16) | ~2M (0.9%) | ~20 min | ~6GB |
| LoRA (r=8) | ~1M (0.4%) | ~15 min | ~5GB |

Use LoRA — same or better accuracy for your dataset size, much faster.

### Expected accuracy targets

| Model | Phase 3 Baseline | Phase 4 Target | Notes |
| --- | --- | --- | --- |
| LID | 98.1% (spaCy) | 99%+ | MuRIL knows Indian languages natively |
| Intent | 70.0% (BOW) | 80%+ | Transformer captures context that BOW misses |

### If Colab disconnects

- Checkpoints are saved every epoch — resume from latest
- Models saved to Google Drive persist across sessions
- Use `trainer.train(resume_from_checkpoint=True)` to continue

### Model saving strategy

Save THREE versions of each model:
1. **LoRA adapter only** (~8MB) — `models/muril_lid_v2/adapter/`
2. **Merged model** (~900MB) — `models/muril_lid_v2/merged/`
3. **Tokenizer** — `models/muril_lid_v2/tokenizer/`

The adapter is tiny (for git), the merged model is for inference.
Add `models/muril_*/merged/` to `.gitignore` (too large for git).

### Integration plan (Phase 8)

After Phase 4, these models replace the spaCy LID and Intent:
- `src/script_detector.py` → add `MuRILLanguageIdentifier` class
- `src/pipeline.py` → use MuRIL LID when GPU available, fall back to spaCy on CPU
- Intent: MuRIL for production, spaCy for lightweight/CPU deployment

---

## After Phase 4 — Next Steps

| Phase | What | When |
| --- | --- | --- |
| Phase 5 | ChromaDB knowledge base (NCERT RAG) | Week 7 |
| Phase 6 | IndicBART Hinglish generator fine-tuning | Week 8 |
| Phase 7 | GEC engine | Week 9 |
| Phase 8 | Streamlit UI + full integration | Week 10–11 |
