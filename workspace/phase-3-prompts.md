# Phase 3 — spaCy Model Training

> **Phase:** 3 of 8 | **Status:** Ready to start
> **Input:** `data/unified_biology_dataset_v2.json` (950 entries, 10 chapters)
> **Output:** Three trained spaCy models in `models/` — LID, NER, Intent
> **Runs on:** CPU (Mac/local) or Colab T4

---

## What Phase 3 Builds

Three spaCy models trained entirely on your own dataset — no external APIs:

| Model | Task | Architecture | Input | Output |
| --- | --- | --- | --- | --- |
| `lid_v2` | Word-level Language ID | tok2vec + token classifier | Hinglish sentence | HI/EN/NE/UNIV per token |
| `ner_v2` | Science Named Entity Recognition | spaCy ner | Hinglish sentence | ORGANELLE/PROCESS/SCIENTIST/INSTRUMENT/CONCEPT |
| `intent_v2` | Student Query Intent | spaCy textcat | Student query text | explain_concept/compare/definition/etc. |

---

## PROMPT — Jatin (paste into Claude Code)

Copy everything between the triple backtick fences and paste into Claude Code:

---

```
I am Jatin, working on EduHinglish — a final-year AI project for 
Hinglish-speaking Indian students. I have just completed Phase 2 
(multi-chapter dataset pipeline). I now need to build Phase 3: 
training three spaCy NLP models on my labeled dataset.

═══════════════════════════════════════════════════════
PROJECT STATE (what already exists)
═══════════════════════════════════════════════════════

Unified dataset: data/unified_biology_dataset_v2.json
  - 950 entries across 10 chapters (Class 9 & 10 Science)
  - Each entry has: id, original_english, hinglish_roman,
    word_level_labels (dict: word → HI/EN/NE/UNIV/MIX),
    topic, chapter, class, code_mixing_type
  - Some entries have: is_student_query (bool), intent (str),
    is_gec_sample (bool), gec_data (dict)

Label tags:
  HI   = Hindi word in Roman script
  EN   = English/science word
  NE   = Named Entity (scientist name, place)
  UNIV = Language-independent (numbers, Sir, OK)
  MIX  = Mixed-script word

Intent values (for student queries):
  explain_concept, compare_concepts, give_example,
  formula_request, definition

No external APIs. Everything runs locally or on Colab T4.
Python 3.10, virtual env already set up (eduhinglish_env).

Already installed: nltk, spacy, pdfplumber, scikit-learn
spaCy model already downloaded: en_core_web_sm

═══════════════════════════════════════════════════════
PHASE 3 SCOPE — THREE FILES TO BUILD
═══════════════════════════════════════════════════════

training/
├── prepare_spacy_data.py    ← TASK 1
├── train_lid.py             ← TASK 2
├── train_ner.py             ← TASK 3
├── train_intent.py          ← TASK 4
├── evaluate_models.py       ← TASK 5
└── configs/
    ├── lid_config.cfg
    ├── ner_config.cfg
    └── intent_config.cfg

models/
├── lid_v2/                  ← saved after train_lid.py
├── ner_v2/                  ← saved after train_ner.py
└── intent_v2/               ← saved after train_intent.py

═══════════════════════════════════════════════════════
TASK 1 — training/prepare_spacy_data.py
═══════════════════════════════════════════════════════

This script converts data/unified_biology_dataset_v2.json into
.spacy binary files that spaCy's training pipeline consumes.

It must produce THREE separate training splits:

── 1a. Token Classification data (for LID model) ──

For each entry in the dataset:
  - Tokenise hinglish_roman using the same whitespace split 
    used in annotation_helper.py (strip punctuation)
  - Map each token to its label from word_level_labels
  - Create a spaCy Doc with token-level BILOU tags
  - Add to train/dev split (80/20)

Output:
  training/data/lid_train_v2.spacy
  training/data/lid_dev_v2.spacy

── 1b. NER data (for science NER model) ──

NER labels to extract (rule-based from existing entries):
  SCIENTIST  — words tagged NE that match a known scientist list
               (Mendel, Darwin, Hooke, Lamarck, Linnaeus, Watson, 
                Crick, Pasteur, Fleming, Lister)
  ORGANELLE  — nucleus, mitochondria, ribosome, chloroplast, 
               vacuole, Golgi, endoplasmic reticulum, centrosome,
               lysosome, cell membrane, cell wall
  PROCESS    — photosynthesis, respiration, digestion, excretion,
               reproduction, transpiration, fertilisation, 
               pollination, osmosis, diffusion, plasmolysis
  CONCEPT    — any EN token that is a multi-word biology concept
               (natural selection, food chain, nitrogen cycle,
                blood group, gene expression, DNA replication)
  INSTRUMENT — microscope, stethoscope, thermometer

Use spacy's PhraseMatcher to find spans in hinglish_roman for each 
category. For each match create a spaCy Doc with .ents set.
Skip entries where no entity is found (don't create empty NER docs).

Output:
  training/data/ner_train_v2.spacy
  training/data/ner_dev_v2.spacy

Print counts at end:
  NER spans by label:
    SCIENTIST   : 23
    ORGANELLE   : 187
    PROCESS     : 134
    CONCEPT     : 89
    INSTRUMENT  : 4

── 1c. Text Classification data (for Intent model) ──

Filter entries where is_student_query == True.
For each, create a spaCy Doc from hinglish_roman with cats set:
  {"explain_concept": 1, "compare_concepts": 0, ...}
(one-hot: only the correct intent gets 1)

If fewer than 60 student query entries exist, print a warning
but still proceed with what's available.

Output:
  training/data/intent_train_v2.spacy
  training/data/intent_dev_v2.spacy

Print counts:
  Intent distribution:
    explain_concept  : 34
    compare_concepts : 8
    give_example     : 12
    formula_request  : 3
    definition       : 7
  Total intent entries: 64

── CLI ──

python training/prepare_spacy_data.py
python training/prepare_spacy_data.py --dataset data/unified_biology_dataset_v2.json
python training/prepare_spacy_data.py --task lid     # only prepare LID data
python training/prepare_spacy_data.py --task ner
python training/prepare_spacy_data.py --task intent

═══════════════════════════════════════════════════════
TASK 2 — training/train_lid.py  +  training/configs/lid_config.cfg
═══════════════════════════════════════════════════════

Train spaCy token classifier for Language ID.

── lid_config.cfg ──

Generate a complete spaCy config for:
  - Pipeline: ["tok2vec", "tagger"]
  - tok2vec: width=96, depth=4, window_size=1 (CPU-friendly)
  - tagger: labels = [HI, EN, NE, UNIV, MIX]
  - optimizer: Adam, lr=0.001
  - batch_size: compounding(1, 32, 1.001)
  - max_epochs: 30
  - eval_frequency: 200

Use spaCy's native config format (cfg files with [training],
[nlp], [components] sections).

── train_lid.py ──

Script that:
1. Loads training/data/lid_train_v2.spacy and lid_dev_v2.spacy
2. Creates a fresh spaCy pipeline with tok2vec + tagger
3. Trains for up to 30 epochs with early stopping 
   (patience=5, stop if dev accuracy doesn't improve)
4. Prints per-epoch stats:
   Epoch 1/30 | Loss: 2.43 | Dev Acc: 67.2% | Best: 67.2% ✓
   Epoch 2/30 | Loss: 1.87 | Dev Acc: 72.1% | Best: 72.1% ✓
   ...
5. Saves best model to models/lid_v2/
6. At end prints:
   ══════════════════════════════════
   LID Model Training Complete
   Best dev accuracy: 84.3%
   Saved to: models/lid_v2/
   Baseline (rule-based): ~75%
   Improvement: +9.3pp
   ══════════════════════════════════

Also compute and print per-label F1 at the end:
   Label  | Precision | Recall | F1
   HI     |   87.2%   | 89.1%  | 88.1%
   EN     |   91.3%   | 90.4%  | 90.8%
   NE     |   62.1%   | 58.3%  | 60.1%
   UNIV   |   79.4%   | 81.2%  | 80.3%

CLI:
  python training/train_lid.py
  python training/train_lid.py --epochs 50 --output models/lid_v2/

═══════════════════════════════════════════════════════
TASK 3 — training/train_ner.py  +  training/configs/ner_config.cfg
═══════════════════════════════════════════════════════

Train spaCy NER for science entities.

── ner_config.cfg ──

Config for:
  - Pipeline: ["tok2vec", "ner"]
  - Same tok2vec as LID (shared architecture, different weights)
  - NER labels: SCIENTIST, ORGANELLE, PROCESS, CONCEPT, INSTRUMENT
  - max_epochs: 40 (NER needs more epochs than token classification)
  - dropout: 0.2

── train_ner.py ──

Same structure as train_lid.py but:
  - Trains ner component
  - Dev metric: F1 (not accuracy, standard for NER)
  - Saves to models/ner_v2/
  - Per-label F1 at end:
    Entity       | P     | R     | F1
    ORGANELLE    | 88.4% | 86.2% | 87.3%
    PROCESS      | 83.1% | 85.6% | 84.3%
    SCIENTIST    | 91.2% | 89.4% | 90.3%
    CONCEPT      | 71.3% | 68.9% | 70.1%
    INSTRUMENT   | 62.0% | 58.3% | 60.1%

CLI:
  python training/train_ner.py
  python training/train_ner.py --epochs 60 --output models/ner_v2/

═══════════════════════════════════════════════════════
TASK 4 — training/train_intent.py  +  training/configs/intent_config.cfg
═══════════════════════════════════════════════════════

Train spaCy text classifier for student query intent.

── intent_config.cfg ──

Config for:
  - Pipeline: ["textcat"]
  - Architecture: BOW (bag-of-words) — best for small datasets (<200)
    If total intent entries > 200, use CNN architecture instead
  - exclusive_classes: true (one intent per query)
  - max_epochs: 50

── train_intent.py ──

Same structure as train_lid.py but:
  - Trains textcat component
  - Dev metric: accuracy (macro-averaged across intents)
  - Prints:
    WARNING: Only 64 intent entries found.
    Training with BOW architecture (recommended for <200 examples).
    Consider adding more student queries to improve accuracy.
  - Saves to models/intent_v2/
  - At end prints confusion matrix (text-based):
    Predicted →  explain  compare  example  formula  definition
    explain  [   29        1        1        0         0      ]
    compare  [    1        5        1        0         0      ]
    ...

CLI:
  python training/train_intent.py
  python training/train_intent.py --arch cnn    # force CNN

═══════════════════════════════════════════════════════
TASK 5 — training/evaluate_models.py
═══════════════════════════════════════════════════════

Unified evaluation script that loads all three saved models
and runs them on a set of test sentences.

Test sentences to always include (hardcoded):

LID test cases:
  "Mitochondria ko cell ka powerhouse kehte hain."
  "Sir, photosynthesis kaise hoti hai?"
  "DNA replication mein enzyme kya role play karta hai?"
  "Mendel ne pea plants par experiments kiye the."
  "Yeh process anaerobic respiration kehlata hai."

NER test cases (same sentences, check entity detection):
  Should find: mitochondria → ORGANELLE, cell → ORGANELLE
  Should find: photosynthesis → PROCESS
  Should find: DNA replication → PROCESS, enzyme → CONCEPT
  Should find: Mendel → SCIENTIST, pea → CONCEPT
  Should find: anaerobic respiration → PROCESS

Intent test cases:
  "Sir, nucleus ka kaam kya hota hai?" → explain_concept
  "Mitochondria aur chloroplast mein kya fark hai?" → compare_concepts
  "Ek example do osmosis ka?" → give_example

Output format:
══════════════════════════════════════════════════════
EduHinglish — Phase 3 Model Evaluation
══════════════════════════════════════════════════════

── LID Model (models/lid_v2/) ──
Input: "Mitochondria ko cell ka powerhouse kehte hain."
  Mitochondria → EN  ✓
  ko           → HI  ✓
  cell         → EN  ✓
  ka           → HI  ✓
  powerhouse   → EN  ✓
  kehte        → HI  ✓
  hain         → HI  ✓

── NER Model (models/ner_v2/) ──
Input: "Mendel ne pea plants par experiments kiye the."
  [Mendel] SCIENTIST
  [pea plants] CONCEPT

── Intent Model (models/intent_v2/) ──
Input: "Mitochondria aur chloroplast mein kya fark hai?"
  Predicted: compare_concepts (conf: 0.87)  ✓

═══════════════════════════════════════════════════════
GIT WORKFLOW
═══════════════════════════════════════════════════════

Branch: feature/jatin-phase3-spacy
Base:   develop

Commit after each task:
  git add . && git commit -m "feat(jatin): [description]"
  git push origin feature/jatin-phase3-spacy

Commit message examples:
  feat(jatin): prepare_spacy_data - LID/NER/intent converters
  feat(jatin): lid_config.cfg + train_lid.py tok2vec tagger
  feat(jatin): ner_config.cfg + train_ner.py science entities
  feat(jatin): intent_config.cfg + train_intent.py BOW classifier
  feat(jatin): evaluate_models.py unified eval script
  model(jatin): lid_v2 trained 84.3% dev accuracy
  model(jatin): ner_v2 trained science NER F1 83.4%
  model(jatin): intent_v2 trained 78.2% intent accuracy

After all models trained, open PR → develop.

═══════════════════════════════════════════════════════
START HERE — EXACT ORDER TO EXECUTE
═══════════════════════════════════════════════════════

Step 1: Create training/ and models/ folder structure
Step 2: Write training/prepare_spacy_data.py (all three tasks)
Step 3: Run prepare_spacy_data.py — verify .spacy files created
Step 4: Write lid_config.cfg + train_lid.py
Step 5: Run train_lid.py — verify models/lid_v2/ saved
Step 6: Write ner_config.cfg + train_ner.py
Step 7: Run train_ner.py — verify models/ner_v2/ saved
Step 8: Write intent_config.cfg + train_intent.py
Step 9: Run train_intent.py — verify models/intent_v2/ saved
Step 10: Write evaluate_models.py
Step 11: Run evaluate_models.py — verify all three models work
Step 12: Final commit + push + PR → develop

Build all scripts (Steps 1-4) first, then I will run them.
Do not hallucinate model accuracy numbers — just print placeholders
like "training in progress" during the actual training loop.
```

---

## Key Technical Notes

### spaCy version
This project uses spaCy 3.x (not v2). Use the `spacy.training` API,
not the deprecated `nlp.begin_training()` / `nlp.update()` v2 API.

### Expected accuracy targets

| Model | Baseline | Target (spaCy) | Notes |
| --- | --- | --- | --- |
| LID | ~75% (rule-based) | 83–88% | More data → higher |
| NER | 0% (no baseline) | 70–85% | Varies by entity type |
| Intent | 0% (no baseline) | 70–80% | Only ~60 examples |

### If training data is too small

- LID: 950 sentences × avg 10 tokens = ~9,500 token examples — sufficient
- NER: depends on PhraseMatcher matches — likely 200–400 spans — borderline
- Intent: ~60–80 queries — small, use BOW, expect 70–75% accuracy

### Integration plan (Phase 8)
After Phase 3, these models slot in as:
- `src/script_detector.py` → replace `WordLevelLID` with `lid_v2`
- `src/pipeline.py` → add NER + intent detection steps using `ner_v2` / `intent_v2`

---

## After Phase 3 — Next Steps

| Phase | What | When |
| --- | --- | --- |
| Phase 4 | MuRIL fine-tuning on Colab T4 (better LID) | Week 6 |
| Phase 5 | ChromaDB knowledge base (NCERT RAG) | Week 7 |
| Phase 6 | IndicBART Hinglish generator fine-tuning | Week 8 |
| Phase 7 | GEC engine | Week 9 |
| Phase 8 | Streamlit UI + full integration | Week 10–11 |
