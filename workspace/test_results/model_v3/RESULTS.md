# v3 Model Results (Biology 9/10 + Social Science 9)

Data: `data/unified_dataset_v3.json`, built with `--silver-lid`. LID dev = 340 gold sentences; intent dev = 1,109 queries; NER dev = 462 docs.

| Model | Where trained | Dev metric | Notes |
|---|---|---|---|
| spaCy LID (`models/lid_v3`) | local CPU | 94.9% acc | NE F1 52.9%; v2 was 98.1% → silver labels likely hurt |
| spaCy Intent (`models/intent_v3`) | local CPU | 71.25% acc (15 classes) | v2 was 70% on 5 classes |
| spaCy NER (`models/ner_v3`) | local CPU | 97.2% micro F1 | 9 entity types incl. social science |
| MuRIL LID (`models/muril_lid_v3`) | Colab T4, 5 epochs, ~11 min | **98.12% acc** (6,389 gold words) | HI 98.5 / EN 97.5 / NE 70.9 / UNIV 100 F1; from base `google/muril-base-cased` |
| MuRIL Intent (`models/muril_intent_v3`) | Colab T4, 10 epochs, ~33 min | **90.53% acc**, 76.46% macro F1 | from base MuRIL; weak on tiny classes: one_word 22% F1 (7 dev), numerical_problem 50% (6), assertion_reason 0% (1) |
| Gemma-3-4B-it (`gemma_eduhinglish_v2`) | Colab | _pending_ | |
| RAG (`src/knowledge_base`) | local GPU | 6/7 retriever test queries on-topic; 5 PASS + 1 WARN (right chapter, score 0.57) in build check | 828 chunks, 18 chapters (ss9 PDF page footers stripped) |

## spaCy vs MuRIL (same gold dev sets)

| Task | spaCy v3 | MuRIL v3 |
|---|---|---|
| LID accuracy | 94.9% | **98.1%** |
| LID NE F1 | 52.9% | **70.9%** |
| Intent accuracy (15 classes) | 71.25% | **90.5%** |

MuRIL wins on both tasks, so it should be the M1 model in the pipeline. The spaCy models are a CPU-only fallback.

## Known issues

- **Hinglish queries about rare English terms miss in RAG.**
  - "Stomata ka function kya hai?" returns ss9 history chunks.
  - "What is the function of stomata?" returns the right Life Processes chunks (0.54).
  - The cause is that the MiniLM embedder handles the Hinglish phrasing poorly. It is not missing data: stomata appears in class9_bio_ch03 and class10_bio_ch05.
  - A fix for pipeline integration is hybrid retrieval (BM25 + embeddings). Translating the query to English before retrieval would also work.
- **The intent model fails on tiny classes:** one_word, numerical_problem and assertion_reason. These need more training examples, not more epochs.

Full logs:
- spaCy run: `spacy_training_log.txt`
- MuRIL dev evaluation: `muril_eval_log.txt` (`python training/evaluate_muril.py`)
- MuRIL training: outputs saved in `training/colab_notebooks/01_train_muril_lid.ipynb` and `02_train_muril_intent.ipynb`
