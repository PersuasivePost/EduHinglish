# v3 Model Results (Biology 9/10 + Social Science 9)

Data: `data/unified_dataset_v3.json`, built with `--silver-lid`. LID dev = 340 gold sentences; intent dev = 1,109 queries; NER dev = 462 docs.

| Model | Where trained | Dev metric | Notes |
|---|---|---|---|
| spaCy LID (`models/lid_v3`) | local CPU | 94.9% acc | NE F1 52.9%; v2 was 98.1% → silver labels likely hurt |
| spaCy Intent (`models/intent_v3`) | local CPU | 71.25% acc (15 classes) | v2 was 70% on 5 classes |
| spaCy NER (`models/ner_v3`) | local CPU | 97.2% micro F1 | 9 entity types incl. social science |
| MuRIL LID (`models/muril_lid_v3`) | Colab T4, 5 epochs, ~11 min | **98.12% acc** (6,389 gold words) | HI 98.5 / EN 97.5 / NE 70.9 / UNIV 100 F1; from base `google/muril-base-cased` |
| MuRIL Intent (`models/muril_intent_v3`) | Colab T4, 10 epochs, ~33 min | **90.53% acc**, 76.46% macro F1 | from base MuRIL; weak on tiny classes: one_word 22% F1 (7 dev), numerical_problem 50% (6), assertion_reason 0% (1) |
| Gemma-3-4B-it QLoRA v2 (`gemma_eduhinglish_v2/adapter`, old ss9 data) | Colab T4, 500 steps (~0.8 epoch), ~77 min | eval loss 1.28 → **0.56** (still falling) | 6/6 hand-written Hinglish tests answer in fluent Hinglish and stay on the context; 2/4 held-out are bad because their reference answers are bad (see below) |
| **Gemma-3-4B-it QLoRA v3** (`gemma_eduhinglish_v3/adapter`, `bio_9_10` + `ss_9_10`, 2,178 entries) | Colab T4, 500 steps (~1.9 epochs), ~56 min | eval loss 1.58 → **1.41** (flat after step 400, no overfitting) | **Use this one.** 4/4 held-out answers in Hinglish and match the reference; no English context copying. Loss is not comparable with v2: v2's eval targets were easy-to-copy English. Issues: an invented fact in a real-life example, some odd Hindi phrasing (see below) |
| RAG (`src/knowledge_base`) | local GPU, 26 s | 6/6 build checks PASS; ss10 probes ("Federalism kya hota hai?", Non-Cooperation, Consumer rights) hit the right chapter | 2,261 chunks, 40 chapters (bio 9/10, ss9, ss10), rebuilt from Jatin's re-cleaned text |

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
- **(v2 only, fixed in v3 by Jatin's `ss_9_10.json`) ss9 Gemma training targets were noisy.** Gemma reproduced them faithfully:
  - 770 entries have no `answer_hinglish`, so the notebook trains on `answer_english` instead.
  - 76 more `answer_hinglish` values contain no Hindi words, so they are effectively English.
  - Together these teach the model to copy the English context back.
  - 10 answers contain degenerate repetition. Some Hinglish answers have garbled words ("derian zilch", "ekder"), and the model produced an "ekta ekta …" loop on one held-out item.
  - Templated questions like "What is the significance of X in <chapter>?" (259) and "Illustrate X with a real-life example" (192) often get a context summary as the answer, not a real answer to the question.
  - Fixes for the next run:
    - Fill or skip entries that have no Hinglish answer.
    - Train with completion-only loss.
    - Use `repetition_penalty` at inference.
- **Gemma v3 can invent facts when the context lacks them.**
  - Asked for a real-life Lok Sabha example, it made up a 2019 Delhi result: AAP with 58% and Raghav Chadha winning 3 seats. AAP won no Delhi seats in 2019.
  - In the pipeline, Gemma should answer only from RAG context, and `application_reallife` answers need a careful check.
- **Gemma v3 sometimes produces odd Hindi phrasing.**
  - Examples: "paani bhi bahut kharab kar dete hain" for "lose water", and "nikalna mushkil nahi hota". The sampled output (temperature 0.7) also produced "dhundh-dhaadi".
  - Use greedy decoding, which is cleaner. Grammar correction (M4) should target this.
- **Some ss10 cleaned text is still messy.**
  - Two-column PDF text is interleaved mid-sentence in places, e.g. "to tensions between the Dutchsmaller in area than the state of".
  - 6 chunks are teacher notes (`NOTES FOR THE TEACHER`, `Sources for Information`) at the start of the ss10 economics chapters.
  - Retrieval still works, but this belongs in `scripts/clean_ncert_text.py`.
- **"Barter system ki problem kya hai?" misses in RAG.** It is the same Hinglish rare-term problem as stomata: the top result is Federalism (0.40) and Money and Credit is second.
- **The intent model fails on tiny classes:** one_word, numerical_problem and assertion_reason. These need more training examples, not more epochs.

Full logs:
- spaCy run: `spacy_training_log.txt`
- MuRIL dev evaluation: `muril_eval_log.txt` (`python training/evaluate_muril.py`)
- MuRIL training: outputs saved in `training/colab_notebooks/01_train_muril_lid.ipynb` and `02_train_muril_intent.ipynb`
- Gemma v3 training and test generations: `training/colab_notebooks/04_gemma_finetune.ipynb` (last cell)
