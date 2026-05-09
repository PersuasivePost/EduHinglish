First follow workspace-setup.md. This file contains instructions for both Ashvatth and Jatin.
Then start workiing on this exact prompts


Here are two complete, self-contained prompts. Ashvatth pastes Prompt A into his Claude Code session. Jatin pastes Prompt B into his. Neither needs to wait for the other to start.

---

## PROMPT A — Ashvatth

Copy everything between the lines and paste into Claude Code:

---

```
I am Ashvatth, working on a final year engineering project called EduHinglish 
— an AI-powered learning companion for Hinglish-speaking Indian students 
(Hindi-English code-mixed language). My teammate Jatin is working in parallel 
on a separate branch. This prompt is my complete independent work scope for 
this weekend.

═══════════════════════════════════════════════════════
PROJECT CONTEXT
═══════════════════════════════════════════════════════

EduHinglish is a 5-module system:
  M1 - Input Processing (preprocessing pipeline) ← NEARLY DONE
  M2 - NCERT Retriever (RAG with ChromaDB)
  M3 - Hinglish Answer Generator (IndicBART fine-tuned)
  M4 - GEC Engine (grammar error correction for Hinglish)
  M5 - Streamlit UI

NO external LLM APIs are used. Everything is local or Google Colab/Kaggle.
No OpenAI, no Anthropic API, no paid services.

Current codebase already has:
  src/preprocessing.py         — English NLP pipeline (tokenize, stem, lemmatize, POS)
  src/script_detector.py       — Script detection, normalization, rule-based LID
  src/pipeline.py              — Unified pipeline combining both
  src/hinglish_dataset_creator.py — 15 labeled Hinglish sentences for Ch5
  src/pdf_extractor.py         — Single PDF extractor (already working)
  requirements.txt             — pdfplumber, nltk, spacy, etc.
  tests/                       — test_preprocessing.py, test_pipeline.py, etc.

Dataset structure for each entry (from hinglish_dataset_creator.py):
{
  "id": 1,
  "original_english": "All living organisms are made up of cells.",
  "hinglish_roman": "Sabhi living organisms cells se bane hote hain.",
  "hinglish_devanagari": "सभी living organisms cells से बने होते हैं।",
  "word_level_labels": {
    "Sabhi": "HI", "living": "EN", "organisms": "EN",
    "cells": "EN", "se": "HI", "bane": "HI", "hote": "HI", "hain": "HI"
  },
  "topic": "Introduction to Cell",
  "chapter": "Chapter 5: The Fundamental Unit of Life",
  "class": "9",
  "code_mixing_type": "intra-sentential",
  "notes": "..."
}

Student query entries also have: "is_student_query": true, "intent": "explain_concept"
GEC entries also have: "is_gec_sample": true, "hinglish_with_error": "...", "error_description": {...}

Word label tags:
  HI   = Hindi word in Roman script
  EN   = English word
  NE   = Named Entity (person, place, org)
  UNIV = Language-independent (numbers, Sir, OK)
  MIX  = Mixed-script word

═══════════════════════════════════════════════════════
MY ROLE THIS WEEKEND (ASHVATTH ONLY)
═══════════════════════════════════════════════════════

I own these files — only I touch these:
  scripts/batch_pdf_extractor.py       ← I write this
  scripts/augment_dataset.py           ← I write this
  scripts/validate_dataset.py          ← I write this
  data/raw/                            ← I place PDFs here
  data/biology/class9/ch*/cleaned_text.txt    ← I produce these
  data/biology/class10/ch*/cleaned_text.txt   ← I produce these
  data/biology/class10/ch*/dataset.json       ← I produce these (Class 10 only)
  data/biology/class9/ch05/dataset_v2.json    ← I contribute factual statements

I DO NOT TOUCH these (Jatin's territory):
  src/script_detector.py
  src/preprocessing.py
  src/pipeline.py
  src/hinglish_dataset_creator.py
  scripts/annotation_helper.py
  scripts/chapter_dataset_builder.py
  data/biology/class9/ch06/dataset.json
  data/biology/class9/ch07/dataset.json
  data/biology/class9/ch13/dataset.json
  data/biology/class9/ch14/dataset.json
  data/biology/class9/ch15/dataset.json

═══════════════════════════════════════════════════════
CHAPTER LIST — COMPLETE SCOPE
═══════════════════════════════════════════════════════

Class 9 Biology chapters:
  ch05 — The Fundamental Unit of Life (Cell Biology)
  ch06 — Tissues
  ch07 — Diversity in Living Organisms
  ch13 — Why Do We Fall Ill
  ch14 — Natural Resources
  ch15 — Improvement in Food Resources

Class 10 Biology chapters:
  ch06 — Life Processes
  ch08 — How do Organisms Reproduce?
  ch09 — Heredity and Evolution
  ch15 — Our Environment
  ch16 — Management of Natural Resources

PDF naming convention I will use:
  data/raw/bio_9_ch05.pdf
  data/raw/bio_9_ch06.pdf
  data/raw/bio_9_ch07.pdf
  data/raw/bio_9_ch13.pdf
  data/raw/bio_9_ch14.pdf
  data/raw/bio_9_ch15.pdf
  data/raw/bio_10_ch06.pdf
  data/raw/bio_10_ch08.pdf
  data/raw/bio_10_ch09.pdf
  data/raw/bio_10_ch15.pdf
  data/raw/bio_10_ch16.pdf

═══════════════════════════════════════════════════════
DATA FOLDER STRUCTURE TO CREATE
═══════════════════════════════════════════════════════

data/
├── raw/                          (PDFs go here - I add them manually)
├── biology/
│   ├── class9/
│   │   ├── ch05/
│   │   │   ├── cleaned_text.txt
│   │   │   ├── raw_text.txt
│   │   │   └── dataset_v2.json   (expanded from 15 → 75 entries)
│   │   ├── ch06/
│   │   │   ├── cleaned_text.txt
│   │   │   └── raw_text.txt
│   │   ├── ch07/
│   │   │   ├── cleaned_text.txt
│   │   │   └── raw_text.txt
│   │   ├── ch13/
│   │   │   ├── cleaned_text.txt
│   │   │   └── raw_text.txt
│   │   ├── ch14/
│   │   │   ├── cleaned_text.txt
│   │   │   └── raw_text.txt
│   │   └── ch15/
│   │       ├── cleaned_text.txt
│   │       └── raw_text.txt
│   └── class10/
│       ├── ch06/
│       │   ├── cleaned_text.txt
│       │   ├── raw_text.txt
│       │   └── dataset.json      (I write Class 10 datasets)
│       ├── ch08/
│       │   ├── cleaned_text.txt
│       │   ├── raw_text.txt
│       │   └── dataset.json
│       ├── ch09/
│       │   ├── cleaned_text.txt
│       │   ├── raw_text.txt
│       │   └── dataset.json
│       ├── ch15/
│       │   ├── cleaned_text.txt
│       │   ├── raw_text.txt
│       │   └── dataset.json
│       └── ch16/
│           ├── cleaned_text.txt
│           ├── raw_text.txt
│           └── dataset.json
└── unified_dataset.json          (built at the end from all datasets)

═══════════════════════════════════════════════════════
TASK 1 — CREATE FOLDER STRUCTURE
═══════════════════════════════════════════════════════

First task: Create ALL the folders and placeholder files listed above.
For placeholder files write a single line: "# placeholder - to be generated"
Commit message format: "chore(ashvatth): scaffold chapter folder structure"

═══════════════════════════════════════════════════════
TASK 2 — scripts/batch_pdf_extractor.py
═══════════════════════════════════════════════════════

Build this script. It should:

1. Scan data/raw/ for all PDFs matching pattern bio_CLASS_chCH.pdf
2. For each PDF found, extract and clean text using pdfplumber
3. Save raw_text.txt and cleaned_text.txt to the correct chapter folder
4. Print a progress summary at the end

Cleaning steps (same as existing src/pdf_extractor.py but batch):
  - Remove NCERT headers/running titles
  - Remove standalone page numbers
  - Remove figure/table captions (Fig. X.X, Table X.X)
  - Fix hyphenation breaks (mem-\nbrane → membrane)
  - Fix mid-sentence line breaks
  - Normalize quotes and dashes
  - Collapse excessive whitespace
  - Remove non-content special characters

CLI usage:
  python scripts/batch_pdf_extractor.py
  python scripts/batch_pdf_extractor.py --raw-dir data/raw --output-dir data/biology

Output printed per chapter:
  [OK] bio_9_ch06.pdf → data/biology/class9/ch06/ (4,231 chars cleaned)
  [SKIP] bio_9_ch07.pdf → cleaned_text.txt already exists, use --force to overwrite

If a PDF is missing, print:
  [MISSING] bio_9_ch13.pdf → place PDF in data/raw/ and rerun

At end print a summary table:
  Processed: X chapters
  Skipped:   X (already extracted)
  Missing:   X PDFs not found

═══════════════════════════════════════════════════════
TASK 3 — scripts/augment_dataset.py
═══════════════════════════════════════════════════════

Build this script. Purpose: take existing labeled sentences and generate
variants automatically to assist in expanding Ch5 from 15 to 75 entries.

Input: data/biology/class9/ch05/ directory (reads hinglish_dataset_creator.py 
       dataset by importing it, or reads a JSON file if present)

Augmentation strategies:
  1. Hindi density shift — replace some EN content words with their Hindi equivalents
     Example: "Cell membrane" → "Koshika membrane" (add HI label)
  2. Question form conversion — convert a statement to a student query
     Example: "Mitochondria produce ATP" → "Sir, mitochondria ATP kaise produce karte hain?"
     Add: is_student_query=True, intent="explain_concept"
  3. Connector swap — swap one Hindi connector for another valid one
     "aur" ↔ "tatha", "lekin" ↔ "parantu", "kyunki" ↔ "isliye"
  4. Formality shift — add "Sir," prefix and "samjhao"/"batao" suffix to statements
  5. Negation form — convert positive to negative
     "Cell wall hoti hai" → "Cell wall nahi hoti"

For each original sentence, generate 2-3 variants.
Each variant must:
  - Have a new unique id (original_id + letter: 1a, 1b, 1c)
  - Update word_level_labels correctly for changed words
  - Have a "augmented_from": original_id field
  - Have "augmentation_type": "question_form" / "connector_swap" etc.

Output: data/biology/class9/ch05/augmented_candidates.json
        (these are CANDIDATES — human reviews and keeps the good ones)

Print at end:
  Original sentences: 15
  Candidates generated: 42
  Review them in: data/biology/class9/ch05/augmented_candidates.json
  Run merge command after review: python scripts/augment_dataset.py --merge

Also implement --merge flag that:
  - Reads augmented_candidates.json
  - Asks for each candidate: Keep? (y/n/e to edit)
  - Saves approved ones to data/biology/class9/ch05/dataset_v2.json

═══════════════════════════════════════════════════════
TASK 4 — scripts/validate_dataset.py
═══════════════════════════════════════════════════════

Build a validation script that checks ALL dataset.json files across all chapters.

Validations to perform per entry:
  1. Required fields present: id, original_english, hinglish_roman, 
     word_level_labels, topic, chapter, class
  2. No duplicate IDs within a file
  3. word_level_labels keys match actual words in hinglish_roman (tokenized by space)
  4. All label values are valid: HI, EN, NE, UNIV, MIX
  5. Entries with is_student_query=True must have an intent field
  6. Entries with is_gec_sample=True must have hinglish_with_error and error_description
  7. CMI (Code-Mixing Index) is computable and reasonable (between 10% and 70%)
  8. hinglish_roman is not identical to original_english

Per-dataset stats to print:
  Chapter: class9/ch05
  Total entries:      75
  Statements:         50
  Student queries:    15
  GEC samples:        10
  Avg CMI:            38.4%
  Validation issues:   0
  ──────────────────────

Cross-dataset stats at end:
  Total entries across all chapters: 625
  Chapters complete (≥50 entries):   8 / 11
  Chapters incomplete (<50 entries): 3
  Total validation issues:           2

CLI:
  python scripts/validate_dataset.py               # validates all
  python scripts/validate_dataset.py --chapter class9/ch05   # single chapter
  python scripts/validate_dataset.py --fix         # auto-fix trivial issues

═══════════════════════════════════════════════════════
TASK 5 — Class 10 Datasets (dataset.json for each chapter)
═══════════════════════════════════════════════════════

After Jatin's annotation_helper.py is merged to develop (Friday night),
pull it and use it to create these datasets:

  data/biology/class10/ch06/dataset.json  — Life Processes (50 entries)
  data/biology/class10/ch08/dataset.json  — How do Organisms Reproduce? (50)
  data/biology/class10/ch09/dataset.json  — Heredity and Evolution (50)
  data/biology/class10/ch15/dataset.json  — Our Environment (50)
  data/biology/class10/ch16/dataset.json  — Management of Natural Resources (50)

Each dataset must have:
  - 30 factual statements (biology concepts from the chapter)
  - 15 student queries (natural Hinglish doubt questions)
  - 5 GEC samples (sentences with common Hindi grammar errors)
  Total: 50 entries minimum per chapter

Same JSON structure as existing hinglish_dataset_creator.py entries.
Chapter-specific IDs: use prefix like "10_06_001" for Class 10 Ch6 entry 1.

For Life Processes topics: nutrition, respiration, transportation, excretion
For Reproduction topics: vegetative, sexual, asexual, human reproduction
For Heredity topics: Mendel's laws, chromosomes, evolution, natural selection
For Environment topics: ecosystem, food chain, ozone, waste management
For Natural Resources topics: water, coal, petroleum, conservation

═══════════════════════════════════════════════════════
GIT WORKFLOW
═══════════════════════════════════════════════════════

Branch: feature/ashvatth-extraction-pipeline
Base:   develop

Commit after each completed task:
  git add . && git commit -m "feat(ashvatth): [description]"
  git push origin feature/ashvatth-extraction-pipeline

Commit message examples:
  feat(ashvatth): scaffold chapter folder structure
  feat(ashvatth): batch pdf extractor for all 11 chapters
  feat(ashvatth): augment_dataset script with 5 augmentation strategies
  feat(ashvatth): validate_dataset cross-chapter stats
  data(ashvatth): class10 ch06 life processes dataset 50 entries
  data(ashvatth): class10 all chapters complete

After Friday night tasks are done, open PR from your branch → develop
so Jatin can pull annotation_helper and you can pull his cleaned texts.

═══════════════════════════════════════════════════════
START HERE — EXACT ORDER TO EXECUTE
═══════════════════════════════════════════════════════

Step 1: Create folder structure + placeholder files
Step 2: Write scripts/batch_pdf_extractor.py
Step 3: Write scripts/augment_dataset.py
Step 4: Write scripts/validate_dataset.py
Step 5: Run batch_pdf_extractor.py on all available PDFs
Step 6: Run augment_dataset.py on Ch5, review candidates
Step 7: Pull Jatin's annotation_helper.py from develop (Saturday morning)
Step 8: Use annotation_helper.py to build Class 10 datasets
Step 9: Run validate_dataset.py on everything, fix errors
Step 10: Final commit + push + PR → develop

Build all 4 scripts completely first (Steps 1-4), then I will run them.
For the dataset entries in Task 5, generate realistic, educationally 
accurate Hinglish sentences about each chapter's biology topics.
Follow the exact JSON structure from hinglish_dataset_creator.py.
Do not invent incorrect biology — stick to NCERT Class 10 content.
```

---

## PROMPT B — Jatin

Copy everything between the lines and paste into Claude Code:

---

```
I am Jatin, working on a final year engineering project called EduHinglish 
— an AI-powered learning companion for Hinglish-speaking Indian students 
(Hindi-English code-mixed language). My teammate Ashvatth is working in 
parallel on a separate branch. This prompt is my complete independent work 
scope for this weekend.

═══════════════════════════════════════════════════════
PROJECT CONTEXT
═══════════════════════════════════════════════════════

EduHinglish is a 5-module system:
  M1 - Input Processing (preprocessing pipeline) ← NEARLY DONE
  M2 - NCERT Retriever (RAG with ChromaDB)
  M3 - Hinglish Answer Generator (IndicBART fine-tuned)
  M4 - GEC Engine (grammar error correction for Hinglish)
  M5 - Streamlit UI

NO external LLM APIs are used. Everything is local or Google Colab/Kaggle.
No OpenAI, no Anthropic API, no paid services.

I (Jatin) originally wrote:
  src/script_detector.py       — Script detection, HinglishNormalizer, WordLevelLID
  src/hinglish_dataset_creator.py — 20 labeled Hinglish sentences for Ch5

Ashvatth originally wrote:
  src/preprocessing.py         — English NLP pipeline
  src/pipeline.py              — Unified pipeline
  src/pdf_extractor.py         — PDF text extractor

Dataset entry structure I created (must follow exactly for consistency):
{
  "id": 1,
  "original_english": "All living organisms are made up of cells.",
  "hinglish_roman": "Sabhi living organisms cells se bane hote hain.",
  "hinglish_devanagari": "सभी living organisms cells से बने होते हैं।",
  "word_level_labels": {
    "Sabhi": "HI", "living": "EN", "organisms": "EN",
    "cells": "EN", "se": "HI", "bane": "HI", "hote": "HI", "hain": "HI"
  },
  "topic": "Introduction to Cell",
  "chapter": "Chapter 5: The Fundamental Unit of Life",
  "class": "9",
  "code_mixing_type": "intra-sentential",
  "notes": "Technical terms kept in English; Hindi provides grammatical structure"
}

Student query entries additionally have:
  "is_student_query": true
  "intent": "explain_concept"   (or compare_concepts / give_example / 
                                  formula_request / definition)

GEC sample entries additionally have:
  "is_gec_sample": true
  "hinglish_with_error": "Cell wall plant cells mein hota hai..."
  "error_description": {
    "error_word": "hote",
    "correct_word": "hoti",
    "error_type": "gender agreement",
    "explanation": "'cell wall' is feminine → 'hoti' not 'hote'"
  }

Word label tags:
  HI   = Hindi word in Roman script (mein, hai, aur, ka, ki, ke, ko, se...)
  EN   = English word or science term
  NE   = Named Entity (Mendel, Darwin, Robert Hooke)
  UNIV = Language-independent (Sir, numbers, OK, please)
  MIX  = Mixed-script word (rare)

Natural Hinglish rules I follow:
  1. Science/biology terms stay in English (always EN tag)
  2. Hindi provides grammar: postpositions (ka/ki/ke/ko/se/mein), 
     verbs (hai/hain/hota/hoti/karta/karti/karte), conjunctions (aur/lekin/kyunki)
  3. Student queries start with "Sir," (UNIV tag for Sir)
  4. Natural ratio: ~40-60% EN, ~40-55% HI, ~0-5% NE/UNIV

═══════════════════════════════════════════════════════
MY ROLE THIS WEEKEND (JATIN ONLY)
═══════════════════════════════════════════════════════

I own these files — only I touch these:
  scripts/annotation_helper.py          ← I write this FIRST (Ashvatth needs it)
  scripts/chapter_dataset_builder.py    ← I write this
  scripts/unified_dataset_builder.py    ← I write this (Sunday)
  data/biology/class9/ch05/dataset_v2.json  ← I expand to 75 entries
  data/biology/class9/ch06/dataset.json     ← I produce (Ch6 Tissues)
  data/biology/class9/ch07/dataset.json     ← I produce (Ch7 Diversity)
  data/biology/class9/ch13/dataset.json     ← I produce (Ch13 Why Do We Fall Ill)
  data/biology/class9/ch14/dataset.json     ← I produce (Ch14 Natural Resources)
  data/biology/class9/ch15/dataset.json     ← I produce (Ch15 Improvement in Food)

I DO NOT TOUCH these (Ashvatth's territory):
  src/preprocessing.py
  src/pipeline.py
  src/pdf_extractor.py
  scripts/batch_pdf_extractor.py
  scripts/augment_dataset.py
  scripts/validate_dataset.py
  data/biology/class10/     (entire Class 10 folder = Ashvatth's)
  data/raw/                 (Ashvatth places PDFs here)

═══════════════════════════════════════════════════════
TASK 1 — scripts/annotation_helper.py  ← DO THIS FIRST
═══════════════════════════════════════════════════════

This is the most important tool this weekend. Both Ashvatth and I will 
use it to create datasets. Build it as a CLI tool that makes manual 
annotation fast.

WORKFLOW it implements:
  1. User provides: English sentence + typed Hinglish version
  2. Tool auto-splits Hinglish into words
  3. For each word, tool makes a prediction (HI/EN/NE/UNIV) using 
     simple rules (check against a built-in Hindi word list + English suffix list)
  4. Shows prediction and asks user to confirm or correct
  5. Builds the complete entry dict
  6. Asks for: topic, is_student_query, intent (if query), is_gec_sample
  7. Saves to specified output JSON file (appending to array)
  8. Shows running count of entries saved

Built-in Hindi word list for auto-prediction (use this exact list):
  HINDI_WORDS = {
    "main","hum","tum","woh","yeh","uska","uski","iska","iski",
    "mera","meri","tera","teri","unka","unki","hamara","tumhara",
    "hai","hain","tha","thi","the","hoga","hogi","hota","hoti","hote",
    "karta","karti","karte","kiya","karo","karna","karke","hona",
    "raha","rahi","rahe","gaya","gayi","aata","aati","jaata","jaati",
    "deta","deti","lete","bana","bante","samjhao","batao","dekho",
    "kehte","kehta","kehti","kaha","dijiye","hokar","jinmein","inmein",
    "ka","ki","ke","ko","se","mein","par","tak","pe","ne","me",
    "aur","ya","lekin","kyunki","isliye","jabki","phir","toh","bhi",
    "hi","sirf","bas","kya","kaise","kyun","kahan","kab","kaun",
    "kitna","bahut","thoda","zyada","kam","achha","bada","bade","badi",
    "chhota","naya","nayi","pehle","baad","andar","bahar","upar",
    "neeche","yahan","wahan","abhi","tab","jab","nahi","nhi","na",
    "mat","ek","do","teen","sabhi","sab","kuch","koi","wala","wale",
    "wali","jaise","taraf","beech","kaam","saath","jinmein","inmein",
    "jo","jo","jabki","tarah","matlab","yaani"
  }

UNIVERSAL_WORDS = {"sir","madam","ok","okay","hello","hi","bye","please","thanks","sorry"}

Auto-prediction logic:
  - Contains Devanagari chars → HI
  - In UNIVERSAL_WORDS → UNIV  
  - In HINDI_WORDS → HI
  - Is a number → UNIV
  - Starts with capital, not first word → NE (candidate)
  - Otherwise → EN

Display format during annotation session:
  ─────────────────────────────────────────
  Word 1/8: "Sabhi"
  Prediction: HI  ← accept? [Enter=yes / type label to change]: 
  ─────────────────────────────────────────
  Word 2/8: "living"
  Prediction: EN  ← accept? [Enter=yes / type label to change]: 

After all words:
  ─────────────────────────────────────────
  Labels: {'Sabhi': 'HI', 'living': 'EN', ...}
  Topic: Introduction to Cell
  Is student query? [y/n]: n
  Is GEC sample? [y/n]: n
  Code mixing type [intra/inter]: intra
  Notes (optional): 
  ─────────────────────────────────────────
  [SAVED] Entry #16 → data/biology/class9/ch05/dataset_v2.json
  Total entries in file: 16
  ─────────────────────────────────────────
  Next sentence? [Enter=continue / q=quit]: 

CLI:
  python scripts/annotation_helper.py --output data/biology/class9/ch05/dataset_v2.json
  python scripts/annotation_helper.py --output data/biology/class9/ch06/dataset.json

Also support --chapter flag to auto-fill chapter and class fields:
  python scripts/annotation_helper.py --chapter class9/ch06 --output data/biology/class9/ch06/dataset.json

When --chapter is given, fill:
  "chapter": "Chapter 6: Tissues"   (map from chapter code)
  "class": "9"

Chapter code → title mapping:
  ix_chap1_science → "Entering the World of Secondary Science"
  ix_chap2_science → "Cell: The Building Block of Life"
  ix_chap3_science → "Tissues in Action"
  ix_chap11_science → "Reproduction: How Life Continues"
  ix_chap12_science → "Patterns in Life: Diversity and Classification"
  x_chap5_science → "Life Processes"
  x_chap6_science → "Control and Coordination"
  x_chap7_science → "How do Organisms Reproduce?"
  x_chap8_science → "Heredity"
  x_chap13_science → "Our Environment"

═══════════════════════════════════════════════════════
TASK 2 — scripts/chapter_dataset_builder.py
═══════════════════════════════════════════════════════

A guided batch entry tool — more structured than annotation_helper.
Used when I want to enter many sentences for a new chapter at once.

It reads the chapter's cleaned_text.txt (produced by Ashvatth's 
batch_pdf_extractor.py) and shows key sentences from it as prompts.

Workflow:
  1. Read cleaned_text.txt for specified chapter
  2. Extract candidate English sentences (longer than 50 chars, 
     containing biology terms) — show 5 at a time as reference
  3. User types the Hinglish version
  4. Calls annotation_helper logic for labeling
  5. Tracks: statements added, queries added, GEC samples added
  6. Shows progress bar toward 50 entry target: [████████░░] 40/50

CLI:
  python scripts/chapter_dataset_builder.py --chapter class9/ch06

At start shows:
  ══════════════════════════════════════
  Chapter 6: Tissues
  Target: 30 statements + 15 queries + 5 GEC = 50 entries
  Output: data/biology/class9/ch06/dataset.json
  ══════════════════════════════════════
  
  REFERENCE SENTENCES FROM NCERT TEXT:
  [1] "A group of cells that are similar in structure and/or work together 
       to achieve a particular function forms a tissue."
  [2] "Plant tissues are of two main types – meristematic and permanent."
  ...
  
  Enter Hinglish version for [1] (or 's' to skip, 'q' to quit):

═══════════════════════════════════════════════════════
TASK 3 — scripts/unified_dataset_builder.py
═══════════════════════════════════════════════════════

Build Sunday night — merges all chapter datasets into one unified file.

Logic:
  1. Scan all data/biology/class*/ch*/dataset*.json files
  2. Load each, add metadata: "source_file", "subject": "biology"
  3. Re-assign global IDs (bio_9_05_001, bio_9_06_001, etc.)
  4. Merge into single list
  5. Print summary table:
     ┌─────────────────────────────────────────────────────┐
     │ Chapter                          │ Entries │ Status │
     ├─────────────────────────────────────────────────────┤
     │ Class 9 Ch5: Fundamental Unit    │  75     │  ✓     │
     │ Class 9 Ch6: Tissues             │  50     │  ✓     │
     │ Class 9 Ch7: Diversity           │  50     │  ✓     │
     │ Class 9 Ch13: Why Do We Fall Ill │  50     │  ✓     │
     │ Class 9 Ch14: Natural Resources  │  50     │  ✓     │
     │ Class 9 Ch15: Food Resources     │  50     │  ✓     │
     │ Class 10 Ch6: Life Processes     │  50     │  ✓     │
     │ Class 10 Ch8: Reproduction       │  50     │  ✓     │
     │ Class 10 Ch9: Heredity           │  50     │  ✓     │
     │ Class 10 Ch15: Environment       │  50     │  ✓     │
     │ Class 10 Ch16: Natural Resources │  50     │  ✓     │
     ├─────────────────────────────────────────────────────┤
     │ TOTAL                            │ 575     │        │
     └─────────────────────────────────────────────────────┘
  6. Save to data/unified_biology_dataset.json

CLI:
  python scripts/unified_dataset_builder.py
  python scripts/unified_dataset_builder.py --output data/unified_biology_dataset.json

═══════════════════════════════════════════════════════
TASK 4 — Dataset Content: Ch5 Expansion (dataset_v2.json)
═══════════════════════════════════════════════════════

The current ch05 dataset has 20 entries (ids 1-20) in 
src/hinglish_dataset_creator.py. I need to expand to 75 entries.
Ashvatth will contribute ~25 factual statements. I contribute:
  - 15 more factual statements (topics not yet covered)
  - 20 more student queries (variety of intents)
  - 5 more GEC samples (different error types)
  = 40 new entries from me → total becomes ~75 with Ashvatth's 25

Topics still needed for Ch5 (not in current 20 entries):
  - Endoplasmic Reticulum (SER vs RER)
  - Ribosomes
  - Cell division (why cells divide)
  - Protoplasm vs cytoplasm
  - Nuclear membrane structure
  - Centrosome
  - Differences between plant and animal cells (tabular)
  - Turgor pressure
  - Plasmolysis
  - Cell theory (all 3 points)

Generate these as properly labeled JSON entries.
Use IDs starting from 21 (since 1-20 already exist).
Save to data/biology/class9/ch05/dataset_v2.json
(This file starts fresh — do NOT import from hinglish_dataset_creator.py,
 Ashvatth will merge at the end)

═══════════════════════════════════════════════════════
TASK 5 — Class 9 Chapter Datasets
═══════════════════════════════════════════════════════

Create 50-entry datasets for each assigned Class 9 chapter.
Each must have: 30 factual statements + 15 student queries + 5 GEC samples.

── Ch6: Tissues ──
Key topics: meristematic tissue, permanent tissue, simple permanent tissue,
complex permanent tissue (xylem, phloem), epithelial tissue, connective tissue,
muscle tissue, nervous tissue, differences plant vs animal tissue

── Ch7: Diversity in Living Organisms ──  
Key topics: basis of classification, Whittaker's 5 kingdoms (Monera, Protista,
Fungi, Plantae, Animalia), Plantae (Thallophyta, Bryophyta, Pteridophyta,
Gymnosperms, Angiosperms), Animalia (Porifera, Coelenterata, Platyhelminthes,
Nematoda, Annelida, Arthropoda, Mollusca, Echinodermata, Protochordata,
Vertebrata), binomial nomenclature, Linnaeus

── Ch13: Why Do We Fall Ill ──
Key topics: health vs disease, infectious vs non-infectious disease, 
acute vs chronic disease, pathogens (bacteria, virus, fungi, protozoa, worms),
means of spread, organ-specific vs tissue-specific, principles of treatment,
principles of prevention, vaccination, antibiotics, immune system

── Ch14: Natural Resources ──
Key topics: biosphere, atmosphere layers, nitrogen cycle, carbon cycle,
water cycle, oxygen cycle, soil formation, biogeochemical cycles, 
role of microorganisms, greenhouse effect, ozone layer

── Ch15: Improvement in Food Resources ──
Key topics: crop improvement (hybridization, GMO), crop protection 
(pesticides, weedicides), animal husbandry, poultry, fisheries, 
apiculture, food storage, manure vs fertilizer, irrigation methods,
mixed farming, intercropping

For each chapter save to:
  data/biology/class9/ch06/dataset.json
  data/biology/class9/ch07/dataset.json
  data/biology/class9/ch13/dataset.json
  data/biology/class9/ch14/dataset.json
  data/biology/class9/ch15/dataset.json

Use chapter-specific ID prefixes:
  Ch6  → ids: "9_06_001", "9_06_002", ...
  Ch7  → ids: "9_07_001", ...
  Ch13 → ids: "9_13_001", ...
  Ch14 → ids: "9_14_001", ...
  Ch15 → ids: "9_15_001", ...

Generate biologically accurate Hinglish sentences.
Key Hindi words to use as connectors (always tag HI):
  ka, ki, ke, ko, se, mein, par, tak, ne, aur, ya, lekin,
  kyunki, jabki, hai, hain, hota, hoti, hote, karta, karti,
  karte, kiya, yeh, woh, jo, jaise, sirf, bhi, nahi, ek

Key biology terms to always tag EN (never translate):
  All organism names, tissue names, process names, scientist names,
  classification terms, scientific concepts from the chapter

═══════════════════════════════════════════════════════
GIT WORKFLOW
═══════════════════════════════════════════════════════

Branch: feature/jatin-dataset-builder
Base:   develop

Commit after each completed task:
  git add . && git commit -m "feat(jatin): [description]"
  git push origin feature/jatin-dataset-builder

Commit message examples:
  feat(jatin): annotation_helper CLI with auto-prediction
  feat(jatin): chapter_dataset_builder with ncert reference display
  feat(jatin): unified_dataset_builder merge script
  data(jatin): ch05 dataset expanded to 75 entries
  data(jatin): ch06 tissues dataset 50 entries complete
  data(jatin): ch07 diversity dataset 50 entries complete
  data(jatin): class9 all chapters complete

CRITICAL — after annotation_helper.py is done (Friday night):
  git add scripts/annotation_helper.py
  git commit -m "feat(jatin): annotation_helper - needed by ashvatth"
  git push origin feature/jatin-dataset-builder
  Open PR → develop and merge immediately so Ashvatth can pull it

═══════════════════════════════════════════════════════
START HERE — EXACT ORDER TO EXECUTE
═══════════════════════════════════════════════════════

Step 1: Write scripts/annotation_helper.py  ← DO THIS FIRST
Step 2: Test annotation_helper.py manually (enter 2-3 sentences)
Step 3: Commit and push annotation_helper.py immediately 
        (Ashvatth is waiting for this Friday night)
Step 4: Write scripts/chapter_dataset_builder.py
Step 5: Generate data/biology/class9/ch05/dataset_v2.json (40 new entries)
Step 6: Generate data/biology/class9/ch06/dataset.json (50 entries)
Step 7: Generate data/biology/class9/ch07/dataset.json (50 entries)
Step 8: Generate data/biology/class9/ch13/dataset.json (50 entries)
Step 9: Generate data/biology/class9/ch14/dataset.json (50 entries)
Step 10: Generate data/biology/class9/ch15/dataset.json (50 entries)
Step 11: Write scripts/unified_dataset_builder.py (Sunday)
Step 12: Run unified builder after Ashvatth's Class 10 datasets are merged
Step 13: Final commit + push + PR → develop

Build scripts first (Steps 1-4), then generate all dataset content.
For dataset content, generate realistic, educationally accurate Hinglish 
sentences based on NCERT Class 9 Biology syllabus.
All biology facts must be accurate — stick to NCERT content only.
Do not invent biology that is not in the NCERT curriculum.
```

---

## After Both Are Done — Sunday Night Merge Sequence

Ashvatth does this (in order):

```bash
# 1. Push your final branch
git add . && git commit -m "feat(ashvatth): phase 1+2 complete" 
git push origin feature/ashvatth-extraction-pipeline

# 2. Merge Jatin's branch into develop first
git checkout develop
git pull origin develop   # gets Jatin's merged work

# 3. Merge your branch
git merge feature/ashvatth-extraction-pipeline

# 4. Run validation
python scripts/validate_dataset.py

# 5. Run unified builder
python scripts/unified_dataset_builder.py

# 6. If all green
git add . && git commit -m "chore: phase 1+2 unified dataset complete"
git push origin develop

# 7. Final PR: develop → main
```