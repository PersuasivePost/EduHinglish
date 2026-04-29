## Weekend Collaboration Plan — Phase 1 + 2 (Jatin & Ashvatth)

---

### Git Branching Strategy

First, set this up right now before anything else.

```bash
# Ashvatth does this on main (both pull after)
git checkout main
git pull origin main

# Create a shared staging branch — this is your "work in progress" branch
git checkout -b develop
git push origin develop

# Ashvatth creates his branch from develop
git checkout -b feature/ashvatth-extraction-pipeline
git push origin feature/ashvatth-extraction-pipeline

# Jatin creates his branch from develop
git checkout -b feature/jatin-dataset-builder
git push origin feature/jatin-dataset-builder
```

**Rule:** Nobody pushes directly to `main`. Everyone pushes to their feature branch → PR into `develop` → at end of weekend, one final PR from `develop` → `main`.

---

### Ownership Map — Zero Conflict Design

This is the most important part. Each person owns completely separate files/folders so there are **no merge conflicts**.

```
EduHinglish/
│
├── data/
│   ├── raw/                          ← ASHVATTH owns (PDFs go here)
│   ├── biology/
│   │   ├── class9/
│   │   │   ├── ch05/
│   │   │   │   ├── cleaned_text.txt  ← ASHVATTH produces
│   │   │   │   └── dataset.json      ← JATIN produces
│   │   │   ├── ch06/
│   │   │   │   ├── cleaned_text.txt  ← ASHVATTH produces
│   │   │   │   └── dataset.json      ← JATIN produces
│   │   │   └── ... (same pattern for all chapters)
│   │   └── class10/
│   │       └── ... (same pattern)
│   └── unified_dataset.json          ← BOTH merge at end (Sunday night)
│
├── scripts/
│   ├── batch_pdf_extractor.py        ← ASHVATTH writes
│   ├── validate_dataset.py           ← ASHVATTH writes
│   ├── augment_dataset.py            ← ASHVATTH writes
│   ├── annotation_helper.py          ← JATIN writes (CRITICAL - needed by both)
│   └── chapter_dataset_builder.py    ← JATIN writes
│
└── src/                              ← DO NOT TOUCH this weekend
```

**Key rule:** Ashvatth never touches `dataset.json`. Jatin never touches `cleaned_text.txt`. No conflicts possible.

---

### Chapter Assignment

```
ASHVATTH — extracts & cleans ALL chapters (PDF → cleaned_text.txt)
  Class 9:  Ch5, Ch6, Ch7, Ch13, Ch14, Ch15
  Class 10: Ch6, Ch8, Ch9, Ch15, Ch16

JATIN — builds dataset.json for assigned chapters
  Phase 1:  Ch5 (expand 15 → 75 sentences)
  Phase 2a: Ch6, Ch7
  Phase 2b: Ch13, Ch14, Ch15

ASHVATTH — also builds dataset.json for his assigned chapters
  Phase 2c: Ch6(10), Ch8(10), Ch9(10), Ch15(10), Ch16(10)
  + helps with Ch5 expansion (factual statements)
```

---

### Hour-by-Hour Weekend Plan

#### Friday Evening (2–3 hours)

**Both (together, 30 min):**
- Run the git commands above
- Create all chapter folders with empty `cleaned_text.txt` and `dataset.json` placeholder files
- Push folder structure to `develop` so both have it

```bash
# Ashvatth runs this to create folder structure
mkdir -p data/biology/class9/ch{05,06,07,13,14,15}
mkdir -p data/biology/class10/ch{06,08,09,15,16}
touch data/biology/class9/ch05/cleaned_text.txt
# ... repeat for all chapters
git add data/
git commit -m "chore: scaffold chapter folder structure"
git push origin develop
# Jatin: git pull origin develop
```

**Ashvatth (Friday evening, 1.5 hrs):**
- Write `scripts/batch_pdf_extractor.py`
- Run it on ALL chapter PDFs — produces `cleaned_text.txt` for every chapter
- Commit and push — **Jatin is now unblocked for all chapters**

```bash
git add scripts/batch_pdf_extractor.py data/biology/
git commit -m "feat(ashvatth): batch extract all chapter texts"
git push origin feature/ashvatth-extraction-pipeline
# Then open PR → develop and merge immediately so Jatin can pull
```

**Jatin (Friday evening, 1.5 hrs):**
- Write `scripts/annotation_helper.py` — this is the single most important tool
- Test it on Ch5 manually
- Commit and push immediately so Ashvatth can also use it Saturday

```bash
git add scripts/annotation_helper.py
git commit -m "feat(jatin): annotation helper CLI for dataset creation"
git push origin feature/jatin-dataset-builder
# PR → develop and merge so Ashvatth can pull Saturday morning
```

---

#### Saturday (Full Day)

**Morning sync (15 min, both):**
```bash
git checkout develop
git pull origin develop
git checkout feature/your-branch
git merge develop   # get each other's Friday work
```

**Ashvatth — Saturday:**

| Time | Task |
|------|------|
| 9am–12pm | Write `augment_dataset.py` + run on Ch5's 15 sentences → generate ~45 candidate variants to review |
| 12pm–1pm | Review and keep best 35 variants → Ch5 now has 50 entries |
| 1pm–2pm | Break |
| 2pm–5pm | Manually write Ch5 factual entries to reach 75 (25 more, ~5 mins each) |
| 5pm–7pm | Write `validate_dataset.py`, run on Ch5, fix issues |
| 7pm–9pm | Start Class 10 datasets using annotation_helper.py |

**Jatin — Saturday:**

| Time | Task |
|------|------|
| 9am–10am | Pull Ashvatth's cleaned texts, write `chapter_dataset_builder.py` |
| 10am–1pm | Ch5 student queries (13–17 style) + GEC samples (18–20 style) → 25+ entries |
| 1pm–2pm | Break |
| 2pm–5pm | Ch6 (Tissues) dataset — 50 sentences using annotation_helper.py |
| 5pm–7pm | Ch7 (Diversity in Living Organisms) — 50 sentences |
| 7pm–9pm | Review and validate Ch6 + Ch7 datasets |

**Saturday night commit (both, 9pm):**
```bash
git add .
git commit -m "feat(ashvatth/jatin): saturday progress - ch5 expanded, ch6, ch7 datasets"
git push origin feature/your-branch
# Both open PRs → develop, review each other's PR, merge
```

---

#### Sunday (Full Day)

**Morning sync (15 min, both):**
```bash
git checkout develop && git pull origin develop
git checkout feature/your-branch && git merge develop
```

**Ashvatth — Sunday:**

| Time | Task |
|------|------|
| 9am–12pm | Class 10: Ch6(Life Processes), Ch8(Reproduction) — 50 each |
| 12pm–1pm | Break |
| 1pm–4pm | Class 10: Ch9(Heredity), Ch15(Environment), Ch16(Natural Resources) — 50 each |
| 4pm–6pm | Write `validate_dataset.py` — runs on ALL datasets, prints CMI stats, missing fields |
| 6pm–8pm | Fix any validation errors across all your chapters |

**Jatin — Sunday:**

| Time | Task |
|------|------|
| 9am–12pm | Class 9: Ch13(Why Do We Fall Ill), Ch14(Natural Resources) — 50 each |
| 12pm–1pm | Break |
| 1pm–3pm | Class 9: Ch15(Improvement in Food Resources) — 50 entries |
| 3pm–5pm | Write `unified_dataset_builder.py` — merges all chapter JSONs into one |
| 5pm–7pm | Run unified builder, validate output, check total counts |
| 7pm–9pm | Fix any cross-chapter issues |

**Sunday night final merge (both together, 9pm):**
```bash
# Both push final feature branches
git add . && git commit -m "feat: complete phase 1+2 datasets" && git push

# Ashvatth does the final merge ceremony
git checkout develop && git pull origin develop
# Merge both feature branches → develop
# Run validate_dataset.py on unified output
# If all green → PR develop → main
git checkout main && git merge develop && git push origin main
```

---

### Commit Message Convention (follow this strictly)

```bash
# Format: type(author): description
feat(ashvatth): batch pdf extractor for all chapters
feat(jatin): ch6 tissues hinglish dataset 50 entries
fix(ashvatth): cleaned ch13 text encoding issue
chore(both): scaffold folder structure
data(jatin): ch7 diversity dataset complete
```

---

### Daily Sync Rule

Both push to your feature branch **at minimum** twice a day — morning and evening. If you finish a chapter dataset, push immediately. Don't sit on uncommitted work.

```bash
# Ashvatth's morning routine
git checkout develop && git pull origin develop
git checkout feature/ashvatth-extraction-pipeline
git merge develop
# work...
git add . && git commit -m "..." && git push
```

---

### What Each Person Needs to Set Up Right Now

**Ashvatth:**
1. All NCERT Biology PDFs placed in `data/raw/` with naming: `bio_9_ch05.pdf`, `bio_9_ch06.pdf`, `bio_10_ch06.pdf` etc.
2. Python environment working with `pdfplumber` installed
3. Test `pdf_extractor.py` works on Ch5

**Jatin:**
1. Pull the current repo and run existing `python scripts/download_nltk_data.py`
2. Open `src/hinglish_dataset_creator.py` and study the structure of entries 13–20 (queries + GEC) — these are the templates you'll scale
3. Plan your 50 entries per chapter: 30 factual statements + 15 student queries + 5 GEC samples

---

When you say go, I'll give both of you the exact Claude Code prompts — Ashvatth gets prompts for `batch_pdf_extractor.py`, `augment_dataset.py`, `validate_dataset.py`. Jatin gets prompts for `annotation_helper.py`, `chapter_dataset_builder.py`, and the dataset entries for each chapter. Run them independently, no waiting needed.