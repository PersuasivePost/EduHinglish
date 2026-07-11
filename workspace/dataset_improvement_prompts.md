# EduHinglish — Dataset Improvement Prompts (Phase 6 v2)

> **Goal:** Expand from ~935 valid samples → ~1,800–2,000 samples with:
> - 35%+ student Q&A pairs (currently 14.4%)
> - 20%+ inter-sentential mixing (currently 5.5%)
> - Balanced length ratios (Hinglish/English: 0.6×–1.8×)
> - Zero null/garbage entries

---

## HOW TO USE THESE PROMPTS

1. Open a **new Claude conversation** for each chapter
2. Paste the **Master System Prompt** first (one time at the top)
3. Paste the chapter-specific prompt below it
4. **Copy the full content of `data/processed/<chapter_folder>/cleaned_text.txt`** and paste it wrapped in `<chapter_text>` tags (see format in each chapter prompt)
5. Download the JSON output and save as `data/chapter_additions/<chapter_id>_additions.json`
6. After all chapters: run the merge script (Step 11)

---

## MASTER SYSTEM PROMPT
> Paste this at the start of every Claude conversation before the chapter prompt.

```
You are a dataset creator for EduHinglish, an AI system that helps Indian students understand
science concepts through Hinglish (a natural mix of Hindi and English used in Indian classrooms).

The dataset covers:
- NCERT Class 9 Biology chapters (class: "9")
- NCERT Class 10 Biology chapters (class: "10")
- General Science chapters that cross-reference multiple subjects (class: "general")
  These are NOT Biology-only — they may cover physics, chemistry, biology, earth science equally.

HINGLISH RULES — follow these strictly:
1. Keep scientific/technical terms in English (photosynthesis, mitochondria, DNA, model, theory, etc.)
2. Use Hindi grammar words naturally: hai, hain, hota, hoti, hote, mein, ka, ki, ke, ko, se, aur,
   jo, jab, tab, isliye, kyunki, lekin, toh
3. NO Devanagari script — Roman only
4. Natural classroom tone — like a teacher explaining to students
5. Two mixing styles:
   - intra-sentential: Hindi and English mixed within the same sentence
     Example: "Mitochondria cell mein energy produce karta hai."
   - inter-sentential: Hindi sentence followed by English sentence (or vice versa)
     Example: "Yeh process bahut important hai. It is called photosynthesis."
6. Length ratio: Hinglish output should be 0.8× to 1.4× the English input word count
7. GROUNDING RULE: Every original_english entry must be based on a specific sentence,
   concept, or example from the provided <chapter_text>. Do NOT invent facts not
   present in the chapter text. For student Q&A entries, the question should be
   answerable from the chapter text, and the hinglish_roman answer should reflect it.

word_level_labels FORMAT:
- Each word in hinglish_roman gets a label: "EN", "HI", "NE" (named entity), or "UNIV"
- Scientific terms → "EN"
- Hindi grammar words → "HI"
- Proper nouns (names, places) → "NE"
- Numbers, OK, Sir → "UNIV"

OUTPUT FORMAT — return a valid JSON array only, no explanation text:
[
  {
    "id": "<chapter_code>_<type>_<3-digit-number>",
    "original_english": "...",
    "hinglish_roman": "...",
    "word_level_labels": { "word": "LABEL", ... },
    "topic": "...",
    "chapter": "<full chapter name>",
    "class": "<9 | 10 | general>",
    "code_mixing_type": "intra-sentential" or "inter-sentential",
    "is_student_query": true or false,
    "intent": "explain_concept" or "compare_concepts" or "give_example" or "define_term" or null
  }
]

NOTE on class field:
- Use "9" for Class 9 Biology chapters (NCERT)
- Use "10" for Class 10 Biology chapters (NCERT)
- Use "general" for cross-disciplinary science chapters (e.g., general1 — Exploration ch01)
```

---

## CHAPTER PROMPTS

---

### CHAPTER 1 — Life Processes (Class 10)
**File to save:** `data/chapter_additions/class10_ch05_additions.json`
**Current:** 95 samples (15 Q&A, 0 inter-sentential)
**Target:** +75 samples (35 Q&A, 20 inter-sentential)

```
Generate 75 new Hinglish dataset entries for NCERT Class 10 Biology — Chapter: "Life Processes".

chapter value = "Life Processes"
class value = "10"
ID format: "10_05_NEW_001", "10_05_NEW_002", etc.

TOPIC COVERAGE — distribute across:
- Autotrophic Nutrition / Photosynthesis (10 entries)
- Heterotrophic Nutrition / Human Digestion (10 entries)
- Respiration (aerobic vs anaerobic) (10 entries)
- Transportation in Plants and Animals (10 entries)
- Excretion (10 entries)
- 5 entries on any topic above

MIXING TARGETS:
- 35 entries: intra-sentential (mixed within sentence)
- 20 entries: inter-sentential (alternating EN/HI sentences)
- 20 entries: intra-sentential student Q&A (is_student_query: true)

STUDENT QUERY EXAMPLES TO INCLUDE (is_student_query: true):
- "Aerobic aur anaerobic respiration mein kya fark hai?"
- "Photosynthesis ke liye kya kya chahiye hota hai?"
- "Kidney mein filtration kaise hoti hai?"
- "Xylem aur phloem ka kya kaam hai?"
- "Hmm, toh stomata sirf gas exchange ke liye hote hain ya aur bhi kuch?"
- Make 15 more similar student questions covering the topics above

INTENT VALUES for student queries:
- "compare_concepts" for X vs Y questions
- "explain_concept" for kaise/kya kaam questions
- "define_term" for kya hai / define questions
- "give_example" for example do questions

Return valid JSON array only. No extra text.
```

> 📂 **Attach chapter text:** After pasting this prompt, add the following block and fill it with the full content of `data/processed/class10_ch05/cleaned_text.txt`:
> ```
> <chapter_text>
> [paste full content of data/processed/class10_ch05/cleaned_text.txt here]
> </chapter_text>
> ```

---

### CHAPTER 2 — Control and Coordination (Class 10)
**File to save:** `data/chapter_additions/class10_ch06_additions.json`
**Current:** 95 samples (15 Q&A, 0 inter-sentential)
**Target:** +75 samples (35 Q&A, 20 inter-sentential)

```
Generate 75 new Hinglish dataset entries for NCERT Class 10 Biology — Chapter: "Control and Coordination".

chapter value = "Control and Coordination"
class value = "10"
ID format: "10_06_NEW_001", "10_06_NEW_002", etc.

TOPIC COVERAGE:
- Nervous System / Neurons (12 entries)
- Reflex Actions / Reflex Arc (10 entries)
- Human Brain — parts and functions (10 entries)
- Hormones in Plants (Auxin, Gibberellins, etc.) (10 entries)
- Endocrine Glands and Hormones in Humans (13 entries)
- Feedback mechanisms (10 entries)

MIXING TARGETS: 35 intra-sentential | 20 inter-sentential | 20 student Q&A

STUDENT QUERIES TO INCLUDE:
- "Reflex action kya hoti hai aur uska example kya hai?"
- "Neuron mein signal kaise travel karta hai?"
- "Nervous system aur hormonal system mein kya difference hai?"
- "Brain ke kaunse part mein kya kaam hota hai?"
- 16 more student questions on the topics above

Return valid JSON array only.
```

> 📂 **Attach chapter text:** After pasting this prompt, add the following block and fill it with the full content of `data/processed/class10_ch06/cleaned_text.txt`:
> ```
> <chapter_text>
> [paste full content of data/processed/class10_ch06/cleaned_text.txt here]
> </chapter_text>
> ```

---

### CHAPTER 3 — How do Organisms Reproduce? (Class 10)
**File to save:** `data/chapter_additions/class10_ch07_additions.json`
**Current:** 95 samples (15 Q&A, 0 inter-sentential)
**Target:** +75 samples (35 Q&A, 20 inter-sentential)


```
Generate 75 new Hinglish dataset entries for NCERT Class 10 Biology — Chapter: "How do Organisms Reproduce?".

chapter value = "How do Organisms Reproduce?"
class value = "10"
ID format: "10_07_NEW_001", "10_07_NEW_002", etc.

TOPIC COVERAGE:
- Asexual Reproduction (Fission, Budding, Fragmentation, Spore formation) (15 entries)
- Vegetative Propagation (10 entries)
- Sexual Reproduction in Plants (Pollination, Fertilisation) (15 entries)
- Human Reproductive System — male and female (15 entries)
- STDs and Contraception (10 entries)
- Importance of Variation (10 entries)

MIXING TARGETS: 35 intra-sentential | 20 inter-sentential | 20 student Q&A

STUDENT QUERIES TO INCLUDE:
- "Asexual aur sexual reproduction mein kya fark hai?"
- "Vegetative propagation kyun use kiya jaata hai?"
- "Double fertilisation kya hoti hai plants mein?"
- "Male aur female reproductive system ke main parts kya hain?"
- 16 more questions on the above topics

Return valid JSON array only.
```

> 📂 **Attach chapter text:** After pasting this prompt, add the following block and fill it with the full content of `data/processed/class10_ch07/cleaned_text.txt`:
> ```
> <chapter_text>
> [paste full content of data/processed/class10_ch07/cleaned_text.txt here]
> </chapter_text>
> ```

---

### CHAPTER 4 — Heredity (Class 10)
**File to save:** `data/chapter_additions/class10_ch08_additions.json`
**Current:** 95 samples (15 Q&A, 0 inter-sentential)
**Target:** +75 samples (35 Q&A, 20 inter-sentential)

```
Generate 75 new Hinglish dataset entries for NCERT Class 10 Biology — Chapter: "Heredity".

chapter value = "Heredity"
class value = "10"
ID format: "10_08_NEW_001", "10_08_NEW_002", etc.

TOPIC COVERAGE:
- Mendel's Experiments (Monohybrid and Dihybrid cross) (15 entries)
- Laws of Inheritance (10 entries)
- Genes, Alleles, Dominant/Recessive traits (12 entries)
- Sex Determination (10 entries)
- Mutations and Evolution basics (10 entries)
- Acquired vs Inherited traits (10 entries)
- Speciation (8 entries)

MIXING TARGETS: 35 intra-sentential | 20 inter-sentential | 20 student Q&A

STUDENT QUERIES TO INCLUDE:
- "Dominant aur recessive traits mein kya fark hota hai?"
- "Mendel ne kaunse experiments kiye the?"
- "Sex determination kaise hota hai humans mein?"
- "Gene aur chromosome mein kya relation hai?"
- 16 more questions on above topics

Return valid JSON array only.
```

> 📂 **Attach chapter text:** After pasting this prompt, add the following block and fill it with the full content of `data/processed/class10_ch08/cleaned_text.txt`:
> ```
> <chapter_text>
> [paste full content of data/processed/class10_ch08/cleaned_text.txt here]
> </chapter_text>
> ```

---

### CHAPTER 5 — Our Environment (Class 10)
**File to save:** `data/chapter_additions/class10_ch13_additions.json`
**Current:** 95 samples (15 Q&A, 0 inter-sentential)
**Target:** +75 samples (35 Q&A, 20 inter-sentential)

```
Generate 75 new Hinglish dataset entries for NCERT Class 10 Biology — Chapter: "Our Environment".

chapter value = "Our Environment"
class value = "10"
ID format: "10_13_NEW_001", "10_13_NEW_002", etc.

TOPIC COVERAGE:
- Food Chain and Food Web (12 entries)
- Trophic Levels and Energy Flow (10 entries)
- Biodegradable vs Non-biodegradable waste (10 entries)
- Biological Magnification (10 entries)
- Ozone Layer Depletion (10 entries)
- Ecosystem components (Producers, Consumers, Decomposers) (13 entries)
- Waste management (10 entries)

MIXING TARGETS: 35 intra-sentential | 20 inter-sentential | 20 student Q&A

STUDENT QUERIES TO INCLUDE:
- "Food chain aur food web mein kya fark hai?"
- "Biological magnification kyun hoti hai?"
- "Ozone layer kaise deplete hoti hai?"
- "Biodegradable waste kya hota hai, example do"
- 16 more questions on above topics

Return valid JSON array only.
```

> 📂 **Attach chapter text:** After pasting this prompt, add the following block and fill it with the full content of `data/processed/class10_ch13/cleaned_text.txt`:
> ```
> <chapter_text>
> [paste full content of data/processed/class10_ch13/cleaned_text.txt here]
> </chapter_text>
> ```

---

### CHAPTER 6 — Cell: The Building Block of Life (Class 9)
**File to save:** `data/chapter_additions/class9_ch02_additions.json`
**Current:** 95 samples (15 Q&A, 0 inter-sentential)
**Target:** +75 samples (35 Q&A, 20 inter-sentential)

```
Generate 75 new Hinglish dataset entries for NCERT Class 9 Biology — Chapter: "Cell: The Building Block of Life".

chapter value = "Cell: The Building Block of Life"
class value = "9"
ID format: "9_02_NEW_001", "9_02_NEW_002", etc.

TOPIC COVERAGE:
- Discovery of Cell (Robert Hooke, Leeuwenhoek) (8 entries)
- Prokaryotic vs Eukaryotic cells (12 entries)
- Plant cell vs Animal cell (12 entries)
- Cell Organelles — Mitochondria, Chloroplast, Nucleus, ER, Golgi, Ribosome, Lysosome (20 entries — 2-3 per organelle)
- Cell Membrane and Osmosis (10 entries)
- Cell Wall (8 entries)
- Vacuoles and Plastids (5 entries)

MIXING TARGETS: 35 intra-sentential | 20 inter-sentential | 20 student Q&A

STUDENT QUERIES TO INCLUDE:
- "Mitochondria ko powerhouse of the cell kyun kehte hain?"
- "Prokaryotic aur eukaryotic cells mein kya difference hai?"
- "Plant cell aur animal cell mein kaunsa organelle sirf plant cell mein hota hai?"
- "Osmosis kaise hoti hai cell membrane mein?"
- 16 more student questions on cell biology

Return valid JSON array only.
```

> 📂 **Attach chapter text:** After pasting this prompt, add the following block and fill it with the full content of `data/processed/class9_ch02/cleaned_text.txt`:
> ```
> <chapter_text>
> [paste full content of data/processed/class9_ch02/cleaned_text.txt here]
> </chapter_text>
> ```

---

### CHAPTER 7 — Tissues in Action (Class 9)  ⚠️ PRIORITY
**File to save:** `data/chapter_additions/class9_ch03_additions.json`
**Current:** 80 samples (0 Q&A, 35 inter-sentential)
> This chapter has ZERO Q&A pairs and is 15 samples short vs other chapters. Do this one first.

**Target:** +90 samples (40 Q&A, 50 intra-sentential)

```
Generate 90 new Hinglish dataset entries for NCERT Class 9 Biology — Chapter: "Tissues in Action".

chapter value = "Tissues in Action"
class value = "9"
ID format: "9_03_NEW_001", "9_03_NEW_002", etc.

TOPIC COVERAGE:
- Introduction to Tissues and why multicellular organisms need them (10 entries)
- Plant Tissues — Meristematic (Apical, Lateral, Intercalary) (10 entries)
- Plant Tissues — Simple Permanent (Parenchyma, Collenchyma, Sclerenchyma) (12 entries)
- Plant Tissues — Complex Permanent (Xylem, Phloem) and Epidermal (10 entries)
- Animal Tissues — Epithelial (Squamous, Columnar, Cuboidal, Ciliated) (12 entries)
- Animal Tissues — Connective (Blood, Bone, Cartilage, Ligament, Tendon) (12 entries)
- Animal Tissues — Muscular (Striated, Smooth, Cardiac) (12 entries)
- Animal Tissues — Nervous Tissue / Neurons (12 entries)

MIXING TARGETS: 50 intra-sentential | 40 student Q&A
NOTE: This chapter needs student Q&A urgently — make 40 of the 90 entries student questions.

STUDENT QUERIES TO INCLUDE (all with intent field set):
- "Meristematic tissue kya hai aur yeh kahan paya jaata hai?"
- "Parenchyma, collenchyma aur sclerenchyma mein kya fark hai?"
- "Xylem aur phloem ka kya function hai?"
- "Striated aur smooth muscle mein kya difference hota hai?"
- "Cardiac muscle special kyun hoti hai?"
- "Neuron ke main parts kya hain?"
- 34 more varied student questions on the topics above

Return valid JSON array only.
```

> 📂 **Attach chapter text:** After pasting this prompt, add the following block and fill it with the full content of `data/processed/class9_ch03/cleaned_text.txt`:
> ```
> <chapter_text>
> [paste full content of data/processed/class9_ch03/cleaned_text.txt here]
> </chapter_text>
> ```

---

### CHAPTER 8 — Reproduction (Class 9, ch11)
**File to save:** `data/chapter_additions/class9_ch11_additions.json`
**Current:** 95 samples (15 Q&A, 0 inter-sentential)
**Target:** +75 samples (35 Q&A, 20 inter-sentential)

```
Generate 75 new Hinglish dataset entries for NCERT Class 9 Biology — Chapter: "Reproduction: How Life Continues".

chapter value = "class9/ch11"
class value = "9"
ID format: "9_11_NEW_001", "9_11_NEW_002", etc.

TOPIC COVERAGE:
- Male and Female Reproductive System in Plants (12 entries)
- Pollination — Self and Cross (10 entries)
- Fertilisation and Seed Formation in Plants (10 entries)
- Vegetative Reproduction in Plants (10 entries)
- Human Male Reproductive System (10 entries)
- Human Female Reproductive System and Menstrual Cycle (13 entries)
- Adolescence and Puberty changes (10 entries)

MIXING TARGETS: 35 intra-sentential | 20 inter-sentential | 20 student Q&A

STUDENT QUERIES TO INCLUDE:
- "Self-pollination aur cross-pollination mein kya fark hai?"
- "Fertilisation ke baad seed kaise banta hai?"
- "Menstrual cycle mein kya hota hai?"
- "Puberty mein kaunse changes hote hain?"
- 16 more questions on above topics

Return valid JSON array only.
```

> 📂 **Attach chapter text:** After pasting this prompt, add the following block and fill it with the full content of `data/processed/class9_ch11/cleaned_text.txt`:
> ```
> <chapter_text>
> [paste full content of data/processed/class9_ch11/cleaned_text.txt here]
> </chapter_text>
> ```

---

### CHAPTER 9 — Patterns in Life: Diversity and Classification (Class 9)
**File to save:** `data/chapter_additions/class9_ch12_additions.json`
**Current:** 95 samples (15 Q&A, 0 inter-sentential)
**Target:** +75 samples (35 Q&A, 20 inter-sentential)

```
Generate 75 new Hinglish dataset entries for NCERT Class 9 Biology — Chapter: "Patterns in Life: Diversity and Classification".

chapter value = "Patterns in Life: Diversity and Classification"
class value = "9"
ID format: "9_12_NEW_001", "9_12_NEW_002", etc.

TOPIC COVERAGE:
- Need for Classification and Basis of Classification (8 entries)
- Binomial Nomenclature (8 entries)
- Five Kingdom Classification (Monera, Protista, Fungi, Plantae, Animalia) (15 entries)
- Plant Kingdom — Thallophyta, Bryophyta, Pteridophyta, Gymnosperms, Angiosperms (12 entries)
- Animal Kingdom — Porifera to Vertebrates overview (12 entries)
- Invertebrates major phyla (10 entries)
- Vertebrates — Fish, Amphibia, Reptilia, Aves, Mammalia (10 entries)

MIXING TARGETS: 35 intra-sentential | 20 inter-sentential | 20 student Q&A

STUDENT QUERIES TO INCLUDE:
- "Classification kyun zaroori hai?"
- "Binomial nomenclature kya hota hai aur iska example kya hai?"
- "Monera aur Protista mein kya fark hai?"
- "Vertebrate aur invertebrate mein kya difference hai?"
- 16 more classification-related student questions

Return valid JSON array only.
```

> 📂 **Attach chapter text:** After pasting this prompt, add the following block and fill it with the full content of `data/processed/class9_ch12/cleaned_text.txt`:
> ```
> <chapter_text>
> [paste full content of data/processed/class9_ch12/cleaned_text.txt here]
> </chapter_text>
> ```

---

### GENERAL SECTION — Exploration: Entering the World of Secondary Science (general1)

> ⚠️ **Classification note:** This is **NOT a Biology chapter**. It is Chapter 1 of a general Science textbook covering the nature of science itself — models, units, laws vs theories, prediction, and estimation. Physics, chemistry, biology, and earth science all appear **equally** as examples. This chapter now lives under `data/processed/general1/` — not under `class9_ch01` — because it cross-references all four science branches rather than belonging to any one subject.

**File to save:** `data/chapter_additions/general1_additions.json`
**Source text:** `data/processed/general1/cleaned_text.txt`
**Existing dataset:** `data/processed/general1/dataset.json`

**Current state (95 entries — needs clean-up before merging):**
- `topic`: all 95 entries labelled `"General"` — needs retroactive topic assignment
- `is_student_query`: field **absent** on 80 of 95 entries (only the 15 Q&A entries have it)
- `class`: all set to `"9"` — acceptable but note this chapter is class-agnostic
- IDs: `9_01_001` → `9_01_095` (old class9_ch01 format — carried forward as-is, don't renumber)

**Target:** +75 new samples (35 Q&A, 20 inter-sentential) with correct topic labels

> ❌ **Topics removed from earlier draft (not present in source text):**
> - ~~"Safety in Laboratory"~~ — zero mentions; fabrication would violate the grounding rule
> - ~~"Introduction to Biology as a science"~~ — biology shares one paragraph with 3 other subjects; not enough distinct content for 10 unique grounded entries

> ℹ️ **Retroactive patch for the 95 existing entries:** After generating new entries, run the patch script in Step 10a (below) to back-fill `topic` and `is_student_query` on existing entries before merging.

```
Generate 75 new Hinglish dataset entries for Class 9 General Science —
Chapter: "Exploration: Entering the World of Secondary Science".

NOTE: This is a general science chapter, NOT a biology chapter. It covers
scientific thinking, models, units, laws, theories, and estimation. Physics,
chemistry, biology, and earth science are referenced equally as branches of
science. Every entry must be grounded in the provided <chapter_text>. Do NOT
invent lab safety content or any fact not present in the chapter text.

chapter value = "Chapter Exploration: Entering Exploration: Entering the World of Secondary"
class value = "general"
section value = "general1"
ID format: "gen_01_NEW_001", "gen_01_NEW_002", etc.

TOPIC COVERAGE — all 6 topics are grounded in the source text:

1. Scientific Models and Assumptions  (topic: "Scientific Models")  — 15 entries
   Draw from: models in physics (point mass car), chemistry (sphere atoms/bonds), biology
   (cell diagrams highlighting key parts), earth science (smooth-sphere Earth), cricket ball
   Example 1.1, Meghnad Saha treating star matter as hot gas, Activity 1.1 (bicycle trip model).
   Core idea: ignoring details is intentional, not a mistake.

2. Measurement and Units  (topic: "Measurement and Units")  — 10 entries
   Draw from: SI units, airplane fuel miscalculation (22,300 kg vs lb/litre), kilogram as
   international standard, symbols m/v/F/I/c, 'c' from Latin celeritas (299792458 m/s),
   unit mix-ups causing real-world errors.

3. Science in Daily Life  (topic: "Science in Daily Life")  — 15 entries
   Draw from: vegetable seller + pan balance, rice estimation for a family, eclipse viral claim
   (no physical/chemical/biological mechanism supports it), mask during COVID
   (physics: particle motion; chemistry: polymer fibres; biology: virus size; maths: airflow model),
   weather forecast limits, science connecting to technology and trade fairness.

4. Laws, Theories and Predictions  (topic: "Laws Theories Predictions")  — 15 entries
   Draw from: Newton's laws → jerk in bus, atomic theory → molecule formation, conservation of
   energy → climbing stairs, prediction of football distance (motion), CO₂ from chemical reactions,
   bread softness, breathing rate change while running, theory ≠ guess, openness to revision,
   Varsha/Meghna rain prediction Example 1.2, weather forecast unreliability beyond a few days.

5. Estimation and Approximate Reasoning  (topic: "Estimation")  — 10 entries
   Draw from: Example 1.3 (litres of air per day — ~10,000 L via balloon method),
   cross-check via 3 balloons/minute × 1440 min ≈ 8640 L, rice for family of four,
   "exact values not always necessary", estimation to detect errors (100 g/month too little;
   a few tonnes too much), Pause and Ponder Q2 (approximate vs exact).

6. Interdisciplinary Nature of Science  (topic: "Interdisciplinary Science")  — 10 entries
   Draw from: the chapter's closing paragraph — divisions into physics/chemistry/biology/earth
   science are human constructs; "natural world has no such boundaries"; climate change,
   medicine, sustainable tech require multiple disciplines; mask example uses all four branches;
   science connects with maths, technology, arts, social sciences.

CROSS-DISCIPLINARY RULE — for every "Interdisciplinary Science" entry:
  - original_english AND hinglish_roman MUST name at least two of:
    physics, chemistry, biology, mathematics (earth science counts too)
  - Use only the chapter's own examples as anchors (mask, climate change, Meghnad Saha,
    Pause and Ponder Q3 — pressure cooker / mobile phone / traffic jam)

MIXING TARGETS:
- 35 entries: intra-sentential
- 20 entries: inter-sentential
- 20 entries: intra-sentential student Q&A (is_student_query: true)

STUDENT QUERIES — all 20 must be answerable from the chapter text:
- "Model banate waqt kaunsi details ignore ki jaati hain aur kyun?"
- "Law aur theory mein kya fark hota hai science mein?"
- "Theory ko 'guess' kyun nahi kehte?"
- "SI units kyun use kiye jaate hain — kya hoga agar har jagah alag units hon?"
- "Estimation itni important kyun hai exact calculation se zyada?"
- "Predictions galat hone par scientist kya karte hain?"
- "Meghnad Saha ne stars ko model karne ke liye kya assume kiya tha?"
- "Ek din mein hum kitna air breathe karte hain — rough estimate kaise lagaein?"
- "Science sirf biology ya chemistry tak kyun nahi rukti?"
- "Mask COVID mein kaise kaam karta hai — kaunse branches of science involved hain?"
- "Weather forecast kabhi kabhi galat kyun hota hai?"
- "Eclipse ke time food khaana harmful hota hai — yeh claim scientifically sahi hai kya?"
- "Mathematics science mein sirf calculation ke liye hai ya aur bhi kuch?"
- "Conservation of energy ka ek roz-marra example kya hai?"
- "Cricket ball ka model banate waqt kya include karte hain, kya ignore?"
- "Speed of light 'c' se kyun denote hoti hai?"
- "Ek kilogram ka standard kyun zaroori hai — sabke liye same kyun hona chahiye?"
- "Approximate reasoning se kaise pata chalega answer reasonable hai ya impossible?"
- "Har cheez ek hi science branch mein kyun nahi aati — example do?"
- "Science mein 'theory' matlab kya hai aur yeh kaise change hoti hai over time?"

Return valid JSON array only. No extra text.
```

> 📂 **Attach chapter text:** After pasting this prompt, add the following block and fill it with the full content of `data/processed/general1/cleaned_text.txt`:
> ```
> <chapter_text>
> [paste full content of data/processed/general1/cleaned_text.txt here]
> </chapter_text>
> ```

---

### STEP 10a — Retroactive Patch for general1 Existing Entries

The 95 existing entries in `data/processed/general1/dataset.json` need two fixes before merging:
1. `topic: "General"` → map to the 6 grounded topics based on `original_english` content
2. `is_student_query` field added to all 80 entries that are missing it (set to `false`)

**Create** `scripts/patch_general1.py`:

```python
"""
Patch data/processed/general1/dataset.json:
  1. Add is_student_query: false to entries missing the field
  2. Remap topic: "General" → one of the 6 grounded topics based on keywords
  3. Update class field to "general" (was "9" — chapter is not class-specific)
Writes to data/processed/general1/dataset_patched.json (inspect before overwriting).
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "data/processed/general1/dataset.json"
OUT = ROOT / "data/processed/general1/dataset_patched.json"

# Keyword → topic mapping (order matters: first match wins)
TOPIC_RULES = [
    ("Interdisciplinary Science", [
        "physics", "chemistry", "biology", "earth science", "climate change",
        "medicines", "sustainable", "multiple disciplines", "branches", "mask",
        "natural world", "divisions", "technology, arts", "social"
    ]),
    ("Estimation", [
        "estimat", "approximate", "rough", "litres of air", "balloon",
        "10,000", "8640", "reasonable", "impossible", "exact value"
    ]),
    ("Laws Theories Predictions", [
        "law", "theory", "theor", "predict", "newton", "atomic theory",
        "conservation of energy", "football", "carbon dioxide", "bread",
        "breathing", "revised", "revision", "evidence", "guess"
    ]),
    ("Measurement and Units", [
        "unit", "kilogram", "kg", "pound", "lb", "SI", "symbol",
        "celeritas", "299792458", "m/s", "airplane", "fuel", "standard"
    ]),
    ("Science in Daily Life", [
        "vegetable", "pan balance", "eclipse", "food", "mask", "covid",
        "weather forecast", "daily life", "rice", "trade", "technology",
        "viral claim", "social media"
    ]),
    ("Scientific Models", [
        "model", "assumption", "ignor", "simplif", "point mass", "sphere",
        "meghnad", "saha", "star", "cricket", "bicycle", "detail"
    ]),
]

def assign_topic(text: str) -> str:
    t = text.lower()
    for topic, keywords in TOPIC_RULES:
        if any(kw in t for kw in keywords):
            return topic
    return "Scientific Models"  # fallback — chapter opens with models

with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

patched = 0
for entry in data:
    changed = False
    # Fix 1: add is_student_query if missing
    if "is_student_query" not in entry:
        entry["is_student_query"] = False
        changed = True
    # Fix 2: remap topic
    if entry.get("topic") == "General":
        src_text = entry.get("original_english", "") + " " + entry.get("hinglish_roman", "")
        entry["topic"] = assign_topic(src_text)
        changed = True
    # Fix 3: update class to "general"
    if str(entry.get("class")) == "9" and entry.get("chapter", "").startswith("Chapter Exploration"):
        entry["class"] = "general"
        changed = True
    if changed:
        patched += 1

print(f"Patched {patched}/{len(data)} entries")

from collections import Counter
topics = Counter(e["topic"] for e in data)
print(f"Topic distribution: {dict(topics)}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"Saved → {OUT}")
print("Review dataset_patched.json, then replace dataset.json if it looks correct.")
```

**Run it:**
```bash
python scripts/patch_general1.py
```

---

## STEP 11 — Merge and Clean All Data

> **Order of operations:**
> 1. Run `patch_general1.py` (Step 10a) first — this fixes topics/is_student_query on the 95 existing general1 entries and writes `dataset_patched.json`
> 2. Manually verify `dataset_patched.json` looks correct, then rename: `cp data/processed/general1/dataset_patched.json data/processed/general1/dataset.json`
> 3. Save all 9 chapter JSON files to `data/chapter_additions/` (class9/class10 Biology chapters)
> 4. Save `data/chapter_additions/general1_additions.json` (the +75 new general entries)
> 5. Run the merge script below

**Create** `scripts/merge_dataset.py`:

```python
"""
Merge all data sources into unified_science_dataset_v2.json:
  - data/unified_biology_dataset.json        (existing class 9/10 Biology entries)
  - data/processed/general1/dataset.json     (patched general section — 95 entries)
  - data/chapter_additions/*.json            (new entries for all chapters)
"""
import json, glob, os
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent

# ── 1. Load existing Biology dataset (class 9/10) ──────────────────────────
with open(ROOT / "data/unified_biology_dataset.json") as f:
    existing = json.load(f)

existing = [d for d in existing
            if d.get("original_english") and d.get("hinglish_roman")]
print(f"Existing Biology entries (valid): {len(existing)}")

# ── 2. Load patched general1 dataset ───────────────────────────────────────
general1_path = ROOT / "data/processed/general1/dataset.json"
if general1_path.exists():
    with open(general1_path, encoding="utf-8") as f:
        general_entries = json.load(f)
    general_entries = [d for d in general_entries
                       if d.get("original_english") and d.get("hinglish_roman")]
    print(f"General1 entries (valid): {len(general_entries)}")
else:
    general_entries = []
    print("WARNING: data/processed/general1/dataset.json not found — skipping")

# ── 3. Load all chapter additions and filter bad length ratios ─────────────
new_entries = []
additions_dir = ROOT / "data/chapter_additions"

for fpath in sorted(glob.glob(str(additions_dir / "*.json"))):
    with open(fpath, encoding="utf-8") as f:
        entries = json.load(f)

    clean = []
    for e in entries:
        if not e.get("original_english") or not e.get("hinglish_roman"):
            continue
        en_len = len(e["original_english"].split())
        hi_len = len(e["hinglish_roman"].split())
        ratio = hi_len / max(en_len, 1)
        if 0.6 <= ratio <= 1.8:
            clean.append(e)

    skipped = len(entries) - len(clean)
    print(f"{os.path.basename(fpath)}: {len(entries)} total, {len(clean)} clean"
          + (f", {skipped} skipped (bad ratio)" if skipped else ""))
    new_entries.extend(clean)

# ── 4. Merge — new entries overwrite on same id ────────────────────────────
all_data = {d["id"]: d for d in existing}
for d in general_entries:
    all_data[d["id"]] = d
for d in new_entries:
    all_data[d["id"]] = d

final = list(all_data.values())

# ── 5. Print stats by section ──────────────────────────────────────────────
qa = sum(1 for d in final if d.get("is_student_query"))
inter = sum(1 for d in final if d.get("code_mixing_type") == "inter-sentential")
by_class = Counter(str(d.get("class", "?")) for d in final)
by_topic = Counter(d.get("topic", "?") for d in final)

print(f"\n{'─'*50}")
print(f"Final dataset: {len(final)} samples")
print(f"  Q&A pairs:        {qa} ({qa/len(final)*100:.1f}%)")
print(f"  inter-sentential: {inter} ({inter/len(final)*100:.1f}%)")
print(f"\nBy section:")
for cls, count in sorted(by_class.items()):
    label = {"9": "Class 9 Biology", "10": "Class 10 Biology",
             "general": "General Science"}.get(cls, cls)
    print(f"  {label}: {count}")
print(f"\nTop topics: {dict(by_topic.most_common(10))}")
print(f"{'─'*50}")

out_path = ROOT / "data/unified_science_dataset_v2.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
print(f"\nSaved → {out_path}")
```

**Run it:**
```bash
mkdir -p data/chapter_additions
python scripts/merge_dataset.py
```

---

## STEP 12 — Rebuild Knowledge Base + Retrain

```bash
# 1. Prepare new seq2seq training data from v2 dataset
python training/prepare_seq2seq_data.py \
  --dataset data/unified_biology_dataset_v2.json \
  --output training/data/indicbart_seq2seq_v2/

# 2. Rebuild vector knowledge base (if NCERT source text changed, else skip)
python src/build_knowledge_base.py --force

# 3. Retrain on Colab
#    Upload training/data/indicbart_seq2seq_v2/ to Colab
#    Run training/colab_notebooks/03_train_indicbart.ipynb
#    Save new weights to models/indicbart_v2/

# 4. Evaluate new model vs old
python training/evaluate_generator.py \
  --model models/indicbart_v2 \
  --compare-base
```

---

## EXPECTED OUTCOME

| Metric | v1 (current) | v2 (expected) |
|---|---|---|
| Dataset size | 935 valid | ~1,700–1,800 |
| Q&A pairs | 14.4% | ~35% |
| Inter-sentential | 5.5% | ~20% |
| Corpus BLEU | 1.17 | 8–15 |
| Length Ratio | 2.93 | 0.9–1.3 |
| CMI Delta | 1.6% | <5% |

> **Start with Tissues in Action (ch03) and Cell (ch02) first** — those two chapters have the biggest gaps and highest evaluation frequency. Test with the new model before generating all 10 chapters.
