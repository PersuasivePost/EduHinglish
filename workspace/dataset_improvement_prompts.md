# EduHinglish — QA Dataset Generation Prompt for New NCERT Social Science Class 9 (2024-25)

---

## About This File

The NCERT Class 9 Social Science syllabus has changed to a new integrated curriculum (2024-25).
This file contains the **copy-pasteable Claude prompt** to generate QA Excel data for each chapter.

Chapters to generate QA for:

| Folder | Chapter | Subject |
|---|---|---|
| `data/processed/ss9_ch02/` | Shaping of the Earth's Surface | Geography |
| `data/processed/ss9_ch03/` | Atmosphere and Climate | Geography |
| `data/processed/ss9_ch04/` | Early Humans and Beginning of Civilisation | History |
| `data/processed/ss9_ch05/` | State and Society up to 1000 CE | History |
| `data/processed/ss9_ch06/` | Democracy | Civics |
| `data/processed/ss9_ch07/` | Elections | Civics |
| `data/processed/ss9_ch08/` | Building Blocks in Economics: The Problem of Choice | Economics |
| `data/processed/ss9_ch09/` | The Price Puzzle: What Drives the Market | Economics |

### How to use

1. Open a **new Claude conversation** for each chapter
2. Paste the **System Prompt** block below (once, at the top)
3. Paste the **Chapter Prompt** block that follows
4. Copy the **full content** of `data/processed/<chapter_folder>/cleaned_text.txt` and paste it inside the `<chapter_text>` tags as shown
5. Save Claude's Excel output as `data/raw/NCERT_Class 9/SS9_<ChapterName>.xlsx` — it should have exactly **4 columns**: `question`, `answer`, `context`, `class`

---

## SYSTEM PROMPT
> Paste this once at the very top of the Claude conversation before anything else.

```
You are an expert NCERT curriculum developer generating a Question-Answer dataset for EduHinglish,
an AI tutoring system for Indian Class 9 students.

Your task is to read the NCERT chapter text provided and generate high-quality QA pairs that will
be used to fine-tune a language model to answer student questions in Hinglish.

STRICT OUTPUT FORMAT:
You must output a single markdown table with exactly 4 columns:
| question | answer | context | class |

Column definitions:
- question  : A clear, natural question a Class 9 student would ask based on the chapter text.
              Write in English. Vary the question types (see types below).
- answer    : A concise, accurate answer (2-5 sentences). Must be grounded ONLY in the provided
              chapter text. Do NOT add external information not present in the chapter.
- context   : 2-4 sentences of relevant NCERT text that directly supports the answer.
              This is used by the RAG retrieval system — it must be a close paraphrase or
              near-verbatim excerpt from the chapter text.
- class     : Always write the number 9 (no quotes, no text).

QUESTION TYPE DISTRIBUTION — generate a balanced mix:
- explain_concept   : "Explain how...", "Why does...", "How does...work?"          (~35%)
- definition        : "What is...?", "What are...?", "Define..."                   (~20%)
- list_enumerate    : "List the...", "Name the types of...", "What are the causes?" (~15%)
- compare_concepts  : "What is the difference between X and Y?"                    (~10%)
- give_example      : "Give an example of...", "Illustrate with an example"         (~8%)
- application       : "How is X important in real life?", "What is the role of...?" (~7%)
- diagram           : "Describe the structure/process of..."                         (~5%)

QUALITY RULES:
1. Every answer must be directly answerable from the chapter text. No hallucination.
2. The context column must be a relevant 2-4 sentence excerpt from the chapter text.
3. Do NOT repeat the same question with slight rephrasing.
4. Cover all major topics and subtopics in the chapter — do not cluster on one section.
5. Generate at least 120-150 QA pairs per chapter.
6. Avoid yes/no questions — every question should require a substantive answer.
7. Do not include questions about chapter number, page numbers, or activities/exercises.
8. Questions should feel natural — like a curious student asking their teacher.

OUTPUT FORMAT: Markdown table only. No introduction, no explanation, no closing remarks.
| question | answer | context | class |
|---|---|---|---|
| ... | ... | ... | 9 |
```

---

## CHAPTER PROMPT
> Use this template for each chapter. Replace the placeholders in [SQUARE BRACKETS].

```
Now process the following NCERT Class 9 chapter and generate the QA dataset.

Chapter details:
- Chapter Title : [CHAPTER TITLE — e.g. "Democracy"]
- Subject       : [SUBJECT — e.g. "Civics"]
- Class         : 9
- Curriculum    : NCERT 2024-25 Integrated Social Science

Generate at least 150 QA pairs covering ALL sections of this chapter.
Ensure the question types are distributed as instructed in the system prompt.

Here is the full chapter text:

<chapter_text>
[PASTE THE FULL CONTENT OF cleaned_text.txt HERE]
</chapter_text>

Output only the markdown table. Start immediately with the header row:
| question | answer | context | class |
```

---

## Chapter-by-Chapter Checklist

Use this to track which chapters have been processed:

| # | Chapter | Subject | Cleaned Text File | Status |
|---|---|---|---|---|
| 1 | Shaping of the Earth's Surface | Geography | `data/processed/ss9_ch02/cleaned_text.txt` | ☐ |
| 2 | Atmosphere and Climate | Geography | `data/processed/ss9_ch03/cleaned_text.txt` | ☐ |
| 3 | Early Humans and Beginning of Civilisation | History | `data/processed/ss9_ch04/cleaned_text.txt` | ☐ |
| 4 | State and Society up to 1000 CE | History | `data/processed/ss9_ch05/cleaned_text.txt` | ☐ |
| 5 | Democracy | Civics | `data/processed/ss9_ch06/cleaned_text.txt` | ☐ |
| 6 | Elections | Civics | `data/processed/ss9_ch07/cleaned_text.txt` | ☐ |
| 7 | Building Blocks in Economics | Economics | `data/processed/ss9_ch08/cleaned_text.txt` | ☐ |
| 8 | The Price Puzzle | Economics | `data/processed/ss9_ch09/cleaned_text.txt` | ☐ |

---

## After Generating All Chapters

Once you have all 8 Excel files saved in `data/raw/NCERT_Class 9/`:
1. Register each new file in `training/convert_excel_to_json.py` under `EXCEL_FILES[9]`
2. Re-run the conversion script:
   ```bash
   python training/convert_excel_to_json.py --class_filter 9 --output data/cbse_qa_dataset_base.json
   ```
3. Check the chapter coverage report in the output — `ch00 (unknown)` should be near 0 for these new chapters since the keywords are already added.
