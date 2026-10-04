"""
generate_ss9_json.py — Generates data/ss9.json from SS9 cleaned chapter texts.
Run: python training/generate_ss9_json.py

Question spectrum (no numericals for SS9 - pure theory subject):
  trivial   : one_word, fill_blank, true_false, name_the
  basic     : definition, short_answer
  medium    : explain_concept, compare_concepts, list_enumerate, give_example,
              application_reallife
  analytical: analytical, assertion_reason, mcq, case_study (from NCERT exercises)
  NCERT     : questions extracted from Questions-and-activities + EXERCISES sections
"""

import re, json, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from ncert_utils import (
    build_passages, key_term, noun_phrases, split_sentences,
    gen_trivial_questions, extract_ncert_questions, is_clean_text,
    is_good_first_sent
)

CHAPTERS = {
    "ss9_ch01": {"chapter": "Understanding Social Science",                      "chapter_num": "ch01", "subject": "Social Science", "subject_code": "ss"},
    "ss9_ch02": {"chapter": "Shaping of the Earth's Surface",                    "chapter_num": "ch02", "subject": "Geography",      "subject_code": "geo"},
    "ss9_ch03": {"chapter": "Atmosphere and Climate",                            "chapter_num": "ch03", "subject": "Geography",      "subject_code": "geo"},
    "ss9_ch04": {"chapter": "Early Humans and Beginning of Civilisation",        "chapter_num": "ch04", "subject": "History",        "subject_code": "hist"},
    "ss9_ch05": {"chapter": "State and Society up to 1000 CE",                  "chapter_num": "ch05", "subject": "History",        "subject_code": "hist"},
    "ss9_ch06": {"chapter": "Democracy",                                         "chapter_num": "ch06", "subject": "Civics",         "subject_code": "civ"},
    "ss9_ch07": {"chapter": "Elections",                                         "chapter_num": "ch07", "subject": "Civics",         "subject_code": "civ"},
    "ss9_ch08": {"chapter": "Building Blocks in Economics: The Problem of Choice","chapter_num": "ch08", "subject": "Economics",     "subject_code": "eco"},
    "ss9_ch09": {"chapter": "The Price Puzzle: What Drives the Market",          "chapter_num": "ch09", "subject": "Economics",     "subject_code": "eco"},
}

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT       = ROOT / "data" / "ss9.json"

STOP_STARTS = {
    'what', 'how', 'why', 'when', 'where', 'which', 'who', 'the', 'a', 'an',
    'it', 'in', 'on', 'at', 'for', 'of', 'and', 'or', 'but', 'these', 'those',
    'this', 'that', 'is', 'are', 'was', 'were', 'be', 'horizontally',
    'vertically', 'however', 'therefore', 'thus', 'hence', 'since', 'although',
    'whereas', 'while', 'now', 'also', 'even', 'just', 'both', 'all', 'each',
    'every', 'such', 'some', 'many', 'most', 'more', 'other', 'another', 'during',
}

# SS9 has numbers (dates, stats, percentages) — generate contextual questions
NUMBER_RE = re.compile(r'\b(\d{3,}[,\d]*)\b')

# ── Question generators ───────────────────────────────────────────────────────

def gen_passage_questions(passage, chapter, subject):
    kt  = key_term(passage)
    nps = noun_phrases(passage)
    sents = split_sentences(passage)
    fs = sents[0] if sents else passage.split('.')[0]

    has_defn    = bool(re.search(r'\b(?:is defined as|refers to|is a |are a |means|can be described as|known as)\b', passage, re.I))
    has_cause   = bool(re.search(r'\b(?:because|due to|causes?|led to|results? in|reason for|therefore)\b', passage, re.I))
    has_compare = bool(re.search(r'\b(?:differ|whereas|while|unlike|compared|contrast|however|distinguish|difference)\b', passage, re.I))
    has_process = bool(re.search(r'\b(?:process|mechanism|forms?|develops?|produc(?:es|ed)|works?|steps?)\b', passage, re.I))
    has_example = bool(re.search(r'\b(?:for example|such as|like|instance|including|e\.g\.)\b', passage, re.I))
    has_import  = bool(re.search(r'\b(?:important|significant|crucial|essential|major|key role|plays a)\b', passage, re.I))
    has_list    = re.search(r'\b(types|kinds|forms|features|characteristics|factors|elements|components|stages|effects)\b', passage, re.I)
    has_number  = NUMBER_RE.search(passage)

    qs = []

    def add(intent, q_en):
        qs.append((intent, q_en, passage))

    # ── Always: core questions ──────────────────────────────────────────
    add("explain_concept", f"Explain {kt} in the context of {chapter}.")
    if is_good_first_sent(fs):
        add("explain_concept", f"Describe in detail: {fs}?")
    add("definition",      f"What is meant by {kt}?")

    if has_defn:
        add("definition",  f"Define {kt} as mentioned in your NCERT textbook.")
        add("short_answer",f"What is {kt}? Write a brief definition in your own words.")

    if has_cause:
        add("explain_concept", f"What are the causes or factors responsible for {kt}?")
        add("short_answer",    f"Why does {kt} occur or exist? Explain with reasons.")

    if has_compare:
        if len(nps) >= 2:
            add("compare_concepts", f"What is the difference between {nps[0]} and {nps[1]}?")
            add("compare_concepts", f"Compare {nps[0]} and {nps[1]} in the context of {subject}.")
        else:
            add("compare_concepts", f"How does {kt} differ from related concepts in {subject}?")

    if has_process:
        add("explain_concept", f"Explain the process of {kt} as described in the chapter.")
        add("explain_concept", f"How does {kt} work? Describe step by step.")

    if has_example:
        add("give_example",    f"Give an example of {kt} as mentioned in the chapter.")
        add("give_example",    f"Illustrate {kt} with a real-life example from Indian context.")

    if has_import:
        add("application_reallife", f"Why is {kt} important? Discuss its significance.")
        add("application_reallife", f"What is the role of {kt} in {subject.lower()}?")

    if has_list:
        lword = has_list.group(1).lower()
        add("list_enumerate", f"List the {lword} of {kt}.")
        add("list_enumerate", f"What are the main {lword} of {kt}?")

    # ── Numerical / statistical context (SS9 - dates, stats) ───────────
    if has_number:
        num = has_number.group(1)
        add("short_answer", f"What does the figure {num} represent in the context of {kt}?")

    # ── Context / significance questions ────────────────────────────────
    if nps:
        add("explain_concept", f"What is the significance of {nps[0]} in {chapter}?")
    if len(nps) >= 2:
        add("explain_concept", f"How are {nps[0]} and {nps[1]} related?")

    add("application_reallife", f"How is {kt} relevant to Indian society or daily life?")
    add("analytical",           f"What would be the consequences if {kt} did not exist in {subject.lower()}?")

    return qs


# ── Entry builder ─────────────────────────────────────────────────────────────

def make_entry(idx, info, intent, q_en, answer_en, topic, source="generated"):
    sc = info["subject_code"]
    cn = info["chapter_num"]
    return {
        "id":               f"c09_{sc}_{cn}_{idx:04d}",
        "class":            "9",
        "subject":          info["subject"],
        "chapter":          info["chapter"],
        "chapter_num":      cn,
        "topic":            topic,
        "source":           source,
        "is_student_query": False,
        "intent":           intent,
        "question_english": q_en,
        "question_hinglish": "",
        "answer_english":   answer_en,
        "answer_hinglish":  "",
        "ncert_context":    answer_en,
        "messages": [
            {"role": "system",    "content": answer_en},
            {"role": "user",      "content": q_en},
            {"role": "assistant", "content": answer_en},
        ]
    }


def process(folder, info, global_seen):
    path = PROCESSED / folder / "cleaned_text.txt"
    if not path.exists():
        print(f"  MISSING: {path}")
        return []
    raw = path.read_text(encoding="utf-8")

    entries = []
    ctr = 0
    chapter = info["chapter"]

    # ── 1. Passage-based questions ────────────────────────────────────────
    passages = build_passages(raw)
    for p in passages:
        if not is_clean_text(p):
            continue
        topic = key_term(p)

        # trivial questions
        for intent, q_en, answer_en in gen_trivial_questions(p, chapter, info["subject"]):
            if q_en.lower() in global_seen: continue
            global_seen.add(q_en.lower())
            ctr += 1
            entries.append(make_entry(ctr, info, intent, q_en, answer_en, topic, "generated_trivial"))

        # medium / analytical questions
        for intent, q_en, answer_en in gen_passage_questions(p, chapter, info["subject"]):
            if q_en.lower() in global_seen: continue
            global_seen.add(q_en.lower())
            ctr += 1
            entries.append(make_entry(ctr, info, intent, q_en, answer_en, topic, "generated"))

    # ── 2. NCERT exercise / activity questions ────────────────────────────
    ncert_qs = extract_ncert_questions(raw, info, seen=global_seen)
    for item in ncert_qs:
        ctr += 1
        entries.append(make_entry(
            ctr, info,
            item['intent'], item['q_en'], item['answer_en'],
            chapter, f"ncert_{item['source']}"
        ))

    return entries


def main():
    print("\n=== Generating SS9 QA Dataset (v2) ===\n")
    all_entries = []
    global_seen = set()

    for folder, info in CHAPTERS.items():
        print(f"  {info['chapter'][:50]:<50} ({info['subject']}) ...", end=" ", flush=True)
        res = process(folder, info, global_seen)
        all_entries.extend(res)
        print(f"{len(res)} entries")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Saved  -> {OUT}")
    print(f"  Total  : {len(all_entries)} entries\n")

    by_subject = Counter(e["subject"] for e in all_entries)
    by_intent  = Counter(e["intent"] for e in all_entries)
    by_source  = Counter(e["source"] for e in all_entries)
    by_chap    = Counter(e["chapter"] for e in all_entries)

    print("  By subject:")
    for s, c in sorted(by_subject.items()):
        print(f"    {s:<20}: {c}")

    print("\n  By intent:")
    for i, c in by_intent.most_common():
        print(f"    {i:<30}: {c}")

    print("\n  By source:")
    for s, c in by_source.most_common():
        print(f"    {s:<30}: {c}")

    print("\n  By chapter:")
    for ch, c in by_chap.most_common():
        print(f"    {ch[:55]:<55}: {c}")


if __name__ == "__main__":
    main()
