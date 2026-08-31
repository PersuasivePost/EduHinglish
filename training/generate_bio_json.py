"""
generate_bio_json.py — Generates data/bio_9_10.json from Biology cleaned chapter texts.
Run: python training/generate_bio_json.py

Question spectrum:
  trivial   : one_word, fill_blank, true_false, name_the
  basic     : definition, short_answer
  medium    : explain_concept, compare_concepts, list_enumerate, give_example,
              describe_function, application_reallife
  analytical: analytical (from NCERT exercises), assertion_reason, mcq, case_study
  numerical : numerical_problem, formula_application (step-by-step)
  reaction  : reaction_equation (balanced chemical equations)
  NCERT     : questions extracted from Revise/Reflect/Refine + EXERCISES sections
"""

import re, json, sys
from pathlib import Path
from collections import Counter

# Use shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from ncert_utils import (
    build_passages, key_term, noun_phrases, split_sentences,
    gen_trivial_questions, extract_ncert_questions,
    get_reaction_questions_for_chapter, is_clean_text, is_good_first_sent
)

CHAPTERS = {
    "class9_ch01": {"chapter": "Exploration: Entering the World of Secondary Science",
                    "chapter_num": "ch01", "class_val": "9", "subject": "Science", "subject_code": "sci"},
    "class9_ch02": {"chapter": "Cell: The Building Block of Life",
                    "chapter_num": "ch02", "class_val": "9", "subject": "Biology", "subject_code": "bio"},
    "class9_ch03": {"chapter": "Tissues in Action",
                    "chapter_num": "ch03", "class_val": "9", "subject": "Biology", "subject_code": "bio"},
    "class9_ch11": {"chapter": "Reproduction: How Life Continues",
                    "chapter_num": "ch11", "class_val": "9", "subject": "Biology", "subject_code": "bio"},
    "class9_ch12": {"chapter": "Patterns in Life: Diversity and Classification",
                    "chapter_num": "ch12", "class_val": "9", "subject": "Biology", "subject_code": "bio"},
    "class10_ch05": {"chapter": "Life Processes",
                     "chapter_num": "ch05", "class_val": "10", "subject": "Biology", "subject_code": "bio"},
    "class10_ch06": {"chapter": "Control and Coordination",
                     "chapter_num": "ch06", "class_val": "10", "subject": "Biology", "subject_code": "bio"},
    "class10_ch07": {"chapter": "How do Organisms Reproduce?",
                     "chapter_num": "ch07", "class_val": "10", "subject": "Biology", "subject_code": "bio"},
    "class10_ch08": {"chapter": "Heredity",
                     "chapter_num": "ch08", "class_val": "10", "subject": "Biology", "subject_code": "bio"},
    "class10_ch13": {"chapter": "Our Environment",
                     "chapter_num": "ch13", "class_val": "10", "subject": "Biology", "subject_code": "bio"},
}

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT       = ROOT / "data" / "bio_9_10.json"

STOP_STARTS = {
    'what', 'how', 'why', 'when', 'where', 'which', 'who', 'the', 'a', 'an',
    'it', 'in', 'on', 'at', 'for', 'of', 'and', 'or', 'but', 'these', 'those',
    'this', 'that', 'is', 'are', 'was', 'were', 'be', 'horizontally',
    'vertically', 'however', 'therefore', 'thus', 'hence', 'since', 'although',
    'whereas', 'while', 'now', 'also', 'even', 'just', 'both', 'all', 'each',
    'every', 'such', 'some', 'many', 'most', 'more', 'other', 'another', 'during',
}

# ── Numerical / formula detection ────────────────────────────────────────────
# Strict: requires actual arithmetic or known bio-math keywords
NUMERICAL_STRONG = re.compile(
    r'('
    r'\d+\.?\d*\s*[xX\u00d7/\u00f7]\s*\d|'         # actual arithmetic: 5 x 2, 100/25
    r'magnification\s*=|estimated size\s*=|'          # formula assignments
    r'actual size\s*=|field.{0,20}diameter|'
    r'energy available at.{0,30}trophic|10%\s*(of|rule|energy)|'
    r'F1.{0,50}F2|cross between.{0,30}(TT|Tt|tt|RR|Rr|rr)|'
    r'\d+\s*:\s*\d+.{0,20}(?:ratio|phenotype|genotype)|'   # explicit ratio for genetics
    r'calculate.{0,60}(energy|size|ratio|number)|'
    r'step\s*\d+\s*(?:\(|:).{0,60}(?:given|formula|calculation)|'  # step-by-step structure
    r'given\s*:\s*.{0,30}=\s*\d'                     # "given: x = number" pattern
    r')',
    re.I
)


FORMULA_STRONG = re.compile(
    r'('
    r'formula.{0,30}=|'
    r'estimated size.{0,20}=|'
    r'magnification.{0,20}=|'
    r'diameter.{0,30}(µm|micrometre|micrometer)|'
    r'\d+\s*(µm|nm|mm)\s*/\s*\d+'                    # unit division
    r')',
    re.I
)

# ── Question generators ───────────────────────────────────────────────────────

def gen_passage_questions(passage, chapter, subject):
    """Generate medium + analytical + numerical questions from a text passage."""
    kt  = key_term(passage)
    nps = noun_phrases(passage)
    sents = split_sentences(passage)
    fs = sents[0] if sents else passage.split('.')[0]

    has_defn    = bool(re.search(r'\b(?:is defined as|refers to|is a |are a |means|can be described as|known as)\b', passage, re.I))
    has_cause   = bool(re.search(r'\b(?:because|due to|causes?|led to|results? in|reason for|therefore)\b', passage, re.I))
    has_compare = bool(re.search(r'\b(?:differ|whereas|while|unlike|compared|contrast|however|distinguish|difference)\b', passage, re.I))
    has_process = bool(re.search(r'\b(?:process|mechanism|forms?|develops?|produc(?:es|ed)|works?|steps?|cycle)\b', passage, re.I))
    has_example = bool(re.search(r'\b(?:for example|such as|like|instance|including|e\.g\.)\b', passage, re.I))
    has_import  = bool(re.search(r'\b(?:important|significant|crucial|essential|major|key role|plays a|function|advantage|use)\b', passage, re.I))
    has_list    = re.search(r'\b(types|kinds|forms|features|characteristics|factors|elements|components|stages|effects|parts|organs|tissues|cells)\b', passage, re.I)
    has_num     = NUMERICAL_STRONG.search(passage)
    has_formula = FORMULA_STRONG.search(passage)
    has_func    = bool(re.search(r'\b(?:function|role|helps|responsible for|allows)\b', passage, re.I))

    qs = []

    def add(intent, q_en):
        qs.append((intent, q_en, passage))

    # ── Always: explain + describe + what is ─────────────────────────────
    add("explain_concept",    f"Explain {kt} in the context of {chapter}.")
    if is_good_first_sent(fs):
        add("explain_concept", f"Describe in detail: {fs}?")
    add("definition",         f"What is meant by {kt}?")

    if has_defn:
        add("definition",     f"Define {kt} as given in your NCERT textbook.")
        add("short_answer",   f"What is {kt}? Give a brief definition in 2–3 sentences.")

    if has_cause:
        add("explain_concept", f"What are the causes or factors responsible for {kt}?")
        add("short_answer",    f"Why does {kt} occur? Give reasons.")

    if has_compare:
        if len(nps) >= 2:
            add("compare_concepts", f"What is the difference between {nps[0]} and {nps[1]}?")
            add("compare_concepts", f"Compare {nps[0]} and {nps[1]}.")
        else:
            add("compare_concepts", f"How does {kt} differ from related biological concepts?")

    if has_process:
        add("explain_concept", f"Explain the biological process of {kt}.")
        add("explain_concept", f"How does {kt} work or develop? Describe step by step.")

    if has_example:
        add("give_example",    f"Give an example of {kt} as mentioned in the chapter.")
        add("give_example",    f"Illustrate {kt} with an example from the NCERT text.")

    if has_import:
        add("application_reallife", f"Why is {kt} important? Discuss its significance.")
        add("application_reallife", f"What is the role or function of {kt} in living organisms?")

    if has_list:
        lword = has_list.group(1).lower()
        add("list_enumerate", f"List the {lword} of {kt}.")
        add("list_enumerate", f"What are the main {lword} of {kt}?")

    if has_func:
        add("describe_function", f"What is the main function of {kt}?")
        add("describe_function", f"Describe the role of {kt} in {chapter}.")

    # ── Numericals (strict: only real math content) ───────────────────────
    if has_num:
        term_match = has_num.group(1)
        add("numerical_problem",
            f"Based on the NCERT text, solve or explain the numerical problem related to {kt}. Show step-by-step calculation.")
        add("numerical_problem",
            f"A numerical problem involves {kt}. Using the information in the chapter, show how to calculate the result step by step.")

    if has_formula:
        add("formula_application",
            f"Using the formula described in the text for {kt}, explain how to compute the result. Give an example with numbers.")

    # ── Context + significance questions ─────────────────────────────────
    if nps:
        add("explain_concept", f"What is the significance of {nps[0]} in {chapter}?")
    if len(nps) >= 2:
        add("explain_concept", f"How are {nps[0]} and {nps[1]} related in biology?")

    add("application_reallife", f"How is {kt} relevant to human life or nature?")
    add("analytical",           f"What would happen if {kt} did not function properly in a living organism?")

    return qs


# ── Entry builder ─────────────────────────────────────────────────────────────

def make_entry(idx, info, intent, q_en, answer_en, topic, source="generated"):
    sc = info["subject_code"]
    cn = info["chapter_num"]
    cv = info["class_val"]
    return {
        "id":               f"c{cv}_{sc}_{cn}_{idx:04d}",
        "class":            cv,
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
    rxn_seen = set()  # track reaction questions per chapter to avoid duplicates

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

        # passage-level medium/analytical questions
        for intent, q_en, answer_en in gen_passage_questions(p, chapter, info["subject"]):
            if q_en.lower() in global_seen: continue
            global_seen.add(q_en.lower())
            ctr += 1
            entries.append(make_entry(ctr, info, intent, q_en, answer_en, topic, "generated"))

        # biological reaction questions (triggered per passage)
        for intent, q_en, answer_en in get_reaction_questions_for_chapter(chapter, p, rxn_seen):
            if q_en.lower() in global_seen: continue
            global_seen.add(q_en.lower())
            ctr += 1
            entries.append(make_entry(ctr, info, intent, q_en, answer_en, chapter, "reaction_equation"))

    # ── 2. NCERT exercise / Revise-Reflect-Refine questions ───────────────
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
    print("\n=== Generating Biology QA Dataset (v2) ===\n")
    all_entries = []
    global_seen = set()   # dedup across all chapters

    for f, info in CHAPTERS.items():
        res = process(f, info, global_seen)
        all_entries.extend(res)
        print(f"  {info['chapter'][:55]:<55}: {len(res):>5} entries")

    OUT.write_text(json.dumps(all_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "="*60)
    print(f"  Saved   -> {OUT}")
    print(f"  Total   : {len(all_entries)} entries\n")

    by_class  = Counter(e["class"] for e in all_entries)
    by_chap   = Counter(e["chapter"] for e in all_entries)
    by_intent = Counter(e["intent"] for e in all_entries)
    by_source = Counter(e["source"] for e in all_entries)

    print("  By class:")
    for k, v in by_class.most_common():
        print(f"    Class {k:<4}: {v}")

    print("\n  By intent:")
    for k, v in by_intent.most_common():
        print(f"    {k:<30}: {v}")

    print("\n  By source:")
    for k, v in by_source.most_common():
        print(f"    {k:<30}: {v}")

    print("\n  By chapter:")
    for k, v in by_chap.most_common():
        print(f"    {k[:55]:<55}: {v}")


if __name__ == "__main__":
    main()
