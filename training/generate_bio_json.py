"""
generate_bio_json.py  —  Generates data/bio_9_10.json from Biology cleaned chapter texts.
Run: python training/generate_bio_json.py
"""
import re, json
from pathlib import Path
from collections import Counter

CHAPTERS = {
    # Class 9 Biology
    "class9_ch01": {"chapter": "Exploration: Entering the World of Secondary Science", "chapter_num":"ch01", "class_val": "9", "subject":"Science", "subject_code":"sci"},
    "class9_ch02": {"chapter": "Cell: The Building Block of Life",                     "chapter_num":"ch02", "class_val": "9", "subject":"Biology", "subject_code":"bio"},
    "class9_ch03": {"chapter": "Tissues in Action",                                    "chapter_num":"ch03", "class_val": "9", "subject":"Biology", "subject_code":"bio"},
    "class9_ch11": {"chapter": "Reproduction: How Life Continues",                     "chapter_num":"ch11", "class_val": "9", "subject":"Biology", "subject_code":"bio"},
    "class9_ch12": {"chapter": "Patterns in Life: Diversity and Classification",       "chapter_num":"ch12", "class_val": "9", "subject":"Biology", "subject_code":"bio"},
    # Class 10 Biology
    "class10_ch05": {"chapter": "Life Processes",                                      "chapter_num":"ch05", "class_val": "10", "subject":"Biology", "subject_code":"bio"},
    "class10_ch06": {"chapter": "Control and Coordination",                            "chapter_num":"ch06", "class_val": "10", "subject":"Biology", "subject_code":"bio"},
    "class10_ch07": {"chapter": "How do Organisms Reproduce?",                         "chapter_num":"ch07", "class_val": "10", "subject":"Biology", "subject_code":"bio"},
    "class10_ch08": {"chapter": "Heredity",                                            "chapter_num":"ch08", "class_val": "10", "subject":"Biology", "subject_code":"bio"},
    "class10_ch13": {"chapter": "Our Environment",                                     "chapter_num":"ch13", "class_val": "10", "subject":"Biology", "subject_code":"bio"},
}

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT       = ROOT / "data" / "bio_9_10.json"

# ── Sentence splitter ────────────────────────────────────────────────────────
def split_sentences(text):
    # protect abbreviations
    text = re.sub(r'\b(etc|vs|Dr|Mr|St|No|Fig|e\.g|i\.e|approx|govt|dept|cm|mm|nm|kg|ml)\.',
                  r'\1<DOT>', text)
    # split on ". " followed by capital
    parts = re.split(r'\.\s+(?=[A-Z])', text)
    out = []
    for p in parts:
        p = p.replace('<DOT>', '.').strip()
        if len(p) > 3:
            out.append(p.rstrip('.') + '.')
    return out

# ── Build passages from a text using sliding sentence windows ─────────────────
SKIP_LINE = re.compile(
    r'^(\d+\.\s|Big Questions?|The\s*$|CChhaapptteerr|iinndddd|LET\'S|BEYOND\s*$|'
    r'UNDERSTANDING\s*$|India and Beyond|https?://|courtesy|wikimedia)', re.I
)
ARTIFACT = re.compile(r'(.)\1{3,}')

def build_passages(raw_text, window=5, step=3, min_len=120):
    lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line: continue
        if SKIP_LINE.match(line): continue
        if re.match(r'^\d+$', line): continue          # bare page numbers
        lines.append(line)

    blob = ' '.join(lines)
    sents = split_sentences(blob)

    passages = []
    for i in range(0, len(sents) - 1, step):
        chunk_sents = sents[i: i + window]
        chunk = ' '.join(chunk_sents)
        if len(chunk) < min_len: continue
        if chunk.count('?') > 3: continue
        passages.append(chunk)
    return passages

# ── Key term extraction ───────────────────────────────────────────────────────
STOP_STARTS = {'what', 'how', 'why', 'when', 'where', 'which', 'who', 'the',
               'a', 'an', 'it', 'in', 'on', 'at', 'for', 'of', 'and', 'or',
               'but', 'these', 'those', 'this', 'that', 'is', 'are', 'was', 'were', 'be'}

def key_term(passage):
    m = re.match(r'([A-Z][^.!?]{3,50}?)\s+(?:is|are|refers to|means|can be defined)', passage)
    if m:
        t = m.group(1).strip()
        words = t.split()
        if words and words[0].lower() not in STOP_STARTS and words[-1].lower() not in STOP_STARTS and len(words) <= 6:
            return t
    caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', passage)
    for c in caps:
        words = c.split()
        if words[0].lower() not in STOP_STARTS and words[-1].lower() not in STOP_STARTS:
            return c
    caps1 = re.findall(r'\b([A-Z][a-z]{3,})\b', passage)
    for c in caps1:
        if c.lower() not in STOP_STARTS:
            return c
    # Fallback: noun phrases
    m = re.search(r'\b(?:the|a|an)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\b', passage)
    if m:
        t = m.group(1).strip()
        words = t.split()
        if words and words[-1].lower() not in STOP_STARTS:
            return t
    return "the topic"

def noun_phrases(passage):
    nps = []
    chunks = re.findall(r'\b([A-Z][a-z]+)\b', passage)
    for c in chunks:
        if c.lower() not in STOP_STARTS: nps.append(c)
    out = []
    for n in nps:
        if n not in out: out.append(n)
        if len(out) == 2: break
    return out

# ── Question generators ────────────────────────────────────────────────────────
def gen_questions(passage, chapter, subject):
    kt  = key_term(passage)
    nps = noun_phrases(passage)
    sents = split_sentences(passage)
    first_sent = sents[0] if sents else passage.split('.')[0]
    fs = first_sent

    has_defn    = bool(re.search(r'\b(?:is defined as|refers to|is a |are a |means|can be described as|known as)\b', passage, re.I))
    has_cause   = bool(re.search(r'\b(?:because|due to|causes?|led to|results? in|reason for|therefore)\b', passage, re.I))
    has_compare = bool(re.search(r'\b(?:differ|whereas|while|unlike|compared|contrast|however|distinguish|difference)\b', passage, re.I))
    has_process = bool(re.search(r'\b(?:process|mechanism|forms?|develops?|produc(?:es|ed)|works?|steps?|cycle)\b', passage, re.I))
    has_example = bool(re.search(r'\b(?:for example|such as|like|instance|including|e\.g\.)\b', passage, re.I))
    has_import  = bool(re.search(r'\b(?:important|significant|crucial|essential|major|key role|plays a|function|advantage|use)\b', passage, re.I))
    has_list    = re.search(r'\b(types|kinds|forms|features|characteristics|factors|elements|components|stages|effects|parts|organs|tissues|cells)\b', passage, re.I)
    has_math_prob = re.search(r'\b(calculate|determine the ratio|cross between|probability of|energy available at|J|joules?|%|percent|ratio|F1|F2|generation|descendant|10%|ten percent|formula|estimated size|µm|micrometre|diameter|magnification)\b', passage, re.I)
    has_formula = bool(re.search(r'\b(formula|estimated size|µm|micrometre|diameter|magnification)\b', passage, re.I) or ('=' in passage and ('/' in passage or 'x' in passage or '+' in passage)))
    has_function = bool(re.search(r'\b(?:function|role|helps|responsible for|allows)\b', passage, re.I))

    qs = []  # (intent, q_en, q_hi)

    def add(intent, q_en, q_hi=""):
        qs.append((intent, q_en, q_hi))

    # Always: explain + describe + what is
    add("explain_concept",    f"Explain {kt} in the context of {chapter}.")
    add("explain_concept",    f"Describe in detail: {fs}?")
    add("definition",         f"What is meant by {kt}?")

    if has_defn:
        add("definition",     f"Define {kt} as given in your NCERT textbook.")
        add("definition",     f"What is {kt}? Give a brief definition.")

    if has_cause:
        add("explain_concept",f"What are the causes or factors responsible for {kt}?")
        add("explain_concept",f"Why does {kt} occur? Explain with reasons.")

    if has_compare:
        if len(nps) >= 2:
            add("compare_concepts", f"What is the difference between {nps[0]} and {nps[1]}?")
            add("compare_concepts", f"Compare {nps[0]} and {nps[1]}.")
        else:
            add("compare_concepts", f"How does {kt} differ from related biological concepts?")

    if has_process:
        add("explain_concept",f"Explain the biological process of {kt}.")
        add("explain_concept",f"How does {kt} work or develop? Describe step by step.")

    if has_example:
        add("give_example",   f"Give an example of {kt} as mentioned in the chapter.")
        add("give_example",   f"Illustrate {kt} with an example from the NCERT text.")

    if has_import:
        add("application_reallife", f"Why is {kt} important? Discuss its significance.")
        add("application_reallife", f"What is the role or function of {kt} in living organisms?")

    if has_list:
        lword = has_list.group(1).lower()
        add("list_enumerate", f"List the {lword} of {kt}.")
        add("list_enumerate", f"What are the main {lword} of {kt}?")
        
    if has_function:
        add("describe_function", f"What is the main function of {kt}?")

    if has_math_prob:
        term = has_math_prob.group(1).lower()
        add("numerical_problem", f"Explain the calculation or mathematical biological concept (like {term}) described regarding {kt}.")
        add("numerical_problem", f"Based on the NCERT text, solve or explain the numerical biological problem related to {kt}.")

    if has_formula:
        add("formula_application", f"Using the formula or calculation provided in the text regarding {kt}, explain how to compute the result.")
        add("formula_application", f"What is the mathematical relationship or calculation described for {kt}?")

    if nps:
        add("explain_concept",f"What is the significance of {nps[0]} in {chapter}?")
    if len(nps) >= 2:
        add("explain_concept",f"How are {nps[0]} and {nps[1]} related?")

    add("application_reallife", f"How is {kt} relevant to human life or nature?")

    return qs

# ── Entry builder ──────────────────────────────────────────────────────────────
def make_entry(idx, info, intent, q_en, q_hi, passage, topic):
    sc  = info["subject_code"]
    cn  = info["chapter_num"]
    cv  = info["class_val"]
    return {
        "id":               f"c{cv}_{sc}_{cn}_{idx:04d}",
        "class":            cv,
        "subject":          info["subject"],
        "chapter":          info["chapter"],
        "chapter_num":      cn,
        "topic":            topic,
        "is_student_query": False,
        "intent":           intent,
        "question_english": q_en,
        "question_hinglish": q_hi,  # Will be blank, ready for enrich_hinglish.py
        "answer_english":   passage,
        "answer_hinglish":  "",
        "ncert_context":    passage,
        "messages": [
            {"role": "system", "content": passage},
            {"role": "user", "content": q_en},
            {"role": "assistant", "content": passage}
        ]
    }

def process(folder, info):
    path = PROCESSED / folder / "cleaned_text.txt"
    if not path.exists():
        print(f"  MISSING: {path}"); return []
    raw = path.read_text(encoding="utf-8")
    plist = build_passages(raw)
    entries, seen, ctr = [], set(), 0
    for p in plist:
        topic = key_term(p)
        for intent, q_en, q_hi in gen_questions(p, info["chapter"], info["subject"]):
            if q_en.lower() in seen: continue
            seen.add(q_en.lower())
            ctr += 1
            entries.append(make_entry(ctr, info, intent, q_en, q_hi, p, topic))
    return entries

def main():
    print("\n=== Generating Biology QA Dataset ===")
    all_entries = []
    for f, info in CHAPTERS.items():
        res = process(f, info)
        all_entries.extend(res)
        print(f"  {info['chapter']} ({info['subject']}) ... {len(res)} entries")

    OUT.write_text(json.dumps(all_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "="*60)
    print(f"✅  Saved → {OUT}")
    print(f"    Total : {len(all_entries)} entries\n")
    
    # Optional: summary by class and chapter
    by_class = Counter(e["class"] for e in all_entries)
    by_chap = Counter(e["chapter"] for e in all_entries)
    by_intent = Counter(e["intent"] for e in all_entries)
    
    print("  By class:")
    for k, v in by_class.most_common(): print(f"    Class {k:<12}: {v}")
    print("\n  By chapter:")
    for k, v in by_chap.most_common(): print(f"    {k:<55}: {v}")
    print("\n  By intent:")
    for k, v in by_intent.most_common(): print(f"    {k:<30}: {v}")

if __name__ == "__main__":
    main()
