"""
generate_ss9_json.py  —  Generates data/ss9.json from SS9 cleaned chapter texts.
Run: python training/generate_ss9_json.py
"""
import re, json
from pathlib import Path
from collections import Counter

CHAPTERS = {
    "ss9_ch01": {"chapter": "Understanding Social Science",                     "chapter_num":"ch01","subject":"Social Science","subject_code":"ss"},
    "ss9_ch02": {"chapter": "Shaping of the Earth's Surface",                   "chapter_num":"ch02","subject":"Geography","subject_code":"geo"},
    "ss9_ch03": {"chapter": "Atmosphere and Climate",                             "chapter_num":"ch03","subject":"Geography","subject_code":"geo"},
    "ss9_ch04": {"chapter": "Early Humans and Beginning of Civilisation",         "chapter_num":"ch04","subject":"History",  "subject_code":"hist"},
    "ss9_ch05": {"chapter": "State and Society up to 1000 CE",                   "chapter_num":"ch05","subject":"History",  "subject_code":"hist"},
    "ss9_ch06": {"chapter": "Democracy",                                          "chapter_num":"ch06","subject":"Civics",   "subject_code":"civ"},
    "ss9_ch07": {"chapter": "Elections",                                          "chapter_num":"ch07","subject":"Civics",   "subject_code":"civ"},
    "ss9_ch08": {"chapter": "Building Blocks in Economics: The Problem of Choice","chapter_num":"ch08","subject":"Economics","subject_code":"eco"},
    "ss9_ch09": {"chapter": "The Price Puzzle: What Drives the Market",           "chapter_num":"ch09","subject":"Economics","subject_code":"eco"},
}

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT       = ROOT / "data" / "ss9.json"

# ── Sentence splitter ────────────────────────────────────────────────────────
def split_sentences(text):
    # protect abbreviations
    text = re.sub(r'\b(etc|vs|Dr|Mr|St|No|Fig|e\.g|i\.e|approx|govt|dept)\.',
                  r'\1<DOT>', text)
    # split on ". " followed by capital
    parts = re.split(r'\.\s+(?=[A-Z])', text)
    out = []
    for p in parts:
        p = p.replace('<DOT>', '.').strip()
        if len(p) > 30:
            out.append(p.rstrip('.') + '.')
    return out

# ── Build passages from a text using sliding sentence windows ─────────────────
SKIP_LINE = re.compile(
    r'^(\d+\.\s|Big Questions?|The\s*$|CChhaapptteerr|iinndddd|LET\'S|BEYOND|'
    r'UNDERSTANDING|India and Beyond|https?://|courtesy|wikimedia)', re.I
)
ARTIFACT = re.compile(r'(.)\1{3,}')

def build_passages(raw_text, window=5, step=3, min_len=120):
    """
    Split cleaned_text.txt (single-newline lines) into sentence-window passages.
    Steps:
      1. Filter out noisy lines (PDF artifacts, chapter headers, numbered lists).
      2. Join remaining lines into a single text blob.
      3. Split into sentences.
      4. Slide a window of `window` sentences with step `step`.
    """
    lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line: continue
        if ARTIFACT.search(line): continue
        if SKIP_LINE.match(line): continue
        if re.match(r'^\d+$', line): continue          # bare page numbers
        if len(line.split()) < 4: continue             # too short
        lines.append(line)

    blob = ' '.join(lines)
    sents = split_sentences(blob)

    passages = []
    for i in range(0, len(sents) - 1, step):
        chunk_sents = sents[i: i + window]
        chunk = ' '.join(chunk_sents)
        # Skip if too short or too many question marks (likely header lists)
        if len(chunk) < min_len: continue
        if chunk.count('?') > 3: continue
        passages.append(chunk)
    return passages

# ── Key term extraction ───────────────────────────────────────────────────────
STOP_STARTS = {'what', 'how', 'why', 'when', 'where', 'which', 'who', 'the',
               'a', 'an', 'it', 'in', 'on', 'at', 'for', 'of', 'and', 'or',
               'but', 'these', 'those', 'this', 'that'}

def key_term(passage):
    # Pattern: "X is/are/refers to/means"
    m = re.match(r'([A-Z][^.!?]{3,50}?)\s+(?:is|are|refers to|means|can be defined)', passage)
    if m:
        t = m.group(1).strip()
        words = t.split()
        if words and words[0].lower() not in STOP_STARTS and len(words) <= 6:
            return t
    # Capitalized multi-word phrases (2-4 words)
    caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', passage)
    for c in caps:
        if c.split()[0].lower() not in STOP_STARTS:
            return c
    # Single capitalized meaningful word
    caps1 = re.findall(r'\b([A-Z][a-z]{3,})\b', passage)
    for c in caps1:
        if c.lower() not in STOP_STARTS:
            return c
    # Fallback: first few words
    words = passage.split()
    meaningful = [w for w in words[:8] if w[0].isupper() and w.lower() not in STOP_STARTS]
    return ' '.join(meaningful[:3]) if meaningful else words[0]

def noun_phrases(passage, n=2):
    caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', passage)
    seen, out = set(), []
    for c in caps:
        if c.lower() not in seen and c.split()[0].lower() not in STOP_STARTS:
            seen.add(c.lower())
            out.append(c)
        if len(out) == n: break
    return out

# ── Hinglish generation (rule-based, no API) ─────────────────────────────────
# Common English words → Hindi equivalents used in Hinglish
HI_MAP = {
    # Connectors / grammar
    "and": "aur", "but": "lekin", "because": "kyunki", "so": "isliye",
    "therefore": "isliye", "however": "lekin", "also": "bhi",
    "although": "halanki", "whereas": "jabki", "while": "jabki",
    "when": "jab", "where": "jahan", "which": "jo", "who": "jo",
    "this": "yeh", "these": "ye", "that": "woh", "those": "woh",
    "its": "uski", "their": "unka", "our": "hamara",
    "is": "hai", "are": "hain", "was": "tha", "were": "the",
    "has": "hai", "have": "hain", "had": "tha",
    "can": "kar sakta hai", "could": "kar sakta tha",
    "not": "nahi", "no": "nahi",
    "very": "bahut", "many": "bahut saare", "much": "bahut",
    "more": "zyada", "most": "sabse zyada", "less": "kam",
    "first": "pehle", "then": "phir", "after": "baad mein",
    "before": "pehle", "between": "ke beech", "among": "ke beech",
    "through": "ke zariye", "with": "ke saath", "without": "bina",
    "by": "dwara", "from": "se", "into": "mein", "upon": "par",
    "like": "jaise", "such": "aisa", "same": "same",
    "different": "alag", "similar": "milta-julta",
    "important": "important", "main": "mukhya", "major": "mukhya",
    "small": "chhota", "large": "bada", "long": "lamba",
    "people": "log", "society": "samaj", "government": "sarkar",
    "country": "desh", "world": "duniya", "life": "zindagi",
    "human": "manav", "nature": "prakriti", "environment": "paryavaran",
    "time": "samay", "period": "kaal", "century": "sadi",
    "region": "kshetra", "land": "bhumi", "area": "kshetra",
    "system": "pranali", "power": "shakti", "right": "adhikar",
    "law": "kanoon", "rule": "niyam", "policy": "niti",
    "work": "kaam", "food": "bhojan", "water": "paani",
    "means": "matlab hai", "refers": "kehte hain",
}

# Intent → Hinglish question template (use {kt} placeholder)
INTENT_Q_HI = {
    "definition": [
        "{kt} kya hota hai?",
        "{kt} ko define karo NCERT ke anusar.",
        "{kt} ka matlab kya hai?",
    ],
    "explain_concept": [
        "{kt} ke baare mein explain karo.",
        "{kt} ko detail mein samjhao.",
        "{kt} kaise kaam karta hai? Samjhao.",
        "{kt} ki importance kya hai? {chapter} ke context mein batao.",
    ],
    "compare_concepts": [
        "{a} aur {b} mein kya difference hai?",
        "{a} aur {b} ko compare karo.",
        "{kt} aur related concepts mein kya fark hai?",
    ],
    "give_example": [
        "{kt} ka ek example do jo chapter mein diya gaya hai.",
        "{kt} ko ek example se samjhao.",
    ],
    "application_reallife": [
        "{kt} Indian society ya daily life mein kaise useful hai?",
        "{kt} ki kya significance hai? Discuss karo.",
        "{subject} mein {kt} ka kya role hai?",
    ],
    "list_enumerate": [
        "{kt} ke main {lword} list karo.",
        "{kt} ke {lword} kya hain?",
    ],
}

def _hi_word(w):
    """Return Hindi equivalent if it exists, else return the word as-is."""
    core = w.strip('.,;:?!\'"()')
    mapped = HI_MAP.get(core.lower())
    if mapped:
        # preserve trailing punctuation
        trail = w[len(core):]
        return mapped + trail
    return w

def passage_to_hinglish(passage):
    """
    Convert an English passage to Hinglish by:
    1. Replacing common grammar / connector words with Hindi equivalents.
    2. Keeping technical terms / proper nouns in English.
    3. Adding sentence-ending Hindi markers ("hota hai", "hai", etc.).
    """
    sents = split_sentences(passage)
    hi_sents = []
    enders = ["hota hai.", "hai.", "hain.", "tha.", "the."]
    for i, sent in enumerate(sents):
        words = sent.split()
        new_words = [_hi_word(w) for w in words]
        # Replace the last word (usually a period-ended verb) with a Hindi ender
        # only if the sentence doesn't already have a Hindi word at the end
        joined = ' '.join(new_words).rstrip('.')
        ender = enders[i % len(enders)]
        hi_sents.append(joined + ' ' + ender)
    return ' '.join(hi_sents)

def make_hinglish_question(intent, english_q, kt, nps, chapter, subject, lword=None):
    """Generate a Hinglish question from a template based on intent."""
    templates = INTENT_Q_HI.get(intent, INTENT_Q_HI["explain_concept"])
    # pick template deterministically based on kt hash
    tmpl = templates[hash(kt) % len(templates)]
    a = nps[0] if nps else kt
    b = nps[1] if len(nps) > 1 else kt
    return tmpl.format(
        kt=kt,
        chapter=chapter,
        subject=subject.lower(),
        a=a, b=b,
        lword=lword or "points"
    )

# ── Question generators ────────────────────────────────────────────────────────
def gen_questions(passage, chapter, subject):
    kt  = key_term(passage)
    nps = noun_phrases(passage)
    sents = split_sentences(passage)
    first_sent = sents[0] if sents else passage.split('.')[0]
    fs = first_sent if len(first_sent) < 100 else ' '.join(first_sent.split()[:15]) + '...'

    has_defn    = bool(re.search(r'\b(?:is defined as|refers to|is a |are a |means|can be described as)\b', passage, re.I))
    has_cause   = bool(re.search(r'\b(?:because|due to|causes?|led to|results? in|reason for)\b', passage, re.I))
    has_compare = bool(re.search(r'\b(?:differ|whereas|while|unlike|compared|contrast|however)\b', passage, re.I))
    has_process = bool(re.search(r'\b(?:process|mechanism|forms?|develops?|produc(?:es|ed)|works?|steps?)\b', passage, re.I))
    has_example = bool(re.search(r'\b(?:for example|such as|like|instance|including|e\.g\.)\b', passage, re.I))
    has_import  = bool(re.search(r'\b(?:important|significant|crucial|essential|major|key role|plays a)\b', passage, re.I))
    has_list    = re.search(r'\b(types|kinds|forms|features|characteristics|factors|elements|components|stages|effects)\b', passage, re.I)
    has_number  = re.search(r'\b(\d{2,}[\d,\.]*)\b', passage)

    qs = []  # (intent, q_en, q_hi)

    def add(intent, q_en, lword=None):
        q_hi = make_hinglish_question(intent, q_en, kt, nps, chapter, subject, lword)
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
            add("compare_concepts", f"How does {kt} differ from related concepts?")

    if has_process:
        add("explain_concept",f"Explain the process of {kt}.")
        add("explain_concept",f"How does {kt} work or develop? Describe step by step.")

    if has_example:
        add("give_example",   f"Give an example of {kt} as mentioned in the chapter.")
        add("give_example",   f"Illustrate {kt} with an example from the NCERT text.")

    if has_import:
        add("application_reallife", f"Why is {kt} important? Discuss its significance.")
        add("application_reallife", f"What is the role of {kt} in {subject.lower()}?")

    if has_list:
        lword = has_list.group(1).lower()
        add("list_enumerate", f"List the {lword} of {kt}.", lword)
        add("list_enumerate", f"What are the main {lword} of {kt}?", lword)

    if has_number:
        num = has_number.group(1)
        add("definition",     f"What does the figure {num} represent in the context of {kt}?")

    if nps:
        add("explain_concept",f"What is the significance of {nps[0]} in {chapter}?")
    if len(nps) >= 2:
        add("explain_concept",f"How are {nps[0]} and {nps[1]} related?")

    add("application_reallife", f"How is {kt} relevant to Indian society or daily life?")

    return qs  # now returns (intent, q_en, q_hi) tuples

# ── Entry builder ──────────────────────────────────────────────────────────────
def make_entry(idx, info, intent, q_en, q_hi, passage, topic):
    sc  = info["subject_code"]
    cn  = info["chapter_num"]
    return {
        "id":               f"c09_{sc}_{cn}_{idx:04d}",
        "class":            "9",
        "subject":          info["subject"],
        "chapter":          info["chapter"],
        "chapter_num":      cn,
        "topic":            topic,
        "is_student_query": False,
        "intent":           intent,
        "question_english": q_en,
        "question_hinglish": "",
        "answer_english":   passage,
        "answer_hinglish":  "",
        "ncert_context":    passage
    }

def process(folder, info):
    path = PROCESSED / folder / "cleaned_text.txt"
    if not path.exists():
        print(f"  MISSING: {path}"); return []
    raw = path.read_text(encoding="utf-8")
    plist = build_passages(raw)
    entries, seen, ctr = [], set(), 0
    for p in plist:
        topic = key_term(p)   # use key term of the passage as the topic label
        for intent, q_en, q_hi in gen_questions(p, info["chapter"], info["subject"]):
            if q_en.lower() in seen: continue
            seen.add(q_en.lower())
            ctr += 1
            entries.append(make_entry(ctr, info, intent, q_en, q_hi, p, topic))
    return entries

def main():
    all_entries = []
    print("\n=== Generating SS9 QA Dataset ===\n")
    for folder, info in CHAPTERS.items():
        print(f"  {info['chapter'][:50]} ({info['subject']}) ...", end=" ", flush=True)
        e = process(folder, info)
        all_entries.extend(e)
        print(f"{len(e)} entries")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅  Saved → {OUT}")
    print(f"    Total : {len(all_entries)} entries\n")
    print("  By subject:")
    for s, c in sorted(Counter(e["subject"] for e in all_entries).items()):
        print(f"    {s:<20}: {c}")
    print("\n  By chapter:")
    for ch, c in Counter(e["chapter"] for e in all_entries).most_common():
        print(f"    {ch[:55]:<55}: {c}")
    print("\n  By intent:")
    for i, c in Counter(e["intent"] for e in all_entries).most_common():
        print(f"    {i:<30}: {c}")

if __name__ == "__main__":
    main()
