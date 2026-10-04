"""
ncert_utils.py — Shared utilities for NCERT dataset generation
==============================================================
Used by generate_bio_json.py and generate_ss9_json.py.
"""

import re
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# ARTIFACT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

MIRROR_ARTIFACT = re.compile(r'([A-Za-z])\1([A-Za-z])\2([A-Za-z])\3')
ARTIFACT_RE = re.compile(r'(.)\1{4,}')

def is_clean_text(text):
    return not MIRROR_ARTIFACT.search(text) and not ARTIFACT_RE.search(text)

# ═══════════════════════════════════════════════════════════════════════════════
# SENTENCE SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════

ABBREV_RE = re.compile(
    r'\b(etc|vs|Dr|Mr|Mrs|St|No|Fig|e\.g|i\.e|approx|govt|dept|'
    r'cm|mm|nm|kg|ml|µm|μm|mL|L|s|min|h|yr)\.',
    re.I
)

def split_sentences(text):
    text = ABBREV_RE.sub(r'\1<DOT>', text)
    parts = re.split(r'\.\s+(?=[A-Z])', text)
    out = []
    for p in parts:
        p = p.replace('<DOT>', '.').strip()
        if len(p) > 3:
            out.append(p.rstrip('.') + '.')
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# PASSAGE BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

SKIP_LINE = re.compile(
    r'^(\d+\.\s|Big Questions?|The\s*$|CChhaapptteerr|iinndddd|LET\'S|BEYOND\s*$|'
    r'UNDERSTANDING\s*$|India and Beyond|https?://|courtesy|wikimedia|'
    r'Reprint\s+\d{4}|ExplorationGrade|Grade\s+\d+)',
    re.I
)

# Pattern to split on page-header artifacts (e.g. CChhaapptteerr--0022..iinndddd)
_PAGE_SPLIT = re.compile(
    r'CChhaapptteerr[^A-Za-z]{0,30}iinndddd[^.]{0,80}(?:PPMM|AAMM|\d{2}:\d{2}:\d{2})|'
    r'ExplorationGrade\d+\s+CChhaapptteerr|'
    r'\d+\s+Reprint\s+\d{4}',
    re.I
)

def _clean_blob(raw_text):
    """
    Convert raw text to a clean single blob:
    - For multi-line files: filter bad lines, join.
    - For single-line files: split on page artifacts, keep clean segments, join.
    """
    lines = raw_text.splitlines()
    if len(lines) > 5:
        # Normal multi-line file
        good = []
        for line in lines:
            line = line.strip()
            if not line: continue
            if SKIP_LINE.match(line): continue
            if re.match(r'^\d+$', line): continue
            if MIRROR_ARTIFACT.search(line): continue
            if ARTIFACT_RE.search(line): continue
            good.append(line)
        blob = ' '.join(good)
    else:
        # Single-line (or few-line) file — split on page markers
        blob = ' '.join(lines)
        segments = _PAGE_SPLIT.split(blob)
        good_segs = []
        for seg in segments:
            seg = MIRROR_ARTIFACT.sub(' ', seg)
            seg = ARTIFACT_RE.sub(' ', seg)
            # Remove inline cross-reference text (e.g. "Grade 7 Curiosity", "Reprint 2026")
            seg = re.sub(r'\b(?:Grade\s+\d+|Reprint\s+\d{4}|ExplorationGrade\d+)\s+\w*', ' ', seg)
            seg = re.sub(r'\bCuriosity\s+', ' ', seg)
            # Remove inline marginal questions ("y Where does", "y How", "y What")
            seg = re.sub(r'\sy\s+(?:Where|How|What|Why|When|Which|Who|Can|Is|Are|Do|Does)\b[^.!?]*[.!?]?', ' ', seg)
            seg = re.sub(r'\s+', ' ', seg).strip()
            if len(seg) > 50:
                good_segs.append(seg)
        blob = ' '.join(good_segs)

    # Strip leading chapter/book title before the first proper sentence
    # Pattern: "Chapter Title ...  First sentence starts here"
    # The chapter title bleeds into the first sentence in single-line PDFs
    blob = re.sub(
        r'^[A-Z][A-Za-z:\s]{5,60}(?:Life|Science|Society|History|Geography|Economics|Civics|Action)\s+'
        r'(?=[A-Z][a-z])',
        '', blob
    )
    # Also remove "Chapter N" prefixes
    blob = re.sub(r'^Chapter\s+\d+\s+', '', blob)
    return blob.strip()


def build_passages(raw_text, window=5, step=3, min_len=250):
    blob = _clean_blob(raw_text)
    sents = split_sentences(blob)
    passages = []
    for i in range(0, len(sents) - 1, step):
        chunk = ' '.join(sents[i: i + window])
        # Strip empty figure references: (), (Fig. X), (X)
        chunk = re.sub(r'\(\s*\)', '', chunk)
        chunk = re.sub(r'\(\s*[Ff]ig(?:ure)?[^)]{0,20}\)', '', chunk)
        # Strip stray page-number refs like "-27" at word boundaries
        chunk = re.sub(r'(?<![\w(])-\s*\d{1,3}(?![\d\w])', '', chunk)
        # Strip marginal book references that slipped through (e.g. "Meet a Scientist")
        chunk = re.sub(r'\bMeet a (?:Scientist|Researcher|Expert)\b', '', chunk)
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        if len(chunk) < min_len: continue
        if chunk.count('?') > 3: continue
        if ARTIFACT_RE.search(chunk): continue
        passages.append(chunk)
    return passages

# ═══════════════════════════════════════════════════════════════════════════════
# KEY TERM EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

STOP_STARTS = {
    # Pronouns / articles / prepositions
    'what', 'how', 'why', 'when', 'where', 'which', 'who', 'the', 'a', 'an',
    'it', 'in', 'on', 'at', 'for', 'of', 'and', 'or', 'but', 'these', 'those',
    'this', 'that', 'is', 'are', 'was', 'were', 'be', 'from', 'by', 'with',
    'to', 'into', 'out', 'up', 'down', 'over', 'under', 'about', 'between',
    # Conjunctions / transitions
    'horizontally', 'vertically', 'however', 'therefore', 'thus', 'hence',
    'since', 'although', 'whereas', 'while', 'now', 'also', 'even', 'just',
    'both', 'all', 'each', 'every', 'such', 'some', 'many', 'most', 'more',
    'other', 'another', 'during', 'after', 'before', 'so', 'yet', 'still',
    'then', 'there', 'here', 'together', 'inside', 'outside', 'above', 'below',
    'when', 'while', 'once', 'upon', 'through', 'across', 'among', 'within',
    # Biology / SS chapter-specific words that are too generic to be key terms
    'trophic', 'various', 'different', 'similar', 'certain', 'particular',
    'following', 'given', 'known', 'called', 'found', 'used', 'seen',
}

def key_term(passage):
    m = re.match(r'([A-Z][^.!?]{3,50}?)\s+(?:is|are|refers to|means|can be defined)', passage)
    if m:
        t = m.group(1).strip()
        words = t.split()
        if (words and words[0].lower() not in STOP_STARTS
                and words[-1].lower() not in STOP_STARTS and len(words) <= 6
                and len(t) >= 4):
            return t
    caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', passage)
    for c in caps:
        words = c.split()
        if (words[0].lower() not in STOP_STARTS
                and words[-1].lower() not in STOP_STARTS
                and len(c) >= 5):   # avoid single short words like "In"
            return c
    caps1 = re.findall(r'\b([A-Z][a-z]{4,})\b', passage)  # min 5 chars (was 3)
    for c in caps1:
        if c.lower() not in STOP_STARTS:
            return c
    words = passage.split()
    meaningful = [
        w for w in words[:12]
        if w[0].isupper() and w.lower() not in STOP_STARTS and w.isalpha() and len(w) > 4
    ]
    return meaningful[0] if meaningful else "the concept"


def is_good_first_sent(s):
    """Return True if a sentence is suitable as a 'Describe in detail' question."""
    if not s or len(s) < 15:
        return False
    # Too long — hard to use as a question topic
    if len(s) > 150:
        return False
    # Contains empty figure references or page numbers
    if re.search(r'\(\s*\)|\(\s*-\s*\d|\(\s*Fig', s):
        return False
    # Starts with a transition / conjunction / preposition word
    first = s.split()[0].lower() if s.split() else ''
    if first in STOP_STARTS or first in {'so', 'but', 'yet', 'also', 'together',
                                          'inside', 'outside', 'therefore', 'however',
                                          'additionally', 'furthermore', 'moreover',
                                          'nevertheless', 'otherwise', 'firstly',
                                          'secondly', 'thirdly', 'finally', 'note',
                                          'observe', 'remember', 'recall'}:
        return False
    # Contains lingering artefacts
    if re.search(r'[A-Z]{3,}|\d{4,}|[_=<>{}\[\]]', s):
        return False
    # Contains marginal cross-reference words
    if re.search(r'\b(Meet a|Curiosity|Grade \d|Reprint|iinndddd|ExplorationGrade)\b', s, re.I):
        return False
    return True

def noun_phrases(passage, n=2):
    caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', passage)
    seen, out = set(), []
    for c in caps:
        if c.lower() not in seen and c.split()[0].lower() not in STOP_STARTS:
            seen.add(c.lower())
            out.append(c)
        if len(out) == n:
            break
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# NCERT QUESTION SECTION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

SECTION_MARKERS = [
    ('Revise, Reflect, Refine', 'rrr'),
    ('Q U E S T I O N S', 'intext'),
    ('E X E R C I S E S', 'exercises'),
    ('Questions and activities', 'activities'),
]

END_MARKERS = [
    'What you have learnt', 'The Journey Beyond', 'At a Glance',
    'The Quest Continues', 'Reprint 20', 'CChhaapptteerr', 'iinndddd',
    'Notes\n', 'Understanding Society:', 'Grade 9 -', 'Grade 10 -',
]

SKIP_Q_RE = re.compile(
    r'\b('
    r'refer to (the )?(fig|figure|picture|diagram|table|image|graph)|'
    r'look at (the )?(fig|figure|picture|diagram|table|image|graph|above)|'
    r'shown in (the )?(fig|figure|above|below|diagram)|'
    r'given in (the )?(fig|figure|above|below)|'
    r'from the (fig|figure|picture|diagram|table|above|graph)|'
    r'(fig|figure)\s*\d+|'
    r'the (picture|diagram|graph|table) (given|shown|above|below)|'
    r'following (figure|picture|diagram|table)'
    r')\b',
    re.I
)

def _find_sections(text):
    sections = []
    for marker, stype in SECTION_MARKERS:
        idx = 0
        while True:
            i = text.find(marker, idx)
            if i < 0: break
            content_start = i + len(marker)
            end = len(text)
            for em in END_MARKERS:
                j = text.find(em, content_start + 20)
                if 0 < j < end: end = j
            for nm, _ in SECTION_MARKERS:
                if nm == marker: continue
                j = text.find(nm, content_start + 20)
                if 0 < j < end: end = j
            content = text[content_start:end].strip()
            if len(content) > 30:
                sections.append((stype, content))
            idx = i + 1
    return sections

def _parse_questions_from_section(section_text):
    parts = re.split(r'(?<!\d)(\d{1,2})\.\s+(?=[A-Z(])', section_text)
    questions = []
    i = 1
    while i < len(parts) - 1:
        q_num = parts[i]
        content = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        option_re = re.compile(
            r'\(([a-d]|i{1,3}v?|vi{0,3})\)\s*([^(]{3,120}?)(?=\s*\([a-divx]|\d{1,2}\.|$)',
            re.S
        )
        options = {}
        for om in option_re.finditer(content):
            label = om.group(1).strip()
            opt_text = om.group(2).strip().rstrip('.,;')
            if len(opt_text) < 150:
                options[label] = opt_text
        q_clean = option_re.sub('', content).strip()
        q_clean = re.sub(r'\s+', ' ', q_clean)
        sub_parts = []
        if len(options) < 2:
            sub_re = re.compile(r'\(([a-f])\)\s*([^(]{15,}?)(?=\s*\([a-f]\)|$)', re.S)
            for sm in sub_re.finditer(content):
                sp_text = sm.group(2).strip()
                if len(sp_text) > 15:
                    sub_parts.append((sm.group(1), sp_text))
        questions.append((q_num, q_clean, options, sub_parts))
        i += 2
    return questions

def _classify_intent(q_text, options, sub_parts):
    qt = q_text.lower()
    if re.search(r'assertion', qt) and re.search(r'reason', qt):
        return 'assertion_reason'
    if len(options) >= 3:
        return 'mcq'
    if sub_parts and len(sub_parts) >= 2:
        return 'case_study'
    if re.search(r'\b(list|name all|enumerate|give (two|three|four|five))\b', qt):
        return 'list_enumerate'
    if re.search(r'\b(define|what is meant by|what do you mean|meaning of)\b', qt):
        return 'definition'
    if re.search(r'\b(difference|compare|distinguish|versus)\b', qt):
        return 'compare_concepts'
    if re.search(r'\b(example|illustrate|give an instance)\b', qt):
        return 'give_example'
    if re.search(r'\b(arrange|order|sequence|correct order)\b', qt):
        return 'list_enumerate'
    if re.search(r'\b(why|give reason|explain why)\b', qt):
        return 'short_answer'
    if re.search(r'\b(how|describe|explain|discuss|outline)\b', qt):
        return 'explain_concept'
    return 'analytical'

def _derive_answer_for_ncert_q(q_text, options, chapter_text):
    if options:
        best_label, best_score = None, -1
        for label, opt_text in options.items():
            words = set(re.findall(r'\b[a-z]{4,}\b', opt_text.lower()))
            score = sum(1 for w in words if w in chapter_text.lower())
            if score > best_score:
                best_score, best_label = score, label
        if best_label:
            return f"The correct answer is ({best_label}) {options[best_label]}."
        return ""
    key_words = set(re.findall(r'\b[a-z]{4,}\b', q_text.lower()))
    sents = split_sentences(chapter_text)
    best_passage, best_score = "", -1
    win = 5
    for i in range(0, max(1, len(sents) - win + 1), 2):
        window = ' '.join(sents[i: i + win])
        score = sum(1 for w in key_words if w in window.lower())
        if score > best_score:
            best_score, best_passage = score, window
    return best_passage

def extract_ncert_questions(raw_text, info, seen=None):
    if seen is None:
        seen = set()
    sections = _find_sections(raw_text)
    results = []
    for stype, section_content in sections:
        parsed = _parse_questions_from_section(section_content)
        for q_num, q_text, options, sub_parts in parsed:
            if len(q_text.strip()) < 10: continue
            if SKIP_Q_RE.search(q_text): continue
            q_clean = q_text.strip()
            if not q_clean.endswith(('?', '.')):
                q_clean += '?'
            q_clean = re.sub(r'\s+', ' ', q_clean)
            if q_clean.lower() in seen: continue
            seen.add(q_clean.lower())
            intent = _classify_intent(q_text, options, sub_parts)
            if options and intent in ('mcq', 'assertion_reason'):
                full_q = (q_clean + ' '
                          + ' '.join(f'({k}) {v}' for k, v in options.items()))
            elif sub_parts:
                full_q = q_clean + ' ' + ' '.join(
                    f'({l}) {t.rstrip(".")}?' for l, t in sub_parts)
            else:
                full_q = q_clean
            answer = _derive_answer_for_ncert_q(q_text, options, raw_text)
            results.append({
                'intent': intent,
                'q_en': full_q,
                'answer_en': answer,
                'source': stype,
            })
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# TRIVIAL QUESTION GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def gen_trivial_questions(passage, chapter, subject):
    kt = key_term(passage)
    sents = split_sentences(passage)
    qs = []
    for sent in sents[:5]:
        s = sent.strip()
        if len(s) < 20 or not is_clean_text(s): continue

        # name_the: "X is called/known as Y"
        m = re.search(
            r'([A-Z][A-Za-z\s]{2,25}?)\s+(?:is|are)\s+'
            r'(?:called|known as|termed|referred to as)\s+([A-Z][A-Za-z\s]{1,25}?)(?:\.|,)',
            s
        )
        if m:
            subj = m.group(1).strip()
            term = m.group(2).strip().rstrip('.,;')
            if subj.split()[0].lower() not in STOP_STARTS and 1 <= len(term.split()) <= 4:
                qs.append(('name_the', f"What is {subj} called?", f"{term}."))

        # fill_blank: "X is defined as / refers to ..."
        m = re.search(
            r'^([A-Z][A-Za-z\s]{2,25}?)\s+'
            r'(?:is defined as|refers to|means|is the process of)\s+'
            r'(.{15,100}?)(?:\.|$)', s
        )
        if m:
            term = m.group(1).strip()
            defn = m.group(2).strip()
            if term.split()[0].lower() not in STOP_STARTS and len(term.split()) <= 4:
                qs.append(('fill_blank', f"_____ is defined as {defn}.", f"{term}."))

        # true_false: short declarative sentences
        wc = len(s.split())
        if 8 <= wc <= 22 and not s.endswith('?'):
            if re.search(r'\b(?:is|are|was|were|have|has|can|produces?|contains?)\b', s, re.I):
                qs.append(('true_false',
                            f"True or False: {s.rstrip('.')}.",
                            f"True. {s.rstrip('.')} as stated in the NCERT textbook."))

        # one_word: process/organ naming
        m = re.search(
            r'(?:process called|process of|organ called|structure called|'
            r'phenomenon called|method called)\s+([A-Z][A-Za-z]{2,25})',
            s, re.I
        )
        if m:
            term = m.group(1).strip()
            pre = s[:m.start()].rsplit('.', 1)[-1].strip()
            if pre and len(pre.split()) >= 4:
                qs.append(('one_word', f"Name the process or structure: {pre}?", f"{term}."))

        # one_word: "What is X?" from "X is a/an/the ..."
        m = re.match(r'^([A-Z][A-Za-z\s]{2,25}?)\s+(?:is|are)\s+(?:a|an|the)\s+', s)
        if m:
            subj = m.group(1).strip()
            if (subj.split()[0].lower() not in STOP_STARTS
                    and len(subj.split()) <= 3 and wc >= 6):
                ans = s if wc <= 20 else ' '.join(s.split()[:20]) + '.'
                qs.append(('one_word', f"What is {subj}?", ans))

    return qs

# ═══════════════════════════════════════════════════════════════════════════════
# BIOLOGICAL REACTIONS DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

BIOLOGICAL_REACTIONS = [
    {
        'name': 'Photosynthesis',
        'trigger': re.compile(
            r'\b(photosynthesis|chlorophyll|chloroplast|autotrophic nutrition|'
            r'carbon dioxide.{0,30}water.{0,30}sunlight)\b', re.I),
        'context_chapters': {'Life Processes', 'Cell: The Building Block of Life', 'Our Environment'},
        'questions': [
            ('reaction_equation', "Write the balanced chemical equation for photosynthesis."),
            ('reaction_equation', "What are the reactants and products in the photosynthesis equation?"),
            ('short_answer', "In the photosynthesis equation 6CO\u2082 + 12H\u2082O \u2192 C\u2086H\u2081\u2082O\u2086 + 6O\u2082 + 6H\u2082O, what role does chlorophyll play?"),
            ('short_answer', "Why is water split during photosynthesis and what happens to the oxygen produced?"),
            ('one_word', "What gas is released as a by-product of photosynthesis?"),
            ('one_word', "Name the green pigment that absorbs sunlight during photosynthesis."),
            ('name_the', "Name the organelle where photosynthesis takes place."),
            ('fill_blank', "During photosynthesis, plants use _____ and _____ in the presence of sunlight and chlorophyll to produce glucose."),
            ('true_false', "True or False: Photosynthesis converts solar energy into chemical energy stored as glucose."),
            ('explain_concept', "Using the equation 6CO\u2082 + 12H\u2082O \u2192 C\u2086H\u2081\u2082O\u2086 + 6O\u2082 + 6H\u2082O, explain what happens during photosynthesis."),
            ('compare_concepts', "Compare the reactants and products of photosynthesis with those of aerobic respiration."),
        ],
        'answer_context': (
            "Photosynthesis is the process by which green plants produce food using sunlight. "
            "Balanced equation: 6CO\u2082 + 12H\u2082O \u2192 C\u2086H\u2081\u2082O\u2086 + 6O\u2082 + 6H\u2082O. "
            "Reactants: 6 molecules of carbon dioxide (CO\u2082) and 12 molecules of water (H\u2082O). "
            "Products: 1 molecule of glucose (C\u2086H\u2081\u2082O\u2086), 6 molecules of oxygen (O\u2082), and 6 molecules of water (H\u2082O). "
            "Conditions: In the presence of chlorophyll and sunlight. Location: Chloroplast. "
            "Chlorophyll absorbs sunlight energy to drive the conversion of CO\u2082 and H\u2082O into glucose. "
            "Oxygen is released as a by-product. "
            "This process is the foundation of all food chains as it converts solar energy into chemical energy."
        ),
    },
    {
        'name': 'Aerobic Respiration',
        'trigger': re.compile(
            r'\b(aerobic respiration|breakdown of glucose.{0,30}oxygen|'
            r'glucose.{0,30}mitochondria|ATP.{0,30}respiration|'
            r'glucose.{0,30}carbon dioxide.{0,30}water)\b', re.I),
        'context_chapters': {'Life Processes'},
        'questions': [
            ('reaction_equation', "Write the balanced chemical equation for aerobic respiration."),
            ('reaction_equation', "What are the products of aerobic respiration of glucose?"),
            ('short_answer', "In aerobic respiration (C\u2086H\u2081\u2082O\u2086 + 6O\u2082 \u2192 6CO\u2082 + 6H\u2082O + Energy), where does each stage take place in the cell?"),
            ('one_word', "Name the energy currency molecule produced during aerobic respiration."),
            ('fill_blank', "In aerobic respiration, glucose reacts with _____ to produce CO\u2082, water, and energy."),
            ('true_false', "True or False: Aerobic respiration produces more energy than anaerobic respiration."),
            ('explain_concept', "Explain the process of aerobic respiration using the equation C\u2086H\u2081\u2082O\u2086 + 6O\u2082 \u2192 6CO\u2082 + 6H\u2082O + Energy (ATP)."),
            ('compare_concepts', "Compare aerobic and anaerobic respiration in terms of oxygen requirement, end products, and energy released."),
        ],
        'answer_context': (
            "Aerobic respiration breaks down glucose using oxygen to release energy. "
            "Balanced equation: C\u2086H\u2081\u2082O\u2086 + 6O\u2082 \u2192 6CO\u2082 + 6H\u2082O + Energy (ATP). "
            "Reactants: 1 molecule of glucose and 6 molecules of oxygen. "
            "Products: 6 molecules of CO\u2082, 6 molecules of water, and approximately 38 ATP molecules. "
            "Stage 1 - Glycolysis (cytoplasm): Glucose (6C) is split into 2 pyruvate molecules (3C each), producing a small amount of ATP. "
            "Stage 2 - Krebs cycle and Electron Transport Chain (mitochondria): Pyruvate is completely oxidised to CO\u2082 and H\u2082O, releasing a large amount of ATP. "
            "Aerobic respiration is more efficient than anaerobic respiration (38 ATP vs 2 ATP)."
        ),
    },
    {
        'name': 'Anaerobic Respiration in Yeast',
        'trigger': re.compile(
            r'\b(fermentation|yeast.{0,30}anaerobic|anaerobic.{0,30}yeast|'
            r'ethanol.{0,30}carbon dioxide|pyruvate.{0,30}ethanol)\b', re.I),
        'context_chapters': {'Life Processes'},
        'questions': [
            ('reaction_equation', "Write the chemical equation for anaerobic respiration (fermentation) in yeast."),
            ('short_answer', "What products are formed when yeast respires anaerobically?"),
            ('one_word', "What gas is released during fermentation by yeast?"),
            ('fill_blank', "During fermentation, yeast converts glucose into _____ and _____ without oxygen."),
            ('true_false', "True or False: Yeast can respire both aerobically and anaerobically."),
            ('compare_concepts', "Compare fermentation in yeast with anaerobic respiration in human muscle cells in terms of products and energy released."),
        ],
        'answer_context': (
            "Anaerobic respiration in yeast (Fermentation): "
            "Equation: C\u2086H\u2081\u2082O\u2086 \u2192 2C\u2082H\u2085OH + 2CO\u2082 + Energy (2 ATP). "
            "In the absence of oxygen, yeast converts glucose \u2192 pyruvate (glycolysis) \u2192 ethanol + CO\u2082. "
            "Products: Ethanol (C\u2082H\u2085OH) and carbon dioxide (CO\u2082). "
            "Energy released: Only 2 ATP molecules (much less than aerobic respiration's 38 ATP). "
            "This process is called fermentation and is used industrially for making bread, beer, and wine. "
            "Compare with anaerobic respiration in muscle cells: glucose \u2192 lactic acid (not ethanol + CO\u2082)."
        ),
    },
    {
        'name': 'Anaerobic Respiration in Muscle Cells',
        'trigger': re.compile(
            r'\b(lactic acid|muscle.{0,30}cramp|anaerobic.{0,30}muscle|'
            r'sudden activity.{0,30}lactic|pyruvate.{0,30}lactic)\b', re.I),
        'context_chapters': {'Life Processes'},
        'questions': [
            ('reaction_equation', "Write the equation for anaerobic respiration in human muscle cells."),
            ('short_answer', "Why do our muscles cramp during vigorous exercise? Which chemical is responsible?"),
            ('one_word', "Name the acid that builds up in muscles during sudden intense activity."),
            ('true_false', "True or False: Lactic acid fermentation in muscles requires oxygen."),
            ('explain_concept', "Explain how anaerobic respiration in muscle cells (C\u2086H\u2081\u2082O\u2086 \u2192 2C\u2083H\u2086O\u2083 + Energy) leads to muscle cramps."),
        ],
        'answer_context': (
            "Anaerobic respiration in muscle cells: "
            "Equation: C\u2086H\u2081\u2082O\u2086 \u2192 2C\u2083H\u2086O\u2083 + Energy (2 ATP). "
            "During sudden intense activity (like sprinting), oxygen supply to muscles is insufficient. "
            "Pathway: Glucose \u2192 Pyruvate (glycolysis) \u2192 Lactic acid (C\u2083H\u2086O\u2083). "
            "The build-up of lactic acid in muscles causes pain and cramps. "
            "Unlike yeast fermentation, muscle cells produce lactic acid, not ethanol and CO\u2082. "
            "Only 2 ATP molecules are produced (vs. 38 ATP in aerobic respiration). "
            "After exercise, when oxygen supply is restored, lactic acid is converted back to pyruvate and broken down aerobically."
        ),
    },
    {
        'name': '10% Energy Flow Rule',
        'trigger': re.compile(
            r'\b(10%.{0,30}(energy|trophic)|trophic.{0,30}10%|energy.{0,30}food chain|'
            r'energy loss.{0,30}trophic|energy available at each)\b', re.I),
        'context_chapters': {'Our Environment'},
        'questions': [
            ('numerical_problem', "If a grassland has 10,000 J of energy at the producer level, calculate the energy available at each trophic level in the food chain: Grass \u2192 Insect \u2192 Frog \u2192 Snake. Show step-by-step calculation."),
            ('numerical_problem', "A food chain is: Grass \u2192 Rabbit \u2192 Snake \u2192 Eagle. If grass has 1,00,000 J of energy, calculate the energy available to each subsequent trophic level using the 10% rule."),
            ('short_answer', "State the 10% law of energy transfer in food chains. Why is energy lost at each trophic level?"),
            ('explain_concept', "Explain why food chains rarely have more than four trophic levels, using the 10% energy transfer rule."),
            ('one_word', "What percentage of energy is transferred from one trophic level to the next?"),
            ('fill_blank', "Only _____% of the energy consumed at one trophic level is available to the next trophic level."),
            ('true_false', "True or False: Energy transfer between trophic levels is 100% efficient."),
        ],
        'answer_context': (
            "The 10% Energy Rule (Lindeman's Law of Energy Transfer): "
            "Only 10% of the energy available at one trophic level is transferred to the next. "
            "The remaining 90% is lost as heat, used in metabolic processes, digestion, and movement. "
            "Worked example - Food chain: Grass \u2192 Insect \u2192 Frog \u2192 Snake \u2192 Eagle. "
            "Step 1 (Given): Energy at Producer (Grass) = 10,000 J. "
            "Step 2 (Formula): Energy at next level = 10% of previous level. "
            "Step 3 (Calculation): "
            "Primary Consumer (Insect) = 10% of 10,000 = 1,000 J. "
            "Secondary Consumer (Frog) = 10% of 1,000 = 100 J. "
            "Tertiary Consumer (Snake) = 10% of 100 = 10 J. "
            "Quaternary Consumer (Eagle) = 10% of 10 = 1 J. "
            "Final Answer: Energy decreases 10-fold at each level. "
            "This is why food chains rarely exceed 4 trophic levels - too little energy remains beyond that."
        ),
    },
    {
        'name': 'Magnification Formula',
        'trigger': re.compile(
            r'\b(magnification|estimated size of.{0,20}cell|microscope.{0,30}formula|'
            r'diameter.{0,30}(µm|micrometre)|observed size|actual size|field of view)\b', re.I),
        'context_chapters': {'Cell: The Building Block of Life'},
        'questions': [
            ('formula_application', "Using the formula: Estimated cell size = Diameter of field of view / Number of cells, calculate the size of an onion cell if the field diameter is 5000 µm and 25 cells are visible across the diameter. Show step-by-step working."),
            ('numerical_problem', "A microscope shows 40 cells across a 4000 µm diameter field of view. What is the estimated size of each cell? Show your working step by step."),
            ('one_word', "What formula is used to calculate the estimated size of a cell under a microscope?"),
            ('fill_blank', "Estimated size of a cell = _____ / Number of cells along the diameter."),
            ('short_answer', "How is the actual size of a cell estimated using a light microscope? What unit is commonly used?"),
        ],
        'answer_context': (
            "Cell Size Estimation using Microscopy. "
            "Formula: Estimated size of cell = Diameter of the visible field (in µm) / Number of cells along the diameter. "
            "Unit conversion: 1 millimetre (mm) = 1000 micrometres (µm). "
            "Worked Example 1: Field diameter = 5000 µm, Number of cells = 25. "
            "Step 1 (Given): Field diameter = 5000 µm, cells along diameter = 25. "
            "Step 2 (Formula): Estimated size = Field diameter \u00f7 Number of cells. "
            "Step 3 (Calculation): Estimated size = 5000 \u00f7 25 = 200 µm. "
            "Final Answer: Each onion cell is approximately 200 µm in size. "
            "Worked Example 2: Field diameter = 4000 µm, cells = 40. "
            "Estimated size = 4000 \u00f7 40 = 100 µm per cell."
        ),
    },
    {
        'name': 'Mendelian Monohybrid Cross',
        'trigger': re.compile(
            r'\b(mendel|monohybrid|F1 progeny|F2 progeny|dominant.{0,20}recessive|'
            r'tall.{0,30}short.{0,30}plant|3\s*:\s*1 ratio|pea plant.{0,30}cross)\b', re.I),
        'context_chapters': {'Heredity'},
        'questions': [
            ('numerical_problem', "In a Mendelian monohybrid cross, a tall pea plant (TT) is crossed with a short plant (tt). Show the F1 and F2 generations and state the phenotype ratio."),
            ('reaction_equation', "Show the cross TT (tall) x tt (short) and give the genotype and phenotype ratios of F1 and F2 generations."),
            ('short_answer', "Mendel crossed tall (TT) and short (tt) pea plants. All F1 plants were tall. Why? What does this tell us about dominance?"),
            ('one_word', "What is the phenotypic ratio in the F2 generation of a Mendelian monohybrid cross?"),
            ('fill_blank', "When TT x tt are crossed, F1 plants are all _____ (phenotype) with genotype _____."),
            ('true_false', "True or False: In a monohybrid cross, F2 generation shows a 3:1 phenotype ratio."),
            ('explain_concept', "Explain Mendel's Law of Dominance using the cross between tall (TT) and short (tt) pea plants."),
        ],
        'answer_context': (
            "Mendelian Monohybrid Cross: TT (tall) x tt (short). "
            "Step 1 (Parental generation P): TT (tall) x tt (short). "
            "Gametes from TT: T and T. Gametes from tt: t and t. "
            "Step 2 (F1 generation - Punnett square): "
            "All offspring are Tt (tall). T is dominant over t. "
            "F1 genotype: 100% Tt. F1 phenotype: 100% Tall. "
            "Step 3 (F1 x F1 to get F2): Tt x Tt. "
            "Punnett square F2: TT : Tt : Tt : tt = 1 TT : 2 Tt : 1 tt. "
            "F2 genotype ratio: 1 TT : 2 Tt : 1 tt. "
            "F2 phenotype ratio: 3 Tall (TT + Tt) : 1 Short (tt) = 3:1. "
            "Conclusion: Mendel's Law of Dominance - in the Tt genotype, T (tall, dominant) completely masks the expression of t (short, recessive)."
        ),
    },
    {
        'name': 'Mendelian Dihybrid Cross',
        'trigger': re.compile(
            r'\b(dihybrid|9\s*:\s*3\s*:\s*3\s*:\s*1|RrYy|RRYY|'
            r'round.{0,15}wrinkled.{0,15}yellow|independent assortment|'
            r'two.{0,20}characteristics.{0,20}bred)\b', re.I),
        'context_chapters': {'Heredity'},
        'questions': [
            ('numerical_problem', "In a Mendelian dihybrid cross (RRYY x rryy), what are the F1 and F2 genotypes and phenotype ratios? Explain using Mendel's Law of Independent Assortment."),
            ('reaction_equation', "Show the F1 and F2 outcomes of crossing RRYY (round yellow seeds) with rryy (wrinkled green seeds)."),
            ('one_word', "What is the F2 phenotype ratio in a Mendelian dihybrid cross?"),
            ('short_answer', "Why did Mendel observe new combinations of traits (like round green and wrinkled yellow) in the F2 generation of dihybrid crosses?"),
            ('explain_concept', "Explain Mendel's Law of Independent Assortment using the dihybrid cross between round yellow (RRYY) and wrinkled green (rryy) pea plants."),
        ],
        'answer_context': (
            "Mendelian Dihybrid Cross: RRYY (Round Yellow) x rryy (Wrinkled Green). "
            "Step 1 (Parental P): RRYY x rryy. R = Round (dominant), r = Wrinkled (recessive), Y = Yellow (dominant), y = Green (recessive). "
            "Step 2 (F1 generation): All offspring are RrYy (Round Yellow). "
            "Both dominant traits (R and Y) are expressed in F1. "
            "Step 3 (F1 x F1: RrYy x RrYy to get F2). "
            "F2 genotype combinations: 16 possible outcomes. "
            "F2 phenotype ratio: 9 Round Yellow : 3 Round Green : 3 Wrinkled Yellow : 1 Wrinkled Green = 9:3:3:1. "
            "New combinations: Round Green and Wrinkled Yellow are new phenotypes not seen in parents. "
            "Conclusion: Mendel's Law of Independent Assortment - genes for seed shape (R/r) and seed colour (Y/y) are on different chromosomes and are inherited independently of each other."
        ),
    },
]


def get_reaction_questions_for_chapter(chapter_name, passage_text, seen=None):
    """
    Return (intent, q_en, answer_en) tuples for biological reactions
    relevant to the current chapter and passage.
    """
    if seen is None:
        seen = set()
    results = []
    for rxn in BIOLOGICAL_REACTIONS:
        relevant = (
            not rxn['context_chapters']
            or chapter_name in rxn['context_chapters']
            or any(c in chapter_name for c in rxn['context_chapters'])
        )
        if not relevant: continue
        if not rxn['trigger'].search(passage_text): continue
        for intent, q_en in rxn['questions']:
            if q_en.lower() in seen: continue
            seen.add(q_en.lower())
            results.append((intent, q_en, rxn['answer_context']))
    return results
