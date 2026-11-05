"""
EduHinglish — Class 10 Social Science Dataset Generator (Large Scale)
======================================================================
Author  : Antigravity
Module  : M1 — Dataset Creation (Class 10 SS)
Purpose : Generate a large ss10.json from NCERT cleaned text for all 22
          chapters of Class 10 Social Science.

          For every extracted NCERT sentence we generate multiple Q&A
          pairs across all intent types — matching ss9.json scale.

          Intent types:
              true_false, explain_concept, definition, short_answer,
              application_reallife, analytical, compare_concepts,
              give_example, list_enumerate, one_word, fill_blank

Usage:
    python scripts/generate_ss10_dataset.py
    python scripts/generate_ss10_dataset.py --preview
"""

from __future__ import annotations
import json, re, argparse, random
from pathlib import Path
from collections import Counter

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH   = PROJECT_ROOT / "data" / "ss10.json"
random.seed(42)

# ── Chapter metadata ──────────────────────────────────────────────────────────
CHAPTERS = [
    {"folder":"ss10_his_ch01","sub_subject":"History","chapter":"The Rise of Nationalism in Europe","chapter_num":"ch01","id_prefix":"c10_ss_his_ch01"},
    {"folder":"ss10_his_ch02","sub_subject":"History","chapter":"Nationalism in India","chapter_num":"ch02","id_prefix":"c10_ss_his_ch02"},
    {"folder":"ss10_his_ch03","sub_subject":"History","chapter":"The Making of a Global World","chapter_num":"ch03","id_prefix":"c10_ss_his_ch03"},
    {"folder":"ss10_his_ch04","sub_subject":"History","chapter":"The Age of Industrialisation","chapter_num":"ch04","id_prefix":"c10_ss_his_ch04"},
    {"folder":"ss10_his_ch05","sub_subject":"History","chapter":"Print Culture and the Modern World","chapter_num":"ch05","id_prefix":"c10_ss_his_ch05"},
    {"folder":"ss10_geo_ch01","sub_subject":"Geography","chapter":"Resources and Development","chapter_num":"ch01","id_prefix":"c10_ss_geo_ch01"},
    {"folder":"ss10_geo_ch02","sub_subject":"Geography","chapter":"Forest and Wildlife Resources","chapter_num":"ch02","id_prefix":"c10_ss_geo_ch02"},
    {"folder":"ss10_geo_ch03","sub_subject":"Geography","chapter":"Water Resources","chapter_num":"ch03","id_prefix":"c10_ss_geo_ch03"},
    {"folder":"ss10_geo_ch04","sub_subject":"Geography","chapter":"Agriculture","chapter_num":"ch04","id_prefix":"c10_ss_geo_ch04"},
    {"folder":"ss10_geo_ch05","sub_subject":"Geography","chapter":"Minerals and Energy Resources","chapter_num":"ch05","id_prefix":"c10_ss_geo_ch05"},
    {"folder":"ss10_geo_ch06","sub_subject":"Geography","chapter":"Manufacturing Industries","chapter_num":"ch06","id_prefix":"c10_ss_geo_ch06"},
    {"folder":"ss10_geo_ch07","sub_subject":"Geography","chapter":"Lifelines of National Economy","chapter_num":"ch07","id_prefix":"c10_ss_geo_ch07"},
    {"folder":"ss10_eco_ch01","sub_subject":"Economics","chapter":"Development","chapter_num":"ch01","id_prefix":"c10_ss_eco_ch01"},
    {"folder":"ss10_eco_ch02","sub_subject":"Economics","chapter":"Sectors of the Indian Economy","chapter_num":"ch02","id_prefix":"c10_ss_eco_ch02"},
    {"folder":"ss10_eco_ch03","sub_subject":"Economics","chapter":"Money and Credit","chapter_num":"ch03","id_prefix":"c10_ss_eco_ch03"},
    {"folder":"ss10_eco_ch04","sub_subject":"Economics","chapter":"Globalisation and the Indian Economy","chapter_num":"ch04","id_prefix":"c10_ss_eco_ch04"},
    {"folder":"ss10_eco_ch05","sub_subject":"Economics","chapter":"Consumer Rights","chapter_num":"ch05","id_prefix":"c10_ss_eco_ch05"},
    {"folder":"ss10_civics_ch01","sub_subject":"Civics","chapter":"Power-sharing","chapter_num":"ch01","id_prefix":"c10_ss_civics_ch01"},
    {"folder":"ss10_civics_ch02","sub_subject":"Civics","chapter":"Federalism","chapter_num":"ch02","id_prefix":"c10_ss_civics_ch02"},
    {"folder":"ss10_civics_ch03","sub_subject":"Civics","chapter":"Gender, Religion and Caste","chapter_num":"ch03","id_prefix":"c10_ss_civics_ch03"},
    {"folder":"ss10_civics_ch04","sub_subject":"Civics","chapter":"Political Parties","chapter_num":"ch04","id_prefix":"c10_ss_civics_ch04"},
    {"folder":"ss10_civics_ch05","sub_subject":"Civics","chapter":"Outcomes of Democracy","chapter_num":"ch05","id_prefix":"c10_ss_civics_ch05"},
]

# ── Topic keywords per chapter ────────────────────────────────────────────────
CHAPTER_TOPICS = {
    "c10_ss_his_ch01":[("Nationalism in Europe",["nation","nationalism","nation-state","liberty","Sorrieu","Renan","utopian"]),("French Revolution",["French Revolution","France","tricolour","Jacobin","Napoleon","Civil Code"]),("Romanticism",["Romanticism","romantic","folk","culture","vernacular"]),("Unification of Germany",["Germany","German","Bismarck","Prussia","Zollverein"]),("Unification of Italy",["Italy","Italian","Cavour","Garibaldi","Mazzini","Sardinia"]),("Nationalism and Imperialism",["imperialism","Balkan","Ottoman","Habsburg","Slavic","empire"])],
    "c10_ss_his_ch02":[("Non-Cooperation Movement",["Non-Cooperation","khadi","boycott","Khilafat","Ali"]),("Civil Disobedience Movement",["Civil Disobedience","Dandi","salt","Salt March","Gandhi-Irwin"]),("Quit India Movement",["Quit India","August 1942","do or die"]),("Indian National Congress",["Congress","INC","Lahore","Purna Swaraj","Nehru"]),("Peasant and Worker Movements",["peasant","tribal","worker","plantation","Gudem","Awadh"]),("Nationalism in India",["nationalism","India","freedom","British","swadeshi","swaraj"])],
    "c10_ss_his_ch03":[("Silk Routes",["Silk Route","silk","spice","cowrie","pre-modern"]),("19th Century Globalisation",["indentured","plantation","migration","trade route"]),("The Great Depression",["Great Depression","depression","unemployment","Wall Street","1929"]),("Bretton Woods",["Bretton Woods","IMF","World Bank","Marshall Plan","GATT"]),("Post-War Recovery",["post-war","recovery","Marshall","reconstruction"]),("Globalisation History",["globalisation","global","trade","exchange"])],
    "c10_ss_his_ch04":[("Proto-industrialisation",["proto-industrialisation","putting-out","stapler","merchant","countryside"]),("Industrial Revolution",["Industrial Revolution","steam engine","factory","mechanisation"]),("Manchester and Bombay",["Manchester","Bombay","cotton","textile mill","Lancashire"]),("Indian Textiles",["Indian textile","weaver","handloom","calico","muslin"]),("Factory Workers",["worker","factory","labour","wages","child labour"]),("Market and Advertising",["advertisement","brand","label","market"])],
    "c10_ss_his_ch05":[("Gutenberg Press",["Gutenberg","printing press","movable type","1448"]),("Print Revolution",["print revolution","printing","manuscript","book"]),("Reformation and Print",["Reformation","Luther","Protestant","Bible","Church"]),("Print in India",["India","Goa","Portuguese","missionary","vernacular press"]),("Newspapers",["newspaper","press","journal","magazine"]),("Books and Knowledge",["knowledge","literacy","education","Rammohun","novel"])],
    "c10_ss_geo_ch01":[("Types of Resources",["resource","renewable","non-renewable","biotic","abiotic"]),("Land Resources",["land","land use","cultivable","fallow"]),("Soil Types",["soil","alluvial","black","laterite","red soil","arid"]),("Soil Conservation",["soil erosion","soil conservation","contour","terrace","shelter belt"]),("Resource Planning",["resource planning","sustainable","conservation"])],
    "c10_ss_geo_ch02":[("Forest Resources",["forest","woodland","tree","timber","vegetation"]),("Wildlife Conservation",["wildlife","animal","species","sanctuary","reserve"]),("Biodiversity",["biodiversity","species","extinction","endemic","habitat"]),("Reserved Forests",["reserved forest","protected forest"]),("Project Tiger",["tiger","Project Tiger","poaching"]),("Community Forests",["community","Chipko","tribal","forest rights"])],
    "c10_ss_geo_ch03":[("Water Scarcity",["water scarcity","shortage","availability","fresh water"]),("Dams",["dam","multipurpose","Bhakra","Hirakud","reservoir"]),("Rainwater Harvesting",["rainwater harvesting","kul","johad","rooftop","palar"]),("Water Conservation",["conservation","watershed","groundwater","recharge"]),("Water Pollution",["pollution","effluent","sewage","contamination"])],
    "c10_ss_geo_ch04":[("Types of Farming",["subsistence","commercial","intensive","extensive","plantation"]),("Crop Seasons",["kharif","rabi","zaid","crop season"]),("Food Crops",["rice","wheat","maize","pulses","food grain"]),("Commercial Crops",["cotton","jute","sugarcane","tea","coffee","rubber"]),("Agricultural Development",["Green Revolution","irrigation","fertiliser","seed","yield"])],
    "c10_ss_geo_ch05":[("Types of Minerals",["mineral","ore","metallic","non-metallic","ferrous"]),("Metallic Minerals",["iron ore","bauxite","copper","manganese","gold","mica"]),("Energy Resources",["coal","petroleum","natural gas","fossil fuel"]),("Conventional Energy",["thermal","hydro","nuclear","conventional"]),("Non-Conventional Energy",["solar","wind","biogas","tidal","non-conventional"])],
    "c10_ss_geo_ch06":[("Types of Industries",["industry","primary","secondary","public sector","private sector"]),("Cotton Textile Industry",["cotton","textile","spinning","weaving","Mumbai","Ahmedabad"]),("Iron and Steel Industry",["iron","steel","Jamshedpur","Bhilai","Bokaro"]),("Chemical Industry",["chemical","fertiliser","petrochemical","pharmaceutical"]),("Industrial Pollution",["pollution","effluent","particulate","waste"]),("Special Economic Zones",["SEZ","Special Economic Zone","export"])],
    "c10_ss_geo_ch07":[("Transport",["transport","transportation","network","mobility"]),("Roadways",["road","highway","National Highway","expressway"]),("Railways",["railway","rail","train","Indian Railways"]),("Waterways",["waterway","river","canal","coastal","shipping","port"]),("Airways",["airway","airport","aviation"]),("Communication",["communication","internet","satellite","radio","television"]),("Trade",["trade","export","import","commerce"])],
    "c10_ss_eco_ch01":[("Development Goals",["development goal","average income","compare","criterion"]),("Income and Development",["income","per capita","national income","GDP"]),("Human Development Index",["HDI","Human Development Index","life expectancy","literacy"]),("Sustainability",["sustainable","sustainability","future generation"]),("National Development",["national development","economic development","progress"])],
    "c10_ss_eco_ch02":[("Primary Sector",["primary sector","agriculture","fishing","mining","forestry"]),("Secondary Sector",["secondary sector","manufacturing","factory","construction"]),("Tertiary Sector",["tertiary","service sector","banking","communication"]),("Organised Sector",["organised sector","formal sector","regular worker"]),("Unorganised Sector",["unorganised sector","informal sector","casual","low wage"]),("Employment",["employment","disguised unemployment","MNREGA","labour"])],
    "c10_ss_eco_ch03":[("Barter System",["barter","double coincidence"]),("Functions of Money",["money","medium of exchange","store of value","unit of account"]),("Formal Credit",["bank","cooperative","formal credit","collateral","loan"]),("Informal Credit",["moneylender","informal credit","landlord","high interest"]),("Self-Help Groups",["self-help group","SHG","microfinance","rural credit"]),("Credit",["credit","debt","repayment","interest rate","borrowing"])],
    "c10_ss_eco_ch04":[("Globalisation",["globalisation","integration","interdependence","global market"]),("Multinational Companies",["multinational","MNC","foreign company","subsidiary"]),("Trade Liberalisation",["liberalisation","tariff","import","export","trade barrier"]),("WTO",["WTO","World Trade Organization","free trade"]),("Impact of Globalisation",["impact","benefit","disadvantage","small producer"])],
    "c10_ss_eco_ch05":[("Consumer Rights",["consumer right","right to safety","right to information"]),("Consumer Protection",["consumer protection","exploitation","adulteration","unfair trade"]),("COPRA",["COPRA","Consumer Protection Act","consumer court","redressal"]),("RTI",["RTI","Right to Information","transparency","accountability"]),("Consumer Courts",["consumer court","district forum","state commission"]),("Quality Marks",["ISI","Hallmark","Agmark","quality mark","certification"])],
    "c10_ss_civics_ch01":[("Power-sharing",["power-sharing","share power","distribution of power"]),("Belgium",["Belgium","Belgian","Flemish","Wallonia","Brussels","Dutch"]),("Sri Lanka",["Sri Lanka","Sinhala","Tamil","Lanka"]),("Majoritarianism",["majoritarianism","majority","minority","dominance"]),("Ethnic Conflict",["civil war","ethnic","conflict","tension","community"]),("Democracy and Power",["democracy","government","organ","legislature","judiciary"])],
    "c10_ss_civics_ch02":[("Federalism",["federalism","federal","federal system","constitution"]),("Centre and State",["Union List","State List","Concurrent List"]),("Linguistic States",["linguistic","language","state reorganisation","Andhra"]),("Decentralisation",["decentralisation","local government","panchayat","municipality"]),("Local Self Government",["panchayati raj","gram sabha","ward","municipal"])],
    "c10_ss_civics_ch03":[("Gender and Politics",["gender","women","female","reservation","patriarchal"]),("Religion in Politics",["religion","communalism","secular","communal"]),("Caste and Politics",["caste","casteism","SC","ST","Dalit"]),("Social Division",["social division","diversity","inequality","discrimination"]),("Communalism",["communalism","communal","riot","partition"])],
    "c10_ss_civics_ch04":[("Political Parties",["political party","party","contest","election","candidate"]),("Functions of Parties",["function","form government","opposition","mobilise","policy"]),("National Parties",["national party","BJP","Congress","INC","BSP"]),("State Parties",["state party","regional party","TDP","DMK"]),("Challenges to Parties",["challenge","dynastic","money power","criminal","reform"])],
    "c10_ss_civics_ch05":[("Democracy and Accountability",["accountable","accountability","transparent","legitimate"]),("Economic Outcomes",["economic","poverty","inequality","distribution"]),("Dignity and Freedom",["dignity","freedom","liberty","rights","equality"]),("Responsive Government",["responsive","citizen","expectation","demand"]),("Democracy vs Dictatorship",["dictatorship","authoritarian","autocracy","non-democratic"])],
}

# ── OCR noise filter ──────────────────────────────────────────────────────────
_NOISE = re.compile(
    r'(?:Reprint\s+\d{4}[-–]\d{2,4}|\.indd\s+\d+|\d{2}[-–]\d{2}[-–]\d{4}|https?://\S+|www\.\S+|\bFor more details.*)',
    re.MULTILINE | re.IGNORECASE
)

# Regex: a "word" that has 4+ consecutive consonants (typical of reversed English)
_CONSONANT_RUN = re.compile(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}')

# Known NCERT margin artifacts (case-insensitive) — add more as discovered
_MARGIN_ARTIFACTS = re.compile(
    r'^\s*('
    r'ni\s+msilanoitaN|msilanoitaN|eporuE|fo\s+esir|'
    r'aera\s+egral|aI\s+ni|dnalsI|naimoR|'
    r'aidnI\s+ni|naciremA|hcnerF|'
    r'gnirahs\S*|rewoP|noigileR|semoctuO|'
    r'[A-Za-z]{2,}\s*[A-Za-z]{1,2}\s*[A-Za-z]{1,2}\s*[A-Za-z]{1,2}\s*$'  # spaced-char vertical text
    r')\s*$',
    re.IGNORECASE
)

# Detect reversed word clusters like "eht dna aidnI" (reversed "India and the")
# Pattern: 2–3 consecutive short all-lowercase words followed by a mixed-case reversed word
_REVERSED_CLUSTER = re.compile(
    r'\b(?:[a-z]{2,4}\s+){1,3}[a-z]+[A-Z][a-z]+\b'   # e.g. "eht dna aidnI"
)


def _is_reversed_text(s: str) -> bool:
    """Heuristic: if the reversed version of a string has significantly more
    vowel–consonant alternation (i.e., looks more like real English), the
    original is likely reversed OCR output."""
    words = s.split()
    if len(words) > 8:
        return False  # Long sentences are unlikely to be entirely reversed
    rev_words = [w[::-1] for w in reversed(words)]
    rev = " ".join(rev_words)

    def _vowel_ratio(text: str) -> float:
        alpha = [c.lower() for c in text if c.isalpha()]
        if not alpha:
            return 0.0
        return sum(1 for c in alpha if c in "aeiou") / len(alpha)

    orig_vr = _vowel_ratio(s)
    rev_vr  = _vowel_ratio(rev)
    # If reversed has a much more "normal" vowel ratio (25-45%) vs. the original
    normal_range = 0.22 <= rev_vr <= 0.50
    original_bad = orig_vr < 0.18 or _CONSONANT_RUN.search(s) is not None
    return normal_range and original_bad


def _is_noise(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 10:
        return True
    # ALL-CAPS short headers
    if s.isupper() and len(s) < 30:
        return True
    # Known NCERT vertical-margin artifact patterns
    if _MARGIN_ARTIFACTS.match(s):
        return True
    # Spaced-out single chars (vertical text extracted character by character)
    tokens = s.split()
    if len(tokens) >= 2 and all(len(t) <= 2 for t in tokens) and len(s) <= 20:
        return True
    # Consonant cluster run — garbled OCR
    if _CONSONANT_RUN.search(s):
        return True
    # Reversed text heuristic
    if _is_reversed_text(s):
        return True
    # Too many non-ASCII for a short line (likely symbol/image text)
    non_ascii = sum(1 for c in s if ord(c) > 127)
    if non_ascii > len(s) * 0.3:
        return True
    return False


def _clean_sentence(s: str) -> str:
    """Strip leading and embedded OCR-garbage tokens from an otherwise valid sentence."""
    words = s.split()
    keep_from = 0
    for i, w in enumerate(words):
        stripped = w.strip(".,;:?!\"'()-")
        # A "garbage" token: has a consonant run OR is a known reversed fragment
        if _CONSONANT_RUN.search(stripped) or _is_reversed_text(stripped):
            keep_from = i + 1
            continue
        # Once we hit a normal-looking capitalised word, stop stripping
        if stripped and stripped[0].isupper():
            break
    cleaned = " ".join(words[keep_from:]).strip()
    # Fall back to original if we stripped too much
    return cleaned if len(cleaned) >= 40 else s


def _has_embedded_corruption(s: str) -> bool:
    """Return True if a sentence contains embedded reversed-word clusters."""
    if _CONSONANT_RUN.search(s):
        return True
    if _REVERSED_CLUSTER.search(s):
        return True
    return False


def extract_sentences(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = [_NOISE.sub(" ", l).strip() for l in text.split("\n") if not _is_noise(l)]
    clean = re.sub(r"\s+", " ", " ".join(lines))
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", clean)
    seen, out = set(), []
    for s in raw:
        s = _clean_sentence(s.strip())
        if len(s) < 60 or len(s) > 600:
            continue
        alpha = sum(c.isalpha() or c == " " for c in s) / len(s)
        if alpha < 0.6:
            continue
        if len(s.split()) < 8:
            continue
        # Reject sentences that start with a lowercase word (fragment / corruption)
        first_char = s[0] if s else ''
        if first_char.islower():
            continue
        # Final gate: reject if sentence still starts with or contains garbled tokens
        first_word = s.split()[0].strip(".,;:?!\"'()")
        if _CONSONANT_RUN.search(first_word) or _is_reversed_text(first_word):
            continue
        # Reject if embedded reversed-cluster found in first 60 chars
        if _has_embedded_corruption(s[:60]):
            continue
        fp = s[:80].lower()
        if fp in seen:
            continue
        seen.add(fp)
        out.append(s)
    return out

def assign_topic(sentence: str, prefix: str) -> str:
    topics = CHAPTER_TOPICS.get(prefix, [])
    if not topics: return "General"
    sl = sentence.lower()
    best, best_n = topics[0][0], 0
    for name, kws in topics:
        n = sum(1 for k in kws if k.lower() in sl)
        if n > best_n: best, best_n = name, n
    return best

# ── Hinglish helpers ──────────────────────────────────────────────────────────
_HI_SUFFIXES = [
    ", yeh NCERT mein clearly bataya gaya hai.",
    ", jo Social Science ki book mein explain kiya gaya hai.",
    ". Yeh ek important point hai.",
    ", isko achhe se yaad kar lo.",
    ", yeh tumhare exam ke liye important hai.",
    ", jo class 10 ka key concept hai.",
]
_HI_STARTERS = [
    "NCERT ke according,","Book mein likha hai ki","Yaad rakho ki",
    "Important point yeh hai ki","Textbook batata hai ki",
    "Class 10 SS mein,","Hum jaante hain ki","Dhyan do,",
]

def _hi_ans(core: str) -> str:
    core = core.rstrip(".")
    return f"{random.choice(_HI_STARTERS)} {core}{random.choice(_HI_SUFFIXES)}"

def _hi_q_stmt(eng_q: str) -> str:
    forms = [
        lambda q: q + " — yeh batao.",
        lambda q: "Socho: " + q,
        lambda q: q.replace("What is","Kya hota hai").replace("What are","Kya hote hain")
                   .replace("How does","Kaise hota hai").replace("Why is","Kyun hai")
                   .replace("Why are","Kyun hote hain").replace("Who was","Kaun tha")
                   .replace("How did","Kaise hua").replace("Who is","Kaun hai"),
    ]
    return random.choice(forms)(eng_q)

# ── Entry builder ─────────────────────────────────────────────────────────────
def sentence_to_entries(sentence: str, ch: dict, topic: str, ctr_start: int) -> list[dict]:
    prefix   = ch["id_prefix"]
    chapter  = ch["chapter"]
    ch_num   = ch["chapter_num"]
    sub_subj = ch["sub_subject"]
    ctx      = sentence.strip()
    words    = ctx.split()

    entries = []
    ctr = ctr_start

    def add(intent, q_en, q_hi, a_en, a_hi):
        nonlocal ctr
        entries.append({
            "id":               f"{prefix}_{ctr:04d}",
            "class":            "10",
            "subject":          "Social Science",
            "sub_subject":      sub_subj,
            "chapter":          chapter,
            "chapter_num":      ch_num,
            "topic":            topic,
            "source":           "generated_trivial",
            "is_student_query": False,
            "intent":           intent,
            "question_english": q_en,
            "question_hinglish": q_hi,
            "answer_english":   a_en,
            "answer_hinglish":  a_hi,
            "ncert_context":    ctx,
            "messages": [
                {"role":"system","content":ctx},
                {"role":"user","content":q_en},
                {"role":"assistant","content":a_en},
            ],
            "messages_hinglish": [
                {"role":"system","content":ctx},
                {"role":"user","content":q_hi},
                {"role":"assistant","content":a_hi},
            ],
        })
        ctr += 1

    # 1. true_false
    add("true_false",
        f"True or False: {ctx}",
        f"{ctx} — Sahi hai ya galat?",
        f"True. {ctx}",
        _hi_ans(f"Yeh bilkul sahi hai. {ctx}"))

    # 2. short_answer
    q = f"What is meant by: '{ctx[:100]}'?" if any(w in ctx.lower() for w in ["is ","are ","was "]) else f"Briefly explain: {ctx[:100]}"
    add("short_answer", q, f"Chhoti explanation do: '{ctx[:80]}'",
        f"{ctx} This is a key point from the chapter on {chapter}.",
        _hi_ans(ctx))

    # 3. explain_concept
    add("explain_concept",
        f"Explain the concept: \"{ctx[:120]}\"",
        f"Is concept ko explain karo: \"{ctx[:80]}\"",
        f"{ctx} Understanding this concept is essential for the chapter '{chapter}'.",
        _hi_ans(f"{chapter} chapter mein yeh concept samjhna zaroori hai. {ctx}"))

    # 4. definition — only if sentence defines something
    DEF_PATS = [" is ", " are ", " refers to ", " means ", " defined as ", " known as "]
    for dp in DEF_PATS:
        if dp in ctx:
            term = ctx.split(dp)[0].strip().rstrip(",")
            if 1 <= len(term.split()) <= 6:
                add("definition",
                    f"Define: {term}",
                    f"Define karo: {term}",
                    ctx,
                    _hi_ans(ctx))
            break

    # 5. fill_blank
    STOP = {"the","a","an","is","are","was","were","in","on","at","by","of","to",
            "and","or","for","with","from","that","this","their","its","it","he",
            "she","they","which","been","be","have","has","had","will","would"}
    if len(words) >= 8:
        blank_idx = None
        for i in range(len(words)-1, max(0, len(words)-7), -1):
            w = words[i].strip(".,;:?!\"'()").lower()
            if w not in STOP and len(w) > 3:
                blank_idx = i; break
        if blank_idx is not None:
            bw = words[blank_idx]
            blanked = words[:]
            blanked[blank_idx] = "_______"
            bs = " ".join(blanked) + "."
            add("fill_blank",
                f"Fill in the blank: {bs}",
                f"Blank bharo: {bs}",
                f"The answer is '{bw.strip('.,;:?!')}'. {ctx}",
                _hi_ans(f"Blank mein '{bw.strip('.,;:?!')}' aayega. {ctx}"))

    # 6. application_reallife
    APPLIC_Q = [
        f"How does the statement '{ctx[:100]}' apply to real life?",
        f"Give a real-life example that demonstrates: '{ctx[:100]}'",
        f"Why is this relevant today: '{ctx[:100]}'?",
    ]
    APPLIC_HI = [
        f"Real life mein kaise apply hota hai: '{ctx[:80]}'?",
        f"Ek real-life example do: '{ctx[:80]}'",
        f"Aaj ke zamane mein kyun relevant hai: '{ctx[:80]}'?",
    ]
    add("application_reallife", random.choice(APPLIC_Q), random.choice(APPLIC_HI),
        f"{ctx} This concept has practical relevance in everyday {sub_subj}.",
        _hi_ans(f"Yeh concept aaj bhi relevant hai. {ctx}"))

    # 7. analytical
    ANALY_Q = [
        f"Critically analyse: '{ctx[:110]}'",
        f"What conclusions can be drawn from: '{ctx[:110]}'?",
        f"Analyse the significance of: '{ctx[:110]}'",
    ]
    ANALY_HI = [
        f"Critically analyse karo: '{ctx[:80]}'",
        f"Is statement se kya conclusions nikale ja sakte hain?",
        f"Iska significance analyse karo: '{ctx[:80]}'",
    ]
    add("analytical", random.choice(ANALY_Q), random.choice(ANALY_HI),
        f"{ctx} This highlights an important aspect of {chapter}.",
        _hi_ans(f"Analysis karte hue, {chapter} ke context mein yeh point bahut important hai. {ctx}"))

    # 8. give_example
    EG_Q  = [f"Give an example related to: '{ctx[:110]}'",
             f"What example supports: '{ctx[:110]}'?"]
    EG_HI = [f"Ek example do jo yeh point support kare: '{ctx[:80]}'",
             f"Koi example batao jisme yeh concept dikhta ho: '{ctx[:80]}'"]
    add("give_example", random.choice(EG_Q), random.choice(EG_HI),
        f"{ctx} This exemplifies the concepts studied in {chapter}.",
        _hi_ans(f"Yeh ek acha example hai jo {chapter} ke concepts dikhata hai. {ctx}"))

    # 9. list_enumerate — only if sentence contains enumerable items
    LIST_TRIGGERS = [", ","; "," such as "," including "," namely "," like "," are: "]
    if any(t in ctx for t in LIST_TRIGGERS):
        LIST_Q  = [f"List the key points in: '{ctx[:110]}'",
                   f"Enumerate the elements mentioned in: '{ctx[:110]}'"]
        LIST_HI = [f"List karo jo is statement mein bataya gaya hai: '{ctx[:80]}'",
                   f"Enumerate karo: '{ctx[:80]}'"]
        add("list_enumerate", random.choice(LIST_Q), random.choice(LIST_HI),
            f"{ctx} These are the key elements from {chapter}.",
            _hi_ans(f"Yeh important points hain jo {chapter} mein bataye gaye hain. {ctx}"))

    # 10. compare_concepts — only if contrast words present
    CONTRAST = ["while","whereas","unlike","however","but","on the other hand",
                "in contrast","although","difference","compared to","both","either"]
    if any(k in ctx.lower() for k in CONTRAST):
        CMP_Q  = [f"Compare the concepts in: '{ctx[:110]}'",
                  f"What contrast is described in: '{ctx[:110]}'?"]
        CMP_HI = [f"Compare karo jo is statement mein hai: '{ctx[:80]}'",
                  f"Is statement mein kya contrast hai: '{ctx[:80]}'?"]
        add("compare_concepts", random.choice(CMP_Q), random.choice(CMP_HI),
            f"{ctx} This highlights the comparison between key concepts in {chapter}.",
            _hi_ans(f"Yeh {chapter} ke important concepts ke beech comparison dikhata hai. {ctx}"))

    # 11. one_word — short sentences with definition pattern
    if len(words) <= 20:
        for dp in [" is ", " are ", " was "]:
            if dp in ctx:
                parts = ctx.split(dp, 1)
                term_words = parts[0].strip().rstrip(",").split()
                if 1 <= len(term_words) <= 4:
                    key = " ".join(term_words)
                    ans_short = " ".join(parts[1].strip().rstrip(".").split()[:6]).rstrip(".,;:")
                    add("one_word",
                        f"In one word or phrase, what is '{key}'?",
                        f"Ek word ya phrase mein batao, '{key}' kya hai?",
                        f"{ans_short.capitalize()}. ({ctx})",
                        _hi_ans(f"'{key}' ko ek phrase mein — {ans_short}. {ctx}"))
                break

    return entries

# ── Main ──────────────────────────────────────────────────────────────────────
def main(preview=False):
    all_entries = []
    print("\n" + "=" * 72)
    print("  EduHinglish — Class 10 Social Science Dataset Generator (Large Scale)")
    print("=" * 72)

    for ch in CHAPTERS:
        cleaned_path = PROCESSED_DIR / ch["folder"] / "cleaned_text.txt"
        if not cleaned_path.exists():
            print(f"  [MISSING] {ch['folder']}/cleaned_text.txt")
            continue

        sentences = extract_sentences(cleaned_path)
        chapter_entries = []
        ctr = 1
        for sent in sentences:
            topic = assign_topic(sent, ch["id_prefix"])
            new  = sentence_to_entries(sent, ch, topic, ctr)
            chapter_entries.extend(new)
            ctr += len(new)

        all_entries.extend(chapter_entries)
        tag = "PREVIEW" if preview else "OK"
        print(f"  [{tag}] {ch['id_prefix']:<32} {len(sentences):4d} sents → {len(chapter_entries):5d} entries  ({ch['chapter'][:40]})")

    print()
    print(f"  {'─'*70}")
    print(f"  Total entries : {len(all_entries):,}")
    print(f"  Chapters      : {len(CHAPTERS)}")
    intents = Counter(e["intent"] for e in all_entries)
    print(f"  Intent distribution:")
    for intent, cnt in sorted(intents.items(), key=lambda x: -x[1]):
        print(f"    {intent:<25} {cnt:6,}")
    print(f"  {'─'*70}")

    if not preview:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)
        size_mb = OUTPUT_PATH.stat().st_size / (1024*1024)
        print(f"\n  ✅ Written → {OUTPUT_PATH}  ({size_mb:.1f} MB)")
    else:
        print(f"\n  ⚡ Preview only — run without --preview to write {OUTPUT_PATH}")
    print("=" * 72 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    main(preview=args.preview)
