"""
EduHinglish — Class 10 Dataset Generator
========================================
Author  : Ashvatth
Module  : M1 — Dataset Creation
Purpose : Generate Class 10 chapter datasets with required counts.

This script generates datasets for these chapter codes (annotation_helper map):
  class10/ch06  — Life Processes
  class10/ch08  — How do Organisms Reproduce?
  class10/ch09  — Heredity and Evolution
  class10/ch15  — Our Environment
  class10/ch16  — Management of Natural Resources
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "biology" / "class10"

HINDI_WORDS = {
    "main", "hum", "tum", "woh", "yeh", "uska", "uski", "iska", "iski",
    "mera", "meri", "tera", "teri", "unka", "unki", "hamara", "tumhara",
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "hota", "hoti", "hote",
    "karta", "karti", "karte", "kiya", "karo", "karna", "karke", "hona",
    "raha", "rahi", "rahe", "gaya", "gayi", "aata", "aati", "jaata", "jaati",
    "deta", "deti", "lete", "bana", "bante", "samjhao", "batao", "dekho",
    "kehte", "kehta", "kehti", "kaha", "dijiye", "hokar", "jinmein", "inmein",
    "ka", "ki", "ke", "ko", "se", "mein", "par", "tak", "pe", "ne", "me",
    "aur", "ya", "lekin", "kyunki", "isliye", "jabki", "phir", "toh", "bhi",
    "hi", "sirf", "bas", "kya", "kaise", "kyun", "kahan", "kab", "kaun",
    "kitna", "bahut", "thoda", "zyada", "kam", "achha", "bada", "bade", "badi",
    "chhota", "naya", "nayi", "pehle", "baad", "andar", "bahar", "upar",
    "neeche", "yahan", "wahan", "abhi", "tab", "jab", "nahi", "nhi", "na",
    "mat", "ek", "do", "teen", "sabhi", "sab", "kuch", "koi", "wala", "wale",
    "wali", "jaise", "taraf", "beech", "kaam", "saath", "jo", "tarah", "matlab",
}

UNIVERSAL_WORDS = {"sir", "madam", "ok", "okay", "hello", "hi", "bye", "please", "thanks", "sorry"}

VALID_LABELS = {"HI", "EN", "NE", "UNIV", "MIX"}


def predict_label(word: str, position: int) -> str:
    clean = word.strip(".,?!;:\"'()")
    if not clean:
        return "EN"

    if any("\u0900" <= c <= "\u097F" for c in clean):
        return "HI"

    lower = clean.lower()
    if lower in UNIVERSAL_WORDS:
        return "UNIV"
    if lower in HINDI_WORDS:
        return "HI"
    if clean.isdigit():
        return "UNIV"
    if position > 0 and clean and clean[0].isupper():
        return "NE"
    return "EN"


def label_sentence(sentence: str) -> dict:
    tokens = sentence.split()
    labels = {}
    for idx, tok in enumerate(tokens):
        labels[tok] = predict_label(tok, idx)
    return labels


def build_entry(entry_id: str, original: str, hinglish: str, topic: str, chapter: str, klass: str,
                is_query: bool = False, intent: str | None = None,
                is_gec: bool = False, hinglish_with_error: str | None = None,
                error_description: dict | None = None) -> dict:
    entry = {
        "id": entry_id,
        "original_english": original,
        "hinglish_roman": hinglish,
        "word_level_labels": label_sentence(hinglish),
        "topic": topic,
        "chapter": chapter,
        "class": klass,
        "code_mixing_type": "intra-sentential",
    }
    if is_query:
        entry["is_student_query"] = True
        entry["intent"] = intent or "explain_concept"
    if is_gec:
        entry["is_gec_sample"] = True
        if hinglish_with_error:
            entry["hinglish_with_error"] = hinglish_with_error
        if error_description:
            entry["error_description"] = error_description
    return entry


def write_dataset(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def make_life_processes() -> list[dict]:
    chapter = "Chapter 6: Life Processes"
    klass = "10"
    entries: list[dict] = []

    statements = [
        ("Nutrition", "Nutrition is the process by which organisms obtain food", "Nutrition ek process hai jisme organisms food lete hain"),
        ("Nutrition", "Autotrophic nutrition uses photosynthesis to make food", "Autotrophic nutrition mein photosynthesis se food banta hai"),
        ("Nutrition", "Heterotrophic nutrition depends on other organisms", "Heterotrophic nutrition dusre organisms par depend karta hai"),
        ("Nutrition", "In humans digestion starts in the mouth", "Humans mein digestion mouth se start hota hai"),
        ("Nutrition", "Salivary amylase breaks down starch", "Salivary amylase starch ko break karta hai"),
        ("Nutrition", "Small intestine absorbs digested food", "Small intestine digested food ko absorb karta hai"),
        ("Respiration", "Respiration releases energy from food", "Respiration food se energy release karta hai"),
        ("Respiration", "Aerobic respiration uses oxygen", "Aerobic respiration oxygen use karta hai"),
        ("Respiration", "Anaerobic respiration occurs without oxygen", "Anaerobic respiration oxygen ke bina hota hai"),
        ("Respiration", "In humans lungs are the main respiratory organ", "Humans mein lungs main respiratory organ hai"),
        ("Respiration", "Alveoli provide large surface area for gas exchange", "Alveoli gas exchange ke liye large surface area deta hai"),
        ("Respiration", "Oxygen is carried by haemoglobin in blood", "Oxygen blood mein haemoglobin ke saath carry hota hai"),
        ("Transportation", "Heart pumps blood throughout the body", "Heart body mein blood pump karta hai"),
        ("Transportation", "Arteries carry blood away from the heart", "Arteries heart se blood bahar le jaati hain"),
        ("Transportation", "Veins carry blood towards the heart", "Veins blood ko heart ki taraf le jaati hain"),
        ("Transportation", "Platelets help in blood clotting", "Platelets blood clotting mein help karte hain"),
        ("Transportation", "Xylem transports water in plants", "Xylem plants mein water transport karta hai"),
        ("Transportation", "Phloem transports food in plants", "Phloem plants mein food transport karta hai"),
        ("Excretion", "Excretion removes harmful metabolic wastes", "Excretion harmful metabolic wastes ko remove karta hai"),
        ("Excretion", "Kidneys filter blood and form urine", "Kidneys blood filter karke urine banate hain"),
        ("Excretion", "Nephrons are the functional units of kidney", "Nephron kidney ka functional unit hota hai"),
        ("Excretion", "Urine contains urea and excess salts", "Urine mein urea aur excess salts hote hain"),
        ("Excretion", "Stomata help in gaseous exchange in plants", "Stomata plants mein gaseous exchange mein help karte hain"),
        ("Nutrition", "Bile emulsifies fats in the small intestine", "Bile small intestine mein fats ko emulsify karta hai"),
        ("Respiration", "Diaphragm movement helps in breathing", "Diaphragm movement breathing mein help karta hai"),
        ("Transportation", "Double circulation keeps oxygenated and deoxygenated blood separate", "Double circulation oxygenated aur deoxygenated blood ko separate rakhta hai"),
        ("Excretion", "Sweat glands remove water and salts", "Sweat glands water aur salts ko remove karte hain"),
        ("Nutrition", "Peristalsis moves food through the alimentary canal", "Peristalsis alimentary canal mein food ko move karta hai"),
        ("Respiration", "Glucose is the main fuel for respiration", "Glucose respiration ke liye main fuel hota hai"),
        ("Excretion", "Dialysis is used when kidneys fail", "Kidney fail ho to dialysis use hota hai"),
    ]

    queries = [
        ("Nutrition", "What is the role of saliva in digestion", "Sir saliva digestion mein kya role play karta hai"),
        ("Nutrition", "Why is bile needed for fat digestion", "Sir bile fat digestion ke liye kyun zaroori hai"),
        ("Nutrition", "How does absorption happen in small intestine", "Sir small intestine mein absorption kaise hota hai"),
        ("Respiration", "Why is oxygen needed for aerobic respiration", "Sir aerobic respiration ke liye oxygen kyun chahiye"),
        ("Respiration", "What is the difference between aerobic and anaerobic respiration", "Sir aerobic aur anaerobic respiration mein kya difference hai"),
        ("Respiration", "How do alveoli help in gas exchange", "Sir alveoli gas exchange mein kaise help karte hain"),
        ("Transportation", "Why do arteries have thick walls", "Sir arteries ki walls thick kyun hoti hain"),
        ("Transportation", "What is the function of platelets", "Sir platelets ka function kya hai"),
        ("Transportation", "How does xylem transport water", "Sir xylem water transport kaise karta hai"),
        ("Transportation", "What is double circulation", "Sir double circulation kya hota hai"),
        ("Excretion", "Why are nephrons important in kidney", "Sir kidney mein nephron important kyun hai"),
        ("Excretion", "How is urine formed", "Sir urine formation kaise hota hai"),
        ("Excretion", "What is dialysis used for", "Sir dialysis kis liye use hota hai"),
        ("Nutrition", "What happens to undigested food", "Sir undigested food ka kya hota hai"),
        ("Respiration", "Why do we breathe faster during exercise", "Sir exercise ke time hum fast kyun breathe karte hain"),
    ]

    gec_samples = [
        ("Respiration", "Aerobic respiration oxygen se hota hai", "Aerobic respiration oxygen se hoti hai", "hota", "hoti", "gender agreement", "Respiration masculine hai isliye hota sahi hai"),
        ("Nutrition", "Digestion mouth mein start hota hai", "Digestion mouth mein start hoti hai", "hota", "hoti", "gender agreement", "Process ke liye hota use hota hai"),
        ("Transportation", "Arteries blood heart se le jata hai", "Arteries blood heart se le jaati hain", "jata", "jaati", "number agreement", "Arteries plural hai"),
        ("Excretion", "Kidneys urine banata hai", "Kidneys urine banate hain", "banata", "banate", "number agreement", "Kidneys plural hai"),
        ("Respiration", "Alveoli surface area deta hai", "Alveoli surface area dete hain", "deta", "dete", "number agreement", "Alveoli plural hai"),
    ]

    idx = 1
    for topic, en, hi in statements:
        entries.append(build_entry(f"10_06_{idx:03d}", en, hi, topic, chapter, klass))
        idx += 1
    for topic, en, hi in queries:
        entries.append(build_entry(f"10_06_{idx:03d}", en, hi, topic, chapter, klass, is_query=True, intent="explain_concept"))
        idx += 1
    for topic, correct, wrong, ew, cw, et, expl in gec_samples:
        entries.append(build_entry(
            f"10_06_{idx:03d}",
            correct,
            correct,
            topic,
            chapter,
            klass,
            is_gec=True,
            hinglish_with_error=wrong,
            error_description={
                "error_word": ew,
                "correct_word": cw,
                "error_type": et,
                "explanation": expl,
            },
        ))
        idx += 1

    return entries


def make_reproduction() -> list[dict]:
    chapter = "Chapter 8: How do Organisms Reproduce?"
    klass = "10"
    entries: list[dict] = []

    statements = [
        ("Asexual reproduction", "Asexual reproduction produces genetically similar offspring", "Asexual reproduction genetically similar offspring produce karta hai"),
        ("Asexual reproduction", "Binary fission is common in Amoeba", "Amoeba mein binary fission common hota hai"),
        ("Asexual reproduction", "Budding occurs in Yeast", "Yeast mein budding hoti hai"),
        ("Asexual reproduction", "Fragmentation is seen in Spirogyra", "Spirogyra mein fragmentation dekha jata hai"),
        ("Asexual reproduction", "Regeneration is ability to regrow body parts", "Regeneration body parts ko regrow karne ki ability hai"),
        ("Vegetative propagation", "Vegetative propagation uses roots stems or leaves", "Vegetative propagation roots stems ya leaves se hota hai"),
        ("Vegetative propagation", "Potato reproduces by stem tuber", "Potato stem tuber se reproduce karta hai"),
        ("Vegetative propagation", "Bryophyllum reproduces by leaf buds", "Bryophyllum leaf buds se reproduce karta hai"),
        ("Sexual reproduction", "Sexual reproduction involves fusion of gametes", "Sexual reproduction mein gametes ka fusion hota hai"),
        ("Sexual reproduction", "Fertilization forms a zygote", "Fertilization se zygote banta hai"),
        ("Sexual reproduction", "DNA copying creates variations", "DNA copying variations create karta hai"),
        ("Human reproduction", "Testes produce sperm", "Testes sperm produce karte hain"),
        ("Human reproduction", "Ovaries produce eggs", "Ovaries eggs produce karti hain"),
        ("Human reproduction", "Menstruation is the monthly cycle in females", "Menstruation females mein monthly cycle hota hai"),
        ("Human reproduction", "Fertilization in humans occurs in the oviduct", "Humans mein fertilization oviduct mein hota hai"),
        ("Human reproduction", "Embryo gets implanted in uterus", "Embryo uterus mein implant hota hai"),
        ("Sexual reproduction", "Pollination transfers pollen to stigma", "Pollination pollen ko stigma tak transfer karta hai"),
        ("Sexual reproduction", "Self pollination occurs in the same flower", "Self pollination same flower mein hota hai"),
        ("Sexual reproduction", "Cross pollination happens between different plants", "Cross pollination different plants ke beech hota hai"),
        ("Sexual reproduction", "Seeds are formed after fertilization", "Fertilization ke baad seeds bante hain"),
        ("Human reproduction", "Placenta provides nutrients to the embryo", "Placenta embryo ko nutrients deta hai"),
        ("Human reproduction", "Puberty begins with hormonal changes", "Puberty hormonal changes se start hoti hai"),
        ("Asexual reproduction", "Spore formation occurs in Rhizopus", "Rhizopus mein spore formation hota hai"),
        ("Sexual reproduction", "Gametes carry half the genetic material", "Gametes genetic material ka half part carry karte hain"),
        ("Human reproduction", "Contraception prevents unwanted pregnancy", "Contraception unwanted pregnancy ko prevent karta hai"),
        ("Human reproduction", "Ovulation is release of egg from ovary", "Ovulation ovary se egg release hona hai"),
        ("Sexual reproduction", "Flowers are the reproductive organs of plants", "Flowers plants ke reproductive organs hote hain"),
        ("Vegetative propagation", "Grafting joins two plant parts", "Grafting do plant parts ko join karta hai"),
        ("Vegetative propagation", "Tissue culture can produce many identical plants", "Tissue culture se many identical plants ban sakte hain"),
        ("Asexual reproduction", "No fertilization is needed in asexual reproduction", "Asexual reproduction mein fertilization ki zaroorat nahi hoti"),
    ]

    queries = [
        ("Asexual reproduction", "Why is asexual reproduction faster", "Sir asexual reproduction fast kyun hota hai"),
        ("Asexual reproduction", "What is the difference between fission and budding", "Sir fission aur budding mein kya difference hai"),
        ("Vegetative propagation", "How does potato reproduce", "Sir potato reproduce kaise karta hai"),
        ("Vegetative propagation", "Why is vegetative propagation useful in agriculture", "Sir vegetative propagation agriculture mein useful kyun hai"),
        ("Sexual reproduction", "What is the role of gametes", "Sir gametes ka role kya hota hai"),
        ("Sexual reproduction", "Why does sexual reproduction create variation", "Sir sexual reproduction variation kyun create karta hai"),
        ("Human reproduction", "What happens during puberty", "Sir puberty mein kya changes hote hain"),
        ("Human reproduction", "Where does fertilization occur in humans", "Sir humans mein fertilization kahan hota hai"),
        ("Human reproduction", "What is the function of placenta", "Sir placenta ka function kya hai"),
        ("Sexual reproduction", "What is pollination", "Sir pollination kya hota hai"),
        ("Sexual reproduction", "Why are seeds important", "Sir seeds important kyun hote hain"),
        ("Human reproduction", "What is ovulation", "Sir ovulation kya hota hai"),
        ("Asexual reproduction", "Give one example of spore formation", "Sir spore formation ka ek example batao"),
        ("Vegetative propagation", "How does grafting help in plants", "Sir grafting plants mein kaise help karta hai"),
        ("Human reproduction", "Why is contraception used", "Sir contraception kyun use hota hai"),
    ]

    gec_samples = [
        ("Human reproduction", "Ovaries eggs produce karti hain", "Ovaries eggs produce karta hai", "karti", "karta", "gender agreement", "Ovaries feminine plural hai"),
        ("Asexual reproduction", "Budding Yeast mein hoti hai", "Budding Yeast mein hota hai", "hoti", "hota", "gender agreement", "Budding feminine process hai"),
        ("Sexual reproduction", "Gametes genetic material carry karte hain", "Gametes genetic material carry karta hai", "karte", "karta", "number agreement", "Gametes plural hai"),
        ("Human reproduction", "Fertilization oviduct mein hota hai", "Fertilization oviduct mein hoti hai", "hota", "hoti", "gender agreement", "Fertilization masculine hai"),
        ("Vegetative propagation", "Potato stem tuber se reproduce karta hai", "Potato stem tuber se reproduce karti hai", "karta", "karti", "gender agreement", "Potato masculine noun ke liye karta"),
    ]

    idx = 1
    for topic, en, hi in statements:
        entries.append(build_entry(f"10_08_{idx:03d}", en, hi, topic, chapter, klass))
        idx += 1
    for topic, en, hi in queries:
        entries.append(build_entry(f"10_08_{idx:03d}", en, hi, topic, chapter, klass, is_query=True, intent="explain_concept"))
        idx += 1
    for topic, correct, wrong, ew, cw, et, expl in gec_samples:
        entries.append(build_entry(
            f"10_08_{idx:03d}",
            correct,
            correct,
            topic,
            chapter,
            klass,
            is_gec=True,
            hinglish_with_error=wrong,
            error_description={
                "error_word": ew,
                "correct_word": cw,
                "error_type": et,
                "explanation": expl,
            },
        ))
        idx += 1

    return entries


def make_heredity() -> list[dict]:
    chapter = "Chapter 9: Heredity and Evolution"
    klass = "10"
    entries: list[dict] = []

    statements = [
        ("Heredity", "Heredity is the transmission of traits from parents to offspring", "Heredity parents se offspring tak traits ka transmission hai"),
        ("Heredity", "Genes are segments of DNA", "Genes DNA ke segments hote hain"),
        ("Heredity", "Chromosomes carry genes", "Chromosomes genes ko carry karte hain"),
        ("Heredity", "Alleles are alternative forms of a gene", "Alleles gene ke alternative forms hote hain"),
        ("Heredity", "Mendel studied inheritance in pea plants", "Mendel ne pea plants mein inheritance study kiya"),
        ("Heredity", "A dominant trait expresses in F1 generation", "Dominant trait F1 generation mein express hota hai"),
        ("Heredity", "A recessive trait appears in F2 generation", "Recessive trait F2 generation mein appear hota hai"),
        ("Heredity", "Genotype refers to genetic makeup", "Genotype genetic makeup ko refer karta hai"),
        ("Heredity", "Phenotype refers to observable traits", "Phenotype observable traits ko refer karta hai"),
        ("Heredity", "Sex is determined by XX and XY chromosomes", "Sex XX aur XY chromosomes se determine hota hai"),
        ("Evolution", "Variations provide the basis for evolution", "Variations evolution ka base provide karte hain"),
        ("Evolution", "Natural selection favors useful variations", "Natural selection useful variations ko favor karta hai"),
        ("Evolution", "Fossils give evidence of evolution", "Fossils evolution ka evidence dete hain"),
        ("Evolution", "Speciation occurs when populations diverge", "Speciation tab hota hai jab populations diverge karte hain"),
        ("Evolution", "Evolution is not a directed process", "Evolution koi directed process nahi hota"),
        ("Heredity", "Blood group inheritance follows multiple alleles", "Blood group inheritance multiple alleles follow karta hai"),
        ("Heredity", "Mutation can create new alleles", "Mutation new alleles create kar sakta hai"),
        ("Evolution", "Homologous organs show common ancestry", "Homologous organs common ancestry show karte hain"),
        ("Evolution", "Analogous organs have similar function but different origin", "Analogous organs ka function similar hota hai par origin different hota hai"),
        ("Evolution", "Acquired traits are not inherited", "Acquired traits inherit nahi hote"),
        ("Heredity", "DNA copying is the basis of inheritance", "DNA copying inheritance ka base hai"),
        ("Heredity", "Traits are controlled by genes", "Traits genes ke dwara control hote hain"),
        ("Evolution", "Genetic drift can change gene frequency", "Genetic drift gene frequency ko change kar sakta hai"),
        ("Evolution", "Small changes over time can lead to new species", "Time ke saath small changes se new species ban sakti hain"),
        ("Heredity", "Sexual reproduction creates more variation", "Sexual reproduction zyada variation create karta hai"),
        ("Heredity", "Trait expression depends on genotype and environment", "Trait expression genotype aur environment par depend karta hai"),
        ("Evolution", "Survival of the fittest is linked to adaptation", "Survival of the fittest adaptation se linked hai"),
        ("Evolution", "Classification uses evolutionary relationships", "Classification evolutionary relationships ko use karta hai"),
        ("Heredity", "Mendel proposed the law of segregation", "Mendel ne law of segregation propose kiya"),
        ("Heredity", "Mendel proposed the law of independent assortment", "Mendel ne law of independent assortment propose kiya"),
    ]

    queries = [
        ("Heredity", "What is the difference between genotype and phenotype", "Sir genotype aur phenotype mein kya difference hai"),
        ("Heredity", "Why does a recessive trait appear in F2", "Sir recessive trait F2 mein kyun aata hai"),
        ("Heredity", "What is the role of chromosomes", "Sir chromosomes ka role kya hai"),
        ("Heredity", "How do genes control traits", "Sir genes traits ko kaise control karte hain"),
        ("Heredity", "What is an allele", "Sir allele kya hota hai"),
        ("Evolution", "How does natural selection work", "Sir natural selection kaise work karta hai"),
        ("Evolution", "Why are variations important", "Sir variations important kyun hote hain"),
        ("Evolution", "What is speciation", "Sir speciation kya hota hai"),
        ("Evolution", "What do fossils tell us", "Sir fossils hume kya batate hain"),
        ("Heredity", "How is sex determined in humans", "Sir humans mein sex determination kaise hota hai"),
        ("Evolution", "What is the difference between homologous and analogous organs", "Sir homologous aur analogous organs mein kya difference hai"),
        ("Evolution", "Why are acquired traits not inherited", "Sir acquired traits inherit kyun nahi hote"),
        ("Heredity", "What is mutation", "Sir mutation kya hota hai"),
        ("Heredity", "What is the law of segregation", "Sir law of segregation kya hai"),
        ("Heredity", "What is the law of independent assortment", "Sir law of independent assortment kya hai"),
    ]

    gec_samples = [
        ("Heredity", "Genes DNA ke segments hote hain", "Genes DNA ke segments hota hai", "hote", "hota", "number agreement", "Genes plural hai"),
        ("Heredity", "Mendel ne pea plants mein experiment kiya", "Mendel ne pea plants mein experiment ki", "kiya", "ki", "tense agreement", "Experiment ke liye kiya sahi hai"),
        ("Evolution", "Fossils evidence dete hain", "Fossils evidence deta hai", "dete", "deta", "number agreement", "Fossils plural hai"),
        ("Evolution", "Natural selection useful traits ko favor karta hai", "Natural selection useful traits ko favor karti hai", "karta", "karti", "gender agreement", "Selection masculine hai"),
        ("Heredity", "Chromosomes genes ko carry karte hain", "Chromosomes genes ko carry karta hai", "karte", "karta", "number agreement", "Chromosomes plural hai"),
    ]

    idx = 1
    for topic, en, hi in statements:
        entries.append(build_entry(f"10_09_{idx:03d}", en, hi, topic, chapter, klass))
        idx += 1
    for topic, en, hi in queries:
        entries.append(build_entry(f"10_09_{idx:03d}", en, hi, topic, chapter, klass, is_query=True, intent="explain_concept"))
        idx += 1
    for topic, correct, wrong, ew, cw, et, expl in gec_samples:
        entries.append(build_entry(
            f"10_09_{idx:03d}",
            correct,
            correct,
            topic,
            chapter,
            klass,
            is_gec=True,
            hinglish_with_error=wrong,
            error_description={
                "error_word": ew,
                "correct_word": cw,
                "error_type": et,
                "explanation": expl,
            },
        ))
        idx += 1

    return entries


def make_environment() -> list[dict]:
    chapter = "Chapter 15: Our Environment"
    klass = "10"
    entries: list[dict] = []

    statements = [
        ("Ecosystem", "An ecosystem includes biotic and abiotic components", "Ecosystem mein biotic aur abiotic components hote hain"),
        ("Ecosystem", "Producers make food using sunlight", "Producers sunlight se food banate hain"),
        ("Ecosystem", "Consumers depend on producers for energy", "Consumers energy ke liye producers par depend karte hain"),
        ("Food chain", "A food chain shows energy transfer", "Food chain energy transfer dikhata hai"),
        ("Food chain", "Food web is a network of food chains", "Food web food chains ka network hota hai"),
        ("Food chain", "Energy decreases at each trophic level", "Har trophic level par energy kam hoti hai"),
        ("Ozone", "Ozone layer protects Earth from UV rays", "Ozone layer Earth ko UV rays se protect karta hai"),
        ("Ozone", "CFCs cause ozone depletion", "CFCs ozone depletion ka cause bante hain"),
        ("Ozone", "Ozone depletion increases UV radiation", "Ozone depletion se UV radiation badhta hai"),
        ("Waste management", "Biodegradable waste can be decomposed", "Biodegradable waste decompose ho sakta hai"),
        ("Waste management", "Non biodegradable waste persists for long time", "Non biodegradable waste long time tak rehta hai"),
        ("Waste management", "Plastics should be reduced and recycled", "Plastics ko reduce aur recycle karna chahiye"),
        ("Ecosystem", "Decomposers recycle nutrients in ecosystem", "Decomposers ecosystem mein nutrients ko recycle karte hain"),
        ("Food chain", "Loss of one species affects the food web", "Ek species ke loss se food web impact hota hai"),
        ("Ecosystem", "Ecosystem maintains balance in nature", "Ecosystem nature mein balance maintain karta hai"),
        ("Waste management", "Segregation of waste helps recycling", "Waste segregation recycling mein help karta hai"),
        ("Ecosystem", "Top consumers are at highest trophic level", "Top consumers highest trophic level par hote hain"),
        ("Food chain", "Only 10 percent energy passes to next level", "Next level ko sirf 10 percent energy pass hoti hai"),
        ("Ozone", "Montreal protocol helped reduce CFC use", "Montreal protocol ne CFC use ko reduce kiya"),
        ("Waste management", "Composting converts organic waste to manure", "Composting organic waste ko manure mein convert karta hai"),
        ("Ecosystem", "Aquatic and terrestrial are major ecosystem types", "Aquatic aur terrestrial major ecosystem types hain"),
        ("Food chain", "Grass is a producer in grassland food chain", "Grassland food chain mein grass producer hota hai"),
        ("Ozone", "Ozone is formed by oxygen molecules in stratosphere", "Ozone stratosphere mein oxygen molecules se banta hai"),
        ("Waste management", "Landfills are used for waste disposal", "Landfills waste disposal ke liye use hote hain"),
        ("Ecosystem", "Human activities can disturb ecosystem balance", "Human activities ecosystem balance ko disturb kar sakti hain"),
        ("Food chain", "Secondary consumers eat primary consumers", "Secondary consumers primary consumers ko khate hain"),
        ("Waste management", "Recycling saves energy and resources", "Recycling energy aur resources save karta hai"),
        ("Ozone", "UV radiation can cause skin cancer", "UV radiation skin cancer cause kar sakta hai"),
        ("Ecosystem", "Biodiversity supports ecosystem stability", "Biodiversity ecosystem stability ko support karti hai"),
        ("Food chain", "Trophic level represents feeding position", "Trophic level feeding position ko represent karta hai"),
    ]

    queries = [
        ("Ecosystem", "What is an ecosystem", "Sir ecosystem kya hota hai"),
        ("Ecosystem", "Why are decomposers important", "Sir decomposers important kyun hote hain"),
        ("Food chain", "What is the difference between food chain and food web", "Sir food chain aur food web mein kya difference hai"),
        ("Food chain", "Why does energy decrease at each trophic level", "Sir trophic level par energy kam kyun hoti hai"),
        ("Food chain", "Give an example of a food chain", "Sir food chain ka example batao"),
        ("Ozone", "How does ozone layer protect us", "Sir ozone layer hume kaise protect karta hai"),
        ("Ozone", "Why do CFCs damage ozone", "Sir CFCs ozone ko damage kyun karte hain"),
        ("Waste management", "What is biodegradable waste", "Sir biodegradable waste kya hota hai"),
        ("Waste management", "Why should we recycle plastics", "Sir plastics recycle kyun karna chahiye"),
        ("Waste management", "How does composting help", "Sir composting kaise help karta hai"),
        ("Ecosystem", "What happens if one species disappears", "Sir ek species gayab ho jaye to kya hota hai"),
        ("Food chain", "Who are primary consumers", "Sir primary consumers kaun hote hain"),
        ("Ozone", "What is the Montreal protocol", "Sir Montreal protocol kya hai"),
        ("Waste management", "What is waste segregation", "Sir waste segregation kya hota hai"),
        ("Ecosystem", "Why is biodiversity important", "Sir biodiversity important kyun hai"),
    ]

    gec_samples = [
        ("Ecosystem", "Ecosystem mein biotic aur abiotic components hote hain", "Ecosystem mein biotic aur abiotic components hota hai", "hote", "hota", "number agreement", "Components plural hai"),
        ("Food chain", "Food chain energy transfer dikhata hai", "Food chain energy transfer dikhati hai", "dikhata", "dikhati", "gender agreement", "Chain masculine hai"),
        ("Ozone", "Ozone layer Earth ko protect karta hai", "Ozone layer Earth ko protect karti hai", "karta", "karti", "gender agreement", "Layer feminine hai"),
        ("Waste management", "Biodegradable waste decompose ho sakta hai", "Biodegradable waste decompose ho sakte hain", "sakta", "sakte", "number agreement", "Waste singular hai"),
        ("Waste management", "Plastics recycle karna chahiye", "Plastics recycle karne chahiye", "karna", "karne", "verb form", "Karna sahi infinitive hai"),
    ]

    idx = 1
    for topic, en, hi in statements:
        entries.append(build_entry(f"10_15_{idx:03d}", en, hi, topic, chapter, klass))
        idx += 1
    for topic, en, hi in queries:
        entries.append(build_entry(f"10_15_{idx:03d}", en, hi, topic, chapter, klass, is_query=True, intent="explain_concept"))
        idx += 1
    for topic, correct, wrong, ew, cw, et, expl in gec_samples:
        entries.append(build_entry(
            f"10_15_{idx:03d}",
            correct,
            correct,
            topic,
            chapter,
            klass,
            is_gec=True,
            hinglish_with_error=wrong,
            error_description={
                "error_word": ew,
                "correct_word": cw,
                "error_type": et,
                "explanation": expl,
            },
        ))
        idx += 1

    return entries


def make_natural_resources() -> list[dict]:
    chapter = "Chapter 16: Management of Natural Resources"
    klass = "10"
    entries: list[dict] = []

    statements = [
        ("Conservation", "Natural resources should be used sustainably", "Natural resources ko sustainably use karna chahiye"),
        ("Conservation", "Water is a renewable resource", "Water renewable resource hai"),
        ("Conservation", "Forests help in soil conservation", "Forests soil conservation mein help karte hain"),
        ("Conservation", "Overuse of resources leads to depletion", "Resources ka overuse depletion ka cause banta hai"),
        ("Water management", "Rainwater harvesting saves water", "Rainwater harvesting water save karta hai"),
        ("Water management", "Watershed management improves water storage", "Watershed management water storage improve karta hai"),
        ("Energy", "Coal and petroleum are non renewable resources", "Coal aur petroleum non renewable resources hain"),
        ("Energy", "Alternative energy reduces fossil fuel use", "Alternative energy fossil fuel use ko reduce karta hai"),
        ("Energy", "Solar energy is clean and renewable", "Solar energy clean aur renewable hai"),
        ("Forest", "Deforestation affects biodiversity", "Deforestation biodiversity ko affect karta hai"),
        ("Forest", "Forest produce oxygen and absorb carbon dioxide", "Forest oxygen produce karte hain aur carbon dioxide absorb karte hain"),
        ("Conservation", "Reduce reuse recycle are key principles", "Reduce reuse recycle key principles hain"),
        ("Conservation", "Sustainable development balances use and protection", "Sustainable development use aur protection ko balance karta hai"),
        ("Water management", "Check dams help recharge groundwater", "Check dams groundwater recharge mein help karte hain"),
        ("Energy", "Biogas is made from animal waste", "Biogas animal waste se banta hai"),
        ("Energy", "Hydroelectric power uses flowing water", "Hydroelectric power flowing water use karta hai"),
        ("Forest", "Afforestation restores forest cover", "Afforestation forest cover restore karta hai"),
        ("Conservation", "Community participation improves resource management", "Community participation resource management improve karti hai"),
        ("Water management", "Irrigation efficiency saves water", "Irrigation efficiency water save karti hai"),
        ("Energy", "Energy conservation reduces demand", "Energy conservation demand ko reduce karta hai"),
        ("Conservation", "Wildlife protection maintains ecological balance", "Wildlife protection ecological balance maintain karta hai"),
        ("Water management", "Pollution reduces usable water", "Pollution usable water ko reduce karta hai"),
        ("Energy", "Fossil fuels are formed over millions of years", "Fossil fuels millions of years mein form hote hain"),
        ("Forest", "Forest fires destroy resources", "Forest fires resources ko destroy karte hain"),
        ("Conservation", "Resource management includes planning and regulation", "Resource management planning aur regulation include karta hai"),
        ("Energy", "Wind energy is renewable", "Wind energy renewable hai"),
        ("Water management", "Drip irrigation saves water", "Drip irrigation water save karta hai"),
        ("Conservation", "Excessive mining damages land", "Excessive mining land ko damage karta hai"),
        ("Forest", "Chipko movement promoted forest conservation", "Chipko movement forest conservation ko promote karta hai"),
        ("Conservation", "Public awareness helps conservation", "Public awareness conservation mein help karta hai"),
    ]

    queries = [
        ("Conservation", "Why is sustainable use important", "Sir sustainable use important kyun hai"),
        ("Water management", "What is rainwater harvesting", "Sir rainwater harvesting kya hota hai"),
        ("Water management", "How do check dams help", "Sir check dams kaise help karte hain"),
        ("Energy", "Why are coal and petroleum called non renewable", "Sir coal aur petroleum non renewable kyun kehte hain"),
        ("Energy", "What is the advantage of solar energy", "Sir solar energy ka advantage kya hai"),
        ("Forest", "How does deforestation affect environment", "Sir deforestation environment ko kaise affect karta hai"),
        ("Conservation", "What are the three R principles", "Sir three R principles kya hain"),
        ("Water management", "Why is drip irrigation useful", "Sir drip irrigation useful kyun hai"),
        ("Energy", "What is biogas", "Sir biogas kya hota hai"),
        ("Forest", "What is the Chipko movement", "Sir Chipko movement kya hai"),
        ("Conservation", "How does community help in conservation", "Sir community conservation mein kaise help karti hai"),
        ("Water management", "How does pollution affect water resources", "Sir pollution water resources ko kaise affect karta hai"),
        ("Energy", "Why is energy conservation needed", "Sir energy conservation kyun zaroori hai"),
        ("Forest", "Why is afforestation important", "Sir afforestation important kyun hai"),
        ("Conservation", "What is sustainable development", "Sir sustainable development kya hota hai"),
    ]

    gec_samples = [
        ("Conservation", "Natural resources ko sustainably use karna chahiye", "Natural resources ko sustainably use karne chahiye", "karna", "karna", "verb form", "Infinitive karna sahi hai"),
        ("Water management", "Rainwater harvesting water save karta hai", "Rainwater harvesting water save karti hai", "karta", "karti", "gender agreement", "Harvesting masculine hai"),
        ("Energy", "Coal aur petroleum non renewable resources hain", "Coal aur petroleum non renewable resources hai", "hain", "hai", "number agreement", "Resources plural hai"),
        ("Forest", "Afforestation forest cover restore karta hai", "Afforestation forest cover restore karti hai", "karta", "karti", "gender agreement", "Afforestation masculine process hai"),
        ("Conservation", "Public awareness conservation mein help karta hai", "Public awareness conservation mein help karti hai", "karta", "karti", "gender agreement", "Awareness masculine noun hai"),
    ]

    idx = 1
    for topic, en, hi in statements:
        entries.append(build_entry(f"10_16_{idx:03d}", en, hi, topic, chapter, klass))
        idx += 1
    for topic, en, hi in queries:
        entries.append(build_entry(f"10_16_{idx:03d}", en, hi, topic, chapter, klass, is_query=True, intent="explain_concept"))
        idx += 1
    for topic, correct, wrong, ew, cw, et, expl in gec_samples:
        entries.append(build_entry(
            f"10_16_{idx:03d}",
            correct,
            correct,
            topic,
            chapter,
            klass,
            is_gec=True,
            hinglish_with_error=wrong,
            error_description={
                "error_word": ew,
                "correct_word": cw,
                "error_type": et,
                "explanation": expl,
            },
        ))
        idx += 1

    return entries


def main() -> None:
    datasets = {
        "ch06": (make_life_processes(), "dataset.json"),
        "ch08": (make_reproduction(), "dataset.json"),
        "ch09": (make_heredity(), "dataset.json"),
        "ch15": (make_environment(), "dataset.json"),
        "ch16": (make_natural_resources(), "dataset.json"),
    }

    for chapter_folder, (entries, filename) in datasets.items():
        out_dir = DATA_ROOT / chapter_folder
        out_path = out_dir / filename
        write_dataset(out_path, entries)
        print(f"[OK] Wrote {len(entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()
