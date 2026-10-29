"""
Add one chapter's hand-written QA entries to the curated datasets:
    Biology        -> data/bio_9_10.json
    Social Science -> data/ss_9_10.json

The input is a JSON list of compact entries:
    {"topic", "intent", "q", "a", "ctx", "q_hi", "a_hi"}

Each entry is validated (intent label, Roman-script Hinglish, context length, no duplicate
questions against the whole dataset) and expanded into the dataset schema (question_english,
answer_hinglish, ncert_context, Gemma-format messages / messages_english / text).

Chapters are named like their data/processed/ folder (class10_bio_ch06, ss9_ch02, ss10_his_ch01).

Usage:
    python scripts/build_dataset.py entries.json --chapter ss10_his_ch01            # append
    python scripts/build_dataset.py entries.json --chapter ss10_his_ch01 --replace  # redo a chapter
    python scripts/build_dataset.py entries.json --chapter ss10_his_ch01 --check    # validate only
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"


class Chapter:
    def __init__(self, cls: str, num: str, title: str, subject: str, id_tag: str, sub_subject: str | None = None):
        self.cls, self.num, self.title = cls, num, title          # cls "9"/"10", num "ch02"
        self.subject, self.sub_subject, self.id_tag = subject, sub_subject, id_tag

    @property
    def id_prefix(self) -> str:
        return f"c{int(self.cls):02d}_{self.id_tag}_{self.num}"

    @property
    def output(self) -> Path:
        return DATA_DIR / ("bio_9_10.json" if self.subject == "Biology" else "ss_9_10.json")


def _bio(cls, num, title):
    return Chapter(cls, num, title, "Biology", "bio")


def _ss(cls, num, title, sub_subject, id_tag):
    return Chapter(cls, num, title, "Social Science", id_tag, sub_subject)


CHAPTERS = {
    "class9_bio_ch02":  _bio("9", "ch02", "Cell: The Building Block of Life"),
    "class9_bio_ch03":  _bio("9", "ch03", "Tissues in Action"),
    "class9_bio_ch11":  _bio("9", "ch11", "Reproduction: How Life Continues"),
    "class9_bio_ch12":  _bio("9", "ch12", "Patterns in Life: Diversity and Classification"),
    "class10_bio_ch05": _bio("10", "ch05", "Life Processes"),
    "class10_bio_ch06": _bio("10", "ch06", "Control and Coordination"),
    "class10_bio_ch07": _bio("10", "ch07", "How do Organisms Reproduce?"),
    "class10_bio_ch08": _bio("10", "ch08", "Heredity"),
    "class10_bio_ch13": _bio("10", "ch13", "Our Environment"),
    # Class 9 Social Science is a single integrated book.
    "ss9_ch01": _ss("9", "ch01", "Understanding Social Science", "General", "ss"),
    "ss9_ch02": _ss("9", "ch02", "Shaping of the Earth's Surface", "Geography", "ss"),
    "ss9_ch03": _ss("9", "ch03", "Atmosphere and Climate", "Geography", "ss"),
    "ss9_ch04": _ss("9", "ch04", "Early Humans and Beginning of Civilisation", "History", "ss"),
    "ss9_ch05": _ss("9", "ch05", "State and Society up to 1000 CE", "History", "ss"),
    "ss9_ch06": _ss("9", "ch06", "Democracy", "Political Science", "ss"),
    "ss9_ch07": _ss("9", "ch07", "Elections", "Political Science", "ss"),
    "ss9_ch08": _ss("9", "ch08", "Building Blocks in Economics: The Problem of Choice", "Economics", "ss"),
    "ss9_ch09": _ss("9", "ch09", "The Price Puzzle: What Drives the Market", "Economics", "ss"),
    "ss10_his_ch01": _ss("10", "ch01", "The Rise of Nationalism in Europe", "History", "his"),
    "ss10_his_ch02": _ss("10", "ch02", "Nationalism in India", "History", "his"),
    "ss10_his_ch03": _ss("10", "ch03", "The Making of a Global World", "History", "his"),
    "ss10_his_ch04": _ss("10", "ch04", "The Age of Industrialisation", "History", "his"),
    "ss10_his_ch05": _ss("10", "ch05", "Print Culture and the Modern World", "History", "his"),
    "ss10_geo_ch01": _ss("10", "ch01", "Resources and Development", "Geography", "geo"),
    "ss10_geo_ch02": _ss("10", "ch02", "Forest and Wildlife Resources", "Geography", "geo"),
    "ss10_geo_ch03": _ss("10", "ch03", "Water Resources", "Geography", "geo"),
    "ss10_geo_ch04": _ss("10", "ch04", "Agriculture", "Geography", "geo"),
    "ss10_geo_ch05": _ss("10", "ch05", "Minerals and Energy Resources", "Geography", "geo"),
    "ss10_geo_ch06": _ss("10", "ch06", "Manufacturing Industries", "Geography", "geo"),
    "ss10_geo_ch07": _ss("10", "ch07", "Lifelines of National Economy", "Geography", "geo"),
    "ss10_civics_ch01": _ss("10", "ch01", "Power-sharing", "Political Science", "civ"),
    "ss10_civics_ch02": _ss("10", "ch02", "Federalism", "Political Science", "civ"),
    "ss10_civics_ch03": _ss("10", "ch03", "Gender, Religion and Caste", "Political Science", "civ"),
    "ss10_civics_ch04": _ss("10", "ch04", "Political Parties", "Political Science", "civ"),
    "ss10_civics_ch05": _ss("10", "ch05", "Outcomes of Democracy", "Political Science", "civ"),
    "ss10_eco_ch01": _ss("10", "ch01", "Development", "Economics", "eco"),
    "ss10_eco_ch02": _ss("10", "ch02", "Sectors of the Indian Economy", "Economics", "eco"),
    "ss10_eco_ch03": _ss("10", "ch03", "Money and Credit", "Economics", "eco"),
    "ss10_eco_ch04": _ss("10", "ch04", "Globalisation and the Indian Economy", "Economics", "eco"),
    "ss10_eco_ch05": _ss("10", "ch05", "Consumer Rights", "Economics", "eco"),
}

INTENTS = {
    "definition", "explain_concept", "short_answer", "describe_function",
    "compare_concepts", "list_enumerate", "give_example", "application_reallife",
    "analytical", "numerical_problem", "mcq", "assertion_reason", "true_false",
    "fill_blank", "one_word",
}

REQUIRED  = ("topic", "intent", "q", "a", "ctx", "q_hi", "a_hi")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def validate(entry: dict, where: str) -> list[str]:
    errors = []
    for key in REQUIRED:
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            errors.append(f"{where}: missing/empty '{key}'")
    if errors:
        return errors

    if entry["intent"] not in INTENTS:
        errors.append(f"{where}: unknown intent '{entry['intent']}'")
    for key in ("q_hi", "a_hi"):
        if DEVANAGARI.search(entry[key]):
            errors.append(f"{where}: '{key}' contains Devanagari (expected Roman Hinglish)")
    if entry["a_hi"].strip() == entry["a"].strip():
        errors.append(f"{where}: a_hi is identical to the English answer")
    if len(entry["ctx"].split()) < 25:
        errors.append(f"{where}: ctx too short ({len(entry['ctx'].split())} words)")
    return errors


# Gemma has no system role, so the NCERT context goes inside the user turn.
# Keep this identical to the prompt used at inference time (src/pipeline.py).
# For a share of records the intent line is left out (intent dropout), so Gemma learns
# to answer from the question itself and treats the MuRIL-predicted intent as a hint.
# At inference, drop the line too when the classifier is unsure.
PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "{intent_line}"
    "{instruction}"
)
INSTRUCTION_HI = "Answer the question in Hinglish using only the given context."
INSTRUCTION_EN = "Answer the question in English using only the given context."


def gemma_messages(context: str, question: str, answer: str, intent: str | None, instruction: str) -> list[dict]:
    intent_line = f"Intent Type: {intent}\n" if intent else ""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question, intent_line=intent_line,
                                    instruction=instruction)
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def gemma_text(messages: list[dict]) -> str:
    # Same output as tokenizer.apply_chat_template(messages, tokenize=False) minus <bos>,
    # which the tokenizer adds itself during training (a doubled <bos> hurts Gemma).
    user, model = messages[0]["content"], messages[1]["content"]
    return f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{model}<end_of_turn>\n"


def keep_intent(record_id: str, dropout: float) -> bool:
    # Hash-based rather than random so the same records lose their intent on every build.
    bucket = int(hashlib.md5(record_id.encode()).hexdigest(), 16) % 10_000 / 10_000
    return bucket >= dropout


def build_record(entry: dict, chapter: Chapter, idx: int, intent_dropout: float) -> dict:
    record_id = f"{chapter.id_prefix}_{idx:04d}"
    prompt_intent = entry["intent"] if keep_intent(record_id, intent_dropout) else None
    messages = gemma_messages(entry["ctx"], entry["q_hi"], entry["a_hi"], prompt_intent, INSTRUCTION_HI)
    messages_english = gemma_messages(entry["ctx"], entry["q"], entry["a"], prompt_intent, INSTRUCTION_EN)
    record = {
        "id": record_id,
        "class": chapter.cls,
        "subject": chapter.subject,
        "chapter": chapter.title,
        "chapter_num": chapter.num,
        "topic": entry["topic"],
        "source": "curated",
        "is_student_query": False,
        "intent": entry["intent"],
        "intent_in_prompt": prompt_intent is not None,
        "question_english": entry["q"],
        "question_hinglish": entry["q_hi"],
        "answer_english": entry["a"],
        "answer_hinglish": entry["a_hi"],
        "ncert_context": entry["ctx"],
        "messages": messages,
        "messages_english": messages_english,
        "text": gemma_text(messages),
    }
    if chapter.sub_subject:
        record = {**{k: v for k, v in record.items() if k != "chapter"}, "sub_subject": chapter.sub_subject,
                  "chapter": chapter.title}
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entries", type=Path, help="JSON list of compact entries for one chapter")
    parser.add_argument("--chapter", required=True, choices=sorted(CHAPTERS), metavar="CHAPTER",
                        help="processed-folder name, e.g. class10_bio_ch06 or ss10_his_ch01")
    parser.add_argument("--replace", action="store_true", help="drop the chapter's existing records first")
    parser.add_argument("--check", action="store_true", help="validate only, do not write")
    parser.add_argument("--intent-dropout", type=float, default=0.12,
                        help="share of records whose prompt omits the intent line (default 0.12)")
    args = parser.parse_args()

    chapter = CHAPTERS[args.chapter]
    output = chapter.output
    dataset = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    in_chapter = lambda r: r["id"].rsplit("_", 1)[0] == chapter.id_prefix
    if args.replace:
        dataset = [r for r in dataset if not in_chapter(r)]

    entries = json.loads(args.entries.read_text(encoding="utf-8"))
    errors = []
    for i, entry in enumerate(entries):
        errors.extend(validate(entry, f"entry[{i}]"))

    existing = {r["question_english"].strip().lower() for r in dataset}
    new_qs = Counter(e["q"].strip().lower() for e in entries if e.get("q"))
    errors.extend(f"duplicate question: {q!r}" for q, n in new_qs.items() if n > 1 or q in existing)

    if errors:
        print("\n".join(errors))
        print(f"\n[FAIL] {len(errors)} problem(s)")
        sys.exit(1)

    start = sum(1 for r in dataset if in_chapter(r)) + 1
    dataset.extend(build_record(e, chapter, start + i, args.intent_dropout) for i, e in enumerate(entries))

    print(f"  {args.chapter} {chapter.title}: +{len(entries)} entries")
    for name, ch in CHAPTERS.items():
        n = sum(1 for r in dataset if r["id"].rsplit("_", 1)[0] == ch.id_prefix)
        if n:
            print(f"    {name:<18} {ch.title:<50} {n}")
    print(f"  total in {output.name}: {len(dataset)}")

    if not args.check:
        output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] updated {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
