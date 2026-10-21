"""
Add one chapter's hand-written QA entries to data/bio_9_10.json.

The input is a JSON list of compact entries:
    {"topic", "intent", "q", "a", "ctx", "q_hi", "a_hi"}

Each entry is validated (intent label, Roman-script Hinglish, context length, no duplicate
questions against the whole dataset) and expanded into the dataset schema (question_english,
answer_hinglish, ncert_context, Gemma-format messages / messages_english / text).

Usage:
    python scripts/build_bio_dataset.py entries.json --chapter 10 ch06            # append
    python scripts/build_bio_dataset.py entries.json --chapter 10 ch06 --replace  # redo a chapter
    python scripts/build_bio_dataset.py entries.json --chapter 10 ch06 --check    # validate only
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH  = PROJECT_ROOT / "data" / "bio_9_10.json"

CHAPTERS = {
    ("09", "ch02"): "Cell: The Building Block of Life",
    ("09", "ch03"): "Tissues in Action",
    ("09", "ch11"): "Reproduction: How Life Continues",
    ("09", "ch12"): "Patterns in Life: Diversity and Classification",
    ("10", "ch05"): "Life Processes",
    ("10", "ch06"): "Control and Coordination",
    ("10", "ch07"): "How do Organisms Reproduce?",
    ("10", "ch08"): "Heredity",
    ("10", "ch13"): "Our Environment",
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


def build_record(entry: dict, cls: str, chapter_num: str, idx: int, intent_dropout: float) -> dict:
    chapter = CHAPTERS[(cls, chapter_num)]
    record_id = f"c{cls}_bio_{chapter_num}_{idx:04d}"
    prompt_intent = entry["intent"] if keep_intent(record_id, intent_dropout) else None
    messages = gemma_messages(entry["ctx"], entry["q_hi"], entry["a_hi"], prompt_intent, INSTRUCTION_HI)
    messages_english = gemma_messages(entry["ctx"], entry["q"], entry["a"], prompt_intent, INSTRUCTION_EN)
    return {
        "id": record_id,
        "class": str(int(cls)),
        "subject": "Biology",
        "chapter": chapter,
        "chapter_num": chapter_num,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entries", type=Path, help="JSON list of compact entries for one chapter")
    parser.add_argument("--chapter", nargs=2, required=True, metavar=("CLASS", "CHAPTER"),
                        help="e.g. 10 ch06")
    parser.add_argument("--replace", action="store_true", help="drop the chapter's existing records first")
    parser.add_argument("--check", action="store_true", help="validate only, do not write")
    parser.add_argument("--intent-dropout", type=float, default=0.12,
                        help="share of records whose prompt omits the intent line (default 0.12)")
    args = parser.parse_args()

    key = (args.chapter[0].zfill(2), args.chapter[1])
    if key not in CHAPTERS:
        sys.exit(f"unknown chapter {key}; known: {sorted(CHAPTERS)}")
    cls, chapter_num = key

    dataset = json.loads(OUTPUT_PATH.read_text(encoding="utf-8")) if OUTPUT_PATH.exists() else []
    in_chapter = lambda r: r["class"] == str(int(cls)) and r["chapter_num"] == chapter_num
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
    records = [build_record(e, cls, chapter_num, start + i, args.intent_dropout) for i, e in enumerate(entries)]
    dataset.extend(records)

    print(f"  c{cls} {chapter_num} {CHAPTERS[key]}: +{len(records)} entries")
    for k in sorted(CHAPTERS):
        n = sum(1 for r in dataset if r["class"] == str(int(k[0])) and r["chapter_num"] == k[1])
        if n:
            print(f"    c{k[0]} {k[1]} {CHAPTERS[k]:<48} {n}")
    print(f"  total: {len(dataset)}")

    if not args.check:
        OUTPUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] updated {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
