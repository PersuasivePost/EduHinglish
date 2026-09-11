"""
EduHinglish — Hinglish Enrichment via Gemini or Ollama
=======================================================
Fills `question_hinglish` and `answer_hinglish` for entries where they are blank.

Backends:
  - gemini : Google Gemini API (free tier: 15 RPM — will bottleneck on large datasets)
  - ollama : Local Ollama server on your own GPU/CPU — no rate limit

Features:
  - Async + concurrent requests (configurable CONCURRENCY)
  - Checkpointing: saves every SAVE_EVERY entries — resume-safe
  - Exponential backoff on rate limits (429) for Gemini / connection errors for Ollama
  - Works with ss9.json, ss10.json, bio9.json, or any dataset in the same format

Usage:
    # Gemini (needs GEMINI_API_KEY in .env)
    python training/enrich_hinglish.py --input data/ss9.json --backend gemini

    # Ollama (needs `ollama serve` running + model pulled: `ollama pull qwen3.5:4b`)
    python training/enrich_hinglish.py --input data/ss9.json --backend ollama --ollama-model qwen3.5:4b
"""

import asyncio
import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Config
PROJECT_ROOT  = Path(__file__).parent.parent
CONCURRENCY   = 2
SAVE_EVERY    = 20
GEMINI_MODEL  = "gemini-3.6-flash"
OLLAMA_MODEL  = "qwen3.5:4b"
OLLAMA_HOST   = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT = """You are a Class 9-12 Indian student who naturally speaks Hinglish — a fluid mix of Hindi
(Roman script) and English, the way students actually talk in class or on WhatsApp, not the way a
translator converts a textbook sentence.

Your job is NOT to translate word-for-word. Your job is to REBUILD the sentence the way a student would
actually say it out loud — casual, natural, often shorter than the English original, and NEVER mirroring
the English sentence's grammar or word order.

Rules:
- Keep technical / subject-specific terms in English (e.g. photosynthesis, democracy, GDP, tectonic plates).
- Use natural Hindi connectors and verbs (kya hai, batao, hota hai, isliye, kyunki, ke baare mein).
- Rebuild the sentence structure from scratch in natural spoken Hindi-English order — don't keep the
  English clause order and just swap words.
- If the English source has redundant or awkward phrasing (e.g. a chapter title repeated twice), drop the
  redundancy instead of carrying it over literally — say it the way a student actually would, once.
- Keep it conversational and reasonably short — a student doesn't ask questions in the same wordy, formal
  register as a textbook.
- If you are not fully confident of the correct Hindi word for something, just say it in English instead.
  Real Hinglish speakers do this constantly — "important", "actually", "obviously" are almost always said
  in English, not forced into Hindi. A natural English word beats a wrong or made-up Hindi word every time.
- Spell any English words you use correctly — don't misspell them.
- Only rephrase what is actually in the English source. Do not add explanations, reasoning, or extra
  clauses that aren't there, even if they'd sound natural.
- Specific words that are easy to get wrong — use these exact translations:
  - "society" / "societies" = "samaj" / "samajon" — NEVER "jagah" (place), which is a different word.
  - "passed down" (as in traditions passed down through generations) has NOTHING to do with "pasand"
    (which means "like/preference"). Say "peedhi dar peedhi chalte aaye hain" or similar instead.
- Output ONLY the Hinglish text. No explanations, no quotes, no labels.

Example — BAD (literal word-substitution, do NOT do this):
English: "Describe in detail: Chapter Understanding Social Science In Grades 6 to 8, we have explored
Social Science through stories of people, places, and events."
Bad: "Chapter Understanding Social Science me, 6th to 8th ke students ne logon, jagahon aur incidenton ke
kahaniyon ke through Social Science ko samajhaya hai, isliye is chapter ko detail me batao?"

Example — GOOD (natural rephrasing, DO this):
Good: "Understanding Social Science chapter ko detail mein explain karo."

Example — BAD:
English: "Give an example of Social Science as mentioned in the chapter."
Bad: "Social Science ke chapter mein diye gaye example ka ek dalo."

Example — GOOD:
Good: "Is chapter mein Social Science ka ek example do na.\""""

Q_PROMPT = """Rephrase this as how a Class 9-12 student would naturally ASK this question out loud in
Hinglish. Don't translate it — rebuild it.

English question: {q_en}
Chapter context: {chapter} ({subject})

Output only the Hinglish question."""

A_PROMPT = """Rephrase this as how a teacher would naturally EXPLAIN this to a student in conversational
Hinglish. Don't translate sentence-by-sentence — restructure it so it sounds spoken, not textbook.
Keep technical terms in English. Use natural Hindi grammar/connectors.

English answer: {a_en}

Output only the Hinglish answer."""

STATEMENT_PROMPT = """This is a STATEMENT (a claim to judge), not a question — do not turn it into a
question. Rephrase it as how a student would naturally STATE this claim out loud in Hinglish, keeping
it a statement. Don't add words that change the meaning — just restate it naturally.

English statement: {stmt_en}
Chapter context: {chapter} ({subject})

Output only the Hinglish statement. No question mark, no "true ya false" — that gets added separately."""


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

def init_gemini_client(model_name: str):
    try:
        import google.generativeai as genai
    except ImportError:
        print("[ERROR] google-generativeai not installed. Run:")
        print("        pip install google-generativeai")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY is not set.")
        print("        Option 1: set it in your terminal:")
        print("            PowerShell: $env:GEMINI_API_KEY='your-key-here'")
        print("            Bash: export GEMINI_API_KEY='your-key-here'")
        print("        Option 2: create a .env file in the project root:")
        print("            GEMINI_API_KEY=your-key-here")
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_PROMPT)


async def gemini_generate(model, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 2048},
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = 2 ** attempt * 5
                print(f"  [rate-limit] waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  [API error] {err}")
                return ""
    return ""


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def check_ollama_ready(host: str, model_name: str):
    if requests is None:
        print("[ERROR] `requests` not installed. Run:")
        print("        pip install requests")
        sys.exit(1)
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception:
        print(f"[ERROR] Can't reach Ollama at {host}.")
        print("        Make sure Ollama is running (it usually starts automatically,")
        print("        or run `ollama serve` in a separate terminal).")
        sys.exit(1)

    names = [m.get("name", "") for m in r.json().get("models", [])]
    if not any(model_name.split(":")[0] in n for n in names):
        print(f"[ERROR] Model '{model_name}' not found locally.")
        print(f"        Pull it first:  ollama pull {model_name}")
        sys.exit(1)


def _ollama_generate_sync(host: str, model_name: str, prompt: str, timeout: int, num_ctx: int, num_predict: int) -> str:
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            # Reasoning models (qwen3, deepseek-r1, etc.) default to thinking mode.
            # On /api/generate, think:false is silently ignored for some of these
            # models and the reasoning trace either leaks into the output or eats
            # the whole num_predict budget, leaving an EMPTY response. /api/chat with
            # think as a top-level field is the combination that reliably disables it.
            "think": False,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": num_predict, "num_ctx": num_ctx},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


async def ollama_generate(host: str, model_name: str, prompt: str, timeout: int, num_ctx: int, num_predict: int, retries: int = 2) -> str:
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(_ollama_generate_sync, host, model_name, prompt, timeout, num_ctx, num_predict)
        except Exception as e:
            print(f"  [ollama error] {e} — retrying...")
            await asyncio.sleep(2 * (attempt + 1))
    return ""


# ---------------------------------------------------------------------------
# Shared enrichment logic
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, data) -> None:
    """Write to a temp file then swap it in, so a crash mid-write never leaves
    the main file half-written (this is what caused the JSON corruption)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def build_messages_hinglish(entry: dict) -> list:
    """Chat-format mirror of `messages`, with user/assistant content in Hinglish.
    System/context stays in English — it's the grounding passage, not something
    that needs to be produced in Hinglish."""
    return [
        {"role": "system", "content": entry.get("ncert_context", "")},
        {"role": "user", "content": entry.get("question_hinglish", "")},
        {"role": "assistant", "content": entry.get("answer_hinglish", "")},
    ]


def strip_chapter_prefix(text: str, chapter: str) -> str:
    """Some upstream entries prepend 'Chapter <chapter title>' verbatim to the start
    of question_english/answer_english (e.g. 'Chapter Understanding Social Science In
    Grades 6 to 8...'). That redundancy, translated literally, is exactly the kind of
    awkward phrasing we don't want in Hinglish output — so strip it before prompting."""
    if not text or not chapter:
        return text
    prefix = f"Chapter {chapter}"
    if text.lower().startswith(prefix.lower()):
        return text[len(prefix):].lstrip(" :,-")
    return text


TRUE_FALSE_PREFIXES = ("true or false:", "true or false -", "true/false:")


def split_true_false(text: str):
    """'True or False: <statement>' is a claim to judge, not a question. Asking the
    model to rephrase it 'as a question a student would ask' makes it try to force a
    declarative statement into question-shape, which is exactly what scrambles it.
    Returns (statement_without_prefix, is_true_false)."""
    stripped = text.strip()
    lower = stripped.lower()
    for p in TRUE_FALSE_PREFIXES:
        if lower.startswith(p):
            return stripped[len(p):].strip(" :-"), True
    return text, False


async def enrich_entry(sem, ctx: dict, entry: dict) -> dict:
    async with sem:
        chapter = entry.get("chapter", "")
        subject = entry.get("subject", "")

        q_clean = strip_chapter_prefix(entry["question_english"], chapter)
        a_clean = strip_chapter_prefix(entry["answer_english"], chapter)[:3000]
        q_statement, is_true_false = split_true_false(q_clean)

        # Reuse an existing translation if we've seen this exact source text before —
        # duplicate source paragraphs (same content, different questions) otherwise get
        # re-translated independently each time, wasting calls and giving inconsistent
        # quality on identical content.
        q_cache_key = ("tf", q_statement) if is_true_false else ("q", q_clean)
        a_cache_key = a_clean

        cached_q = ctx["q_cache"].get(q_cache_key)
        cached_a = ctx["a_cache"].get(a_cache_key)

        if cached_q is not None:
            q_hi = cached_q
        else:
            if is_true_false:
                q_prompt = STATEMENT_PROMPT.format(stmt_en=q_statement, chapter=chapter, subject=subject)
            else:
                q_prompt = Q_PROMPT.format(q_en=q_clean, chapter=chapter, subject=subject)
            if ctx["backend"] == "ollama":
                q_hi = await ollama_generate(ctx["host"], ctx["model_name"], q_prompt, ctx["timeout"], ctx["num_ctx"], ctx["num_predict"])
            else:
                q_hi = await gemini_generate(ctx["model"], q_prompt)
            if is_true_false and q_hi:
                q_hi = f"{q_hi} — Sahi ya galat?"
            if q_hi:
                ctx["q_cache"][q_cache_key] = q_hi

        if cached_a is not None:
            a_hi = cached_a
        else:
            a_prompt = A_PROMPT.format(a_en=a_clean)
            if ctx["backend"] == "ollama":
                a_hi = await ollama_generate(ctx["host"], ctx["model_name"], a_prompt, ctx["timeout"], ctx["num_ctx"], ctx["num_predict"])
            else:
                a_hi = await gemini_generate(ctx["model"], a_prompt)
            if a_hi:
                ctx["a_cache"][a_cache_key] = a_hi

        entry["question_hinglish"] = q_hi
        entry["answer_hinglish"] = a_hi
        entry["messages_hinglish"] = build_messages_hinglish(entry)
        return entry


async def run(input_path: Path, backend: str, concurrency: int, save_every: int,
              model_name: str, host: str, ollama_timeout: int, num_ctx: int, num_predict: int, limit: int = None):
    ctx = {"backend": backend, "model_name": model_name, "host": host, "model": None,
           "timeout": ollama_timeout, "num_ctx": num_ctx, "num_predict": num_predict,
           "q_cache": {}, "a_cache": {}}

    if backend == "ollama":
        check_ollama_ready(host, model_name)
    else:
        ctx["model"] = init_gemini_client(model_name)

    print(f"\n[EduHinglish Enrichment] -> {input_path.name}")
    print(f"  Backend    : {backend}")
    print(f"  Model      : {model_name}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Checkpoint : every {save_every} entries\n")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    total = len(data)

    # Seed the reuse cache from entries already translated in past runs, so identical
    # source text (many entries share the same source paragraph) doesn't get
    # re-translated — and re-rolled with different quality each time.
    for e in data:
        chapter = e.get("chapter", "")
        if e.get("question_hinglish"):
            q_clean = strip_chapter_prefix(e["question_english"], chapter)
            q_stmt, is_tf = split_true_false(q_clean)
            key = ("tf", q_stmt) if is_tf else ("q", q_clean)
            ctx["q_cache"].setdefault(key, e["question_hinglish"])
        if e.get("answer_hinglish"):
            a_clean = strip_chapter_prefix(e["answer_english"], chapter)[:3000]
            ctx["a_cache"].setdefault(a_clean, e["answer_hinglish"])

    print(f"  Reuse cache seeded: {len(ctx['q_cache'])} unique questions, {len(ctx['a_cache'])} unique answers")

    # Backfill messages_hinglish for entries enriched before this field existed —
    # no API calls needed, we already have the Hinglish text.
    backfilled = 0
    for e in data:
        if e.get("question_hinglish") and e.get("answer_hinglish") and not e.get("messages_hinglish"):
            e["messages_hinglish"] = build_messages_hinglish(e)
            backfilled += 1
    if backfilled:
        print(f"  Backfilled messages_hinglish for {backfilled} already-enriched entries (no API calls)")
        atomic_write_json(input_path, data)

    pending_idx = [
        i for i, e in enumerate(data)
        if not e.get("question_hinglish") or not e.get("answer_hinglish")
    ]
    print(f"  Total entries : {total}")
    print(f"  Already filled: {total - len(pending_idx)}")

    if limit and limit > 0:
        pending_idx = pending_idx[:limit]
        print(f"  To process    : {len(pending_idx)} (limited to {limit})\n")
    else:
        print(f"  To process    : {len(pending_idx)}\n")

    if not pending_idx:
        print("All entries already enriched. Nothing to do.")
        return

    sem = asyncio.Semaphore(concurrency)
    done = 0
    t_start = time.time()

    for batch_start in range(0, len(pending_idx), save_every):
        batch = pending_idx[batch_start: batch_start + save_every]
        tasks = [enrich_entry(sem, ctx, data[i]) for i in batch]
        results = await asyncio.gather(*tasks)

        for i, enriched in zip(batch, results):
            data[i] = enriched
            done += 1

        atomic_write_json(input_path, data)

        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        remain = (len(pending_idx) - done) / rate if rate > 0 else 0
        print(
            f"  [{done}/{len(pending_idx)}] checkpoint saved"
            f"  |  {rate:.1f} entries/s"
            f"  |  ~{remain / 60:.1f} min remaining"
        )

    print(f"\nDone! Enriched {done} entries -> {input_path}")


def main():
    parser = argparse.ArgumentParser(description="EduHinglish - Hinglish Enrichment (Gemini or Ollama)")
    parser.add_argument("--input", required=True, help="Path to dataset JSON (e.g. data/ss9.json)")
    parser.add_argument("--backend", choices=["gemini", "ollama"], default="gemini",
                         help="Which backend to use (default: gemini)")
    parser.add_argument("--model", type=str, default=None,
                         help=f"Model name. Defaults: gemini -> {GEMINI_MODEL}, ollama -> {OLLAMA_MODEL}")
    parser.add_argument("--ollama-host", type=str, default=OLLAMA_HOST, help="Ollama server URL")
    parser.add_argument("--ollama-timeout", type=int, default=300,
                         help="Seconds to wait for a single Ollama response before retrying (default: 300)")
    parser.add_argument("--num-ctx", type=int, default=4096,
                         help="Ollama context window size (default: 4096). Lower reduces per-request VRAM use.")
    parser.add_argument("--num-predict", type=int, default=2048,
                         help="Max output tokens per generation (default: 2048). Lower reduces per-request VRAM use.")
    parser.add_argument("--concurrency", type=int, default=None,
                         help="Concurrency limit (default: 2 for gemini, 1 for ollama — a single local GPU "
                              "doesn't benefit from parallel requests)")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY,
                         help=f"Save checkpoint every N entries (default: {SAVE_EVERY})")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of entries to process (useful for testing)")
    args = parser.parse_args()
    inpath = PROJECT_ROOT / args.input

    if not inpath.exists():
        print(f"[ERROR] File not found: {inpath}")
        sys.exit(1)

    model_name = args.model or (OLLAMA_MODEL if args.backend == "ollama" else GEMINI_MODEL)
    concurrency = args.concurrency if args.concurrency is not None else (1 if args.backend == "ollama" else CONCURRENCY)

    asyncio.run(run(inpath, args.backend, concurrency, args.save_every, model_name, args.ollama_host,
                    args.ollama_timeout, args.num_ctx, args.num_predict, args.limit))


if __name__ == "__main__":
    main()