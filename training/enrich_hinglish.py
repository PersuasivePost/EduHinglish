"""
EduHinglish — Hinglish Enrichment via Gemini API
=================================================
Fills `question_hinglish` and `answer_hinglish` for entries where they are blank.

Features:
  - Async + concurrent requests (configurable CONCURRENCY)
  - Checkpointing: saves every SAVE_EVERY entries — resume-safe
  - Exponential backoff on rate limits (429)
  - Works with ss9.json, ss10.json, bio9.json, or any dataset in the same format

Usage:
    export GEMINI_API_KEY="your-key-here"
    python3 training/enrich_hinglish.py --input data/ss9.json
    python3 training/enrich_hinglish.py --input data/ss10.json --concurrency 10
"""

import asyncio
import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("[ERROR] google-generativeai not installed. Run:")
    print("        pip install google-generativeai")
    sys.exit(1)

# Config
PROJECT_ROOT = Path(__file__).parent.parent
CONCURRENCY  = 8
SAVE_EVERY   = 100
MODEL_NAME   = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are an expert at writing natural Hinglish — a mix of Hindi (Roman script) and English,
as spoken by Indian school students (Class 6-12).

Rules:
- Keep technical / subject-specific terms in English (e.g. photosynthesis, democracy, GDP, tectonic plates).
- Use Hindi grammar words, connectors, and verbs (e.g. "kya hai", "batao", "hota hai", "isliye", "kyunki").
- Keep it natural, conversational, and how a student would actually ask or answer in class.
- Do NOT transliterate every word — only replace common English grammar words with Hindi equivalents.
- Output ONLY the requested Hinglish text. No explanations, no quotes, no labels."""

Q_PROMPT = """Convert this English question to natural Hinglish (Roman script, Hindi+English mix):

English question: {q_en}
Chapter context: {chapter} ({subject})

Output only the Hinglish question."""

A_PROMPT = """Convert this English answer to natural Hinglish (Roman script, Hindi+English mix).
Keep technical terms in English. Use Hindi grammar/connectors.

English answer: {a_en}

Output only the Hinglish answer."""


def init_client():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable not set.")
        print("        export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
    )


async def generate(model, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 512},
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


async def enrich_entry(sem, model, entry: dict) -> dict:
    async with sem:
        q_hi = await generate(
            model,
            Q_PROMPT.format(
                q_en=entry["question_english"],
                chapter=entry.get("chapter", ""),
                subject=entry.get("subject", ""),
            ),
        )
        a_hi = await generate(
            model,
            A_PROMPT.format(a_en=entry["answer_english"][:1200]),
        )
        entry["question_hinglish"] = q_hi
        entry["answer_hinglish"]   = a_hi
        return entry


async def run(input_path: Path, concurrency: int, save_every: int):
    model = init_client()
    print(f"\n[EduHinglish Enrichment] -> {input_path.name}")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Checkpoint : every {save_every} entries\n")

    data  = json.loads(input_path.read_text(encoding="utf-8"))
    total = len(data)

    pending_idx = [
        i for i, e in enumerate(data)
        if not e.get("question_hinglish") or not e.get("answer_hinglish")
    ]
    print(f"  Total entries : {total}")
    print(f"  Already filled: {total - len(pending_idx)}")
    print(f"  To process    : {len(pending_idx)}\n")

    if not pending_idx:
        print("All entries already enriched. Nothing to do.")
        return

    sem     = asyncio.Semaphore(concurrency)
    done    = 0
    t_start = time.time()

    for batch_start in range(0, len(pending_idx), save_every):
        batch   = pending_idx[batch_start : batch_start + save_every]
        tasks   = [enrich_entry(sem, model, data[i]) for i in batch]
        results = await asyncio.gather(*tasks)

        for i, enriched in zip(batch, results):
            data[i] = enriched
            done   += 1

        input_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        elapsed = time.time() - t_start
        rate    = done / elapsed if elapsed > 0 else 0
        remain  = (len(pending_idx) - done) / rate if rate > 0 else 0
        print(
            f"  [{done}/{len(pending_idx)}] checkpoint saved"
            f"  |  {rate:.1f} entries/s"
            f"  |  ~{remain/60:.1f} min remaining"
        )

    print(f"\nDone! Enriched {done} entries -> {input_path}")


def main():
    parser = argparse.ArgumentParser(description="EduHinglish - Gemini Hinglish Enrichment")
    parser.add_argument("--input", required=True, help="Path to dataset JSON (e.g. data/ss9.json)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--save-every",  type=int, default=SAVE_EVERY)
    args   = parser.parse_args()
    inpath = PROJECT_ROOT / args.input

    if not inpath.exists():
        print(f"[ERROR] File not found: {inpath}")
        sys.exit(1)

    asyncio.run(run(inpath, args.concurrency, args.save_every))


if __name__ == "__main__":
    main()
