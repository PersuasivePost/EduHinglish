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


def _ollama_generate_sync(host: str, model_name: str, prompt: str) -> str:
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
            "options": {"temperature": 0.3, "num_predict": 2048, "num_ctx": 4096},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


async def ollama_generate(host: str, model_name: str, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(_ollama_generate_sync, host, model_name, prompt)
        except Exception as e:
            print(f"  [ollama error] {e} — retrying...")
            await asyncio.sleep(2 * (attempt + 1))
    return ""


# ---------------------------------------------------------------------------
# Shared enrichment logic
# ---------------------------------------------------------------------------

async def enrich_entry(sem, ctx: dict, entry: dict) -> dict:
    async with sem:
        q_prompt = Q_PROMPT.format(
            q_en=entry["question_english"],
            chapter=entry.get("chapter", ""),
            subject=entry.get("subject", ""),
        )
        a_prompt = A_PROMPT.format(a_en=entry["answer_english"][:3000])

        if ctx["backend"] == "ollama":
            q_hi = await ollama_generate(ctx["host"], ctx["model_name"], q_prompt)
            a_hi = await ollama_generate(ctx["host"], ctx["model_name"], a_prompt)
        else:
            q_hi = await gemini_generate(ctx["model"], q_prompt)
            a_hi = await gemini_generate(ctx["model"], a_prompt)

        entry["question_hinglish"] = q_hi
        entry["answer_hinglish"] = a_hi
        return entry


async def run(input_path: Path, backend: str, concurrency: int, save_every: int,
              model_name: str, host: str, limit: int = None):
    ctx = {"backend": backend, "model_name": model_name, "host": host, "model": None}

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

        input_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

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

    asyncio.run(run(inpath, args.backend, concurrency, args.save_every, model_name, args.ollama_host, args.limit))


if __name__ == "__main__":
    main()