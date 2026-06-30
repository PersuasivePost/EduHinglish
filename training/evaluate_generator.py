"""
EduHinglish — Phase 6 Generator Evaluation
============================================
Author  : Jatin
Module  : Evaluation — IndicBART Hinglish Generator
Purpose : Evaluate the fine-tuned IndicBART generator quality using automated
          metrics (BLEU, ROUGE, CMI, length ratio) and human-readable sample
          outputs. Supports comparison between base and fine-tuned models.

Usage:
    python training/evaluate_generator.py                          # full evaluation
    python training/evaluate_generator.py --samples 20             # show 20 samples
    python training/evaluate_generator.py --model models/indicbart_v1  # custom model path
    python training/evaluate_generator.py --compare-base           # also evaluate base model
"""

from __future__ import annotations

import json
import re
import argparse
import time
from pathlib import Path
from collections import Counter, defaultdict

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "unified_biology_dataset.json"
SEQ2SEQ_DIR  = PROJECT_ROOT / "training" / "data" / "indicbart_seq2seq_dataset"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
DEFAULT_MODEL = "models/indicbart_v1"
SEED = 42


# ═════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═════════════════════════════════════════════════════════════════════════════

def _thick():
    print("  " + "═" * 60)


def _thin():
    print("  " + "─" * 60)


# ── BLEU ──────────────────────────────────────────────────────────────────────

def compute_bleu_scores(references: list[str], hypotheses: list[str]) -> dict:
    """
    Compute corpus-level and per-sentence BLEU scores.

    Tries sacrebleu first, falls back to nltk.translate.bleu_score.

    Returns:
        dict with "corpus_bleu", "per_sentence" (list of floats).
    """
    per_sentence = []

    try:
        import sacrebleu

        # Corpus-level BLEU
        corpus = sacrebleu.corpus_bleu(hypotheses, [references])
        corpus_bleu = round(corpus.score, 2)

        # Per-sentence BLEU
        for ref, hyp in zip(references, hypotheses):
            sent = sacrebleu.sentence_bleu(hyp, [ref])
            per_sentence.append(round(sent.score, 2))

    except ImportError:
        from nltk.translate.bleu_score import (
            sentence_bleu,
            corpus_bleu as nltk_corpus_bleu,
            SmoothingFunction,
        )

        smoothing = SmoothingFunction().method1

        ref_tokenized = [r.split() for r in references]
        hyp_tokenized = [h.split() for h in hypotheses]

        # Corpus-level BLEU
        corpus_bleu = round(
            nltk_corpus_bleu(
                [[r] for r in ref_tokenized],
                hyp_tokenized,
                smoothing_function=smoothing,
            ) * 100,
            2,
        )

        # Per-sentence BLEU
        for ref_tok, hyp_tok in zip(ref_tokenized, hyp_tokenized):
            score = sentence_bleu(
                [ref_tok], hyp_tok, smoothing_function=smoothing
            )
            per_sentence.append(round(score * 100, 2))

    return {"corpus_bleu": corpus_bleu, "per_sentence": per_sentence}


# ── ROUGE ─────────────────────────────────────────────────────────────────────

def compute_rouge_scores(references: list[str], hypotheses: list[str]) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Returns:
        dict with "rouge1", "rouge2", "rougeL" (each an F1 float),
        and "per_sentence" list of dicts.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print(f"  {Fore.YELLOW}[WARN] rouge-score not installed. "
              f"Run: pip install rouge-score")
        return {
            "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
            "per_sentence": [],
        }

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )

    r1_scores, r2_scores, rl_scores = [], [], []
    per_sentence = []

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        r1 = round(scores["rouge1"].fmeasure * 100, 2)
        r2 = round(scores["rouge2"].fmeasure * 100, 2)
        rl = round(scores["rougeL"].fmeasure * 100, 2)

        r1_scores.append(r1)
        r2_scores.append(r2)
        rl_scores.append(rl)
        per_sentence.append({"rouge1": r1, "rouge2": r2, "rougeL": rl})

    return {
        "rouge1": round(sum(r1_scores) / max(len(r1_scores), 1), 2),
        "rouge2": round(sum(r2_scores) / max(len(r2_scores), 1), 2),
        "rougeL": round(sum(rl_scores) / max(len(rl_scores), 1), 2),
        "per_sentence": per_sentence,
    }


# ── Code-Mixing Index (CMI) ──────────────────────────────────────────────────

# Inline word-level LID for CMI calculation (mirrors scripts/validate_dataset.py
# and src/script_detector.py logic, without importing the full module)

_HINDI_WORDS = {
    "hai", "hain", "hota", "hoti", "hote", "karta", "karti", "karte",
    "ka", "ki", "ke", "ko", "se", "mein", "par", "pe", "ne",
    "ek", "yeh", "woh", "jo", "jab", "tab", "kya", "kaise", "kahan",
    "kaun", "kitna", "kitni", "kitne", "aur", "ya", "lekin", "agar",
    "toh", "bhi", "hi", "sirf", "iske", "uske", "unke", "isme", "usme",
    "jisme", "inhe", "unhe", "isko", "usko", "apna", "apni", "apne",
    "bahut", "thoda", "zyada", "kam", "achha", "bura", "bada", "bade",
    "badi", "chhota", "naya", "nayi", "pehle", "baad", "andar", "bahar",
    "upar", "neeche", "yahan", "wahan", "abhi", "nahi", "nhi", "na",
    "mat", "sabhi", "sab", "kuch", "koi", "wala", "wale", "wali",
    "jaise", "tarah", "matlab", "yaani", "saath", "taraf", "beech",
    "kaam", "cheez", "jagah", "tarika", "hota", "hoti", "hote",
    "kehte", "kehta", "kehti", "bolte", "bolta", "bolti",
    "samajh", "samjho", "samjhao", "batao", "padhai", "padhte",
    "milta", "milti", "milte", "dete", "deta", "deti",
    "rakhte", "rakhta", "rakhti", "karte", "karti", "karta",
}

_SCIENCE_TERMS = {
    "cell", "cells", "membrane", "membranes", "nucleus", "chromosome",
    "chromosomes", "gene", "genes", "dna", "rna", "protein", "proteins",
    "mitochondria", "ribosome", "ribosomes", "lysosome", "lysosomes",
    "endoplasmic", "reticulum", "golgi", "apparatus", "vacuole", "vacuoles",
    "plastid", "plastids", "chloroplast", "chloroplasts", "chlorophyll",
    "osmosis", "diffusion", "photosynthesis", "respiration", "atp",
    "enzyme", "enzymes", "glucose", "oxygen", "carbon", "dioxide",
    "nitrogen", "hydrogen", "energy", "organism", "organisms",
    "tissue", "tissues", "organ", "organs", "heredity", "reproduction",
    "stomata", "xylem", "phloem", "neuron", "neurons", "synapse",
    "hormone", "hormones", "ecosystem", "environment", "biodegradable",
}

_EN_SUFFIXES = (
    "tion", "sion", "ness", "ment", "able", "ible", "ous", "ive",
    "ful", "less", "ing", "ed", "er", "est", "ly", "al", "ity",
    "ence", "ance", "ism", "ist", "ize", "ise",
)


def _identify_word(word: str) -> str:
    """Lightweight word-level language ID (mirrors src/script_detector.py)."""
    clean = re.sub(r"[^\w]", "", word)
    if not clean:
        return "PUNCT"

    has_dev = any("\u0900" <= c <= "\u097F" for c in clean)
    has_rom = any(c.isascii() and c.isalpha() for c in clean)
    if has_dev and has_rom:
        return "MIX"
    if has_dev:
        return "HI"

    lower = clean.lower()

    if lower.isdigit():
        return "UNIV"
    if lower in _SCIENCE_TERMS:
        return "EN"
    if lower in _HINDI_WORDS:
        return "HI"

    if len(lower) > 4:
        for sfx in _EN_SUFFIXES:
            if lower.endswith(sfx):
                return "EN"

    return "EN"


def compute_cmi(text: str) -> float:
    """
    Compute Code-Mixing Index for a single text string.

    CMI = (1 - max(HI, EN) / (HI + EN)) * 100
    Returns 0.0 if no HI/EN words found.
    """
    words = text.split()
    labels = [_identify_word(w) for w in words]

    hi = sum(1 for l in labels if l == "HI")
    en = sum(1 for l in labels if l == "EN")
    total = hi + en

    if total == 0:
        return 0.0

    return round((1 - max(hi, en) / total) * 100, 1)


def compute_cmi_batch(texts: list[str]) -> dict:
    """
    Compute CMI statistics for a batch of texts.

    Returns:
        dict with "mean_cmi", "min_cmi", "max_cmi", "per_text" list.
    """
    cmi_values = [compute_cmi(t) for t in texts]
    return {
        "mean_cmi": round(sum(cmi_values) / max(len(cmi_values), 1), 1),
        "min_cmi": round(min(cmi_values), 1) if cmi_values else 0.0,
        "max_cmi": round(max(cmi_values), 1) if cmi_values else 0.0,
        "per_text": cmi_values,
    }


# ── Length ratio ──────────────────────────────────────────────────────────────

def compute_length_ratio(references: list[str], hypotheses: list[str]) -> dict:
    """
    Compute avg length ratio: avg(len(generated)) / avg(len(reference)).

    Returns:
        dict with "ratio", "avg_ref_len", "avg_gen_len".
    """
    ref_lens = [len(r.split()) for r in references]
    gen_lens = [len(h.split()) for h in hypotheses]

    avg_ref = sum(ref_lens) / max(len(ref_lens), 1)
    avg_gen = sum(gen_lens) / max(len(gen_lens), 1)
    ratio = avg_gen / avg_ref if avg_ref > 0 else 0.0

    return {
        "ratio": round(ratio, 3),
        "avg_ref_words": round(avg_ref, 1),
        "avg_gen_words": round(avg_gen, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_dev_split() -> list[dict]:
    """
    Load the dev split for evaluation.

    Strategy:
      1. Try loading pre-split dev set from training/data/indicbart_seq2seq_dataset/
      2. Fall back to loading full dataset and recreating the split (SEED=42)
    """
    # Attempt 1: Pre-split dev set
    dev_file = SEQ2SEQ_DIR / "dev.json"
    if dev_file.exists():
        print(f"  Loading dev split from: {dev_file}")
        with open(dev_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  {Fore.GREEN}[OK] Loaded {len(data)} dev examples from pre-split file")
        return data

    # Attempt 2: Recreate split from full dataset
    if not DATASET_PATH.exists():
        print(f"  {Fore.RED}[ERROR] Dataset not found: {DATASET_PATH}")
        print(f"         Run scripts/validate_dataset.py or ensure dataset exists.")
        return []

    print(f"  Pre-split dev set not found at: {SEQ2SEQ_DIR}")
    print(f"  Recreating split from: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    import random
    rng = random.Random(SEED)
    rng.shuffle(full_data)

    # 80/10/10 split (matching training setup)
    n = len(full_data)
    train_end = int(0.8 * n)
    dev_end = int(0.9 * n)
    dev_data = full_data[train_end:dev_end]

    print(f"  {Fore.GREEN}[OK] Created dev split: {len(dev_data)} examples "
          f"(from {n} total, SEED={SEED})")
    return dev_data


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation runner
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model_path: str,
    dev_data: list[dict],
    use_lora: bool = True,
    label: str = "Fine-tuned",
) -> dict:
    """
    Run full evaluation pipeline on dev data.

    Args:
        model_path: Path to LoRA adapter weights (relative to project root).
        dev_data:   List of dataset entries with original_english and hinglish_roman.
        use_lora:   Whether to load LoRA adapter.
        label:      Label for this model in the report.

    Returns:
        dict with all metrics and per-example results.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from generator import HinglishGenerator

    print(f"\n  Loading model: {label} ({model_path})")
    generator = HinglishGenerator(
        model_path=model_path,
        use_lora=use_lora,
    )

    # ── Generate Hinglish for all dev entries ─────────────────────────────────
    references = []
    hypotheses = []
    per_example = []

    print(f"\n  Generating Hinglish for {len(dev_data)} dev examples...")
    start_time = time.time()

    for i, entry in enumerate(dev_data):
        english = entry.get("original_english", "")
        reference = entry.get("hinglish_roman", "")

        result = generator.generate(english)
        generated = result["output"]

        references.append(reference)
        hypotheses.append(generated)

        per_example.append({
            "id": entry.get("id", f"dev_{i:03d}"),
            "english": english,
            "reference": reference,
            "generated": generated,
            "chapter": entry.get("chapter", "Unknown"),
            "topic": entry.get("topic", "Unknown"),
        })

        # Progress indicator
        if (i + 1) % 25 == 0 or i == len(dev_data) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    [{i + 1}/{len(dev_data)}] "
                  f"({elapsed:.1f}s, {rate:.1f} examples/sec)")

    total_time = time.time() - start_time
    print(f"  {Fore.GREEN}[OK] Generation complete: {len(dev_data)} examples "
          f"in {total_time:.1f}s")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print(f"\n  Computing metrics...")

    bleu_results = compute_bleu_scores(references, hypotheses)
    rouge_results = compute_rouge_scores(references, hypotheses)
    ref_cmi = compute_cmi_batch(references)
    gen_cmi = compute_cmi_batch(hypotheses)
    length_ratio = compute_length_ratio(references, hypotheses)

    # Attach per-sentence scores to per_example
    for i, ex in enumerate(per_example):
        ex["bleu"] = bleu_results["per_sentence"][i] if i < len(bleu_results["per_sentence"]) else 0.0
        if rouge_results["per_sentence"] and i < len(rouge_results["per_sentence"]):
            ex["rouge"] = rouge_results["per_sentence"][i]
        else:
            ex["rouge"] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        ex["ref_cmi"] = ref_cmi["per_text"][i] if i < len(ref_cmi["per_text"]) else 0.0
        ex["gen_cmi"] = gen_cmi["per_text"][i] if i < len(gen_cmi["per_text"]) else 0.0

    # ── Per-chapter breakdown ─────────────────────────────────────────────────
    chapter_groups = defaultdict(list)
    for ex in per_example:
        chapter_groups[ex["chapter"]].append(ex)

    chapter_metrics = {}
    for chapter, examples in sorted(chapter_groups.items()):
        ch_refs = [ex["reference"] for ex in examples]
        ch_hyps = [ex["generated"] for ex in examples]
        ch_bleu = compute_bleu_scores(ch_refs, ch_hyps)
        ch_cmi = compute_cmi_batch(ch_hyps)

        chapter_metrics[chapter] = {
            "count": len(examples),
            "corpus_bleu": ch_bleu["corpus_bleu"],
            "mean_cmi": ch_cmi["mean_cmi"],
        }

    return {
        "label": label,
        "model_path": model_path,
        "use_lora": use_lora,
        "num_examples": len(dev_data),
        "generation_time_sec": round(total_time, 1),
        "metrics": {
            "bleu": {
                "corpus": bleu_results["corpus_bleu"],
                "mean_sentence": round(
                    sum(bleu_results["per_sentence"])
                    / max(len(bleu_results["per_sentence"]), 1), 2
                ),
            },
            "rouge": {
                "rouge1": rouge_results["rouge1"],
                "rouge2": rouge_results["rouge2"],
                "rougeL": rouge_results["rougeL"],
            },
            "cmi": {
                "reference_mean": ref_cmi["mean_cmi"],
                "generated_mean": gen_cmi["mean_cmi"],
                "cmi_delta": round(abs(gen_cmi["mean_cmi"] - ref_cmi["mean_cmi"]), 1),
            },
            "length_ratio": length_ratio,
        },
        "chapter_metrics": chapter_metrics,
        "per_example": per_example,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Report printing
# ═════════════════════════════════════════════════════════════════════════════

def print_metrics_table(results: dict):
    """Print the overall metrics table."""
    m = results["metrics"]

    _thick()
    print(f"  Evaluation Results: {results['label']}")
    print(f"  Model: {results['model_path']} | "
          f"Examples: {results['num_examples']} | "
          f"Time: {results['generation_time_sec']}s")
    _thick()

    print(f"\n  {'Metric':<30} {'Score':>10}")
    _thin()

    # BLEU
    bleu_color = Fore.GREEN if m["bleu"]["corpus"] >= 15 else Fore.YELLOW if m["bleu"]["corpus"] >= 5 else Fore.RED
    print(f"  {'Corpus BLEU':<30} {bleu_color}{m['bleu']['corpus']:>9.2f}{Style.RESET_ALL}")
    print(f"  {'Mean Sentence BLEU':<30} {m['bleu']['mean_sentence']:>9.2f}")

    # ROUGE
    print(f"  {'ROUGE-1 (F1)':<30} {m['rouge']['rouge1']:>9.2f}")
    print(f"  {'ROUGE-2 (F1)':<30} {m['rouge']['rouge2']:>9.2f}")
    print(f"  {'ROUGE-L (F1)':<30} {m['rouge']['rougeL']:>9.2f}")

    # CMI
    ref_cmi = m["cmi"]["reference_mean"]
    gen_cmi = m["cmi"]["generated_mean"]
    cmi_delta = m["cmi"]["cmi_delta"]
    cmi_color = Fore.GREEN if cmi_delta <= 10 else Fore.YELLOW if cmi_delta <= 20 else Fore.RED
    print(f"  {'Reference CMI (avg)':<30} {ref_cmi:>9.1f}%")
    print(f"  {'Generated CMI (avg)':<30} {gen_cmi:>9.1f}%")
    print(f"  {'CMI Delta (|ref - gen|)':<30} {cmi_color}{cmi_delta:>9.1f}%{Style.RESET_ALL}")

    # Length ratio
    lr = m["length_ratio"]
    lr_color = Fore.GREEN if 0.8 <= lr["ratio"] <= 1.2 else Fore.YELLOW
    print(f"  {'Length Ratio (gen/ref)':<30} {lr_color}{lr['ratio']:>9.3f}{Style.RESET_ALL}")
    print(f"  {'Avg Reference Words':<30} {lr['avg_ref_words']:>9.1f}")
    print(f"  {'Avg Generated Words':<30} {lr['avg_gen_words']:>9.1f}")
    print()


def print_chapter_breakdown(results: dict):
    """Print per-chapter metrics breakdown."""
    _thin()
    print(f"  Per-Chapter Breakdown")
    _thin()

    cm = results["chapter_metrics"]

    # Header
    print(f"  {'Chapter':<42} {'N':>4} {'BLEU':>7} {'CMI':>6}")
    print(f"  {'-'*42} {'-'*4} {'-'*7} {'-'*6}")

    for chapter, metrics in sorted(cm.items()):
        # Truncate long chapter names
        ch_display = chapter[:40] + ".." if len(chapter) > 42 else chapter
        print(f"  {ch_display:<42} {metrics['count']:>4} "
              f"{metrics['corpus_bleu']:>6.1f} {metrics['mean_cmi']:>5.1f}%")
    print()


def print_samples(results: dict, num_samples: int = 10):
    """Print sample outputs for human inspection."""
    _thin()
    print(f"  Sample Outputs ({num_samples} examples)")
    _thin()

    examples = results["per_example"]

    # Pick evenly spaced samples
    step = max(1, len(examples) // num_samples)
    sample_indices = list(range(0, len(examples), step))[:num_samples]

    for idx in sample_indices:
        ex = examples[idx]
        bleu = ex.get("bleu", 0.0)
        bleu_color = Fore.GREEN if bleu >= 20 else Fore.YELLOW if bleu >= 5 else Fore.RED

        print(f"\n  [{ex['id']}] Chapter: {ex['chapter']}")
        print(f"  {Fore.BLUE}English:{Style.RESET_ALL}")
        print(f"    \"{ex['english'][:150]}{'...' if len(ex['english']) > 150 else ''}\"")
        print(f"  {Fore.MAGENTA}Reference Hinglish:{Style.RESET_ALL}")
        print(f"    \"{ex['reference'][:150]}{'...' if len(ex['reference']) > 150 else ''}\"")
        print(f"  {Fore.CYAN}Generated Hinglish:{Style.RESET_ALL}")
        print(f"    \"{ex['generated'][:150]}{'...' if len(ex['generated']) > 150 else ''}\"")
        print(f"  BLEU: {bleu_color}{bleu:.1f}{Style.RESET_ALL} | "
              f"CMI ref: {ex.get('ref_cmi', 0):.1f}% | "
              f"CMI gen: {ex.get('gen_cmi', 0):.1f}%")

    print()


def print_worst_examples(results: dict, num_worst: int = 5):
    """Print worst examples by BLEU for error analysis."""
    _thin()
    print(f"  Worst {num_worst} Examples (lowest BLEU) — Error Analysis")
    _thin()

    examples = sorted(results["per_example"], key=lambda x: x.get("bleu", 0.0))

    for i, ex in enumerate(examples[:num_worst]):
        bleu = ex.get("bleu", 0.0)

        print(f"\n  {Fore.RED}[{i + 1}] BLEU: {bleu:.1f}{Style.RESET_ALL} | "
              f"ID: {ex['id']} | Chapter: {ex['chapter']}")
        print(f"  English:   \"{ex['english'][:120]}{'...' if len(ex['english']) > 120 else ''}\"")
        print(f"  Reference: \"{ex['reference'][:120]}{'...' if len(ex['reference']) > 120 else ''}\"")
        print(f"  Generated: \"{ex['generated'][:120]}{'...' if len(ex['generated']) > 120 else ''}\"")

    print()


def print_comparison(results_finetuned: dict, results_base: dict):
    """Print side-by-side comparison of fine-tuned vs base model."""
    _thick()
    print(f"  Model Comparison: Fine-tuned vs Base IndicBART")
    _thick()

    m_ft = results_finetuned["metrics"]
    m_base = results_base["metrics"]

    print(f"\n  {'Metric':<30} {'Fine-tuned':>12} {'Base':>12} {'Delta':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")

    rows = [
        ("Corpus BLEU", m_ft["bleu"]["corpus"], m_base["bleu"]["corpus"]),
        ("ROUGE-1 (F1)", m_ft["rouge"]["rouge1"], m_base["rouge"]["rouge1"]),
        ("ROUGE-2 (F1)", m_ft["rouge"]["rouge2"], m_base["rouge"]["rouge2"]),
        ("ROUGE-L (F1)", m_ft["rouge"]["rougeL"], m_base["rouge"]["rougeL"]),
        ("Generated CMI", m_ft["cmi"]["generated_mean"], m_base["cmi"]["generated_mean"]),
        ("Length Ratio", m_ft["length_ratio"]["ratio"], m_base["length_ratio"]["ratio"]),
    ]

    for name, ft_val, base_val in rows:
        delta = ft_val - base_val
        delta_color = Fore.GREEN if delta > 0 else Fore.RED if delta < 0 else ""
        # For length ratio and CMI, closer to reference is better, so delta interpretation differs
        if name in ("Length Ratio", "Generated CMI"):
            delta_color = ""  # neutral for these metrics
        print(f"  {name:<30} {ft_val:>11.2f} {base_val:>11.2f} "
              f"{delta_color}{delta:>+9.2f}{Style.RESET_ALL}")

    print()


# ═════════════════════════════════════════════════════════════════════════════
# Save results
# ═════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, output_path: Path):
    """Save evaluation results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Make serializable (tuples → lists, etc.)
    serializable = json.loads(json.dumps(results, default=str, ensure_ascii=False))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"  {Fore.GREEN}[SAVED] Results → {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IndicBART Hinglish Generator (Phase 6)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Path to fine-tuned model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of sample outputs to display (default: 10)"
    )
    parser.add_argument(
        "--compare-base", action="store_true",
        help="Also evaluate base IndicBART (no LoRA) for comparison"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Custom output path for results JSON"
    )
    args = parser.parse_args()

    _thick()
    print(f"  EduHinglish — Phase 6 Generator Evaluation")
    _thick()
    print()

    # ── Load dev data ─────────────────────────────────────────────────────────
    dev_data = load_dev_split()
    if not dev_data:
        print(f"  {Fore.RED}[ERROR] No dev data loaded. Exiting.")
        return

    # ── Run fine-tuned model evaluation ───────────────────────────────────────
    results_ft = run_evaluation(
        model_path=args.model,
        dev_data=dev_data,
        use_lora=True,
        label="Fine-tuned IndicBART + LoRA",
    )

    print_metrics_table(results_ft)
    print_chapter_breakdown(results_ft)
    print_samples(results_ft, num_samples=args.samples)
    print_worst_examples(results_ft)

    # ── Optionally run base model evaluation ──────────────────────────────────
    results_base = None
    if args.compare_base:
        results_base = run_evaluation(
            model_path=args.model,
            dev_data=dev_data,
            use_lora=False,
            label="Base IndicBART (no fine-tuning)",
        )

        print_metrics_table(results_base)
        print_comparison(results_ft, results_base)

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = Path(args.output) if args.output else (
        OUTPUT_DIR / "generator_eval_results.json"
    )

    save_payload = {
        "finetuned": results_ft,
    }
    if results_base:
        save_payload["base"] = results_base

    save_results(save_payload, output_path)

    _thick()
    print(f"  {Fore.GREEN}[OK] Evaluation complete")
    _thick()


if __name__ == "__main__":
    main()
