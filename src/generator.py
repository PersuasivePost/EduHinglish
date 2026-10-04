"""
EduHinglish — Hinglish Answer Generator
=========================================
Author  : Jatin
Module  : M3 — Hinglish Answer Generator (Phase 6)
Purpose : Inference module that loads the fine-tuned IndicBART model and
          generates Hinglish explanations from English NCERT text + student
          queries. Integrates with M1 (preprocessing) and M2 (retriever)
          to form the complete M1→M2→M3 pipeline.

Usage:
    python src/generator.py
    # Runs a quick generation test with sample NCERT sentences.
"""

import time
import warnings
from pathlib import Path
from colorama import Fore, Style, init as colorama_init

# Suppress transformers deprecation: IndicBART config uses old `use_return_dict`
warnings.filterwarnings(
    "ignore",
    message=".*use_return_dict.*",
    category=FutureWarning,
)

colorama_init(autoreset=True)


class HinglishGenerator:
    """
    M3 — Hinglish Answer Generator

    Loads fine-tuned IndicBART and generates Hinglish explanations
    from NCERT English text. Supports LoRA adapter loading when
    fine-tuned weights are available, with automatic fallback to
    the base model.
    """

    def __init__(
        self,
        model_path: str = "models/indicbart_v1",
        base_model: str = "ai4bharat/IndicBART",
        device: str = None,
        use_lora: bool = True,
    ):
        """
        Load model + tokenizer.

        Args:
            model_path: Path to LoRA adapter weights directory.
            base_model: HuggingFace model ID for IndicBART base.
            device:     'cuda', 'cpu', or None for auto-detect.
            use_lora:   If True, attempt to load LoRA adapter from model_path.
        """
        print(f"\n  Loading Hinglish generator: {base_model}")

        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch not installed. Run:\n"
                "  pip install torch"
            )

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            raise ImportError(
                "transformers not installed. Run:\n"
                "  pip install transformers"
            )

        # ── Device selection ──────────────────────────────────────────────────
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # ── Load tokenizer ────────────────────────────────────────────────────
        self.base_model_name = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # ── Load model (base + optional LoRA) ─────────────────────────────────
        self.lora_loaded = False

        # Resolve model_path relative to project root
        project_root = Path(__file__).parent.parent
        adapter_path = project_root / model_path

        if use_lora and adapter_path.exists():
            # Attempt to load base model + LoRA adapter
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    "peft not installed (needed for LoRA adapter loading). Run:\n"
                    "  pip install peft"
                )

            print(f"  Loading base model + LoRA adapter from {adapter_path}")
            base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
            self.model = PeftModel.from_pretrained(base, str(adapter_path))
            self.model = self.model.merge_and_unload()
            self.lora_loaded = True

        elif use_lora and not adapter_path.exists():
            # LoRA requested but adapter not found — fallback to base
            print(f"  {Fore.YELLOW}[WARN] LoRA adapter not found at: {adapter_path}")
            print(f"        Falling back to base IndicBART model (no fine-tuning)")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

        else:
            # use_lora=False — load base model only
            print(f"  Loading base IndicBART model (LoRA disabled)")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

        self.model = self.model.to(self.device)
        self.model.eval()

        # ── Status report ─────────────────────────────────────────────────────
        model_desc = "IndicBART + LoRA" if self.lora_loaded else "IndicBART base"
        lora_status = "loaded" if self.lora_loaded else "not found (using base)"

        print(f"  {Fore.GREEN}[OK] HinglishGenerator initialized")
        print(f"       Model:        {model_desc}")
        print(f"       Device:       {self.device}")
        print(f"       LoRA adapter: {lora_status}")

    # ── Single generation ────────────────────────────────────────────────────

    def generate(
        self,
        english_text: str,
        max_length: int = 128,
        num_beams: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        prefix: str = "Translate to Hinglish: ",
    ) -> dict:
        """
        Generate Hinglish explanation from English input.

        Args:
            english_text: NCERT English paragraph or sentence.
            max_length:   Maximum generation length in tokens.
            num_beams:    Beam search width (4 is a good default).
            temperature:  Sampling temperature (lower = more conservative).
            top_p:        Nucleus sampling threshold.
            do_sample:    Whether to use sampling (False = greedy/beam search).
            prefix:       Task prefix prepended to input (matches training format).

        Returns:
            dict with keys:
              - "input":             original English text
              - "output":            generated Hinglish text
              - "generation_config": dict of all generation parameters used
        """
        import torch

        # Guard: handle None or empty input gracefully
        if not english_text or not isinstance(english_text, str):
            return {
                "input": english_text,
                "output": "",
                "generation_config": {},
            }

        # Build input with task prefix
        input_text = prefix + english_text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Cap output length relative to input to prevent runaway generation
        input_len = inputs["input_ids"].shape[-1]
        effective_max = min(max_length, input_len + 80)

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": effective_max,
            "num_beams": num_beams,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.3,
            "length_penalty": -1.0,   # negative = penalise long outputs
            "early_stopping": True,
        }

        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode and strip stray special tokens from output
        output_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        # Strip stray BOS artifact '<s>' that IndicBART sometimes leaves
        # even after skip_special_tokens (known tokenizer quirk)
        if output_text.startswith("<s>"):
            output_text = output_text[3:].strip()

        generation_config = {
            "max_length": max_length,
            "num_beams": num_beams,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "prefix": prefix,
        }

        return {
            "input": english_text,
            "output": output_text,
            "generation_config": generation_config,
        }

    # ── Pipeline answer generation (M1→M2→M3) ────────────────────────────────

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[dict],
        max_length: int = 100,
    ) -> dict:
        """
        Generate a Hinglish answer given a student query and retrieved NCERT chunks.
        This is the PRIMARY method used in the full pipeline (M1→M2→M3).

        Args:
            query:            Student's question (could be Hinglish or English).
            retrieved_chunks: List of chunk dicts from NCERTRetriever.search().
                              Each has: text, metadata, relevance_score.
            max_length:       Maximum generation length in tokens.

        Returns:
            dict with keys:
              - "query":            original student query
              - "context_used":     the combined chunk text
              - "context_sources":  list of (chapter_title, chunk_id, relevance_score)
              - "hinglish_answer":  generated Hinglish text
              - "generation_config": generation parameters
        """
        # 1. Combine top retrieved chunks into context (2-3 chunks, ~400 words max)
        context_parts = []
        context_sources = []
        word_count = 0
        max_context_words = 400

        for chunk in retrieved_chunks[:3]:
            chunk_text = chunk.get("text", "")
            chunk_words = chunk_text.split()

            if word_count + len(chunk_words) > max_context_words and context_parts:
                break

            context_parts.append(chunk_text)
            word_count += len(chunk_words)

            # Track sources for attribution
            meta = chunk.get("metadata", {})
            context_sources.append((
                meta.get("chapter_title", "Unknown"),
                chunk.get("chunk_id", "unknown"),
                chunk.get("relevance_score", 0.0),
            ))

        context = " ".join(context_parts)

        # 2. Build structured prompt
        chapter_title = context_sources[0][0] if context_sources else "NCERT Biology"
        prompt = (
            f"Topic: {chapter_title} | "
            f"Context: {context} | "
            f"Student asks: {query} | "
            f"Explain in Hinglish:"
        )

        # 3. Generate using self.generate() with a pipeline-specific prefix
        result = self.generate(
            prompt,
            max_length=max_length,
            prefix="",  # prompt already contains the task instruction
        )

        return {
            "query": query,
            "context_used": context,
            "context_sources": context_sources,
            "hinglish_answer": result["output"],
            "generation_config": result["generation_config"],
        }

    # ── Batch generation ─────────────────────────────────────────────────────

    def generate_batch(
        self,
        english_texts: list[str],
        batch_size: int = 8,
        **kwargs,
    ) -> list[dict]:
        """
        Batch generation for evaluation. Generates for multiple inputs at once.

        Args:
            english_texts: List of English strings.
            batch_size:    Number to process at once.
            **kwargs:      Additional keyword arguments passed to generate().

        Returns:
            List of output dicts (same format as generate()).
        """
        import torch

        results = []
        total = len(english_texts)

        print(f"\n  Batch generation: {total} inputs (batch_size={batch_size})")

        start_time = time.time()

        for i in range(0, total, batch_size):
            batch_texts = english_texts[i : i + batch_size]
            batch_end = min(i + batch_size, total)

            print(f"    Processing batch {i // batch_size + 1} "
                  f"({i + 1}–{batch_end} of {total})...")

            # Process each item in the batch individually
            # (IndicBART handles variable-length inputs better this way)
            for text in batch_texts:
                result = self.generate(text, **kwargs)
                results.append(result)

        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 0

        print(f"  Batch complete: {total} items in {elapsed:.1f}s "
              f"({rate:.1f} items/sec)")

        return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI — Quick generation test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Load generator and test with sample NCERT sentences."""
    print(f"\n{'='*65}")
    print(f"  EduHinglish -- Hinglish Generator Test")
    print(f"{'='*65}")

    generator = HinglishGenerator()

    # ── Test 1: generate() with sample NCERT sentences ────────────────────────
    test_sentences = [
        "Photosynthesis is the process by which green plants make food using sunlight.",
        "The nucleus contains chromosomes which carry genetic information.",
        "Respiration involves breaking down of glucose to release energy.",
        "Stomata are tiny pores on leaves that allow gas exchange.",
        "DNA replication is the process of making an identical copy of DNA.",
    ]

    print(f"\n  {'─'*60}")
    print(f"  Test 1: generate() — English → Hinglish")
    print(f"  {'─'*60}")

    for i, sentence in enumerate(test_sentences):
        result = generator.generate(sentence)

        print(f"\n  [{i + 1}] English:")
        print(f"      \"{sentence}\"")
        print(f"      {Fore.CYAN}Hinglish:{Style.RESET_ALL}")
        print(f"      \"{result['output']}\"")

    # ── Test 2: generate_answer() with mock retrieved chunks ──────────────────
    print(f"\n  {'─'*60}")
    print(f"  Test 2: generate_answer() — Pipeline simulation (M1→M2→M3)")
    print(f"  {'─'*60}")

    mock_chunks = [
        {
            "text": "Photosynthesis occurs in chloroplasts of green plants. "
                    "The process uses sunlight, carbon dioxide and water to "
                    "produce glucose and oxygen. Chlorophyll pigment captures "
                    "light energy which drives the reaction.",
            "metadata": {"chapter_title": "Life Processes"},
            "chunk_id": "class10_ch05_chunk_012",
            "relevance_score": 0.85,
        },
        {
            "text": "The overall equation for photosynthesis is: 6CO2 + 6H2O "
                    "→ C6H12O6 + 6O2. This process occurs in two stages: "
                    "light reactions and dark reactions (Calvin cycle).",
            "metadata": {"chapter_title": "Life Processes"},
            "chunk_id": "class10_ch05_chunk_013",
            "relevance_score": 0.72,
        },
    ]

    test_query = "Photosynthesis kaise hoti hai?"

    print(f"\n  Query: \"{test_query}\"")
    print(f"  Retrieved chunks: {len(mock_chunks)}")

    for j, chunk in enumerate(mock_chunks):
        score = chunk["relevance_score"]
        color = Fore.GREEN if score >= 0.5 else Fore.YELLOW
        print(f"    Chunk {j + 1} ({color}score: {score:.2f}{Style.RESET_ALL}): "
              f"{chunk['metadata']['chapter_title']}")

    answer_result = generator.generate_answer(test_query, mock_chunks)

    print(f"\n  {Fore.CYAN}Hinglish Answer:{Style.RESET_ALL}")
    print(f"  \"{answer_result['hinglish_answer']}\"")

    print(f"\n  Context sources:")
    for title, chunk_id, score in answer_result["context_sources"]:
        print(f"    - {title} ({chunk_id}, score: {score:.2f})")

    print(f"\n{'='*65}")
    print(f"  {Fore.GREEN}[OK] Generator test complete")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
