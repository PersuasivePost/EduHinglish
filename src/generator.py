"""
EduHinglish — Hinglish Answer Generator
=========================================
Author  : Jatin
Module  : M3 — Hinglish Answer Generator (Phase 6)
Purpose : Inference module that loads Gemma-3-4b-it with the fine-tuned LoRA
          adapter and answers a student's question from retrieved NCERT text.
          Integrates with M1 (preprocessing) and M2 (retriever) to form the
          complete M1→M2→M3 pipeline.

Usage:
    python src/generator.py
    # Runs a quick generation test with mock retrieved chunks.
"""

import os
import time
from pathlib import Path
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

BASE_MODEL = "google/gemma-3-4b-it"
# v4 = all five subjects (bio/chem/phys/maths/ss, class 9-10); falls back to v3
# (bio + ss) when v4 has not been downloaded. EDUHINGLISH_ADAPTER overrides both
# and may be a local folder or a Hugging Face Hub repo id ("user/repo").
ADAPTER_CANDIDATES = [
    "models/gemma_eduhinglish_v4/adapter",
    "models/gemma_eduhinglish_v3/adapter",
]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_adapter(adapter_path: str = None) -> tuple[str | None, bool]:
    """
    Return (path_or_repo_id, is_local). None means no adapter was found.
    """
    adapter_path = adapter_path or os.environ.get("EDUHINGLISH_ADAPTER")
    if adapter_path:
        p = Path(adapter_path)
        for candidate in (p, PROJECT_ROOT / p):
            if (candidate / "adapter_config.json").exists():
                return str(candidate), True
        # Not on disk: treat "user/repo" as a Hub repo id
        if "/" in adapter_path and not p.is_absolute() and not adapter_path.startswith("."):
            return adapter_path, False
        return None, True
    for rel in ADAPTER_CANDIDATES:
        if (PROJECT_ROOT / rel / "adapter_config.json").exists():
            return str(PROJECT_ROOT / rel), True
    return None, True


# Gemma has no system role, so the NCERT context goes inside the user turn.
# Keep this identical to PROMPT_TEMPLATE / INSTRUCTION_* in scripts/build_dataset.py,
# which renders the training data.
PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "{intent_line}"
    "{instruction}"
)
INSTRUCTIONS = {
    "hinglish": "Answer the question in Hinglish using only the given context.",
    "english": "Answer the question in English using only the given context.",
}

# The training data drops the intent line for ~12% of records, so the model
# treats the classifier's intent as a hint. Below this confidence we drop it too.
INTENT_CONFIDENCE_THRESHOLD = 0.6


def build_prompt(context: str, question: str, intent: str = None,
                 language: str = "hinglish") -> str:
    """Build the user-turn text exactly as it was rendered for training."""
    intent_line = f"Intent Type: {intent}\n" if intent else ""
    return PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        intent_line=intent_line,
        instruction=INSTRUCTIONS[language],
    )


class HinglishGenerator:
    """
    M3 — Hinglish Answer Generator

    Loads google/gemma-3-4b-it plus the LoRA adapter and generates answers
    grounded in retrieved NCERT chunks. Falls back to the base model when the
    adapter is not found.
    """

    def __init__(
        self,
        adapter_path: str = None,
        base_model: str = BASE_MODEL,
        device: str = None,
        use_lora: bool = True,
        load_in_4bit: bool = None,
    ):
        """
        Load model + tokenizer.

        Args:
            adapter_path: LoRA adapter folder (absolute, or relative to the project
                          root) or a Hub repo id. None = $EDUHINGLISH_ADAPTER,
                          then ADAPTER_CANDIDATES.
            base_model:   HuggingFace model ID of the Gemma base model.
            device:       'cuda', 'mps', 'cpu', or None for auto-detect.
            use_lora:     If True, attempt to load the LoRA adapter.
            load_in_4bit: NF4 quantisation (CUDA + bitsandbytes only).
                          None = enable automatically when available.
        """
        print(f"\n  Loading Hinglish generator: {base_model}")

        try:
            import torch
        except ImportError:
            raise ImportError("torch not installed. Run:\n  pip install torch")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise ImportError("transformers not installed. Run:\n  pip install transformers")

        # ── Device selection ──────────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # ── 4-bit availability ────────────────────────────────────────────────
        if load_in_4bit is None:
            try:
                import bitsandbytes  # noqa: F401
                load_in_4bit = device == "cuda"
            except ImportError:
                load_in_4bit = False
        if load_in_4bit and device != "cuda":
            raise ValueError("4-bit loading needs a CUDA device.")
        self.load_in_4bit = load_in_4bit

        # ── Tokenizer ─────────────────────────────────────────────────────────
        self.base_model_name = base_model
        _hf_token = os.environ.get("HF_TOKEN") or None
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, token=_hf_token)
        # Gemma ends a reply with <end_of_turn>; stop there as well as at <eos>.
        end_turn = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        self.eos_token_ids = [self.tokenizer.eos_token_id] + (
            [end_turn] if isinstance(end_turn, int) and end_turn != self.tokenizer.unk_token_id else []
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Base model ────────────────────────────────────────────────────────
        load_kwargs = {}
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = {
                "cuda": torch.bfloat16,
                "mps": torch.float16,
                "cpu": torch.float32,
            }[device]

        self.model = AutoModelForCausalLM.from_pretrained(base_model, token=_hf_token, **load_kwargs)

        # ── Optional LoRA adapter ─────────────────────────────────────────────
        self.lora_loaded = False
        self.adapter_source = None
        adapter, _ = resolve_adapter(adapter_path) if use_lora else (None, True)

        if adapter:
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError("peft not installed (needed for the LoRA adapter). Run:\n  pip install peft")

            print(f"  Loading LoRA adapter from {adapter}")
            self.model = PeftModel.from_pretrained(self.model, adapter)
            if not load_in_4bit:
                self.model = self.model.merge_and_unload()
            self.lora_loaded = True
            self.adapter_source = adapter
        elif use_lora:
            print(f"  {Fore.YELLOW}[WARN] No LoRA adapter found ({adapter_path or ADAPTER_CANDIDATES})")
            print(f"        Falling back to the base Gemma model (no fine-tuning)")

        if not load_in_4bit:
            self.model = self.model.to(self.device)
        self.model.eval()

        # ── Status report ─────────────────────────────────────────────────────
        model_desc = "Gemma-3-4b-it + LoRA" if self.lora_loaded else "Gemma-3-4b-it base"
        print(f"  {Fore.GREEN}[OK] HinglishGenerator initialized")
        print(f"       Model:        {model_desc}")
        print(f"       Device:       {self.device}{' (4-bit)' if load_in_4bit else ''}")
        print(f"       LoRA adapter: {self.adapter_source if self.lora_loaded else 'not found (using base)'}")

    # ── Single generation ────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.1,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        """
        Generate a reply for an already-built user prompt (see build_prompt()).

        Returns:
            dict with keys "input", "output" and "generation_config".
        """
        import torch

        if not prompt or not isinstance(prompt, str):
            return {"input": prompt, "output": "", "generation_config": {}}

        # The chat template adds <bos>, so tokenize without special tokens
        # (a doubled <bos> hurts Gemma).
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.eos_token_ids,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens (skip the echoed prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "input": prompt,
            "output": output_text,
            "generation_config": {k: v for k, v in gen_kwargs.items()
                                  if k not in ("pad_token_id", "eos_token_id")},
        }

    # ── Pipeline answer generation (M1→M2→M3) ────────────────────────────────

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[dict],
        intent: str = None,
        intent_confidence: float = None,
        language: str = "hinglish",
        max_new_tokens: int = 256,
    ) -> dict:
        """
        Generate an answer given a student query and retrieved NCERT chunks.
        This is the PRIMARY method used in the full pipeline (M1→M2→M3).

        Args:
            query:             Student's question (Hinglish or English).
            retrieved_chunks:  Chunk dicts from NCERTRetriever.search().
                               Each has: text, metadata, relevance_score.
            intent:            Predicted intent label (optional).
            intent_confidence: Classifier confidence. The intent line is left
                               out of the prompt when this is below
                               INTENT_CONFIDENCE_THRESHOLD.
            language:          "hinglish" or "english".
            max_new_tokens:    Maximum generation length in tokens.

        Returns:
            dict with keys:
              - "query":             original student query
              - "context_used":      the combined chunk text
              - "context_sources":   list of (chapter_title, chunk_id, relevance_score)
              - "hinglish_answer":   generated answer text
              - "prompt_intent":     intent placed in the prompt, or None
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

            meta = chunk.get("metadata", {})
            context_sources.append((
                meta.get("chapter_title", "Unknown"),
                chunk.get("chunk_id", "unknown"),
                chunk.get("relevance_score", 0.0),
            ))

        context = " ".join(context_parts)

        # 2. Decide whether to show the intent hint
        use_intent = intent is not None and (
            intent_confidence is None or intent_confidence >= INTENT_CONFIDENCE_THRESHOLD
        )
        prompt_intent = intent if use_intent else None

        # 3. Build the training-format prompt and generate
        prompt = build_prompt(context, query, prompt_intent, language)
        result = self.generate(prompt, max_new_tokens=max_new_tokens)

        return {
            "query": query,
            "context_used": context,
            "context_sources": context_sources,
            "hinglish_answer": result["output"],
            "prompt_intent": prompt_intent,
            "generation_config": result["generation_config"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI — Quick generation test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Load generator and test with mock retrieved chunks."""
    print(f"\n{'='*65}")
    print(f"  EduHinglish -- Hinglish Generator Test")
    print(f"{'='*65}")

    generator = HinglishGenerator()

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
    for j, chunk in enumerate(mock_chunks):
        score = chunk["relevance_score"]
        color = Fore.GREEN if score >= 0.5 else Fore.YELLOW
        print(f"    Chunk {j + 1} ({color}score: {score:.2f}{Style.RESET_ALL}): "
              f"{chunk['metadata']['chapter_title']}")

    t0 = time.time()
    answer_result = generator.generate_answer(
        test_query, mock_chunks, intent="explain_concept", intent_confidence=0.9
    )
    elapsed = time.time() - t0

    print(f"\n  {Fore.CYAN}Hinglish Answer:{Style.RESET_ALL}")
    print(f"  \"{answer_result['hinglish_answer']}\"")
    print(f"\n  Intent in prompt: {answer_result['prompt_intent']}")
    print(f"  Context sources:")
    for title, chunk_id, score in answer_result["context_sources"]:
        print(f"    - {title} ({chunk_id}, score: {score:.2f})")

    print(f"\n{'='*65}")
    print(f"  {Fore.GREEN}[OK] Generator test complete ({elapsed:.1f}s)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
