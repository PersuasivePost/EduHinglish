"""
EduHinglish -- Phase 6, Prompt 2: Train IndicBART Hinglish Generator
=====================================================================
Author  : Ashvatth
Module  : Phase 6 -- IndicBART Fine-tuning (English -> Hinglish Seq2Seq)
Purpose : Fine-tune ai4bharat/IndicBART with LoRA (PEFT) using
          Seq2SeqTrainer on the seq2seq dataset prepared by
          training/prepare_seq2seq_data.py.

Architecture:
  IndicBART-base (MBartForConditionalGeneration, 244M params, frozen)
      -> LoRA adapters (rank=16, alpha=32) on q_proj + v_proj
      -> generates natural Hinglish from NCERT English text

Designed to run on Google Colab T4 (16 GB VRAM).
Can run on CPU but will be extremely slow.

Usage:
  python training/train_indicbart.py                          # defaults
  python training/train_indicbart.py --epochs 3               # fewer epochs
  python training/train_indicbart.py --batch-size 4           # low-VRAM mode
  python training/train_indicbart.py --lr 5e-4                # custom LR
  python training/train_indicbart.py --test                   # generate after train
  python training/train_indicbart.py --resume models/indicbart_v1/checkpoint-XXX
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Suppress transformers deprecation: IndicBART config uses old `use_return_dict`
warnings.filterwarnings(
    "ignore",
    message=".*use_return_dict.*",
    category=FutureWarning,
)

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# -------------------------------------------------------------------------
# DEPENDENCY CHECK
# -------------------------------------------------------------------------

_missing = []
try:
    import torch
except ImportError:
    _missing.append("torch")
try:
    from transformers import (
        AutoTokenizer,
        MBartForConditionalGeneration,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
except ImportError:
    _missing.append("transformers>=4.40.0")
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    _missing.append("peft>=0.10.0")
try:
    from datasets import load_from_disk
except ImportError:
    _missing.append("datasets>=2.19.0")

if _missing:
    print("  [ERROR] Missing packages: " + ", ".join(_missing))
    print("  Run: pip install " + " ".join(_missing))
    sys.exit(1)


# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------

DATASET_DIR  = Path("training/data/indicbart_seq2seq_dataset")
OUTPUT_DIR   = Path("models/indicbart_v1")
MODEL_NAME   = "ai4bharat/IndicBART"
SEED         = 42

# LoRA defaults
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1

# Training defaults
DEFAULT_EPOCHS     = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR         = 3e-4
GRAD_ACCUM_STEPS   = 2
MAX_GEN_LENGTH     = 256


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

def _thick() -> None:
    print("  " + "=" * 58)


def _divider() -> None:
    print("  " + "-" * 58)


def check_gpu() -> str:
    """Detect GPU and print device info."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU detected : {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"  CUDA version : {torch.version.cuda}")
        return "cuda"
    else:
        print("  [WARNING] No GPU detected!")
        print("  Training on CPU will be EXTREMELY slow.")
        print("  Recommendation: Use the Colab notebook instead:")
        print("    training/colab_notebooks/03_train_indicbart.ipynb")
        print()
        return "cpu"


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.1f} hours"


def load_dataset_config(config_path: Path) -> dict:
    """Load the config.json saved by prepare_seq2seq_data.py."""
    if not config_path.exists():
        print(f"  [WARNING] config.json not found at {config_path}")
        return {}
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------------
# CUSTOM DATA COLLATOR
# -------------------------------------------------------------------------

@dataclass
class Seq2SeqDataCollator:
    """
    Pads input_ids, attention_mask, and labels to the longest sequence
    in each batch. Labels are padded with -100 (ignored in cross-entropy loss).

    This is used instead of DataCollatorForSeq2Seq to keep full control
    and match the project's coding style.
    """
    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # ── Determine max lengths in this batch ───────────────────────────
        max_src = max(len(f["input_ids"]) for f in features)
        max_tgt = max(len(f["labels"])    for f in features)

        input_ids_batch      = []
        attention_mask_batch = []
        labels_batch         = []

        for feature in features:
            src_len = len(feature["input_ids"])
            tgt_len = len(feature["labels"])

            # Right-pad source
            src_pad   = max_src - src_len
            input_ids = feature["input_ids"] + [self.pad_token_id] * src_pad
            attn_mask = feature["attention_mask"] + [0] * src_pad

            # Right-pad labels with -100
            tgt_pad = max_tgt - tgt_len
            labels  = feature["labels"] + [self.label_pad_id] * tgt_pad

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attn_mask)
            labels_batch.append(labels)

        return {
            "input_ids":      torch.tensor(input_ids_batch,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch,  dtype=torch.long),
            "labels":         torch.tensor(labels_batch,          dtype=torch.long),
        }


# -------------------------------------------------------------------------
# GENERATION TEST
# -------------------------------------------------------------------------

def run_generation_test(
    model,
    tokenizer,
    dev_dataset,
    device: str,
    n_samples: int = 5,
) -> None:
    """
    Generate Hinglish output for up to *n_samples* examples from the dev set
    and print the source + generated text side by side.
    """
    print()
    _thick()
    print("  Generation Test — Sample Outputs")
    _thick()

    model.eval()
    model_device = next(model.parameters()).device

    n = min(n_samples, len(dev_dataset))
    for i in range(n):
        row        = dev_dataset[i]
        input_ids  = torch.tensor([row["input_ids"]], dtype=torch.long).to(model_device)
        attn_mask  = torch.tensor([row["attention_mask"]], dtype=torch.long).to(model_device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_length=MAX_GEN_LENGTH,
                num_beams=4,
                early_stopping=True,
            )

        src_text = tokenizer.decode(row["input_ids"],    skip_special_tokens=True)
        gen_text = tokenizer.decode(generated[0],        skip_special_tokens=True)
        ref_text = tokenizer.decode(
            [t for t in row["labels"] if t != -100],
            skip_special_tokens=True,
        )

        _divider()
        print(f"  Example {i + 1}")
        print(f"  SRC : {src_text[:120]}")
        print(f"  GEN : {gen_text[:120]}")
        print(f"  REF : {ref_text[:120]}")

    _divider()
    print()


# -------------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------------

def train(
    max_epochs:  int   = DEFAULT_EPOCHS,
    batch_size:  int   = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
    lora_r:      int   = LORA_R,
    lora_alpha:  int   = LORA_ALPHA,
    output_dir:  Path  = OUTPUT_DIR,
    resume_from: str | None = None,
    run_test:    bool  = False,
) -> None:
    """Main IndicBART fine-tuning function."""
    print()
    _thick()
    print("  EduHinglish -- Phase 6: Train IndicBART Hinglish Generator")
    _thick()
    print()

    # -- 1. Device check ---------------------------------------------------
    print("  [1/6] Checking device...")
    device   = check_gpu()
    use_fp16 = (device == "cuda")
    print()

    # -- 2. Load dataset ---------------------------------------------------
    print("  [2/6] Loading seq2seq dataset...")
    if not DATASET_DIR.exists():
        print(f"  [ERROR] Dataset not found: {DATASET_DIR}")
        print("  Run: python training/prepare_seq2seq_data.py")
        sys.exit(1)

    ds       = load_from_disk(str(DATASET_DIR))
    train_ds = ds["train"]
    dev_ds   = ds["dev"]

    print(f"  Train examples : {len(train_ds)}")
    print(f"  Dev examples   : {len(dev_ds)}")

    # Load config.json for metadata (non-blocking if missing)
    ds_config = load_dataset_config(DATASET_DIR / "config.json")
    if ds_config:
        print(f"  Pair types     : "
              f"A={ds_config.get('type_a_count', '?')}  "
              f"B={ds_config.get('type_b_count', '?')}  "
              f"C={ds_config.get('type_c_count', '?')}")
    print()

    # -- 3. Load model + tokenizer -----------------------------------------
    print(f"  [3/6] Loading IndicBART + applying LoRA...")
    print(f"  Model : {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

    # ── Apply LoRA ─────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Print trainable vs total params
    trainable, total = model.get_nb_trainable_parameters()
    pct = trainable / total * 100
    print(f"  LoRA rank      : {lora_r}")
    print(f"  LoRA alpha     : {lora_alpha}")
    print(f"  Trainable      : {trainable:,} / {total:,} ({pct:.2f}%)")
    print()

    # -- 4. Data collator --------------------------------------------------
    print("  [4/6] Configuring training...")
    pad_id       = tokenizer.pad_token_id or 0
    data_collator = Seq2SeqDataCollator(pad_token_id=pad_id)

    # -- 5. Training arguments --------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        predict_with_generate=True,
        generation_max_length=MAX_GEN_LENGTH,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    print(f"  Epochs         : {max_epochs}")
    print(f"  Batch size     : {batch_size}  (grad_accum={GRAD_ACCUM_STEPS})")
    print(f"  Learning rate  : {learning_rate}")
    print(f"  FP16           : {use_fp16}")
    print(f"  Output dir     : {output_dir}")
    if resume_from:
        print(f"  Resume from    : {resume_from}")
    print()

    # -- 6. Train ---------------------------------------------------------
    _divider()
    print("  [5/6] Training started...")
    _divider()
    print()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
    )

    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    elapsed      = time.time() - start_time

    print()
    _divider()
    print(f"  Training complete in {format_time(elapsed)}")
    _divider()
    print()

    # ── Extract final losses ───────────────────────────────────────────────
    log_history     = trainer.state.log_history
    final_train_loss = None
    final_eval_loss  = None
    for entry in reversed(log_history):
        if final_train_loss is None and "loss" in entry:
            final_train_loss = entry["loss"]
        if final_eval_loss is None and "eval_loss" in entry:
            final_eval_loss = entry["eval_loss"]
        if final_train_loss is not None and final_eval_loss is not None:
            break

    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None) or str(output_dir)

    # -- Save adapter + tokenizer -----------------------------------------
    print("  [6/6] Saving model artifacts...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"  [OK] LoRA adapter saved : {output_dir}/")

    # ── Save training_summary.json ─────────────────────────────────────────
    summary = {
        "model_name":          MODEL_NAME,
        "epochs":              max_epochs,
        "final_train_loss":    round(final_train_loss, 4) if final_train_loss is not None else None,
        "final_eval_loss":     round(final_eval_loss,  4) if final_eval_loss  is not None else None,
        "total_training_time": format_time(elapsed),
        "training_time_seconds": round(elapsed, 1),
        "trainable_params":    trainable,
        "total_params":        total,
        "lora_r":              lora_r,
        "lora_alpha":          lora_alpha,
        "batch_size":          batch_size,
        "learning_rate":       learning_rate,
        "best_checkpoint_path": best_ckpt,
        "device":              device,
        "fp16":                use_fp16,
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] Training summary   : {summary_path}")

    # ── Print final summary ────────────────────────────────────────────────
    print()
    _thick()
    print("  IndicBART Fine-tuning Complete")
    _thick()
    print(f"  Model           : {MODEL_NAME} + LoRA")
    print(f"  Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    if final_train_loss is not None:
        print(f"  Final train loss: {final_train_loss:.4f}")
    if final_eval_loss is not None:
        print(f"  Final eval loss : {final_eval_loss:.4f}")
    print(f"  Training time   : {format_time(elapsed)} on {device.upper()}")
    print(f"  Saved to        : {output_dir}/")
    print(f"  Best checkpoint : {best_ckpt}")
    _thick()
    print()

    # -- Optional generation test -----------------------------------------
    if run_test:
        run_generation_test(model, tokenizer, dev_ds, device)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train_indicbart",
        description=(
            "Fine-tune IndicBART (MBart seq2seq) with LoRA (PEFT) "
            "for English -> Hinglish generation."
        ),
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Per-device batch size (default: {DEFAULT_BATCH_SIZE}). "
             "Use 4 for limited VRAM.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR}).",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=LORA_R,
        help=f"LoRA rank (default: {LORA_R}).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=LORA_ALPHA,
        help=f"LoRA alpha scaling factor (default: {LORA_ALPHA}).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for saved model (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="Resume training from a saved checkpoint directory.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="After training, generate Hinglish for 5 dev samples and print them.",
    )
    args = parser.parse_args()

    train(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        output_dir=args.output,
        resume_from=args.resume,
        run_test=args.test,
    )


if __name__ == "__main__":
    main()
