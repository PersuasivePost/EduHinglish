"""
EduHinglish -- Phase 4, Task 2: Train MuRIL LID Model
=======================================================
Authors : Ashvatth & Jatin
Module  : Phase 4 -- MuRIL Fine-tuning (LID Token Classification)
Purpose : Fine-tune google/muril-base-cased with LoRA (PEFT) for
          word-level Language Identification on Hinglish text.

Labels: HI (0), EN (1), NE (2), UNIV (3), MIX (4)

Architecture:
  MuRIL-base (frozen) -> LoRA adapters (rank=16, alpha=32)
      -> Linear head (768 -> 5 classes)

Designed to run on Google Colab T4 (16GB VRAM).
Can run on CPU but will be very slow (~2 hours).

Usage:
  python training/train_muril_lid.py
  python training/train_muril_lid.py --epochs 15 --lr 3e-4
  python training/train_muril_lid.py --output models/muril_lid_v2/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

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
    import numpy as np
except ImportError:
    _missing.append("numpy")
try:
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    _missing.append("transformers")
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    _missing.append("peft")
try:
    from datasets import load_from_disk
except ImportError:
    _missing.append("datasets")

if _missing:
    print("[ERROR] Missing packages: " + ", ".join(_missing))
    print("  Run: pip install " + " ".join(_missing))
    sys.exit(1)


# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------

DATASET_DIR = Path("training/data/muril_lid_dataset")
OUTPUT_DIR  = Path("models/muril_lid_v1")
MODEL_NAME  = "google/muril-base-cased"
SEED        = 42

LABEL2ID = {"HI": 0, "EN": 1, "NE": 2, "UNIV": 3, "MIX": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

PHASE3_BASELINE = 98.1  # spaCy tok2vec LID accuracy


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

def _thick():
    print("  " + "=" * 58)

def _divider():
    print("  " + "-" * 58)


def check_gpu():
    """Detect GPU and print device info."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"  CUDA version: {torch.version.cuda}")
        return "cuda"
    else:
        print("  [WARNING] No GPU detected!")
        print("  Training on CPU will be VERY slow (~2 hours).")
        print("  Recommendation: Use the Colab notebook instead:")
        print("    training/colab_notebooks/01_train_muril_lid.ipynb")
        print()
        return "cpu"


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.1f} hours"


# -------------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Compute token-level accuracy and per-label precision/recall/F1.
    Ignores tokens with label -100 (subword continuations, special tokens).
    """
    predictions, labels = eval_pred
    # predictions shape: (batch, seq_len, num_labels)
    preds = np.argmax(predictions, axis=-1)

    # Flatten and filter out -100
    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(preds, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_labels.append(l)
                pred_labels.append(p)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Overall accuracy
    accuracy = (true_labels == pred_labels).mean() * 100

    # Per-label precision, recall, F1
    label_metrics = {}
    for label_id in range(NUM_LABELS):
        label_name = ID2LABEL[label_id]
        tp = int(((pred_labels == label_id) & (true_labels == label_id)).sum())
        fp = int(((pred_labels == label_id) & (true_labels != label_id)).sum())
        fn = int(((pred_labels != label_id) & (true_labels == label_id)).sum())

        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall    = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        label_metrics[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    return {
        "accuracy": accuracy,
        "label_metrics": label_metrics,
    }


# -------------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------------

def train(
    max_epochs: int = 10,
    learning_rate: float = 2e-4,
    batch_size: int = 16,
    lora_r: int = 16,
    lora_alpha: int = 32,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Main training function."""
    print()
    _thick()
    print("  EduHinglish -- Phase 4: Train MuRIL LID Model")
    _thick()
    print()

    # -- 1. Device check --------------------------------------------------
    print("  [1/6] Checking device...")
    device = check_gpu()
    use_fp16 = (device == "cuda")
    print()

    # -- 2. Load dataset --------------------------------------------------
    print("  [2/6] Loading dataset...")
    if not DATASET_DIR.exists():
        print(f"  [ERROR] Dataset not found: {DATASET_DIR}")
        print("  Run: python training/prepare_hf_data.py")
        sys.exit(1)

    ds = load_from_disk(str(DATASET_DIR))
    train_ds = ds["train"]
    dev_ds   = ds["dev"]

    # Remove word_ids column (not needed for training, causes issues)
    if "word_ids" in train_ds.column_names:
        train_ds = train_ds.remove_columns(["word_ids"])
    if "word_ids" in dev_ds.column_names:
        dev_ds = dev_ds.remove_columns(["word_ids"])

    print(f"  Train examples: {len(train_ds)}")
    print(f"  Dev examples:   {len(dev_ds)}")
    print()

    # -- 3. Load model + tokenizer ----------------------------------------
    print("  [3/6] Loading MuRIL model + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # -- Apply LoRA -------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value", "dense"],
    )

    model = get_peft_model(model, lora_config)

    # Print trainable params
    trainable, total = model.get_nb_trainable_parameters()
    pct = trainable / total * 100
    print(f"  Base model:      {MODEL_NAME}")
    print(f"  LoRA rank:       {lora_r}")
    print(f"  LoRA alpha:      {lora_alpha}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({pct:.1f}%)")
    print()

    # -- 4. Training arguments --------------------------------------------
    print("  [4/6] Configuring training...")

    checkpoint_dir = str(output_dir.parent / "muril_lid_checkpoints")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        fp16=use_fp16,
        report_to="none",
        seed=SEED,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Data collator handles padding dynamically per batch
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=128,
    )

    # -- 5. Create Trainer and train --------------------------------------
    print(f"  Epochs:          {max_epochs}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  FP16:            {use_fp16}")
    print(f"  Checkpoints:     {checkpoint_dir}")
    print()

    _divider()
    print("  [5/6] Training started...")
    _divider()
    print()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    print()
    _divider()
    print(f"  Training complete in {format_time(elapsed)}")
    _divider()
    print()

    # -- 6. Final evaluation on best model --------------------------------
    print("  [6/6] Running final evaluation...")
    eval_result = trainer.evaluate()

    accuracy      = eval_result.get("eval_accuracy", 0.0)
    label_metrics = eval_result.get("eval_label_metrics", {})

    # -- Save model -------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Save merged model (adapter merged into base) for easy inference
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    # Save training metadata
    metadata = {
        "model_name": MODEL_NAME,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "epochs": max_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_accuracy": round(accuracy, 2),
        "phase3_baseline": PHASE3_BASELINE,
        "improvement": round(accuracy - PHASE3_BASELINE, 2),
        "trainable_params": trainable,
        "total_params": total,
        "training_time_seconds": round(elapsed, 1),
        "device": device,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "label_metrics": label_metrics,
    }
    meta_path = output_dir / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # -- Print summary ----------------------------------------------------
    improvement = accuracy - PHASE3_BASELINE
    imp_sign = "+" if improvement >= 0 else ""

    print()
    _thick()
    print("  MuRIL LID Training Complete")
    _thick()
    print(f"  Best dev accuracy : {accuracy:.1f}%")
    print(f"  Phase 3 baseline  : {PHASE3_BASELINE:.1f}% (spaCy tok2vec)")
    print(f"  Improvement       : {imp_sign}{improvement:.1f}pp")
    print(f"  Trainable params  : {trainable:,} / {total:,} ({pct:.1f}%)")
    print(f"  Training time     : {format_time(elapsed)} on {device.upper()}")
    print(f"  Saved adapter     : {adapter_dir}/")
    print(f"  Saved merged      : {merged_dir}/")
    print(f"  Saved metadata    : {meta_path}")
    _thick()

    # Per-label F1 table
    if label_metrics:
        print()
        print(f"  Per-label metrics:")
        print(f"  {'Label':>6s} | {'Precision':>10s} | {'Recall':>10s} | {'F1':>10s} | {'Support':>8s}")
        print(f"  {'-'*6:>6s}-+-{'-'*10:>10s}-+-{'-'*10:>10s}-+-{'-'*10:>10s}-+-{'-'*8:>8s}")
        for label_name in ["HI", "EN", "NE", "UNIV", "MIX"]:
            m = label_metrics.get(label_name, {})
            if not m:
                continue
            print(
                f"  {label_name:>6s} | "
                f"{m.get('precision', 0):>9.1f}% | "
                f"{m.get('recall', 0):>9.1f}% | "
                f"{m.get('f1', 0):>9.1f}% | "
                f"{m.get('support', 0):>8d}"
            )
        print()

    # Clean up checkpoints to save disk space
    ckpt_path = Path(checkpoint_dir)
    if ckpt_path.exists():
        shutil.rmtree(ckpt_path, ignore_errors=True)
        print(f"  Cleaned up checkpoints: {checkpoint_dir}")

    print()


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train_muril_lid",
        description="Fine-tune MuRIL for token-level LID with LoRA (PEFT).",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4).",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Training batch size (default: 16).",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for saved model (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    train(
        max_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
