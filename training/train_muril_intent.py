"""
EduHinglish -- Phase 4, Task 3: Train MuRIL Intent Model
==========================================================
Authors : Ashvatth & Jatin
Module  : Phase 4 -- MuRIL Fine-tuning (Intent Classification)
Purpose : Fine-tune google/muril-base-cased with LoRA (PEFT) for
          sequence-level student query intent classification.

Intents: explain_concept (0), compare_concepts (1), give_example (2),
         formula_request (3), definition (4)

Architecture:
  MuRIL-base (frozen) -> LoRA adapters (rank=8, alpha=16)
      -> [CLS] pooling -> Classification head (768 -> 5 intents)

NOTE: Only 120 labeled queries. Uses class-weighted loss to handle
the skewed distribution (53 explain vs 5 formula).

Designed to run on Google Colab T4 (16GB VRAM).
Can run on CPU (dataset is small, ~10 min on CPU).

Usage:
  python training/train_muril_intent.py
  python training/train_muril_intent.py --epochs 30 --lr 1e-4
  python training/train_muril_intent.py --output models/muril_intent_v2/
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
    import torch.nn as nn
except ImportError:
    _missing.append("torch")
try:
    import numpy as np
except ImportError:
    _missing.append("numpy")
try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
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

DATASET_DIR = Path("training/data/muril_intent_dataset_v2")
OUTPUT_DIR  = Path("models/muril_intent_v2")
MODEL_NAME  = "google/muril-base-cased"
SEED        = 42

INTENT2ID = {
    "explain_concept":  0,
    "compare_concepts": 1,
    "give_example":     2,
    "formula_request":  3,
    "definition":       4,
}
ID2INTENT   = {v: k for k, v in INTENT2ID.items()}
NUM_INTENTS = len(INTENT2ID)
INTENT_NAMES = list(INTENT2ID.keys())

PHASE3_BASELINE = 70.0  # spaCy BOW textcat intent accuracy


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
        print("  [WARNING] No GPU detected.")
        print("  Intent dataset is small (120 examples) -- CPU is usable (~10 min).")
        print("  For faster training, use the Colab notebook:")
        print("    training/colab_notebooks/02_train_muril_intent.ipynb")
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


def load_class_weights() -> list[float] | None:
    """Load class weights from label_maps.json created by prepare_hf_data.py."""
    maps_path = DATASET_DIR / "label_maps.json"
    if not maps_path.exists():
        return None
    with open(maps_path, encoding="utf-8") as f:
        maps = json.load(f)
    cw = maps.get("class_weights", {})
    if not cw:
        return None
    # Build ordered weight list matching intent IDs
    weights = []
    for intent_name in INTENT_NAMES:
        weights.append(cw.get(intent_name, 1.0))
    return weights


# -------------------------------------------------------------------------
# CUSTOM TRAINER WITH CLASS-WEIGHTED LOSS
# -------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """
    Trainer subclass that uses class-weighted CrossEntropyLoss
    to handle the imbalanced intent distribution.
    (53 explain_concept vs 5 formula_request)
    """

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# -------------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Compute accuracy, macro-F1, and per-class precision/recall/F1
    for intent classification.
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)

    # Overall accuracy
    accuracy = (preds == labels).mean() * 100

    # Per-class metrics
    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for class_id in range(NUM_INTENTS):
        intent_name = ID2INTENT[class_id]
        tp = int(((preds == class_id) & (labels == class_id)).sum())
        fp = int(((preds == class_id) & (labels != class_id)).sum())
        fn = int(((preds != class_id) & (labels == class_id)).sum())

        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall    = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[intent_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # Macro averages
    f1_macro = np.mean(f1s)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "per_class": per_class,
    }


def build_confusion_matrix(preds, labels, n_classes):
    """Build a confusion matrix as a 2D list."""
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for true, pred in zip(labels, preds):
        matrix[true][pred] += 1
    return matrix


def print_confusion_matrix(matrix, class_names):
    """Pretty-print a confusion matrix."""
    print()
    print("  Confusion Matrix (rows=true, cols=predicted):")
    print()

    # Header with short names
    short = [n.replace("_concept", "").replace("_concepts", "")
                .replace("_request", "").replace("_", " ")[:8]
             for n in class_names]

    header = "  {:>12s}".format("") + "".join(f" {s:>8s}" for s in short)
    print(header)
    print("  " + "-" * (12 + 8 * len(short) + len(short)))

    for i, row in enumerate(matrix):
        row_name = class_names[i][:12]
        cells = "".join(f" {v:>8d}" for v in row)
        print(f"  {row_name:>12s}{cells}")
    print()


# -------------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------------

def train(
    max_epochs: int = 20,
    learning_rate: float = 2e-4,
    batch_size: int = 8,
    lora_r: int = 8,
    lora_alpha: int = 16,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Main training function."""
    print()
    _thick()
    print("  EduHinglish -- Phase 4: Train MuRIL Intent Model")
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

    print(f"  Train examples: {len(train_ds)}")
    print(f"  Dev examples:   {len(dev_ds)}")

    # Load class weights
    class_weights = load_class_weights()
    if class_weights:
        print(f"  Class weights:  {[round(w, 2) for w in class_weights]}")
    print()

    # -- 3. Load model + tokenizer ----------------------------------------
    print("  [3/6] Loading MuRIL model + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_INTENTS,
        id2label=ID2INTENT,
        label2id=INTENT2ID,
    )

    # -- Apply LoRA -------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.15,
        bias="none",
        target_modules=["query", "key", "value", "dense"],
    )

    model = get_peft_model(model, lora_config)

    # Print trainable params
    trainable, total = model.get_nb_trainable_parameters()
    pct = trainable / total * 100
    print(f"  Base model:       {MODEL_NAME}")
    print(f"  LoRA rank:        {lora_r}")
    print(f"  LoRA alpha:       {lora_alpha}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({pct:.1f}%)")
    print()

    # -- 4. Training arguments --------------------------------------------
    print("  [4/6] Configuring training...")

    checkpoint_dir = str(output_dir.parent / "muril_intent_checkpoints")

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
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=5,
        fp16=use_fp16,
        report_to="none",
        seed=SEED,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Data collator for sequence classification (pads input_ids + attention_mask)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=128,
    )

    # -- 5. Create Trainer and train --------------------------------------
    print(f"  Epochs:          {max_epochs}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  FP16:            {use_fp16}")
    print(f"  Best metric:     f1_macro")
    print(f"  Checkpoints:     {checkpoint_dir}")
    print()

    _divider()
    print("  [5/6] Training started...")
    _divider()
    print()

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    print()
    _divider()
    print(f"  Training complete in {format_time(elapsed)}")
    _divider()
    print()

    # -- 6. Final evaluation + confusion matrix ---------------------------
    print("  [6/6] Running final evaluation...")
    eval_result = trainer.evaluate()

    accuracy   = eval_result.get("eval_accuracy", 0.0)
    f1_macro   = eval_result.get("eval_f1_macro", 0.0)
    per_class  = eval_result.get("eval_per_class", {})

    # Build confusion matrix from predictions on dev set
    dev_preds = trainer.predict(dev_ds)
    pred_labels = np.argmax(dev_preds.predictions, axis=-1)
    true_labels = dev_preds.label_ids
    cm = build_confusion_matrix(pred_labels.tolist(), true_labels.tolist(), NUM_INTENTS)

    # -- Save model -------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Save merged model for easy inference
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    # Save training metadata
    metadata = {
        "model_name": MODEL_NAME,
        "task": "intent_classification",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "epochs": max_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_accuracy": round(accuracy, 2),
        "best_f1_macro": round(f1_macro, 2),
        "phase3_baseline": PHASE3_BASELINE,
        "improvement": round(f1_macro - PHASE3_BASELINE, 2),
        "trainable_params": trainable,
        "total_params": total,
        "training_time_seconds": round(elapsed, 1),
        "device": device,
        "intent2id": INTENT2ID,
        "id2intent": {str(k): v for k, v in ID2INTENT.items()},
        "class_weights": class_weights,
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
    }
    meta_path = output_dir / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # -- Print summary ----------------------------------------------------
    improvement = f1_macro - PHASE3_BASELINE
    imp_sign = "+" if improvement >= 0 else ""

    print()
    _thick()
    print("  MuRIL Intent Training Complete")
    _thick()
    print(f"  Best dev accuracy  : {accuracy:.1f}%")
    print(f"  Best dev F1 (macro): {f1_macro:.1f}%")
    print(f"  Phase 3 baseline   : {PHASE3_BASELINE:.1f}% (spaCy BOW)")
    print(f"  Improvement        : {imp_sign}{improvement:.1f}pp")
    print(f"  Trainable params   : {trainable:,} / {total:,} ({pct:.1f}%)")
    print(f"  Training time      : {format_time(elapsed)} on {device.upper()}")
    print(f"  Saved adapter      : {adapter_dir}/")
    print(f"  Saved merged       : {merged_dir}/")
    print(f"  Saved metadata     : {meta_path}")
    _thick()

    # Per-class F1 table
    if per_class:
        print()
        print(f"  Per-intent metrics:")
        print(f"  {'Intent':>18s} | {'Prec':>8s} | {'Recall':>8s} | {'F1':>8s} | {'N':>5s}")
        print(f"  {'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}")
        for intent_name in INTENT_NAMES:
            m = per_class.get(intent_name, {})
            if not m:
                continue
            print(
                f"  {intent_name:>18s} | "
                f"{m.get('precision', 0):>7.1f}% | "
                f"{m.get('recall', 0):>7.1f}% | "
                f"{m.get('f1', 0):>7.1f}% | "
                f"{m.get('support', 0):>5d}"
            )

    # Confusion matrix
    print_confusion_matrix(cm, INTENT_NAMES)

    # Clean up checkpoints
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
        prog="train_muril_intent",
        description="Fine-tune MuRIL for intent classification with LoRA (PEFT).",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=20,
        help="Number of training epochs (default: 20).",
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
        default=8,
        help="Training batch size (default: 8).",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor (default: 16).",
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
