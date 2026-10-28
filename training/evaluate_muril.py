"""
EduHinglish — MuRIL Model Evaluation
=====================================
Scores the downloaded MuRIL models on the gold dev splits produced by
`training/prepare_hf_data.py`, then runs a few spot-check sentences.

Usage:
    python training/evaluate_muril.py
    python training/evaluate_muril.py --lid-only
    python training/evaluate_muril.py --intent-only
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))
from label_config import DATA_VERSION

PROJECT_ROOT = Path(__file__).parent.parent
LID_MODEL_DIR = PROJECT_ROOT / "models" / f"muril_lid_{DATA_VERSION}"
INTENT_MODEL_DIR = PROJECT_ROOT / "models" / f"muril_intent_{DATA_VERSION}"
LID_DATA_DIR = PROJECT_ROOT / "training" / "data" / f"muril_lid_dataset_{DATA_VERSION}"
INTENT_DATA_DIR = PROJECT_ROOT / "training" / "data" / f"muril_intent_dataset_{DATA_VERSION}"
BATCH_SIZE = 32

LID_TESTS = [
    "Mitochondria ko cell ka powerhouse kehte hain.",
    "Sir, photosynthesis kaise hoti hai?",
    "Mendel ne pea plants par experiments kiye the.",
    "Harappan cities mein drainage system kaafi advanced tha.",
]

INTENT_TESTS = [
    ("Sir, nucleus ka kaam kya hota hai?", "explain_concept"),
    ("Mitochondria aur chloroplast mein kya fark hai?", "compare_concepts"),
    ("Ek example do osmosis ka?", "give_example"),
    ("Democracy ki definition kya hai?", "definition"),
    ("Lok Sabha aur Rajya Sabha mein kya antar hai?", "compare_concepts"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _thick():
    print("  " + "═" * 54)


def print_header():
    _thick()
    print(f"  EduHinglish — MuRIL Model Evaluation ({DATA_VERSION})")
    _thick()
    print(f"  Device: {DEVICE}\n")


def load_model(model_dir: Path, model_cls):
    if not (model_dir / "config.json").exists():
        sys.exit(f"  [ERROR] No model at {model_dir}\n"
                 f"          Download it from Drive (MyDrive/EduHinglish/models/{model_dir.name}) "
                 f"into models/ first.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = model_cls.from_pretrained(model_dir).to(DEVICE).eval()
    return model, tokenizer


def predict_batches(model, tokenizer, dev):
    """Yields (logits, batch) for the pre-tokenized dev split."""
    for start in range(0, len(dev), BATCH_SIZE):
        batch = dev[start:start + BATCH_SIZE]
        features = [{"input_ids": ids, "attention_mask": mask}
                    for ids, mask in zip(batch["input_ids"], batch["attention_mask"])]
        enc = tokenizer.pad(features, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits.cpu()
        yield logits, batch


def print_report(gold: list[str], pred: list[str], labels: list[str]):
    correct = sum(g == p for g, p in zip(gold, pred))
    print(f"  Dev accuracy: {correct / len(gold):.2%}  ({correct}/{len(gold)})")

    gold_n, pred_n, tp = Counter(gold), Counter(pred), Counter(g for g, p in zip(gold, pred) if g == p)
    f1s = []
    print(f"  {'Label':>20} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'N':>5}")
    for label in labels:
        if gold_n[label] == 0:
            continue
        prec = tp[label] / pred_n[label] if pred_n[label] else 0.0
        rec = tp[label] / gold_n[label]
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        print(f"  {label:>20} | {prec:>6.1%} | {rec:>6.1%} | {f1:>6.1%} | {gold_n[label]:>5}")
    print(f"  Macro F1 (labels present in dev): {sum(f1s) / len(f1s):.2%}\n")


def predict_lid(text: str, model, tokenizer) -> list[tuple[str, str]]:
    '''Returns [(word, label), ...] using the first subword of each word.'''
    # Same word split as prepare_hf_data.py
    words = re.findall(r"\w+|[^\w\s]", text)
    encoding = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        predictions = model(**encoding).logits.argmax(-1).squeeze(0).tolist()

    result, seen = [], set()
    for idx, word_idx in enumerate(encoding.word_ids()):
        if word_idx is None or word_idx in seen:
            continue
        seen.add(word_idx)
        result.append((words[word_idx], model.config.id2label[predictions[idx]]))
    return result


def eval_lid():
    print(f"  ── MuRIL LID ({LID_MODEL_DIR.relative_to(PROJECT_ROOT)}/) ──")
    model, tokenizer = load_model(LID_MODEL_DIR, AutoModelForTokenClassification)
    id2label = model.config.id2label
    dev = load_from_disk(str(LID_DATA_DIR))["dev"]

    gold, pred = [], []
    for logits, batch in predict_batches(model, tokenizer, dev):
        for row_logits, labels in zip(logits, batch["labels"]):
            row_pred = row_logits.argmax(-1).tolist()
            # Only the first subword of each word carries a label; the rest are -100
            for i, label_id in enumerate(labels):
                if label_id != -100:
                    gold.append(id2label[label_id])
                    pred.append(id2label[row_pred[i]])

    print(f"  Gold dev set: {len(dev)} sentences, {len(gold)} words")
    print_report(gold, pred, list(id2label.values()))

    print("  Spot checks:")
    for sent in LID_TESTS:
        tagged = " ".join(f"{w}/{l}" for w, l in predict_lid(sent, model, tokenizer))
        print(f"    {tagged}")
    print()


def eval_intent():
    print(f"  ── MuRIL Intent ({INTENT_MODEL_DIR.relative_to(PROJECT_ROOT)}/) ──")
    model, tokenizer = load_model(INTENT_MODEL_DIR, AutoModelForSequenceClassification)
    id2label = model.config.id2label
    dev = load_from_disk(str(INTENT_DATA_DIR))["dev"]

    gold, pred = [], []
    for logits, batch in predict_batches(model, tokenizer, dev):
        gold.extend(id2label[i] for i in batch["labels"])
        pred.extend(id2label[i] for i in logits.argmax(-1).tolist())

    print(f"  Dev set: {len(dev)} queries")
    print_report(gold, pred, list(id2label.values()))

    print("  Spot checks:")
    for sent, expected in INTENT_TESTS:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).squeeze(0)
        conf, pred_id = probs.max(-1)
        label = id2label[pred_id.item()]
        marker = "✓" if label == expected else f"✗ (expected {expected})"
        print(f"    {sent:<50} → {label} ({conf.item():.2f}) {marker}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate MuRIL models for LID and Intent.")
    parser.add_argument("--lid-only", action="store_true", help="Evaluate only the LID model.")
    parser.add_argument("--intent-only", action="store_true", help="Evaluate only the Intent model.")
    args = parser.parse_args()

    print_header()
    if not args.intent_only:
        eval_lid()
    if not args.lid_only:
        eval_intent()
    print("  spaCy vs MuRIL comparison: workspace/test_results/model_v3/RESULTS.md")


if __name__ == "__main__":
    main()
