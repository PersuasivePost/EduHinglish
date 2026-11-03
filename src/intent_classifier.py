"""
EduHinglish — MuRIL Intent Classifier (inference)
===================================================
Module  : M1 — Input Processing
Purpose : Load the fine-tuned MuRIL sequence classifier (models/muril_intent_v3,
          trained by training/train_muril_intent.py / Colab notebook 02) and
          predict one of the 15 intents in training/label_config.py.

          The prediction is passed to Gemma as "Intent Type: <intent>" (see
          src/generator.py), so the labels must match the QA datasets.

The model path can be a local folder or a Hugging Face Hub repo id, e.g.
    EDUHINGLISH_INTENT_MODEL=your-hf-user/eduhinglish-muril-intent

Usage:
    python src/intent_classifier.py "Photosynthesis kya hota hai?"
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INTENT_MODEL = "models/muril_intent_v3"


def resolve_model_path(path_or_repo: str) -> str:
    """Local folder (absolute or relative to the project root) or a Hub repo id."""
    p = Path(path_or_repo)
    if p.is_absolute() and p.exists():
        return str(p)
    if (PROJECT_ROOT / p).exists():
        return str(PROJECT_ROOT / p)
    return path_or_repo  # assume "user/repo" on the Hugging Face Hub


class IntentClassifier:
    def __init__(self, model_path: str = None, device: str = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = resolve_model_path(
            model_path or os.environ.get("EDUHINGLISH_INTENT_MODEL", DEFAULT_INTENT_MODEL)
        )
        # MuRIL is small (~240M params). Keep it on CPU by default so the GPU is
        # left for Gemma; set EDUHINGLISH_INTENT_DEVICE=cuda to move it.
        self.device = device or os.environ.get("EDUHINGLISH_INTENT_DEVICE", "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self._torch = torch
        print(f"  [OK] IntentClassifier loaded from {model_path} ({len(self.id2label)} labels, {self.device})")

    def predict(self, text: str, top_k: int = 3) -> dict:
        """Return {"intent", "confidence", "top_k": [(label, prob), ...]}."""
        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits[0], dim=-1)
        values, indices = probs.topk(min(top_k, probs.shape[-1]))
        ranked = [(self.id2label[i.item()], round(v.item(), 4)) for v, i in zip(values, indices)]
        return {"intent": ranked[0][0], "confidence": ranked[0][1], "top_k": ranked}


if __name__ == "__main__":
    clf = IntentClassifier()
    queries = sys.argv[1:] or [
        "Photosynthesis kya hota hai?",
        "Aerobic aur anaerobic respiration mein kya fark hai?",
        "Ohm's law ka ek numerical solve karo: V = 10V, R = 5 ohm, current kitna hoga?",
        "Federalism ke 3 features batao",
    ]
    for q in queries:
        r = clf.predict(q)
        print(f"  {r['intent']:<22} {r['confidence']:.2f}  {q}")
