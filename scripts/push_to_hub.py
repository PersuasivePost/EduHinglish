"""
Upload a trained EduHinglish model folder to the Hugging Face Hub.

Works for both kinds of model we serve:
    gemma  — LoRA adapter folder (adapter_config.json + adapter_model.safetensors)
    intent — MuRIL intent classifier (config.json + model.safetensors)

A model card (README.md) is written into the upload. Repos are private by default.

Setup (once):
    pip install huggingface_hub
    huggingface-cli login            # paste a WRITE token from https://huggingface.co/settings/tokens

Usage:
    python scripts/push_to_hub.py gemma  models/gemma_eduhinglish_v4/adapter  your-hf-user/eduhinglish-gemma-3-4b-it-lora
    python scripts/push_to_hub.py intent models/muril_intent_v3               your-hf-user/eduhinglish-muril-intent
    add --public to make the repo public
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GEMMA_CARD = """---
base_model: google/gemma-3-4b-it
library_name: peft
license: gemma
language: [en, hi]
tags: [lora, qlora, hinglish, education, ncert, code-mixing]
pipeline_tag: text-generation
---
# EduHinglish — Gemma-3-4B-it LoRA adapter

Answers NCERT Class 9–10 questions (Biology, Chemistry, Physics, Mathematics, Social Science)
in Roman-script Hinglish, grounded in a retrieved textbook passage. Part of the EduHinglish
project (K.J. Somaiya College of Engineering).

Adapter: LoRA r={r}, alpha={alpha}, targets {targets}.

## Prompt format (single user turn, Gemma chat template)
```
Context:
<NCERT passage>

Question:
<student question>

Intent Type: <intent>          # optional line
Answer the question in Hinglish using only the given context.
```

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
base = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo}")
```
Use of this adapter is subject to the Gemma Terms of Use.
"""

INTENT_CARD = """---
base_model: google/muril-base-cased
library_name: transformers
license: apache-2.0
language: [en, hi]
tags: [hinglish, intent-classification, education, code-mixing]
pipeline_tag: text-classification
---
# EduHinglish — MuRIL query-intent classifier

Classifies a student's Hinglish/English question into one of {n} intents
({labels}). The prediction is passed to the EduHinglish Gemma generator as a hint.

```python
from transformers import pipeline
clf = pipeline("text-classification", model="{repo}")
clf("Photosynthesis kya hota hai?")
```
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["gemma", "intent"])
    ap.add_argument("folder", type=Path)
    ap.add_argument("repo_id", help="e.g. your-hf-user/eduhinglish-gemma-3-4b-it-lora")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    folder = args.folder if args.folder.is_absolute() else PROJECT_ROOT / args.folder
    if args.kind == "gemma":
        need = ["adapter_config.json", "adapter_model.safetensors"]
    else:
        need = ["config.json", "model.safetensors"]
    missing = [f for f in need if not (folder / f).exists()]
    if missing:
        sys.exit(f"[FAIL] {folder} is missing {missing}")

    if args.kind == "gemma":
        cfg = json.loads((folder / "adapter_config.json").read_text())
        card = GEMMA_CARD.format(r=cfg["r"], alpha=cfg["lora_alpha"],
                                 targets=", ".join(sorted(cfg["target_modules"])), repo=args.repo_id)
    else:
        cfg = json.loads((folder / "config.json").read_text())
        labels = [cfg["id2label"][k] for k in sorted(cfg["id2label"], key=int)]
        card = INTENT_CARD.format(n=len(labels), labels=", ".join(labels), repo=args.repo_id)

    api = HfApi()
    api.create_repo(args.repo_id, private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, folder_path=str(folder),
                      ignore_patterns=["README.md", "checkpoint-*", "*.pt", "optimizer*", "training_args.bin"],
                      commit_message=f"Upload {args.kind} model from {folder.name}")
    api.upload_file(repo_id=args.repo_id, path_or_fileobj=card.encode(), path_in_repo="README.md",
                    commit_message="Model card")
    print(f"[OK] https://huggingface.co/{args.repo_id}  ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
