"""
Publish the EduHinglish QA datasets to Kaggle (first upload or a new version).

Stages these files in outputs/kaggle_dataset/ and uploads them with the Kaggle CLI:
    bio_9_10.json  chem_9_10.json  phys_9_10.json  maths_9_10.json  ss_9_10.json
    README.md (dataset card with per-subject counts)
    ncert_text/<chapter>.txt   only with --with-text (cleaned NCERT text used for RAG)

Setup (once):
    pip install kaggle
    Kaggle → Settings → API → "Create New Token"  → save kaggle.json to ~/.kaggle/kaggle.json
    (or export KAGGLE_USERNAME / KAGGLE_KEY)

Usage:
    python scripts/publish_kaggle_dataset.py your-kaggle-user/eduhinglish-ncert-hinglish-qa
    python scripts/publish_kaggle_dataset.py your-kaggle-user/eduhinglish-ncert-hinglish-qa -m "add class 10 maths"
    python scripts/publish_kaggle_dataset.py ... --stage-only     # build the folder, don't upload
    python scripts/publish_kaggle_dataset.py ... --public         # first upload only; default is private
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STAGE_DIR = PROJECT_ROOT / "outputs" / "kaggle_dataset"
DATA_FILES = ["bio_9_10.json", "chem_9_10.json", "phys_9_10.json", "maths_9_10.json", "ss_9_10.json"]


def build_card(stats: dict, total: int, with_text: bool) -> str:
    rows = "\n".join(f"| `{f}` | {s['subject']} | {s['n']} | {s['c9']} | {s['c10']} | {s['chapters']} |"
                     for f, s in stats.items())
    text_note = ("\n- `ncert_text/<chapter>.txt` — cleaned NCERT chapter text used to build the RAG index.\n"
                 if with_text else "")
    return f"""# EduHinglish — NCERT Class 9–10 Hinglish QA

Curated question–answer pairs from NCERT Class 9 and 10 textbooks, each with an English and a
Roman-script **Hinglish** version, the supporting NCERT passage, and an intent label. Built for the
EduHinglish project (fine-tuning Gemma-3-4B-it to answer students' doubts in Hinglish).

**{total} records**

| File | Subject | Records | Class 9 | Class 10 | Chapters |
|---|---|---|---|---|---|
{rows}

## Fields
- `id`, `class`, `subject`, `sub_subject` (Social Science only), `chapter`, `chapter_num`, `topic`
- `intent` — one of 15 labels (definition, explain_concept, short_answer, describe_function,
  compare_concepts, list_enumerate, give_example, application_reallife, analytical,
  numerical_problem, mcq, assertion_reason, true_false, fill_blank, one_word)
- `question_english`, `answer_english`, `question_hinglish`, `answer_hinglish`
- `ncert_context` — the NCERT passage the answer is grounded in
- `messages` / `messages_english` — Gemma chat-format training pairs (Hinglish / English)
- `text` — `messages` rendered with the Gemma chat template (what the model is trained on)
- `intent_in_prompt` — whether the `Intent Type:` line was kept in the prompt (~12% dropped)
{text_note}
## Source & licence
Passages are taken from NCERT textbooks (© NCERT), used here for non-commercial educational
research. Questions, answers and Hinglish translations were written by the EduHinglish team.
"""


def stage(with_text: bool) -> None:
    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    STAGE_DIR.mkdir(parents=True)
    stats, total = {}, 0
    for name in DATA_FILES:
        records = json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
        classes = Counter(r["class"] for r in records)
        stats[name] = {"subject": records[0]["subject"], "n": len(records), "c9": classes["9"],
                       "c10": classes["10"], "chapters": len({(r["class"], r["chapter"]) for r in records})}
        total += len(records)
        shutil.copy2(DATA_DIR / name, STAGE_DIR / name)
        print(f"  {name:18s} {len(records):5d}")

    if with_text:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from chunker import NCERTChunker
        (STAGE_DIR / "ncert_text").mkdir()
        for folder in NCERTChunker.CHAPTER_MAP:
            src = DATA_DIR / "processed" / folder / "cleaned_text.txt"
            if src.exists():
                shutil.copy2(src, STAGE_DIR / "ncert_text" / f"{folder}.txt")
        print(f"  ncert_text/        {len(list((STAGE_DIR / 'ncert_text').iterdir())):5d} chapters")

    (STAGE_DIR / "README.md").write_text(build_card(stats, total, with_text), encoding="utf-8")
    print(f"  total              {total:5d}  → {STAGE_DIR.relative_to(PROJECT_ROOT)}")


def write_metadata(slug: str) -> None:
    meta = {
        "title": "EduHinglish NCERT Class 9-10 Hinglish QA",
        "subtitle": "Bio, Chem, Phys, Maths, Social Science QA pairs in English + Hinglish",
        "id": slug,
        "licenses": [{"name": "other"}],
        "keywords": ["education", "nlp", "india", "text generation"],
        "description": (STAGE_DIR / "README.md").read_text(encoding="utf-8"),
    }
    (STAGE_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug", help="kaggle-username/dataset-name")
    ap.add_argument("-m", "--message", default="Update EduHinglish datasets")
    ap.add_argument("--with-text", action="store_true", help="also include cleaned NCERT chapter text")
    ap.add_argument("--public", action="store_true", help="make the dataset public on first upload")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--new", action="store_true", default=None,
                    help="force 'create' (first upload) instead of auto-detecting")
    args = ap.parse_args()
    if args.slug.count("/") != 1:
        sys.exit("slug must look like kaggle-username/dataset-name")

    stage(args.with_text)
    write_metadata(args.slug)
    if args.stage_only:
        return

    if shutil.which("kaggle") is None:
        sys.exit("kaggle CLI not found: pip install kaggle")
    if args.new is None:
        probe = subprocess.run(["kaggle", "datasets", "files", args.slug], capture_output=True, text=True)
        out = (probe.stdout + probe.stderr).lower()
        exists = probe.returncode == 0 and "404" not in out and "not found" not in out
    else:
        exists = not args.new
    if exists:
        cmd = ["kaggle", "datasets", "version", "-p", str(STAGE_DIR), "-m", args.message, "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(STAGE_DIR), "--dir-mode", "zip"]
        if args.public:
            cmd.append("--public")
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[OK] https://www.kaggle.com/datasets/{args.slug}")


if __name__ == "__main__":
    main()
