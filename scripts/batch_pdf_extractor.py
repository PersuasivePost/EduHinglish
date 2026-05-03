"""
EduHinglish — Batch PDF Extractor
===================================
Author  : Ashvatth
Module  : M1 — Input Processing (Step 0, Batch Version)
Purpose : Extract and clean text from ALL NCERT Biology chapter PDFs
          in one run. Reuses the existing NCERTPDFExtractor class from
          src/pdf_extractor.py. Outputs raw_text.txt and cleaned_text.txt
          to data/processed/<chapter_key>/ for each chapter.

Usage:
    python scripts/batch_pdf_extractor.py
    python scripts/batch_pdf_extractor.py --raw-dir data/raw --output-dir data/processed
    python scripts/batch_pdf_extractor.py --force          # overwrite existing
    python scripts/batch_pdf_extractor.py --list           # list all PDFs found

Data directory scanned:
    data/raw/biology/class_ix/   — Class 9 PDFs
    data/raw/biology/class_x/    — Class 10 PDFs

Output structure:
    data/processed/
    ├── class9_ch05/
    │   ├── raw_text.txt
    │   └── cleaned_text.txt
    ├── class9_ch06/
    │   ├── raw_text.txt
    │   └── cleaned_text.txt
    └── ... (one folder per chapter)
"""

import sys
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Add src/ to path so we can import NCERTPDFExtractor
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pdf_extractor import NCERTPDFExtractor  # Ashvatth's existing extractor


# ─────────────────────────────────────────────────────────────
# CHAPTER MAP
# ─────────────────────────────────────────────────────────────
# Maps chapter_key → chapter metadata + expected PDF filename(s).
#
# HOW TO READ THIS:
#   "chapter_key"  → the folder name under data/processed/
#   "pdf_names"    → list of filenames to look for in data/raw/
#                    (searches recursively; first match wins)
#   "title"        → human-readable chapter title
#   "class"        → "9" or "10"
#   "chapter_num"  → NCERT chapter number
#
# ADDING A NEW PDF:
#   If you have the PDF but with a different filename, add it to the
#   "pdf_names" list for that chapter. The script tries each name
#   in order and uses the first one it finds.
#
# STANDARD NAMING CONVENTION (what to name your PDFs when you add them):
#   Class 9 : bio_9_ch05.pdf, bio_9_ch06.pdf, ... bio_9_ch15.pdf
#   Class 10: bio_10_ch06.pdf, bio_10_ch08.pdf, ... bio_10_ch16.pdf
# ─────────────────────────────────────────────────────────────

CHAPTER_MAP = {

    # ── Class 9 Biology ──────────────────────────────────────────────────────

    "class9_ch05": {
        "title"      : "The Fundamental Unit of Life",
        "class"      : "9",
        "chapter_num": "05",
        "pdf_names"  : [
            "bio_9_ch05.pdf",         # standard name (add your PDF with this name)
            "ix_chap5_science.pdf",   # alternate NCERT naming
            "class9_ch5.pdf",
            "ch5_class9.pdf",
        ],
    },

    "class9_ch06": {
        "title"      : "Tissues",
        "class"      : "9",
        "chapter_num": "06",
        "pdf_names"  : [
            "bio_9_ch06.pdf",
            "ix_chap6_science.pdf",
            "class9_ch6.pdf",
            "ch6_class9.pdf",
        ],
    },

    "class9_ch07": {
        "title"      : "Diversity in Living Organisms",
        "class"      : "9",
        "chapter_num": "07",
        "pdf_names"  : [
            "bio_9_ch07.pdf",
            "ix_chap7_science.pdf",
            "class9_ch7.pdf",
            "ch7_class9.pdf",
        ],
    },

    "class9_ch13": {
        "title"      : "Why Do We Fall Ill",
        "class"      : "9",
        "chapter_num": "13",
        "pdf_names"  : [
            "bio_9_ch13.pdf",
            "ix_chap13_science.pdf",
            "class9_ch13.pdf",
            "ch13_class9.pdf",
        ],
    },

    "class9_ch14": {
        "title"      : "Natural Resources",
        "class"      : "9",
        "chapter_num": "14",
        "pdf_names"  : [
            "bio_9_ch14.pdf",
            "ix_chap14_science.pdf",
            "class9_ch14.pdf",
            "ch14_class9.pdf",
        ],
    },

    "class9_ch15": {
        "title"      : "Improvement in Food Resources",
        "class"      : "9",
        "chapter_num": "15",
        "pdf_names"  : [
            "bio_9_ch15.pdf",
            "ix_chap15_science.pdf",
            "class9_ch15.pdf",
            "ch15_class9.pdf",
        ],
    },

    # ── Class 10 Biology ─────────────────────────────────────────────────────

    "class10_ch06": {
        "title"      : "Life Processes",
        "class"      : "10",
        "chapter_num": "06",
        "pdf_names"  : [
            "bio_10_ch06.pdf",
            "x_chap6_science.pdf",    # this file EXISTS in data/raw/biology/class_x/
            "class10_ch6.pdf",
            "ch6_class10.pdf",
        ],
    },

    "class10_ch08": {
        "title"      : "How do Organisms Reproduce?",
        "class"      : "10",
        "chapter_num": "08",
        "pdf_names"  : [
            "bio_10_ch08.pdf",
            "x_chap8_science.pdf",    # this file EXISTS in data/raw/biology/class_x/
            "class10_ch8.pdf",
            "ch8_class10.pdf",
        ],
    },

    "class10_ch09": {
        "title"      : "Heredity and Evolution",
        "class"      : "10",
        "chapter_num": "09",
        "pdf_names"  : [
            "bio_10_ch09.pdf",
            "x_chap9_science.pdf",
            "class10_ch9.pdf",
            "ch9_class10.pdf",
        ],
    },

    "class10_ch15": {
        "title"      : "Our Environment",
        "class"      : "10",
        "chapter_num": "15",
        "pdf_names"  : [
            "bio_10_ch15.pdf",
            "x_chap15_science.pdf",
            "class10_ch15.pdf",
            "ch15_class10.pdf",
        ],
    },

    "class10_ch16": {
        "title"      : "Management of Natural Resources",
        "class"      : "10",
        "chapter_num": "16",
        "pdf_names"  : [
            "bio_10_ch16.pdf",
            "x_chap16_science.pdf",
            "class10_ch16.pdf",
            "ch16_class10.pdf",
        ],
    },
}


# ─────────────────────────────────────────────────────────────
# HELPER — find a PDF by scanning data/raw/ recursively
# ─────────────────────────────────────────────────────────────

def find_pdf(raw_dir: Path, pdf_names: list) -> Path | None:
    """
    Search raw_dir recursively for any filename in pdf_names.
    Returns the first match found, or None if nothing matches.
    """
    for name in pdf_names:
        # Direct match in any subdirectory
        matches = list(raw_dir.rglob(name))
        if matches:
            return matches[0]
    return None


# ─────────────────────────────────────────────────────────────
# MAIN EXTRACTION FUNCTION
# ─────────────────────────────────────────────────────────────

def run_batch(raw_dir: Path, output_dir: Path, force: bool = False) -> dict:
    """
    Iterate over CHAPTER_MAP, find each PDF, extract + clean, save outputs.

    Returns:
        {
          "extracted": [...],   # chapter keys successfully processed
          "skipped"  : [...],   # already extracted (no --force)
          "missing"  : [...],   # PDF not found in raw_dir
          "errors"   : [...],   # extraction crashed
        }
    """
    results = {
        "extracted": [],
        "skipped"  : [],
        "missing"  : [],
        "errors"   : [],
    }

    print()
    print("=" * 70)
    print("  EduHinglish — Batch PDF Extractor")
    print("=" * 70)
    print(f"  Raw PDF dir : {raw_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Force mode  : {'YES — will overwrite' if force else 'NO — skip if exists'}")
    print(f"  Chapters    : {len(CHAPTER_MAP)}")
    print("=" * 70)
    print()

    for chapter_key, info in CHAPTER_MAP.items():
        title       = info["title"]
        pdf_names   = info["pdf_names"]
        out_folder  = output_dir / chapter_key
        clean_path  = out_folder / "cleaned_text.txt"
        raw_path    = out_folder / "raw_text.txt"

        # ── Check if already extracted ────────────────────────
        if clean_path.exists() and not force:
            print(f"  [SKIP]    {chapter_key:<20} -> '{title}' - already extracted (use --force to redo)")
            results["skipped"].append(chapter_key)
            continue

        # ── Find the PDF ──────────────────────────────────────
        pdf_path = find_pdf(raw_dir, pdf_names)

        if pdf_path is None:
            print(f"  [MISSING] {chapter_key:<20} -> '{title}'")
            print(f"            Place one of these in data/raw/ (or any subfolder):")
            for name in pdf_names[:2]:   # show top 2 expected names
                print(f"              * {name}")
            results["missing"].append(chapter_key)
            continue

        # ── Extract and clean ─────────────────────────────────
        print(f"  [RUN]     {chapter_key:<20} -> '{title}'")
        print(f"            PDF: {pdf_path.relative_to(raw_dir.parent)}")

        try:
            extractor = NCERTPDFExtractor(str(pdf_path))

            # Suppress the verbose page-by-page output for batch runs
            # by redirecting; we only care about the final summary line
            extractor.extract_text()
            extractor.clean_extracted_text()

            # Create output folder
            out_folder.mkdir(parents=True, exist_ok=True)

            # Save both files using explicit UTF-8 encoding.
            # We write directly here (not via extractor.save_*) because
            # src/pdf_extractor.py opens files without encoding= which fails
            # on Windows cp1252 when NCERT PDFs contain special characters.
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(extractor.raw_text)
            print(f"  [SAVED] Raw text     -> {raw_path}")

            with open(clean_path, "w", encoding="utf-8") as f:
                f.write(extractor.clean_text)
            print(f"  [SAVED] Cleaned text -> {clean_path}")

            clean_chars = len(extractor.clean_text)
            raw_chars   = len(extractor.raw_text)

            print(
                f"  [OK]      {chapter_key:<20} -> '{title}' "
                f"({clean_chars:,} chars cleaned / {raw_chars:,} raw)"
            )
            results["extracted"].append(chapter_key)

        except Exception as exc:
            print(f"  [ERROR]   {chapter_key:<20} -> {exc}")
            results["errors"].append(chapter_key)

        print()

    return results


# ─────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def print_summary(results: dict, output_dir: Path):
    """Print a final summary table after the batch run."""
    n_ok      = len(results["extracted"])
    n_skip    = len(results["skipped"])
    n_miss    = len(results["missing"])
    n_err     = len(results["errors"])
    total     = len(CHAPTER_MAP)

    print()
    print("=" * 70)
    print("  BATCH EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"  Extracted : {n_ok:>3}  chapters")
    print(f"  Skipped   : {n_skip:>3}  (already done)")
    print(f"  Missing   : {n_miss:>3}  PDFs not found in data/raw/")
    print(f"  Errors    : {n_err:>3}  (extraction failed)")
    print(f"  Total     : {n_ok + n_skip:>3} / {total} chapters available")
    print("=" * 70)

    if results["missing"]:
        print()
        print("  MISSING PDFs — download from ncert.nic.in and place in data/raw/")
        print("  Rename to the standard format:  bio_CLASS_chCHAPTER.pdf")
        print()
        for key in results["missing"]:
            info = CHAPTER_MAP[key]
            std_name = f"bio_{info['class']}_ch{info['chapter_num']}.pdf"
            print(f"    {key:<20} -> {std_name}  ({info['title']})")

    if results["extracted"]:
        print()
        print(f"  Output saved to: {output_dir}/")
        for key in results["extracted"]:
            folder = output_dir / key
            cfile  = folder / "cleaned_text.txt"
            size   = cfile.stat().st_size if cfile.exists() else 0
            print(f"    {key:<20} -> {folder.name}/cleaned_text.txt  ({size:,} bytes)")

    if results["errors"]:
        print()
        print("  ERRORS — these chapters failed:")
        for key in results["errors"]:
            print(f"    {key}")

    print()


# ─────────────────────────────────────────────────────────────
# LIST MODE — show all PDFs currently in data/raw/
# ─────────────────────────────────────────────────────────────

def list_raw_pdfs(raw_dir: Path):
    """Print every PDF found in raw_dir (recursive) with its path."""
    print()
    print("=" * 70)
    print(f"  PDFs found in {raw_dir}  (recursive scan)")
    print("=" * 70)

    pdfs = sorted(raw_dir.rglob("*.pdf"))
    if not pdfs:
        print("  (none found)")
    else:
        for p in pdfs:
            rel = p.relative_to(raw_dir)
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {str(rel):<55} {size_mb:>6.1f} MB")

    print(f"\n  Total: {len(pdfs)} PDF(s)")
    print()

    # Cross-reference with CHAPTER_MAP
    print("  CHAPTER MAP STATUS:")
    print(f"  {'Chapter Key':<22} {'Title':<40} {'PDF Found?'}")
    print(f"  {'-'*22} {'-'*40} ----------")
    for key, info in CHAPTER_MAP.items():
        found = find_pdf(raw_dir, info["pdf_names"])
        status = f"YES -> {found.name}" if found else "NO"
        print(f"  {key:<22} {info['title']:<40} {status}")
    print()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EduHinglish — Batch PDF Extractor for all 11 biology chapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/batch_pdf_extractor.py
  python scripts/batch_pdf_extractor.py --force
  python scripts/batch_pdf_extractor.py --list
  python scripts/batch_pdf_extractor.py --raw-dir data/raw --output-dir data/processed
        """,
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Directory containing NCERT Biology PDFs (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Root output directory for extracted text (default: data/processed)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite already-extracted chapters (default: skip existing)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all PDFs in raw-dir and their CHAPTER_MAP status, then exit",
    )
    parser.add_argument(
        "--chapter",
        type=str,
        default=None,
        metavar="KEY",
        help="Process a single chapter only, e.g. --chapter class9_ch06",
    )

    args = parser.parse_args()

    raw_dir    = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    # Validate raw dir
    if not raw_dir.exists():
        print(f"[ERROR] raw-dir does not exist: {raw_dir}")
        sys.exit(1)

    # ── List mode ─────────────────────────────────────────────
    if args.list:
        list_raw_pdfs(raw_dir)
        sys.exit(0)

    # ── Single chapter mode ───────────────────────────────────
    global CHAPTER_MAP
    if args.chapter:
        if args.chapter not in CHAPTER_MAP:
            print(f"[ERROR] Unknown chapter key: '{args.chapter}'")
            print(f"        Valid keys: {', '.join(CHAPTER_MAP.keys())}")
            sys.exit(1)

        # Temporarily restrict to that one chapter
        target = {args.chapter: CHAPTER_MAP[args.chapter]}
        CHAPTER_MAP = target

    # ── Batch run ─────────────────────────────────────────────
    results = run_batch(raw_dir, output_dir, force=args.force)
    print_summary(results, output_dir)


if __name__ == "__main__":
    main()
