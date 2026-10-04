"""
EduHinglish — PDF Text Extractor
=================================
Author  : Ashvatth
Module  : M1 — Input Processing (Step 0)
Purpose : Extract and clean text from NCERT Biology PDF chapter.
          Output is saved to data/processed/ for downstream pipeline use.

Usage:
    python pdf_extractor.py
    python pdf_extractor.py --pdf ../data/raw/ncert_class9_science.pdf
"""

import re
import json
import os
import argparse
import itertools
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Try importing pdfplumber; fall back with a helpful message
# ─────────────────────────────────────────────────────────────
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("[WARN] pdfplumber not installed. Run: pip install pdfplumber")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Constants — adjust if chapter pages differ
# ─────────────────────────────────────────────────────────────
CHAPTER_TITLE    = "The Fundamental Unit of Life"
CHAPTER_NUMBER   = 5
SUBJECT          = "Science"
CLASS_LEVEL      = "9"
TARGET_CHAPTER   = "chapter5"

# NCERT Ch5 typically spans pages 59–73 in the combined science PDF
# Set to None to process ALL pages
CHAPTER_START_PAGE = None
CHAPTER_END_PAGE   = None


class NCERTPDFExtractor:
    """
    Extract and clean text from NCERT textbook PDFs.

    Workflow:
        PDF  →  raw text  →  cleaned text  →  saved JSON + TXT
    """

    def __init__(self, pdf_path: str):
        self.pdf_path   = Path(pdf_path)
        self.raw_text   = ""
        self.clean_text = ""
        self.pages_data = []

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}\n"
                "Download NCERT Class 9 Science from https://ncert.nic.in/textbook.php\n"
                "Place it in: data/raw/"
            )

    # ──────────────────────────────────────────────────────────
    # EXTRACTION
    # ──────────────────────────────────────────────────────────

    def extract_text(self, start_page: int = None, end_page: int = None) -> str:
        """
        Extract raw text from PDF pages.

        Args:
            start_page: 1-indexed first page to extract (None = first page)
            end_page  : 1-indexed last  page to extract (None = last  page)
        """
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("Install pdfplumber: pip install pdfplumber")

        print(f"\n{'='*60}")
        print("STEP 0 — PDF TEXT EXTRACTION")
        print(f"{'='*60}")
        print(f"  File : {self.pdf_path.name}")

        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            s = (start_page - 1) if start_page else 0
            e = end_page if end_page else total_pages

            print(f"  Total pages : {total_pages}")
            print(f"  Extracting  : pages {s+1}–{e}")
            print()

            for i, page in enumerate(pdf.pages[s:e], start=s + 1):
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 20:
                    self.pages_data.append({
                        "page_number": i,
                        "raw_text"   : page_text,
                        "char_count" : len(page_text),
                    })
                    self.raw_text += page_text + "\n\n"
                    print(f"  [OK] Page {i:>3} — {len(page_text):>6} chars")
                else:
                    print(f"  [--] Page {i:>3} — skipped (image/empty)")

        print(f"\n  Total extracted : {len(self.raw_text):,} characters")
        return self.raw_text

    # ──────────────────────────────────────────────────────────
    # CLEANING
    # ──────────────────────────────────────────────────────────

    def clean_extracted_text(self, text: str = None) -> str:
        """
        Clean raw extracted text by removing PDF artefacts.

        Steps applied:
            1. Remove chapter headers / running titles
            2. Remove standalone page numbers
            3. Remove figure & table captions
            4. Fix broken words (across line-breaks)
            5. Fix hyphenation artefacts
            6. Normalize quotes and dashes
            7. Collapse excessive whitespace
            8. Strip non-content special characters
        """
        if text is None:
            text = self.raw_text

        print(f"\n{'='*60}")
        print("CLEANING EXTRACTED TEXT")
        print(f"{'='*60}")
        original_len = len(text)

        # 0 - Fix PDF extraction character duplication bugs (e.g., DDDDDOOOOO -> DO)
        def replacer(match):
            word = match.group(0)
            groups = [''.join(g) for k, g in itertools.groupby(word)]
            lengths = [len(g) for g in groups]
            if not lengths: return word
            if all(l >= 4 for l in lengths) or (len(lengths) > 1 and all(l >= 3 for l in lengths)):
                return ''.join(g[0] for g in groups)
            if len(lengths) == 1 and lengths[0] >= 4:
                if groups[0][0].isalpha() or not groups[0][0].isalnum():
                    return groups[0][0]
            return word

        words = text.split()
        text = " ".join([re.sub(r'\S+', replacer, w) for w in words])
        print("  [OK] Collapsed duplicated character artefacts")

        # 1 — Headers / running titles
        text = re.sub(
            r'(?i)(the fundamental unit of life|chapter\s*\d+|'
            r'science|class\s*(ix|x|9|10)|ncert)',
            '', text
        )
        print("  [OK] Removed headers and running titles")

        # 1.5 — QR codes and Sidebar artifacts
        text = re.sub(r'\b\d{4}CH\d{2}\b', '', text, flags=re.I)
        text = re.sub(r'(?i)\b(Think It Over)\b', '', text)
        print("  [OK] Removed QR codes and Sidebar artifacts")

        # 2 — Standalone page numbers
        text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
        print("  [OK] Removed page numbers")

        # 3 — Figure / table captions
        text = re.sub(r'Fig(?:ure|\.)\s*\d+[\.\d]*\s*[:\-]?\s*', '', text)
        text = re.sub(r'Table\s*\d+[\.\d]*\s*[:\-]?\s*', '', text)
        print("  [OK] Removed figure/table references")

        # 4 — Fix broken words across line-breaks
        #      e.g. "mem-\nbrane" → "membrane"
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        print("  [OK] Fixed hyphenation breaks")

        # 5 — Fix sentences split across lines (lowercase char after newline)
        text = re.sub(r'([a-z,;])\n([a-zA-Z])', r'\1 \2', text)
        print("  [OK] Fixed mid-sentence line breaks")

        # 6 — Normalize quotation marks and dashes
        for old, new in [('"', '"'), ('"', '"'), (''', "'"), (''', "'"),
                         ('–', '-'), ('—', '-'), ('…', '...')]:
            text = text.replace(old, new)
        print("  [OK] Normalized special characters")

        # 7 — Collapse excessive whitespace / blank lines
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        print("  [OK] Collapsed excessive whitespace")

        # 8 — Remove remaining non-content characters
        #      Keep: word chars, spaces, punctuation, scientific symbols
        text = re.sub(
            r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\%\°\/\+\=\²\³\₂\₃\→\←\↔]',
            '',
            text
        )
        print("  [OK] Removed non-content characters")

        self.clean_text = text.strip()

        reduction = (1 - len(self.clean_text) / original_len) * 100
        print(f"\n  Before : {original_len:,} chars")
        print(f"  After  : {len(self.clean_text):,} chars")
        print(f"  Reduced: {reduction:.1f}%")

        return self.clean_text

    # ──────────────────────────────────────────────────────────
    # STATISTICS
    # ──────────────────────────────────────────────────────────

    def get_text_stats(self) -> dict:
        """Print and return statistics about the cleaned text."""
        text      = self.clean_text or self.raw_text
        words     = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        paragraphs= [p for p in text.split('\n\n') if p.strip()]

        stats = {
            "characters" : len(text),
            "words"      : len(words),
            "sentences"  : len(sentences),
            "paragraphs" : len(paragraphs),
            "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 1),
        }

        print(f"\n{'='*50}")
        print("TEXT STATISTICS")
        print(f"{'='*50}")
        for k, v in stats.items():
            print(f"  {k:<28}: {v}")
        print(f"{'='*50}")

        return stats

    # ──────────────────────────────────────────────────────────
    # SAVING
    # ──────────────────────────────────────────────────────────

    def save_raw_text(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.raw_text)
        print(f"  [SAVED] Raw text     → {output_path}")

    def save_clean_text(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.clean_text)
        print(f"  [SAVED] Cleaned text → {output_path}")

    def save_pages_data(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.pages_data, f, indent=2, ensure_ascii=False)
        print(f"  [SAVED] Pages JSON   → {output_path}")

    def preview(self, n_chars: int = 600):
        """Print first n_chars of clean text as a sanity check."""
        text = self.clean_text or self.raw_text
        print(f"\n{'='*60}")
        print(f"PREVIEW — first {n_chars} characters of clean text")
        print(f"{'='*60}")
        print(text[:n_chars])
        print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# CLI / direct run
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract text from NCERT PDF")
    parser.add_argument(
        "--pdf",
        default=str(Path(__file__).parent.parent / "data" / "raw" / "ncert_class9_science.pdf"),
        help="Path to the NCERT PDF file",
    )
    parser.add_argument("--start-page", type=int, default=CHAPTER_START_PAGE)
    parser.add_argument("--end-page",   type=int, default=CHAPTER_END_PAGE)
    args = parser.parse_args()

    extractor = NCERTPDFExtractor(args.pdf)

    # Extract
    extractor.extract_text(
        start_page=args.start_page,
        end_page=args.end_page,
    )

    # Clean
    extractor.clean_extracted_text()

    # Save
    base = Path(__file__).parent.parent / "data" / "processed"
    extractor.save_raw_text(str(base / "chapter_text_raw.txt"))
    extractor.save_clean_text(str(base / "chapter_text_clean.txt"))
    extractor.save_pages_data(str(base / "chapter_pages.json"))

    # Stats + preview
    extractor.get_text_stats()
    extractor.preview()


if __name__ == "__main__":
    main()