"""
EduHinglish — Class 10 Social Science PDF Extractor & Standardizer
==================================================================
Author  : Ashvatth & Antigravity
Module  : M1 — Input Processing (Class 10 Social Science)
Purpose : Renames raw NCERT Class 10 Social Science PDFs to standardized
          consistent names and extracts/cleans all 22 chapters into
          data/processed/ss10_<subject>_ch<num>/ with raw_text.txt,
          cleaned_text.txt, and chapter_pages.json.

Usage:
    python scripts/extract_ss10_social_science.py
    python scripts/extract_ss10_social_science.py --force
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

# Add src/ to path so we can import NCERTPDFExtractor
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pdf_extractor import NCERTPDFExtractor

# ─────────────────────────────────────────────────────────────
# CHAPTER DEFINITIONS
# ─────────────────────────────────────────────────────────────
SS10_CHAPTERS = [
    # ── Geography (Contemporary India - II) ──────────────────
    {
        "subject": "Geography",
        "subj_code": "geo",
        "raw_folder": "10th-geo",
        "chapter_num": 1,
        "chapter_str": "ch01",
        "title": "Resources and Development",
        "old_names": ["jess101.pdf"],
        "std_pdf_name": "ss10_geo_ch01_resources_and_development.pdf",
        "out_folder": "ss10_geo_ch01",
    },
    {
        "subject": "Geography",
        "subj_code": "geo",
        "raw_folder": "10th-geo",
        "chapter_num": 2,
        "chapter_str": "ch02",
        "title": "Forest and Wildlife Resources",
        "old_names": ["jess102.pdf"],
        "std_pdf_name": "ss10_geo_ch02_forest_and_wildlife.pdf",
        "out_folder": "ss10_geo_ch02",
    },
    {
        "subject": "Geography",
        "subj_code": "geo",
        "raw_folder": "10th-geo",
        "chapter_num": 3,
        "chapter_str": "ch03",
        "title": "Water Resources",
        "old_names": ["jess103.pdf"],
        "std_pdf_name": "ss10_geo_ch03_water_resources.pdf",
        "out_folder": "ss10_geo_ch03",
    },
    {
        "subject": "Geography",
        "subj_code": "geo",
        "raw_folder": "10th-geo",
        "chapter_num": 4,
        "chapter_str": "ch04",
        "title": "Agriculture",
        "old_names": ["jess104.pdf"],
        "std_pdf_name": "ss10_geo_ch04_agriculture.pdf",
        "out_folder": "ss10_geo_ch04",
    },
    {
        "subject": "Geography",
        "subj_code": "geo",
        "raw_folder": "10th-geo",
        "chapter_num": 5,
        "chapter_str": "ch05",
        "title": "Minerals and Energy Resources",
        "old_names": ["jess105.pdf"],
        "std_pdf_name": "ss10_geo_ch05_minerals_and_energy.pdf",
        "out_folder": "ss10_geo_ch05",
    },
    {
        "subject": "Geography",
        "subj_code": "geo",
        "raw_folder": "10th-geo",
        "chapter_num": 6,
        "chapter_str": "ch06",
        "title": "Manufacturing Industries",
        "old_names": ["jess106.pdf"],
        "std_pdf_name": "ss10_geo_ch06_manufacturing_industries.pdf",
        "out_folder": "ss10_geo_ch06",
    },
    {
        "subject": "Geography",
        "subj_code": "geo",
        "raw_folder": "10th-geo",
        "chapter_num": 7,
        "chapter_str": "ch07",
        "title": "Lifelines of National Economy",
        "old_names": ["jess107.pdf"],
        "std_pdf_name": "ss10_geo_ch07_lifelines_of_national_economy.pdf",
        "out_folder": "ss10_geo_ch07",
    },

    # ── Economics (Understanding Economic Development) ───────
    {
        "subject": "Economics",
        "subj_code": "eco",
        "raw_folder": "10th-eco",
        "chapter_num": 1,
        "chapter_str": "ch01",
        "title": "Development",
        "old_names": ["jess201.pdf"],
        "std_pdf_name": "ss10_eco_ch01_development.pdf",
        "out_folder": "ss10_eco_ch01",
    },
    {
        "subject": "Economics",
        "subj_code": "eco",
        "raw_folder": "10th-eco",
        "chapter_num": 2,
        "chapter_str": "ch02",
        "title": "Sectors of the Indian Economy",
        "old_names": ["jess202.pdf"],
        "std_pdf_name": "ss10_eco_ch02_sectors_of_indian_economy.pdf",
        "out_folder": "ss10_eco_ch02",
    },
    {
        "subject": "Economics",
        "subj_code": "eco",
        "raw_folder": "10th-eco",
        "chapter_num": 3,
        "chapter_str": "ch03",
        "title": "Money and Credit",
        "old_names": ["jess203.pdf"],
        "std_pdf_name": "ss10_eco_ch03_money_and_credit.pdf",
        "out_folder": "ss10_eco_ch03",
    },
    {
        "subject": "Economics",
        "subj_code": "eco",
        "raw_folder": "10th-eco",
        "chapter_num": 4,
        "chapter_str": "ch04",
        "title": "Globalisation and the Indian Economy",
        "old_names": ["jess204.pdf"],
        "std_pdf_name": "ss10_eco_ch04_globalisation.pdf",
        "out_folder": "ss10_eco_ch04",
    },
    {
        "subject": "Economics",
        "subj_code": "eco",
        "raw_folder": "10th-eco",
        "chapter_num": 5,
        "chapter_str": "ch05",
        "title": "Consumer Rights",
        "old_names": ["jess205.pdf"],
        "std_pdf_name": "ss10_eco_ch05_consumer_rights.pdf",
        "out_folder": "ss10_eco_ch05",
    },

    # ── History (India and the Contemporary World - II) ─────
    {
        "subject": "History",
        "subj_code": "his",
        "raw_folder": "10th-his",
        "chapter_num": 1,
        "chapter_str": "ch01",
        "title": "The Rise of Nationalism in Europe",
        "old_names": ["jess301.pdf"],
        "std_pdf_name": "ss10_his_ch01_nationalism_in_europe.pdf",
        "out_folder": "ss10_his_ch01",
    },
    {
        "subject": "History",
        "subj_code": "his",
        "raw_folder": "10th-his",
        "chapter_num": 2,
        "chapter_str": "ch02",
        "title": "Nationalism in India",
        "old_names": ["jess302.pdf"],
        "std_pdf_name": "ss10_his_ch02_nationalism_in_india.pdf",
        "out_folder": "ss10_his_ch02",
    },
    {
        "subject": "History",
        "subj_code": "his",
        "raw_folder": "10th-his",
        "chapter_num": 3,
        "chapter_str": "ch03",
        "title": "The Making of a Global World",
        "old_names": ["jess303.pdf"],
        "std_pdf_name": "ss10_his_ch03_making_of_global_world.pdf",
        "out_folder": "ss10_his_ch03",
    },
    {
        "subject": "History",
        "subj_code": "his",
        "raw_folder": "10th-his",
        "chapter_num": 4,
        "chapter_str": "ch04",
        "title": "The Age of Industrialisation",
        "old_names": ["jess304.pdf"],
        "std_pdf_name": "ss10_his_ch04_age_of_industrialisation.pdf",
        "out_folder": "ss10_his_ch04",
    },
    {
        "subject": "History",
        "subj_code": "his",
        "raw_folder": "10th-his",
        "chapter_num": 5,
        "chapter_str": "ch05",
        "title": "Print Culture and the Modern World",
        "old_names": ["jess305.pdf"],
        "std_pdf_name": "ss10_his_ch05_print_culture.pdf",
        "out_folder": "ss10_his_ch05",
    },

    # ── Civics (Democratic Politics - II) ────────────────────
    {
        "subject": "Civics",
        "subj_code": "civics",
        "raw_folder": "10th-civics",
        "chapter_num": 1,
        "chapter_str": "ch01",
        "title": "Power-sharing",
        "old_names": ["jess401.pdf"],
        "std_pdf_name": "ss10_civics_ch01_power_sharing.pdf",
        "out_folder": "ss10_civics_ch01",
    },
    {
        "subject": "Civics",
        "subj_code": "civics",
        "raw_folder": "10th-civics",
        "chapter_num": 2,
        "chapter_str": "ch02",
        "title": "Federalism",
        "old_names": ["jess402.pdf"],
        "std_pdf_name": "ss10_civics_ch02_federalism.pdf",
        "out_folder": "ss10_civics_ch02",
    },
    {
        "subject": "Civics",
        "subj_code": "civics",
        "raw_folder": "10th-civics",
        "chapter_num": 3,
        "chapter_str": "ch03",
        "title": "Gender, Religion and Caste",
        "old_names": ["jess403.pdf"],
        "std_pdf_name": "ss10_civics_ch03_gender_religion_caste.pdf",
        "out_folder": "ss10_civics_ch03",
    },
    {
        "subject": "Civics",
        "subj_code": "civics",
        "raw_folder": "10th-civics",
        "chapter_num": 4,
        "chapter_str": "ch04",
        "title": "Political Parties",
        "old_names": ["jess404.pdf"],
        "std_pdf_name": "ss10_civics_ch04_political_parties.pdf",
        "out_folder": "ss10_civics_ch04",
    },
    {
        "subject": "Civics",
        "subj_code": "civics",
        "raw_folder": "10th-civics",
        "chapter_num": 5,
        "chapter_str": "ch05",
        "title": "Outcomes of Democracy",
        "old_names": ["jess405.pdf"],
        "std_pdf_name": "ss10_civics_ch05_outcomes_of_democracy.pdf",
        "out_folder": "ss10_civics_ch05",
    },
]


def flatten_subdirectories(raw_dir: Path):
    """Flatten any nested subdirectories like 10th-geo/jess1dd/."""
    for folder in ["10th-geo", "10th-eco", "10th-his", "10th-civics"]:
        fpath = raw_dir / folder
        if not fpath.exists():
            continue
        for child in list(fpath.iterdir()):
            if child.is_dir():
                for p in child.glob("*.pdf"):
                    dest = fpath / p.name
                    if not dest.exists():
                        shutil.move(str(p), str(dest))
                        print(f"  [MOVED] {p.name} -> {folder}/{p.name}")
                try:
                    child.rmdir()
                    print(f"  [CLEANED] Removed empty directory {child.name}")
                except OSError:
                    pass


def standardize_pdf_filenames(raw_dir: Path):
    """Rename raw PDF files to standardized names."""
    print("\n" + "=" * 70)
    print("STEP 1: STANDARDIZING RAW PDF FILENAMES")
    print("=" * 70)

    for item in SS10_CHAPTERS:
        subject_dir = raw_dir / item["raw_folder"]
        std_path = subject_dir / item["std_pdf_name"]

        if std_path.exists():
            print(f"  [EXISTS] {item['raw_folder']}/{item['std_pdf_name']}")
            continue

        # Look for old names
        found = False
        for old_name in item["old_names"]:
            old_path = subject_dir / old_name
            if old_path.exists():
                old_path.rename(std_path)
                print(f"  [RENAMED] {old_name} -> {item['std_pdf_name']}")
                found = True
                break
        
        if not found:
            # Check if matching pdf exists with similar name
            for f in subject_dir.glob("*.pdf"):
                if item["old_names"][0].lower() in f.name.lower():
                    f.rename(std_path)
                    print(f"  [RENAMED] {f.name} -> {item['std_pdf_name']}")
                    found = True
                    break

        if not found and not std_path.exists():
            print(f"  [WARN] Could not find raw PDF for {item['title']} in {subject_dir}")


def extract_all_chapters(raw_dir: Path, processed_dir: Path, force: bool = False):
    """Extract, clean, and save text + page data for all 22 chapters."""
    print("\n" + "=" * 70)
    print("STEP 2: EXTRACTING & CLEANING CHAPTER TEXT")
    print("=" * 70)

    results = []

    for item in SS10_CHAPTERS:
        subject_dir = raw_dir / item["raw_folder"]
        pdf_path = subject_dir / item["std_pdf_name"]
        if not pdf_path.exists():
            # Fallback to check old names
            for old_name in item["old_names"]:
                cand = subject_dir / old_name
                if cand.exists():
                    pdf_path = cand
                    break

        if not pdf_path.exists():
            print(f"  [MISSING] PDF not found: {item['std_pdf_name']}")
            results.append({"item": item, "status": "MISSING", "clean_chars": 0, "pages": 0})
            continue

        out_folder = processed_dir / item["out_folder"]
        raw_txt_path = out_folder / "raw_text.txt"
        clean_txt_path = out_folder / "cleaned_text.txt"
        pages_json_path = out_folder / "chapter_pages.json"

        if clean_txt_path.exists() and raw_txt_path.exists() and pages_json_path.exists() and not force:
            print(f"  [SKIPPED] {item['out_folder']} ({item['title']}) already exists. Use --force to redo.")
            clean_size = clean_txt_path.stat().st_size
            results.append({"item": item, "status": "SKIPPED", "clean_chars": clean_size, "pages": 0})
            continue

        print(f"\n>> Processing {item['out_folder']} : {item['title']} ({item['subject']})")
        print(f"   PDF: {pdf_path.name}")

        try:
            extractor = NCERTPDFExtractor(str(pdf_path))
            extractor.extract_text()
            extractor.clean_extracted_text()

            # Create output folder
            out_folder.mkdir(parents=True, exist_ok=True)

            # Save raw and cleaned text with explicit utf-8
            with open(raw_txt_path, "w", encoding="utf-8") as f:
                f.write(extractor.raw_text)

            with open(clean_txt_path, "w", encoding="utf-8") as f:
                f.write(extractor.clean_text)

            # Build chapter_pages.json
            pages_summary = []
            for p in extractor.pages_data:
                pages_summary.append({
                    "page": p["page_number"],
                    "chars": p["char_count"]
                })

            chapter_meta = {
                "chapter": item["chapter_num"],
                "title": item["title"],
                "subject": item["subject"],
                "chapter_id": item["chapter_str"],
                "total_pages": len(extractor.pages_data),
                "pages": pages_summary
            }

            with open(pages_json_path, "w", encoding="utf-8") as f:
                json.dump(chapter_meta, f, indent=2, ensure_ascii=False)

            print(f"  [SAVED] {item['out_folder']}/ -> raw_text.txt, cleaned_text.txt, chapter_pages.json")
            print(f"  [STATS] Raw: {len(extractor.raw_text):,} chars | Clean: {len(extractor.clean_text):,} chars | Pages: {len(extractor.pages_data)}")

            results.append({
                "item": item,
                "status": "OK",
                "clean_chars": len(extractor.clean_text),
                "pages": len(extractor.pages_data)
            })

        except Exception as e:
            print(f"  [ERROR] Failed to extract {item['title']}: {e}")
            results.append({"item": item, "status": "ERROR", "clean_chars": 0, "pages": 0})

    return results


def print_final_summary(results: list, processed_dir: Path):
    """Print complete summary table."""
    print("\n" + "=" * 90)
    print("EXTRACTION SUMMARY TABLE — CLASS 10 SOCIAL SCIENCE")
    print("=" * 90)
    print(f"{'Folder':<22} {'Subject':<12} {'Chapter Title':<38} {'Pages':<7} {'Status'}")
    print("-" * 90)

    ok_count = 0
    for r in results:
        item = r["item"]
        folder = item["out_folder"]
        subj = item["subject"]
        title = item["title"][:36]
        pgs = r["pages"]
        status = r["status"]
        if status in ("OK", "SKIPPED"):
            ok_count += 1
        print(f"{folder:<22} {subj:<12} {title:<38} {pgs:<7} {status}")

    print("=" * 90)
    print(f"Total Available: {ok_count} / {len(SS10_CHAPTERS)} chapters")
    print(f"Output Directory: {processed_dir.resolve()}")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract Class 10 Social Science NCERT PDFs")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing extracted chapters")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    print("=" * 70)
    print("EduHinglish — Class 10 Social Science Extraction Pipeline")
    print(f"Raw Directory      : {raw_dir}")
    print(f"Processed Directory: {processed_dir}")
    print("=" * 70)

    # 1. Flatten nested directories
    flatten_subdirectories(raw_dir)

    # 2. Standardize raw PDF filenames
    standardize_pdf_filenames(raw_dir)

    # 3. Extract and clean chapters
    results = extract_all_chapters(raw_dir, processed_dir, force=args.force)

    # 4. Print summary
    print_final_summary(results, processed_dir)


if __name__ == "__main__":
    main()
