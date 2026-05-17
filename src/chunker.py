"""
EduHinglish — NCERT Chapter Chunker
====================================
Author  : Ashvatth
Module  : M2 — NCERT Retriever (Phase 5, Task 1)
Purpose : Split cleaned NCERT chapter text into paragraph-level chunks
          with metadata, suitable for embedding and vector storage.

Usage:
    python src/chunker.py
    python src/chunker.py --chunk-size 150 --overlap 20
    python src/chunker.py --save-json src/chunks_preview.json
"""

import re
import json
import argparse
from pathlib import Path
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


class NCERTChunker:
    """
    Splits cleaned NCERT chapter text into paragraph-level chunks
    with rich metadata for vector storage and retrieval.

    Each chunk targets ~200 words with 30-word overlap between
    consecutive chunks to preserve context at boundaries.
    """

    # ── Chapter metadata mapping ─────────────────────────────────────────────
    CHAPTER_MAP = {
        "class9_ch01": {
            "subject": "biology", "class_num": "9", "chapter_num": "ch01",
            "chapter_title": "Exploration: Entering the World of Secondary Science",
        },
        "class9_ch02": {
            "subject": "biology", "class_num": "9", "chapter_num": "ch02",
            "chapter_title": "Cell: The Building Block of Life",
        },
        "class9_ch03": {
            "subject": "biology", "class_num": "9", "chapter_num": "ch03",
            "chapter_title": "Tissues in Action",
        },
        "class9_ch11": {
            "subject": "biology", "class_num": "9", "chapter_num": "ch11",
            "chapter_title": "Reproduction: How Life Continues",
        },
        "class9_ch12": {
            "subject": "biology", "class_num": "9", "chapter_num": "ch12",
            "chapter_title": "Patterns in Life: Diversity and Classification",
        },
        "class10_ch05": {
            "subject": "biology", "class_num": "10", "chapter_num": "ch05",
            "chapter_title": "Life Processes",
        },
        "class10_ch06": {
            "subject": "biology", "class_num": "10", "chapter_num": "ch06",
            "chapter_title": "Control and Coordination",
        },
        "class10_ch07": {
            "subject": "biology", "class_num": "10", "chapter_num": "ch07",
            "chapter_title": "How do Organisms Reproduce?",
        },
        "class10_ch08": {
            "subject": "biology", "class_num": "10", "chapter_num": "ch08",
            "chapter_title": "Heredity",
        },
        "class10_ch13": {
            "subject": "biology", "class_num": "10", "chapter_num": "ch13",
            "chapter_title": "Our Environment",
        },
    }

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 30):
        """
        Args:
            chunk_size:    Target number of words per chunk.
            chunk_overlap: Number of overlapping words between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[OK] NCERTChunker initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")

    # ── Text cleaning ────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """
        Remove OCR artifacts, page markers, and formatting noise
        from cleaned_text.txt content.
        """
        lines = text.split("\n")
        cleaned = []

        for line in lines:
            stripped = line.strip()

            # Keep blank lines (used later as paragraph separators)
            if not stripped:
                cleaned.append("")
                continue

            # Remove lines that are purely page markers
            if re.match(r"^Reprint\s+\d{4}[-–]\d{2,4}$", stripped, re.IGNORECASE):
                continue
            if "Reprint 2025-26" in stripped or "Reprint 2024-25" in stripped:
                continue

            # Remove lines that are purely figure references
            if re.match(r"^(Fig\.|Figure|Table)\s*[\d.]+", stripped, re.IGNORECASE):
                continue

            # Remove lines that are purely Activity headers
            if re.match(r"^Activity\s*_{2,}\s*\d+", stripped):
                continue

            # Remove garbled OCR headers (repeated block capitals like "TTTTT FFFFF UUUUU")
            if re.match(r"^[A-Z]{3,}(\s+[A-Z]{3,}){2,}$", stripped):
                continue

            # Remove garbled chapter footers (repeated digits like "5555588888")
            if re.match(r"^\d{5,}(\s*\d{5,})*\s*$", stripped):
                continue

            # Remove reversed text artifacts (e.g. "wonk ot eroM", "rethguaD sllec")
            if stripped in ("wonk ot eroM", "rethguaD sllec", "Meiosis-II", "Meiosis-I"):
                continue

            # Remove garbled double-letter headers ("CChhaapptteerr...")
            if re.match(r"^[A-Z][a-z][A-Z][a-z]", stripped) and len(stripped) > 10:
                # Check if characters repeat in pairs
                pairs_match = True
                clean_chars = stripped.replace(" ", "").replace("-", "").replace(".", "")
                for i in range(0, len(clean_chars) - 1, 2):
                    if i + 1 < len(clean_chars) and clean_chars[i].lower() != clean_chars[i + 1].lower():
                        pairs_match = False
                        break
                if pairs_match and len(clean_chars) >= 10:
                    continue

            # Remove lines shorter than 20 characters (likely headers/artifacts)
            if len(stripped) < 20:
                continue

            cleaned.append(stripped)

        # Collapse multiple consecutive blank lines into a single blank line
        result = "\n".join(cleaned)
        result = re.sub(r"\n{3,}", "\n\n", result)

        # Remove inline OCR garbled patterns (e.g. "AAAAAccccctttttiiiiivvvvviiiiitttttyyyyy 77777.....33333")
        # These are character-repeated figure/activity references embedded within text
        result = re.sub(r"[A-Z]{4,}[a-z]{4,}[A-Z][\w.()]*(?:\s+[\d.()]{4,})*", "", result)

        # Clean up any resulting double-spaces
        result = re.sub(r"  +", " ", result)

        return result.strip()

    # ── Paragraph splitting & merging ────────────────────────────────────────

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split cleaned text into paragraphs using double-newline."""
        paragraphs = re.split(r"\n\n+", text)
        return [p.strip().replace("\n", " ") for p in paragraphs if p.strip()]

    def _merge_short_paragraphs(self, paragraphs: list[str], min_words: int = 30) -> list[str]:
        """
        Merge paragraphs with fewer than min_words words into the
        next paragraph to avoid tiny, low-context chunks.
        """
        if not paragraphs:
            return []

        merged = []
        buffer = ""

        for para in paragraphs:
            if buffer:
                buffer = buffer + " " + para
            else:
                buffer = para

            if len(buffer.split()) >= min_words:
                merged.append(buffer)
                buffer = ""

        # Flush remaining buffer
        if buffer:
            if merged:
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer)

        return merged

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        """
        Split a paragraph exceeding chunk_size words into
        sub-chunks with chunk_overlap word overlap.
        """
        words = paragraph.split()
        if len(words) <= self.chunk_size:
            return [paragraph]

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(chunk_text)

            if end >= len(words):
                break

            start = end - self.chunk_overlap

        return chunks

    # ── Main chunking methods ────────────────────────────────────────────────

    def chunk_file(self, file_path, metadata: dict) -> list[dict]:
        """
        Chunk a single cleaned_text.txt file into paragraph-level chunks.

        Args:
            file_path: Path to the cleaned_text.txt file.
            metadata:  Dict with keys: subject, class_num, chapter_num,
                       chapter_title, folder_name.

        Returns:
            List of chunk dicts with chunk_id, text, word_count, and metadata.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"  {Fore.RED}[MISSING] {file_path}")
            return []

        # Step 1: Read the file
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")

        # Step 2: Clean artifacts
        cleaned_text = self._clean_text(raw_text)

        # Step 3: Split into paragraphs
        paragraphs = self._split_into_paragraphs(cleaned_text)

        # Step 4a: Merge short paragraphs
        paragraphs = self._merge_short_paragraphs(paragraphs, min_words=30)

        # Step 4b: Split long paragraphs into sub-chunks
        all_chunk_texts = []
        for para in paragraphs:
            sub_chunks = self._split_long_paragraph(para)
            all_chunk_texts.extend(sub_chunks)

        # Step 5: Build chunk dicts with metadata
        folder_name = metadata.get("folder_name", "unknown")
        total_chunks = len(all_chunk_texts)
        chunks = []

        for idx, text in enumerate(all_chunk_texts):
            word_count = len(text.split())

            # Skip chunks that ended up too short after processing
            if word_count < 10:
                continue

            chunks.append({
                "chunk_id": f"{folder_name}_chunk_{idx:03d}",
                "text": text,
                "word_count": word_count,
                "metadata": {
                    "subject": metadata.get("subject", "biology"),
                    "class": metadata.get("class_num", ""),
                    "chapter_num": metadata.get("chapter_num", ""),
                    "chapter_title": metadata.get("chapter_title", ""),
                    "folder_name": folder_name,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                },
            })

        return chunks

    def chunk_all_chapters(self, processed_dir: str = "data/processed") -> list[dict]:
        """
        Scan processed_dir for all chapter directories and chunk them all.

        Args:
            processed_dir: Path to the directory containing class*_ch*/ folders.

        Returns:
            Merged list of all chunks across all chapters.
        """
        processed_path = Path(processed_dir)
        if not processed_path.exists():
            print(f"{Fore.RED}[ERROR] Directory not found: {processed_dir}")
            return []

        print(f"\n{'='*55}")
        print(f"  NCERT Chapter Chunking")
        print(f"{'='*55}")
        print(f"  Source:     {processed_dir}")
        print(f"  Chunk size: {self.chunk_size} words")
        print(f"  Overlap:    {self.chunk_overlap} words")
        print(f"{'='*55}\n")

        all_chunks = []
        chapter_stats = []

        # Find all chapter directories matching class*_ch*
        chapter_dirs = sorted([
            d for d in processed_path.iterdir()
            if d.is_dir() and re.match(r"class\d+_ch\d+", d.name)
        ])

        for chapter_dir in chapter_dirs:
            folder_name = chapter_dir.name
            cleaned_file = chapter_dir / "cleaned_text.txt"

            # Skip folders not in our map
            if folder_name not in self.CHAPTER_MAP:
                print(f"  {Fore.YELLOW}[SKIP] {folder_name} — not in CHAPTER_MAP")
                continue

            if not cleaned_file.exists():
                print(f"  {Fore.RED}[MISSING] {folder_name}/cleaned_text.txt")
                continue

            # Build metadata from map
            map_entry = self.CHAPTER_MAP[folder_name]
            metadata = {**map_entry, "folder_name": folder_name}

            # Chunk the file
            chunks = self.chunk_file(cleaned_file, metadata)
            all_chunks.extend(chunks)

            # Collect stats
            file_size_kb = round(cleaned_file.stat().st_size / 1024, 1)
            chapter_stats.append({
                "folder": folder_name,
                "num_chunks": len(chunks),
                "file_size_kb": file_size_kb,
            })

            print(f"  {Fore.GREEN}[OK] {folder_name:20s} -> {len(chunks):3d} chunks  ({file_size_kb}KB)")

        # ── Print summary ────────────────────────────────────────────────────
        if all_chunks:
            word_counts = [c["word_count"] for c in all_chunks]

            print(f"\n{'='*55}")
            print(f"  NCERT Chunking Complete")
            print(f"{'='*55}")
            print(f"  Chapters processed: {len(chapter_stats)}")
            print(f"  Total chunks:       {len(all_chunks)}")
            print(f"  Avg words/chunk:    {sum(word_counts) / len(word_counts):.0f}")
            print(f"  Min words/chunk:    {min(word_counts)}")
            print(f"  Max words/chunk:    {max(word_counts)}")
            print(f"\n  Per-chapter breakdown:")
            for s in chapter_stats:
                print(f"    {s['folder']:20s}: {s['num_chunks']:3d} chunks  ({s['file_size_kb']}KB)")
            print(f"{'='*55}")
        else:
            print(f"\n  {Fore.RED}No chunks generated. Check that cleaned_text.txt files exist.")

        return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# CLI — direct run
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NCERT Chapter Chunker")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Target words per chunk (default: 200)")
    parser.add_argument("--overlap", type=int, default=30,
                        help="Word overlap between chunks (default: 30)")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Path to processed data directory")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Save chunks to JSON file for inspection")
    args = parser.parse_args()

    # Resolve processed dir relative to project root
    project_root = Path(__file__).parent.parent
    processed_dir = args.processed_dir or str(project_root / "data" / "processed")

    chunker = NCERTChunker(chunk_size=args.chunk_size, chunk_overlap=args.overlap)
    chunks = chunker.chunk_all_chapters(processed_dir)

    # Optionally save JSON preview
    if args.save_json and chunks:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"\n  [SAVED] {len(chunks)} chunks -> {out_path}")

    return chunks


if __name__ == "__main__":
    main()
