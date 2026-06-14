"""
EduHinglish — Build NCERT Knowledge Base
==========================================
Author  : Ashvatth
Module  : M2 — NCERT Retriever (Phase 5, Task 4)
Purpose : One-time script that orchestrates the full RAG pipeline:
          cleaned_text.txt files -> chunks -> embeddings -> ChromaDB

Usage:
    python scripts/build_knowledge_base.py
    python scripts/build_knowledge_base.py --chunk-size 150
    python scripts/build_knowledge_base.py --force       # rebuild from scratch
    python scripts/build_knowledge_base.py --db-path src/knowledge_base
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


def main():
    parser = argparse.ArgumentParser(description="Build NCERT Knowledge Base (RAG Pipeline)")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Target words per chunk (default: 200)")
    parser.add_argument("--overlap", type=int, default=30,
                        help="Word overlap between chunks (default: 30)")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path for ChromaDB storage (default: src/knowledge_base)")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Path to processed data directory")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing knowledge base and rebuild from scratch")
    args = parser.parse_args()

    # Resolve paths
    db_path = args.db_path or str(project_root / "src" / "knowledge_base")
    processed_dir = args.processed_dir or str(project_root / "data" / "processed")

    print(f"\n{'='*65}")
    print(f"  EduHinglish -- NCERT Knowledge Base Builder")
    print(f"{'='*65}")
    print(f"  Source:        {processed_dir}")
    print(f"  DB path:       {db_path}")
    print(f"  Chunk size:    {args.chunk_size} words")
    print(f"  Overlap:       {args.overlap} words")
    print(f"  Force rebuild: {args.force}")
    print(f"{'='*65}")

    total_start = time.time()

    # ── Import modules ───────────────────────────────────────────────────────
    from chunker import NCERTChunker
    from embedder import NCERTEmbedder
    from retriever import NCERTRetriever

    # ── Step 1: Initialize retriever and check existing data ─────────────────
    print(f"\n{'_'*65}")
    print(f"  Step 1: Initialize vector database")
    print(f"{'_'*65}")

    retriever = NCERTRetriever(db_path=db_path)
    stats = retriever.get_collection_stats()

    if stats["total_documents"] > 0 and not args.force:
        print(f"\n  Knowledge base already exists with {stats['total_documents']} documents.")
        print(f"  Use --force to rebuild from scratch.")
        return

    if args.force and stats["total_documents"] > 0:
        print(f"\n  Deleting existing collection ({stats['total_documents']} documents)...")
        retriever.delete_collection()

    # ── Step 2: Chunk all chapters ───────────────────────────────────────────
    print(f"\n{'_'*65}")
    print(f"  Step 2: Chunk NCERT chapters")
    print(f"{'_'*65}")

    chunker = NCERTChunker(chunk_size=args.chunk_size, chunk_overlap=args.overlap)
    chunks = chunker.chunk_all_chapters(processed_dir)

    if not chunks:
        print(f"\n  {Fore.RED}No chunks generated. Aborting.")
        return

    # ── Step 3: Embed all chunks ─────────────────────────────────────────────
    print(f"\n{'_'*65}")
    print(f"  Step 3: Embed chunks")
    print(f"{'_'*65}")

    embedder = NCERTEmbedder()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_texts(texts)

    # ── Step 4: Store in ChromaDB ────────────────────────────────────────────
    print(f"\n{'_'*65}")
    print(f"  Step 4: Store in ChromaDB")
    print(f"{'_'*65}")

    retriever.add_documents(chunks, embeddings)

    # ── Step 5: Verification queries ─────────────────────────────────────────
    print(f"\n{'_'*65}")
    print(f"  Step 5: Verification queries")
    print(f"{'_'*65}")

    verification_queries = [
        ("mitochondria function", ["powerhouse", "mitochondria", "atp"]),
        ("photosynthesis chloroplast", ["photosynthesis", "chloroplast", "sunlight"]),
        ("cell division mitosis meiosis", ["mitosis", "meiosis", "division"]),
    ]

    all_passed = True
    for query_text, expected_keywords in verification_queries:
        results = retriever.search(query_text, embedder, top_k=1)
        if results:
            top_text = results[0]["text"].lower()
            found_any = any(kw.lower() in top_text for kw in expected_keywords)
            chapter = results[0]["metadata"].get("chapter_title", "Unknown")
            score = results[0]["relevance_score"]

            if found_any:
                print(f"  {Fore.GREEN}[PASS] \"{query_text}\" -> {chapter} (score: {score:.2f}){Style.RESET_ALL}")
            else:
                print(f"  {Fore.YELLOW}[WARN] \"{query_text}\" -> {chapter} (score: {score:.2f}){Style.RESET_ALL}")
                all_passed = False
        else:
            print(f"  {Fore.RED}[FAIL] \"{query_text}\" -> No results{Style.RESET_ALL}")
            all_passed = False

    # ── Final summary ────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    final_stats = retriever.get_collection_stats()
    word_counts = [c["word_count"] for c in chunks]
    num_chapters = len(set(c["metadata"]["folder_name"] for c in chunks))

    print(f"\n{'='*65}")
    print(f"  EduHinglish -- NCERT Knowledge Base Build Complete")
    print(f"{'='*65}")
    print(f"\n  Chunking:")
    print(f"    Chapters processed:  {num_chapters}")
    print(f"    Total chunks:        {len(chunks)}")
    print(f"    Avg words/chunk:     {sum(word_counts)/len(word_counts):.0f}")
    print(f"\n  Embedding:")
    print(f"    Model:               paraphrase-multilingual-MiniLM-L12-v2")
    print(f"    Dimension:           {embedder.dimension}")
    print(f"    Device:              {embedder.device}")
    print(f"\n  Storage:")
    print(f"    Database:            ChromaDB (local)")
    print(f"    Location:            {db_path}")
    print(f"    Collection:          {final_stats['collection_name']}")
    print(f"    Total vectors:       {final_stats['total_documents']}")
    print(f"\n  Verification:          {'All passed' if all_passed else 'Some queries need review'}")
    print(f"  Total build time:      {total_time:.1f}s")
    print(f"\n  {Fore.GREEN}Knowledge base ready for Phase 6 (Hinglish generator)!")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
