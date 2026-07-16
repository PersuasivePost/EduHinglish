# Phase 5 — NCERT Knowledge Base / RAG Pipeline

> **Phase:** 5 of 8 | **Status:** Ready to start
> **Input:** `data/processed/` cleaned text files (10 chapters, ~350KB total text)
> **Output:** ChromaDB vector database + retrieval pipeline (chunker → embedder → retriever)
> **Runs on:** CPU (local) or Colab T4 — no GPU required for this phase
> **Prerequisite:** Phase 4 complete (MuRIL models trained)

---

## What Phase 5 Builds

A fully local RAG (Retrieval-Augmented Generation) knowledge base that stores all NCERT chapter content as searchable vectors so the system retrieves the right paragraph for any student query. No internet needed at runtime.

| Component | Purpose | Technology |
| --- | --- | --- |
| `chunker.py` | Split chapter text into paragraph-level chunks with metadata | Pure Python |
| `embedder.py` | Convert text chunks into 384-dim vectors | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `retriever.py` | Search ChromaDB for top-k relevant chunks | ChromaDB (local, on-disk) |
| `build_knowledge_base.py` | One-time script: all chapters → ChromaDB | Orchestrator script |

### Why This Architecture?

- **Embedding model** (`paraphrase-multilingual-MiniLM-L12-v2`): 118M params, handles Hindi+English mixed sentences natively, runs fast on CPU, free, local, no API
- **Vector DB** (ChromaDB): Runs fully local as files on disk, zero setup, no API keys, pip install only
- **Designed for easy migration**: Retriever uses a wrapper class so switching to Pinecone later requires changing only the backend parameter

---

## PROMPT — Ashvatth (paste into Claude Code)

Copy everything between the triple backtick fences and paste into Claude Code:

---

```
I am Ashvatth, working on EduHinglish — a final-year AI project for
Hinglish-speaking Indian students. I have completed Phase 4
(MuRIL fine-tuning). I now need to build Phase 5:
the NCERT Knowledge Base / RAG pipeline using ChromaDB.

═══════════════════════════════════════════════════════
PROJECT STATE (what already exists)
═══════════════════════════════════════════════════════

Cleaned NCERT chapter text files in data/processed/:
  data/processed/class9_ch01/cleaned_text.txt   (18KB)
  data/processed/class9_ch02/cleaned_text.txt   (50KB)
  data/processed/class9_ch03/cleaned_text.txt   (46KB)
  data/processed/class9_ch11/cleaned_text.txt   (53KB)
  data/processed/class9_ch12/cleaned_text.txt   (59KB)
  data/processed/class10_ch05/cleaned_text.txt  (50KB)
  data/processed/class10_ch06/cleaned_text.txt  (30KB)
  data/processed/class10_ch07/cleaned_text.txt  (37KB)
  data/processed/class10_ch08/cleaned_text.txt  (15KB)
  data/processed/class10_ch13/cleaned_text.txt  (20KB)
  Total: ~380KB of cleaned English NCERT text across 10 chapters.

Chapter metadata mapping (folder → chapter info):
  class9_ch01  → Class 9,  "Exploration: Entering the World of Secondary Science"
  class9_ch02  → Class 9,  "Cell: The Building Block of Life"
  class9_ch03  → Class 9,  "Tissues in Action"
  class9_ch11  → Class 9,  "Reproduction: How Life Continues"
  class9_ch12  → Class 9,  "Patterns in Life: Diversity and Classification"
  class10_ch05 → Class 10, "Life Processes"
  class10_ch06 → Class 10, "Control and Coordination"
  class10_ch07 → Class 10, "How do Organisms Reproduce?"
  class10_ch08 → Class 10, "Heredity"
  class10_ch13 → Class 10, "Our Environment"

Unified dataset: data/unified_biology_dataset_v2.json (950 entries)

Existing modules:
  src/pipeline.py          — M1 preprocessing pipeline
  src/script_detector.py   — ScriptDetector, HinglishNormalizer, WordLevelLID
  src/preprocessing.py     — EnglishPreprocessor
  models/lid_v2/           — spaCy LID model
  models/ner_v2/           — spaCy NER model
  models/intent_v2/        — spaCy intent model

requirements.txt currently has (relevant):
  pdfplumber, spacy, nltk, pandas, numpy, transformers, datasets,
  peft, accelerate, sentencepiece

.gitignore already has:
  ncert_vectordb/
  chroma_db/

No external APIs. Everything runs locally. Python 3.10.
Virtual env: eduhinglish_env

═══════════════════════════════════════════════════════
PHASE 5 SCOPE — FILES TO BUILD
═══════════════════════════════════════════════════════

src/
├── chunker.py                 ← TASK 1
├── embedder.py                ← TASK 2
├── retriever.py               ← TASK 3
scripts/
└── build_knowledge_base.py    ← TASK 4
tests/
└── test_retriever.py          ← TASK 5

Output directories (created by build script):
  src/knowledge_base/          ← ChromaDB persisted files

═══════════════════════════════════════════════════════
TASK 1 — src/chunker.py
═══════════════════════════════════════════════════════

This module splits cleaned NCERT chapter text into paragraph-level
chunks suitable for embedding and retrieval.

── Class: NCERTChunker ──

__init__(self, chunk_size=200, chunk_overlap=30):
  - chunk_size: target words per chunk (NOT characters)
  - chunk_overlap: number of overlapping words between consecutive chunks
    (helps preserve context at chunk boundaries)

chunk_file(self, file_path, metadata) -> list[dict]:
  Input:
    - file_path: path to a cleaned_text.txt file
    - metadata: dict with keys: subject, class_num, chapter_num,
      chapter_title, folder_name

  Processing steps:
    1. Read the entire cleaned_text.txt file
    2. Clean the text further:
       - Remove lines that are purely page markers ("Reprint 2025-26")
       - Remove lines that are purely figure references ("Fig. X.X")
       - Remove lines that are purely "Activity ______________X.X"
       - Remove lines shorter than 20 characters (headers, artifacts)
       - Collapse multiple blank lines into single blank line
    3. Split into paragraphs using double-newline as separator
    4. For each paragraph:
       - If paragraph has fewer than 30 words, merge with next paragraph
       - If paragraph has more than chunk_size words, split into
         sub-chunks of chunk_size words with chunk_overlap word overlap
    5. For each resulting chunk, create a dict:
       {
         "chunk_id": "{folder_name}_chunk_{index:03d}",
         "text": "the chunk text...",
         "word_count": 187,
         "metadata": {
           "subject": "biology",
           "class": "9",
           "chapter_num": "ch01",
           "chapter_title": "Exploration: Entering the World of...",
           "folder_name": "class9_ch01",
           "chunk_index": 0,
           "total_chunks": 45
         }
       }

  Return: list of chunk dicts

chunk_all_chapters(self, processed_dir="data/processed") -> list[dict]:
  - Scan processed_dir for all subdirectories matching class*_ch*
  - For each, look for cleaned_text.txt
  - Map folder name to chapter metadata using CHAPTER_MAP (hardcoded dict)
  - Call chunk_file() for each
  - Return merged list of all chunks
  - Print progress and summary

── CHAPTER_MAP (hardcode this inside the class) ──

  CHAPTER_MAP = {
    "class9_ch01": {"subject": "biology", "class_num": "9",
      "chapter_num": "ch01",
      "chapter_title": "Exploration: Entering the World of Secondary Science"},
    "class9_ch02": {"subject": "biology", "class_num": "9",
      "chapter_num": "ch02",
      "chapter_title": "Cell: The Building Block of Life"},
    "class9_ch03": {"subject": "biology", "class_num": "9",
      "chapter_num": "ch03",
      "chapter_title": "Tissues in Action"},
    "class9_ch11": {"subject": "biology", "class_num": "9",
      "chapter_num": "ch11",
      "chapter_title": "Reproduction: How Life Continues"},
    "class9_ch12": {"subject": "biology", "class_num": "9",
      "chapter_num": "ch12",
      "chapter_title": "Patterns in Life: Diversity and Classification"},
    "class10_ch05": {"subject": "biology", "class_num": "10",
      "chapter_num": "ch05",
      "chapter_title": "Life Processes"},
    "class10_ch06": {"subject": "biology", "class_num": "10",
      "chapter_num": "ch06",
      "chapter_title": "Control and Coordination"},
    "class10_ch07": {"subject": "biology", "class_num": "10",
      "chapter_num": "ch07",
      "chapter_title": "How do Organisms Reproduce?"},
    "class10_ch08": {"subject": "biology", "class_num": "10",
      "chapter_num": "ch08",
      "chapter_title": "Heredity"},
    "class10_ch13": {"subject": "biology", "class_num": "10",
      "chapter_num": "ch13",
      "chapter_title": "Our Environment"},
  }

── Print at end of chunk_all_chapters() ──

  ══════════════════════════════════════════════════
  NCERT Chunking Complete
  ══════════════════════════════════════════════════
  Chapters processed: 10
  Total chunks:       ~350-500 (depends on chunk_size)
  Avg words/chunk:    ~170
  Min words/chunk:    32
  Max words/chunk:    210

  Per-chapter breakdown:
    class9_ch01  : 28 chunks  (18KB)
    class9_ch02  : 55 chunks  (50KB)
    ...

── CLI (when run directly) ──

  python src/chunker.py
  python src/chunker.py --chunk-size 150 --overlap 20
  python src/chunker.py --save-json src/chunks_preview.json

When run directly, chunk all chapters and optionally save a
JSON preview file for inspection.

═══════════════════════════════════════════════════════
TASK 2 — src/embedder.py
═══════════════════════════════════════════════════════

Wrapper around sentence-transformers for embedding text chunks.

── Class: NCERTEmbedder ──

__init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
  - Load the sentence-transformer model
  - Print model info: name, embedding dimension (384), device (cpu/cuda)
  - Handle the case where the model is not yet downloaded:
    print a message and download it automatically
  - IMPORTANT: First time will download ~500MB. Print a progress note.

embed_texts(self, texts: list[str], batch_size=32, show_progress=True) -> list[list[float]]:
  - Encode all texts in batches using model.encode()
  - Show a tqdm progress bar if show_progress=True
  - Return list of embedding vectors (each is list of 384 floats)

embed_query(self, query: str) -> list[float]:
  - Encode a single query string
  - Return one 384-dim vector

── Print at end ──

  When called with a list of texts, print:
    Embedded 420 chunks in 12.3s (34.1 chunks/sec)
    Embedding dimension: 384
    Device: cpu

── CLI (when run directly) ──

  python src/embedder.py
  # Runs a quick test: embeds 3 example sentences, prints similarity matrix

  Test sentences:
    "Mitochondria is the powerhouse of the cell"
    "Mitochondria ko cell ka powerhouse kehte hain"
    "Photosynthesis occurs in chloroplasts of plant cells"

  Expected output:
    Similarity matrix:
              Sent1   Sent2   Sent3
    Sent1     1.000   0.85+   0.40-0.60
    Sent2     0.85+   1.000   0.35-0.55
    Sent3     0.40+   0.35+   1.000

  The English and Hinglish versions of the same concept should
  have high similarity (>0.80) — this proves the multilingual
  model handles Hinglish retrieval correctly.

═══════════════════════════════════════════════════════
TASK 3 — src/retriever.py
═══════════════════════════════════════════════════════

ChromaDB interface for storing and searching NCERT chunks.
IMPORTANT: Design this with a db_type parameter so we can
migrate to Pinecone later with minimal code changes.

── Class: NCERTRetriever ──

__init__(self, db_path="src/knowledge_base", collection_name="ncert_biology", db_type="chroma"):
  - db_type: "chroma" for now (future: "pinecone")
  - If db_type == "chroma":
    - import chromadb
    - Create PersistentClient at db_path
    - Get or create collection with name collection_name
    - Use cosine distance (distance_fn="cosine" in collection metadata)
  - Print: "[OK] NCERTRetriever initialized (ChromaDB @ {db_path})"

add_documents(self, chunks: list[dict], embeddings: list[list[float]]):
  - chunks: list of chunk dicts from NCERTChunker
  - embeddings: corresponding vectors from NCERTEmbedder
  - For ChromaDB:
    collection.add(
      ids=[chunk["chunk_id"] for chunk in chunks],
      embeddings=embeddings,
      documents=[chunk["text"] for chunk in chunks],
      metadatas=[chunk["metadata"] for chunk in chunks]
    )
  - Print: "Added {len(chunks)} documents to ChromaDB collection '{collection_name}'"

query(self, query_embedding: list[float], top_k=3, filters=None) -> list[dict]:
  - filters: optional dict for metadata filtering
    e.g. {"class": "9"} or {"chapter_num": "ch05"}
  - For ChromaDB:
    results = collection.query(
      query_embeddings=[query_embedding],
      n_results=top_k,
      where=filters if filters else None,
      include=["documents", "metadatas", "distances"]
    )
  - Return list of dicts:
    [
      {
        "chunk_id": "class10_ch05_chunk_003",
        "text": "Mitochondria are known as the powerhouses...",
        "metadata": {"subject": "biology", "class": "10", ...},
        "distance": 0.234,
        "relevance_score": 0.766  # 1 - distance for cosine
      },
      ...
    ]

get_collection_stats(self) -> dict:
  - Return: {"total_documents": N, "collection_name": "...", "db_type": "chroma"}

search(self, query_text: str, embedder, top_k=3, filters=None) -> list[dict]:
  - Convenience method: embeds query_text using the provided embedder,
    then calls self.query()
  - This is the PRIMARY method used by the downstream pipeline

── CLI (when run directly) ──

  python src/retriever.py
  # Loads existing knowledge base and runs test queries

  Test queries:
    "Mitochondria ka kaam kya hai?"
    "Photosynthesis kaise hoti hai?"
    "DNA replication mein enzyme kya role play karta hai?"
    "Stomata ka function kya hai?"
    "Cell division mein mitosis aur meiosis ka fark batao"

  For each query, print:
    ──────────────────────────────────────────────
    Query: "Mitochondria ka kaam kya hai?"
    ──────────────────────────────────────────────
    Result 1 (score: 0.87):
      Chapter: Life Processes (Class 10)
      Text: "Mitochondria are known as the powerhouses
             of the cell. Mitochondria have two membrane
             coverings..."
    Result 2 (score: 0.82):
      Chapter: Cell: The Building Block of Life (Class 9)
      Text: "The energy required for various chemical..."
    Result 3 (score: 0.71):
      Chapter: Life Processes (Class 10)
      Text: "ATP is known as the energy currency..."
    ──────────────────────────────────────────────

═══════════════════════════════════════════════════════
TASK 4 — scripts/build_knowledge_base.py
═══════════════════════════════════════════════════════

One-time script that orchestrates the full pipeline:
  cleaned_text.txt files → chunks → embeddings → ChromaDB

── Flow ──

  1. Import NCERTChunker, NCERTEmbedder, NCERTRetriever
  2. Initialize all three
  3. Chunk all chapters: chunker.chunk_all_chapters()
  4. Embed all chunks: embedder.embed_texts([c["text"] for c in chunks])
  5. Store in ChromaDB: retriever.add_documents(chunks, embeddings)
  6. Verify: run 3 test queries and print results
  7. Print final summary

── Output ──

  ══════════════════════════════════════════════════════
  EduHinglish — NCERT Knowledge Base Build Complete
  ══════════════════════════════════════════════════════

  Chunking:
    Chapters processed:  10
    Total chunks:        ~400
    Avg words/chunk:     ~170

  Embedding:
    Model:               paraphrase-multilingual-MiniLM-L12-v2
    Dimension:           384
    Time:                ~15s on CPU

  Storage:
    Database:            ChromaDB (local)
    Location:            src/knowledge_base/
    Collection:          ncert_biology
    Total vectors:       ~400

  Verification queries:
    ✓ "mitochondria function" → top hit from Life Processes
    ✓ "photosynthesis chloroplast" → top hit from Cell Biology
    ✓ "Mendel pea plants" → top hit from Heredity

  Knowledge base ready for Phase 6 (Hinglish generator)!
  ══════════════════════════════════════════════════════

── CLI ──

  python scripts/build_knowledge_base.py
  python scripts/build_knowledge_base.py --chunk-size 150
  python scripts/build_knowledge_base.py --db-path src/knowledge_base
  python scripts/build_knowledge_base.py --force  # rebuild from scratch

  The --force flag deletes existing knowledge_base/ directory and rebuilds.
  Without --force, if the collection already has documents, skip and print:
    "Knowledge base already exists with 420 documents. Use --force to rebuild."

═══════════════════════════════════════════════════════
TASK 5 — tests/test_retriever.py
═══════════════════════════════════════════════════════

Integration test that verifies the full RAG pipeline works
end-to-end with both English and Hinglish queries.

── Test cases ──

  Test 1: English query retrieves correct chapter
    Query: "What is the function of mitochondria?"
    Expected: top result mentions "powerhouse" or "ATP" or "mitochondria"
    Expected: top result from class10_ch05 or class9_ch01

  Test 2: Hinglish query retrieves correct chapter
    Query: "Sir mitochondria ka kaam kya hai?"
    Expected: top result mentions "powerhouse" or "ATP" or "mitochondria"
    NOTE: This tests that the multilingual embedder handles Hinglish

  Test 3: Chapter filter works
    Query: "cell division"
    Filter: {"class_num": "9"}
    Expected: all results have class="9"

  Test 4: Cross-lingual similarity
    Query English: "photosynthesis in plants"
    Query Hinglish: "plants mein photosynthesis kaise hoti hai"
    Expected: both queries return overlapping top-3 results

  Test 5: Retriever returns correct format
    - Each result has: chunk_id, text, metadata, distance, relevance_score
    - relevance_score is between 0 and 1
    - Results are sorted by relevance (highest first)

── Run ──

  python tests/test_retriever.py
  # or
  python -m pytest tests/test_retriever.py -v

── Output format ──

  ══════════════════════════════════════════════════════
  EduHinglish — Phase 5 RAG Pipeline Tests
  ══════════════════════════════════════════════════════
  Test 1: English query retrieval        ✓ PASS
  Test 2: Hinglish query retrieval       ✓ PASS
  Test 3: Chapter filter                 ✓ PASS
  Test 4: Cross-lingual similarity       ✓ PASS
  Test 5: Result format validation       ✓ PASS
  ──────────────────────────────────────────────────────
  All 5 tests passed!

═══════════════════════════════════════════════════════
DEPENDENCIES TO ADD
═══════════════════════════════════════════════════════

Add these to requirements.txt under a new section:

  # --- Phase 5: NCERT Knowledge Base (RAG) ---
  chromadb>=0.4.24
  sentence-transformers>=2.7.0

Also install torch if not already present (sentence-transformers needs it):
  pip install torch  # if not installed from Phase 4

═══════════════════════════════════════════════════════
GIT WORKFLOW
═══════════════════════════════════════════════════════

Branch: feature/ashvatth-phase5-rag
Base:   develop

Commit after each task:
  git add . && git commit -m "feat(ashvatth): [description]"
  git push origin feature/ashvatth-phase5-rag

Commit message examples:
  feat(ashvatth): chunker.py - paragraph splitter with metadata tagging
  feat(ashvatth): embedder.py - multilingual sentence-transformer wrapper
  feat(ashvatth): retriever.py - ChromaDB interface with migration support
  feat(ashvatth): build_knowledge_base.py - full RAG pipeline builder
  feat(ashvatth): test_retriever.py - end-to-end RAG tests
  feat(ashvatth): requirements.txt - add chromadb + sentence-transformers
  feat(ashvatth): phase 5 knowledge base built - 420 vectors indexed

After all tests pass, open PR → develop.

═══════════════════════════════════════════════════════
START HERE — EXACT ORDER TO EXECUTE
═══════════════════════════════════════════════════════

Step 1:  Add chromadb and sentence-transformers to requirements.txt
Step 2:  Install: pip install chromadb sentence-transformers
Step 3:  Write src/chunker.py
Step 4:  Run chunker.py directly — verify chunks are created, inspect JSON output
Step 5:  Write src/embedder.py
Step 6:  Run embedder.py directly — verify similarity matrix looks correct
Step 7:  Write src/retriever.py
Step 8:  Write scripts/build_knowledge_base.py
Step 9:  Run build_knowledge_base.py — verify ChromaDB populated
Step 10: Write tests/test_retriever.py
Step 11: Run tests — verify all 5 pass
Step 12: Run retriever.py directly — verify Hinglish queries return correct results
Step 13: Final commit + push + PR → develop

Build all scripts (Steps 3-10) first, then run them.
Do not hallucinate retrieval results — print actual computed
distances and retrieved text from the real ChromaDB.

IMPORTANT NOTES:
- The cleaned_text.txt files have OCR artifacts and formatting
  issues (merged lines, garbled headers like "TTTTT FFFFF").
  The chunker MUST clean these before chunking.
- ChromaDB data goes in src/knowledge_base/ — this directory
  is already in .gitignore (as ncert_vectordb/ and chroma_db/).
  Add src/knowledge_base/ to .gitignore as well.
- The embedder downloads ~500MB on first run. Warn the user.
- All file paths should use pathlib.Path for cross-platform support.
- Use colorama for colored terminal output (already in requirements.txt).
```

---

## Key Technical Notes

### Embedding model choice

`paraphrase-multilingual-MiniLM-L12-v2` is specifically chosen because:
- It produces 384-dimensional embeddings (compact, fast to search)
- It's trained on 50+ languages including Hindi and English
- It handles Romanized Hindi (Hinglish) without any special preprocessing
- A Hinglish query like "mitochondria ka kaam kya hai" will have high cosine similarity with the English chunk about mitochondria

### ChromaDB persistence

ChromaDB stores data as local files. The `PersistentClient` API:
```python
import chromadb
client = chromadb.PersistentClient(path="src/knowledge_base")
collection = client.get_or_create_collection(
    name="ncert_biology",
    metadata={"hnsw:space": "cosine"}
)
```
This creates a folder at `src/knowledge_base/` with SQLite + binary files.

### Chunk size rationale

- **200 words** is optimal for NCERT textbook paragraphs
- Too small (<100 words): loses context, retrieves fragments
- Too large (>400 words): dilutes relevance, wastes embedding capacity
- **30-word overlap**: ensures no concept is split at a boundary without any representation in adjacent chunks

### Expected scale

| Metric | Expected Value |
| --- | --- |
| Total chapters | 10 |
| Total chunks | 350–500 |
| Embedding dimension | 384 |
| Total ChromaDB size on disk | ~5–10 MB |
| Build time (CPU) | ~30–60 seconds |
| Query latency (CPU) | <50 ms |

### Migration to Pinecone (future)

The `retriever.py` is designed with `db_type` parameter. To migrate later:
1. Sign up for Pinecone free tier, create index with 384 dimensions
2. `pip install pinecone`
3. Add `elif db_type == "pinecone"` branches in retriever.py
4. Re-run `build_knowledge_base.py` with `--db-type pinecone`

### Query flow (Phase 8 integration)

```
Student: "Sir mitochondria ka kaam kya hai?"
  ↓ M1 pipeline: normalize + LID + intent detection
  ↓ Key terms extracted: mitochondria, kaam (function)
  ↓ Phase 5 retriever: embed query → ChromaDB top-3 search
  ↓ Returns: 3 NCERT paragraphs about mitochondria
  ↓ Phase 6 generator: IndicBART generates Hinglish answer
  ↓ Phase 8 UI: displays answer in Streamlit
```

---

## After Phase 5 — Next Steps

| Phase | What | When |
| --- | --- | --- |
| Phase 6 | IndicBART Hinglish generator fine-tuning | Week 8 |
| Phase 7 | GEC engine | Week 9 |
| Phase 8 | Streamlit UI + full integration | Week 10–11 |
