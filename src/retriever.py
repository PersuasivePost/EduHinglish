"""
EduHinglish — NCERT Retriever (ChromaDB Interface)
====================================================
Author  : Ashvatth
Module  : M2 — NCERT Retriever (Phase 5, Task 3)
Purpose : ChromaDB vector database interface for storing and searching
          NCERT chapter chunks. Designed with db_type parameter for
          easy future migration to Pinecone.

Usage:
    python src/retriever.py
    # Loads existing knowledge base and runs test queries.
"""

from pathlib import Path
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


class NCERTRetriever:
    """
    Vector database interface for NCERT chunk storage and retrieval.

    Currently supports ChromaDB (local, on-disk). Designed with a
    db_type parameter so switching to Pinecone later requires
    changing only the backend initialization.
    """

    def __init__(
        self,
        db_path: str = "src/knowledge_base",
        collection_name: str = "ncert_biology",
        db_type: str = "chroma",
    ):
        """
        Initialize the retriever.

        Args:
            db_path:         Path where ChromaDB stores its files.
            collection_name: Name of the ChromaDB collection.
            db_type:         "chroma" (future: "pinecone").
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.db_type = db_type

        if db_type == "chroma":
            self._init_chroma()
        elif db_type == "pinecone":
            raise NotImplementedError(
                "Pinecone support not yet implemented. "
                "Use db_type='chroma' for now. Migration guide in workspace/phase-5-prompts.md."
            )
        else:
            raise ValueError(f"Unsupported db_type: {db_type}. Use 'chroma'.")

    def _init_chroma(self):
        """Initialize ChromaDB persistent client and collection."""
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb not installed. Run:\n"
                "  pip install chromadb"
            )

        # Ensure the directory exists
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        doc_count = self.collection.count()
        print(f"  {Fore.GREEN}[OK] NCERTRetriever initialized (ChromaDB @ {self.db_path})")
        print(f"       Collection:  {self.collection_name}")
        print(f"       Documents:   {doc_count}")

    # ── Add documents ────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[dict], embeddings: list[list[float]]):
        """
        Add chunked documents with their embeddings to the vector store.

        Args:
            chunks:     List of chunk dicts from NCERTChunker (must have
                        chunk_id, text, metadata).
            embeddings: List of embedding vectors from NCERTEmbedder.
        """
        if self.db_type == "chroma":
            self._add_chroma(chunks, embeddings)

    def _add_chroma(self, chunks: list[dict], embeddings: list[list[float]]):
        """Add documents to ChromaDB collection."""
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]

        # ChromaDB metadata values must be str, int, float, or bool
        metadatas = []
        for chunk in chunks:
            meta = {}
            for k, v in chunk["metadata"].items():
                if isinstance(v, (int, float)):
                    meta[k] = v
                elif isinstance(v, bool):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        # Add in batches to avoid potential memory issues
        batch_size = 100
        total_added = 0

        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            total_added += batch_end - i

        print(f"  Added {total_added} documents to ChromaDB collection '{self.collection_name}'")

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, query_embedding: list[float], top_k: int = 3, filters: dict = None) -> list[dict]:
        """
        Search for the most relevant chunks given a query embedding.

        Args:
            query_embedding: 384-dim embedding vector of the query.
            top_k:           Number of results to return.
            filters:         Optional metadata filter dict.
                             e.g. {"class": "9"} or {"chapter_num": "ch05"}

        Returns:
            List of result dicts sorted by relevance (highest first).
        """
        if self.db_type == "chroma":
            return self._query_chroma(query_embedding, top_k, filters)
        return []

    def _query_chroma(self, query_embedding: list[float], top_k: int, filters: dict = None) -> list[dict]:
        """Query ChromaDB collection."""
        # Build where clause for metadata filtering
        where_clause = None
        if filters:
            if len(filters) == 1:
                key, value = list(filters.items())[0]
                where_clause = {key: str(value)}
            else:
                # Multiple filters: use $and
                where_clause = {
                    "$and": [{k: str(v)} for k, v in filters.items()]
                }

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # Parse results into clean format
        parsed = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                parsed.append({
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": round(distance, 4),
                    "relevance_score": round(1 - distance, 4),  # cosine: 1 - distance
                })

        # Sort by relevance (highest first)
        parsed.sort(key=lambda x: x["relevance_score"], reverse=True)
        return parsed

    # ── Convenience search ───────────────────────────────────────────────────

    def search(self, query_text: str, embedder, top_k: int = 3, filters: dict = None) -> list[dict]:
        """
        Convenience method: embed query text and search in one call.
        This is the PRIMARY method used by the downstream pipeline.

        Args:
            query_text: Raw query string (English or Hinglish).
            embedder:   NCERTEmbedder instance for embedding the query.
            top_k:      Number of results to return.
            filters:    Optional metadata filters.

        Returns:
            List of result dicts.
        """
        query_embedding = embedder.embed_query(query_text)
        return self.query(query_embedding, top_k=top_k, filters=filters)

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_collection_stats(self) -> dict:
        """Get basic statistics about the collection."""
        count = self.collection.count() if self.db_type == "chroma" else 0
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "db_type": self.db_type,
            "db_path": self.db_path,
        }

    def delete_collection(self):
        """Delete the collection (for --force rebuild)."""
        if self.db_type == "chroma":
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"  Collection '{self.collection_name}' deleted and recreated.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — Test queries against existing knowledge base
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Load existing knowledge base and run test queries."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from embedder import NCERTEmbedder

    print(f"\n{'='*65}")
    print(f"  EduHinglish -- NCERT Retriever Test")
    print(f"{'='*65}")

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    db_path = str(project_root / "src" / "knowledge_base")

    retriever = NCERTRetriever(db_path=db_path)
    stats = retriever.get_collection_stats()

    if stats["total_documents"] == 0:
        print(f"\n  {Fore.RED}Knowledge base is empty! Run build_knowledge_base.py first.")
        return

    print(f"\n  Knowledge base: {stats['total_documents']} documents indexed\n")

    embedder = NCERTEmbedder()

    test_queries = [
        "Mitochondria ka kaam kya hai?",
        "Photosynthesis kaise hoti hai?",
        "DNA replication mein enzyme kya role play karta hai?",
        "Stomata ka function kya hai?",
        "Cell division mein mitosis aur meiosis ka fark batao",
    ]

    for query in test_queries:
        results = retriever.search(query, embedder, top_k=3)

        print(f"\n  {'_'*60}")
        print(f"  Query: \"{query}\"")
        print(f"  {'_'*60}")

        for i, result in enumerate(results):
            meta = result["metadata"]
            chapter = meta.get("chapter_title", "Unknown")
            class_num = meta.get("class", "?")
            score = result["relevance_score"]

            # Truncate text for display
            text_preview = result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]

            color = Fore.GREEN if score >= 0.5 else Fore.YELLOW if score >= 0.3 else Fore.RED
            print(f"  Result {i+1} ({color}score: {score:.2f}{Style.RESET_ALL}):")
            print(f"    Chapter: {chapter} (Class {class_num})")
            print(f"    Text: \"{text_preview}\"")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()
