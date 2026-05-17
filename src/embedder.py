"""
EduHinglish — NCERT Text Embedder
==================================
Author  : Ashvatth
Module  : M2 — NCERT Retriever (Phase 5, Task 2)
Purpose : Wrapper around sentence-transformers for embedding NCERT text chunks.
          Uses a multilingual model that handles Hindi+English (Hinglish) natively.

Usage:
    python src/embedder.py
    # Runs a quick similarity test with English and Hinglish sentences.
"""

import time
import numpy as np
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


class NCERTEmbedder:
    """
    Sentence-transformer wrapper for embedding NCERT text chunks.

    Default model: paraphrase-multilingual-MiniLM-L12-v2
      - 118M params, 384-dimensional embeddings
      - Trained on 50+ languages including Hindi and English
      - Handles Romanized Hindi (Hinglish) without special preprocessing
      - Runs fast on CPU, faster on GPU
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Load the sentence-transformer model.

        NOTE: First run will download ~500MB. Subsequent runs use cached model.
        """
        print(f"\n  Loading embedding model: {model_name}")
        print(f"  (First run will download ~500MB - subsequent runs use cache)")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run:\n"
                "  pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        self.device = str(self.model.device)

        print(f"  {Fore.GREEN}[OK] NCERTEmbedder initialized")
        print(f"       Model:     {model_name.split('/')[-1]}")
        print(f"       Dimension: {self.dimension}")
        print(f"       Device:    {self.device}")

    def embed_texts(self, texts: list[str], batch_size: int = 32, show_progress: bool = True) -> list[list[float]]:
        """
        Encode a list of text strings into embedding vectors.

        Args:
            texts:         List of text strings to embed.
            batch_size:    Batch size for encoding.
            show_progress: Show tqdm progress bar.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        start_time = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        elapsed = time.time() - start_time
        rate = len(texts) / elapsed if elapsed > 0 else 0

        print(f"\n  Embedded {len(texts)} chunks in {elapsed:.1f}s ({rate:.1f} chunks/sec)")
        print(f"  Embedding dimension: {self.dimension}")
        print(f"  Device: {self.device}")

        # Convert numpy arrays to Python lists for ChromaDB compatibility
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Encode a single query string into an embedding vector.

        Args:
            query: The query text to embed.

        Returns:
            A single embedding vector (list of floats).
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# CLI — Quick similarity test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run a quick similarity test to verify the embedder handles Hinglish."""
    print(f"\n{'='*65}")
    print(f"  EduHinglish -- Embedder Similarity Test")
    print(f"{'='*65}")

    embedder = NCERTEmbedder()

    test_sentences = [
        "Mitochondria is the powerhouse of the cell",
        "Mitochondria ko cell ka powerhouse kehte hain",
        "Photosynthesis occurs in chloroplasts of plant cells",
    ]

    labels = ["EN: mitochondria", "HI: mitochondria", "EN: photosynthesis"]

    print(f"\n  Test sentences:")
    for i, (label, sent) in enumerate(zip(labels, test_sentences)):
        print(f"    [{i+1}] {label}")
        print(f"        \"{sent}\"")

    # Embed all
    embeddings = embedder.embed_texts(test_sentences, show_progress=False)
    embeddings_np = np.array(embeddings)

    # Compute cosine similarity matrix
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    normalized = embeddings_np / norms
    similarity_matrix = np.dot(normalized, normalized.T)

    print(f"\n  Cosine Similarity Matrix:")
    # Header
    print(f"  {'':18s}", end="")
    for label in labels:
        print(f"  {label:18s}", end="")
    print()

    # Rows
    for i, label in enumerate(labels):
        print(f"  {label:18s}", end="")
        for j in range(len(labels)):
            score = similarity_matrix[i][j]
            if i == j:
                color = ""
            elif score >= 0.80:
                color = Fore.GREEN
            elif score >= 0.50:
                color = Fore.YELLOW
            else:
                color = Fore.RED
            print(f"  {color}{score:>6.3f}{Style.RESET_ALL}            ", end="")
        print()

    # Verify cross-lingual similarity
    en_hi_sim = similarity_matrix[0][1]
    en_photo_sim = similarity_matrix[0][2]

    print(f"\n  {'='*60}")
    print(f"  Verification:")
    if en_hi_sim >= 0.75:
        print(f"  {Fore.GREEN}[PASS] EN-Hinglish similarity: {en_hi_sim:.3f} (>=0.75 -- Hinglish retrieval works!)")
    else:
        print(f"  {Fore.RED}[FAIL] EN-Hinglish similarity: {en_hi_sim:.3f} (<0.75 -- may need different model)")

    if en_hi_sim > en_photo_sim:
        print(f"  {Fore.GREEN}[PASS] Same-concept similarity ({en_hi_sim:.3f}) > different-concept ({en_photo_sim:.3f})")
    else:
        print(f"  {Fore.YELLOW}[WARN] Same-concept similarity not higher than different-concept")
    print(f"  {'='*60}")


if __name__ == "__main__":
    main()
