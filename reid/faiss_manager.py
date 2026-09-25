"""
reid/faiss_manager.py
---------------------
Local CPU-based FAISS vector index manager for person Re-ID.
Works 100% offline. Supports nearest-neighbor search with Cosine Similarity (IndexFlatIP).
Saves and loads index and student mapping metadata to/from disk.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSManager:
    """
    Manages the offline FAISS index and the vector_idx -> student_id mapping.
    """

    def __init__(self, embedding_dim: int = 672):
        self.embedding_dim = embedding_dim
        # IndexFlatIP computes exact Inner Product; for L2-normalized vectors this is Cosine Similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        # Mapping: vector_index (int) -> dict with student_id, etc.
        self.metadata: List[Dict] = []

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def add_embedding(self, embedding: np.ndarray, student_id: str, extra_meta: Optional[Dict] = None) -> int:
        """
        Add a single L2-normalized embedding to the index with its associated student_id.
        Returns the assigned vector index.
        """
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        # Ensure L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        vector_idx = self.index.ntotal
        self.index.add(vec)

        entry = {
            "vector_idx": vector_idx,
            "student_id": student_id,
        }
        if extra_meta:
            entry.update(extra_meta)
        self.metadata.append(entry)
        return vector_idx

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search the FAISS index for nearest neighbors.
        Returns list of dicts: [{"student_id": ..., "score": float, "vector_idx": int}, ...]
        sorted in descending order of similarity score.
        """
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        scores, indices = self.index.search(vec, k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "student_id": meta["student_id"],
                "score": float(score),
                "vector_idx": int(idx),
                "metadata": meta,
            })
        return results

    def save(self, index_path: str, meta_path: str):
        """Save FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "embedding_dim": self.embedding_dim,
                "total_vectors": self.index.ntotal,
                "metadata": self.metadata,
            }, f, indent=2)
        logger.info("Saved FAISS index to %s (%d vectors)", index_path, self.index.ntotal)

    def load(self, index_path: str, meta_path: str) -> bool:
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return False

        try:
            self.index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.embedding_dim = data.get("embedding_dim", self.embedding_dim)
            self.metadata = data.get("metadata", [])
            logger.info("Loaded FAISS index from %s (%d vectors)", index_path, self.index.ntotal)
            return True
        except Exception as exc:
            logger.warning("Failed to load FAISS index from %s: %s", index_path, exc)
            return False

    def reset(self):
        """Reset the index and metadata."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []
