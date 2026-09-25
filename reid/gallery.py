"""
reid/gallery.py
---------------
Student Registration Gallery Manager.
Stores 3–5 reference image crops and embeddings per student.
Persists gallery metadata and FAISS index offline to disk.
"""

import os
import cv2
import json
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

from .embedding_model import BaseReIDModel, PersonReIDExtractor
from .faiss_manager import FAISSManager

logger = logging.getLogger(__name__)


class StudentGallery:
    """
    Manages student registration, reference crops, embeddings, and FAISS indexing.
    Directory structure:
      gallery_dir/
        <student_id>/
          crop_0.jpg
          crop_1.jpg
          ...
      faiss.index
      metadata.json
    """

    def __init__(
        self,
        base_dir: str = None,
        extractor: BaseReIDModel = None,
        faiss_manager: FAISSManager = None,
    ):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reid")
        self.base_dir = base_dir
        self.gallery_dir = os.path.join(base_dir, "gallery")
        self.index_path = os.path.join(base_dir, "faiss.index")
        self.meta_path = os.path.join(base_dir, "metadata.json")

        os.makedirs(self.gallery_dir, exist_ok=True)

        self.extractor = extractor or PersonReIDExtractor()
        self.faiss_mgr = faiss_manager or FAISSManager(embedding_dim=self.extractor.embedding_dim)

        # In-memory dictionary: student_id -> list of embeddings
        self.students: Dict[str, List[np.ndarray]] = {}

        # Attempt to load existing gallery
        self.load()

    def register_student(
        self,
        student_id: str,
        crops: List[np.ndarray],
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> int:
        """
        Register a student with 3–5 reference crops.
        Saves crops to disk, extracts embeddings, and adds them to the FAISS index.
        Returns the number of embeddings added.
        """
        student_dir = os.path.join(self.gallery_dir, student_id)
        os.makedirs(student_dir, exist_ok=True)

        if student_id not in self.students:
            self.students[student_id] = []

        added_count = 0
        for i, crop in enumerate(crops[:5]):  # Up to 5 reference crops
            if crop is None or crop.size == 0:
                continue

            # Save reference crop to disk
            crop_path = os.path.join(student_dir, f"crop_{len(self.students[student_id]) + added_count}.jpg")
            cv2.imwrite(crop_path, crop)

            # Compute or get embedding
            if embeddings is not None and i < len(embeddings):
                emb = embeddings[i]
            else:
                emb = self.extractor.extract(crop)

            # Add to FAISS index
            self.faiss_mgr.add_embedding(
                embedding=emb,
                student_id=student_id,
                extra_meta={"crop_path": os.path.relpath(crop_path, self.base_dir)}
            )
            self.students[student_id].append(emb)
            added_count += 1

        logger.info("Registered student '%s' with %d reference embeddings.", student_id, added_count)
        return added_count

    def get_registered_students(self) -> List[str]:
        """Return list of all registered student IDs."""
        return sorted(list(self.students.keys()))

    def save(self):
        """Save FAISS index and metadata to disk."""
        self.faiss_mgr.save(self.index_path, self.meta_path)

    def load(self) -> bool:
        """Load gallery from disk if available."""
        loaded = self.faiss_mgr.load(self.index_path, self.meta_path)
        if loaded:
            self.students = {}
            for item in self.faiss_mgr.metadata:
                sid = item["student_id"]
                if sid not in self.students:
                    self.students[sid] = []
        return loaded

    def reset(self):
        """Reset gallery and index in memory and on disk."""
        self.students = {}
        self.faiss_mgr.reset()
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
