"""
reid/enrolled_gallery.py
------------------------
Persistent Named-Student Gallery Manager.

Replaces the anonymous "Student 01, Student 02, ..." system with
explicitly named enrolled identities:
    Aagam / Swayam / Harshil / Soham

Structure on disk:
    reid/data/
        identities.json              -- master metadata
        galleries/
            Aagam/
                face_ref.jpg         -- original face photo
                face_embedding.npy   -- 384-dim face feature vector
                body_crop_000.jpg    -- body appearance crop
                body_crop_001.jpg
                ...
                body_embeddings.npy  -- (N, 672) stacked body embeddings
            Swayam/
                ...
            Harshil/
                ...
            Soham/
                ...
        faiss.index                  -- unified FAISS index (all students)
        faiss_meta.json              -- index -> student_id mapping

KEY DESIGN DECISIONS:
- Multiple body embeddings per student (NOT just one).
- Adaptive gallery: high-confidence observations get added, up to MAX_BODY_EMBEDDINGS.
- Diversity check: reject new embedding if too similar to an existing one (avoid redundancy).
- Persistent to disk: survives across video runs.
- Face embeddings stored for enrollment verification and fallback matching.
"""

import os
import json
import logging
import numpy as np
import cv2
import faiss
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


def _safe_imread(path: str) -> Optional[np.ndarray]:
    """
    Read an image file robustly.
    Ultralytics patches cv2.imdecode with a broken implementation,
    so we use PIL as primary reader and convert to BGR numpy array.
    """
    try:
        from PIL import Image
        img_pil = Image.open(path).convert("RGB")
        img_rgb = np.array(img_pil, dtype=np.uint8)
        img_bgr = img_rgb[:, :, ::-1].copy()
        return img_bgr
    except Exception:
        pass
    # Fallback: read raw bytes and use cv2 original binding
    try:
        import ctypes
        # Read file bytes and decode
        with open(path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        # Access original C-level imdecode to bypass Python patch
        img = cv2.imdecode.__wrapped__(raw, cv2.IMREAD_COLOR) if hasattr(cv2.imdecode, '__wrapped__') else None
        if img is not None:
            return img
    except Exception:
        pass
    return None


# Maximum body embeddings stored per student to keep gallery bounded
MAX_BODY_EMBEDDINGS = 20
# Minimum diversity: only add new embedding if cosine distance > this threshold from all existing
MIN_DIVERSITY_DIST = 0.08
# Body embedding dimension (MobileNetV3 576 + HSV 96 = 672)
BODY_EMB_DIM = 672
# Face embedding dimension
FACE_EMB_DIM = 384


@dataclass
class StudentProfile:
    """All data for one enrolled student."""
    name: str                                    # e.g. "Aagam"
    face_embedding: Optional[np.ndarray] = None  # (384,) float32
    body_embeddings: List[np.ndarray] = field(default_factory=list)  # list of (672,) float32
    enrollment_complete: bool = False
    body_crop_count: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "enrollment_complete": self.enrollment_complete,
            "num_body_embeddings": len(self.body_embeddings),
            "body_crop_count": self.body_crop_count,
        }


class EnrolledGallery:
    """
    Manages the enrollment gallery for the four known students.
    Handles:
      - Face embedding from photos (enrollment-time).
      - Body embedding gallery (multi-embedding per student).
      - FAISS index for fast nearest-neighbor search.
      - Adaptive self-learning gallery updates.
      - Full disk persistence.
    """

    KNOWN_STUDENTS = ["Aagam", "Swayam", "Harshil", "Soham"]

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Default: reid/data/ relative to this file
            data_dir = os.path.join(os.path.dirname(__file__), "data")

        self.data_dir = data_dir
        self.galleries_dir = os.path.join(data_dir, "galleries")
        self.index_path = os.path.join(data_dir, "faiss.index")
        self.meta_path = os.path.join(data_dir, "faiss_meta.json")
        self.identities_path = os.path.join(data_dir, "identities.json")

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.galleries_dir, exist_ok=True)

        # Initialize FAISS index (IndexFlatIP = inner product, cosine sim for L2-normed vecs)
        self.faiss_index = faiss.IndexFlatIP(BODY_EMB_DIM)
        self.faiss_meta: List[Dict] = []  # [{student_name, vec_idx, crop_path}]

        # In-memory profiles
        self.profiles: Dict[str, StudentProfile] = {}

        # Load existing gallery if available
        self._load()

    # ------------------------------------------------------------------
    # Enrollment Phase
    # ------------------------------------------------------------------

    def enroll_face_photo(self, student_name: str, photo_path: str, face_embedder) -> bool:
        """
        Enroll a student's face from their reference photo.
        Extracts face embedding and stores it.
        Returns True on success.
        """
        if student_name not in self.KNOWN_STUDENTS:
            logger.warning("Unknown student name: %s. Must be one of %s", student_name, self.KNOWN_STUDENTS)
            return False

        img = _safe_imread(photo_path)
        if img is None:
            logger.error("Could not read face photo: %s", photo_path)
            return False

        face_emb = face_embedder.extract_from_image(img)
        if face_emb is None:
            logger.warning("Could not extract face embedding from %s", photo_path)
            # Still create the profile without face embedding
            face_emb = None

        if student_name not in self.profiles:
            self.profiles[student_name] = StudentProfile(name=student_name)

        self.profiles[student_name].face_embedding = face_emb

        # Save face reference image
        student_dir = os.path.join(self.galleries_dir, student_name)
        os.makedirs(student_dir, exist_ok=True)
        ref_path = os.path.join(student_dir, "face_ref.jpg")
        cv2.imwrite(ref_path, img)

        if face_emb is not None:
            np.save(os.path.join(student_dir, "face_embedding.npy"), face_emb)
            logger.info("Enrolled face for %s (embedding dim=%d)", student_name, len(face_emb))
        else:
            logger.warning("Enrolled %s without face embedding (detection failed)", student_name)

        return True

    def add_body_crop(
        self,
        student_name: str,
        body_crop: np.ndarray,
        body_embedding: np.ndarray,
        force: bool = False,
    ) -> bool:
        """
        Add a body appearance crop+embedding to a student's gallery.

        Checks:
          - Gallery not at max capacity (or force=True skips this).
          - Embedding is diverse enough from existing embeddings.

        Returns True if the embedding was added.
        """
        if student_name not in self.profiles:
            self.profiles[student_name] = StudentProfile(name=student_name)

        profile = self.profiles[student_name]

        # Check max capacity
        if not force and len(profile.body_embeddings) >= MAX_BODY_EMBEDDINGS:
            logger.debug("Gallery full for %s (%d embeddings). Skipping.", student_name, MAX_BODY_EMBEDDINGS)
            return False

        # Diversity check: don't add if too similar to existing embeddings
        if not force and len(profile.body_embeddings) > 0:
            for existing_emb in profile.body_embeddings:
                sim = float(np.dot(body_embedding, existing_emb))
                if sim > (1.0 - MIN_DIVERSITY_DIST):
                    logger.debug(
                        "Skipping near-duplicate body embedding for %s (sim=%.3f)", student_name, sim
                    )
                    return False

        # Save crop to disk
        student_dir = os.path.join(self.galleries_dir, student_name)
        os.makedirs(student_dir, exist_ok=True)
        idx = profile.body_crop_count
        crop_path = os.path.join(student_dir, f"body_crop_{idx:03d}.jpg")
        if body_crop is not None and body_crop.size > 0:
            cv2.imwrite(crop_path, body_crop)
        profile.body_crop_count += 1

        # Add embedding to memory + FAISS
        profile.body_embeddings.append(body_embedding.copy())
        vec = np.asarray(body_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        vec_idx = self.faiss_index.ntotal
        self.faiss_index.add(vec)
        self.faiss_meta.append({
            "student_name": student_name,
            "vec_idx": vec_idx,
            "crop_path": os.path.relpath(crop_path, self.data_dir),
        })

        logger.debug(
            "Added body embedding #%d for %s (gallery size=%d)",
            idx, student_name, len(profile.body_embeddings)
        )
        return True

    def mark_enrollment_complete(self, student_name: str):
        """Mark a student's enrollment as complete after body gallery is built."""
        if student_name in self.profiles:
            profile = self.profiles[student_name]
            profile.enrollment_complete = True
            logger.info(
                "Enrollment complete for %s: %d body embeddings",
                student_name, len(profile.body_embeddings)
            )

    # ------------------------------------------------------------------
    # Search / Matching
    # ------------------------------------------------------------------

    def search_body(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Nearest-neighbor body Re-ID search in FAISS.
        Returns list of {student_name, score, vec_idx} sorted descending.
        """
        if self.faiss_index.ntotal == 0:
            return []

        k = min(top_k, self.faiss_index.ntotal)
        vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        scores, indices = self.faiss_index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.faiss_meta):
                continue
            meta = self.faiss_meta[idx]
            results.append({
                "student_name": meta["student_name"],
                "score": float(score),
                "vec_idx": int(idx),
            })
        return results

    def search_face(self, query_face_emb: np.ndarray, threshold: float = 0.75) -> Optional[Tuple[str, float]]:
        """
        Match a face embedding against enrolled face references.
        Returns (student_name, similarity) of best match above threshold, or None.
        """
        if query_face_emb is None:
            return None

        best_name = None
        best_sim = -1.0
        for name, profile in self.profiles.items():
            if profile.face_embedding is None:
                continue
            sim = float(np.dot(query_face_emb, profile.face_embedding))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_name is not None and best_sim >= threshold:
            return (best_name, best_sim)
        return None

    def get_enrolled_students(self) -> List[str]:
        """Return list of students with at least some body embeddings."""
        return [name for name, p in self.profiles.items() if len(p.body_embeddings) > 0]

    def get_profile(self, student_name: str) -> Optional[StudentProfile]:
        return self.profiles.get(student_name)

    def total_body_embeddings(self) -> int:
        return self.faiss_index.ntotal

    # ------------------------------------------------------------------
    # Adaptive Gallery Update (self-learning)
    # ------------------------------------------------------------------

    def adaptive_update(
        self,
        student_name: str,
        body_crop: np.ndarray,
        body_embedding: np.ndarray,
        confidence: float,
        confidence_thresh: float = 0.88,
    ) -> bool:
        """
        Add a new observation to the gallery only if:
          - Confidence is above threshold (high-confidence match).
          - Gallery not full.
          - Embedding is sufficiently diverse.

        Returns True if gallery was updated.
        """
        if confidence < confidence_thresh:
            return False
        return self.add_body_crop(student_name, body_crop, body_embedding, force=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Save FAISS index, metadata, and identity profiles to disk."""
        try:
            faiss.write_index(self.faiss_index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "embedding_dim": BODY_EMB_DIM,
                    "total_vectors": self.faiss_index.ntotal,
                    "metadata": self.faiss_meta,
                }, f, indent=2)

            # Save face embeddings and body embeddings arrays per student
            identities_data = {}
            for name, profile in self.profiles.items():
                student_dir = os.path.join(self.galleries_dir, name)
                os.makedirs(student_dir, exist_ok=True)

                if profile.face_embedding is not None:
                    np.save(os.path.join(student_dir, "face_embedding.npy"), profile.face_embedding)

                if profile.body_embeddings:
                    body_arr = np.stack(profile.body_embeddings, axis=0)
                    np.save(os.path.join(student_dir, "body_embeddings.npy"), body_arr)

                identities_data[name] = profile.to_dict()

            with open(self.identities_path, "w", encoding="utf-8") as f:
                json.dump(identities_data, f, indent=2)

            logger.info(
                "Saved enrolled gallery: %d students, %d body vectors.",
                len(self.profiles), self.faiss_index.ntotal
            )
        except Exception as e:
            logger.error("Failed to save enrolled gallery: %s", e)

    def _load(self) -> bool:
        """Load existing gallery from disk."""
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            logger.info("No existing enrolled gallery found. Starting fresh.")
            return False

        try:
            self.faiss_index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.faiss_meta = data.get("metadata", [])

            # Load per-student data
            if os.path.exists(self.identities_path):
                with open(self.identities_path, "r", encoding="utf-8") as f:
                    identities_data = json.load(f)

                for name, info in identities_data.items():
                    profile = StudentProfile(
                        name=name,
                        enrollment_complete=info.get("enrollment_complete", False),
                        body_crop_count=info.get("body_crop_count", 0),
                    )
                    student_dir = os.path.join(self.galleries_dir, name)

                    # Load face embedding
                    face_emb_path = os.path.join(student_dir, "face_embedding.npy")
                    if os.path.exists(face_emb_path):
                        profile.face_embedding = np.load(face_emb_path)

                    # Load body embeddings
                    body_emb_path = os.path.join(student_dir, "body_embeddings.npy")
                    if os.path.exists(body_emb_path):
                        body_arr = np.load(body_emb_path)
                        profile.body_embeddings = [body_arr[i] for i in range(body_arr.shape[0])]

                    self.profiles[name] = profile

            logger.info(
                "Loaded enrolled gallery: %d students, %d body vectors.",
                len(self.profiles), self.faiss_index.ntotal
            )
            return True

        except Exception as e:
            logger.warning("Failed to load enrolled gallery: %s. Starting fresh.", e)
            self.faiss_index = faiss.IndexFlatIP(BODY_EMB_DIM)
            self.faiss_meta = []
            self.profiles = {}
            return False

    def is_enrollment_ready(self) -> bool:
        """Return True if all 4 known students have at least 3 body embeddings."""
        for name in self.KNOWN_STUDENTS:
            if name not in self.profiles:
                return False
            if len(self.profiles[name].body_embeddings) < 3:
                return False
        return True

    def enrollment_status(self) -> Dict[str, int]:
        """Return dict of student -> number of body embeddings."""
        return {
            name: len(self.profiles[name].body_embeddings)
            if name in self.profiles else 0
            for name in self.KNOWN_STUDENTS
        }
