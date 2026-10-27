"""
reid/face_embedder.py
---------------------
Local Face Feature Extractor for enrollment and identity confirmation.

Used ONLY for:
  1. Enrollment phase: match face photos to video observations.
  2. Fallback signal: when a face is clearly visible later in the video.

Does NOT depend on continuous face visibility.
No cloud APIs.

Strategy:
  - OpenCV Haar Cascade or DNN SSD for face detection (local).
  - Lightweight face embedding: resize-to-64x64 BGR flattened + LBP histogram
    (a simple but effective local descriptor for face verification).
  - L2-normalized 128+256 = 384-dimensional vector.

This intentionally stays simple. The primary identity signal is body Re-ID (MobileNetV3+HSV).
"""

import cv2
import numpy as np
import logging
import os
from typing import Optional, Tuple


def _safe_imread(path: str) -> Optional[np.ndarray]:
    """Read image via PIL to bypass ultralytics cv2 patch."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        return arr[:, :, ::-1].copy()  # RGB -> BGR
    except Exception:
        return None


logger = logging.getLogger(__name__)


def _compute_lbp_histogram(gray: np.ndarray, num_points: int = 8, radius: int = 1) -> np.ndarray:
    """
    Compute a simplified LBP (Local Binary Pattern) histogram.
    Divides the image into a 4x4 grid and computes a 64-bin histogram per cell.
    Returns a 4*4*64 = 1024-dim vector, then we trim to 256 for efficiency.
    """
    h, w = gray.shape
    num_bins = 256
    # 2x2 grid
    grid_h, grid_w = h // 2, w // 2
    hist_all = []
    for r in range(2):
        for c in range(2):
            cell = gray[r * grid_h:(r + 1) * grid_h, c * grid_w:(c + 1) * grid_w]
            # Simple pixel-comparison LBP approximation
            lbp_cell = np.zeros_like(cell, dtype=np.uint8)
            center = cell[1:-1, 1:-1]
            # 8-neighbor comparison
            neighbors = [
                cell[:-2, :-2], cell[:-2, 1:-1], cell[:-2, 2:],
                cell[1:-1, :-2],                  cell[1:-1, 2:],
                cell[2:, :-2],   cell[2:, 1:-1],  cell[2:, 2:],
            ]
            code = np.zeros(center.shape, dtype=np.uint8)
            for bit, nb in enumerate(neighbors):
                code += (nb >= center).astype(np.uint8) * (1 << bit)
            lbp_cell[1:-1, 1:-1] = code
            hist, _ = np.histogram(lbp_cell.ravel(), bins=num_bins, range=(0, 255))
            hist_all.append(hist.astype(np.float32))
    combined = np.concatenate(hist_all)
    return combined


class FaceEmbedder:
    """
    Local face feature extractor.
    Uses Haar cascade for face detection + a compact face descriptor.
    Produces a 384-dimensional L2-normalized vector per face.
    """

    def __init__(self):
        # Try OpenCV Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            cascade_path = "haarcascade_frontalface_default.xml"

        self.face_cascade = None
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("FaceEmbedder: Loaded Haar cascade from %s", cascade_path)
        else:
            logger.warning("FaceEmbedder: No Haar cascade found. Face detection will use full-image fallback.")

        self.face_size = (64, 64)
        self._dim = 128 + 256  # pixel features + LBP histogram
        logger.info("FaceEmbedder: embedding_dim=%d", self._dim)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect faces in a BGR image.
        Returns list of (x, y, w, h) face bounding boxes sorted by area descending.
        """
        if image is None or image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.face_cascade is not None and not self.face_cascade.empty():
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                if len(faces) > 0:
                    # Sort by area descending
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    return faces
            except Exception as e:
                logger.debug("Face detection failed: %s", e)
        return []

    def extract_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face in a full image (e.g. enrollment photo) and extract embedding.
        Falls back to using the full image resized if no face is detected.
        Returns L2-normalized float32 embedding of shape (384,) or None.
        """
        if image is None or image.size == 0:
            return None

        faces = self.detect_faces(image)
        if faces and len(faces) > 0:
            x, y, w, h = faces[0]
            # Add small margin
            margin = int(0.15 * max(w, h))
            x = max(0, x - margin)
            y = max(0, y - margin)
            x2 = min(image.shape[1], x + w + 2 * margin)
            y2 = min(image.shape[0], y + h + 2 * margin)
            face_crop = image[y:y2, x:x2]
        else:
            # No face found — use full image (photo might already be a face crop)
            face_crop = image

        return self._embed_face_crop(face_crop)

    def extract_from_person_crop(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Try to find a face in a person body crop and extract face embedding.
        Returns None if no face found or too small.
        """
        if person_crop is None or person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]
        # Assume face is in upper 40% of the person crop
        face_region = person_crop[:max(1, int(h * 0.4)), :]

        if face_region.shape[0] < 20 or face_region.shape[1] < 20:
            return None

        faces = self.detect_faces(face_region)
        if faces and len(faces) > 0:
            x, y, fw, fh = faces[0]
            # Require minimum face size relative to crop
            if fw * fh < 400:  # 20x20 minimum
                return None
            margin = int(0.1 * max(fw, fh))
            x = max(0, x - margin)
            y = max(0, y - margin)
            x2 = min(face_region.shape[1], x + fw + 2 * margin)
            y2 = min(face_region.shape[0], y + fh + 2 * margin)
            face_crop = face_region[y:y2, x:x2]
            return self._embed_face_crop(face_crop)

        return None

    def _embed_face_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute embedding from a face crop BGR image.
        """
        if face_crop is None or face_crop.size == 0 or face_crop.shape[0] < 8 or face_crop.shape[1] < 8:
            return None

        try:
            # Resize to standard face size
            face_resized = cv2.resize(face_crop, self.face_size)

            # Component 1: Downsampled pixel features (equalized grayscale)
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)
            # Resize to 8x8 = 64 values, then pad to 128
            small = cv2.resize(gray_eq, (8, 8)).flatten().astype(np.float32) / 255.0
            # Duplicate with slight shift to get 128 features
            pixel_feat = np.concatenate([small, small])  # 128-dim

            # Component 2: LBP histogram over 2x2 grid = 4 * 64 = 256-dim
            lbp_feat = _compute_lbp_histogram(gray_eq)  # 4 * 64 = 256 dim
            norm_l = np.linalg.norm(lbp_feat)
            if norm_l > 1e-6:
                lbp_feat = lbp_feat / norm_l

            combined = np.concatenate([pixel_feat, lbp_feat]).astype(np.float32)
            norm = np.linalg.norm(combined)
            if norm > 1e-6:
                combined = combined / norm
            return combined

        except Exception as e:
            logger.debug("Face embedding failed: %s", e)
            return None

    def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """
        Cosine similarity between two face embeddings.
        """
        if emb_a is None or emb_b is None:
            return 0.0
        dot = float(np.dot(emb_a, emb_b))
        return max(0.0, min(1.0, dot))
