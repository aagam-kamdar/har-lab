"""
enroll_students.py
------------------
Enrolls 4 known students (Aagam, Swayam, Harshil, Soham) from their face photos.

For each student:
  - FRONT photos → ArcFace face profile (512-dim, averaged + re-normalized)
  - ALL photos (front + back) → Body gallery via PersonReIDExtractor
    with conservative augmentations (no vertical flip, no extreme rotations)

Augmentations used (user-approved):
  - Horizontal flip
  - Rotation ±5°, ±10°
  - Brightness/Contrast (mild)
  - HSV saturation jitter
  - Gaussian blur (light)
  - Gaussian noise
  - Random cutout patches

Augmentations NOT used (per user rules):
  - Vertical flip
  - 90°/180° rotations
  - Extreme distortions
"""

from __future__ import annotations

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FaceFeatureExtractor, BodyFeatureExtractor, OSNetBodyFeatureExtractor, BodyGallery

logger = logging.getLogger(__name__)

KNOWN_STUDENTS = ["aagam", "swayam", "harshil", "soham"]
FACE_FOLDER = Path("face")


# --------------------------------------------------------------------------- #
#  Augmentation helpers                                                        #
# --------------------------------------------------------------------------- #

def augment_image(img: np.ndarray) -> List[np.ndarray]:
    """Generate augmented variants of an image. Returns a list of BGR arrays."""
    augmented = []

    # 1. Horizontal flip
    augmented.append(cv2.flip(img, 1))

    # 2. Rotations ±5°, ±10°
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)

    # 3. Brightness / contrast
    for alpha, beta in [(1.2, 15), (0.85, -15), (1.1, 5)]:
        adj = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        augmented.append(adj)

    # 4. HSV saturation jitter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    for scale in [1.3, 0.75]:
        h_ch, s_ch, v_ch = cv2.split(hsv)
        s_ch = np.clip(s_ch * scale, 0, 255)
        jitter = cv2.merge([h_ch, s_ch, v_ch]).astype(np.uint8)
        augmented.append(cv2.cvtColor(jitter, cv2.COLOR_HSV2BGR))

    # 5. Gaussian blur
    augmented.append(cv2.GaussianBlur(img, (3, 3), 0.8))

    # 6. Gaussian noise
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy)

    # 7. Random cutout
    cutout = img.copy()
    patch_h, patch_w = h // 5, w // 5
    py = np.random.randint(0, max(1, h - patch_h))
    px = np.random.randint(0, max(1, w - patch_w))
    cutout[py : py + patch_h, px : px + patch_w] = 128
    augmented.append(cutout)

    return augmented


def is_front_image(filename: str) -> bool:
    """Heuristic: if filename contains 'back', it's a back/desk image."""
    name_lower = filename.lower()
    return "back" not in name_lower


# --------------------------------------------------------------------------- #
#  Enrollment                                                                  #
# --------------------------------------------------------------------------- #

class StudentEnroller:
    """
    Enrolls known students from face photos and builds:
      - face_profiles: {student_name: 512-dim L2-normalized ArcFace embedding}
      - body_gallery: BodyGallery with augmented body embeddings per student
    """

    def __init__(self):
        self.face_extractor = FaceFeatureExtractor(det_size=(320, 320))
        self.body_extractor = BodyFeatureExtractor()  # LOCKED: MobileNetV3-Small + HSV (672-dim)
        self.body_gallery = BodyGallery(dim=672)
        self.face_profiles: Dict[str, np.ndarray] = {}
        self._enrollment_stats: Dict[str, Dict] = {}

    def enroll_all(self) -> Tuple[Dict[str, np.ndarray], BodyGallery]:
        """
        Enumerate face/ folder and enroll each student found.
        Returns (face_profiles, body_gallery).
        """
        if not FACE_FOLDER.exists():
            raise FileNotFoundError(f"Face folder not found: {FACE_FOLDER.resolve()}")

        for student_folder in sorted(FACE_FOLDER.iterdir()):
            if not student_folder.is_dir():
                continue
            name = student_folder.name.lower()
            if name not in KNOWN_STUDENTS:
                logger.warning("Skipping unknown folder: %s", student_folder)
                continue
            self._enroll_student(name, student_folder)

        logger.info("Enrollment complete. Face profiles: %s", list(self.face_profiles.keys()))
        logger.info("Body gallery sizes: %s", self.body_gallery.sizes())
        return self.face_profiles, self.body_gallery

    def _enroll_student(self, name: str, folder: Path):
        image_files = [
            f for f in sorted(folder.iterdir())
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        logger.info("Enrolling %s from %d images in %s", name, len(image_files), folder)

        face_embeddings: List[np.ndarray] = []
        body_added = 0
        body_augmented = 0

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Could not read: %s", img_path)
                continue

            fname = img_path.name
            front = is_front_image(fname)

            # --- Face embedding (front images only) ---
            if front:
                emb = self.face_extractor.extract_from_image(img)
                if emb is not None:
                    face_embeddings.append(emb)
                    logger.debug("  Face embedding from %s: OK", fname)
                else:
                    logger.debug("  Face embedding from %s: no face detected", fname)

            # --- Body embedding (all images: front + back) ---
            body_emb = self.body_extractor.extract(img)
            if body_emb is not None:
                added = self.body_gallery.add(name, body_emb)
                if added:
                    body_added += 1

                # Augmented variants
                for aug_img in augment_image(img):
                    aug_emb = self.body_extractor.extract(aug_img)
                    if aug_emb is not None:
                        ok = self.body_gallery.add(name, aug_emb)
                        if ok:
                            body_augmented += 1

        # --- Average + re-normalize face profile ---
        if face_embeddings:
            avg_face = np.mean(face_embeddings, axis=0)
            norm = np.linalg.norm(avg_face)
            if norm > 1e-8:
                avg_face = avg_face / norm
            self.face_profiles[name] = avg_face.astype(np.float32)
            logger.info(
                "  %s face profile: %d embeddings averaged (%.4f norm)",
                name, len(face_embeddings), float(np.linalg.norm(avg_face)),
            )
        else:
            logger.warning("  %s: No face embeddings obtained!", name)

        self._enrollment_stats[name] = {
            "images": len(image_files),
            "face_embeddings": len(face_embeddings),
            "body_base": body_added,
            "body_augmented": body_augmented,
            "body_total": self.body_gallery.sizes().get(name, 0),
        }
        logger.info("  %s enrollment: %s", name, self._enrollment_stats[name])

    def stats(self) -> Dict:
        return dict(self._enrollment_stats)


def enroll_students(verbose: bool = True) -> Tuple[Dict[str, np.ndarray], BodyGallery]:
    """
    Convenience function. Returns (face_profiles, body_gallery).
    Call once before running the tracker.
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
    enroller = StudentEnroller()
    face_profiles, body_gallery = enroller.enroll_all()
    if verbose:
        logger.info("=== Enrollment Stats ===")
        for name, stats in enroller.stats().items():
            logger.info("  %s: %s", name, stats)
    return face_profiles, body_gallery


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    face_profiles, body_gallery = enroll_students()
    print("\n[DONE] Enrollment complete.")
    print(f"  Face profiles: {list(face_profiles.keys())}")
    print(f"  Body gallery sizes: {body_gallery.sizes()}")
