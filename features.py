"""
features.py
-----------
Feature extraction utilities:
  - ArcFace face embeddings via InsightFace buffalo_sc
  - Body Re-ID embeddings via OSNetBodyFeatureExtractor (OSNet-x0.25, 512-dim)
    Active model wired into pipeline. Replaces MobileNetV3+HSV (672-dim).
  - Cosine similarity helpers
  - FAISS-based body gallery with deduplication and bounded size
"""

from __future__ import annotations

import os
import logging
import warnings
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Thresholds imported from identity_lock_manager to stay consistent
from identity_lock_manager import BODY_GALLERY_MAX, BODY_GALLERY_DEDUP_THRESH


# --------------------------------------------------------------------------- #
#  Face Feature Extractor (InsightFace buffalo_sc — ArcFace 512-dim)         #
# --------------------------------------------------------------------------- #

class FaceFeatureExtractor:
    """
    Wraps InsightFace buffalo_sc for fast face detection + ArcFace embedding.
    Uses only 'detection' and 'recognition' modules (det_500m + w600k_mbf).
    """

    def __init__(self, model_name: str = "buffalo_sc", det_size: Tuple[int, int] = (320, 320)):
        import insightface
        from insightface.app import FaceAnalysis

        self.model_name = model_name
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=-1, det_size=det_size)  # CPU
        self.det_size = det_size
        self._app_640: Optional[FaceAnalysis] = None
        logger.info("FaceFeatureExtractor: buffalo_sc ready (det_size=%s)", det_size)

    def get_app_640(self):
        """Lazy initializer for 640x640 detector to avoid unnecessary startup memory/latency."""
        if self._app_640 is None:
            from insightface.app import FaceAnalysis
            app_640 = FaceAnalysis(
                name=self.model_name,
                allowed_modules=["detection", "recognition"],
            )
            app_640.prepare(ctx_id=-1, det_size=(640, 640))
            self._app_640 = app_640
            logger.info("FaceFeatureExtractor: secondary buffalo_sc ready (det_size=(640, 640))")
        return self._app_640

    def extract_from_image(self, bgr_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest face and return its 512-dim L2-normalized ArcFace embedding.
        Returns None if no face found.
        """
        if bgr_image is None or bgr_image.size == 0:
            return None
        try:
            faces = self.app.get(bgr_image)
        except Exception as exc:
            logger.debug("FaceExtract error: %s", exc)
            return None

        if not faces:
            return None

        # Choose largest face by bounding-box area
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = best.normed_embedding  # already L2-normalized by insightface
        return emb.astype(np.float32)

    def extract_from_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.3,
        use_640: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Given a person bbox (x1,y1,x2,y2), crop the upper portion (head region)
        with optional padding and extract ArcFace embedding.
        Supports time-based escalation to det_size=(640, 640) via use_640.
        Mandatory geometric face-to-body rule applies at both 320 and 640.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        # Crop upper ~45% of person bbox as head region
        head_y2 = y1 + int(bh * 0.45)
        pad = int(bw * padding)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, head_y2 + pad)

        if cx2 - cx1 < 20 or cy2 - cy1 < 20:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        if crop is None or crop.size == 0:
            return None

        app = self.get_app_640() if use_640 else self.app
        try:
            faces = app.get(crop)
        except Exception as exc:
            logger.debug("FaceExtract error: %s", exc)
            return None

        if not faces:
            return None

        # Filter candidate faces: face center must adhere to top-center face-to-body rule:
        # 1. Horizontal face center must be within central 30% of person bbox (|fcx - body_cx| / body_w <= 0.30)
        # 2. Vertical face center must be within top 45% of person bbox ((fcy - body_top) / body_h <= 0.45)
        # This prevents background or adjacent neighbors from bleeding into the target crop.
        valid_faces = []
        body_cx = (x1 + x2) / 2.0
        for f in faces:
            fcx = (f.bbox[0] + f.bbox[2]) / 2.0 + cx1
            fcy = (f.bbox[1] + f.bbox[3]) / 2.0 + cy1
            norm_x = abs(fcx - body_cx) / max(bw, 1)
            norm_y = (fcy - y1) / max(bh, 1)
            if norm_x <= 0.30 and 0.0 <= norm_y <= 0.45:
                valid_faces.append(f)

        if not valid_faces:
            return None

        # Select largest face among valid candidates
        best = max(valid_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = best.normed_embedding
        return emb.astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors."""
        return float(np.dot(a, b))

    def top_match(
        self,
        query_emb: np.ndarray,
        gallery: Dict[str, np.ndarray],
    ) -> Tuple[str, float, float]:
        """
        Match query against a {name: embedding} gallery.
        Returns (best_name, best_score, second_best_score).
        """
        scores = {name: self.cosine_similarity(query_emb, emb) for name, emb in gallery.items()}
        sorted_scores = sorted(scores.values(), reverse=True)
        best_name = max(scores, key=scores.get)
        best_score = sorted_scores[0]
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        return best_name, best_score, second_score


# --------------------------------------------------------------------------- #
#  Body Feature Extractor — LEGACY (MobileNetV3 + HSV Spatial Color)          #
#  NOT wired into the pipeline. Kept for reference / A/B comparison only.     #
# --------------------------------------------------------------------------- #

class BodyFeatureExtractor:
    """
    Wraps PersonReIDExtractor from reid/embedding_model.py.
    Extracts 672-dim body Re-ID embeddings.

    LEGACY: Not wired into robust_tracker.py.
    Use OSNetBodyFeatureExtractor (512-dim) for the active pipeline.
    """

    def __init__(self):
        from reid.embedding_model import PersonReIDExtractor
        self.extractor = PersonReIDExtractor()
        logger.info("BodyFeatureExtractor (LEGACY): PersonReIDExtractor ready (672-dim)")

    def extract(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract body Re-ID embedding from a BGR person crop.
        Returns L2-normalized 672-dim float32 vector or None.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        try:
            emb = self.extractor.extract(bgr_crop)
            if emb is None:
                return None
            emb = emb.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                return None
            return emb / norm
        except Exception as exc:
            logger.debug("BodyExtract error: %s", exc)
            return None

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


# --------------------------------------------------------------------------- #
#  Body Feature Extractor — ACTIVE (OSNet-x0.25, 512-dim)                     #
# --------------------------------------------------------------------------- #

class OSNetBodyFeatureExtractor:
    """
    Wraps OSNet-x0.25 from reid/osnet.py for body Re-ID.
    Output: 512-dim L2-normalized float32 embedding.

    This is the ACTIVE body Re-ID model wired into robust_tracker.py.
    Replaces MobileNetV3+HSV (672-dim) as of the OSNet integration swap.

    Weights: reid/weights/osnet_x0_25_msmt17.pth (MSMT17-trained).

    Benchmark results (offline, vs. MobileNetV3+HSV):
      - FPR: 0.36% vs 15.68%  (97.9% precision vs 50.78%)
      - FNR: 49.2% vs 50.88%  (recall similar: ~50%)
      - Mean class separation: 0.2024 vs 0.0704
      - Threshold used: 0.68 (vs 0.78 for MobileNetV3)
      - Margin used: 0.05 (vs 0.06 for MobileNetV3)

    EVT_BT_2 status: OPEN. Both MobileNetV3 and OSNet-x0.25 independently
    produce FALSE_RECOVERY for EVT_BT_2 (dark-clothing aagam→soham confusion).
    OSNet scores aagam@soham=0.8956 with margin 0.2397 — a stronger wrong match.
    EVT_BT_2 remains an OPEN KNOWN LIMITATION regardless of model choice.
    """

    WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "reid", "weights", "osnet_x0_25_msmt17.pth")
    # Preprocessing constants (ImageNet mean/std, consistent with MSMT17 training)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    DIM = 512

    def __init__(self, weights_path: Optional[str] = None):
        import importlib.util
        import torch

        wp = weights_path or self.WEIGHTS_PATH
        self._torch = torch

        # Load reid/osnet.py directly to avoid triggering reid/__init__.py
        # (which imports FAISSManager → faiss, an optional dependency not
        # needed for OSNet inference).
        _osnet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reid", "osnet.py")
        _spec = importlib.util.spec_from_file_location("reid_osnet_standalone", _osnet_path)
        osnet_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(osnet_mod)

        self.model = osnet_mod.osnet_x0_25(num_classes=1000, pretrained=False)
        ckpt = torch.load(wp, map_location="cpu")
        matched = {
            k: v for k, v in ckpt.items()
            if k in self.model.state_dict()
            and v.shape == self.model.state_dict()[k].shape
        }
        self.model.load_state_dict(matched, strict=False)
        self.model.eval()
        logger.info(
            "OSNetBodyFeatureExtractor: OSNet-x0.25 ready (%d-dim), weights=%s, matched_keys=%d",
            self.DIM, wp, len(matched),
        )

    def extract(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 512-dim L2-normalized body Re-ID embedding from a BGR person crop.
        Input is resized to 128x256 (W x H) before inference.
        Returns float32 ndarray or None on failure.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        if bgr_crop.shape[0] < 12 or bgr_crop.shape[1] < 12:
            return None
        try:
            # Resize to OSNet standard input: width=128, height=256
            res = cv2.resize(bgr_crop, (128, 256))
            # BGR → RGB, normalize to [0,1], then ImageNet-normalize
            rgb = res[:, :, ::-1].astype(np.float32) / 255.0
            norm = (rgb - self._MEAN) / self._STD
            tensor = self._torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
            with self._torch.no_grad():
                out = self.model(tensor).squeeze().numpy()
            l2 = np.linalg.norm(out)
            if l2 < 1e-8:
                return None
            return (out / l2).astype(np.float32)
        except Exception as exc:
            logger.debug("OSNetBodyExtract error: %s", exc)
            return None

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


# --------------------------------------------------------------------------- #
#  Body Gallery with FAISS backend + deduplication + bounded size            #
# --------------------------------------------------------------------------- #

class BodyGallery:
    """
    Per-student body embedding gallery backed by FAISS (IndexFlatIP).
    Enforces:
      - Max BODY_GALLERY_MAX embeddings per student
      - Near-duplicate removal at BODY_GALLERY_DEDUP_THRESH
    """

    def __init__(self, dim: int = 672):
        import faiss
        self.dim = dim
        self._galleries: Dict[str, faiss.IndexFlatIP] = {}
        self._counts: Dict[str, int] = {}
        self._raw: Dict[str, List[np.ndarray]] = {}  # keep raw for dedup check

    def _ensure_student(self, name: str):
        import faiss
        if name not in self._galleries:
            self._galleries[name] = faiss.IndexFlatIP(self.dim)
            self._counts[name] = 0
            self._raw[name] = []

    def add(self, name: str, embedding: np.ndarray) -> bool:
        """
        Add embedding for student. Returns True if added, False if rejected.
        Rejects if: near-duplicate detected OR gallery full.
        """
        self._ensure_student(name)

        n = self._counts[name]
        if n >= BODY_GALLERY_MAX:
            return False  # Gallery full — no eviction for simplicity

        # Near-duplicate check
        if n > 0:
            sim = self._max_sim(name, embedding)
            if sim >= BODY_GALLERY_DEDUP_THRESH:
                return False  # Near-duplicate

        vec = embedding.astype(np.float32).reshape(1, -1)
        self._galleries[name].add(vec)
        self._raw[name].append(embedding)
        self._counts[name] += 1
        return True

    def search(
        self,
        query: np.ndarray,
        candidate_students: Set[str],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Search student galleries strictly among candidate_students.
        Returns list of (name, best_score) sorted descending.
        Only candidate_students are scored — unowned/allowed students only.
        """
        if not candidate_students:
            return []

        results = []
        q = query.astype(np.float32).reshape(1, -1)
        for name in candidate_students:
            index = self._galleries.get(name)
            if index is None or index.ntotal == 0:
                continue
            k = min(top_k, index.ntotal)
            scores, _ = index.search(q, k)
            best = float(scores[0][0])
            results.append((name, best))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def top_match(
        self,
        query: np.ndarray,
        candidate_students: Set[str],
    ) -> Tuple[Optional[str], float, float]:
        """
        Returns (best_name, best_score, second_best_score) restricted to candidate_students.
        If candidate_students is empty or no valid candidates found, returns (None, 0.0, 0.0).
        """
        if not candidate_students:
            return (None, 0.0, 0.0)

        results = self.search(query, candidate_students=candidate_students, top_k=3)
        if not results:
            return (None, 0.0, 0.0)
        best_name, best_score = results[0]
        second_score = results[1][1] if len(results) > 1 else 0.0
        return best_name, best_score, second_score

    def sizes(self) -> Dict[str, int]:
        return {name: cnt for name, cnt in self._counts.items()}

    def _max_sim(self, name: str, query: np.ndarray) -> float:
        q = query.astype(np.float32).reshape(1, -1)
        k = min(3, self._counts[name])
        scores, _ = self._galleries[name].search(q, k)
        return float(scores[0][0])


# --------------------------------------------------------------------------- #
#  Cosine similarity utility                                                  #
# --------------------------------------------------------------------------- #

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))
