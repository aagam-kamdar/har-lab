"""
identity_adapter.py - Thin public adapter wrapping the frozen identity & tracking pipeline.

Consumes the verified frozen components:
  - ByteTrack (short-term tracking)
  - SCRFD + ArcFace (authoritative identity locking)
  - OSNet-x0.25 (body Re-ID gap recovery)
  - IdentityLockManager (bijective ownership invariants & sticky locks)

Exports a clean per-frame interface:
  adapter.update(frame, frame_idx, det_boxes=None) -> List[Dict]
  yielding:
    [{ "track_id": int, "person_id": str, "display_name": str,
       "bbox": [x1, y1, x2, y2], "identity_confidence": float, "is_locked": bool,
       "status": str, "identity_state": str }]
"""

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytetrack import BYTETracker, BYTETrackerArgs
from features import FaceFeatureExtractor, BodyGallery
from run_final_reid_similarity_benchmark import ExtractorB
from enroll_students import enroll_students, KNOWN_STUDENTS
from identity_lock_manager import (
    IdentityLockManager,
    FACE_LOCK_THRESHOLD,
    FACE_AMBIGUITY_MARGIN,
    FACE_MIN_CONFIRMATIONS,
    BODY_RECOVERY_THRESHOLD,
    BODY_RECOVERY_AMBIGUITY,
    FACE_FRAME_GAP_FOR_RECOVERY,
)

logger = logging.getLogger("IdentityAdapter")

GALLERY_NPZ = "galleries_osnet_x0_25/gallery_embeddings.npz"

_CACHED_FACE_PROFILES = None
_CACHED_BODY_GALLERY = None
_CACHED_FACE_EXTRACTOR = None
_CACHED_BODY_EXTRACTOR = None


class IdentityState:
    UNIDENTIFIED = "UNIDENTIFIED"
    FACE_CANDIDATE = "FACE_CANDIDATE"
    IDENTITY_LOCKED = "IDENTITY_LOCKED"
    TRACK_OCCLUDED = "TRACK_OCCLUDED"
    REIDENTIFICATION_PENDING = "REIDENTIFICATION_PENDING"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"
    UNKNOWN = "UNKNOWN"


# Physical Lab Ground Truth Desk Mappings (Desk 1 = Harshil, Desk 3 = Aagam)
HOME_DESKS = {
    "harshil": "D01",
    "swayam": "D02",
    "aagam": "D03",
    "soham": "D04",
}

DESK_ROIS = {
    "D01": [450, 0, 570, 360],
    "D02": [540, 20, 660, 360],
    "D03": [640, 40, 780, 380],
    "D04": [770, 50, 960, 420],
}


@dataclass
class PersistentIdentity:
    """Persistent student identity record maintained independently from transient tracker IDs."""
    person_id: str
    display_name: str
    home_desk: str
    active_track_id: Optional[int] = None
    previous_track_id: Optional[int] = None
    identity_state: str = IdentityState.UNIDENTIFIED
    confidence: float = 0.0
    last_seen_frame: int = -1
    last_seen_t: float = 0.0
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_center: Optional[Tuple[float, float]] = None
    occluded_since_frame: Optional[int] = None
    lock_authority: Optional[str] = None
    desk_frames: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


def load_osnet_gallery(dim: int = 512, npz_path: str = GALLERY_NPZ) -> BodyGallery:
    """Load pre-enrolled OSNet body appearance gallery."""
    global _CACHED_BODY_GALLERY
    if _CACHED_BODY_GALLERY is not None:
        return _CACHED_BODY_GALLERY
    gal = BodyGallery(dim=dim)
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        for student in KNOWN_STUDENTS:
            if student in data:
                for vec in data[student]:
                    gal.add(student, vec)
    _CACHED_BODY_GALLERY = gal
    return gal


class IdentityAdapter:
    """
    Person-centric identity tracker wrapping frozen ByteTrack + ArcFace + OSNet Re-ID.
    Implements closed-set windowed assignment (Hungarian matching) across enrolled students
    with quality-gated, event-driven face recognition and sticky track identities.
    """

    def __init__(
        self,
        fps: float = 30.0,
        yolo_model_path: str = "yolo26n.pt",
        identifying_timeout_s: float = 5.0,
        live_gate_evidence_s: float = 1.5,
        clear_margin: float = 0.15,
        arcface_confirm_s: float = 0.5,
        body_stable_s: float = 3.0,
        mode: str = "redesigned",
        min_affinity_thresh: float = 0.50,
        body_only_cross_assignment: bool = False,
        stitching_enabled: bool = True,
        stitching_max_gap_s: float = 5.0,
        stitching_max_dist_px: float = 100.0,
        stitching_body_veto_thresh: float = 0.40,
        face_min_size: int = 24,
        face_min_det_conf: float = 0.55,
        face_min_sharpness: float = 30.0,
        face_temporal_independence_s: float = 0.20,
        face_min_confirmations: int = 2,
        face_event_interval_frames: int = 10,
    ):
        global _CACHED_FACE_PROFILES, _CACHED_FACE_EXTRACTOR, _CACHED_BODY_EXTRACTOR
        self.fps = float(fps)
        self.target_fps = fps
        self.identifying_timeout_s = float(identifying_timeout_s)
        self.live_gate_evidence_s = float(live_gate_evidence_s)
        self.clear_margin = float(clear_margin)
        self.arcface_confirm_s = float(arcface_confirm_s)
        self.body_stable_s = float(body_stable_s)
        self.mode = mode.lower()  # 'legacy', 'permissive', 'redesigned', 'stitching'
        self.min_affinity_thresh = float(min_affinity_thresh)
        self.body_only_cross_assignment = bool(body_only_cross_assignment)
        self.stitching_enabled = bool(stitching_enabled)
        self.stitching_max_gap_s = float(stitching_max_gap_s)
        self.stitching_max_dist_px = float(stitching_max_dist_px)
        self.stitching_body_veto_thresh = float(stitching_body_veto_thresh)

        # Quality gating & event-driven parameters
        self.face_min_size = int(face_min_size)
        self.face_min_det_conf = float(face_min_det_conf)
        self.face_min_sharpness = float(face_min_sharpness)
        self.face_temporal_independence_s = float(face_temporal_independence_s)
        self.face_min_confirmations = int(face_min_confirmations)
        self.face_event_interval_frames = int(face_event_interval_frames)

        # Spatio-temporal stitching state
        self.recent_ended_tracks: List[Dict] = []
        self.stitched_tracks: Dict[int, str] = {}
        self.track_last_center: Dict[int, Tuple[float, float]] = {}
        self.track_last_seen_f: Dict[int, int] = {}
        self.track_last_bbox: Dict[int, Tuple[int, int, int, int]] = {}

        # Candidate stability & lock tracking
        self.track_candidate: Dict[int, Optional[str]] = {}
        self.track_candidate_first_frame: Dict[int, int] = {}
        self.track_locked_since_frame: Dict[int, int] = {}

        # 1. ByteTrack configured for target FPS
        bt_args = BYTETrackerArgs()
        bt_args.track_thresh = 0.35
        bt_args.track_buffer = int(max(30, 90 * (self.fps / 30.0)))
        bt_args.match_thresh = 0.85
        bt_args.mot20 = False
        self.tracker = BYTETracker(bt_args, frame_rate=int(max(1, self.fps)))

        # 2. ArcFace Face Extractor & Enrolled Face Profiles (cached)
        if _CACHED_FACE_EXTRACTOR is None:
            _CACHED_FACE_EXTRACTOR = FaceFeatureExtractor(det_size=(320, 320))
        self.face_extractor = _CACHED_FACE_EXTRACTOR

        if _CACHED_FACE_PROFILES is None:
            _CACHED_FACE_PROFILES, _ = enroll_students(verbose=False)
        self.face_profiles = _CACHED_FACE_PROFILES

        # 3. OSNet Body Extractor & Gallery (cached)
        if _CACHED_BODY_EXTRACTOR is None:
            _CACHED_BODY_EXTRACTOR = ExtractorB()
        self.body_extractor = _CACHED_BODY_EXTRACTOR
        self.body_gallery = load_osnet_gallery(dim=512, npz_path=GALLERY_NPZ)

        # 4. Identity Lock Manager (standard single-resolution gate)
        self.lock_mgr = IdentityLockManager(require_dual_agreement=False)

        # 5. YOLO person detector fallback
        self.yolo_model = YOLO(yolo_model_path) if yolo_model_path and os.path.exists(yolo_model_path) else None

        # Tracking state
        self.face_confirmations = defaultdict(lambda: deque(maxlen=FACE_MIN_CONFIRMATIONS))
        self.recovered_map: Dict[int, str] = {}
        self._last_body_emb: Dict[int, np.ndarray] = {}

        # Closed-set tracking state
        self.track_first_seen_frame: Dict[int, int] = {}
        self.track_face_ema: Dict[int, Dict[str, float]] = defaultdict(lambda: {s: 0.0 for s in KNOWN_STUDENTS})
        self.track_body_ema: Dict[int, Dict[str, float]] = defaultdict(lambda: {s: 0.0 for s in KNOWN_STUDENTS})
        self.track_assigned: Dict[int, Tuple[str, float, bool]] = {}  # tid -> (student, conf, is_locked)
        self.track_assignment_margin: Dict[int, float] = {}
        self.track_assignment_affinity: Dict[int, float] = {}

        # Quality-gated candidate buffer & explicit identity states
        self.face_candidate_observations: Dict[int, List[Dict]] = defaultdict(list)
        self.track_last_face_frame: Dict[int, int] = {}
        self.pending_reid_tracks: Set[int] = set()
        self.conflict_tracks: Set[int] = set()
        self.track_identity_state: Dict[int, str] = {}

        # Structured Re-ID events history
        self.reid_events: List[Dict] = []
        self.contradictory_observations: Dict[int, List[Dict]] = defaultdict(list)
        self.duplicate_of: Dict[int, int] = {}

        # Persistent identity states (decoupled from transient tracker IDs)
        self.persistent_identities: Dict[str, PersistentIdentity] = {
            s: PersistentIdentity(
                person_id=s,
                display_name=s.capitalize(),
                home_desk=HOME_DESKS.get(s, ""),
            )
            for s in KNOWN_STUDENTS
        }

    @staticmethod
    def match_desk_zone(bbox: Tuple[int, int, int, int] | List[int]) -> Optional[str]:
        """Check if person bbox centroid falls inside a known desk ROI."""
        if not bbox or len(bbox) < 4:
            return None
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        for did, roi in DESK_ROIS.items():
            if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                return did
        return None

    def log_reid_event(
        self,
        track_id: int,
        prev_identity: str,
        prev_confidence: float,
        prev_track_id: Optional[int],
        trigger_event: str,
        face_meta: Optional[Dict] = None,
        candidates: Optional[Dict[str, float]] = None,
        decision: str = "PENDING",
        reason: str = "",
    ):
        """Structured logging for every transition into or out of REIDENTIFICATION_PENDING."""
        meta = face_meta or {}
        qual = meta.get("quality", 0.0)
        det_sc = meta.get("det_score", 0.0)
        sharp = meta.get("sharpness", 0.0)
        event_dict = {
            "track_id": track_id,
            "prev_identity": prev_identity,
            "prev_confidence": round(prev_confidence, 3),
            "prev_track_id": prev_track_id,
            "trigger_event": trigger_event,
            "face_quality": round(qual, 3),
            "det_score": round(det_sc, 3),
            "sharpness": round(sharp, 1),
            "candidates": candidates or {},
            "decision": decision,
            "reason": reason,
        }
        self.reid_events.append(event_dict)
        logger.info(
            "[RE-ID EVENT] Track #%d | Prev: %s (conf: %.1f%%, prev_tid: %s) | "
            "Trigger: %s | FaceQual: %.2f (det: %.2f, sharp: %.1f) | "
            "Candidates: %s | Decision: %s | Reason: %s",
            track_id, prev_identity, prev_confidence * 100, prev_track_id,
            trigger_event, qual, det_sc, sharp,
            candidates or {}, decision, reason
        )

    def extract_quality_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.3,
        use_640: bool = False,
    ) -> Optional[Tuple[np.ndarray, float, Dict]]:
        """
        Quality-gated face extraction from person bounding box.
        Applies SCRFD face detection and InsightFace ArcFace embedding,
        enforcing rigorous quality gates:
          - Bounding box dimensions (w >= face_min_size, h >= face_min_size)
          - Aspect ratio (0.55 <= w/h <= 1.55)
          - Detection confidence (det_score >= face_min_det_conf)
          - Laplacian variance blur/sharpness (var >= face_min_sharpness)
          - Geometric face-to-body centering (top-center rule)
          - 5-point landmark geometry (eye distance >= 8px, yaw asymmetry <= 3.8)
        Returns:
          (normed_embedding, quality_score, metadata_dict) or None if rejected.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        head_y2 = y1 + int(bh * 0.45)
        pad = int(bw * padding)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, head_y2 + pad)

        if cx2 - cx1 < 24 or cy2 - cy1 < 24:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        if crop is None or crop.size == 0:
            return None

        app = self.face_extractor.get_app_640() if use_640 else self.face_extractor.app
        try:
            faces = app.get(crop)
        except Exception:
            return None

        if not faces:
            return None

        body_cx = (x1 + x2) / 2.0
        best_candidate = None
        best_quality = -1.0
        best_meta = None

        for f in faces:
            fw = float(f.bbox[2] - f.bbox[0])
            fh = float(f.bbox[3] - f.bbox[1])
            if fw < self.face_min_size or fh < self.face_min_size:
                continue

            ar = fw / max(fh, 1.0)
            if ar < 0.55 or ar > 1.55:
                continue

            if float(f.det_score) < self.face_min_det_conf:
                continue

            fcx = (f.bbox[0] + f.bbox[2]) / 2.0 + cx1
            fcy = (f.bbox[1] + f.bbox[3]) / 2.0 + cy1
            norm_x = abs(fcx - body_cx) / max(bw, 1)
            norm_y = (fcy - y1) / max(bh, 1)
            if norm_x > 0.35 or norm_y > 0.45:
                continue

            fx1 = max(0, int(f.bbox[0]))
            fy1 = max(0, int(f.bbox[1]))
            fx2 = min(crop.shape[1], int(f.bbox[2]))
            fy2 = min(crop.shape[0], int(f.bbox[3]))
            face_img = crop[fy1:fy2, fx1:fx2]
            if face_img.size == 0:
                continue

            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if lap_var < self.face_min_sharpness:
                continue

            if hasattr(f, "kps") and f.kps is not None and len(f.kps) == 5:
                kps = f.kps
                eye_dist = float(np.linalg.norm(kps[0] - kps[1]))
                if eye_dist < 8.0:
                    continue
                d_left = float(np.linalg.norm(kps[0] - kps[2]))
                d_right = float(np.linalg.norm(kps[1] - kps[2]))
                asym = max(d_left, d_right) / max(min(d_left, d_right), 1e-4)
                if asym > 3.8:
                    continue

            norm_sharp = min(1.0, lap_var / 120.0)
            norm_size = min(1.0, (fw * fh) / (55.0 * 55.0))
            quality = 0.4 * float(f.det_score) + 0.3 * norm_sharp + 0.3 * norm_size

            if quality > best_quality:
                best_quality = quality
                best_candidate = f.normed_embedding.astype(np.float32)
                best_meta = {
                    "det_score": float(f.det_score),
                    "sharpness": lap_var,
                    "quality": quality,
                    "bbox": (fx1 + cx1, fy1 + cy1, fx2 + cx1, fy2 + cy1),
                }

        if best_candidate is not None:
            return best_candidate, best_quality, best_meta
        return None

    def get_track_identity_state(self, track_id: int) -> str:
        """Query the explicit identity state of a track."""
        if track_id in self.conflict_tracks:
            return IdentityState.IDENTITY_CONFLICT
        if track_id in self.pending_reid_tracks:
            return IdentityState.REIDENTIFICATION_PENDING
        state = self.lock_mgr.get_state(track_id)
        if state:
            if state.status == "LOST":
                return IdentityState.TRACK_OCCLUDED
            if state.locked:
                return IdentityState.IDENTITY_LOCKED
        return self.track_identity_state.get(track_id, IdentityState.UNIDENTIFIED)

    def update(self, frame: np.ndarray, frame_idx: int, det_boxes: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Process a single frame and return all active tracks with resolved identities.

        Args:
            frame: BGR numpy image (H, W, 3).
            frame_idx: monotonically increasing integer frame count.
            det_boxes: Optional numpy array of shape [N, 5] (x1, y1, x2, y2, conf) for person class.
                       If None, runs person detection via internal YOLO model.

        Returns:
            List of track dicts:
              [{
                  "track_id": int,
                  "person_id": str ("aagam", "swayam", "harshil", "soham", or "unknown"),
                  "display_name": str ("Aagam", "Swayam", or "Unknown #N"),
                  "bbox": [x1, y1, x2, y2],
                  "identity_confidence": float,
                  "is_locked": bool,
                  "status": str ("LOCKED", "ASSIGNED", "CANDIDATE", "IDENTIFYING", "CONFLICT", "UNKNOWN"),
                  "identity_state": str (IdentityState enum string)
              }]
        """
        h, w = frame.shape[:2]

        # 1. Person detections
        if det_boxes is None:
            if self.yolo_model is not None:
                res = self.yolo_model(frame, classes=[0], conf=0.25, verbose=False)[0]
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                if len(boxes) > 0:
                    det_boxes = np.hstack([boxes, confs[:, None]]).astype(np.float32)
                else:
                    det_boxes = np.empty((0, 5), dtype=np.float32)
            else:
                det_boxes = np.empty((0, 5), dtype=np.float32)

        # 2. ByteTrack update
        online_targets = self.tracker.update(det_boxes, [h, w], [h, w])
        tracked_ids = [int(t.track_id) for t in self.tracker.tracked_stracks]
        lost_ids = [int(t.track_id) for t in self.tracker.lost_stracks]
        removed_ids = [int(t.track_id) for t in self.tracker.removed_stracks]

        # Collect ended locked tracks before lock_mgr resets them
        ended_locked = {}
        for r_tid in removed_ids:
            state = self.lock_mgr.get_state(r_tid)
            if state and state.locked and state.locked_name:
                ended_locked[r_tid] = state.locked_name
            elif r_tid in self.track_assigned and self.track_assigned[r_tid][2]:
                ended_locked[r_tid] = self.track_assigned[r_tid][0]

        # 3. Synchronize lifecycle with IdentityLockManager
        self.lock_mgr.update_lifecycle(tracked_ids, lost_ids, removed_ids, current_frame=frame_idx)

        # Clean up removed/lost tracks from local assigned state and save to recent_ended_tracks
        for r_tid in removed_ids:
            named_student = ended_locked.get(r_tid)

            if named_student and r_tid in self.track_last_center:
                last_f = self.track_last_seen_f.get(r_tid, frame_idx)
                self.recent_ended_tracks.append({
                    "track_id": r_tid,
                    "student": named_student,
                    "frame_idx": last_f,
                    "timestamp_s": last_f / self.fps,
                    "last_center": self.track_last_center[r_tid],
                    "last_bbox": self.track_last_bbox.get(r_tid, (0, 0, 0, 0)),
                })
                p_id = self.persistent_identities.get(named_student)
                if p_id:
                    p_id.occluded_since_frame = frame_idx
                    p_id.previous_track_id = r_tid
                    p_id.active_track_id = None
                    p_id.identity_state = IdentityState.TRACK_OCCLUDED

            self.track_assigned.pop(r_tid, None)
            self.track_first_seen_frame.pop(r_tid, None)
            self.track_face_ema.pop(r_tid, None)
            self.track_body_ema.pop(r_tid, None)
            self.track_last_center.pop(r_tid, None)
            self.track_last_seen_f.pop(r_tid, None)
            self.track_last_bbox.pop(r_tid, None)
            self.stitched_tracks.pop(r_tid, None)
            self.face_candidate_observations.pop(r_tid, None)
            self.contradictory_observations.pop(r_tid, None)
            self.track_last_face_frame.pop(r_tid, None)
            self.pending_reid_tracks.discard(r_tid)
            self.conflict_tracks.discard(r_tid)
            self.track_identity_state.pop(r_tid, None)
            self.duplicate_of.pop(r_tid, None)

        # Prune old ended tracks beyond 2x stitching_max_gap_s
        curr_t_s = frame_idx / self.fps
        self.recent_ended_tracks = [
            e for e in self.recent_ended_tracks
            if (curr_t_s - e["timestamp_s"]) <= (self.stitching_max_gap_s * 2.0)
        ]

        active_tids = [int(target.track_id) for target in online_targets]
        for tid in active_tids:
            if tid not in self.track_first_seen_frame:
                self.track_first_seen_frame[tid] = frame_idx

        # 4. Extract face & body features for active tracks
        target_bboxes = {}
        for target in online_targets:
            tid = int(target.track_id)
            tlwh = target.tlwh
            x1 = max(0, int(tlwh[0]))
            y1 = max(0, int(tlwh[1]))
            x2 = min(w, int(tlwh[0] + tlwh[2]))
            y2 = min(h, int(tlwh[1] + tlwh[3]))
            bbox = (x1, y1, x2, y2)
            target_bboxes[tid] = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self.track_last_center[tid] = (cx, cy)
            self.track_last_seen_f[tid] = frame_idx
            self.track_last_bbox[tid] = bbox

        # Duplicate track detection & suppression: detect split/overlapping boxes on same person
        self.duplicate_of.clear()
        for i in range(len(active_tids)):
            t1 = active_tids[i]
            b1 = target_bboxes[t1]
            for j in range(i + 1, len(active_tids)):
                t2 = active_tids[j]
                b2 = target_bboxes[t2]
                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                if iw > 0 and ih > 0:
                    inter = iw * ih
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    union = a1 + a2 - inter
                    iou = inter / max(1.0, union)
                    if iou >= 0.45:
                        s1 = self.lock_mgr.get_state(t1)
                        s2 = self.lock_mgr.get_state(t2)
                        t1_l = bool(s1 and s1.locked and s1.locked_name)
                        t2_l = bool(s2 and s2.locked and s2.locked_name)
                        if t1_l and not t2_l:
                            self.duplicate_of[t2] = t1
                        elif t2_l and not t1_l:
                            self.duplicate_of[t1] = t2
                        elif self.track_first_seen_frame.get(t1, 0) <= self.track_first_seen_frame.get(t2, 0):
                            self.duplicate_of[t2] = t1
                        else:
                            self.duplicate_of[t1] = t2

        for target in online_targets:
            tid = int(target.track_id)
            bbox = target_bboxes[tid]
            x1, y1, x2, y2 = bbox

            self.lock_mgr.update_seen(tid, frame_idx)
            state = self.lock_mgr.get_state(tid)
            is_firmly_assigned = (tid in self.track_assigned and self.track_assigned[tid][1] >= 0.65)
            is_locked = bool(state and state.locked and state.locked_name)

            # Event-Driven Face Recognition Decision
            # 1. Duplicate tracks skip ArcFace to prevent stealing locks or causing ghost tracks
            # 2. Seated and locked tracks: ArcFace is bypassed to preserve sticky tracking without CPU overhead or angle-induced flips
            # 3. Meaningful events that trigger face evaluation:
            #    - Unassigned, candidate, pending re-id, or conflict tracks
            #    - Track undergoes significant relocation away from locked home desk
            run_face_event = False
            if tid not in self.duplicate_of:
                is_relocated = False
                if is_locked and state and state.locked_name:
                    home_d = HOME_DESKS.get(state.locked_name)
                    curr_d = self.match_desk_zone(bbox)
                    if curr_d and home_d and curr_d != home_d:
                        is_relocated = True

                if not is_locked or (tid in self.conflict_tracks) or (tid in self.pending_reid_tracks) or is_relocated:
                    run_face_event = (frame_idx % self.face_event_interval_frames == 0) or (tid not in self.track_face_ema)

            if run_face_event:
                face_res = self.extract_quality_face(frame, bbox)
                if face_res is not None:
                    emb, quality_score, meta = face_res
                    for s in KNOWN_STUDENTS:
                        sim = max(0.0, float(np.dot(emb, self.face_profiles[s])))
                        prev = self.track_face_ema[tid].get(s, 0.0)
                        self.track_face_ema[tid][s] = 0.5 * prev + 0.5 * sim

                    # Contradictory evidence check on already locked tracks
                    if state and state.locked and state.locked_name:
                        cur_locked = state.locked_name
                        best_any, score_any, second_any = self.face_extractor.top_match(emb, self.face_profiles)
                        margin_any = score_any - second_any
                        if best_any != cur_locked and score_any >= FACE_LOCK_THRESHOLD and margin_any >= FACE_AMBIGUITY_MARGIN:
                            self.contradictory_observations[tid].append({
                                "student": best_any,
                                "score": score_any,
                                "margin": margin_any,
                                "frame_idx": frame_idx,
                            })
                            recent_contra = [c for c in self.contradictory_observations[tid] if c["student"] == best_any]
                            if len(recent_contra) >= self.face_min_confirmations:
                                old_stud = cur_locked
                                old_conf = float(state.locked_face_score or 0.0)
                                if self.lock_mgr.student_owner.get(old_stud) == tid:
                                    self.lock_mgr.student_owner.pop(old_stud, None)
                                self.lock_mgr.track_to_student.pop(tid, None)
                                state.locked = False
                                state.locked_name = None
                                state.locked_face_score = 0.0
                                self.lock_mgr._lock(state, best_any, score_any, frame_idx)
                                self.track_assigned[tid] = (best_any, score_any, True)
                                self.contradictory_observations[tid].clear()
                                self.log_reid_event(
                                    track_id=tid,
                                    prev_identity=old_stud,
                                    prev_confidence=old_conf,
                                    prev_track_id=tid,
                                    trigger_event="contradictory_face_evidence",
                                    face_meta=meta,
                                    candidates={best_any: score_any, old_stud: float(np.dot(emb, self.face_profiles[old_stud]))},
                                    decision="LOCK_INVALIDATED_AND_REASSIGNED",
                                    reason=f"Face match strongly contradicted previous lock ({old_stud} -> {best_any}, score={score_any:.3f})",
                                )
                                continue

                    # Also submit to frozen LockManager if meeting high threshold
                    cand_set = set(KNOWN_STUDENTS) - set(self.lock_mgr.student_owner.keys())
                    if state and state.locked_name:
                        cand_set.add(state.locked_name)
                    if tid in self.conflict_tracks:
                        cand_set = set(KNOWN_STUDENTS)

                    filtered_profiles = {k: v for k, v in self.face_profiles.items() if k in cand_set}
                    if filtered_profiles:
                        best_name, best_score, second_score = self.face_extractor.top_match(emb, filtered_profiles)
                        margin = best_score - second_score

                        # Temporal independence check: require samples separated in time
                        last_f = self.track_last_face_frame.get(tid, -999)
                        min_gap_f = int(max(1, self.face_temporal_independence_s * self.fps))
                        if (frame_idx - last_f) >= min_gap_f:
                            self.track_last_face_frame[tid] = frame_idx

                            if best_score >= FACE_LOCK_THRESHOLD and margin >= FACE_AMBIGUITY_MARGIN:
                                self.face_candidate_observations[tid].append({
                                    "student": best_name,
                                    "score": best_score,
                                    "margin": margin,
                                    "quality": quality_score,
                                    "frame_idx": frame_idx,
                                })
                                if len(self.face_candidate_observations[tid]) > 5:
                                    self.face_candidate_observations[tid] = self.face_candidate_observations[tid][-5:]

                                recent_cands = [o for o in self.face_candidate_observations[tid] if o["student"] == best_name]
                                if len(recent_cands) >= self.face_min_confirmations:
                                    avg_sc = float(np.mean([o["score"] for o in recent_cands]))
                                    avg_mg = float(np.mean([o["margin"] for o in recent_cands]))
                                    self.lock_mgr.submit_face_match(tid, frame_idx, best_name, avg_sc, avg_mg)
                                    self.face_confirmations[tid].clear()
                                    self.face_candidate_observations[tid].clear()
                                    self.pending_reid_tracks.discard(tid)
                                    self.conflict_tracks.discard(tid)

            # OSNet Body Re-ID Appearance (supporting appearance only)
            if not is_locked:
                check_body = (not is_firmly_assigned and (frame_idx % 3 == 0 or tid not in self.track_body_ema)) or (is_firmly_assigned and frame_idx % 15 == 0)
                if check_body and (x2 - x1 >= 20 and y2 - y1 >= 40):
                    person_crop = frame[y1:y2, x1:x2]
                    b_emb = self.body_extractor.extract(person_crop)
                    if b_emb is not None:
                        res = self.body_gallery.search(b_emb, candidate_students=set(KNOWN_STUDENTS), top_k=len(KNOWN_STUDENTS))
                        for s, b_score in res:
                            prev = self.track_body_ema[tid].get(s, 0.0)
                            self.track_body_ema[tid][s] = 0.5 * prev + 0.5 * float(b_score)

        # 5. Closed-set Assignment
        assigned_students = set()
        # Authoritative ArcFace locks take top priority
        for tid in active_tids:
            state = self.lock_mgr.get_state(tid)
            if state and state.locked and state.locked_name:
                self.track_assigned[tid] = (state.locked_name, float(state.locked_face_score or 1.0), True)
                self.track_assignment_margin[tid] = float(state.locked_face_score or 1.0)
                self.track_assignment_affinity[tid] = float(state.locked_face_score or 1.0)
                if tid in self.stitched_tracks and self.stitched_tracks[tid] != state.locked_name:
                    del self.stitched_tracks[tid]
                assigned_students.add(state.locked_name)

        # Spatio-Temporal Stitching for unassigned tracks
        if self.mode != "legacy" and self.stitching_enabled:
            for tid in active_tids:
                if tid in self.track_assigned and self.track_assigned[tid][2]:
                    continue  # already face locked
                if tid in self.stitched_tracks:
                    st_name = self.stitched_tracks[tid]
                    if st_name not in assigned_students:
                        self.track_assigned[tid] = (st_name, 0.70, False)
                        assigned_students.add(st_name)
                    continue

                curr_c = self.track_last_center.get(tid)
                if curr_c is None:
                    continue

                best_match = None
                best_cand_idx = -1
                min_d = 999999.0
                for cand_idx, cand in enumerate(reversed(self.recent_ended_tracks)):
                    cand_name = cand["student"]
                    # 1. No overlap with another active track with this name
                    if cand_name in assigned_students:
                        continue
                    # 2. Time gap within stitching_max_gap_s
                    dt = curr_t_s - cand["timestamp_s"]
                    if not (0.0 <= dt <= self.stitching_max_gap_s):
                        continue
                    # 3. Temporal co-existence invariant:
                    # A track cannot be a continuation of cand if it was already alive for > 5 frames before cand ended!
                    first_seen = self.track_first_seen_frame.get(tid, frame_idx)
                    if first_seen < cand["frame_idx"] - 5:
                        continue
                    # 4. Spatial distance within stitching_max_dist_px
                    cand_c = cand["last_center"]
                    dist_px = float(np.hypot(curr_c[0] - cand_c[0], curr_c[1] - cand_c[1]))
                    if dist_px > self.stitching_max_dist_px:
                        continue
                    # 5. Workstation boundary invariant:
                    cand_desk = self.match_desk_zone(cand.get("last_bbox", (0, 0, 0, 0)))
                    curr_desk = self.match_desk_zone(target_bboxes[tid])
                    b_sim = self.track_body_ema[tid].get(cand_name, 0.0)
                    if cand_desk and curr_desk and cand_desk != curr_desk:
                        if b_sim < 0.55:
                            continue
                    # 6. Body appearance veto
                    if b_sim > 0.0 and b_sim < self.stitching_body_veto_thresh:
                        continue  # Appearance vetoed

                    if dist_px < min_d:
                        min_d = dist_px
                        best_match = (cand_name, b_sim or 0.65, dt, cand.get("track_id"))
                        best_cand_idx = len(self.recent_ended_tracks) - 1 - cand_idx

                if best_match:
                    st_name, st_score, dt, prev_tid = best_match
                    self.track_assigned[tid] = (st_name, float(st_score), False)
                    self.track_assignment_affinity[tid] = float(st_score)
                    self.track_assignment_margin[tid] = 0.20
                    self.stitched_tracks[tid] = st_name
                    assigned_students.add(st_name)
                    if dt > 1.5:
                        self.pending_reid_tracks.add(tid)
                        self.log_reid_event(
                            track_id=tid,
                            prev_identity=st_name,
                            prev_confidence=float(st_score),
                            prev_track_id=prev_tid,
                            trigger_event="stitching_gap",
                            face_meta=None,
                            candidates={st_name: float(st_score)},
                            decision="REIDENTIFICATION_PENDING",
                            reason=f"Spatio-temporal stitch with dt={dt:.2f}s > 1.5s",
                        )
                    if 0 <= best_cand_idx < len(self.recent_ended_tracks):
                        self.recent_ended_tracks.pop(best_cand_idx)

        if self.mode != "legacy":
            unassigned_tids = [tid for tid in active_tids if tid not in self.track_assigned or not self.track_assigned[tid][2]]
            unassigned_tids = [tid for tid in unassigned_tids if tid not in self.stitched_tracks]
            unassigned_students = [s for s in KNOWN_STUDENTS if s not in assigned_students]

            if unassigned_tids and unassigned_students:
                A = np.zeros((len(unassigned_tids), len(unassigned_students)), dtype=np.float32)
                for i, tid in enumerate(unassigned_tids):
                    for j, s in enumerate(unassigned_students):
                        f_s = self.track_face_ema[tid].get(s, 0.0)
                        b_s = self.track_body_ema[tid].get(s, 0.0)
                        if f_s >= 0.35 and b_s > 0.40:
                            A[i, j] = 0.60 * f_s + 0.40 * b_s
                        elif f_s >= 0.35:
                            A[i, j] = f_s
                        elif self.body_only_cross_assignment and b_s > 0.65:
                            A[i, j] = b_s
                        else:
                            A[i, j] = 0.0

                # Solve 1-to-1 Hungarian matching
                row_ind, col_ind = linear_sum_assignment(1.0 - A)
                for r, c in zip(row_ind, col_ind):
                    tid = unassigned_tids[r]
                    stud = unassigned_students[c]
                    aff = float(A[r, c])

                    # Minimum affinity rejection check:
                    if aff < self.min_affinity_thresh:
                        continue

                    other_affs = [float(A[r, k]) for k in range(len(unassigned_students)) if k != c]
                    margin = aff - (max(other_affs) if other_affs else 0.0)

                    f_s = self.track_face_ema[tid].get(stud, 0.0)

                    # Acceptance gating:
                    if (aff >= self.min_affinity_thresh and margin >= 0.08 and (f_s >= 0.35 or self.body_only_cross_assignment)):
                        self.track_assigned[tid] = (stud, aff, False)
                        self.track_assignment_margin[tid] = float(margin)
                        self.track_assignment_affinity[tid] = float(aff)
                        assigned_students.add(stud)

        # Conflict Detection & Spatial Desk Arbitration
        active_by_student = defaultdict(list)
        for tid in active_tids:
            if tid in self.duplicate_of:
                continue
            if tid in self.track_assigned:
                stud = self.track_assigned[tid][0]
                active_by_student[stud].append(tid)

        for stud, tids in active_by_student.items():
            if len(tids) > 1:
                home_d = HOME_DESKS.get(stud)
                home_roi = DESK_ROIS.get(home_d) if home_d else None

                def conflict_rank(t_id):
                    c = self.track_last_center.get(t_id, (0.0, 0.0))
                    in_home = False
                    if home_roi:
                        in_home = (home_roi[0] <= c[0] <= home_roi[2] and home_roi[1] <= c[1] <= home_roi[3])
                    s = self.lock_mgr.get_state(t_id)
                    is_l = bool(s and s.locked and s.locked_name == stud)
                    conf = self.track_assigned[t_id][1]
                    return (1 if in_home else 0, 1 if is_l else 0, conf)

                ranked_tids = sorted(tids, key=conflict_rank, reverse=True)
                winner = ranked_tids[0]
                for loser in ranked_tids[1:]:
                    loser_conf = float(self.track_assigned[loser][1]) if loser in self.track_assigned else 0.0
                    self.conflict_tracks.add(loser)
                    del self.track_assigned[loser]
                    self.stitched_tracks.pop(loser, None)
                    self.pending_reid_tracks.add(loser)
                    self.log_reid_event(
                        track_id=loser,
                        prev_identity=stud,
                        prev_confidence=loser_conf,
                        prev_track_id=winner,
                        trigger_event="spatial_conflict_lost",
                        face_meta=None,
                        candidates={stud: 0.0},
                        decision="IDENTITY_CONFLICT",
                        reason=f"Duplicate active claim for {stud}; Track #{winner} won arbitration",
                    )

        # 6. Format output tracks
        output_tracks = []
        for target in online_targets:
            tid = int(target.track_id)
            bbox = target_bboxes[tid]
            state = self.lock_mgr.get_state(tid)
            is_locked = bool(state and state.locked and state.locked_name)

            if self.mode == "legacy":
                # Legacy mode: original pipeline lock decisions, no extra gate, no Hungarian override
                if is_locked:
                    person_id = state.locked_name.lower()
                    display_name = state.locked_name.capitalize()
                    confidence = float(state.locked_face_score or 1.0)
                    status = "LOCKED"
                    identity_state = IdentityState.IDENTITY_LOCKED
                else:
                    person_id = f"unknown_{tid}"
                    display_name = f"Unknown #{tid}"
                    confidence = 0.0
                    status = "UNKNOWN"
                    identity_state = IdentityState.UNKNOWN
                affinity = confidence
                margin = 0.0
                likely_stud = state.locked_name if is_locked else None

            elif tid in self.duplicate_of:
                par_id = self.duplicate_of[tid]
                par_state = self.lock_mgr.get_state(par_id)
                if par_state and par_state.locked and par_state.locked_name:
                    stud = par_state.locked_name
                    person_id = stud.lower()
                    display_name = f"{stud.capitalize()} (Duplicate)"
                    confidence = float(par_state.locked_face_score or 1.0)
                    status = "DUPLICATE"
                    identity_state = IdentityState.IDENTITY_LOCKED
                elif par_id in self.track_assigned:
                    stud, conf, _ = self.track_assigned[par_id]
                    person_id = stud.lower()
                    display_name = f"{stud.capitalize()} (Duplicate)"
                    confidence = float(conf)
                    status = "DUPLICATE"
                    identity_state = IdentityState.FACE_CANDIDATE
                else:
                    person_id = f"duplicate_{tid}"
                    display_name = f"Duplicate #{tid}"
                    confidence = 0.0
                    status = "DUPLICATE"
                    identity_state = IdentityState.UNIDENTIFIED
                affinity = confidence
                margin = 0.0
                likely_stud = person_id if not person_id.startswith("duplicate") else None

            elif tid in self.conflict_tracks:
                identity_state = IdentityState.IDENTITY_CONFLICT
                status = "CONFLICT"
                person_id = f"conflict_{tid}"
                display_name = f"Identity Conflict #{tid}"
                confidence = 0.0
                affinity = 0.0
                margin = 0.0
                likely_stud = None

            elif is_locked and (tid in self.track_assigned) and self.track_assigned[tid][2]:
                stud, conf, _ = self.track_assigned[tid]
                identity_state = IdentityState.IDENTITY_LOCKED
                status = "LOCKED"
                person_id = stud.lower()
                display_name = stud.capitalize()
                confidence = float(conf)
                affinity = self.track_assignment_affinity.get(tid, float(conf))
                margin = self.track_assignment_margin.get(tid, 0.20)
                likely_stud = stud

            elif tid in self.pending_reid_tracks and (tid in self.track_assigned):
                stud, conf, _ = self.track_assigned[tid]
                identity_state = IdentityState.REIDENTIFICATION_PENDING
                status = "REID_PENDING"
                person_id = stud.lower()
                display_name = f"{stud.capitalize()} (Re-ID Pending)"
                confidence = float(conf)
                affinity = self.track_assignment_affinity.get(tid, float(conf))
                margin = self.track_assignment_margin.get(tid, 0.10)
                likely_stud = stud

            elif tid in self.track_assigned:
                stud, conf, is_l = self.track_assigned[tid]
                affinity = self.track_assignment_affinity.get(tid, float(conf))
                margin = self.track_assignment_margin.get(tid, 0.0)

                # Determine candidate stability
                prev_cand = self.track_candidate.get(tid)
                if prev_cand == stud:
                    cand_stable_f = self.track_candidate_first_frame.get(tid, frame_idx)
                else:
                    self.track_candidate[tid] = stud
                    self.track_candidate_first_frame[tid] = frame_idx
                    cand_stable_f = frame_idx
                cand_stable_s = (frame_idx - cand_stable_f) / max(1.0, self.fps)

                # Track ArcFace lock confirmation duration
                if is_locked:
                    if tid not in self.track_locked_since_frame:
                        self.track_locked_since_frame[tid] = frame_idx
                    lock_dur_s = (frame_idx - self.track_locked_since_frame[tid]) / max(1.0, self.fps)
                else:
                    self.track_locked_since_frame.pop(tid, None)
                    lock_dur_s = 0.0

                cands = self.face_candidate_observations.get(tid, [])
                if self.mode == "permissive":
                    if cand_stable_s >= self.live_gate_evidence_s or margin >= self.clear_margin:
                        person_id = stud.lower()
                        display_name = stud.capitalize()
                        confidence = float(conf)
                        status = "LOCKED" if is_locked else "ASSIGNED"
                        identity_state = IdentityState.IDENTITY_LOCKED if is_locked else IdentityState.FACE_CANDIDATE
                    else:
                        person_id = f"identifying_{tid}"
                        display_name = "Identifying..."
                        confidence = 0.0
                        status = "IDENTIFYING"
                        identity_state = IdentityState.UNIDENTIFIED
                elif is_locked and lock_dur_s >= self.arcface_confirm_s:
                    person_id = stud.lower()
                    display_name = stud.capitalize()
                    confidence = float(conf)
                    status = "LOCKED"
                    identity_state = IdentityState.IDENTITY_LOCKED
                elif tid in self.stitched_tracks:
                    person_id = stud.lower()
                    display_name = stud.capitalize()
                    confidence = float(conf)
                    status = "ASSIGNED"
                    identity_state = IdentityState.IDENTITY_LOCKED if conf >= 0.70 else IdentityState.REIDENTIFICATION_PENDING
                elif (not is_locked) and cand_stable_s >= self.body_stable_s:
                    person_id = stud.lower()
                    display_name = stud.capitalize()
                    confidence = float(conf)
                    status = "ASSIGNED"
                    identity_state = IdentityState.FACE_CANDIDATE
                elif len(cands) > 0:
                    person_id = f"identifying_{tid}"
                    display_name = f"Identifying (likely: {stud.capitalize()})"
                    confidence = float(conf)
                    status = "IDENTIFYING"
                    identity_state = IdentityState.FACE_CANDIDATE
                else:
                    person_id = f"identifying_{tid}"
                    display_name = f"Identifying (likely: {stud.capitalize()})"
                    confidence = 0.0
                    status = "IDENTIFYING"
                    identity_state = IdentityState.FACE_CANDIDATE
                likely_stud = self.track_candidate.get(tid)

            else:
                self.track_candidate[tid] = None
                self.track_candidate_first_frame.pop(tid, None)
                self.track_locked_since_frame.pop(tid, None)
                first_f = self.track_first_seen_frame.get(tid, frame_idx)
                duration_s = (frame_idx - first_f) / max(1.0, self.fps)
                cands = self.face_candidate_observations.get(tid, [])
                if len(cands) > 0:
                    top_cand = cands[-1]["student"]
                    person_id = f"identifying_{tid}"
                    display_name = f"Identifying (likely: {top_cand.capitalize()})"
                    confidence = 0.0
                    status = "IDENTIFYING"
                    identity_state = IdentityState.FACE_CANDIDATE
                    likely_stud = top_cand
                elif duration_s < self.identifying_timeout_s:
                    person_id = f"identifying_{tid}"
                    display_name = "Identifying..."
                    confidence = 0.0
                    status = "IDENTIFYING"
                    identity_state = IdentityState.UNIDENTIFIED
                    likely_stud = None
                else:
                    person_id = f"unknown_{tid}"
                    display_name = f"Unknown (not enrolled) #{tid}"
                    confidence = 0.0
                    status = "UNKNOWN"
                    identity_state = IdentityState.UNKNOWN
                    likely_stud = None
                affinity = 0.0
                margin = 0.0

            self.track_identity_state[tid] = identity_state

            output_tracks.append({
                "track_id": tid,
                "person_id": person_id,
                "display_name": display_name,
                "likely_student": likely_stud,
                "bbox": bbox,
                "identity_confidence": round(confidence, 3),
                "affinity": round(affinity, 3),
                "margin": round(margin, 3),
                "is_locked": is_locked,
                "status": status,
                "identity_state": identity_state,
            })

        # 7. Synchronize persistent student identities
        for s in KNOWN_STUDENTS:
            p_id = self.persistent_identities[s]
            matching = [tr for tr in output_tracks if tr["person_id"] == s and tr["status"] != "DUPLICATE"]
            if matching:
                best_tr = matching[0]
                if p_id.active_track_id is not None and p_id.active_track_id != best_tr["track_id"]:
                    p_id.previous_track_id = p_id.active_track_id
                p_id.active_track_id = best_tr["track_id"]
                p_id.identity_state = best_tr["identity_state"]
                p_id.confidence = best_tr["identity_confidence"]
                p_id.last_seen_frame = frame_idx
                p_id.last_seen_t = frame_idx / self.fps
                p_id.last_bbox = best_tr["bbox"]
                cx = (best_tr["bbox"][0] + best_tr["bbox"][2]) / 2.0
                cy = (best_tr["bbox"][1] + best_tr["bbox"][3]) / 2.0
                p_id.last_center = (cx, cy)
                p_id.occluded_since_frame = None
                p_id.lock_authority = best_tr["status"]
                desk_id = self.match_desk_zone(best_tr["bbox"])
                if desk_id and best_tr.get("is_locked"):
                    p_id.desk_frames[desk_id] += 1
            else:
                if p_id.active_track_id is not None:
                    p_id.previous_track_id = p_id.active_track_id
                    p_id.active_track_id = None
                    p_id.occluded_since_frame = frame_idx
                    p_id.identity_state = IdentityState.TRACK_OCCLUDED

        return output_tracks
