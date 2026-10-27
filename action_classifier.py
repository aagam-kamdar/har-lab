"""
action_classifier.py
--------------------
Skeleton-based action classifier for Student Productivity Tracker.

ACTIONS (7 classes):
  1. sitting       - Seated, relatively still
  2. writing       - One-handed motion at desk level (pen/pencil)
  3. typing        - Both wrists symmetric at desk level
  4. phone         - Wrist near face/ear region
  5. sleeping      - Head (nose) significantly below shoulder level
  6. standing      - Large hip-ankle vertical span; or walking (displacement)
  7. talking       - Close to another tracked person + head turned laterally

Detection strategy:
  - Primary: YOLO-Pose 17-keypoint geometry rules (zero training needed)
  - Optional override: pretrained YOLO detection model (plug-in via CLASSIFIER_MODEL_PATH)

COCO Keypoint indices (yolo/ultralytics convention):
  0: nose         1: left_eye     2: right_eye
  3: left_ear     4: right_ear    5: left_shoulder
  6: right_shoulder 7: left_elbow  8: right_elbow
  9: left_wrist   10: right_wrist  11: left_hip
  12: right_hip   13: left_knee   14: right_knee
  15: left_ankle  16: right_ankle
"""

from __future__ import annotations

import os
import time
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2

logger = logging.getLogger("ActionClassifier")

# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #

KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16,
}

ACTION_LABELS = [
    "sitting",   # neutral attentive seated
    "reading",   # reading notes / screen / textbook
    "writing",   # holding pen/pencil writing at desk
    "typing",    # typing on laptop/keyboard
    "phone",     # holding or looking at phone
    "sleeping",  # head on desk / slumped
    "standing",  # upright stationary posture
    "walking",   # dynamic movement / walking
    "discuss",   # discussing / talking with peers
]

# Smoothing window: majority-vote over N frames
SMOOTH_WINDOW = 15

# Confidence threshold below which keypoint is considered invisible
KP_CONF_THRESH = 0.3

# Pretrained YOLO action detection model path (plug in SCB or Roboflow model here)
CLASSIFIER_MODEL_PATH = os.environ.get("ACTION_MODEL_PATH", "")


# --------------------------------------------------------------------------- #
#  Data structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class ActionResult:
    action: str                  # One of ACTION_LABELS
    confidence: float            # 0..1 heuristic confidence
    raw_scores: Dict[str, float] = field(default_factory=dict)
    source: str = "rules"        # "rules" | "yolo_model"
    note: str = ""


# --------------------------------------------------------------------------- #
#  Optional YOLO plug-in model                                                 #
# --------------------------------------------------------------------------- #

class _YOLOActionModel:
    """Wraps an optional pretrained YOLO detection model for action recognition."""

    def __init__(self, model_path: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.names = self.model.names
        logger.info("Loaded YOLO action model from %s — classes: %s", model_path, self.names)

    def predict(self, crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """Returns (best_class_name, confidence) or None."""
        if crop is None or crop.size == 0:
            return None
        try:
            res = self.model(crop, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                return None
            best_idx = int(res.boxes.cls[res.boxes.conf.argmax()])
            best_conf = float(res.boxes.conf.max())
            return self.names[best_idx], best_conf
        except Exception as e:
            logger.debug("YOLO action model error: %s", e)
            return None


# --------------------------------------------------------------------------- #
#  Keypoint utility functions                                                   #
# --------------------------------------------------------------------------- #

def _get_kp(kps: np.ndarray, name: str) -> Optional[Tuple[float, float, float]]:
    """
    kps: shape (17, 3) — x, y, conf  OR shape (17, 2) — x, y (conf=1)
    Returns (x, y, conf) or None if invisible.
    """
    idx = KP[name]
    if idx >= len(kps):
        return None
    if kps.shape[1] == 3:
        x, y, c = float(kps[idx, 0]), float(kps[idx, 1]), float(kps[idx, 2])
        if c < KP_CONF_THRESH or (x == 0 and y == 0):
            return None
        return x, y, c
    else:
        x, y = float(kps[idx, 0]), float(kps[idx, 1])
        if x == 0 and y == 0:
            return None
        return x, y, 1.0


def _mid(a: Optional[Tuple], b: Optional[Tuple]) -> Optional[Tuple[float, float]]:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _dy(a: Optional[Tuple], b: Optional[Tuple]) -> Optional[float]:
    """Vertical distance: positive = a is ABOVE b (smaller y value in image coords)."""
    if a is None or b is None:
        return None
    return b[1] - a[1]   # positive means a is above b


def _dist(a: Optional[Tuple], b: Optional[Tuple]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))


# --------------------------------------------------------------------------- #
#  Rule-based classifier                                                        #
# --------------------------------------------------------------------------- #

def _classify_rules(
    kps: np.ndarray,
    bbox_h: float,
    nearby_students: List[str],
    position_history: deque,  # deque of (cx, cy)
) -> ActionResult:
    """
    Apply geometric rules on COCO keypoints to classify action.

    Args:
        kps: (17, 2) or (17, 3) keypoint array
        bbox_h: height of person bounding box in pixels (scale reference)
        nearby_students: list of student names whose bbox centres are close
        position_history: deque of recent (cx, cy) positions for motion detection

    Returns: ActionResult
    """
    unit = max(bbox_h, 50.0)   # normalisation unit

    # Extract key landmarks
    nose      = _get_kp(kps, "nose")
    l_sh      = _get_kp(kps, "l_shoulder")
    r_sh      = _get_kp(kps, "r_shoulder")
    l_el      = _get_kp(kps, "l_elbow")
    r_el      = _get_kp(kps, "r_elbow")
    l_wr      = _get_kp(kps, "l_wrist")
    r_wr      = _get_kp(kps, "r_wrist")
    l_hip     = _get_kp(kps, "l_hip")
    r_hip     = _get_kp(kps, "r_hip")
    l_kn      = _get_kp(kps, "l_knee")
    r_kn      = _get_kp(kps, "r_knee")
    l_an      = _get_kp(kps, "l_ankle")
    r_an      = _get_kp(kps, "r_ankle")
    l_ear     = _get_kp(kps, "l_ear")
    r_ear     = _get_kp(kps, "r_ear")

    mid_sh    = _mid(l_sh, r_sh)
    mid_hip   = _mid(l_hip, r_hip)
    mid_kn    = _mid(l_kn, r_kn)
    mid_an    = _mid(l_an, r_an)
    mid_wr    = _mid(l_wr, r_wr)

    scores: Dict[str, float] = {a: 0.0 for a in ACTION_LABELS}

    # ------------------------------------------------------------------ #
    # SLEEPING: nose below shoulder level by significant margin
    # ------------------------------------------------------------------ #
    if nose and mid_sh:
        dy_nose_sh = _dy(nose, mid_sh)   # negative = nose is BELOW shoulders
        if dy_nose_sh is not None and dy_nose_sh < -0.08 * unit:
            scores["sleeping"] += 0.7
        if dy_nose_sh is not None and dy_nose_sh < -0.15 * unit:
            scores["sleeping"] += 0.3

    # ------------------------------------------------------------------ #
    # PHONE: wrist near ear/nose region
    # ------------------------------------------------------------------ #
    face_refs = [p for p in [nose, l_ear, r_ear] if p is not None]
    if face_refs:
        face_y = float(np.mean([p[1] for p in face_refs]))
        face_x = float(np.mean([p[0] for p in face_refs]))
        for wr in [l_wr, r_wr]:
            if wr is not None:
                d_face = float(np.sqrt((wr[0] - face_x)**2 + (wr[1] - face_y)**2))
                if d_face < 0.20 * unit:
                    scores["phone"] += 0.55
                if d_face < 0.10 * unit:
                    scores["phone"] += 0.30

    # ------------------------------------------------------------------ #
    # STANDING / WALKING
    # ------------------------------------------------------------------ #
    hip_ankle_dy = None
    if mid_hip and mid_an:
        hip_ankle_dy = _dy(mid_hip, mid_an)   # large positive = standing
    elif mid_hip and mid_kn:
        hip_ankle_dy = _dy(mid_hip, mid_kn)

    if hip_ankle_dy is not None:
        if hip_ankle_dy > 0.55 * unit:
            scores["standing"] += 0.6
        elif hip_ankle_dy > 0.40 * unit:
            scores["standing"] += 0.3

    # Walking: dynamic displacement across frames (distinct from stationary standing)
    if len(position_history) >= 5:
        pts = list(position_history)[-5:]
        displacements = [
            float(np.sqrt((pts[i][0] - pts[i-1][0])**2 + (pts[i][1] - pts[i-1][1])**2))
            for i in range(1, len(pts))
        ]
        avg_disp = float(np.mean(displacements)) if displacements else 0.0
        if avg_disp > 0.035 * unit:
            scores["walking"] += min(avg_disp / (0.07 * unit), 0.75)
            scores["standing"] = max(0.0, scores["standing"] - 0.45)

    # ------------------------------------------------------------------ #
    # SITTING: hip-to-knee angle is roughly horizontal; small body span
    # ------------------------------------------------------------------ #
    if mid_hip and mid_kn:
        hip_knee_dy = _dy(mid_hip, mid_kn)
        if hip_knee_dy is not None and hip_knee_dy < 0.35 * unit:
            scores["sitting"] += 0.4
        if mid_sh and _dy(mid_sh, mid_hip) is not None:
            torso_h = _dy(mid_sh, mid_hip)
            body_span = hip_ankle_dy if hip_ankle_dy is not None else torso_h
            if body_span is not None and body_span < 0.45 * unit:
                scores["sitting"] += 0.3

    # ------------------------------------------------------------------ #
    # TYPING: both wrists visible, similar height, at desk (low in frame)
    #         symmetric horizontal positions, small lateral movement
    # ------------------------------------------------------------------ #
    if l_wr and r_wr and mid_sh:
        wr_dy = abs(l_wr[1] - r_wr[1])           # vertical symmetry
        wr_dx = abs(l_wr[0] - r_wr[0])           # horizontal spread
        sh_to_lwr = _dy(mid_sh, l_wr)
        sh_to_rwr = _dy(mid_sh, r_wr)

        # Both wrists below shoulder (positive dy) and similar height
        if (sh_to_lwr is not None and sh_to_lwr > 0.05 * unit and
                sh_to_rwr is not None and sh_to_rwr > 0.05 * unit):
            scores["typing"] += 0.3
        if wr_dy < 0.12 * unit:   # wrists at similar heights
            scores["typing"] += 0.25
        if 0.15 * unit < wr_dx < 0.70 * unit:  # typical keyboard spread
            scores["typing"] += 0.2
        # Both elbows bent (not arms down)
        if l_el and r_el:
            if (l_el[1] < l_wr[1] and r_el[1] < r_wr[1]):  # elbows above wrists
                scores["typing"] += 0.15

    # ------------------------------------------------------------------ #
    # WRITING: one wrist (dominant) at desk level, other may be anywhere
    #          head slightly forward/down, asymmetric wrist use
    # ------------------------------------------------------------------ #
    desk_wrist = None
    other_wrist = None
    if l_wr and r_wr and mid_sh:
        sh_to_lwr = _dy(mid_sh, l_wr)
        sh_to_rwr = _dy(mid_sh, r_wr)
        if sh_to_lwr is not None and sh_to_rwr is not None:
            if abs(sh_to_lwr - sh_to_rwr) > 0.10 * unit:
                # Asymmetric - one wrist clearly lower
                if sh_to_lwr > sh_to_rwr:
                    desk_wrist, other_wrist = l_wr, r_wr
                else:
                    desk_wrist, other_wrist = r_wr, l_wr
    elif l_wr and mid_sh:
        sh_to_lwr = _dy(mid_sh, l_wr)
        if sh_to_lwr is not None and sh_to_lwr > 0.10 * unit:
            desk_wrist = l_wr
    elif r_wr and mid_sh:
        sh_to_rwr = _dy(mid_sh, r_wr)
        if sh_to_rwr is not None and sh_to_rwr > 0.10 * unit:
            desk_wrist = r_wr

    if desk_wrist is not None:
        scores["writing"] += 0.45
        # Head tilted forward (nose below shoulders)
        if nose and mid_sh:
            dy_nose_sh = _dy(nose, mid_sh)
            if dy_nose_sh is not None and -0.08 * unit < dy_nose_sh < 0.0:
                scores["writing"] += 0.20
        # Sitting + writing is stronger
        if scores["sitting"] > 0.3:
            scores["writing"] += 0.15
    else:
        # Seated with head tilted forward looking at desk/book/screen without active writing = READING
        if scores["sitting"] > 0.3 and nose and mid_sh:
            dy_nose_sh = _dy(nose, mid_sh)
            if dy_nose_sh is not None and -0.09 * unit < dy_nose_sh < 0.0:
                scores["reading"] += 0.50

    # ------------------------------------------------------------------ #
    # DISCUSS: Close to another tracked person + head turned laterally
    # ------------------------------------------------------------------ #
    if nearby_students:
        scores["discuss"] += 0.40
        # Head turned: one ear more visible than other
        if l_ear and not r_ear:
            scores["discuss"] += 0.20  # head turned right
        elif r_ear and not l_ear:
            scores["discuss"] += 0.20  # head turned left
        elif l_ear and r_ear:
            ear_dx = abs(l_ear[0] - r_ear[0])
            if ear_dx < 0.08 * unit:  # face profile (not frontal)
                scores["discuss"] += 0.15

    # ------------------------------------------------------------------ #
    # Resolve: pick action with highest score; apply simple hierarchy
    # ------------------------------------------------------------------ #
    # Hierarchy: sleeping > phone > talking > (typing vs writing) > standing > sitting
    priority_override = None
    if scores["sleeping"] >= 0.6:
        priority_override = "sleeping"
    elif scores["phone"] >= 0.5:
        priority_override = "phone"
    elif scores["discuss"] >= 0.5:
        priority_override = "discuss"
    elif scores["walking"] >= 0.45:
        priority_override = "walking"

    if priority_override:
        best = priority_override
    else:
        best = max(scores, key=lambda k: scores[k])
    conf = min(scores[best], 1.0)
    if conf < 0.15:
        best = "sitting"   # default fallback
        conf = 0.3

    return ActionResult(
        action=best,
        confidence=round(conf, 3),
        raw_scores={k: round(v, 3) for k, v in scores.items()},
        source="rules",
    )


# --------------------------------------------------------------------------- #
#  Per-student temporal smoother                                                #
# --------------------------------------------------------------------------- #

class ActionSmoother:
    """Majority-vote smoother over a sliding window of action labels."""

    def __init__(self, window: int = SMOOTH_WINDOW):
        self.window = window
        self._history: deque = deque(maxlen=window)

    def update(self, result: ActionResult) -> ActionResult:
        self._history.append(result.action)
        if len(self._history) < 3:
            return result
        counts = defaultdict(int)
        for a in self._history:
            counts[a] += 1
        best = max(counts, key=lambda k: counts[k])
        smoothed_conf = counts[best] / len(self._history)
        return ActionResult(
            action=best,
            confidence=round(smoothed_conf, 3),
            raw_scores=result.raw_scores,
            source=result.source + "_smoothed",
            note=f"window={len(self._history)}",
        )

    def reset(self):
        self._history.clear()


# --------------------------------------------------------------------------- #
#  Main ActionClassifier class                                                  #
# --------------------------------------------------------------------------- #

class ActionClassifier:
    """
    Classifies student actions from pose keypoints.

    Usage:
        classifier = ActionClassifier()

        # Each frame, for each student:
        result = classifier.update(
            student_id="Aagam",
            track_id=3,
            frame_idx=120,
            keypoints=kps_array,          # (17, 3) with confidence
            bbox=(x1, y1, x2, y2),
            nearby_students=["Swayam"],   # students within proximity threshold
        )
        print(result.action, result.confidence)
    """

    def __init__(self, smooth_window: int = SMOOTH_WINDOW):
        self._smoothers: Dict[str, ActionSmoother] = {}    # keyed by student_id
        self._pos_history: Dict[str, deque] = {}           # keyed by student_id
        self._yolo_model: Optional[_YOLOActionModel] = None

        # Load optional YOLO action model
        if CLASSIFIER_MODEL_PATH and os.path.exists(CLASSIFIER_MODEL_PATH):
            try:
                self._yolo_model = _YOLOActionModel(CLASSIFIER_MODEL_PATH)
                logger.info("YOLO action model loaded from %s", CLASSIFIER_MODEL_PATH)
            except Exception as e:
                logger.warning("Could not load YOLO action model: %s", e)

        self._smooth_window = smooth_window

    def _get_smoother(self, student_id: str) -> ActionSmoother:
        if student_id not in self._smoothers:
            self._smoothers[student_id] = ActionSmoother(self._smooth_window)
        return self._smoothers[student_id]

    def _get_pos_history(self, student_id: str) -> deque:
        if student_id not in self._pos_history:
            self._pos_history[student_id] = deque(maxlen=30)
        return self._pos_history[student_id]

    def update(
        self,
        student_id: str,
        track_id: int,
        frame_idx: int,
        keypoints: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        nearby_students: Optional[List[str]] = None,
        crop: Optional[np.ndarray] = None,
    ) -> ActionResult:
        """
        Update action classification for a student this frame.

        Args:
            student_id:       Named identity (e.g. "Aagam") or "UNKNOWN"
            track_id:         ByteTrack track_id
            frame_idx:        Current frame index
            keypoints:        np.ndarray of shape (17, 2) or (17, 3); None if unavailable
            bbox:             (x1, y1, x2, y2) in pixels
            nearby_students:  List of other student names whose bbox centres are close
            crop:             Person crop image (used for YOLO action model if loaded)

        Returns:
            ActionResult with .action (string) and .confidence
        """
        x1, y1, x2, y2 = bbox
        bbox_h = max(y2 - y1, 1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        pos_hist = self._get_pos_history(student_id)
        pos_hist.append((cx, cy))

        # --- Try YOLO model first if available ---
        if self._yolo_model and crop is not None:
            yolo_pred = self._yolo_model.predict(crop)
            if yolo_pred is not None:
                yolo_action, yolo_conf = yolo_pred
                # Map SCB-dataset classes to our labels
                mapped = _map_scb_class(yolo_action)
                raw_result = ActionResult(
                    action=mapped,
                    confidence=float(yolo_conf),
                    source="yolo_model",
                    note=f"raw={yolo_action}",
                )
                smoother = self._get_smoother(student_id)
                return smoother.update(raw_result)

        # --- Rule-based fallback ---
        if keypoints is None:
            raw_result = ActionResult(action="sitting", confidence=0.3, source="default")
        else:
            kps = np.array(keypoints)
            raw_result = _classify_rules(
                kps=kps,
                bbox_h=float(bbox_h),
                nearby_students=nearby_students or [],
                position_history=pos_hist,
            )

        smoother = self._get_smoother(student_id)
        return smoother.update(raw_result)

    def reset_student(self, student_id: str):
        """Reset history for a student (e.g. after a long gap)."""
        if student_id in self._smoothers:
            self._smoothers[student_id].reset()
        if student_id in self._pos_history:
            self._pos_history[student_id].clear()


# --------------------------------------------------------------------------- #
#  SCB-Dataset class name mapper                                                #
# --------------------------------------------------------------------------- #

_SCB_MAP = {
    # Genuine matches only:
    "reading":           "reading",
    "study":             "reading",
    "studying":          "reading",
    "writing":           "writing",
    "typing":            "typing",
    "phone":             "phone",
    "using-phone":       "phone",
    "using_phone":       "phone",
    "phone use":         "phone",
    "sleeping":          "sleeping",
    "head-down":         "sleeping",
    "standing":          "standing",
    "walking":           "walking",
    "discuss":           "discuss",
    "discussion":        "discuss",
    "talking":           "discuss",
    "sitting":           "sitting",
    "looking-ahead":     "sitting",

    # Unwanted / ambiguous classes are DISCARDED (None):
    "hand-raising":      None,
    "hand raise":        None,
    "hand_raising":      None,
    "bowing-head":       None,
    "bowing_head":       None,
    "leaning-over-table":None,
    "leaning_over_table":None,
}

def _map_scb_class(raw: str) -> Optional[str]:
    key = raw.lower().strip()
    return _SCB_MAP.get(key, None)


# --------------------------------------------------------------------------- #
#  Proximity helper (for talking detection)                                     #
# --------------------------------------------------------------------------- #

def find_nearby_students(
    student_id: str,
    all_bboxes: Dict[str, Tuple[int, int, int, int]],
    proximity_px: float = 150.0,
) -> List[str]:
    """
    Return names of other students whose bbox centres are within proximity_px pixels.
    """
    if student_id not in all_bboxes:
        return []
    x1, y1, x2, y2 = all_bboxes[student_id]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nearby = []
    for other, bbox in all_bboxes.items():
        if other == student_id:
            continue
        ox1, oy1, ox2, oy2 = bbox
        ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
        d = float(np.sqrt((cx - ocx)**2 + (cy - ocy)**2))
        if d <= proximity_px:
            nearby.append(other)
    return nearby


# --------------------------------------------------------------------------- #
#  Action colour palette (BGR)                                                  #
# --------------------------------------------------------------------------- #

ACTION_COLORS_BGR = {
    "sitting":  (180, 180, 180),   # Gray
    "reading":  (200, 160, 40),    # Amber/Gold
    "writing":  (50,  200, 100),   # Green
    "typing":   (80,  180, 255),   # Cyan
    "phone":    (0,   60,  255),   # Red
    "sleeping": (120, 60,  200),   # Purple
    "standing": (0,   200, 220),   # Yellow
    "walking":  (0,   230, 160),   # Teal-Green
    "discuss":  (255, 165, 0),     # Orange
    "UNKNOWN":  (80,  80,  80),
}

ACTION_EMOJI = {
    "sitting":  "🪑",
    "reading":  "📖",
    "writing":  "✍️",
    "typing":   "⌨️",
    "phone":    "📱",
    "sleeping": "😴",
    "standing": "🧍",
    "walking":  "🚶",
    "discuss":  "💬",
    "UNKNOWN":  "❓",
}
