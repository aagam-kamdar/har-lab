"""
byte_tracker.py
---------------
A minimal, self-contained ByteTrack implementation suitable for a college IPD project.

ByteTrack paper: https://arxiv.org/abs/2110.06864

What this file does (plain English for viva):
  1. Every frame we receive a list of detections (bounding boxes + confidence).
  2. We split them into HIGH-confidence detections (conf >= high_thresh) and
     LOW-confidence detections (conf between low_thresh and high_thresh).
  3. We predict where each existing tracked person *would* be using a Kalman Filter
     (a mathematical prediction model – like estimating where a moving ball will be).
  4. We match HIGH-conf detections to existing tracks using IoU (Intersection over Union)
     – how much two boxes overlap.
  5. Unmatched tracks are given a second chance and matched against LOW-conf detections.
  6. Detections still unmatched after both rounds start new "tentative" tracks.
  7. A track becomes CONFIRMED after it survives `min_hits` frames.
  8. If a track is unmatched for `max_age` consecutive frames, it is removed.
  9. Each confirmed track has a unique integer track_id that stays stable as long
     as the person remains visible.

Limitations (important for viva):
  - ByteTrack uses only spatial position (IoU) for matching. It does NOT know what
    the person looks like (appearance). If two people cross paths, IDs may swap.
  - If a person leaves the frame and returns much later, they get a NEW track_id.
  - This is short-term tracking only. Re-ID (appearance embedding) would be needed
    for long-term stable identity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kalman Filter for a bounding-box track
# ---------------------------------------------------------------------------
# State vector: [cx, cy, s, r, vcx, vcy, vs]
#   cx, cy  = center x, center y
#   s       = area (width * height)
#   r       = aspect ratio (width / height) — treated as constant
#   vcx, vcy, vs = velocities
#
# We use a constant-velocity model.

class KalmanBoxTracker:
    """Tracks a single bounding box using a Kalman Filter."""

    _id_counter = 0  # class-level counter so IDs are globally unique

    def __init__(self, bbox: np.ndarray):
        """
        bbox: [x1, y1, x2, y2]
        """
        KalmanBoxTracker._id_counter += 1
        self.track_id: int = KalmanBoxTracker._id_counter

        # --- Kalman filter matrices (7-dimensional state) ---
        # State vector layout:
        #   index 0: cx   (center x)
        #   index 1: cy   (center y)
        #   index 2: s    (area = width * height)
        #   index 3: r    (aspect ratio = width / height) — CONSTANT, no velocity
        #   index 4: vcx  (velocity of cx)
        #   index 5: vcy  (velocity of cy)
        #   index 6: vs   (velocity of s)
        #
        # Constant-velocity model:
        #   cx_new  = cx  + vcx   → F[0, 4] = 1
        #   cy_new  = cy  + vcy   → F[1, 5] = 1
        #   s_new   = s   + vs    → F[2, 6] = 1
        #   r_new   = r           → no off-diagonal (aspect ratio is constant)
        #   vcx/vcy/vs unchanged  → already handled by np.eye(7)
        #
        # BUG that was here before (DO NOT restore):
        #   for i in range(4): self.F[i, i+3] = 1.0
        #   That set F[0,3]=1 (cx+=r), F[1,4]=1 (cy+=vcx),
        #              F[2,5]=1 (s+=vcy), F[3,6]=1 (r+=vs)  ← all wrong
        self.F = np.eye(7)
        self.F[0, 4] = 1.0   # cx  += vcx
        self.F[1, 5] = 1.0   # cy  += vcy
        self.F[2, 6] = 1.0   # s   += vs
        # index 3 (r) intentionally has no velocity term

        # Measurement matrix (we observe cx, cy, s, r)
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        # Measurement noise covariance
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])

        # Process noise covariance
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])

        # Initial state covariance
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0])

        # State: initialize from first detection
        cx, cy, s, r = _bbox_to_z(bbox)
        self.x = np.array([[cx], [cy], [s], [r], [0.0], [0.0], [0.0]])

        self.hits = 1           # how many frames this track was matched
        self.hit_streak = 1     # consecutive frames matched
        self.age = 0            # total frames since creation
        self.time_since_update = 0  # frames since last matched

    # --- Kalman predict step ---
    def predict(self) -> np.ndarray:
        """Advance state by one time step. Returns predicted [x1,y1,x2,y2]."""
        # Prevent negative area
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return _z_to_bbox(self.x[:4].flatten())

    # --- Kalman update step ---
    def update(self, bbox: np.ndarray):
        """Correct the state with a matched detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        z = np.array(_bbox_to_z(bbox)).reshape(4, 1)
        y = z - self.H @ self.x                         # innovation
        S = self.H @ self.P @ self.H.T + self.R         # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_bbox(self) -> np.ndarray:
        """Return current estimated [x1, y1, x2, y2]."""
        return _z_to_bbox(self.x[:4].flatten())


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _bbox_to_z(bbox: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert [x1,y1,x2,y2] → (cx, cy, area, aspect_ratio)."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return cx, cy, s, r


def _z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, area, aspect_ratio) → [x1, y1, x2, y2]."""
    cx, cy, s, r = z[0], z[1], z[2], z[3]
    w = np.sqrt(max(s * r, 0.0))
    h = s / (w + 1e-6)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2])


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes.
    bboxes_a: (N, 4)  bboxes_b: (M, 4)  →  returns (N, M)
    """
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))

    # Expand dims for broadcasting
    a = bboxes_a[:, np.newaxis, :]   # (N, 1, 4)
    b = bboxes_b[np.newaxis, :, :]   # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    iou = inter_area / (area_a + area_b - inter_area + 1e-6)
    return iou


# ---------------------------------------------------------------------------
# Linear assignment using scipy (wrapper around lapjv / Hungarian)
# ---------------------------------------------------------------------------

def _linear_assignment(cost_matrix: np.ndarray):
    """
    Solve the linear assignment problem.
    Returns array of shape (K, 2) with matched [row, col] pairs.
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.stack([row_ind, col_ind], axis=1)


# ---------------------------------------------------------------------------
# Association helper
# ---------------------------------------------------------------------------

def _associate(
    detections: np.ndarray,   # (N, 4) [x1,y1,x2,y2]
    trackers: np.ndarray,     # (M, 4) predicted boxes
    iou_threshold: float,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Match detections to trackers using IoU + Hungarian algorithm.

    Returns:
      matches:           (K, 2) array of [det_idx, trk_idx]
      unmatched_dets:    list of detection indices that were NOT matched
      unmatched_trks:    list of tracker indices that were NOT matched
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(len(detections))),
            [],
        )

    iou_matrix = _iou_batch(detections, trackers)  # (N, M)
    cost_matrix = 1.0 - iou_matrix                  # minimize cost = maximize IoU

    matched_indices = _linear_assignment(cost_matrix)

    unmatched_dets = [
        d for d in range(len(detections))
        if d not in matched_indices[:, 0]
    ]
    unmatched_trks = [
        t for t in range(len(trackers))
        if t not in matched_indices[:, 1]
    ]

    # Filter out matches with IoU below threshold
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m)

    matches = np.array(matches, dtype=int).reshape(-1, 2) if matches else np.empty((0, 2), dtype=int)
    return matches, unmatched_dets, unmatched_trks


# ---------------------------------------------------------------------------
# TrackedObject  –  what the rest of the code gets back
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    track_id: int
    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2)
    confidence: float
    class_name: str


# ---------------------------------------------------------------------------
# ByteTracker  –  the main public class
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    Minimal ByteTrack implementation.

    Parameters (viva-friendly explanation):
      high_thresh  – Only detections above this confidence are used in the FIRST
                     matching round. High-confidence means we're very sure it's a person.
      low_thresh   – Detections between low_thresh and high_thresh get a SECOND chance –
                     they are matched against tracks that didn't match anything in round 1.
                     This is the "Byte" idea: don't throw away low-confidence boxes.
      max_age      – How many consecutive frames a track can go unmatched before we
                     delete it. A value of 30 at 30 fps means ~1 second of patience.
      min_hits     – A new track must be matched for this many frames before we call
                     it "confirmed" and give it a public track_id. Avoids false positives
                     getting IDs immediately.
      iou_threshold – Minimum overlap required to consider two boxes as the same person.
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        max_age: int = 30,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[dict],   # each dict: {bbox:[x1,y1,x2,y2], conf:float, class_name:str}
    ) -> List[TrackedObject]:
        """
        Feed one frame's detections into the tracker.

        Returns a list of TrackedObject (one per CONFIRMED active track).
        If detections is empty, existing tracks are predicted forward;
        confirmed tracks that still have hits >= min_hits are returned
        (they'll disappear after max_age frames).
        """
        self.frame_count += 1

        # --- Step 1: Predict new locations for all existing tracks ---
        predicted_boxes = []
        for trk in self.trackers:
            predicted_boxes.append(trk.predict())

        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))

        # --- Step 2: Split detections by confidence ---
        high_dets = [d for d in detections if d["conf"] >= self.high_thresh]
        low_dets  = [d for d in detections if self.low_thresh <= d["conf"] < self.high_thresh]

        high_boxes = np.array([d["bbox"] for d in high_dets]) if high_dets else np.empty((0, 4))

        # --- Step 3: First association – high-confidence dets vs all tracks ---
        matches_h, unmatched_dets_h, unmatched_trks_h = _associate(
            high_boxes, predicted_boxes, self.iou_threshold
        )

        # Update matched tracks (high-confidence)
        for det_i, trk_i in matches_h:
            self.trackers[trk_i].update(high_boxes[det_i])
            self.trackers[trk_i].confidence = high_dets[det_i]["conf"]
            self.trackers[trk_i].class_name = high_dets[det_i]["class_name"]

        # --- Step 4: Second association – low-confidence dets vs unmatched tracks ---
        if low_dets and unmatched_trks_h:
            low_boxes = np.array([d["bbox"] for d in low_dets])
            unmatched_predicted = predicted_boxes[unmatched_trks_h]

            matches_l, _, still_unmatched_trks = _associate(
                low_boxes, unmatched_predicted, self.iou_threshold
            )

            for det_i, local_trk_i in matches_l:
                global_trk_i = unmatched_trks_h[local_trk_i]
                self.trackers[global_trk_i].update(low_boxes[det_i])
                self.trackers[global_trk_i].confidence = low_dets[det_i]["conf"]
                self.trackers[global_trk_i].class_name = low_dets[det_i]["class_name"]

            # Tracks truly unmatched in both rounds reset hit_streak
            for local_trk_i in still_unmatched_trks:
                global_trk_i = unmatched_trks_h[local_trk_i]
                self.trackers[global_trk_i].hit_streak = 0
        else:
            # No low-conf dets; all unmatched tracks reset hit_streak
            for trk_i in unmatched_trks_h:
                self.trackers[trk_i].hit_streak = 0

        # --- Step 5: Create new tracks for unmatched HIGH-conf detections ---
        for det_i in unmatched_dets_h:
            new_trk = KalmanBoxTracker(high_boxes[det_i])
            new_trk.confidence = high_dets[det_i]["conf"]   # type: ignore[attr-defined]
            new_trk.class_name = high_dets[det_i]["class_name"]  # type: ignore[attr-defined]
            self.trackers.append(new_trk)

        # --- Step 6: Remove dead tracks ---
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        # --- Step 7: Return confirmed active tracks ---
        # A track is returned when:
        #   1. It was matched in THIS frame (time_since_update == 0), AND
        #   2. It has been matched for at least min_hits frames (hits >= min_hits).
        # This prevents false-positive detections from immediately getting an ID.
        active: List[TrackedObject] = []
        for trk in self.trackers:
            if trk.time_since_update == 0 and trk.hits >= self.min_hits:
                bbox = trk.get_bbox()
                obj = TrackedObject(
                    track_id=trk.track_id,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=getattr(trk, "confidence", 0.0),
                    class_name=getattr(trk, "class_name", "person"),
                )
                active.append(obj)

        return active

    def reset(self):
        """Reset all tracks and the ID counter (useful between videos)."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker._id_counter = 0
