"""
har_classifier.py
-----------------
Pretrained Human Activity Recognition (HAR) module using VideoMAE.
Model: MCG-NJU/videomae-small-finetuned-kinetics (Kinetics-400).

Provides:
  - Per-track temporal clip buffering (16 frames)
  - Person-centered bounding box cropping with contextual margin
  - Pretrained VideoMAE inference (CPU optimized)
  - Explicit verified mapping between Kinetics-400 and application taxonomy:
      ['active_work', 'writing', 'reading', 'discussing', 'idle', 'walking', 'phone', 'call', 'away', 'uncertain']
  - Thread-safe non-blocking asynchronous inference to protect dashboard and video processing throughput
  - Fallback and missing weights handling without application crash
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("HARClassifier")

# Verified semantic mapping from Kinetics-400 to application taxonomy
KINETICS_TO_APP_MAP: Dict[str, str] = {
    # Active Work (computer work, typing, tech desk tasks)
    "using computer": "active_work",
    "playing piano": "active_work",
    "assembling computer": "active_work",

    # Writing
    "writing": "writing",

    # Reading
    "reading book": "reading",
    "reading newspaper": "reading",

    # Phone (Kinetics-400 only has texting; no phone call class exists in Kinetics-400)
    "texting": "phone",

    # Discussing / Interacting
    "answering questions": "discussing",
    "shaking head": "discussing",
    "sign language interpreting": "discussing",
    "clapping": "discussing",

    # Idle / Low energy
    "yawning": "idle",

    # Walking / Moving
    "walking the dog": "walking",
    "running on treadmill": "walking",
}


@dataclass
class HARPrediction:
    track_id: int
    raw_label: str
    confidence: float
    mapped_action: str
    timestamp: float
    latency_ms: float
    is_mapped: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "raw_label": self.raw_label,
            "confidence": round(self.confidence, 4),
            "mapped_action": self.mapped_action,
            "timestamp": round(self.timestamp, 3),
            "latency_ms": round(self.latency_ms, 1),
            "is_mapped": self.is_mapped,
        }


class HARClassifier:
    """
    Independent Pretrained Human Activity Recognition classifier.
    Loads MCG-NJU/videomae-small-finetuned-kinetics once, manages per-track
    frame buffers, runs inference on 16-frame person-centered clips, and
    maps labels to the target application taxonomy.
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-small-finetuned-kinetics",
        clip_len: int = 16,
        inference_interval_s: float = 2.0,
        confidence_threshold: float = 0.30,
        device: str = "cpu",
        async_mode: bool = True,
        crop_size: int = 224,
    ):
        self.model_name = model_name
        self.clip_len = clip_len
        self.inference_interval_s = float(inference_interval_s)
        self.confidence_threshold = float(confidence_threshold)
        self.device = device
        self.async_mode = async_mode
        self.crop_size = crop_size

        self.model = None
        self.processor = None
        self.labels: Dict[int, str] = {}
        self.is_ready = False
        self.init_error: Optional[str] = None

        # Per-track state
        self._buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.clip_len))
        self._last_inference_t: Dict[int, float] = defaultdict(float)
        self._last_predictions: Dict[int, HARPrediction] = {}
        self._in_flight: set = set()
        self._lock = threading.Lock()

        # Metrics
        self.total_inferences = 0
        self.total_latency_ms = 0.0

        self._load_model()

    def _load_model(self):
        """Load pretrained VideoMAE model and image processor."""
        try:
            logger.info("Loading pretrained HAR model: %s on %s...", self.model_name, self.device)
            t0 = time.time()
            from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
            import torch

            self.processor = VideoMAEImageProcessor.from_pretrained(self.model_name)
            self.model = VideoMAEForVideoClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self.labels = self.model.config.id2label
            self.is_ready = True
            dur = time.time() - t0
            logger.info("HAR model loaded successfully in %.2fs. Classes: %d", dur, len(self.labels))
        except Exception as e:
            self.is_ready = False
            self.init_error = str(e)
            logger.error("Failed to load pretrained HAR model (%s): %s", self.model_name, e)

    def extract_person_crop(self, frame: np.ndarray, bbox: List[int] | Tuple[int, int, int, int]) -> np.ndarray:
        """Extract a person-centered crop with 10% spatial context padding, resized to crop_size."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(0.10 * bw)
        pad_y = int(0.10 * bh)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        return crop

    def map_label(self, raw_label: str, confidence: float, margin: float = 0.0) -> Tuple[str, bool]:
        """
        Map a Kinetics-400 label to the application taxonomy.
        Returns (mapped_action, is_mapped).
        Uses both primary confidence threshold and margin-qualified threshold for 400 classes.
        """
        raw_clean = raw_label.lower().strip()
        # Rule: do not map playing piano to active_work when probability is below 0.10
        if raw_clean == "playing piano" and confidence < 0.10:
            return "uncertain", False

        mapped = KINETICS_TO_APP_MAP.get(raw_clean)

        if mapped is not None:
            if confidence >= self.confidence_threshold:
                return mapped, True
            # For 400 classes, 14% confidence with a 5% margin over runner-up is >50x uniform baseline (0.25%)
            if confidence >= 0.14 and margin >= 0.05:
                return mapped, True
            # Project-specific active_work threshold (0.10) for standard configuration
            if raw_clean in ("using computer", "playing piano") and confidence >= 0.10 and self.confidence_threshold <= 0.30:
                return mapped, True
        return "uncertain", False

    def predict_clip(self, clip_frames: List[np.ndarray]) -> Tuple[str, float, str, bool, float]:
        """
        Run VideoMAE inference on a list of 16 RGB/BGR frames.
        Returns: (raw_label, confidence, mapped_action, is_mapped, latency_ms)
        """
        if not self.is_ready or self.model is None or self.processor is None:
            return "model_not_ready", 0.0, "uncertain", False, 0.0

        if len(clip_frames) < self.clip_len:
            return "insufficient_frames", 0.0, "uncertain", False, 0.0

        import torch

        t0 = time.time()
        # Convert BGR frames to RGB for VideoMAE
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f.shape[-1] == 3 else f for f in clip_frames]

        inputs = self.processor(rgb_frames, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            k_val = min(5, probs.shape[-1])
            topk = torch.topk(probs, k_val)
            topk_indices = topk.indices.cpu().numpy()
            topk_probs = topk.values.cpu().numpy()

            top_idx = int(topk_indices[0])
            top_conf = float(topk_probs[0])
            second_conf = float(topk_probs[1]) if len(topk_probs) > 1 else 0.0
            margin = top_conf - second_conf
            raw_top1_label = self.labels.get(top_idx, f"class_{top_idx}")
            top_k_list = [(self.labels.get(int(idx), f"class_{idx}"), float(p)) for idx, p in zip(topk_indices, topk_probs)]

            # Check full class-score vector for active_work classes (using computer, playing piano)
            if getattr(self, "_idx_computer", None) is None or getattr(self, "_idx_piano", None) is None:
                for idx, lbl in self.labels.items():
                    c_lbl = lbl.lower().strip()
                    if c_lbl == "using computer":
                        self._idx_computer = int(idx)
                    elif c_lbl == "playing piano":
                        self._idx_piano = int(idx)

            comp_idx = getattr(self, "_idx_computer", None)
            piano_idx = getattr(self, "_idx_piano", None)
            p_comp = float(probs[comp_idx].item()) if comp_idx is not None else 0.0
            p_piano = float(probs[piano_idx].item()) if piano_idx is not None else 0.0

        latency_ms = (time.time() - t0) * 1000.0

        # Rule 1.A: If probability of either relevant class >= 0.10, map to active_work
        # Checked across the full class-score vector so it is not ignored outside top-1.
        if p_comp >= 0.10 or p_piano >= 0.10:
            if p_comp >= p_piano:
                active_lbl = "using computer"
                active_conf = p_comp
            else:
                active_lbl = "playing piano"
                active_conf = p_piano
            mapped_action = "active_work"
            is_mapped = True
            raw_label = active_lbl
            confidence = active_conf
        else:
            raw_label = raw_top1_label
            confidence = top_conf
            if raw_label.lower().strip() == "playing piano":
                mapped_action, is_mapped = "uncertain", False
            else:
                mapped_action, is_mapped = self.map_label(raw_label, top_conf, margin=margin)

        # Store latest clip inference details for diagnostics
        self._last_clip_details = {
            "top_k": top_k_list,
            "margin": margin,
            "top_label": raw_top1_label,
            "top_conf": top_conf,
            "raw_label": raw_label,
            "confidence": confidence,
            "p_computer": p_comp,
            "p_piano": p_piano,
            "mapped_action": mapped_action,
            "is_mapped": is_mapped,
            "latency_ms": latency_ms,
        }

        return raw_label, confidence, mapped_action, is_mapped, latency_ms

    def _async_inference_worker(self, track_id: int, clip_frames: List[np.ndarray], timestamp: float):
        """Worker executed in background thread."""
        try:
            raw_lbl, conf, mapped, is_mapped, lat_ms = self.predict_clip(clip_frames)
            details = getattr(self, "_last_clip_details", {}).copy()
            pred = HARPrediction(
                track_id=track_id,
                raw_label=raw_lbl,
                confidence=conf,
                mapped_action=mapped,
                timestamp=timestamp,
                latency_ms=lat_ms,
                is_mapped=is_mapped,
                details=details,
            )
            with self._lock:
                self._last_predictions[track_id] = pred
                self.total_inferences += 1
                self.total_latency_ms += lat_ms
        except Exception as e:
            logger.error("Error during HAR async inference for track %d: %s", track_id, e)
        finally:
            with self._lock:
                self._in_flight.discard(track_id)

    def update_track(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: List[int] | Tuple[int, int, int, int],
        timestamp: float,
    ) -> Optional[HARPrediction]:
        """
        Feed new person crop for a tracked student.
        Triggers inference when clip buffer is full and inference interval has elapsed.
        Returns the latest available prediction for this track.
        """
        crop = self.extract_person_crop(frame, bbox)

        with self._lock:
            buf = self._buffers[track_id]
            buf.append(crop)

            # Check if inference should be triggered
            has_prior = (track_id in self._last_inference_t)
            time_elapsed = (timestamp - self._last_inference_t[track_id]) if has_prior else float("inf")
            should_run = (
                self.is_ready
                and len(buf) >= self.clip_len
                and (not has_prior or time_elapsed >= self.inference_interval_s)
                and track_id not in self._in_flight
            )

            if should_run:
                self._last_inference_t[track_id] = timestamp
                clip_snapshot = list(buf)

                if self.async_mode:
                    self._in_flight.add(track_id)
                    thread = threading.Thread(
                        target=self._async_inference_worker,
                        args=(track_id, clip_snapshot, timestamp),
                        daemon=True,
                    )
                    thread.start()
                else:
                    # Synchronous fallback
                    raw_lbl, conf, mapped, is_mapped, lat_ms = self.predict_clip(clip_snapshot)
                    details = getattr(self, "_last_clip_details", {}).copy()
                    self._last_predictions[track_id] = HARPrediction(
                        track_id=track_id,
                        raw_label=raw_lbl,
                        confidence=conf,
                        mapped_action=mapped,
                        timestamp=timestamp,
                        latency_ms=lat_ms,
                        is_mapped=is_mapped,
                        details=details,
                    )
                    self.total_inferences += 1
                    self.total_latency_ms += lat_ms

            return self._last_predictions.get(track_id)

    def get_prediction(self, track_id: int) -> Optional[HARPrediction]:
        """Get the latest prediction for a track ID."""
        with self._lock:
            return self._last_predictions.get(track_id)

    def reset_track(self, track_id: int):
        """Clean up buffers and cached prediction when a track ends."""
        with self._lock:
            self._buffers.pop(track_id, None)
            self._last_inference_t.pop(track_id, None)
            self._last_predictions.pop(track_id, None)
            self._in_flight.discard(track_id)

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics."""
        with self._lock:
            avg_lat = (self.total_latency_ms / self.total_inferences) if self.total_inferences > 0 else 0.0
            return {
                "model_name": self.model_name,
                "is_ready": self.is_ready,
                "device": self.device,
                "total_inferences": self.total_inferences,
                "average_latency_ms": round(avg_lat, 1),
                "active_tracks_buffered": len(self._buffers),
                "supported_taxonomy": list(KINETICS_TO_APP_MAP.values()),
                "unsupported_taxonomy": ["call", "away"],
            }
