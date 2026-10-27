"""
enrolled_tracking_pipeline.py
------------------------------
PHASE B + C: Full Lab Video Tracking with Named Enrolled Students

Pipeline:
  1. Load enrolled gallery (Aagam / Swayam / Harshil / Soham) from reid/data/.
  2. Run YOLO person detection on every frame.
  3. Feed detections to ByteTrack for short-term tracking.
  4. For each track: run body Re-ID + optional face matching -> NamedIdentityResolver.
  5. Persistent student_id (named) remains stable across ByteTrack ID changes.
  6. Adaptive gallery updates when high-confidence observations are encountered.
  7. Output annotated MP4 with: "Track 4 | Aagam" banners.
  8. Print full evaluation report at the end.

USAGE:
    python enrolled_tracking_pipeline.py

NOTES:
  - Run enrollment_runner.py first if reid/data/ does not exist.
  - The original video_processor.py + run_reid_pipeline_test.py remain UNTOUCHED.
  - This script uses the new named enrollment system in parallel.
"""

from __future__ import annotations

import os
import sys
import cv2
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from ultralytics import YOLO
from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack, STrack, TrackState

from reid.embedding_model import PersonReIDExtractor
from reid.face_embedder import FaceEmbedder
from reid.enrolled_gallery import EnrolledGallery
from reid.named_identity_resolver import NamedIdentityResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enrolled_tracker")


# ========== CONFIGURATION ==========

VIDEO_PATH = "new lab video.mp4"
YOLO_MODEL = "yolo26n.pt"
GALLERY_DATA_DIR = os.path.join("reid", "data")

OUTPUT_DIR = os.path.join("static", "results", "enrolled_tracking")
OUTPUT_FILENAME = "enrolled_output_tracked.mp4"

CONF_THRESH = 0.30
TRACK_THRESH = 0.50
TRACK_BUFFER = 60          # Longer buffer to survive occlusion at desks
MATCH_THRESH = 0.80

SIMILARITY_THRESH = 0.72   # Body Re-ID cosine similarity threshold (calibrated for seated desk poses)
AMBIGUITY_MARGIN = 0.020   # Top-1 vs top-2 distinction margin
MIN_OBSERVATIONS = 5       # Frames before confirming identity via temporal consensus

ADAPTIVE_UPDATE = True
ADAPTIVE_CONF_THRESH = 0.90

# Named student color palette (BGR)  Aagam=Teal, Swayam=Pink, Harshil=Indigo, Soham=Amber, UNKNOWN=Gray
STUDENT_COLORS = {
    "Aagam":   (212, 234, 94),    # Teal-ish
    "Swayam":  (182, 114, 244),   # Pink
    "Harshil": (248, 140, 129),   # Indigo/Blue
    "Soham":   (36, 191, 251),    # Amber/Gold
    "UNKNOWN": (128, 128, 128),   # Gray
}
DEFAULT_TRACK_COLOR = (200, 200, 200)
TEXT_COLOR = (10, 10, 10)


# ---------------------------------------------------------------------------
# Video Writer
# ---------------------------------------------------------------------------

class VideoWriter:
    """H.264 PyAV writer with cv2 fallback."""

    def __init__(self, path: str, fps: float, w: int, h: int):
        self.path = path
        self.mode = "cv2"
        self.container = None
        self.stream = None
        self.cv2_writer = None
        try:
            import av
            self.container = av.open(path, mode="w")
            self.stream = self.container.add_stream("h264", rate=int(round(fps)))
            self.stream.width = w
            self.stream.height = h
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"preset": "veryfast", "crf": "22"}
            self.mode = "av"
            logger.info("VideoWriter: H.264/PyAV initialized")
        except Exception as e:
            logger.warning("PyAV unavailable (%s). Using cv2.", e)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.cv2_writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    def write(self, frame_bgr: np.ndarray):
        if self.mode == "av":
            import av
            vf = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
            for pkt in self.stream.encode(vf):
                self.container.mux(pkt)
        elif self.cv2_writer:
            self.cv2_writer.write(frame_bgr)

    def release(self):
        if self.mode == "av" and self.container:
            for pkt in self.stream.encode():
                self.container.mux(pkt)
            self.container.close()
        elif self.cv2_writer:
            self.cv2_writer.release()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_tracks(
    frame: np.ndarray,
    tracks: List[STrack],
    resolver: NamedIdentityResolver,
    frame_idx: int,
    phase_label: str = "",
) -> np.ndarray:
    """Draw bounding boxes and identity labels on frame."""
    h_fr, w_fr = frame.shape[:2]

    for trk in tracks:
        x1, y1, x2, y2 = [int(v) for v in trk.tlbr]
        x1 = max(0, min(x1, w_fr - 1))
        y1 = max(0, min(y1, h_fr - 1))
        x2 = max(0, min(x2, w_fr - 1))
        y2 = max(0, min(y2, h_fr - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        student_name = resolver.get_student_name(trk.track_id)

        if student_name:
            color = STUDENT_COLORS.get(student_name, DEFAULT_TRACK_COLOR)
            if student_name == "UNKNOWN":
                label = f"Track {trk.track_id} | UNKNOWN"
            else:
                label = f"Track {trk.track_id} | {student_name}"
        else:
            color = DEFAULT_TRACK_COLOR
            label = f"Track {trk.track_id} [?]"

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Label banner
        fscale = max(0.45, min(0.70, (y2 - y1) / 140.0))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, 2)
        ly = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, ly), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale, TEXT_COLOR, 2, cv2.LINE_AA)

    # Phase label overlay
    if phase_label:
        cv2.putText(frame, phase_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Frame counter
    cv2.putText(frame, f"Frame {frame_idx}", (10, h_fr - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def determine_phase(frame_idx: int, fps: float) -> str:
    """Label video phases for overlay."""
    t = frame_idx / fps
    if t < 45:
        return "PHASE A: Enrollment Segment"
    elif t < 90:
        return "PHASE B: Desk Transition"
    else:
        return "PHASE C: Desk Tracking"


# ---------------------------------------------------------------------------
# Main Tracking Pipeline
# ---------------------------------------------------------------------------

def run_enrolled_tracking():
    """Run the full named-student tracking pipeline."""
    print("=" * 72)
    print("ENROLLED STUDENT TRACKING PIPELINE")
    print("ByteTrack + Named Re-ID (Aagam / Swayam / Harshil / Soham)")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # Load enrolled gallery
    # ----------------------------------------------------------------
    print("\n[1/5] Loading enrolled gallery...")
    gallery = EnrolledGallery(data_dir=GALLERY_DATA_DIR)

    if gallery.total_body_embeddings() == 0:
        print("ERROR: No enrolled students found. Run enrollment_runner.py first.")
        sys.exit(1)

    status = gallery.enrollment_status()
    for name, count in status.items():
        print(f"  {name}: {count} body embeddings")
    print(f"  Total FAISS body vectors: {gallery.total_body_embeddings()}")

    # ----------------------------------------------------------------
    # Load models
    # ----------------------------------------------------------------
    print("\n[2/5] Loading models...")
    body_extractor = PersonReIDExtractor()
    face_embedder = FaceEmbedder()
    yolo = YOLO(YOLO_MODEL)
    print("  Models loaded.")

    # ----------------------------------------------------------------
    # Open video
    # ----------------------------------------------------------------
    print("\n[3/5] Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_PATH}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {width}x{height} @ {fps:.1f} fps, ~{total_frames_est} frames")

    # ----------------------------------------------------------------
    # Initialize ByteTrack + NamedIdentityResolver
    # ----------------------------------------------------------------
    BaseTrack.reset_id()
    tracker_args = BYTETrackerArgs(
        track_thresh=TRACK_THRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH,
        mot20=False,
    )
    tracker = BYTETracker(args=tracker_args, frame_rate=int(round(fps)))

    resolver = NamedIdentityResolver(
        gallery=gallery,
        body_extractor=body_extractor,
        face_embedder=face_embedder,
        similarity_thresh=SIMILARITY_THRESH,
        ambiguity_margin=AMBIGUITY_MARGIN,
        min_observations=MIN_OBSERVATIONS,
        fps=fps,
        adaptive_update=ADAPTIVE_UPDATE,
        adaptive_confidence_thresh=ADAPTIVE_CONF_THRESH,
    )

    # ----------------------------------------------------------------
    # Video writer
    # ----------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    writer = VideoWriter(out_path, fps, width, height)

    # ----------------------------------------------------------------
    # Main tracking loop
    # ----------------------------------------------------------------
    print("\n[4/5] Running tracking loop...")
    frame_idx = 0
    total_detections = 0
    unique_track_ids: Set[int] = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # YOLO detection
            yolo_results = yolo(frame, conf=CONF_THRESH, classes=[0], verbose=False)[0]
            n_dets = len(yolo_results.boxes)
            total_detections += n_dets

            if n_dets > 0:
                xyxy = yolo_results.boxes.xyxy.cpu().numpy()
                confs = yolo_results.boxes.conf.cpu().numpy()[:, None]
                detections = np.hstack([xyxy, confs])
            else:
                detections = np.empty((0, 5), dtype=float)

            before_lost = {t.track_id for t in tracker.lost_stracks}

            # ByteTrack update
            online_targets: List[STrack] = tracker.update(
                detections,
                img_info=(height, width),
                img_size=(height, width),
            )

            # Record short-term reactivations
            after_tracked = {t.track_id for t in tracker.tracked_stracks}
            reactivated = after_tracked & before_lost
            for tid in reactivated:
                resolver.total_reid_attempts  # access to ensure initialized

            # Identity resolution for each active track
            for trk in online_targets:
                tid = trk.track_id
                unique_track_ids.add(tid)

                if resolver.get_student_name(tid) is None:
                    x1, y1, x2, y2 = [int(v) for v in trk.tlbr]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    if x2 > x1 + 10 and y2 > y1 + 10:
                        body_crop = frame[y1:y2, x1:x2]
                        # Extract upper-body region for face fallback
                        h_box = y2 - y1
                        upper_crop = frame[y1:y1 + max(1, int(h_box * 0.45)), x1:x2]

                        resolver.resolve_track(
                            track_id=tid,
                            crop=body_crop,
                            frame_idx=frame_idx,
                            face_crop=upper_crop,
                        )

            # Draw annotations
            phase = determine_phase(frame_idx, fps)
            annotated = draw_tracks(frame, online_targets, resolver, frame_idx, phase)
            writer.write(annotated)

            # Progress log
            if frame_idx % 150 == 0 or frame_idx == 1:
                active_str = [
                    f"T{t.track_id}:{resolver.get_student_name(t.track_id) or '?'}"
                    for t in online_targets
                ]
                pct = frame_idx / max(1, total_frames_est) * 100
                logger.info(
                    "Frame %4d/%d (%.0f%%) | dets=%d | active: %s",
                    frame_idx, total_frames_est, pct, n_dets,
                    ", ".join(active_str)
                )

    finally:
        cap.release()
        writer.release()
        resolver.finalize()
        gallery.save()

    # ----------------------------------------------------------------
    # Evaluation Report
    # ----------------------------------------------------------------
    print(f"\n[5/5] Generating evaluation report...")
    _print_report(
        frame_idx, total_detections, unique_track_ids,
        resolver, gallery, out_path, fps
    )


def _print_report(
    total_frames: int,
    total_detections: int,
    unique_track_ids: Set[int],
    resolver: NamedIdentityResolver,
    gallery: EnrolledGallery,
    out_path: str,
    fps: float,
):
    """Print the full evaluation report."""
    m = resolver.get_metrics()
    SEP = "=" * 72
    SEP2 = "-" * 72

    print("\n" + SEP)
    print("ENROLLED STUDENT TRACKING — EVALUATION REPORT")
    print(SEP)
    print(f"  Total Frames Processed          : {total_frames}")
    print(f"  Video Duration                  : {total_frames / fps:.1f} sec")
    print(f"  Total YOLO Person Detections    : {total_detections}")
    print(f"  Avg Detections / Frame          : {total_detections / max(1, total_frames):.2f}")
    print()
    print(f"  Temporary ByteTrack IDs         : {len(unique_track_ids)}")
    print(f"  Enrolled Students Tracked       : {', '.join(m['enrolled_students_tracked']) or 'None'}")
    print()
    print(f"  Re-ID Attempts                  : {m['total_reid_attempts']}")
    print(f"  Body Re-ID Matches              : {m['body_reid_matches']}")
    print(f"  Face-Assisted Matches           : {m['face_assisted_matches']}")
    print(f"  Total Successful Matches        : {m['total_successful_matches']}")
    print()
    print(f"  UNKNOWN Periods                 : {m['unknown_periods']}")
    print(f"  Identity Recoveries             : {m['recoveries']}")
    print(f"  Identity Switches (errors)      : {m['identity_switches']}")
    print(f"  Adaptive Gallery Updates        : {m['gallery_updates']}")
    print()
    print(f"  Similarity — Min / Avg / Max    : {m['min_similarity']:.3f} / {m['avg_similarity']:.3f} / {m['max_similarity']:.3f}")
    print()

    # Gallery state at end
    print(SEP2)
    print("FINAL GALLERY STATE:")
    print(SEP2)
    gs = gallery.enrollment_status()
    for name, cnt in gs.items():
        bar = "█" * min(cnt, 20)
        print(f"  {name:10s}: {bar} {cnt} body embeddings")

    # Recovery log
    if resolver.recovery_log:
        print()
        print(SEP2)
        print("IDENTITY RECOVERY LOG:")
        print(SEP2)
        print(f"  {'Student':<10} {'Old Track':>9} {'New Track':>9} {'Gap Frames':>11} {'Gap Sec':>9} {'Score':>7}")
        print("  " + "-" * 60)
        for ev in resolver.recovery_log:
            print(
                f"  {ev['student']:<10} {ev['old_track']:>9d} {ev['new_track']:>9d} "
                f"{ev['gap_frames']:>11d} {ev['gap_sec']:>9.2f}s {ev['score']:>7.3f}"
            )
    else:
        print(f"\n  No recoveries logged (all tracks may have stayed continuous).")

    # Switch log
    if resolver.switch_log:
        print()
        print(SEP2)
        print("IDENTITY SWITCH LOG (potential errors):")
        print(SEP2)
        for ev in resolver.switch_log:
            print(f"  Frame {ev['frame']:5d}: Track {ev['track_id']} {ev['from']} -> {ev['to']}")

    # Track -> Student assignment table
    print()
    print(SEP2)
    print("TRACK -> STUDENT ASSIGNMENTS:")
    print(SEP2)
    # Group by student
    student_tracks: Dict[str, List[int]] = defaultdict(list)
    for tid, sname in sorted(resolver.track_to_student.items()):
        student_tracks[sname].append(tid)
    for sname, tids in sorted(student_tracks.items()):
        print(f"  {sname:10s}: Tracks {tids}")

    print()
    print(SEP)
    out_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print(f"  OUTPUT VIDEO : {out_path}")
    print(f"  File Size    : {out_size:,} bytes ({out_size / 1e6:.1f} MB)")
    print(SEP)


if __name__ == "__main__":
    run_enrolled_tracking()
