"""
robust_tracker.py
-----------------
Robust Identity-Locking Student Tracker.

Pipeline:
  new lab video.mp4
      ↓
  YOLO person-only detection
      ↓
  BYTETracker (official)
      ↓
  IdentityLockManager
      - Face match (ArcFace) → LOCK identity
      - Body Re-ID (FAISS) → supporting evidence only
      - Gap recovery after long absence
      ↓
  Annotated MP4 output + JSON report

Usage:
  python robust_tracker.py [--max-frames N] [--output-dir DIR]
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import warnings
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack, STrack, TrackState as BTTrackState
from identity_lock_manager import IdentityLockManager, FACE_FRAME_GAP_FOR_RECOVERY
from features import FaceFeatureExtractor, BodyFeatureExtractor, OSNetBodyFeatureExtractor, BodyGallery
from enroll_students import enroll_students, KNOWN_STUDENTS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Visual constants                                                    #
# ------------------------------------------------------------------ #
STUDENT_COLORS = {
    "aagam":   (0, 215, 255),
    "swayam":  (255, 105, 180),
    "harshil": (50, 205, 50),
    "soham":   (255, 140, 0),
    "UNKNOWN": (130, 130, 130),
}


def get_color(name: str) -> Tuple[int, int, int]:
    key = name.lower() if name != "UNKNOWN" else "UNKNOWN"
    return STUDENT_COLORS.get(key, (200, 200, 200))


# ------------------------------------------------------------------ #
#  H.264 Video Writer                                                  #
# ------------------------------------------------------------------ #

class VideoWriter:
    def __init__(self, path: str, fps: float, width: int, height: int):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.mode = "cv2"
        self.writer = None

        try:
            import av
            self.mode = "av"
            self.container = av.open(path, mode="w")
            self.stream = self.container.add_stream("h264", rate=int(round(fps)))
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"preset": "veryfast", "crf": "22"}
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        if self.mode == "av":
            import av
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in self.stream.encode(av_frame):
                self.container.mux(pkt)
        else:
            self.writer.write(frame)

    def close(self):
        if self.mode == "av":
            for pkt in self.stream.encode():
                self.container.mux(pkt)
            self.container.close()
        elif self.writer:
            self.writer.release()


# ------------------------------------------------------------------ #
#  Annotation helper                                                   #
# ------------------------------------------------------------------ #

def draw_track(frame: np.ndarray, bbox: Tuple, track_id: int, label: str) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = get_color(label)
    display = f"ID{track_id} | {label.title()}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    (tw, th), baseline = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    bg_y1 = max(0, y1 - th - baseline - 4)
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, display,
        (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
    )
    return frame


# ------------------------------------------------------------------ #
#  Main Tracker                                                        #
# ------------------------------------------------------------------ #

class RobustTracker:
    """
    Connects BYTETracker → IdentityLockManager → FaceExtractor → BodyGallery.
    Implements the strict 11-rule locking policy.
    """

def calculate_escalation_frames(fps: float) -> int:
    """
    Calculates the number of frames corresponding to 1.0 real second at the given FPS.
    - 30.0 FPS -> 30 frames
    - 12.0 FPS -> 12 frames
    """
    effective_fps = float(fps) if fps and fps > 0 else 30.0
    return max(1, round(1.0 * effective_fps))


class RobustTracker:
    """
    Pipeline orchestrator integrating YOLOv8 + BYTETracker + ArcFace + OSNet Re-ID + IdentityLockManager.
    """

    def __init__(
        self,
        face_profiles: Dict[str, np.ndarray],
        body_gallery: BodyGallery,
        yolo_model,
        face_extractor: FaceFeatureExtractor,
        body_extractor: BodyFeatureExtractor,
        fps: float = 30.0,
    ):
        self.face_profiles = face_profiles
        self.body_gallery = body_gallery
        self.yolo = yolo_model
        self.face_extractor = face_extractor
        self.body_extractor = body_extractor
        self.lock_manager = IdentityLockManager(require_dual_agreement=True)
        self.set_fps(fps)

        # Per-track last-known embedding for gap recovery
        self._last_body_emb: Dict[int, np.ndarray] = {}
        self._track_last_active: Dict[int, int] = {}

        # Escalation & throttling state per track
        self._last_escalation_frame: Dict[int, int] = {}
        self._last_locked_face_check: Dict[int, int] = {}
        self._last_locked_body_emb_frame: Dict[int, int] = {}

        # Switch/blocked-switch counters for stats deduplication
        self._prev_switches: Dict[int, int] = {}
        self._prev_blocked: Dict[int, int] = {}

        # Stats
        self.stats = defaultdict(int)
        self.frame_log: List[Dict] = []

    def set_fps(self, fps: float):
        """Set runtime video FPS and update time-based frame thresholds."""
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.unknown_escalation_frames = calculate_escalation_frames(self.fps)
        self.locked_face_check_interval = max(1, round(1.0 * self.fps))

    def should_escalate_to_640(self, tid: int, frame_idx: int, first_seen_frame: int, is_locked: bool) -> bool:
        """
        Option B Persistence:
        - UNKNOWN for 1.0 real second -> use SCRFD det_size=(640,640) once on next face check.
        - Return to 320 immediately.
        - Re-escalate only after another full 1.0 real second of continued UNKNOWN.
        - Locked tracks never escalate to 640 (always return to 320).
        """
        if is_locked:
            return False

        frames_unknown = frame_idx - first_seen_frame
        if frames_unknown < self.unknown_escalation_frames:
            return False

        last_esc = self._last_escalation_frame.get(tid, -99999)
        if (frame_idx - last_esc) >= self.unknown_escalation_frames:
            self._last_escalation_frame[tid] = frame_idx
            return True

        return False

    def should_check_locked_face(self, tid: int, frame_idx: int) -> bool:
        """
        Compute reduction Part B1:
        Throttles locked-track contradiction face checks to ~1.0 real second using actual FPS.
        """
        last_check = self._last_locked_face_check.get(tid, -99999)
        if (frame_idx - last_check) >= self.locked_face_check_interval:
            self._last_locked_face_check[tid] = frame_idx
            return True
        return False

    def process_frame(self, frame: np.ndarray, frame_idx: int, tracker: BYTETracker) -> np.ndarray:
        h, w = frame.shape[:2]

        # 1. YOLO person detection
        results = self.yolo(frame, classes=[0], verbose=False)
        dets = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    dets.append([x1, y1, x2, y2, conf])

        self.stats["total_detections"] += len(dets)

        # 2. BYTETracker update
        if dets:
            det_array = np.array(dets, dtype=np.float32)
        else:
            det_array = np.empty((0, 5), dtype=np.float32)

        online_targets = tracker.update(det_array, [h, w], [h, w])
        self.stats["total_frames"] += 1

        # 3. Notify lock manager of ByteTrack lifecycle states (TRACKED / LOST / REMOVED)
        tracked_ids = [int(t.track_id) for t in tracker.tracked_stracks if t.is_activated]
        lost_ids = [int(t.track_id) for t in tracker.lost_stracks]
        removed_ids = [int(t.track_id) for t in tracker.removed_stracks]
        self.lock_manager.update_lifecycle(tracked_ids, lost_ids, removed_ids, frame_idx)

        # 4. Per-frame ownership invariant validation
        violations = self.lock_manager.validate_ownership()
        if violations:
            for v in violations:
                logger.error("OWNERSHIP VIOLATION at frame %d: %s", frame_idx, v)
            self.stats["ownership_violations"] += len(violations)

        annotated = frame.copy()
        frame_log_entry = {
            "frame": frame_idx,
            "detections": len(dets),
            "tracks": [],
            "ownership": dict(self.lock_manager.student_owner),
        }

        # Candidate students available (Part B3: scoped unowned students)
        candidate_students = set(KNOWN_STUDENTS) - set(self.lock_manager.student_owner.keys())

        for target in online_targets:
            tid = int(target.track_id)
            tlwh = target.tlwh
            x1 = int(tlwh[0])
            y1 = int(tlwh[1])
            x2 = int(tlwh[0] + tlwh[2])
            y2 = int(tlwh[1] + tlwh[3])
            bbox = (x1, y1, x2, y2)

            # Clamp bbox
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w, x2)
            y2c = min(h, y2)
            person_crop = frame[y1c:y2c, x1c:x2c]

            # -- Update seen --
            state = self.lock_manager.update_seen(tid, frame_idx)

            # -- Gap recovery check (before face/body extraction) --
            frames_gap = frame_idx - self._track_last_active.get(tid, frame_idx)
            is_new_appearance = frames_gap >= FACE_FRAME_GAP_FOR_RECOVERY
            if is_new_appearance and tid in self._last_body_emb and candidate_students:
                body_emb = self._last_body_emb[tid]
                b_name, b_score, b_sec = self.body_gallery.top_match(body_emb, candidate_students=candidate_students)
                if b_name is not None and b_name != "UNKNOWN":
                    recovered = self.lock_manager.attempt_gap_recovery(
                        tid, frame_idx, b_name, b_score, b_sec
                    )
                    if recovered:
                        self.stats["gap_recoveries"] += 1

            # -- Face extraction decision --
            was_locked = state.locked
            run_face = False
            use_640 = False

            if not state.locked:
                run_face = True
                # Part A: Time-based 640x640 escalation under Option B
                use_640 = self.should_escalate_to_640(tid, frame_idx, state.first_seen_frame, is_locked=False)
                if use_640:
                    self.stats["escalations_to_640"] += 1
            else:
                # Part B1: Throttled contradiction checks on locked tracks (~1.0 real second)
                # Part A5: Once locked, always returns to 320 (use_640 remains False)
                if self.should_check_locked_face(tid, frame_idx):
                    run_face = True

            if run_face:
                if not state.locked:
                    # Initial lock evaluation requires dual 320 and 640 agreement
                    # 1. First extract at 320 (fast standard resolution)
                    face_emb_320 = self.face_extractor.extract_from_crop(frame, bbox, padding=0.25, use_640=False)
                    self.stats["face_extractions"] += 1

                    if face_emb_320 is not None:
                        f_name_320, f_score_320, f_sec_320 = self.face_extractor.top_match(
                            face_emb_320, self.face_profiles
                        )
                        # If 320 candidate meets lock threshold and margin, verify with 640 before finalizing lock
                        if f_score_320 >= FACE_LOCK_THRESHOLD and (f_score_320 - f_sec_320) >= FACE_AMBIGUITY_MARGIN:
                            face_emb_640 = self.face_extractor.extract_from_crop(frame, bbox, padding=0.25, use_640=True)
                            self.stats["face_extractions"] += 1
                            if face_emb_640 is not None:
                                f_name_640, f_score_640, f_sec_640 = self.face_extractor.top_match(
                                    face_emb_640, self.face_profiles
                                )
                            else:
                                f_name_640, f_score_640, f_sec_640 = None, 0.0, 0.0

                            self.lock_manager.submit_face_match(
                                tid, frame_idx, f_name_320, f_score_320, f_sec_320,
                                name_640=f_name_640, score_640=f_score_640, second_score_640=f_sec_640
                            )
                            if not was_locked and state.locked:
                                self.stats["face_locks"] += 1
                        else:
                            # Below lock threshold or margin -> submit to keep state updated (holds UNKNOWN)
                            self.lock_manager.submit_face_match(
                                tid, frame_idx, f_name_320, f_score_320, f_sec_320
                            )
                    else:
                        # 320 found no face. If time-based escalation is active, try 640
                        if use_640:
                            face_emb_640 = self.face_extractor.extract_from_crop(frame, bbox, padding=0.25, use_640=True)
                            self.stats["face_extractions"] += 1
                            if face_emb_640 is not None:
                                f_name_640, f_score_640, f_sec_640 = self.face_extractor.top_match(
                                    face_emb_640, self.face_profiles
                                )
                                self.lock_manager.submit_face_match(
                                    tid, frame_idx, f_name_640, 0.0, 0.0,
                                    name_640=f_name_640, score_640=f_score_640, second_score_640=f_sec_640
                                )
                else:
                    # Already locked track: contradiction check (320 only, no 640)
                    face_emb = self.face_extractor.extract_from_crop(frame, bbox, padding=0.25, use_640=False)
                    self.stats["face_extractions"] += 1
                    if face_emb is not None:
                        f_name, f_score, f_sec = self.face_extractor.top_match(
                            face_emb, self.face_profiles
                        )
                        self.lock_manager.submit_face_match(
                            tid, frame_idx, f_name, f_score, f_sec
                        )
            else:
                self.stats["face_extractions_skipped"] += 1

            # Authoritative rendering label: ALWAYS state.locked_name if locked, else UNKNOWN
            if state.locked:
                label = state.locked_name
            else:
                label = "UNKNOWN"

            # -- Body embedding (Compute Reduction: Skip redundant computation) --
            # Part B2: Skip expensive OSNet extraction whenever unnecessary
            # Part B3: Check candidate_students BEFORE body embedding extraction
            run_body_extract = False
            if person_crop.size > 0 and (x2c - x1c) > 20 and (y2c - y1c) > 40:
                if state.locked:
                    # Locked track: only extract periodically to refresh template for future gap recovery
                    last_b = self._last_locked_body_emb_frame.get(tid, -99999)
                    if tid not in self._last_body_emb or (frame_idx - last_b) >= self.locked_face_check_interval:
                        run_body_extract = True
                else:
                    # Unlocked track: only extract if there are candidate students available
                    if len(candidate_students) > 0:
                        run_body_extract = True

                if run_body_extract:
                    body_emb = self.body_extractor.extract(person_crop)
                    if body_emb is not None:
                        self._last_body_emb[tid] = body_emb
                        if state.locked:
                            self._last_locked_body_emb_frame[tid] = frame_idx
                            cand = {state.locked_name}
                        else:
                            cand = candidate_students

                        b_name, b_score, b_sec = self.body_gallery.top_match(body_emb, candidate_students=cand)
                        if b_name is not None and b_name != "UNKNOWN":
                            self.lock_manager.submit_body_evidence(tid, frame_idx, b_name, b_score)
                        self.stats["body_extractions"] += 1
                else:
                    self.stats["body_extractions_skipped"] += 1

            self._track_last_active[tid] = frame_idx

            # -- Annotate --
            annotated = draw_track(annotated, bbox, tid, label)
            frame_log_entry["tracks"].append({
                "track_id": tid,
                "label": label,
                "locked": state.locked,
                "contradictions": state.contradiction_count,
                "bbox": [x1, y1, x2, y2],
            })

        # Overlay stats
        n_locked = sum(1 for t in frame_log_entry["tracks"] if t["locked"])
        n_unknown = len(frame_log_entry["tracks"]) - n_locked
        overlay = f"Frame {frame_idx} | Tracks: {len(online_targets)} | Locked: {n_locked} | Unknown: {n_unknown}"
        cv2.putText(annotated, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self.frame_log.append(frame_log_entry)
        return annotated

    def build_report(self, video_path: str, fps: float, total_frames: int) -> Dict:
        all_states = self.lock_manager.all_states()
        summary = self.lock_manager.summary()

        track_details = []
        for tid, state in sorted(all_states.items()):
            lifetime = state.last_seen_frame - state.first_seen_frame + 1
            track_details.append({
                "track_id": tid,
                "name": state.identity_label(),
                "locked": state.locked,
                "locked_at_frame": state.locked_at_frame,
                "locked_face_score": round(state.locked_face_score, 4),
                "first_frame": state.first_seen_frame,
                "last_frame": state.last_seen_frame,
                "lifetime_frames": lifetime,
                "lifetime_seconds": round(lifetime / fps, 2),
                "face_matches": state.face_matches,
                "body_matches": state.body_matches,
                "recovery_count": state.recovery_count,
                "contradictions": state.contradiction_count,
            })

        # Per-student coverage
        name_coverage: Dict[str, List] = defaultdict(list)
        for entry in self.frame_log:
            for t in entry["tracks"]:
                label = t["label"]
                name_coverage[label].append(entry["frame"])

        student_coverage = {}
        for name, frames in name_coverage.items():
            student_coverage[name] = {
                "frames": len(frames),
                "coverage_pct": round(100.0 * len(frames) / max(1, total_frames), 1),
            }

        # Final ownership state
        final_ownership = self.lock_manager.ownership_snapshot()

        return {
            "video": video_path,
            "total_frames_processed": total_frames,
            "fps": fps,
            "duration_seconds": round(total_frames / fps, 2),
            "summary": summary,
            "stats": dict(self.stats),
            "student_coverage": student_coverage,
            "tracks": track_details,
            "body_gallery_sizes": self.body_gallery.sizes(),
            "final_student_owner": final_ownership["student_owner"],
            "final_track_to_student": {str(k): v for k, v in final_ownership["track_to_student"].items()},
            "ownership_violations_detected": self.stats.get("ownership_violations", 0),
        }


# ------------------------------------------------------------------ #
#  Runner                                                              #
# ------------------------------------------------------------------ #

def run(
    video_path: str = "new lab video.mp4",
    max_frames: Optional[int] = None,
    output_dir: str = "test_outputs",
    output_stem: str = "robust_tracking_2min",
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("=== Robust Identity-Locking Tracker ===")
    logger.info("Video: %s | Max frames: %s | Output: %s", video_path, max_frames, output_dir)

    # -- Enrollment --
    logger.info("Starting student enrollment...")
    t0 = time.time()
    face_profiles, body_gallery = enroll_students(verbose=False)
    logger.info("Enrollment done in %.1fs", time.time() - t0)
    logger.info("Enrolled: %s", list(face_profiles.keys()))
    logger.info("Body gallery: %s", body_gallery.sizes())

    # -- Load models --
    from ultralytics import YOLO
    yolo = YOLO("yolo26n.pt")
    face_extractor = FaceFeatureExtractor(det_size=(320, 320))
    body_extractor = BodyFeatureExtractor()  # LOCKED: MobileNetV3-Small + HSV (672-dim)

    # -- Open video --
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info("Video: %dx%d @ %.2f FPS, total frames: %d", vid_w, vid_h, fps, total_frames_in_video)

    if max_frames is not None:
        process_limit = min(max_frames, total_frames_in_video)
    else:
        process_limit = total_frames_in_video

    logger.info("Will process %d frames (%.1fs)", process_limit, process_limit / fps)

    # -- Bytetrack --
    BaseTrack.reset_id()
    args = BYTETrackerArgs()
    args.track_thresh = 0.35
    args.track_buffer = 90    # 3 seconds at 30fps
    args.match_thresh = 0.85
    args.mot20 = False
    tracker = BYTETracker(args, frame_rate=int(fps))

    # -- Output --
    os.makedirs(output_dir, exist_ok=True)
    video_out = os.path.join(output_dir, f"{output_stem}.mp4")
    report_out = os.path.join(output_dir, f"{output_stem}_report.json")

    writer = VideoWriter(video_out, fps, vid_w, vid_h)
    robust_tracker = RobustTracker(face_profiles, body_gallery, yolo, face_extractor, body_extractor)

    # -- Main loop --
    logger.info("Starting tracking loop...")
    t_start = time.time()
    frame_idx = 0

    while frame_idx < process_limit:
        ok, frame = cap.read()
        if not ok:
            break

        annotated = robust_tracker.process_frame(frame, frame_idx, tracker)
        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 150 == 0:
            elapsed = time.time() - t_start
            fps_actual = frame_idx / max(elapsed, 0.001)
            remaining = (process_limit - frame_idx) / max(fps_actual, 0.001)
            lm = robust_tracker.lock_manager.summary()
            logger.info(
                "Frame %d/%d | %.1f FPS | locked=%d unknown=%d | ETA: %.0fs",
                frame_idx, process_limit, fps_actual,
                lm["locked_tracks"], lm["unknown_tracks"], remaining,
            )

    cap.release()
    writer.close()

    elapsed_total = time.time() - t_start
    logger.info("Tracking complete: %d frames in %.1fs (%.1f FPS)", frame_idx, elapsed_total, frame_idx / elapsed_total)

    # -- Report --
    report = robust_tracker.build_report(video_path, fps, frame_idx)
    report["processing_time_seconds"] = round(elapsed_total, 2)
    report["processing_fps"] = round(frame_idx / elapsed_total, 2)

    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=== FINAL REPORT ===")
    logger.info("Output video: %s", video_out)
    logger.info("Report JSON: %s", report_out)
    logger.info("Total tracks: %d", report["summary"]["total_tracks"])
    logger.info("Locked tracks: %d", report["summary"]["locked_tracks"])
    logger.info("Unknown tracks: %d", report["summary"]["unknown_tracks"])
    logger.info("Identity assignments: %s", report["summary"]["name_assignments"])
    logger.info("Student coverage: %s", report["student_coverage"])
    logger.info("Stats: %s", report["stats"])
    logger.info("Body gallery final sizes: %s", report["body_gallery_sizes"])

    return report


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Identity-Locking Student Tracker")
    parser.add_argument("--video", default="new lab video.mp4", help="Input video path")
    parser.add_argument("--max-frames", type=int, default=3600, help="Max frames to process (default: 3600 = 2 min at 30fps)")
    parser.add_argument("--output-dir", default="test_outputs", help="Output directory")
    parser.add_argument("--output-stem", default="robust_tracking_2min", help="Output filename stem")
    args = parser.parse_args()

    run(
        video_path=args.video,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
    )
