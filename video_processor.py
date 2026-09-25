"""
video_processor.py
------------------
100% LOCAL VIDEO TRACKING & IDENTITY RESOLUTION PIPELINE:
  Input Video (.mp4)
       ↓
  Local YOLO (Person Class Only: classes=[0])
       ↓
  Convert detections to [x1, y1, x2, y2, confidence]
       ↓
  Official BYTETracker.update(...) (Sequential frames)
       ↓
  Local Person Re-ID + FAISS Vector Index (Temporal Confirmation)
       ↓
  Persistent Student Identity Resolution (Track 19 | Student 03)
       ↓
  Single browser-playable MP4 output (H.264 / PyAV)

Zero cloud calls. Zero Roboflow dependencies.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set

import cv2
import numpy as np
from ultralytics import YOLO

from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack, STrack, TrackState
from reid import PersonReIDExtractor, StudentGallery, IdentityResolver

logger = logging.getLogger(__name__)

# Bounding box color palette (BGR)
COLOR_PALETTE = [
    (94, 234, 212),   # Teal
    (244, 114, 182),  # Pink
    (129, 140, 248),  # Indigo
    (251, 191, 36),   # Amber
    (52, 211, 153),   # Emerald
    (248, 113, 113),  # Rose
    (167, 139, 250),  # Purple
    (56, 189, 248),   # Sky
    (234, 179, 8),    # Yellow
    (14, 165, 233),   # Light Blue
    (249, 115, 22),   # Orange
    (168, 85, 247),   # Violet
]
TEXT_COLOR = (11, 15, 14)  # Dark charcoal text


# ---------------------------------------------------------------------------
# Result Data Model
# ---------------------------------------------------------------------------

@dataclass
class LocalTrackingResult:
    output_video_filename: str
    output_video_path: str
    total_frames: int
    fps: float
    width: int
    height: int
    unique_track_ids: list[int]
    persistent_student_ids: list[str]
    duration_sec: float
    reid_metrics: dict = field(default_factory=dict)
    recovery_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Universal Video Writer (H.264 browser-playable with cv2 fallback)
# ---------------------------------------------------------------------------

class UniversalVideoWriter:
    """
    Writes browser-playable MP4 video.
    Prefers PyAV with libx264 (playable in all HTML5 browsers without plugins),
    with graceful fallback to OpenCV cv2.VideoWriter.
    """

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = output_path
        self.fps = max(1.0, float(fps))
        self.width = width
        self.height = height
        self.mode = "av"
        self.container = None
        self.stream = None
        self.cv2_writer = None

        try:
            import av

            self.container = av.open(output_path, mode="w")
            self.stream = self.container.add_stream("h264", rate=int(round(self.fps)))
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"preset": "veryfast", "crf": "23"}
            self.mode = "av"
            logger.info("UniversalVideoWriter: initialized H.264 stream via PyAV")
        except Exception as exc:
            logger.warning(
                "PyAV unavailable (%s). Falling back to cv2.VideoWriter (mp4v).", exc
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.cv2_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps, (width, height)
            )
            self.mode = "cv2"

    def write(self, bgr_frame: np.ndarray):
        if self.mode == "av":
            import av

            video_frame = av.VideoFrame.from_ndarray(bgr_frame, format="bgr24")
            for packet in self.stream.encode(video_frame):
                self.container.mux(packet)
        elif self.cv2_writer is not None:
            self.cv2_writer.write(bgr_frame)

    def release(self):
        if self.mode == "av" and self.container is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
            logger.info("UniversalVideoWriter: closed PyAV container")
        elif self.cv2_writer is not None:
            self.cv2_writer.release()
            logger.info("UniversalVideoWriter: released cv2.VideoWriter")


# ---------------------------------------------------------------------------
# Annotation / Drawing
# ---------------------------------------------------------------------------

def draw_tracks_on_frame(
    frame: np.ndarray,
    tracks: List[STrack],
    identity_resolver: Optional[IdentityResolver] = None,
) -> np.ndarray:
    """
    Draw persistent bounding boxes and track ID / student ID banners on the frame.
    Format: "Track 19 | Student 03" or "Track 1"
    Color is locked to the persistent student identity once resolved.
    """
    h_fr, w_fr = frame.shape[:2]

    for track in tracks:
        tlbr = track.tlbr
        x1, y1, x2, y2 = (int(v) for v in tlbr)

        # Clamp coordinates to frame boundary
        x1 = max(0, min(x1, w_fr - 1))
        y1 = max(0, min(y1, h_fr - 1))
        x2 = max(0, min(x2, w_fr - 1))
        y2 = max(0, min(y2, h_fr - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        student_id = identity_resolver.get_student_id(track.track_id) if identity_resolver else None

        # Lock color to persistent student_id so identity continuity is visually clear
        if student_id:
            try:
                s_num = int(student_id.split()[-1])
                color = COLOR_PALETTE[(s_num - 1) % len(COLOR_PALETTE)]
            except Exception:
                color = COLOR_PALETTE[abs(hash(student_id)) % len(COLOR_PALETTE)]
            label = f"Track {track.track_id} | {student_id}"
        else:
            color = COLOR_PALETTE[track.track_id % len(COLOR_PALETTE)]
            label = f"Track {track.track_id}"

        # Bounding box border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Font scale adapted to box size
        box_h = y2 - y1
        font_scale = max(0.45, min(0.65, box_h / 140.0))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

        # Filled banner above box
        label_top = max(0, y1 - th - 8)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + tw + 8, y1),
            color,
            -1,
        )

        # Label text
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    return frame


# ---------------------------------------------------------------------------
# Core Local Video Tracking + Re-ID Function
# ---------------------------------------------------------------------------

def process_video_local(
    video_path: str,
    output_dir: str,
    model_path: str = "yolo26n.pt",
    conf_thresh: float = 0.30,
    track_thresh: float = 0.50,
    track_buffer: int = 30,
    match_thresh: float = 0.80,
    similarity_thresh: float = 0.80,
    ambiguity_margin: float = 0.04,
    min_observations: int = 3,
    output_filename: str = "output_tracked.mp4",
) -> LocalTrackingResult:
    """
    Process an uploaded video completely locally with Re-ID:
      1. Open video with OpenCV.
      2. Sequentially read EVERY frame.
      3. Run local YOLO person detection (classes=[0]).
      4. Convert detections to BYTETracker format: [x1, y1, x2, y2, score].
      5. Feed sequential frames to official BYTETracker.update(...).
      6. Query Re-ID embedding & FAISS index for new/unconfirmed tracks.
      7. Apply temporal voting to confirm student_id.
      8. Draw "Track X | Student Y" onto every frame.
      9. Write annotated frames into output_tracked.mp4.
      10. Return LocalTrackingResult with tracking & Re-ID metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps is None or native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Opened video '%s': %dx%d @ %.2f fps, ~%d frames",
        os.path.basename(video_path),
        width,
        height,
        native_fps,
        total_frames_est,
    )

    # 1. Load Local YOLO detector
    actual_weights = model_path
    if not os.path.exists(actual_weights):
        for candidate in ("yolo26n.pt", "yolo11n.pt", "yolov8n.pt"):
            if os.path.exists(candidate):
                actual_weights = candidate
                break

    logger.info("Loading local YOLO model from: %s", actual_weights)
    yolo_model = YOLO(actual_weights)

    # 2. Initialize Official BYTETracker
    BaseTrack.reset_id()
    tracker_args = BYTETrackerArgs(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        mot20=False,
    )
    tracker = BYTETracker(args=tracker_args, frame_rate=int(round(native_fps)))

    # 3. Initialize Re-ID Pipeline (Local Gallery + FAISS Index + Identity Resolver)
    reid_base_dir = os.path.join(output_dir, "reid")
    os.makedirs(reid_base_dir, exist_ok=True)
    extractor = PersonReIDExtractor()
    gallery = StudentGallery(base_dir=reid_base_dir, extractor=extractor)
    gallery.reset()  # Clean gallery for fresh video run

    resolver = IdentityResolver(
        gallery=gallery,
        extractor=extractor,
        similarity_thresh=similarity_thresh,
        ambiguity_margin=ambiguity_margin,
        min_observations=min_observations,
        fps=native_fps,
    )

    # 4. Initialize Video Writer
    out_video_path = os.path.join(output_dir, output_filename)
    writer = UniversalVideoWriter(out_video_path, native_fps, width, height)

    unique_track_ids: Set[int] = set()
    total_person_dets = 0
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1

            # ---------------------------------------------------------------
            # Step A: Local YOLO Person Detection (classes=[0] -> PERSON ONLY)
            # ---------------------------------------------------------------
            yolo_results = yolo_model(
                frame,
                conf=conf_thresh,
                classes=[0],
                verbose=False,
            )[0]

            num_dets = len(yolo_results.boxes)
            total_person_dets += num_dets

            if num_dets > 0:
                xyxy = yolo_results.boxes.xyxy.cpu().numpy()
                conf = yolo_results.boxes.conf.cpu().numpy()[:, None]
                detections = np.hstack([xyxy, conf])
            else:
                detections = np.empty((0, 5), dtype=float)

            # Record before-lost tracks to detect ByteTrack reactivations
            before_lost_tids = {t.track_id for t in tracker.lost_stracks}

            # ---------------------------------------------------------------
            # Step B: Official BYTETracker Sequential Update
            # ---------------------------------------------------------------
            online_targets: List[STrack] = tracker.update(
                detections,
                img_info=(height, width),
                img_size=(height, width),
            )

            # Record ByteTrack short-term reactivations (Requirement 9)
            after_tracked_tids = {t.track_id for t in tracker.tracked_stracks}
            reactivated_tids = after_tracked_tids & before_lost_tids
            for tid in reactivated_tids:
                resolver.record_short_term_reactivation(tid)

            # ---------------------------------------------------------------
            # Step C: Re-ID & Identity Resolution (Triggered only when needed)
            # ---------------------------------------------------------------
            for trk in online_targets:
                tid = trk.track_id
                unique_track_ids.add(tid)

                # If track does not have a confirmed student_id yet, resolve it
                if resolver.get_student_id(tid) is None:
                    # Extract crop from current frame
                    x1, y1, x2, y2 = [int(v) for v in trk.tlbr]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    if x2 > x1 + 10 and y2 > y1 + 10:
                        crop = frame[y1:y2, x1:x2]
                        resolver.resolve_track(
                            track_id=tid,
                            crop=crop,
                            frame_idx=frame_index,
                            allow_new_registration=True,
                        )

            # Console log periodically
            if frame_index % 100 == 0 or frame_index == 1:
                active_str = [
                    f"T{t.track_id}:{resolver.get_student_id(t.track_id) or '?'}"
                    for t in online_targets
                ]
                logger.info(
                    "Frame %4d: %d persons | Active: %s",
                    frame_index,
                    len(online_targets),
                    ", ".join(active_str),
                )

            # ---------------------------------------------------------------
            # Step D: Draw Annotations (Track X | Student Y)
            # ---------------------------------------------------------------
            annotated_frame = draw_tracks_on_frame(frame, online_targets, resolver)

            # ---------------------------------------------------------------
            # Step E: Write to Output MP4 Video
            # ---------------------------------------------------------------
            writer.write(annotated_frame)

    finally:
        cap.release()
        writer.release()
        resolver.finalize()
        # Save gallery and FAISS index
        gallery.save()

    duration_sec = frame_index / native_fps if native_fps > 0 else 0.0
    sorted_unique_tracks = sorted(unique_track_ids)
    persistent_students = sorted(list(set(resolver.track_to_student.values())))
    reid_metrics = resolver.get_metrics()
    reid_metrics["total_person_dets"] = total_person_dets
    reid_metrics["avg_dets_per_frame"] = round(total_person_dets / frame_index, 2) if frame_index > 0 else 0.0

    logger.info(
        "Finished local tracking + Re-ID: %d frames (%.2fs), %d track IDs -> %d persistent student IDs",
        frame_index,
        duration_sec,
        len(sorted_unique_tracks),
        len(persistent_students),
    )

    return LocalTrackingResult(
        output_video_filename=output_filename,
        output_video_path=out_video_path,
        total_frames=frame_index,
        fps=native_fps,
        width=width,
        height=height,
        unique_track_ids=sorted_unique_tracks,
        persistent_student_ids=persistent_students,
        duration_sec=duration_sec,
        reid_metrics=reid_metrics,
        recovery_log=resolver.recovery_log,
    )


# Alias for backward compatibility
def process_video(video_path: str, output_dir: str, **kwargs) -> LocalTrackingResult:
    return process_video_local(video_path=video_path, output_dir=output_dir, **kwargs)