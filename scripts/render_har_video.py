"""
scripts/render_har_video.py
---------------------------
Renders a visualization video demonstrating activity recognition EXCLUSIVELY
through the pretrained Human Activity Recognition (HAR) model:
  Model: MCG-NJU/videomae-small-finetuned-kinetics (Kinetics-400)

Features:
  - 100% Geometry-free activity classification
  - ByteTrack tracking + ArcFace/OSNet identity resolution (stitching mode)
  - Per-student 16-frame VideoMAE clip extraction and inference
  - Displays real-time mapped application action + raw Kinetics-400 label + confidence score
  - Top HUD showing frame index, video time, and model status
  - Saves to test_outputs/pretrained_har_video_demo.mp4
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter
from har_classifier import HARClassifier

VIDEO_PATH = "new video.mp4"
CACHE_DETS = None
OUTPUT_PATH = "test_outputs/pretrained_har_video_demo.mp4"

# Color definitions (BGR)
ACTION_COLORS = {
    "active_work": (60, 200, 75),    # Vibrant Green
    "writing":     (80, 220, 120),   # Mint Green
    "reading":     (240, 180, 50),   # Cyan
    "phone":       (40, 40, 240),    # Red
    "call":        (30, 120, 255),   # Orange
    "discussing":  (220, 140, 40),   # Blue-Amber
    "idle":        (30, 210, 240),   # Yellow
    "walking":     (200, 80, 180),   # Purple
    "away":        (120, 120, 120),  # Gray
    "uncertain":   (160, 160, 160),  # Light Gray
}
COLOR_BG_DARK = (20, 20, 20)
COLOR_WHITE = (255, 255, 255)

DESK_ZONES = {
    "D01": (450, 0, 570, 360, "Desk 1 (Harshil)"),
    "D02": (540, 20, 660, 360, "Desk 2 (Swayam)"),
    "D03": (640, 40, 780, 380, "Desk 3 (Aagam)"),
    "D04": (770, 50, 960, 420, "Desk 4 (Soham)")
}


def match_desk(bbox):
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    for zid, (rx1, ry1, rx2, ry2, name) in DESK_ZONES.items():
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            return zid
    return None


def render_har_video(max_frames=600):
    print(f"[INFO] Initializing Pretrained HAR video rendering ({max_frames} frames)...")

    cached_dets = None
    if CACHE_DETS and os.path.exists(CACHE_DETS):
        npz = np.load(CACHE_DETS, allow_pickle=True)
        cached_dets = {int(k[2:]): npz[k] for k in npz.files}
        print(f"[INFO] Loaded {len(cached_dets)} cached detection frames.")

    adapter = IdentityAdapter(
        fps=30.0,
        yolo_model_path="yolo26n.pt",
        identifying_timeout_s=5.0,
        live_gate_evidence_s=1.5,
        clear_margin=0.15,
        arcface_confirm_s=0.5,
        body_stable_s=3.0,
        mode="stitching",
        min_affinity_thresh=0.50,
        body_only_cross_assignment=False,
        stitching_enabled=True,
        stitching_max_gap_s=5.0,
        stitching_max_dist_px=100.0,
        stitching_body_veto_thresh=0.40
    )

    print("[INFO] Loading VideoMAE Pretrained HAR Classifier...")
    har = HARClassifier(
        model_name="MCG-NJU/videomae-small-finetuned-kinetics",
        clip_len=16,
        inference_interval_s=2.0,
        confidence_threshold=0.25,
        device="cpu",
        async_mode=False  # synchronous for video rendering to guarantee exact frames
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_idx = 0
    t0 = time.time()
    total_har_calls = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        now_sim = frame_idx / fps

        # Identity tracking pass
        det_boxes = cached_dets.get(frame_idx, None) if cached_dets else None
        tracks = adapter.update(frame, frame_idx, det_boxes=det_boxes)

        # Draw Desk Zones
        for zid, (rx1, ry1, rx2, ry2, zname) in DESK_ZONES.items():
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (65, 65, 65), 1)
            cv2.putText(frame, zid, (rx1 + 5, ry1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1)

        active_har_count = 0

        for t in tracks:
            tid = t["track_id"]
            bbox = [int(v) for v in t["bbox"]]
            x1, y1, x2, y2 = bbox
            disp_name = t["display_name"]
            status = t["status"]

            # Feed frame to HAR classifier for this student track
            pred = har.update_track(track_id=tid, frame=frame, bbox=bbox, timestamp=now_sim)
            if pred is not None:
                active_har_count += 1
                total_har_calls += 1

            desk = match_desk(bbox)
            desk_str = f" [{desk}]" if desk else ""

            # Action mapping & display attributes
            if pred is not None:
                mapped_act = pred.mapped_action
                raw_lbl = pred.raw_label
                conf = pred.confidence
                lat = pred.latency_ms
            else:
                buf_len = len(har._buffers.get(tid, []))
                mapped_act = "uncertain"
                raw_lbl = f"buffering ({buf_len}/16)"
                conf = 0.0
                lat = 0.0

            color = ACTION_COLORS.get(mapped_act, (160, 160, 160))

            # Draw person bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Top label banner: Student name + Identity State + Mapped action
            id_state = t.get("identity_state", "UNIDENTIFIED")
            top_label = f"#{tid} {disp_name} [{id_state}]{desk_str} | {mapped_act.upper()}"
            (tw1, th1), _ = cv2.getTextSize(top_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw1 + 10, y1), color, -1)
            cv2.putText(frame, top_label, (x1 + 5, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BG_DARK, 1, cv2.LINE_AA)

            # Bottom sub-banner: Raw Kinetics-400 label + confidence score
            conf_str = f" ({conf*100:.1f}%)" if conf > 0 else ""
            sub_label = f"Kinetics-400: '{raw_lbl}'{conf_str}"
            (tw2, th2), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(frame, (x1, y2), (x1 + tw2 + 8, min(height, y2 + 20)), COLOR_BG_DARK, -1)
            cv2.putText(frame, sub_label, (x1 + 4, min(height - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        # Top HUD Dashboard
        hud_h = 52
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, hud_h), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        time_s = frame_idx / fps
        hud_left = f"FRAME: {frame_idx:04d}/{max_frames} | TIME: {time_s:5.2f}s | FPS: {fps:.1f} | TRACKS: {len(tracks)}"
        hud_right = f"PIPELINE: PRETRAINED HAR (VideoMAE Kinetics-400) | NO GEOMETRY RULES"

        cv2.putText(frame, hud_left, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, hud_right, (width - 710, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 240, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"  [RENDER] Processed frame {frame_idx}/{max_frames} ({frame_idx/max_frames*100:.0f}%)...")

    cap.release()
    out.release()
    elapsed = time.time() - t0
    print(f"\n[DONE] Rendered {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS) -> {OUTPUT_PATH}")
    print(f"       Total HAR inferences executed: {har.total_inferences}, Avg Latency: {har.get_stats()['average_latency_ms']} ms")


if __name__ == "__main__":
    render_har_video(max_frames=300)
