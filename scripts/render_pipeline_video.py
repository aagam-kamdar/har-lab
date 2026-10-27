"""
scripts/render_pipeline_video.py
--------------------------------
Renders an annotated visualization video of the final pipeline:
- ByteTrack + ArcFace Lock + Spatio-Temporal Stitching + Gate Display
- Overlays bounding boxes, student names, status tags ([LOCKED], [STITCHED], [IDENTIFYING]),
  desk zones, and a real-time HUD.
- Saves to test_outputs/final_pipeline_stitching_demo.mp4
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter

VIDEO_PATH = "new lab video.mp4"
CACHE_DETS = "test_outputs/cached_new_lab_yolo_dets.npz"
OUTPUT_PATH = "test_outputs/final_pipeline_stitching_demo.mp4"

# Color definitions (BGR)
COLOR_GREEN = (40, 200, 60)      # LOCKED
COLOR_CYAN = (230, 180, 40)      # STITCHED
COLOR_YELLOW = (40, 210, 240)    # IDENTIFYING
COLOR_GRAY = (140, 140, 140)     # UNKNOWN
COLOR_BG_DARK = (20, 20, 20)
COLOR_WHITE = (255, 255, 255)

DESK_ZONES = {
    "D01": (450, 0, 570, 360, "Desk 1 (Soham)"),
    "D02": (540, 20, 660, 360, "Desk 2 (Swayam)"),
    "D03": (640, 40, 780, 380, "Desk 3 (Harshil)"),
    "D04": (770, 50, 960, 420, "Desk 4 (Aagam)")
}


def match_desk(bbox):
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    for zid, (rx1, ry1, rx2, ry2, name) in DESK_ZONES.items():
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            return zid
    return None


def render_video(max_frames=1200):
    print(f"[INFO] Initializing video rendering up to frame {max_frames}...")

    npz = np.load(CACHE_DETS, allow_pickle=True)
    cached_dets = {int(k[2:]): npz[k] for k in npz.files}

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

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_idx = 0
    t0 = time.time()

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        det_boxes = cached_dets.get(frame_idx, np.empty((0, 5), dtype=np.float32))
        tracks = adapter.update(frame, frame_idx, det_boxes=det_boxes)

        # Draw Desk Zones (subtle dashed/thin rectangles)
        for zid, (rx1, ry1, rx2, ry2, zname) in DESK_ZONES.items():
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (70, 70, 70), 1)
            cv2.putText(frame, zid, (rx1 + 5, ry1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        # Draw Tracks
        locked_count = 0
        stitched_count = 0

        for t in tracks:
            tid = t["track_id"]
            bbox = [int(v) for v in t["bbox"]]
            x1, y1, x2, y2 = bbox
            disp_name = t["display_name"]
            status = t["status"]
            aff = t.get("affinity", 0.0)

            desk = match_desk(bbox)
            desk_str = f" [{desk}]" if desk else ""

            if status == "LOCKED":
                color = COLOR_GREEN
                tag = "FACE LOCKED"
                locked_count += 1
            elif tid in adapter.stitched_tracks:
                color = COLOR_CYAN
                tag = "STITCHED"
                stitched_count += 1
            elif status == "ASSIGNED":
                color = COLOR_CYAN
                tag = "ASSIGNED"
            elif status == "IDENTIFYING":
                color = COLOR_YELLOW
                tag = "IDENTIFYING"
            else:
                color = COLOR_GRAY
                tag = "UNKNOWN"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label banner
            label_text = f"#{tid} {disp_name}{desk_str} | {tag}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 5, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BG_DARK, 1, cv2.LINE_AA)

        # Top HUD Dashboard
        hud_h = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, hud_h), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        time_s = frame_idx / fps
        hud_left = f"FRAME: {frame_idx:04d} / {max_frames} | TIME: {time_s:5.2f}s | FPS: {fps:.1f} | ACTIVE TRACKS: {len(tracks)}"
        hud_right = f"LOCKED: {locked_count} | STITCHED: {stitched_count} | PIPELINE: SPATIO-TEMPORAL STITCHING"

        cv2.putText(frame, hud_left, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, hud_right, (width - 660, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 240, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    elapsed = time.time() - t0
    print(f"\n[DONE] Rendered {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    render_video(max_frames=1200)
