"""
run_12fps_full_video.py
-----------------------
Executes the identity tracking pipeline at 12 FPS (alternating stride +2/+3)
on the FULL 6,005-frame (3 min 20 sec @ 30 FPS) 'new lab video.mp4'.

Configuration:
  - Sampling: 12 FPS (alternating stride +2/+3 from 30 FPS source -> 2,402 frames total)
  - Tracker: ByteTrack with frame_rate=12 (buffer_size = 36 steps = 1.20s real video time)
  - Detector: YOLO26n-pose (0-1799 from frozen cache, 1800-6004 live batched inference)
  - Face Verification: SCRFD + ArcFace (512-dim, sticky locks)
  - Re-ID: OSNet-x0.25 (with strict candidate-pool scoping)
  - Outputs:
      - test_outputs/full_video_12fps_annotated.mp4 (annotated 12 FPS playback)
      - test_outputs/downtime_distribution_12fps_full_video.json (detailed metric log)
"""

import sys, os, io

# Ensure unbuffered / line-buffered UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

import time
import json
import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
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
)
from video_processor import UniversalVideoWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("12FPSFullVideo")

VIDEO_PATH = "new lab video.mp4"
CACHED_DETECTIONS_1800F = "test_outputs/cached_yolo26pose_detections_1800f.json"
OUTPUT_VIDEO_PATH = "test_outputs/full_video_12fps_annotated.mp4"
SUMMARY_JSON_PATH = "test_outputs/downtime_distribution_12fps_full_video.json"
GALLERY_NPZ = "galleries_osnet_x0_25/gallery_embeddings.npz"
YOLO_MODEL_PATH = "yolo26n-pose.pt"

TOTAL_FRAMES = 6005
SOURCE_FPS = 30.0
TARGET_FPS = 12.0
BATCH_SIZE = 32

# Colors (BGR)
COLOR_GREEN = (40, 220, 40)       # ArcFace Locked
COLOR_YELLOW = (0, 215, 255)      # Re-ID Recovered
COLOR_ORANGE = (0, 140, 255)      # UNKNOWN
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (180, 180, 180)
COLOR_RED = (40, 40, 240)


def format_mm_ss(frame_idx: int, fps: float = 30.0) -> str:
    total_seconds = frame_idx / fps
    mins = int(total_seconds // 60)
    secs = int(total_seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def load_osnet_gallery(dim: int, npz_path: str) -> BodyGallery:
    gal = BodyGallery(dim=dim)
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        for student in KNOWN_STUDENTS:
            if student in data:
                for vec in data[student]:
                    gal.add(student, vec)
    return gal


def draw_hud(frame: np.ndarray, frame_idx: int, total_frames: int, lock_mgr: IdentityLockManager, active_count: int):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (330, 210), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (10, 10), (330, 210), (90, 90, 90), 1)

    time_str = f"{format_mm_ss(frame_idx)} / {format_mm_ss(total_frames)}  (12 FPS Track)"
    cv2.putText(frame, time_str, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Active Tracks: {active_count}", (22, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (20, 72), (320, 72), (70, 70, 70), 1)

    row_y = 96
    for s_name in KNOWN_STUDENTS:
        owner_tid = lock_mgr.student_owner.get(s_name)
        if owner_tid is not None:
            state = lock_mgr.get_state(owner_tid)
            if state.locked:
                status_txt = "FACE-LOCKED"
                txt_col = COLOR_GREEN
            elif state.status == "LOST":
                status_txt = "BUFFER-LOST"
                txt_col = COLOR_ORANGE
            else:
                status_txt = "RE-ID ACTIVE"
                txt_col = COLOR_YELLOW
            line = f"  {s_name.capitalize():<8}: Track {owner_tid:<2} [{status_txt}]"
        else:
            line = f"  {s_name.capitalize():<8}: UNOWNED / UNKNOWN"
            txt_col = COLOR_GRAY

        cv2.putText(frame, line, (22, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, txt_col, 1, cv2.LINE_AA)
        row_y += 26


def draw_bounding_box(frame: np.ndarray, bbox: Tuple[int, int, int, int], track_id: int, label_name: str, color: Tuple[int, int, int], sub_tag: Optional[str] = None):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    corner_len = min(16, (x2 - x1) // 4, (y2 - y1) // 4)
    if corner_len > 0:
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

    text = f"Track {track_id} | {label_name.upper()}"
    if sub_tag:
        text += f" [{sub_tag}]"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    tag_y1 = max(0, y1 - th - 8)
    tag_y2 = y1
    tag_x2 = min(frame.shape[1], x1 + tw + 10)

    cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
    text_color = (10, 10, 10) if color in [COLOR_YELLOW, COLOR_GREEN, COLOR_ORANGE] else (255, 255, 255)
    cv2.putText(frame, text, (x1 + 5, tag_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_color, 1, cv2.LINE_AA)


def main():
    print("=" * 72)
    print("STARTING FULL 3 MIN 20 SEC VIDEO RUN AT 12 FPS (6,005 frames)")
    print("=" * 72)
    os.makedirs("test_outputs", exist_ok=True)

    # 1. Pre-cached Detections for frames 0-1799
    cached_detections_map = {}
    if os.path.exists(CACHED_DETECTIONS_1800F):
        with open(CACHED_DETECTIONS_1800F, "r") as f:
            cache_data = json.load(f)
        for item in cache_data["detections"]:
            cached_detections_map[item["frame_idx"]] = item["boxes"]
        print(f"[INFO] Loaded {len(cached_detections_map)} pre-cached detection frames (0-1799).")

    # 2. YOLO Model for frames 1800+
    print(f"[INFO] Loading {YOLO_MODEL_PATH} for live inference on frames 1800+ ...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    # 3. ByteTrack at 12 FPS
    bt_args = BYTETrackerArgs()
    bt_args.track_thresh = 0.35
    bt_args.track_buffer = 90
    bt_args.match_thresh = 0.85
    bt_args.mot20 = False
    tracker = BYTETracker(bt_args, frame_rate=int(TARGET_FPS))
    buf_size = int(TARGET_FPS / SOURCE_FPS * bt_args.track_buffer)
    print(f"[INFO] ByteTrack configured for 12 FPS: buffer_size = {buf_size} steps ({buf_size/SOURCE_FPS:.2f}s video time)")

    # 4. ArcFace & OSNet
    print("[INFO] Initializing FaceFeatureExtractor & OSNet ...")
    face_extractor = FaceFeatureExtractor(det_size=(320, 320))
    face_profiles, _ = enroll_students(verbose=False)
    body_extractor = ExtractorB()
    body_gallery = load_osnet_gallery(dim=512, npz_path=GALLERY_NPZ)

    # 5. Lock Manager & Video Stream
    lock_mgr = IdentityLockManager()
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_frames = min(TOTAL_FRAMES, total_video_frames)

    writer = UniversalVideoWriter(OUTPUT_VIDEO_PATH, fps=TARGET_FPS, width=width, height=height)

    # Build 12 FPS frame sample sequence (alternating stride 2/3)
    frames_12fps = []
    f, use2 = 0, True
    while f < actual_frames:
        frames_12fps.append(f)
        f += 2 if use2 else 3
        use2 = not use2

    print(f"[INFO] Total 12 FPS frames to process: {len(frames_12fps)} (out of {actual_frames} source frames)")

    # Tracking & Metrics state
    removal_pending = {}  # student -> (removed_sec, old_tid, removed_fid)
    downtime_events = []
    recovered_map = {}
    prev_owned = {}
    face_confirmations = defaultdict(lambda: deque(maxlen=FACE_MIN_CONFIRMATIONS))

    t0 = time.time()

    # Process in batches for live YOLO if beyond 1800
    for idx, fid in enumerate(frames_12fps):
        video_sec = fid / SOURCE_FPS
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            break

        # Get Detections
        if fid in cached_detections_map:
            raw_boxes = cached_detections_map[fid]
            det_boxes = np.array(raw_boxes, dtype=np.float32) if raw_boxes else np.empty((0, 5), dtype=np.float32)
        else:
            res = yolo_model(frame, classes=[0], conf=0.25, verbose=False)[0]
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            if len(boxes_xyxy) > 0:
                det_boxes = np.hstack([boxes_xyxy, confs[:, None]]).astype(np.float32)
            else:
                det_boxes = np.empty((0, 5), dtype=np.float32)

        # ByteTrack Update
        img_info = [frame.shape[0], frame.shape[1]]
        online_targets = tracker.update(det_boxes, img_info, img_info)

        tracked_ids = [t.track_id for t in tracker.tracked_stracks]
        lost_ids = [t.track_id for t in tracker.lost_stracks]
        removed_ids = [t.track_id for t in tracker.removed_stracks]

        # Sync Lifecycle
        lock_mgr.update_lifecycle(tracked_ids, lost_ids, removed_ids, current_frame=fid)

        # Detect Removals
        curr_owned = dict(lock_mgr.student_owner)
        for student, old_tid in prev_owned.items():
            if student not in curr_owned and student not in removal_pending:
                removal_pending[student] = (video_sec, old_tid, fid)
                print(f"  [REMOVAL] f{fid} ({video_sec:.2f}s): {student} (old track {old_tid})")
        prev_owned = dict(curr_owned)

        # Process Active Tracks
        for target in online_targets:
            tid = target.track_id
            tlwh = target.tlwh
            x1, y1, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])
            bbox = (max(0, x1), max(0, y1), min(width, x1 + w), min(height, y1 + h))

            lock_mgr.update_seen(tid, fid)
            state = lock_mgr.get_state(tid)

            # ArcFace Identification
            if not state.locked and idx % 3 == 0:
                emb = face_extractor.extract_from_crop(frame, bbox)
                if emb is not None:
                    cand_set = set(KNOWN_STUDENTS) - set(lock_mgr.student_owner.keys())
                    if state.locked_name:
                        cand_set.add(state.locked_name)
                    filtered_profiles = {k: v for k, v in face_profiles.items() if k in cand_set}
                    if filtered_profiles:
                        best_name, best_score, second_score = face_extractor.top_match(emb, filtered_profiles)
                        margin = best_score - second_score
                        if best_score >= FACE_LOCK_THRESHOLD and margin >= FACE_AMBIGUITY_MARGIN:
                            face_confirmations[tid].append((best_name, best_score, second_score))
                            buf = list(face_confirmations[tid])
                            if len(buf) >= FACE_MIN_CONFIRMATIONS and all(b[0] == best_name for b in buf[-FACE_MIN_CONFIRMATIONS:]):
                                prior_s = lock_mgr.track_to_student.get(tid)
                                lock_mgr.submit_face_match(tid, fid, best_name, best_score, second_score)
                                new_s = lock_mgr.track_to_student.get(tid)
                                if new_s and new_s != prior_s:
                                    recovered_map.pop(tid, None)
                                    if new_s in removal_pending:
                                        old_sec, old_tid, old_fid = removal_pending.pop(new_s)
                                        downtime = video_sec - old_sec
                                        pool_sz = len(set(KNOWN_STUDENTS) - set(lock_mgr.student_owner.keys())) + 1
                                        ev = dict(
                                            student=new_s, old_tid=old_tid, old_removed_sec=old_sec,
                                            new_tid=tid, recovery_sec=video_sec, downtime_sec=downtime,
                                            method="ArcFace", score=best_score, margin=margin, candidate_pool_size=pool_sz
                                        )
                                        downtime_events.append(ev)
                                        print(f"  [RECOVERY-ArcFace] f{fid} ({video_sec:.2f}s): {new_s} on Track {tid} (downtime: {downtime:.2f}s)")
                                face_confirmations[tid].clear()

            # OSNet Body Re-ID Recovery (for UNKNOWN tracks)
            if not state.locked and tid not in recovered_map and idx % 2 == 0:
                cands = set(KNOWN_STUDENTS) - set(lock_mgr.student_owner.keys())
                if cands and (bbox[2] - bbox[0] >= 20 and bbox[3] - bbox[1] >= 40):
                    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    b_emb = body_extractor.extract(crop)
                    if b_emb is not None:
                        res = body_gallery.top_match(b_emb, candidate_students=cands)
                        if res is not None:
                            b_name, b_score, b_margin = res
                            b_second = b_score - b_margin
                            if b_score >= 0.68 and b_margin >= 0.05:
                                rec_s = lock_mgr.attempt_gap_recovery(tid, fid, b_name, b_score, b_second)
                                if rec_s == b_name:
                                    recovered_map[tid] = b_name
                                    if b_name in removal_pending:
                                        old_sec, old_tid, old_fid = removal_pending.pop(b_name)
                                        downtime = video_sec - old_sec
                                        ev = dict(
                                            student=b_name, old_tid=old_tid, old_removed_sec=old_sec,
                                            new_tid=tid, recovery_sec=video_sec, downtime_sec=downtime,
                                            method="OSNet", score=b_score, margin=b_margin, candidate_pool_size=len(cands)
                                        )
                                        downtime_events.append(ev)
                                        print(f"  [RECOVERY-OSNet] f{fid} ({video_sec:.2f}s): {b_name} on Track {tid} (downtime: {downtime:.2f}s)")

            # Render Annotation Box
            label = lock_mgr.get_label(tid)
            is_locked = state.locked if state is not None else False
            if is_locked:
                color = COLOR_GREEN
                tag = "LOCKED"
            elif tid in recovered_map:
                label = recovered_map[tid]
                color = COLOR_YELLOW
                tag = "RE-ID"
            else:
                label = "UNKNOWN"
                color = COLOR_ORANGE
                tag = None

            draw_bounding_box(frame, bbox, tid, label, color, sub_tag=tag)

        # Render HUD & Write Output Frame
        draw_hud(frame, fid, actual_frames, lock_mgr, len(online_targets))
        writer.write(frame)

        if idx % 50 == 0:
            elapsed = time.time() - t0
            pct = (idx / len(frames_12fps)) * 100
            print(f"  [PROGRESS] frame {fid}/{actual_frames} ({pct:.1f}%) | elapsed: {elapsed:.1f}s | active_owners: {dict(lock_mgr.student_owner)}", flush=True)

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"\n[INFO] Full video processing finished in {elapsed:.1f}s ({len(frames_12fps)/elapsed:.2f} FPS throughput)")

    # Summary Distribution
    downtimes = [ev["downtime_sec"] for ev in downtime_events]
    print("\n" + "=" * 72)
    print("FULL VIDEO 12 FPS RECOVERY & DOWNTIME SUMMARY")
    print("=" * 72)
    print(f"Total Recovery Events: {len(downtime_events)}")
    if downtime_events:
        print(f"  Min Downtime:    {min(downtimes):.2f}s")
        print(f"  Median Downtime: {np.median(downtimes):.2f}s")
        print(f"  Mean Downtime:   {np.mean(downtimes):.2f}s")
        print(f"  Max Downtime:    {max(downtimes):.2f}s")
    print(f"Unresolved Removals: {len(removal_pending)} students {list(removal_pending.keys())}")
    print(f"Annotated Video Saved: {OUTPUT_VIDEO_PATH}")

    # Save JSON summary
    summary = {
        "video_duration_sec": actual_frames / SOURCE_FPS,
        "sampling_fps": TARGET_FPS,
        "total_sampled_frames": len(frames_12fps),
        "downtime_events": downtime_events,
        "distribution": {
            "n": len(downtime_events),
            "min": float(min(downtimes)) if downtimes else None,
            "median": float(np.median(downtimes)) if downtimes else None,
            "mean": float(np.mean(downtimes)) if downtimes else None,
            "max": float(max(downtimes)) if downtimes else None,
        },
        "unresolved_removals": {
            s: {"removed_sec": v[0], "old_tid": v[1], "removed_frame": v[2]} for s, v in removal_pending.items()
        },
    }
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Report JSON Saved: {SUMMARY_JSON_PATH}\n")


if __name__ == "__main__":
    main()
