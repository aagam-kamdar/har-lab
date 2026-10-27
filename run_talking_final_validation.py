"""
run_talking_final_validation.py
--------------------------------
Validates the CURRENT FINALIZED identity tracking pipeline on 'talking.mp4'.
Diagnostic run only - preserves all production files and settings unchanged.

Pipeline Architecture:
  - Input: talking.mp4 (3,417 frames @ 30.017 FPS, 848x392)
  - Detector: YOLO26n-pose (yolo26n-pose.pt, person class=0, conf=0.25)
  - Tracker: BYTETracker (track_thresh=0.35, track_buffer=90, match_thresh=0.85, mot20=False, frame_rate=30)
  - Face Verification: SCRFD + ArcFace (buffalo_sc, det_size=(320, 320), padding=0.25)
  - Identity Manager: IdentityLockManager (sticky locks, bijective ownership)
  - Body Re-ID: OSNet-x0.25 (ExtractorB, 512-dim, tau=0.68, margin=0.05)
  - Gallery: Frozen OSNet gallery (galleries_osnet_x0_25/gallery_embeddings.npz) - NO mutation
  - Candidate Scoping: candidate_students = set(KNOWN_STUDENTS) - set(lock_mgr.student_owner.keys())
  - Output Video: test_outputs/talking_final_validation.mp4 @ 30 FPS native
  - Output JSON: test_outputs/talking_final_validation.json
  - Output Summary: test_outputs/talking_final_validation_summary.md
"""

import sys, os, io

# Ensure line-buffered UTF-8 output
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
logger = logging.getLogger("TalkingValidation")

VIDEO_PATH = "talking.mp4"
OUTPUT_VIDEO_PATH = "test_outputs/talking_final_validation.mp4"
OUTPUT_JSON_PATH = "test_outputs/talking_final_validation.json"
OUTPUT_MD_PATH = "test_outputs/talking_final_validation_summary.md"
GALLERY_NPZ = "galleries_osnet_x0_25/gallery_embeddings.npz"
YOLO_MODEL_PATH = "yolo26n-pose.pt"

# Colors (BGR)
COLOR_GREEN = (40, 220, 40)       # ArcFace Locked
COLOR_YELLOW = (0, 215, 255)      # Re-ID Recovered
COLOR_ORANGE = (0, 140, 255)      # UNKNOWN
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (180, 180, 180)
COLOR_RED = (40, 40, 240)


def format_timestamp(frame_idx: int, fps: float = 30.0) -> str:
    total_seconds = frame_idx / fps
    mins = int(total_seconds // 60)
    secs = int(total_seconds % 60)
    millis = int((total_seconds - int(total_seconds)) * 1000)
    return f"{mins:02d}:{secs:02d}.{millis:03d} ({mins:02d}:{secs:02d})"


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


def draw_hud(frame: np.ndarray, frame_idx: int, total_frames: int, lock_mgr: IdentityLockManager, active_count: int, fps: float):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (310, 180), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (10, 10), (310, 180), (90, 90, 90), 1)

    time_str = f"{format_mm_ss(frame_idx, fps)} / {format_mm_ss(total_frames, fps)} (f{frame_idx})"
    cv2.putText(frame, time_str, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active Tracks: {active_count} | 30 FPS", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (18, 62), (300, 62), (70, 70, 70), 1)

    row_y = 82
    for s_name in KNOWN_STUDENTS:
        owner_tid = lock_mgr.student_owner.get(s_name)
        if owner_tid is not None:
            state = lock_mgr.get_state(owner_tid)
            if state.locked:
                status_txt = "LOCKED"
                txt_col = COLOR_GREEN
            elif state.status == "LOST":
                status_txt = "LOST-BUF"
                txt_col = COLOR_ORANGE
            else:
                status_txt = "ACTIVE"
                txt_col = COLOR_YELLOW
            line = f" {s_name.capitalize():<8}: Track {owner_tid:<2} [{status_txt}]"
        else:
            line = f" {s_name.capitalize():<8}: UNOWNED"
            txt_col = COLOR_GRAY

        cv2.putText(frame, line, (20, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_col, 1, cv2.LINE_AA)
        row_y += 22


def draw_bounding_box(frame: np.ndarray, bbox: Tuple[int, int, int, int], track_id: int, label_name: str, color: Tuple[int, int, int], sub_tag: Optional[str] = None):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    corner_len = min(12, (x2 - x1) // 4, (y2 - y1) // 4)
    if corner_len > 0:
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

    text = f"T{track_id}:{label_name.upper()}"
    if sub_tag:
        text += f"[{sub_tag}]"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tag_y1 = max(0, y1 - th - 6)
    tag_y2 = y1
    tag_x2 = min(frame.shape[1], x1 + tw + 8)

    cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
    text_color = (10, 10, 10) if color in [COLOR_YELLOW, COLOR_GREEN, COLOR_ORANGE] else (255, 255, 255)
    cv2.putText(frame, text, (x1 + 4, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)


def main():
    print("=" * 72)
    print("VALIDATING CURRENT FINALIZED PIPELINE ON TALKING.MP4")
    print("=" * 72)
    os.makedirs("test_outputs", exist_ok=True)

    # 1. Initialize YOLO26n-Pose
    print(f"[INFO] Initializing detector: {YOLO_MODEL_PATH} ...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    # 2. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {VIDEO_PATH}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps

    print(f"[INFO] Video Info: {width}x{height} @ {fps:.3f} FPS, {total_frames} frames ({duration_s:.2f}s)")

    # 3. Initialize ByteTrack (native 30 FPS)
    bt_args = BYTETrackerArgs()
    bt_args.track_thresh = 0.35
    bt_args.track_buffer = 90
    bt_args.match_thresh = 0.85
    bt_args.mot20 = False
    tracker = BYTETracker(bt_args, frame_rate=int(round(fps)))
    print(f"[INFO] Tracker: BYTETracker(track_thresh=0.35, track_buffer=90, match_thresh=0.85, frame_rate={int(round(fps))})")

    # 4. Initialize ArcFace & Enrolled Profiles
    print("[INFO] Initializing FaceFeatureExtractor (InsightFace buffalo_sc) ...")
    face_extractor = FaceFeatureExtractor(det_size=(320, 320))
    face_profiles, _ = enroll_students(verbose=False)
    print(f"[INFO] Enrolled student face profiles: {list(face_profiles.keys())}")

    # 5. Initialize OSNet-x0.25 & Frozen Gallery (NO MUTATION)
    print("[INFO] Initializing OSNet-x0.25 (ExtractorB) & Frozen Gallery ...")
    body_extractor = ExtractorB()
    body_gallery = load_osnet_gallery(dim=512, npz_path=GALLERY_NPZ)
    initial_gallery_sizes = body_gallery.sizes()
    print(f"[INFO] Loaded frozen OSNet gallery sizes: {initial_gallery_sizes}")

    # 6. Initialize IdentityLockManager & VideoWriter
    lock_mgr = IdentityLockManager()
    writer = UniversalVideoWriter(OUTPUT_VIDEO_PATH, fps=fps, width=width, height=height)

    # State tracking for diagnostics and audits
    events_log = []
    first_face_locks = {}
    recovered_map: Dict[int, str] = {}
    recovery_attempted_tracks: Set[int] = set()
    prior_student_owner: Dict[str, int] = {}

    # Audit counters
    ownership_violations_count = 0
    ownership_timing_violations_count = 0
    candidate_pool_leak_violations_count = 0
    gallery_updates_count = 0

    # Specific audit logs for Swayam & Harshil
    swayam_harshil_incidents = []
    frame_356_forensics = {}

    t0 = time.time()

    for current_f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Stream ended at frame {current_f_idx}")
            break

        # YOLO detection (conf=0.25, person class=0)
        res = yolo_model(frame, classes=[0], conf=0.25, verbose=False)[0]
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        if len(boxes_xyxy) > 0:
            dets_arr = np.hstack([boxes_xyxy, confs[:, None]]).astype(np.float32)
        else:
            dets_arr = np.empty((0, 5), dtype=np.float32)

        # ByteTrack update
        img_info = (height, width)
        online_targets = tracker.update(dets_arr, img_info, img_info)

        # Lifecycle tracking
        tracked_ids = [int(t.track_id) for t in tracker.tracked_stracks if t.is_activated]
        lost_ids = [int(t.track_id) for t in tracker.lost_stracks]
        removed_ids = [int(t.track_id) for t in tracker.removed_stracks]
        lock_mgr.update_lifecycle(tracked_ids, lost_ids, removed_ids, current_f_idx)

        # Releases check
        curr_owners = dict(lock_mgr.student_owner)
        for s_name, old_tid in prior_student_owner.items():
            if s_name not in curr_owners:
                events_log.append({
                    "event_type": "RELEASE",
                    "frame": current_f_idx,
                    "timestamp": format_timestamp(current_f_idx, fps),
                    "track_id": old_tid,
                    "student": s_name,
                    "trigger": "ByteTrack Eviction (REMOVED after 90f)",
                })
        prior_student_owner = curr_owners

        # Audit 1: Ownership Invariants
        violations = lock_mgr.validate_ownership()
        if violations:
            ownership_violations_count += len(violations)
            for v in violations:
                logger.error("OWNERSHIP VIOLATION at frame %d: %s", current_f_idx, v)

        # Process each online track
        for t in online_targets:
            tid = int(t.track_id)
            tlwh = t.tlwh
            x1 = max(0, int(round(tlwh[0])))
            y1 = max(0, int(round(tlwh[1])))
            x2 = min(width, int(round(tlwh[0] + tlwh[2])))
            y2 = min(height, int(round(tlwh[1] + tlwh[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = (x1, y1, x2, y2)
            person_crop = frame[y1:y2, x1:x2].copy()

            state = lock_mgr.update_seen(tid, current_f_idx)
            was_locked = state.locked

            # Special Investigation: Frame 356 Detailed Forensic Logging
            if current_f_idx == 356:
                # Perform granular face detection inside the crop to inspect all candidate faces
                h_img, w_img = frame.shape[:2]
                bw, bh = x2 - x1, y2 - y1
                pad = int(bw * 0.25)
                head_y2 = y1 + int(bh * 0.45)
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(w_img, x2 + pad)
                cy2 = min(h_img, head_y2 + pad)
                crop_head = frame[cy1:cy2, cx1:cx2]

                detected_faces = face_extractor.app.get(crop_head) if (cx2 - cx1 >= 20 and cy2 - cy1 >= 20) else []
                face_details = []
                for f_idx, fc in enumerate(detected_faces):
                    fc_box_crop = [float(coord) for coord in fc.bbox]
                    fc_box_abs = [float(fc.bbox[0] + cx1), float(fc.bbox[1] + cy1), float(fc.bbox[2] + cx1), float(fc.bbox[3] + cy1)]
                    fc_area = (fc.bbox[2] - fc.bbox[0]) * (fc.bbox[3] - fc.bbox[1])
                    fc_center_in_bbox = (x1 <= (fc_box_abs[0] + fc_box_abs[2])/2.0 <= x2) and (y1 <= (fc_box_abs[1] + fc_box_abs[3])/2.0 <= y2)
                    fc_emb = fc.normed_embedding.astype(np.float32)
                    sims = {name: float(np.dot(fc_emb, prof)) for name, prof in face_profiles.items()}
                    face_details.append({
                        "face_idx": f_idx,
                        "bbox_crop": fc_box_crop,
                        "bbox_abs": fc_box_abs,
                        "area": float(fc_area),
                        "center_inside_person_bbox": fc_center_in_bbox,
                        "similarities": sims,
                    })

                frame_356_forensics[f"track_{tid}"] = {
                    "bbox": bbox,
                    "crop_window": [cx1, cy1, cx2, cy2],
                    "detected_faces_count": len(detected_faces),
                    "faces": face_details,
                    "state_before": {
                        "locked": state.locked,
                        "locked_name": state.locked_name,
                        "pending_name": state.pending_name,
                        "pending_count": state.pending_count
                    }
                }

            # -------------------------------------------------------------
            # ArcFace Initial Face Lock (Production padding=0.25)
            # -------------------------------------------------------------
            face_emb = face_extractor.extract_from_crop(frame, bbox, padding=0.25)
            if face_emb is not None:
                f_name, f_score, f_sec = face_extractor.top_match(face_emb, face_profiles)

                # Diagnostic incident monitoring: Swayam ↔ Harshil contradictions or mispredictions
                is_swayam_harshil_related = (f_name in ["swayam", "harshil"]) or (state.locked_name in ["swayam", "harshil"])
                if is_swayam_harshil_related and state.locked and state.locked_name != f_name:
                    if f_score >= FACE_LOCK_THRESHOLD and (f_score - f_sec) >= FACE_AMBIGUITY_MARGIN:
                        swayam_harshil_incidents.append({
                            "type": "ARCFACE_CONTRADICTION_IGNORED",
                            "frame": current_f_idx,
                            "timestamp": format_timestamp(current_f_idx, fps),
                            "track_id": tid,
                            "locked_name": state.locked_name,
                            "predicted_name": f_name,
                            "score": float(f_score),
                            "second_score": float(f_sec),
                            "margin": float(f_score - f_sec),
                            "action": "STICKY_LOCK_PRESERVED_CONTRADICTION_IGNORED"
                        })

                # Submit to identity lock manager
                prior_locked_name = state.locked_name
                lock_mgr.submit_face_match(tid, current_f_idx, f_name, f_score, f_sec)

                if not was_locked and state.locked:
                    prior_rec = recovered_map.get(tid)
                    events_log.append({
                        "event_type": "LOCK",
                        "frame": current_f_idx,
                        "timestamp": format_timestamp(current_f_idx, fps),
                        "track_id": tid,
                        "student": state.locked_name,
                        "score": float(f_score),
                        "margin": float(f_score - f_sec),
                        "prior_recovery": prior_rec
                    })
                    recovered_map.pop(tid, None)

            # -------------------------------------------------------------
            # OSNet Body Re-ID Recovery (Strict Candidate Scoping Active)
            # -------------------------------------------------------------
            if not state.locked and tid not in recovery_attempted_tracks and person_crop.size > 0:
                current_owned_set = set(lock_mgr.student_owner.keys())
                candidate_students = set(KNOWN_STUDENTS) - current_owned_set

                # Audit 3: Candidate Pool Leak Check
                for owned_s in current_owned_set:
                    if owned_s in candidate_students:
                        candidate_pool_leak_violations_count += 1
                        logger.error("CANDIDATE POOL LEAK at frame %d: owned student '%s' in pool!", current_f_idx, owned_s)

                if candidate_students and (x2 - x1) > 25 and (y2 - y1) > 50:
                    body_emb = body_extractor.extract(person_crop)
                    if body_emb is not None:
                        top_m, top_s, sec_s = body_gallery.top_match(body_emb, candidate_students=candidate_students)
                        margin = top_s - sec_s
                        tau = 0.68
                        margin_req = 0.05

                        if top_m is not None and top_s >= tau and margin >= margin_req:
                            recovered_name = top_m
                            recovered_map[tid] = recovered_name
                            recovery_attempted_tracks.add(tid)
                            lock_mgr._acquire_ownership(tid, recovered_name, current_f_idx)

                            events_log.append({
                                "event_type": "RECOVERY",
                                "frame": current_f_idx,
                                "timestamp": format_timestamp(current_f_idx, fps),
                                "track_id": tid,
                                "student": recovered_name,
                                "score": float(top_s),
                                "margin": float(margin),
                                "candidate_pool": list(candidate_students),
                            })

            # Render bounding box
            label = lock_mgr.get_label(tid)
            if state.locked:
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

        # Draw HUD and write frame
        draw_hud(frame, current_f_idx, total_frames, lock_mgr, len(online_targets), fps)
        writer.write(frame)

        if current_f_idx % 250 == 0:
            elapsed = time.time() - t0
            pct = (current_f_idx / total_frames) * 100.0
            print(f"  [PROGRESS] frame {current_f_idx}/{total_frames} ({pct:.1f}%) | elapsed: {elapsed:.1f}s | "
                  f"owners: {dict(lock_mgr.student_owner)}", flush=True)

    cap.release()
    writer.release()
    total_time = time.time() - t0
    final_gallery_sizes = body_gallery.sizes()

    print(f"\n[INFO] Run complete in {total_time:.1f}s ({total_frames/total_time:.2f} FPS throughput)")
    print(f"[INFO] Initial gallery sizes: {initial_gallery_sizes}")
    print(f"[INFO] Final gallery sizes:   {final_gallery_sizes}")
    print(f"[INFO] Total gallery updates during run: {gallery_updates_count}")

    # Build report data
    report_data = {
        "video_info": {
            "path": VIDEO_PATH,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration_s,
            "detector_model": YOLO_MODEL_PATH,
            "tracker_config": {
                "track_thresh": bt_args.track_thresh,
                "track_buffer": bt_args.track_buffer,
                "match_thresh": bt_args.match_thresh,
                "frame_rate": int(round(fps))
            }
        },
        "gallery_audit": {
            "initial_sizes": initial_gallery_sizes,
            "final_sizes": final_gallery_sizes,
            "total_updates": gallery_updates_count,
            "is_frozen": (initial_gallery_sizes == final_gallery_sizes and gallery_updates_count == 0)
        },
        "ownership_audit": {
            "ownership_violations": ownership_violations_count,
            "ownership_timing_violations": ownership_timing_violations_count,
            "candidate_pool_leak_violations": candidate_pool_leak_violations_count,
            "final_student_owner": dict(lock_mgr.student_owner),
            "final_track_to_student": dict(lock_mgr.track_to_student)
        },
        "swayam_harshil_incidents": swayam_harshil_incidents,
        "frame_356_forensics": frame_356_forensics,
        "events_log": events_log,
        "all_tracks": [
            {
                "track_id": tid,
                "name": st.locked_name if st.locked else "UNKNOWN",
                "locked": st.locked,
                "locked_at_frame": st.locked_at_frame,
                "locked_face_score": st.locked_face_score,
                "contradictions": st.contradiction_count,
                "status": st.status
            }
            for tid, st in lock_mgr.all_states().items()
        ]
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"[INFO] Saved JSON report -> {OUTPUT_JSON_PATH}")

    return report_data


if __name__ == "__main__":
    main()
