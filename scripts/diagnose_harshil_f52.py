"""
scripts/diagnose_harshil_f52.py
-------------------------------
Diagnoses Harshil lock at frame 52 vs frame 960 by running:
  1. The ORIGINAL script logic (run_full_video_verification.py with YOLO26n-pose cached detections)
  2. The Adapter logic (identity_adapter.py with YOLO26n detection)
on frames 0-100 of 'new lab video.mp4' for the D03 track (Harshil's desk: x ~ 640-780).
Dumps per frame:
  - Person bbox
  - Detector name & version
  - Face detection success & crop bbox
  - ArcFace score & margin
  - Lock status & trigger
"""

import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bytetrack import BYTETracker, BYTETrackerArgs
from features import FaceFeatureExtractor, BodyGallery
from run_final_reid_similarity_benchmark import ExtractorB
from enroll_students import enroll_students, KNOWN_STUDENTS
from identity_lock_manager import IdentityLockManager
from identity_adapter import IdentityAdapter

VIDEO_PATH = "new lab video.mp4"
CACHED_POSE_JSON = "test_outputs/cached_yolo26pose_detections_1800f.json"


def run_original_f0_100():
    print("=" * 80)
    print("RUNNING ORIGINAL SCRIPT PIPELINE (YOLO26n-pose + IdentityLockManager) on frames 0-100")
    print("=" * 80)

    with open(CACHED_POSE_JSON, "r") as f:
        pose_data = json.load(f)
    pose_dets_by_frame = {d["frame_idx"]: np.array(d["boxes"], dtype=np.float32) for d in pose_data["detections"] if d["frame_idx"] <= 100}

    bt_args = BYTETrackerArgs()
    bt_args.track_thresh = 0.35
    bt_args.track_buffer = 90
    bt_args.match_thresh = 0.85
    bt_args.mot20 = False
    tracker = BYTETracker(bt_args, frame_rate=30)

    face_extractor = FaceFeatureExtractor(det_size=(320, 320))
    face_profiles, _ = enroll_students(verbose=False)
    body_extractor = ExtractorB()
    body_gallery = BodyGallery(dim=512)
    gal_npz = "galleries_osnet_x0_25/gallery_embeddings.npz"
    if os.path.exists(gal_npz):
        data = np.load(gal_npz)
        for s in KNOWN_STUDENTS:
            if s in data:
                for vec in data[s]:
                    body_gallery.add(s, vec)

    lock_mgr = IdentityLockManager(require_dual_agreement=False)
    recovered_map = {}

    cap = cv2.VideoCapture(VIDEO_PATH)
    original_d03_log = []

    for f_idx in range(101):
        ok, frame = cap.read()
        if not ok:
            break

        det_boxes = pose_dets_by_frame.get(f_idx, np.empty((0, 5), dtype=np.float32))
        h, w = frame.shape[:2]
        online_targets = tracker.update(det_boxes, [h, w], [h, w])

        tracked_ids = [int(t.track_id) for t in tracker.tracked_stracks]
        lost_ids = [int(t.track_id) for t in tracker.lost_stracks]
        removed_ids = [int(t.track_id) for t in tracker.removed_stracks]
        lock_mgr.update_lifecycle(tracked_ids, lost_ids, removed_ids, current_frame=f_idx)

        for target in online_targets:
            tid = int(target.track_id)
            tlwh = target.tlwh
            x1, y1 = max(0, int(tlwh[0])), max(0, int(tlwh[1]))
            x2, y2 = min(w, int(tlwh[0] + tlwh[2])), min(h, int(tlwh[1] + tlwh[3]))
            bbox = (x1, y1, x2, y2)
            person_crop = frame[y1:y2, x1:x2].copy()

            # D03 desk zone: x in [630, 800]
            cx = (x1 + x2) / 2.0
            is_d03 = (630 <= cx <= 800)

            state = lock_mgr.update_seen(tid, f_idx)
            was_locked = state.locked if state else False

            # ArcFace Initial Face Lock
            face_emb = face_extractor.extract_from_crop(frame, bbox, padding=0.25)
            f_match, f_score, f_sec = (None, 0.0, 0.0)
            if face_emb is not None:
                f_match, f_score, f_sec = face_extractor.top_match(face_emb, face_profiles)
                lock_mgr.submit_face_match(tid, f_idx, f_match, f_score, f_sec)

            # OSNet Body Re-ID Recovery (if unowned) - exact run_full_video_verification logic
            current_owned_set = set(lock_mgr.student_owner.keys())
            candidate_students = set(KNOWN_STUDENTS) - current_owned_set
            rec_match, rec_score, rec_margin = (None, 0.0, 0.0)
            if not state.locked and candidate_students and (x2 - x1) > 25 and (y2 - y1) > 50:
                body_emb = body_extractor.extract(person_crop)
                if body_emb is not None:
                    top_m, top_s, sec_s = body_gallery.top_match(body_emb, candidate_students=candidate_students)
                    margin = top_s - sec_s
                    rec_match, rec_score, rec_margin = top_m, top_s, margin
                    tau = 0.68
                    margin_req = 0.05
                    if top_m is not None and top_s >= tau and margin >= margin_req:
                        recovered_map[tid] = top_m
                        lock_mgr._acquire_ownership(tid, top_m, f_idx)

            new_state = lock_mgr.get_state(tid)
            is_now_locked = new_state.locked if new_state else False
            locked_name = new_state.locked_name if new_state else None

            if is_d03:
                original_d03_log.append({
                    "frame": f_idx,
                    "pipeline": "ORIGINAL (run_full_video_verification)",
                    "detector": "yolo26n-pose.pt (via cached_yolo26pose_detections_1800f.json)",
                    "track_id": tid,
                    "bbox": [x1, y1, x2, y2],
                    "face_detected": face_emb is not None,
                    "face_top_match": f_match,
                    "face_score": round(float(f_score), 4),
                    "face_margin": round(float(f_score - f_sec), 4),
                    "body_match": rec_match,
                    "body_score": round(float(rec_score), 4),
                    "body_margin": round(float(rec_margin), 4),
                    "locked": is_now_locked,
                    "locked_name": locked_name
                })

    cap.release()
    return original_d03_log


def run_adapter_f0_100():
    print("=" * 80)
    print("RUNNING ADAPTER PIPELINE (YOLO26n + IdentityAdapter) on frames 0-100")
    print("=" * 80)

    adapter = IdentityAdapter(
        fps=30.0,
        yolo_model_path="yolo26n.pt",
        identifying_timeout_s=5.0,
        live_gate_evidence_s=1.5,
        clear_margin=0.15,
        arcface_confirm_s=0.5,
        body_stable_s=3.0,
        mode="redesigned"
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    adapter_d03_log = []

    for f_idx in range(101):
        ok, frame = cap.read()
        if not ok:
            break

        tracks = adapter.update(frame, f_idx)
        for t in tracks:
            tid = t["track_id"]
            bbox = [int(v) for v in t["bbox"]]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            is_d03 = (630 <= cx <= 800)

            if is_d03:
                adapter_d03_log.append({
                    "frame": f_idx,
                    "pipeline": "ADAPTER (identity_adapter.py)",
                    "detector": "yolo26n.pt (live ultralytics detector)",
                    "track_id": tid,
                    "bbox": bbox,
                    "display_name": t["display_name"],
                    "likely_student": t.get("likely_student"),
                    "status": t["status"],
                    "affinity": t.get("affinity", 0.0),
                    "margin": t.get("margin", 0.0),
                    "face_ema": {s: round(adapter.track_face_ema[tid].get(s, 0.0), 3) for s in KNOWN_STUDENTS},
                    "body_ema": {s: round(adapter.track_body_ema[tid].get(s, 0.0), 3) for s in KNOWN_STUDENTS}
                })

    cap.release()
    return adapter_d03_log


def main():
    orig_log = run_original_f0_100()
    adapt_log = run_adapter_f0_100()

    out_file = "test_outputs/harshil_f52_vs_f960_diagnostic.json"
    with open(out_file, "w") as f:
        json.dump({"original_pipeline": orig_log, "adapter_pipeline": adapt_log}, f, indent=2)
    print(f"\n[INFO] Diagnostic dumped to {out_file}")

    print("\n" + "=" * 80)
    print("KEY FRAMES COMPARISON FOR HARSHIL'S DESK D03 (Track 2)")
    print("=" * 80)
    orig_by_f = {e["frame"]: e for e in orig_log}
    adapt_by_f = {e["frame"]: e for e in adapt_log}

    print(f"{'Frame':<6} | {'Orig Bbox':<24} | {'Orig Face':<18} | {'Orig Lock':<15} | {'Adapter Bbox':<24} | {'Adapter Display':<22}")
    print("-" * 115)
    for f in [0, 1, 9, 20, 40, 50, 52, 60, 80, 100]:
        o = orig_by_f.get(f, {})
        a = adapt_by_f.get(f, {})
        o_box = str(o.get("bbox", "N/A"))
        o_face = f"{o.get('face_top_match')}:{o.get('face_score')}" if o.get("face_detected") else "No Face"
        o_lock = f"{o.get('locked_name')} ({'LOCK' if o.get('locked') else 'UNLOCKED'})" if o.get("locked_name") else "None"

        a_box = str(a.get("bbox", "N/A"))
        a_disp = str(a.get("display_name", "N/A"))
        print(f"f{f:<5} | {o_box:<24} | {o_face:<18} | {o_lock:<15} | {a_box:<24} | {a_disp:<22}")


if __name__ == "__main__":
    main()
