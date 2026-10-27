"""
run_writi_yolo26pose_2min.py
----------------------------
Executes Pipeline 6 with YOLO26n-pose detector (yolo26n-pose.pt) on 'writi.mp4'
for 2 minutes (3,600 frames @ 30 FPS).

Features of Pipeline 6:
1. Detector: YOLO26n-pose (yolo26n-pose.pt, classes=[0], conf=0.25)
2. Tracker: BYTETracker (track_thresh=0.35, track_buffer=90, match_thresh=0.85)
3. Biometrics: InsightFace SCRFD + ArcFace
4. Geometric Safeguard: 0.0 <= norm_y <= 0.45 and norm_x <= 0.30
5. Dual-Resolution Agreement: Both 320 & 640 must yield score >= 0.380 on same student
6. Option B Escalation: 1.0s real time before single-frame 640 attempt
7. IdentityLockManager: Bijective 1-to-1 ownership, permanent sticky locks
8. Candidate-Gated Body Re-ID: MobileNetV3 + HSV color embeddings
9. Output Video: test_outputs/writi_yolo26pose_2min.mp4
10. Report: test_outputs/writi_yolo26pose_2min_report.json
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, ".")
from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack
from features import FaceFeatureExtractor, OSNetBodyFeatureExtractor, BodyGallery
from enroll_students import enroll_students, KNOWN_STUDENTS
from robust_tracker import RobustTracker, draw_track
from video_processor import UniversalVideoWriter
from identity_lock_manager import FACE_LOCK_THRESHOLD, FACE_AMBIGUITY_MARGIN

def run_writi_2min():
    print("=" * 75, flush=True)
    print("RUNNING PIPELINE 6 ON writi.mp4 WITH YOLO26n-POSE (2 MINUTES)", flush=True)
    print("=" * 75, flush=True)

    video_path = "writi.mp4"
    output_video_path = "test_outputs/writi_yolo26pose_osnet_2min.mp4"
    output_report_path = "test_outputs/writi_yolo26pose_osnet_2min_report.json"
    model_path = "yolo26n-pose.pt"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2 minutes of video
    target_duration_seconds = 120.0
    frames_to_run = min(total_video_frames, int(round(target_duration_seconds * fps)))

    print(f"Video: {video_path} ({w}x{h} @ {fps:.2f} FPS)", flush=True)
    print(f"Running for: {target_duration_seconds:.1f}s ({frames_to_run} frames)", flush=True)

    # 1. Load enrollment galleries
    print("Loading student enrollment galleries...", flush=True)
    face_profiles, body_gallery = enroll_students(verbose=False)
    face_extractor = FaceFeatureExtractor(det_size=(320, 320))
    body_extractor = OSNetBodyFeatureExtractor()

    # 2. Load YOLO26n-pose model
    print(f"Loading detector: {model_path}...", flush=True)
    detector = YOLO(model_path)

    # 3. Initialize Tracker
    tracker_wrapper = RobustTracker(
        face_profiles, body_gallery, None, face_extractor, body_extractor, fps=fps
    )

    BaseTrack.reset_id()
    args = BYTETrackerArgs()
    args.track_thresh = 0.35
    args.track_buffer = 90
    args.match_thresh = 0.85
    args.mot20 = False
    byte_tracker = BYTETracker(args, frame_rate=int(round(fps)))

    # 4. Universal Video Writer
    writer = UniversalVideoWriter(output_video_path, fps=fps, width=w, height=h)
    print(f"Rendering annotated video to: {output_video_path}...", flush=True)

    locks = []
    contradictions = []
    ownership_violations = 0
    candidate_pool_violations = 0

    t0 = time.time()
    for frame_idx in range(frames_to_run):
        ret, frame = cap.read()
        if not ret:
            break

        # A. YOLO26n-pose detection (classes=[0] -> person)
        res = detector(frame, classes=[0], verbose=False, conf=0.25)
        dets = []
        for r in res:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    dets.append([x1, y1, x2, y2, conf])

        det_arr = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)

        # B. BYTETracker update
        online_targets = byte_tracker.update(det_arr, [h, w], [h, w])

        # C. Lifecycle management
        tracked_ids = [int(t.track_id) for t in byte_tracker.tracked_stracks if t.is_activated]
        lost_ids = [int(t.track_id) for t in byte_tracker.lost_stracks]
        removed_ids = [int(t.track_id) for t in byte_tracker.removed_stracks]
        tracker_wrapper.lock_manager.update_lifecycle(tracked_ids, lost_ids, removed_ids, frame_idx)

        # Invariant check
        v_list = tracker_wrapper.lock_manager.validate_ownership()
        if v_list:
            ownership_violations += len(v_list)
            for v in v_list:
                print(f"  [OWNERSHIP VIOLATION] frame {frame_idx}: {v}", flush=True)

        annotated = frame.copy()
        candidate_students = set(KNOWN_STUDENTS) - set(tracker_wrapper.lock_manager.student_owner.keys())

        for target in online_targets:
            tid = int(target.track_id)
            tlwh = target.tlwh
            x1, y1 = int(tlwh[0]), int(tlwh[1])
            x2, y2 = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
            bbox = (x1, y1, x2, y2)

            x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            person_crop = frame[y1c:y2c, x1c:x2c]

            state = tracker_wrapper.lock_manager.update_seen(tid, frame_idx)
            was_locked = state.locked

            # Gap recovery check
            frames_gap = frame_idx - tracker_wrapper._track_last_active.get(tid, frame_idx)
            is_new_appearance = frames_gap >= 90
            if is_new_appearance and tid in tracker_wrapper._last_body_emb and candidate_students:
                b_name, b_score, b_sec = tracker_wrapper.body_gallery.top_match(
                    tracker_wrapper._last_body_emb[tid], candidate_students=candidate_students
                )
                if b_name is not None and b_name != "UNKNOWN":
                    if b_name in tracker_wrapper.lock_manager.student_owner:
                        candidate_pool_violations += 1
                    recovered = tracker_wrapper.lock_manager.attempt_gap_recovery(
                        tid, frame_idx, b_name, b_score, b_sec
                    )
                    if recovered:
                        tracker_wrapper.stats["gap_recoveries"] += 1

            # Face extraction decision
            run_face = False
            use_640 = False
            if not state.locked:
                run_face = True
                use_640 = tracker_wrapper.should_escalate_to_640(tid, frame_idx, state.first_seen_frame, is_locked=False)
                if use_640:
                    tracker_wrapper.stats["escalations_to_640"] += 1
            else:
                if tracker_wrapper.should_check_locked_face(tid, frame_idx):
                    run_face = True

            if run_face:
                if not state.locked:
                    # Pipeline 6 Dual-Resolution Agreement Gate:
                    # 1. First extract at 320x320
                    face_emb_320 = tracker_wrapper.face_extractor.extract_from_crop(
                        frame, bbox, padding=0.25, use_640=False
                    )
                    tracker_wrapper.stats["face_extractions"] += 1
                    if face_emb_320 is not None:
                        f_name_320, f_score_320, f_sec_320 = tracker_wrapper.face_extractor.top_match(
                            face_emb_320, tracker_wrapper.face_profiles
                        )
                        # If candidate passes 320 threshold and margin, verify with 640x640 before lock
                        if f_score_320 >= FACE_LOCK_THRESHOLD and (f_score_320 - f_sec_320) >= FACE_AMBIGUITY_MARGIN:
                            face_emb_640 = tracker_wrapper.face_extractor.extract_from_crop(
                                frame, bbox, padding=0.25, use_640=True
                            )
                            tracker_wrapper.stats["face_extractions"] += 1
                            if face_emb_640 is not None:
                                f_name_640, f_score_640, f_sec_640 = tracker_wrapper.face_extractor.top_match(
                                    face_emb_640, tracker_wrapper.face_profiles
                                )
                            else:
                                f_name_640, f_score_640, f_sec_640 = None, 0.0, 0.0

                            tracker_wrapper.lock_manager.submit_face_match(
                                tid, frame_idx, f_name_320, f_score_320, f_sec_320,
                                name_640=f_name_640, score_640=f_score_640, second_score_640=f_sec_640
                            )
                            if not was_locked and state.locked:
                                tracker_wrapper.stats["face_locks"] += 1
                                lock_record = {
                                    "frame": frame_idx,
                                    "timestamp": round(frame_idx / fps, 2),
                                    "tid": tid,
                                    "name": state.locked_name,
                                    "score_320": round(float(f_score_320), 3),
                                    "score_640": round(float(f_score_640), 3),
                                }
                                locks.append(lock_record)
                                print(f"  --> LOCK: Frame {frame_idx:4d} (t={lock_record['timestamp']:6.2f}s) | Track {tid:2d} -> {state.locked_name} (320={f_score_320:.3f}, 640={f_score_640:.3f})", flush=True)
                        else:
                            tracker_wrapper.lock_manager.submit_face_match(
                                tid, frame_idx, f_name_320, f_score_320, f_sec_320
                            )
                    else:
                        if use_640:
                            face_emb_640 = tracker_wrapper.face_extractor.extract_from_crop(
                                frame, bbox, padding=0.25, use_640=True
                            )
                            tracker_wrapper.stats["face_extractions"] += 640
                            if face_emb_640 is not None:
                                f_name_640, f_score_640, f_sec_640 = tracker_wrapper.face_extractor.top_match(
                                    face_emb_640, tracker_wrapper.face_profiles
                                )
                                tracker_wrapper.lock_manager.submit_face_match(
                                    tid, frame_idx, f_name_640, 0.0, 0.0,
                                    name_640=f_name_640, score_640=f_score_640, second_score_640=f_sec_640
                                )
                else:
                    # Contradiction check on locked tracks (throttled ~1.0s, 320 only)
                    face_emb = tracker_wrapper.face_extractor.extract_from_crop(
                        frame, bbox, padding=0.25, use_640=False
                    )
                    tracker_wrapper.stats["face_extractions"] += 1
                    if face_emb is not None:
                        f_name, f_score, f_sec = tracker_wrapper.face_extractor.top_match(
                            face_emb, tracker_wrapper.face_profiles
                        )
                        if f_score >= FACE_LOCK_THRESHOLD and (f_score - f_sec) >= FACE_AMBIGUITY_MARGIN and f_name != state.locked_name:
                            contradictions.append({
                                "frame": frame_idx,
                                "tid": tid,
                                "locked": state.locked_name,
                                "candidate": f_name,
                                "score": round(float(f_score), 3),
                            })
                        tracker_wrapper.lock_manager.submit_face_match(
                            tid, frame_idx, f_name, f_score, f_sec
                        )
            else:
                tracker_wrapper.stats["face_extractions_skipped"] += 1

            label = state.locked_name if state.locked else "UNKNOWN"

            # Body Re-ID (candidate pool gated)
            run_body_extract = False
            if person_crop.size > 0 and (x2c - x1c) > 20 and (y2c - y1c) > 40:
                if state.locked:
                    last_b = tracker_wrapper._last_locked_body_emb_frame.get(tid, -99999)
                    if tid not in tracker_wrapper._last_body_emb or (frame_idx - last_b) >= tracker_wrapper.locked_face_check_interval:
                        run_body_extract = True
                else:
                    if len(candidate_students) > 0:
                        run_body_extract = True

                if run_body_extract:
                    body_emb = tracker_wrapper.body_extractor.extract(person_crop)
                    if body_emb is not None:
                        tracker_wrapper._last_body_emb[tid] = body_emb
                        cand = {state.locked_name} if state.locked else candidate_students
                        b_name, b_score, b_sec = tracker_wrapper.body_gallery.top_match(body_emb, candidate_students=cand)
                        if b_name is not None and b_name != "UNKNOWN":
                            tracker_wrapper.lock_manager.submit_body_evidence(tid, frame_idx, b_name, b_score)
                        tracker_wrapper.stats["body_extractions"] += 1
                else:
                    tracker_wrapper.stats["body_extractions_skipped"] += 1

            tracker_wrapper._track_last_active[tid] = frame_idx

            # Draw track on frame
            annotated = draw_track(annotated, bbox, tid, label)

        # Draw HUD
        hud_line1 = f"Frame {frame_idx}/{frames_to_run} | FPS: {fps:.1f} | Active Tracks: {len(online_targets)} | YOLO26n-pose"
        hud_line2 = f"Owners: {tracker_wrapper.lock_manager.student_owner} | Violations: {ownership_violations}"
        cv2.putText(annotated, hud_line1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(annotated, hud_line1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.putText(annotated, hud_line2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        cv2.putText(annotated, hud_line2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)

        writer.write(annotated)

        if (frame_idx + 1) % 300 == 0:
            elapsed = time.time() - t0
            cur_fps = (frame_idx + 1) / max(0.1, elapsed)
            print(f"  Frame {frame_idx + 1}/{frames_to_run} ({(frame_idx+1)/frames_to_run*100:.1f}%) | Throughput: {cur_fps:.1f} FPS | Locks: {len(locks)}", flush=True)

    cap.release()
    writer.release()

    runtime = time.time() - t0
    effective_fps = frames_to_run / max(0.1, runtime)

    report = {
        "video": video_path,
        "detector": model_path,
        "duration_seconds": round(frames_to_run / fps, 2),
        "total_frames": frames_to_run,
        "runtime_seconds": round(runtime, 2),
        "effective_fps": round(effective_fps, 2),
        "video_fps": fps,
        "confirmed_locks": locks,
        "total_locks": len(locks),
        "contradictions": contradictions,
        "ownership_violations": ownership_violations,
        "candidate_pool_violations": candidate_pool_violations,
        "stats": dict(tracker_wrapper.stats),
        "final_ownership": dict(tracker_wrapper.lock_manager.student_owner),
        "output_video": output_video_path,
    }

    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 75, flush=True)
    print("RUN COMPLETE: writi.mp4 (2 MINUTES) WITH YOLO26n-POSE", flush=True)
    print("=" * 75, flush=True)
    print(f"Processed {frames_to_run} frames in {runtime:.1f}s ({effective_fps:.1f} FPS)")
    print(f"Confirmed Locks ({len(locks)}):")
    for lk in locks:
        print(f"  Frame {lk['frame']:4d} ({lk['timestamp']:6.2f}s) | Track {lk['tid']:2d} -> {lk['name']} (320={lk['score_320']}, 640={lk['score_640']})")
    print(f"Ownership Violations: {ownership_violations}")
    print(f"Candidate Pool Violations: {candidate_pool_violations}")
    print(f"Final Ownership: {tracker_wrapper.lock_manager.student_owner}")
    print(f"Output Video: {output_video_path}")
    print("=" * 75, flush=True)

if __name__ == "__main__":
    run_writi_2min()
