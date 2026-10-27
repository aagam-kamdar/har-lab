"""
run_pipeline_writi.py
---------------------
Executes the finalized production pipeline on 'writi.mp4' (7,485 frames @ 30 FPS):
1. Online YOLOv8/26n person detection (conf=0.25) + caching to disk.
2. Official BYTETracker association.
3. Dual-resolution initial lock agreement (320 & 640 both >= 0.38 for same student).
4. Geometric face-inside-bbox filter (0.0 <= norm_y <= 0.45).
5. Compute reduction:
   - Throttled contradiction checks on locked tracks (~1.0s, 320 only)
   - Candidate pool gating before body Re-ID
6. Renders browser-playable 30 FPS MP4: test_outputs/writi_annotated.mp4
7. Saves full report to: test_outputs/writi_report.json
"""

import os
import sys
import time
import json
import logging
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, ".")
from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack
from features import FaceFeatureExtractor, BodyFeatureExtractor, BodyGallery
from enroll_students import enroll_students, KNOWN_STUDENTS
from robust_tracker import RobustTracker, draw_track
from video_processor import UniversalVideoWriter
from identity_lock_manager import FACE_LOCK_THRESHOLD, FACE_AMBIGUITY_MARGIN

def run_writi():
    print("=" * 70, flush=True)
    print("RUNNING FINALIZED PIPELINE ON writi.mp4", flush=True)
    print("=" * 70, flush=True)

    video_path = "writi.mp4"
    output_video_path = "test_outputs/writi_annotated.mp4"
    output_report_path = "test_outputs/writi_report.json"
    cached_dets_path = "test_outputs/cached_writi_yolo26n_dets.json"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path} ({w}x{h} @ {fps:.2f} FPS, {total_frames} frames)", flush=True)

    # 1. Load enrollment profiles
    print("Loading student enrollment galleries...", flush=True)
    face_profiles, body_gallery = enroll_students(verbose=False)
    face_extractor = FaceFeatureExtractor(det_size=(320, 320))
    body_extractor = BodyFeatureExtractor()

    # 2. Initialize YOLO detector
    print("Loading YOLO detector (yolo26n.pt)...", flush=True)
    yolo_model = YOLO("yolo26n.pt")

    # Check if cached detections already exist
    cached_dets = {}
    use_existing_cache = False
    if os.path.exists(cached_dets_path):
        try:
            with open(cached_dets_path, "r") as f:
                cached_dets = json.load(f)
            if len(cached_dets) >= total_frames:
                print(f"Loaded existing cached detections from {cached_dets_path} ({len(cached_dets)} frames).", flush=True)
                use_existing_cache = True
        except Exception as e:
            print(f"Could not load existing cache: {e}. Will compute online.", flush=True)

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
    print(f"Rendering annotated video to {output_video_path}...", flush=True)

    locks = []
    contradictions = []
    ownership_violations = 0
    candidate_pool_violations = 0
    initial_gallery_sizes = dict(body_gallery.sizes())

    t0 = time.time()
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # A. Detections
        if use_existing_cache:
            dets = cached_dets.get(str(frame_idx), cached_dets.get(frame_idx, []))
        else:
            res = yolo_model(frame, classes=[0], verbose=False, conf=0.25)
            dets = []
            for r in res:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        dets.append([x1, y1, x2, y2, conf])
            cached_dets[str(frame_idx)] = dets

        det_arr = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)

        # B. BYTETracker update
        online_targets = byte_tracker.update(det_arr, [h, w], [h, w])

        # C. Lifecycle
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
            x1 = int(tlwh[0])
            y1 = int(tlwh[1])
            x2 = int(tlwh[0] + tlwh[2])
            y2 = int(tlwh[1] + tlwh[3])
            bbox = (x1, y1, x2, y2)

            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w, x2)
            y2c = min(h, y2)
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

            # Face extraction
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
                    # Initial lock requires dual 320 and 640 agreement
                    face_emb_320 = tracker_wrapper.face_extractor.extract_from_crop(
                        frame, bbox, padding=0.25, use_640=False
                    )
                    tracker_wrapper.stats["face_extractions"] += 1
                    if face_emb_320 is not None:
                        f_name_320, f_score_320, f_sec_320 = tracker_wrapper.face_extractor.top_match(
                            face_emb_320, tracker_wrapper.face_profiles
                        )
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
                            tracker_wrapper.stats["face_extractions"] += 1
                            if face_emb_640 is not None:
                                f_name_640, f_score_640, f_sec_640 = tracker_wrapper.face_extractor.top_match(
                                    face_emb_640, tracker_wrapper.face_profiles
                                )
                                tracker_wrapper.lock_manager.submit_face_match(
                                    tid, frame_idx, f_name_640, 0.0, 0.0,
                                    name_640=f_name_640, score_640=f_score_640, second_score_640=f_sec_640
                                )
                else:
                    # Contradiction check on locked track (throttled, 320 only)
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

            # Body Re-ID
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

            # Draw track
            annotated = draw_track(annotated, bbox, tid, label)

        # Draw HUD
        hud_line1 = f"Frame {frame_idx}/{total_frames} | FPS: {fps:.1f} | Active Tracks: {len(online_targets)}"
        hud_line2 = f"Owners: {tracker_wrapper.lock_manager.student_owner} | Violations: {ownership_violations}"
        cv2.putText(annotated, hud_line1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(annotated, hud_line1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.putText(annotated, hud_line2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        cv2.putText(annotated, hud_line2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)

        writer.write(annotated)

        if (frame_idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            cur_fps = (frame_idx + 1) / max(0.1, elapsed)
            print(f"  Frame {frame_idx + 1}/{total_frames} ({(frame_idx+1)/total_frames*100:.1f}%) | Throughput: {cur_fps:.1f} FPS | Locks: {len(locks)}", flush=True)

    cap.release()
    writer.release()

    runtime = time.time() - t0
    effective_fps = total_frames / max(0.1, runtime)

    # Save cached detections if computed online
    if not use_existing_cache:
        print(f"Saving cached detections to {cached_dets_path}...", flush=True)
        with open(cached_dets_path, "w") as f:
            json.dump(cached_dets, f)

    final_gallery_sizes = dict(body_gallery.sizes())
    gallery_mutations = sum(abs(final_gallery_sizes.get(k, 0) - initial_gallery_sizes.get(k, 0)) for k in initial_gallery_sizes)

    report = {
        "video": video_path,
        "total_frames": total_frames,
        "runtime_seconds": round(runtime, 2),
        "effective_fps": round(effective_fps, 2),
        "video_fps": fps,
        "confirmed_locks": locks,
        "total_locks": len(locks),
        "contradictions": contradictions,
        "ownership_violations": ownership_violations,
        "candidate_pool_violations": candidate_pool_violations,
        "gallery_mutations": gallery_mutations,
        "stats": dict(tracker_wrapper.stats),
        "final_ownership": dict(tracker_wrapper.lock_manager.student_owner),
        "output_video": output_video_path,
    }

    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("RUN COMPLETE ON writi.mp4", flush=True)
    print("=" * 70, flush=True)
    print(f"Total Frames: {total_frames} in {runtime:.1f}s ({effective_fps:.1f} FPS)")
    print(f"Confirmed Locks ({len(locks)}):")
    for lk in locks:
        print(f"  Frame {lk['frame']:4d} ({lk['timestamp']:6.2f}s) | Track {lk['tid']:2d} -> {lk['name']} (320={lk['score_320']}, 640={lk['score_640']})")
    print(f"Ownership Violations: {ownership_violations}")
    print(f"Candidate Pool Violations: {candidate_pool_violations}")
    print(f"Final Ownership: {tracker_wrapper.lock_manager.student_owner}")
    print(f"Output Video: {output_video_path}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_writi()
