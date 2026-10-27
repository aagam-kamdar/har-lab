"""
run_regression_and_render.py
-----------------------------
Executes the full production regression with:
1. Time-based 640x640 escalation (Option B persistence).
2. Compute reduction:
   - Throttled contradiction face checks on locked tracks (~1.0s)
   - Candidate-pool scoped body Re-ID (skip when candidate pool empty)
   - Throttled body extraction on locked tracks
3. Renders browser-playable 30 FPS output MP4 videos:
   - test_outputs/talking_adaptive_640.mp4
   - test_outputs/new_lab_adaptive_640.mp4
4. Validates the strict Regression Gate.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np

sys.path.insert(0, ".")
from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack
from features import FaceFeatureExtractor, BodyFeatureExtractor, OSNetBodyFeatureExtractor, BodyGallery
from enroll_students import enroll_students, KNOWN_STUDENTS
from robust_tracker import RobustTracker, draw_track
from video_processor import UniversalVideoWriter


def run_video_pipeline(
    video_path: str,
    cached_dets_path: str,
    output_video_path: str,
    max_frames: int,
    face_extractor: FaceFeatureExtractor,
    body_extractor: OSNetBodyFeatureExtractor,
    face_profiles: Dict[str, np.ndarray],
    body_gallery: BodyGallery,
) -> Dict:
    print(f"\n{'='*70}\nSTARTING PIPELINE RUN: {video_path}\n{'='*70}", flush=True)
    t0 = time.time()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_run = min(total_video_frames, max_frames)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize Tracker wrapper with actual runtime FPS
    tracker_wrapper = RobustTracker(
        face_profiles,
        body_gallery,
        None,
        face_extractor,
        body_extractor,
        fps=fps,
    )

    BaseTrack.reset_id()
    args = BYTETrackerArgs()
    args.track_thresh = 0.35
    args.track_buffer = 90
    args.match_thresh = 0.85
    args.mot20 = False
    byte_tracker = BYTETracker(args, frame_rate=int(round(fps)))

    # Load cached detections
    with open(cached_dets_path, "r") as f:
        data = json.load(f)
    if "detections" in data:
        cached_dets = {d["frame_idx"]: d["boxes"] for d in data["detections"]}
    else:
        cached_dets = {int(k): v for k, v in data.items()}

    # Initialize video writer
    writer = UniversalVideoWriter(output_video_path, fps=fps, width=w, height=h)
    print(f"Rendering to {output_video_path} ({w}x{h} @ {fps:.1f} FPS, {frames_to_run} frames)...", flush=True)

    locks = []
    contradictions = []
    escalation_events = []
    ownership_violations = 0
    candidate_pool_violations = 0

    initial_gallery_sizes = dict(body_gallery.sizes())

    for frame_idx in range(frames_to_run):
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detections
        dets = cached_dets.get(frame_idx, [])
        det_arr = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)

        # 2. BYTETracker update
        online_targets = byte_tracker.update(det_arr, [h, w], [h, w])

        # 3. Update lifecycle
        tracked_ids = [int(t.track_id) for t in byte_tracker.tracked_stracks if t.is_activated]
        lost_ids = [int(t.track_id) for t in byte_tracker.lost_stracks]
        removed_ids = [int(t.track_id) for t in byte_tracker.removed_stracks]
        tracker_wrapper.lock_manager.update_lifecycle(tracked_ids, lost_ids, removed_ids, frame_idx)

        # 4. Check ownership invariants
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
                        print(f"  [CANDIDATE POOL VIOLATION] frame {frame_idx}: {b_name} already owned!", flush=True)
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
                    escalation_events.append({
                        "frame": frame_idx,
                        "tid": tid,
                        "timestamp": round(frame_idx / fps, 2),
                    })
            else:
                # Contradiction face check on locked tracks (throttled)
                if tracker_wrapper.should_check_locked_face(tid, frame_idx):
                    run_face = True

            if run_face:
                if not state.locked:
                    # Initial lock evaluation requires dual 320 and 640 agreement
                    # 1. First extract at 320 (fast standard resolution)
                    face_emb_320 = tracker_wrapper.face_extractor.extract_from_crop(
                        frame, bbox, padding=0.25, use_640=False
                    )
                    tracker_wrapper.stats["face_extractions"] += 1

                    if face_emb_320 is not None:
                        f_name_320, f_score_320, f_sec_320 = tracker_wrapper.face_extractor.top_match(
                            face_emb_320, tracker_wrapper.face_profiles
                        )
                        # If 320 candidate meets lock threshold and margin, verify with 640 before finalizing lock
                        if f_score_320 >= 0.38 and (f_score_320 - f_sec_320) >= 0.06:
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
                                    "score": round(f_score_320, 3),
                                }
                                locks.append(lock_record)
                                print(f"  --> LOCK: Frame {frame_idx} (t={lock_record['timestamp']}s) | Track {tid} -> {state.locked_name} (score={f_score_320:.3f})", flush=True)
                        else:
                            # Below lock threshold or margin -> submit to keep state updated (holds UNKNOWN)
                            tracker_wrapper.lock_manager.submit_face_match(
                                tid, frame_idx, f_name_320, f_score_320, f_sec_320
                            )
                    else:
                        # 320 found no face. If time-based escalation is active, try 640
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
                    # Contradiction face check on locked tracks (throttled, 320 only)
                    face_emb = tracker_wrapper.face_extractor.extract_from_crop(
                        frame, bbox, padding=0.25, use_640=False
                    )
                    tracker_wrapper.stats["face_extractions"] += 1
                    if face_emb is not None:
                        f_name, f_score, f_sec = tracker_wrapper.face_extractor.top_match(
                            face_emb, tracker_wrapper.face_profiles
                        )
                        if f_score >= 0.38 and (f_score - f_sec) >= 0.06 and f_name != state.locked_name:
                            contradictions.append({
                                "frame": frame_idx,
                                "tid": tid,
                                "locked": state.locked_name,
                                "candidate": f_name,
                                "score": round(f_score, 3),
                            })
                        tracker_wrapper.lock_manager.submit_face_match(
                            tid, frame_idx, f_name, f_score, f_sec
                        )
            else:
                tracker_wrapper.stats["face_extractions_skipped"] += 1

            # Authoritative label
            label = state.locked_name if state.locked else "UNKNOWN"

            # Body Re-ID (Compute reduction: skip when unnecessary)
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
                        if state.locked:
                            tracker_wrapper._last_locked_body_emb_frame[tid] = frame_idx
                            cand = {state.locked_name}
                        else:
                            cand = candidate_students
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
        hud_line1 = f"Frame {frame_idx}/{frames_to_run} | FPS: {fps:.1f} | Active Tracks: {len(online_targets)}"
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

    final_gallery_sizes = dict(body_gallery.sizes())
    gallery_mutations = sum(abs(final_gallery_sizes.get(k, 0) - initial_gallery_sizes.get(k, 0)) for k in initial_gallery_sizes)

    report = {
        "video": video_path,
        "total_frames": frames_to_run,
        "runtime_seconds": round(runtime, 2),
        "effective_fps": round(effective_fps, 2),
        "video_fps": fps,
        "confirmed_locks": locks,
        "total_locks": len(locks),
        "contradictions": contradictions,
        "escalation_events_count": len(escalation_events),
        "sample_escalation_events": escalation_events[:10],
        "ownership_violations": ownership_violations,
        "candidate_pool_violations": candidate_pool_violations,
        "gallery_mutations": gallery_mutations,
        "stats": dict(tracker_wrapper.stats),
        "final_ownership": dict(tracker_wrapper.lock_manager.student_owner),
        "output_video": output_video_path,
    }
    return report


if __name__ == "__main__":
    print("Initializing Face & Body Extractors and Reference Galleries...", flush=True)
    face_profiles, body_gallery = enroll_students(verbose=False)
    face_extractor = FaceFeatureExtractor(det_size=(320, 320))
    body_extractor = OSNetBodyFeatureExtractor()  # ACTIVE: OSNet-x0.25 512-dim

    # 1. Run talking.mp4
    report_talking = run_video_pipeline(
        video_path="talking.mp4",
        cached_dets_path="test_outputs/cached_talking_yolo26n_dets.json",
        output_video_path="test_outputs/talking_adaptive_640.mp4",
        max_frames=3417,
        face_extractor=face_extractor,
        body_extractor=body_extractor,
        face_profiles=face_profiles,
        body_gallery=body_gallery,
    )
    with open("test_outputs/talking_adaptive_640_report.json", "w") as f:
        json.dump(report_talking, f, indent=2)

    # 2. Run new lab video.mp4
    report_new_lab = run_video_pipeline(
        video_path="new lab video.mp4",
        cached_dets_path="test_outputs/cached_yolo26pose_detections_1800f.json",
        output_video_path="test_outputs/new_lab_adaptive_640.mp4",
        max_frames=1800,
        face_extractor=face_extractor,
        body_extractor=body_extractor,
        face_profiles=face_profiles,
        body_gallery=body_gallery,
    )
    with open("test_outputs/new_lab_adaptive_640_report.json", "w") as f:
        json.dump(report_new_lab, f, indent=2)

    print("\nALL PIPELINE RUNS COMPLETED SUCCESSFULLY!", flush=True)
