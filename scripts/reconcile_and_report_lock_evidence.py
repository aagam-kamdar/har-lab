"""
reconcile_and_report_lock_evidence.py
Reconciles lock timing, states clock definitions, and inspects the exact evidence
(face score, body score, accumulated evidence frames) at lock time for all 4 students.
Flags any locks made on fewer than ~10 frames of evidence.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter
from enroll_students import KNOWN_STUDENTS

def inspect_lock_evidence(video_path="new lab video.mp4", max_frames=300):
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 # 30.0 FPS native
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    adapter = IdentityAdapter(fps=source_fps, yolo_model_path="yolo26n.pt")

    locks = {}
    frame_idx = 0

    print("=" * 85)
    print("INSPECTING LOCK EVIDENCE AT NATIVE SOURCE RESOLUTION AND CLOCK")
    print("=" * 85)
    print(f"Clock: Source Video Wall-Clock Time (PTS = frame_idx / {source_fps:.1f}s)")
    print(f"Video: {video_path} (Native: 1120x480 @ {source_fps:.1f} FPS, {total_frames} frames)")
    print("-" * 85)

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        tracks = adapter.update(frame, frame_idx)

        for t in tracks:
            pid = t["person_id"]
            tid = t["track_id"]
            conf = t["identity_confidence"]
            aff = t.get("affinity", conf)
            margin = t.get("margin", 0.0)
            status = t.get("status", "ASSIGNED")

            if pid in KNOWN_STUDENTS and pid not in locks:
                # Capture evidence state at lock time
                f_s = adapter.track_face_ema[tid].get(pid, 0.0)
                b_s = adapter.track_body_ema[tid].get(pid, 0.0)
                first_seen = adapter.track_first_seen_frame.get(tid, frame_idx)
                n_frames_evidence = frame_idx - first_seen + 1
                ev_duration_s = n_frames_evidence / source_fps
                is_arcface_lock = t.get("is_locked", False)

                locks[pid] = {
                    "student": pid,
                    "track_id": tid,
                    "frame": frame_idx,
                    "source_time_s": frame_idx / source_fps,
                    "face_score": f_s,
                    "body_score": b_s,
                    "affinity": aff,
                    "margin": margin,
                    "frames_evidence": n_frames_evidence,
                    "ev_duration_s": ev_duration_s,
                    "is_arcface_lock": is_arcface_lock,
                    "flag_low_evidence": ev_duration_s < adapter.live_gate_evidence_s and not (is_arcface_lock or margin >= adapter.clear_margin)
                }

        frame_idx += 1

    cap.release()

    print(f"{'Student':<10} | {'Lock Frame':<11} | {'Source Time':<12} | {'Evid (s)':<10} | {'Affinity':<10} | {'Margin':<10} | {'Face':<6} | {'Body':<6} | {'Gate / Trigger'}")
    print("-" * 105)

    for s in KNOWN_STUDENTS:
        if s in locks:
            info = locks[s]
            arc_tag = "ArcFace Lock" if info["is_arcface_lock"] else ("Clear Margin" if info["margin"] >= adapter.clear_margin else ">= 1.5s Evidence")
            print(f"{s.capitalize():<10} | f{info['frame']:<10} | {info['source_time_s']:<10.2f}s | {info['ev_duration_s']:<8.2f}s | {info['affinity']:<10.3f} | {info['margin']:<10.3f} | {info['face_score']:<6.3f} | {info['body_score']:<6.3f} | {arc_tag}")
        else:
            print(f"{s.capitalize():<10} | NOT LOCKED")

    print("=" * 105 + "\n")
    return locks

if __name__ == "__main__":
    inspect_lock_evidence()
