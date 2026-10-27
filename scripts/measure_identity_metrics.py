"""
measure_identity_metrics.py
Measures:
1. Tracks per person
2. Time-to-lock per person (seconds from appearance to name assignment)
3. Percentage of person-seconds spent Unknown vs Identified on new lab video.mp4
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from collections import defaultdict
from identity_adapter import IdentityAdapter
from enroll_students import KNOWN_STUDENTS

def run_measurement(adapter_cls=IdentityAdapter, max_frames=600, fps=None, label="MATCHED"):
    video_path = "new lab video.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Video {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if fps is None:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    adapter = adapter_cls(fps=fps, yolo_model_path="yolo26n.pt")

    frame_idx = 0
    person_frames_known = defaultdict(int)
    person_frames_unknown = defaultdict(int)
    tracks_per_person = defaultdict(set)
    first_appearance_frame = {}
    first_lock_frame = {}
    track_to_assigned_person = {}

    print(f"\n=======================================================")
    print(f"RUNNING IDENTITY MEASUREMENT ({label}) | max_frames={max_frames}")
    print(f"=======================================================")

    while cap.isOpened() and frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        tracks = adapter.update(frame, frame_idx)

        # Record active tracks
        for t in tracks:
            tid = t["track_id"]
            pid = t["person_id"] # 'aagam', 'harshil', 'soham', 'swayam', or 'unknown'

            if tid not in first_appearance_frame:
                first_appearance_frame[tid] = frame_idx

            if pid != "unknown" and pid in KNOWN_STUDENTS:
                tracks_per_person[pid].add(tid)
                track_to_assigned_person[tid] = pid
                person_frames_known[pid] += 1
                if pid not in first_lock_frame:
                    first_lock_frame[pid] = frame_idx
            else:
                # Track is unknown in this frame
                # If we later know who this track belongs to, we can attribute it
                person_frames_unknown[tid] += 1

        frame_idx += 1

    cap.release()

    # Attribute unknown frames of resolved tracks to the resolved student
    student_total_frames = defaultdict(int)
    student_known_frames = defaultdict(int)
    student_unknown_frames = defaultdict(int)

    for tid, pid in track_to_assigned_person.items():
        unk = person_frames_unknown[tid]
        kwn = person_frames_known[pid]
        student_unknown_frames[pid] += unk

    for s in KNOWN_STUDENTS:
        kwn = person_frames_known[s]
        unk = student_unknown_frames[s]
        tot = kwn + unk
        student_total_frames[s] = tot
        student_known_frames[s] = kwn

    print(f"\n--- RESULTS: {label} (Evaluated on {frame_idx} frames / {frame_idx/fps:.1f}s) ---")
    print(f"{'Student':<10} | {'Tracks':<6} | {'Time-to-Lock (s)':<18} | {'Total Time':<10} | {'Unknown Time':<12} | {'% Unknown':<10}")
    print("-" * 75)

    total_kwn_all = 0
    total_unk_all = 0

    for s in KNOWN_STUDENTS:
        t_count = len(tracks_per_person[s])
        first_lock = first_lock_frame.get(s)
        lock_sec_str = f"{first_lock / fps:.2f}s (f{first_lock})" if first_lock is not None else "NOT LOCKED"
        tot_sec = student_total_frames[s] / fps
        unk_sec = student_unknown_frames[s] / fps
        kwn_sec = student_known_frames[s] / fps
        pct_unk = (unk_sec / tot_sec * 100.0) if tot_sec > 0 else 0.0

        total_kwn_all += kwn_sec
        total_unk_all += unk_sec

        print(f"{s.capitalize():<10} | {t_count:<6} | {lock_sec_str:<18} | {tot_sec:<10.1f}s | {unk_sec:<12.1f}s | {pct_unk:<9.1f}%")

    all_tot = total_kwn_all + total_unk_all
    overall_pct_unk = (total_unk_all / all_tot * 100.0) if all_tot > 0 else 0.0
    print("-" * 75)
    print(f"Overall Across Enrolled Students: {total_unk_all:.1f}s Unknown / {all_tot:.1f}s Total ({overall_pct_unk:.1f}% Unknown)")
    print(f"=======================================================\n")

    return {
        "label": label,
        "tracks_per_person": {s: len(tracks_per_person[s]) for s in KNOWN_STUDENTS},
        "first_lock_frame": first_lock_frame,
        "student_unknown_frames": dict(student_unknown_frames),
        "student_known_frames": dict(student_known_frames),
        "overall_pct_unk": overall_pct_unk,
    }

if __name__ == "__main__":
    run_measurement(max_frames=300, fps=8.0, label="BASELINE")
