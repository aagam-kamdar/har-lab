"""
measure_live_vs_retro_locks.py
Measures and reports BOTH:
1. Live Time-to-Lock: when the student's name is first displayed live on the dashboard/stream.
2. Retroactive Resolution Time: when earlier historical frames are retroactively rewritten.
3. Total person-time spent as 'Identifying...', 'Unknown', and 'Named'.
"""

import os
import sys
import cv2
from collections import defaultdict
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter
from enroll_students import KNOWN_STUDENTS

def measure_locks(video_path="new lab video.mp4", max_frames=600):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    adapter = IdentityAdapter(fps=fps, yolo_model_path="yolo26n.pt")

    live_lock_frame = {}
    retro_resolution_frame = {}
    first_seen_frame = {}

    frame_idx = 0
    student_live_frames = defaultdict(int)
    student_identifying_frames = defaultdict(int)
    student_unknown_frames = defaultdict(int)

    # Track lifecycle history
    track_history = defaultdict(list) # tid -> [(frame_idx, person_id, display_name)]

    print(f"\nAnalyzing {video_path} for Live vs Retroactive Lock Times (fps={fps:.1f})...")

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        tracks = adapter.update(frame, frame_idx)

        for t in tracks:
            tid = t["track_id"]
            pid = t["person_id"]
            dname = t["display_name"]

            if tid not in first_seen_frame:
                first_seen_frame[tid] = frame_idx

            track_history[tid].append((frame_idx, pid, dname))

            if pid in KNOWN_STUDENTS:
                student_live_frames[pid] += 1
                if pid not in live_lock_frame:
                    live_lock_frame[pid] = frame_idx
                    # The retroactive resolution happens at this moment, covering from first_seen_frame[tid] to frame_idx
                    retro_resolution_frame[pid] = (first_seen_frame[tid], frame_idx)
            elif "identifying" in pid:
                student_identifying_frames[tid] += 1
            else:
                student_unknown_frames[tid] += 1

        frame_idx += 1

    cap.release()

    print("\n" + "=" * 85)
    print("IDENTITY RESOLUTION REPORT: LIVE VS RETROACTIVE LOCK TIMES")
    print("=" * 85)
    print(f"{'Student':<10} | {'First Seen':<12} | {'Live Lock (Time)':<20} | {'Retroactive Coverage':<22} | {'Status'}")
    print("-" * 85)

    for s in KNOWN_STUDENTS:
        first_f = None
        # Find earliest appearance of track that became s
        for tid, (f_start, f_res) in retro_resolution_frame.items():
            pass

        if s in live_lock_frame:
            f_live = live_lock_frame[s]
            t_live = f_live / fps
            f_start, f_res = retro_resolution_frame[s]
            t_start = f_start / fps
            retro_str = f"t={t_start:.2f}s -> t={t_live:.2f}s"
            live_str = f"{t_live:.2f}s (f{f_live})"
            first_str = f"{t_start:.2f}s (f{f_start})"
            status = "RESOLVED"
        else:
            first_str = "N/A"
            live_str = "NOT LOCKED"
            retro_str = "None"
            status = "PENDING"

        print(f"{s.capitalize():<10} | {first_str:<12} | {live_str:<20} | {retro_str:<22} | {status}")

    print("=" * 85 + "\n")

if __name__ == "__main__":
    measure_locks()
