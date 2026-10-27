"""
measure_replay_metrics.py - Measure tracks per person, time to lock, and unknown person-seconds.
"""
import os
import re
from collections import defaultdict

def analyze_summary_events():
    path = "test_outputs/final_identity_lock_full_video_summary.md"
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    events = []
    for line in lines:
        l = line.strip()
        if l.startswith("|") and "`f" in l:
            parts = [p.strip() for p in l.split("|")[1:-1]]
            if len(parts) >= 6:
                frame_str = parts[0].replace("`", "").strip()
                ts_str = parts[1].replace("`", "").strip()
                ev_type = parts[2].replace("*", "").strip()
                track = parts[3].replace("Track", "").replace("`", "").strip()
                student = parts[4].replace("*", "").strip().lower()
                detail = parts[5].strip()
                events.append((frame_str, ts_str, ev_type, track, student, detail))

    print(f"Total milestone events recorded: {len(events)}")
    tracks_per_student = defaultdict(set)
    locks = {}

    for frame, ts, ev_type, track, student, detail in events:
        tracks_per_student[student].add(track)
        if ev_type == "LOCK" and student not in locks:
            locks[student] = (frame, ts, track)

    for s in ["soham", "harshil", "swayam", "aagam"]:
        t_ids = sorted(list(tracks_per_student[s]), key=lambda x: int(x) if x.isdigit() else 999)
        print(f"Student: {s.upper()}")
        print(f"  Total Tracks ({len(t_ids)}): {t_ids}")
        print(f"  First Lock: {locks.get(s)}")

if __name__ == "__main__":
    analyze_summary_events()
