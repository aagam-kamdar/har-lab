"""
diagnose_identity.py
Runs IdentityAdapter on new video.mp4, logging every frame where:
- Track at Desk 2 (Swayam's desk, cx ~ 540-660) is labeled Aagam,
- Or Track at Desk 1 (Aagam's desk, cx ~ 450-570) is labeled Swayam,
- Or Track at Desk 3 (Harshil's desk, cx ~ 640-780) is labeled incorrectly,
- Or any lock / Hungarian assignment occurs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

from identity_adapter import IdentityAdapter, KNOWN_STUDENTS

def run_diagnostics(video_path="new video.mp4", max_frames=600):
    print(f"=== Diagnosing Identity on {video_path} (up to {max_frames} frames) ===")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return

    adapter = IdentityAdapter(fps=30.0, mode="stitching")
    
    error_count = 0
    total_evaluated_tracks = 0

    for f_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        tracks = adapter.update(frame, f_idx)
        for t in tracks:
            tid = t["track_id"]
            pid = t["person_id"]
            bbox = t["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            
            # Ground truth expected person based on desk ROI:
            # D01: 450-570 -> Aagam
            # D02: 540-660 -> Swayam
            # D03: 640-780 -> Harshil
            # D04: 770-960 -> Soham
            expected = None
            if 420 <= cx < 550:
                expected = "aagam"
            elif 550 <= cx < 650:
                expected = "swayam"
            elif 650 <= cx < 780:
                expected = "harshil"
            elif 780 <= cx <= 980:
                expected = "soham"

            face_sims = {s: round(adapter.track_face_ema[tid].get(s, 0.0), 3) for s in KNOWN_STUDENTS}
            body_sims = {s: round(adapter.track_body_ema[tid].get(s, 0.0), 3) for s in KNOWN_STUDENTS}

            is_error = False
            if expected and pid in KNOWN_STUDENTS and pid != expected:
                is_error = True
                error_count += 1

            if f_idx % 60 == 0 or is_error or (t.get("is_locked") and f_idx % 30 == 0):
                lock_state = "LOCKED" if t.get("is_locked") else "UNLOCKED"
                err_flag = " *** ERROR ***" if is_error else ""
                print(f"[f{f_idx:04d} t={f_idx/30.0:5.1f}s] Tid={tid:2d} (cx={cx:5.1f}, cy={cy:5.1f}) "
                      f"Assigned={pid:<10} (exp={expected or 'None':<7}) [{lock_state}] {err_flag}\n"
                      f"   Face EMA: {face_sims}\n"
                      f"   Body EMA: {body_sims}\n"
                      f"   Track status: {t.get('status')} conf={t.get('identity_confidence'):.3f}")

    cap.release()
    print(f"\nDiagnostic finished: {error_count} identity assignment errors detected across {max_frames} frames.")

if __name__ == "__main__":
    run_diagnostics()
