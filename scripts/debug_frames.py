import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter, KNOWN_STUDENTS

adapter = IdentityAdapter(fps=30.0, mode='stitching')
cap = cv2.VideoCapture('new video.mp4')

for f_idx in range(535):
    ret, frame = cap.read()
    if not ret:
        break
    tracks = adapter.update(frame, f_idx)
    if 495 <= f_idx <= 515:
        print(f"=== FRAME {f_idx} (t={f_idx/30.0:.2f}s) ===")
        print("  Recent ended:", [(e['track_id'], e['student'], e['frame_idx']) for e in adapter.recent_ended_tracks])
        print("  Stitched:", dict(adapter.stitched_tracks))
        print("  Assigned:", {tid: (val[0], round(val[1], 3), val[2]) for tid, val in adapter.track_assigned.items()})
        for t in tracks:
            tid = t['track_id']
            f_ema = {s: round(adapter.track_face_ema[tid].get(s, 0.0), 3) for s in KNOWN_STUDENTS}
            b_ema = {s: round(adapter.track_body_ema[tid].get(s, 0.0), 3) for s in KNOWN_STUDENTS}
            print(f"    Track {tid} bbox={t['bbox']} assigned={t['person_id']} lock={t['is_locked']} margin={adapter.track_assignment_margin.get(tid, 0):.2f}")
            print(f"      Face EMA: {f_ema}")
            print(f"      Body EMA: {b_ema}")

cap.release()
