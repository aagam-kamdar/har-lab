import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter, KNOWN_STUDENTS
from features import FaceFeatureExtractor
from enroll_students import enroll_students

def scan_video():
    print("=== Scanning new video.mp4 for Face Detections and Identity Assignments ===")
    cap = cv2.VideoCapture("new video.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    face_extractor = FaceFeatureExtractor(det_size=(320, 320))
    face_profiles, _ = enroll_students(verbose=False)

    # Let's sample every 10 frames across the first 3000 frames (first 100s)
    # and find all face detections
    detections = []
    for f_idx in range(0, min(3000, total_frames), 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_extractor.app.get(frame)
        for f in faces:
            bbox = [int(x) for x in f.bbox]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            emb = f.normed_embedding
            sims = {s: float(np.dot(emb, face_profiles[s])) for s in KNOWN_STUDENTS}
            best_s = max(sims, key=sims.get)
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            top1, score1 = sorted_sims[0]
            top2, score2 = sorted_sims[1]
            margin = score1 - score2
            
            if score1 >= 0.25:
                detections.append({
                    "frame": f_idx,
                    "time_s": f_idx / 30.0,
                    "bbox": bbox,
                    "cx": cx,
                    "cy": cy,
                    "best": top1,
                    "score1": score1,
                    "second": top2,
                    "score2": score2,
                    "margin": margin,
                    "all": sims
                })

    cap.release()
    print(f"\nFound {len(detections)} face detections with score >= 0.25:")
    for d in detections:
        print(f"Frame {d['frame']:04d} (t={d['time_s']:5.1f}s, cx={d['cx']:5.1f}): "
              f"Best={d['best']} ({d['score1']:.3f}), 2nd={d['second']} ({d['score2']:.3f}), "
              f"margin={d['margin']:.3f}, bbox={d['bbox']}")

if __name__ == "__main__":
    scan_video()
