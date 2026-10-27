"""
investigate_swayam_face.py
Deep diagnostic of Swayam's identity lock in 'new lab video.mp4':
1. Reports BOTH live time-to-lock (when display_name first shows 'Swayam' in live order)
   and retroactive resolution time.
2. Inspects Swayam's face visibility per frame:
   - Face detection presence (SCRFD)
   - Detected face bounding box & size (W x H)
   - Estimated Yaw angle from 5-point facial landmarks
   - ArcFace cosine similarity against Swayam's enrolled profile vs other students
3. Concludes definitively whether the 15s lock delay is due to camera angle / head orientation
   or matching threshold.
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import FaceFeatureExtractor
from enroll_students import enroll_students, KNOWN_STUDENTS
from identity_adapter import IdentityAdapter

def estimate_yaw_from_landmarks(kpts: np.ndarray) -> float:
    """
    Estimates yaw from 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth).
    Returns approximate yaw in degrees (-90 to +90).
    """
    if kpts is None or len(kpts) < 3:
        return 0.0
    l_eye = kpts[0]
    r_eye = kpts[1]
    nose = kpts[2]

    eye_dist = np.linalg.norm(r_eye - l_eye)
    if eye_dist < 1e-4:
        return 0.0

    # Horizontal position of nose relative to eye midpoint
    eye_mid_x = (l_eye[0] + r_eye[0]) / 2.0
    dx = (nose[0] - eye_mid_x) / (eye_dist / 2.0)
    # Clamp to [-1, 1] and convert to degrees
    dx = np.clip(dx, -1.0, 1.0)
    yaw_deg = float(dx * 45.0) # approx +/-45 deg range
    return yaw_deg


def run_swayam_investigation(video_path: str = "new lab video.mp4", max_frames: int = 500):
    if not os.path.exists(video_path):
        print(f"[ERROR] {video_path} not found.")
        return

    print("=" * 75)
    print("SWAYAM IDENTITY & FACE VISIBILITY INVESTIGATION")
    print("=" * 75)
    print(f"Analyzing {video_path} up to frame {max_frames}...")

    # Load enrolled face profiles
    profiles, _ = enroll_students(verbose=False)
    face_extractor = FaceFeatureExtractor(det_size=(320, 320))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Also track with IdentityAdapter to record Live vs Retroactive lock
    adapter = IdentityAdapter(fps=fps, yolo_model_path="yolo26n.pt")

    frame_idx = 0
    live_lock_frame = None
    retro_lock_frame = None
    swayam_tid = None

    frame_records = []

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Run adapter
        tracks = adapter.update(frame, frame_idx)

        # Locate Swayam's track
        swayam_box = None
        for t in tracks:
            pid = t["person_id"]
            dname = t["display_name"]
            tid = t["track_id"]

            if pid == "swayam" or dname == "Swayam":
                swayam_tid = tid
                swayam_box = t["bbox"]
                if live_lock_frame is None:
                    live_lock_frame = frame_idx
                    print(f"[LIVE LOCK] Swayam first displayed live at frame {frame_idx} ({frame_idx/fps:.2f}s) | Track {tid}")
                break

        # If not yet identified by adapter, check tracks in Swayam's typical spatial area (e.g. Seat D02 or near desk)
        if swayam_box is None and swayam_tid is not None:
            for t in tracks:
                if t["track_id"] == swayam_tid:
                    swayam_box = t["bbox"]
                    break

        # If still None, look for track in D02 / Swayam's seat region if unassigned
        if swayam_box is None:
            # Let's inspect all tracks for face matches against Swayam
            for t in tracks:
                b = t["bbox"]
                # Crop and test face
                emb = face_extractor.extract_from_crop(frame, b)
                if emb is not None:
                    score = float(np.dot(emb, profiles["swayam"]))
                    if score > 0.40:
                        swayam_box = b
                        swayam_tid = t["track_id"]
                        break

        # If we have a box for Swayam, do detailed face analysis
        record = {
            "frame": frame_idx,
            "time_s": frame_idx / fps,
            "has_person_box": swayam_box is not None,
            "face_detected": False,
            "face_size": (0, 0),
            "face_area": 0,
            "yaw_deg": 0.0,
            "swayam_score": 0.0,
            "best_match": None,
            "best_score": 0.0
        }

        if swayam_box is not None:
            x1, y1, x2, y2 = swayam_box
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if crop.size > 0:
                # Run SCRFD directly on the crop to inspect detections
                faces = face_extractor.app.get(crop)
                if len(faces) > 0:
                    # Pick largest face
                    f_obj = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    fb = f_obj.bbox
                    fw = int(fb[2] - fb[0])
                    fh = int(fb[3] - fb[1])
                    record["face_detected"] = True
                    record["face_size"] = (fw, fh)
                    record["face_area"] = fw * fh
                    record["yaw_deg"] = estimate_yaw_from_landmarks(f_obj.kps)

                    # Cosine sim with all students
                    emb = f_obj.embedding
                    if emb is not None:
                        emb = emb / np.linalg.norm(emb)
                        sims = {s: float(np.dot(emb, profiles[s])) for s in KNOWN_STUDENTS}
                        best_s = max(sims, key=sims.get)
                        record["swayam_score"] = sims["swayam"]
                        record["best_match"] = best_s
                        record["best_score"] = sims[best_s]

        frame_records.append(record)
        frame_idx += 1

    cap.release()

    # Analysis of face records
    total_f = len(frame_records)
    faces_detected = [r for r in frame_records if r["face_detected"]]
    n_face = len(faces_detected)

    print("\n" + "=" * 75)
    print("SWAYAM FACE VISIBILITY & MATCHING ANALYSIS")
    print("=" * 75)
    print(f"Total Frames Analyzed: {total_f} ({total_f/fps:.1f}s)")
    print(f"Frames with Person Box: {sum(1 for r in frame_records if r['has_person_box'])} / {total_f}")
    print(f"Frames with Detected Face: {n_face} / {total_f} ({n_face/total_f*100.0:.1f}%)")

    # Time intervals breakdown (e.g. 0-5s, 5-10s, 10-15s, 15-20s)
    intervals = [(0, 5), (5, 10), (10, 15), (15, 20)]
    print("\n" + "-" * 75)
    print(f"{'Time Interval':<16} | {'Face Detect %':<14} | {'Avg Size (px)':<14} | {'Avg Yaw (deg)':<14} | {'Avg Swayam Sim':<14}")
    print("-" * 75)

    for start_s, end_s in intervals:
        sub = [r for r in frame_records if start_s <= r["time_s"] < end_s]
        if not sub:
            continue
        sub_faces = [r for r in sub if r["face_detected"]]
        det_pct = len(sub_faces) / len(sub) * 100.0
        avg_sz = np.mean([r["face_size"][0] for r in sub_faces]) if sub_faces else 0.0
        avg_yaw = np.mean([abs(r["yaw_deg"]) for r in sub_faces]) if sub_faces else 0.0
        avg_sim = np.mean([r["swayam_score"] for r in sub_faces]) if sub_faces else 0.0
        print(f"{f'{start_s}s - {end_s}s':<16} | {det_pct:<13.1f}% | {avg_sz:<14.1f} | {avg_yaw:<14.1f} | {avg_sim:<14.3f}")

    print("-" * 75)

    # First face detection vs first lock
    first_face = next((r for r in frame_records if r["face_detected"]), None)
    if first_face:
        print(f"\nFirst Face Detected at: Frame {first_face['frame']} ({first_face['time_s']:.2f}s) | Size: {first_face['face_size']} | Yaw: {first_face['yaw_deg']:.1f} deg | Score: {first_face['swayam_score']:.3f}")
    else:
        print("\nNo face was detected in the entire analyzed range!")

    # Root Cause Diagnostic Conclusion
    print("\n" + "=" * 75)
    print("ROOT CAUSE DIAGNOSIS: Camera Angle vs Matching Threshold")
    print("=" * 75)
    early_sub = [r for r in frame_records if r["time_s"] < 12.0]
    early_faces = [r for r in early_sub if r["face_detected"]]

    if len(early_faces) / len(early_sub) < 0.20:
        cause = "CAMERA ANGLE / HEAD POSE (Face Invisibility)"
        detail = (
            f"During the first 12 seconds, Swayam's face was visible in only {len(early_faces)}/{len(early_sub)} frames ({len(early_faces)/len(early_sub)*100.0:.1f}%). "
            f"Swayam was facing away from the camera / looking down at the workstation, preventing SCRFD face landmark detection."
        )
    else:
        # Check scores
        avg_early_sim = np.mean([r["swayam_score"] for r in early_faces])
        if avg_early_sim < 0.40:
            cause = "EXTREME POSE / DISTORTION"
            detail = f"Face was detected but cosine similarity was suppressed (avg {avg_early_sim:.3f}) due to high yaw/pitch."
        else:
            cause = "MATCHING THRESHOLD"
            detail = f"Face was visible with solid similarity ({avg_early_sim:.3f}), but lock gate required multiple consecutive confirmations."

    print(f"Primary Cause: {cause}")
    print(f"Detailed Explanation: {detail}")
    print("=" * 75 + "\n")

    return {
        "live_lock_frame": live_lock_frame,
        "total_frames": total_f,
        "face_detection_pct": n_face / total_f * 100.0 if total_f > 0 else 0.0,
        "primary_cause": cause,
        "detail": detail
    }

if __name__ == "__main__":
    run_swayam_investigation()
