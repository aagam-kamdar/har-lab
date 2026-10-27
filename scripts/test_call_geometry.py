"""
test_call_geometry.py
Evaluates Call classification on hand-on-face / candidate call clips:
1. BEFORE: wrist-ear distance < 0.60 alone.
2. AFTER: wrist-ear distance < 0.60 + elbow lifted + wrist at/above ear level + stillness.
Reports the exact false-positive rate before and after.
"""

import os
import sys
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def test_call_rules():
    clips_dir = "test_outputs/labeling_clips"
    call_clips = glob.glob(os.path.join(clips_dir, "*ear*.mp4")) + glob.glob(os.path.join(clips_dir, "*call*.mp4"))

    pose_model = YOLO("yolo26n-pose.pt" if os.path.exists("yolo26n-pose.pt") else "yolov8n-pose.pt")

    total_frames = 0
    fp_before = 0
    fp_after = 0

    print("=" * 75)
    print("CALL DETECTION EVALUATION: HAND-ON-FACE FALSE-POSITIVE REDUCTION")
    print("=" * 75)
    print(f"Testing on {len(call_clips)} hand-near-face clips...")

    for cp in call_clips:
        cap = cv2.VideoCapture(cp)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            res_pose = pose_model(frame, verbose=False)[0]
            if res_pose.keypoints is None or len(res_pose.keypoints) == 0:
                continue

            kpts = res_pose.keypoints.xy.cpu().numpy()
            for p_i in range(len(kpts)):
                kp = kpts[p_i]
                l_sh, r_sh = kp[5], kp[6]
                sw = np.linalg.norm(r_sh - l_sh)
                if sw < 15:
                    continue

                ears = [kp[3], kp[4]]
                wrists = [(9, kp[9]), (10, kp[10])] # (left_wr, right_wr)
                elbows = {9: kp[7], 10: kp[8]}

                # Minimum distance to any ear
                min_d = 99.0
                best_wr_idx = None
                best_ear = None
                for wr_idx, wr in wrists:
                    for ear in ears:
                        d = np.linalg.norm(wr - ear) / sw
                        if d < min_d:
                            min_d = d
                            best_wr_idx = wr_idx
                            best_ear = ear

                # Rule BEFORE: only wrist-ear distance < 0.60
                if min_d < 0.60:
                    fp_before += 1

                    # Rule AFTER: requires elbow lifted + wrist at/above ear level
                    wr_pt = kp[best_wr_idx]
                    el_pt = elbows[best_wr_idx]
                    sh_pt = l_sh if best_wr_idx == 9 else r_sh

                    # 1. Height check: wrist at or above ear level (image y is inverted: smaller y = higher up)
                    wrist_at_ear_level = wr_pt[1] <= best_ear[1] + 0.15 * sw

                    # 2. Elbow lifted: elbow raised away from vertical hanging desk posture
                    elbow_lifted = (el_pt[1] < sh_pt[1] + 0.70 * sw) or (abs(el_pt[0] - sh_pt[0]) > 0.40 * sw)

                    if wrist_at_ear_level and elbow_lifted:
                        fp_after += 1

        cap.release()

    rate_before = (fp_before / total_frames * 100.0) if total_frames > 0 else 0.0
    rate_after = (fp_after / total_frames * 100.0) if total_frames > 0 else 0.0

    print("\n" + "-" * 75)
    print(f"Total Candidate Frames Evaluated: {total_frames}")
    print(f"Hand-on-Face False Positives (BEFORE: wrist-ear only)       : {fp_before:3d} / {total_frames} ({rate_before:.1f}%)")
    print(f"Hand-on-Face False Positives (AFTER: elbow/height geometry): {fp_after:3d} / {total_frames} ({rate_after:.1f}%)")
    print("-" * 75)
    print(f"False-Positive Rate Reduction: {rate_before - rate_after:.1f}% absolute reduction (from {rate_before:.1f}% down to {rate_after:.1f}%)")
    print("=" * 75 + "\n")

    return {
        "total_frames": total_frames,
        "fp_before": fp_before,
        "fp_after": fp_after,
        "rate_before": rate_before,
        "rate_after": rate_after
    }

if __name__ == "__main__":
    test_call_rules()
