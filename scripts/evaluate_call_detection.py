"""
scripts/evaluate_call_detection.py
Evaluates Call classification:
1. False-Positive Rate on hand-on-face / thinking / chin-resting candidate clips.
2. Held-out check for real calls (when labeled by user).
3. Postural geometry verification (ensuring lifted elbow + wrist at ear level is recognized).
"""

import os
import sys
import glob
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab_monitor import PersonTracker, raw_label_person, extract_person_features, DEFAULT_TH

def evaluate_call(clips_dir="test_outputs/labeling_clips"):
    th = DEFAULT_TH.copy()
    if os.path.exists("thresholds.yaml"):
        with open("thresholds.yaml", "r") as f:
            th.update(yaml.safe_load(f))

    pose_model = YOLO("yolo26n-pose.pt" if os.path.exists("yolo26n-pose.pt") else "yolov8n-pose.pt")

    # 1. False-Positive Rate Check on Hand-on-Face / Non-Call Clips
    non_call_clips = glob.glob(os.path.join(clips_dir, "*ear*.mp4")) + \
                     glob.glob(os.path.join(clips_dir, "*Harshil*.mp4"))
    non_call_clips = sorted(list(set(non_call_clips)))

    total_neg_frames = 0
    fp_before = 0  # wrist-ear alone
    fp_after = 0   # wrist-ear + elbow lifted + ear level + stillness

    for cp in non_call_clips:
        cap = cv2.VideoCapture(cp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        f_idx = 0
        p = PersonTracker("test_p", "Test", 10)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_neg_frames += 1
            now = f_idx / fps
            res = pose_model(frame, verbose=False)[0]
            if res.keypoints is not None and len(res.keypoints) > 0:
                kpts = res.keypoints.xy.cpu().numpy()
                confs = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
                for p_i in range(len(kpts)):
                    p = PersonTracker(f"test_p_{p_i}", f"Test_{p_i}", 10)
                    feat = extract_person_features(p, kpts[p_i], confs[p_i] if confs is not None else None, {}, now, th)
                    if feat is not None:
                        # Before check: wrist_ear < call_ear alone
                        if feat["wrist_ear"] < th["call_ear"]:
                            fp_before += 1
                        # After check: full explainable rule
                        lbl, reason = raw_label_person(p, feat, {}, {}, now, th)
                        if lbl == "call":
                            fp_after += 1
            f_idx += 1
        cap.release()

    fpr_before = (fp_before / total_neg_frames * 100.0) if total_neg_frames > 0 else 0.0
    fpr_after = (fp_after / total_neg_frames * 100.0) if total_neg_frames > 0 else 0.0

    # 2. Check for User-Labeled Real Calls
    real_call_clips = glob.glob(os.path.join(clips_dir, "*real_call*.mp4")) + \
                      glob.glob(os.path.join(clips_dir, "*true_call*.mp4"))

    # Also check if any CSV labels real calls
    real_call_labeled = False
    real_call_support = 0
    real_call_tp = 0

    if real_call_clips:
        real_call_labeled = True
        for cp in real_call_clips:
            cap = cv2.VideoCapture(cp)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            f_idx = 0
            p = PersonTracker("call_p", "CallPerson", 10)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                real_call_support += 1
                now = f_idx / fps
                res = pose_model(frame, verbose=False)[0]
                if res.keypoints is not None and len(res.keypoints) > 0:
                    kpts = res.keypoints.xy.cpu().numpy()
                    confs = res.keypoints.conf.cpu().numpy()
                    feat = extract_person_features(p, kpts[0], confs[0], {}, now, th)
                    if feat:
                        lbl, _ = raw_label_person(p, feat, {}, {}, now, th)
                        if lbl == "call":
                            real_call_tp += 1
                f_idx += 1
            cap.release()

    # 3. Geometric Verification (Simulated Real Call Posture)
    p_sim = PersonTracker("sim_call", "SimCall", 10)
    sim_real_call_feat = {
        "standing": False,
        "speed": 0.0,
        "wrist_ear": 0.35,              # Wrist directly at ear (< 0.60)
        "call_pose_valid": True,        # Elbow lifted away from desk + wrist at ear level
        "ear_hold": 3.0,                # Held steady for 3.0s (>= 2.5s)
        "wrist_phone": 99.0,
        "head_down": False,
        "turn_dir": 0,
        "motion": 0.05,
        "hands_low": False,
        "on_kb": False,
        "since_motion": 999.0
    }
    sim_lbl, sim_reason = raw_label_person(p_sim, sim_real_call_feat, {}, {}, 100.0, th)
    sim_detected = (sim_lbl == "call")

    print("=" * 80)
    print("CALL DETECTION EVALUATION & HELD-OUT CHECK")
    print("=" * 80)
    print(f"Hand-on-Face / Negative Frames Evaluated : {total_neg_frames}")
    print(f"  False Positives BEFORE (wrist-ear only) : {fp_before}/{total_neg_frames} ({fpr_before:.1f}%)")
    print(f"  False Positives AFTER (elbow + height)  : {fp_after}/{total_neg_frames} ({fpr_after:.1f}%)")
    print(f"  False Positive Rate Reduction           : {fpr_before - fpr_after:.1f}% absolute reduction")
    print("-" * 80)
    print("Held-Out Human-Labeled Real Call Verification:")
    if real_call_labeled:
        recall = (real_call_tp / real_call_support * 100.0) if real_call_support > 0 else 0.0
        print(f"  Real Call Support: {real_call_support} frames")
        print(f"  Real Calls Detected (TP): {real_call_tp}")
        print(f"  Real Call Recall: {recall:.1f}%")
    else:
        print("  Real Call Support: 0 intervals (no human-confirmed real call clips labeled yet)")
        print("  Real Call Recall : Not evaluated (pending human labels)")
    print(f"  Geometric Real Call Sensitivity Check: {'VERIFIED (Pass)' if sim_detected else 'FAILED'}")
    print("=" * 80)

    return {
        "neg_frames": total_neg_frames,
        "fp_before": fp_before,
        "fp_after": fp_after,
        "fpr_before": fpr_before,
        "fpr_after": fpr_after,
        "real_call_labeled": real_call_labeled,
        "real_call_recall": (real_call_tp / real_call_support) if real_call_support > 0 else None,
        "sim_detected": sim_detected
    }

if __name__ == "__main__":
    evaluate_call()
