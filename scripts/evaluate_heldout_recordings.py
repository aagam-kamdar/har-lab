"""
evaluate_heldout_recordings.py
Evaluates rule cascade on held-out recordings (talking.mp4 and writi.mp4) split by RECORDING.
Strictly enforces evaluation hygiene:
- Reports per-class Precision, Recall, and Support.
- Classes with 0 support (e.g. phone, call) are explicitly marked "Not evaluated (0 support)"
- Does NOT claim any unverified scores.
"""

import os
import sys
import cv2
import numpy as np
import yaml
from collections import defaultdict
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter
from lab_monitor import PersonTracker, raw_label_person, extract_person_features, DEFAULT_TH

def evaluate_heldout():
    th_path = "thresholds.yaml"
    th = DEFAULT_TH.copy()
    if os.path.exists(th_path):
        with open(th_path, "r") as f:
            th.update(yaml.safe_load(f))

    recordings = [
        {"file": "talking.mp4", "ground_truth_label": "discussing", "max_frames": 300, "stride": 15},
        {"file": "writi.mp4", "ground_truth_label": "writing", "max_frames": 300, "stride": 15}
    ]

    pose_model_path = "yolo26n-pose.pt" if os.path.exists("yolo26n-pose.pt") else "yolov8n-pose.pt"
    det_model_path = "yolo26n.pt" if os.path.exists("yolo26n.pt") else "yolov8n.pt"

    pose_model = YOLO(pose_model_path)
    det_model = YOLO(det_model_path)

    all_classes = ["active_work", "writing", "reading", "discussing", "idle", "walking", "phone", "call", "away", "uncertain"]

    # Confusion counts: class -> {"TP": 0, "FP": 0, "FN": 0, "Support": 0}
    stats = {c: {"TP": 0, "FP": 0, "FN": 0, "Support": 0} for c in all_classes}

    print("=" * 80)
    print("EVALUATING RULE CASCADE ON HELD-OUT RECORDINGS (By Recording Split)")
    print("=" * 80)

    for rec in recordings:
        v_path = rec["file"]
        gt_label = rec["ground_truth_label"]
        max_f = rec["max_frames"]
        stride = rec["stride"]

        if not os.path.exists(v_path):
            print(f"[WARNING] {v_path} not found. Skipping.")
            continue

        print(f"\nProcessing held-out recording: {v_path} (Dominant Ground-Truth: {gt_label})...")
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        f_idx = 0
        evaluated_samples = 0
        persons = {}

        while cap.isOpened() and f_idx < max_f:
            ret, frame = cap.read()
            if not ret:
                break

            if f_idx % stride == 0:
                now = f_idx / fps
                h, w = frame.shape[:2]

                # Run pose and det
                res_det = det_model(frame, verbose=False)[0]
                objs = defaultdict(list)
                for box, cls_id in zip(res_det.boxes.xyxy.cpu().numpy(), res_det.boxes.cls.cpu().numpy()):
                    objs[int(cls_id)].append(box)

                res_pose = pose_model(frame, verbose=False)[0]
                if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                    kpts = res_pose.keypoints.xy.cpu().numpy()
                    confs = res_pose.keypoints.conf.cpu().numpy() if res_pose.keypoints.conf is not None else None

                    all_f = {}
                    current_pids = []
                    for p_i in range(len(kpts)):
                        pid = f"person_{p_i}"
                        current_pids.append(pid)
                        if pid not in persons:
                            persons[pid] = PersonTracker(pid, pid, 10)
                        p = persons[pid]

                        feat = extract_person_features(p, kpts[p_i], confs[p_i] if confs is not None else None, objs, now, th)
                        all_f[pid] = feat

                    for pid in current_pids:
                        p = persons[pid]
                        feat = all_f[pid]
                        pred_label, reason = raw_label_person(p, feat, all_f, persons, now, th)

                        evaluated_samples += 1
                        stats[gt_label]["Support"] += 1

                        if pred_label == gt_label:
                            stats[gt_label]["TP"] += 1
                        else:
                            stats[gt_label]["FN"] += 1
                            stats[pred_label]["FP"] += 1

            f_idx += 1

        cap.release()
        print(f"Evaluated {evaluated_samples} person-instances on {v_path}")

    print("\n" + "=" * 80)
    print("HELD-OUT RECORDINGS EVALUATION RESULTS (Strict Hygiene)")
    print("=" * 80)
    print(f"{'Activity Class':<14} | {'Precision':<18} | {'Recall':<18} | {'F1-Score':<14} | {'Support'}")
    print("-" * 80)

    for c in all_classes:
        sup = stats[c]["Support"]
        tp = stats[c]["TP"]
        fp = stats[c]["FP"]
        fn = stats[c]["FN"]

        if sup == 0:
            p_str = "Not evaluated"
            r_str = "Not evaluated"
            f1_str = "Not evaluated"
            sup_str = "0 (no ground truth)"
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            p_str = f"{prec:.3f}"
            r_str = f"{rec:.3f}"
            f1_str = f"{f1:.3f}"
            sup_str = str(sup)

        print(f"{c:<14} | {p_str:<18} | {r_str:<18} | {f1_str:<14} | {sup_str}")

    print("=" * 80 + "\n")
    return stats

if __name__ == "__main__":
    evaluate_heldout()
