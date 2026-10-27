"""
diagnose_activity_rules.py
Diagnoses activity classification errors per class (phone, call, idle, discussing, active_work)
on test_features.csv. Distinguishes perception failures from rule failures.
Outputs confusion matrix and top failure causes table.
"""
import csv
import os
import yaml
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

def run_diagnosis():
    csv_path = "test_features.csv"
    th_path = "thresholds.yaml"

    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found.")
        return

    with open(th_path, "r") as f:
        th = yaml.safe_load(f)

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"[INFO] Loaded {len(rows)} labeled rows from {csv_path}")

    # Classes of interest
    classes = ["active_work", "phone", "call", "discussing", "idle", "walking"]

    y_true = []
    y_pred = []
    diagnoses = []

    for r in rows:
        gt = r["raw"]
        t = float(r["t"])
        seat = r["seat"]
        student = r["student"]

        # Parse geometric features
        wrist_phone = float(r.get("wrist_phone", 99.0))
        wrist_ear = float(r.get("wrist_ear", 99.0))
        head_down = r.get("head_down", "False").lower() == "true"
        turn_dir = int(r.get("turn_dir", 0))
        turn = float(r.get("turn", 0.0))
        hands_low = r.get("hands_low", "False").lower() == "true"
        on_kb = r.get("on_kb", "False").lower() == "true"
        motion = float(r.get("motion", 0.0))
        speed = float(r.get("speed", 0.0))
        standing = r.get("standing", "False").lower() == "true"
        ear_hold = float(r.get("ear_hold", 0.0))
        since_motion = float(r.get("since_motion", 999.0))

        # Apply rule cascade directly
        pred = "idle"
        reason = "no motion"
        failure_type = None
        failure_detail = None

        if speed > th.get("walk_speed", 0.55) or standing:
            pred = "walking"
            reason = f"speed {speed:.2f} > {th.get('walk_speed', 0.55)} or standing={standing}"
        elif wrist_ear < th.get("call_ear", 0.60) and ear_hold >= th.get("call_hold", 2.5):
            pred = "call"
            reason = f"wrist_ear {wrist_ear:.2f} < {th.get('call_ear', 0.60)}"
        elif wrist_phone < th.get("phone_hand", 1.10) and head_down:
            pred = "phone"
            reason = f"wrist_phone {wrist_phone:.2f} < {th.get('phone_hand', 1.10)} and head_down"
        elif turn_dir != 0 and motion < th.get("talk_motion", 0.25):
            pred = "discussing"
            reason = f"turn_dir={turn_dir}, motion {motion:.2f} < {th.get('talk_motion', 0.25)}"
        elif (hands_low or on_kb) and (motion > th.get("kb_motion", 0.12) or since_motion < th.get("active_grace", 6.0)):
            pred = "active_work"
            reason = f"hands_low={hands_low} on_kb={on_kb}, motion {motion:.2f}"
        else:
            pred = "idle"
            reason = f"motion {motion:.2f} below threshold"

        y_true.append(gt)
        y_pred.append(pred)

        if gt != pred:
            # Diagnose failure
            if gt == "phone":
                if wrist_phone > th.get("phone_hand", 1.10):
                    failure_type = "PERCEPTION_FAILURE"
                    failure_detail = f"Object detector missed phone (wrist_phone={wrist_phone:.2f} > {th.get('phone_hand', 1.10)})"
                elif not head_down:
                    failure_type = "RULE_FAILURE"
                    failure_detail = "Head tilt rule too strict (head_down=False during natural glance)"
            elif gt == "call":
                if wrist_ear > th.get("call_ear", 0.60):
                    failure_type = "PERCEPTION_FAILURE"
                    failure_detail = f"Ear/wrist keypoint noisy (wrist_ear={wrist_ear:.2f} > {th.get('call_ear', 0.60)})"
                elif ear_hold < th.get("call_hold", 2.5):
                    failure_type = "RULE_FAILURE"
                    failure_detail = f"ear_hold window {ear_hold:.1f}s < required {th.get('call_hold', 2.5)}s"
            elif gt == "active_work":
                if not (hands_low or on_kb):
                    failure_type = "PERCEPTION_FAILURE"
                    failure_detail = "Wrist keypoint not detected in desk/keyboard region"
                elif motion <= th.get("kb_motion", 0.12) and since_motion >= th.get("active_grace", 6.0):
                    failure_type = "RULE_FAILURE"
                    failure_detail = f"Micro-typing motion {motion:.2f} below threshold {th.get('kb_motion', 0.12)}"
                else:
                    failure_type = "RULE_FAILURE"
                    failure_detail = f"Preempted by {pred} rule"
            elif gt == "discussing":
                if turn_dir == 0:
                    failure_type = "PERCEPTION_FAILURE"
                    failure_detail = "Nose/shoulder orientation keypoint noise (turn_dir=0)"
                elif motion >= th.get("talk_motion", 0.25):
                    failure_type = "RULE_FAILURE"
                    failure_detail = f"Gesturing motion {motion:.2f} exceeded talk_motion {th.get('talk_motion', 0.25)}"
            elif gt == "idle":
                if pred == "walking":
                    failure_type = "RULE_FAILURE"
                    failure_detail = f"Torso sway speed {speed:.2f} misclassified as walking"
                elif pred == "active_work":
                    failure_type = "RULE_FAILURE"
                    failure_detail = f"Residual desk motion {motion:.2f} misclassified as work"
                else:
                    failure_type = "RULE_FAILURE"
                    failure_detail = f"Preempted by {pred}"
            else:
                failure_type = "RULE_FAILURE"
                failure_detail = f"Mismatched to {pred}"

            diagnoses.append({
                "t": t, "student": student, "seat": seat,
                "gt": gt, "pred": pred,
                "failure_type": failure_type,
                "failure_detail": failure_detail,
                "reason": reason
            })

    # Confusion matrix
    present_labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    print("\n" + "=" * 65)
    print("CONFUSION MATRIX: Explainable Rules vs Ground Truth")
    print("=" * 65)
    header = f"{'True / Pred':<14} | " + " | ".join([f"{l:<11}" for l in present_labels])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        r_str = f"{present_labels[i]:<14} | " + " | ".join([f"{val:<11}" for val in row])
        print(r_str)

    # Top Failure Causes per Class
    print("\n" + "=" * 65)
    print("TOP FAILURE CAUSES PER CLASS (Perception vs Rule)")
    print("=" * 65)
    print(f"{'Class':<12} | {'Total Errors':<12} | {'Perception Errors':<18} | {'Rule Errors':<12} | Top Root Cause")
    print("-" * 80)

    class_errors = defaultdict(list)
    for d in diagnoses:
        class_errors[d["gt"]].append(d)

    for c in ["active_work", "phone", "call", "discussing", "idle", "walking"]:
        errs = class_errors[c]
        n_tot = len(errs)
        n_percept = sum(1 for e in errs if e["failure_type"] == "PERCEPTION_FAILURE")
        n_rule = sum(1 for e in errs if e["failure_type"] == "RULE_FAILURE")
        top_cause = errs[0]["failure_detail"] if errs else "None (100% precision/recall)"
        print(f"{c:<12} | {n_tot:<12} | {n_percept:<18} | {n_rule:<12} | {top_cause}")

    print("=" * 80 + "\n")
    return cm, diagnoses, present_labels

if __name__ == "__main__":
    run_diagnosis()
