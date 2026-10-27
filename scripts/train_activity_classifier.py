"""
train_activity_classifier.py
Trains a lightweight window classifier (LightGBM) on 2-3s rolling windows of explainable features.
Evaluates using Leave-One-Student-Out (LOSO) cross-validation against the rule cascade.
Exposes feature importance for model explainability.
Enforces the Ship Gate: only ship if LOSO macro-F1 beats the rule cascade.
"""

import os
import sys
import csv
import pickle
import yaml
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import lightgbm as lgb

def extract_features_and_windows(csv_path: str = "test_features.csv", window_size: int = 5):
    """
    Loads test_features.csv and computes rolling window statistics per student sequence.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group rows by student maintaining chronological order
    by_student = defaultdict(list)
    for r in rows:
        by_student[r["student"]].append(r)

    feature_names = [
        # Instantaneous
        "wrist_phone", "wrist_ear", "head_down", "turn", "turn_dir",
        "hands_low", "on_kb", "motion", "speed", "standing", "ear_hold", "since_motion",
        # Rolling window aggregations
        "roll_motion_mean", "roll_motion_max", "roll_motion_std",
        "roll_speed_mean", "roll_speed_max",
        "roll_on_kb_prop", "roll_hands_low_prop", "roll_turn_prop",
        "roll_head_down_prop", "roll_wrist_phone_min"
    ]

    all_X = []
    all_y = []
    all_students = []
    all_rules_pred = []

    # Load thresholds for rule evaluation
    th_path = "thresholds.yaml"
    th = {}
    if os.path.exists(th_path):
        with open(th_path, "r") as f:
            th = yaml.safe_load(f)

    for student, seq in by_student.items():
        # Sort by t
        seq = sorted(seq, key=lambda x: float(x["t"]))
        n = len(seq)

        for i in range(n):
            r = seq[i]
            # Window slice [max(0, i - window_size + 1) : i + 1]
            w_slice = seq[max(0, i - window_size + 1) : i + 1]

            # Instantaneous values
            w_phone = float(r.get("wrist_phone", 99.0))
            w_ear = float(r.get("wrist_ear", 99.0))
            h_down = 1.0 if r.get("head_down", "False").lower() == "true" else 0.0
            turn = float(r.get("turn", 0.0))
            turn_dir = float(r.get("turn_dir", 0))
            h_low = 1.0 if r.get("hands_low", "False").lower() == "true" else 0.0
            on_kb = 1.0 if r.get("on_kb", "False").lower() == "true" else 0.0
            motion = float(r.get("motion", 0.0))
            speed = float(r.get("speed", 0.0))
            standing = 1.0 if r.get("standing", "False").lower() == "true" else 0.0
            ear_hold = float(r.get("ear_hold", 0.0))
            since_motion = float(r.get("since_motion", 999.0))

            # Rolling stats over window
            motions = [float(x.get("motion", 0.0)) for x in w_slice]
            speeds = [float(x.get("speed", 0.0)) for x in w_slice]
            kbs = [1.0 if x.get("on_kb", "False").lower() == "true" else 0.0 for x in w_slice]
            lows = [1.0 if x.get("hands_low", "False").lower() == "true" else 0.0 for x in w_slice]
            turns = [1.0 if int(x.get("turn_dir", 0)) != 0 else 0.0 for x in w_slice]
            downs = [1.0 if x.get("head_down", "False").lower() == "true" else 0.0 for x in w_slice]
            phones = [float(x.get("wrist_phone", 99.0)) for x in w_slice]

            feat_vec = [
                w_phone, w_ear, h_down, turn, turn_dir,
                h_low, on_kb, motion, speed, standing, ear_hold, since_motion,
                float(np.mean(motions)), float(np.max(motions)), float(np.std(motions)),
                float(np.mean(speeds)), float(np.max(speeds)),
                float(np.mean(kbs)), float(np.mean(lows)), float(np.mean(turns)),
                float(np.mean(downs)), float(np.min(phones))
            ]

            all_X.append(feat_vec)
            all_y.append(r["raw"])
            all_students.append(student)

            # Rule prediction
            rule_pred = "idle"
            if speed > th.get("walk_speed", 0.55) or standing > 0.5:
                rule_pred = "walking"
            elif w_ear < th.get("call_ear", 0.60) and ear_hold >= th.get("call_hold", 2.5):
                rule_pred = "call"
            elif w_phone < th.get("phone_hand", 1.10) and h_down > 0.5:
                rule_pred = "phone"
            elif turn_dir != 0 and motion < th.get("talk_motion", 0.25):
                rule_pred = "discussing"
            elif (h_low > 0.5 or on_kb > 0.5) and (motion > th.get("kb_motion", 0.12) or since_motion < th.get("active_grace", 6.0)):
                rule_pred = "active_work"
            else:
                rule_pred = "idle"

            all_rules_pred.append(rule_pred)

    return (
        np.array(all_X, dtype=np.float32),
        np.array(all_y),
        np.array(all_students),
        np.array(all_rules_pred),
        feature_names
    )


def run_loso_evaluation():
    X, y, students, rule_preds, feature_names = extract_features_and_windows()
    unique_students = np.unique(students)
    classes = sorted(list(np.unique(y)))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for i, c in enumerate(classes)}
    y_idx = np.array([label_to_idx[c] for c in y])

    print("=" * 70)
    print("LEAVE-ONE-STUDENT-OUT (LOSO) CROSS-VALIDATION")
    print("=" * 70)
    print(f"Total samples: {len(y)} across {len(unique_students)} students: {list(unique_students)}")
    print(f"Classes: {classes}")
    print(f"Number of windowed features: {len(feature_names)}")

    lgb_preds_all = []
    y_test_all = []
    rule_preds_all = []

    fold_results = []
    feature_importances = np.zeros(len(feature_names), dtype=np.float32)

    for fold_idx, held_out in enumerate(unique_students):
        test_mask = (students == held_out)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y_idx[train_mask]
        X_test, y_test = X[test_mask], y_idx[test_mask]
        rules_test = rule_preds[test_mask]

        clf = lgb.LGBMClassifier(
            n_estimators=40,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.08,
            min_child_samples=3,
            random_state=42,
            verbosity=-1
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        pred_labels = [idx_to_label[p] for p in preds]
        true_labels = [idx_to_label[p] for p in y_test]

        f1_model = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        f1_rules = f1_score(true_labels, rules_test, average="macro", zero_division=0)

        feature_importances += clf.feature_importances_

        fold_results.append({
            "held_out": held_out,
            "n_samples": len(y_test),
            "f1_model": f1_model,
            "f1_rules": f1_rules
        })

        lgb_preds_all.extend(pred_labels)
        y_test_all.extend(true_labels)
        rule_preds_all.extend(rules_test)

    print("\n" + "-" * 70)
    print(f"{'Held-Out Student':<18} | {'N Samples':<10} | {'Rules Macro-F1':<16} | {'LightGBM Macro-F1':<18} | {'Delta':<8}")
    print("-" * 70)
    for r in fold_results:
        delta = r["f1_model"] - r["f1_rules"]
        delta_str = f"{delta:+.3f}"
        print(f"{r['held_out']:<18} | {r['n_samples']:<10} | {r['f1_rules']:<16.3f} | {r['f1_model']:<18.3f} | {delta_str:<8}")

    overall_f1_rules = f1_score(y_test_all, rule_preds_all, average="macro", zero_division=0)
    overall_f1_model = f1_score(y_test_all, lgb_preds_all, average="macro", zero_division=0)
    print("-" * 70)
    print(f"{'OVERALL (LOSO)':<18} | {len(y_test_all):<10} | {overall_f1_rules:<16.3f} | {overall_f1_model:<18.3f} | {overall_f1_model - overall_f1_rules:+.3f}")
    print("=" * 70)

    # Feature Importance
    print("\n" + "=" * 70)
    print("TOP FEATURE IMPORTANCES (LightGBM Window Classifier)")
    print("=" * 70)
    feature_importances /= len(unique_students)
    ranked_indices = np.argsort(feature_importances)[::-1]
    for rank, idx in enumerate(ranked_indices[:10], start=1):
        print(f"  {rank:2d}. {feature_names[idx]:<24} : {feature_importances[idx]:.2f}")

    # Ship Gate Decision
    print("\n" + "=" * 70)
    print("SHIP GATE VERIFICATION (Part B.4)")
    print("=" * 70)
    if overall_f1_model > overall_f1_rules:
        status = "PASSED"
        msg = "LightGBM model strictly beats explainable rules on held-out students. APPROVED to ship as ambiguous case re-ranker."
        # Save production model trained on all data
        os.makedirs("models", exist_ok=True)
        final_clf = lgb.LGBMClassifier(
            n_estimators=40, max_depth=3, num_leaves=7,
            learning_rate=0.08, min_child_samples=3, random_state=42, verbosity=-1
        )
        final_clf.fit(X, y_idx)
        model_payload = {
            "model": final_clf,
            "classes": classes,
            "feature_names": feature_names,
            "overall_f1_model": overall_f1_model,
            "overall_f1_rules": overall_f1_rules,
            "feature_importances": dict(zip(feature_names, feature_importances))
        }
        with open("models/activity_classifier_lgbm.pkl", "wb") as f:
            pickle.dump(model_payload, f)
        print(f"[{status}] {msg}")
        print(f"[INFO] Serialized production model to models/activity_classifier_lgbm.pkl")
    else:
        status = "REJECTED (RULE CASCADE PRESERVED)"
        msg = f"LightGBM model ({overall_f1_model:.3f}) did NOT strictly beat rule cascade ({overall_f1_rules:.3f}) on held-out students. As required by Part B.4, model is NOT shipped; explainable rule cascade remains authoritative."
        print(f"[{status}] {msg}")

    print("=" * 70 + "\n")
    return {
        "overall_f1_rules": overall_f1_rules,
        "overall_f1_model": overall_f1_model,
        "fold_results": fold_results,
        "top_features": [(feature_names[i], float(feature_importances[i])) for i in ranked_indices[:10]]
    }

if __name__ == "__main__":
    run_loso_evaluation()
