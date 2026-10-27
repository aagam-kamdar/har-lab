"""
labeling_helper.py
Builds an active labeling helper:
1. Identifies borderline, low-confidence, or rule-disagreement segments.
2. Exports short video clips (around +/- 1.5s of the ambiguous event) with visual bounding/timestamp overlays.
3. Generates a clean CSV template (ambiguous_segments_template.csv) for manual ground-truth annotation.
"""

import os
import sys
import csv
import cv2
import yaml
import numpy as np
from typing import List, Dict, Tuple

def detect_ambiguities_from_features(csv_path: str, th_path: str = "thresholds.yaml") -> List[Dict]:
    """Scans test_features.csv to identify ambiguous/borderline segments."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] Features file {csv_path} not found.")
        return []

    th = {}
    if os.path.exists(th_path):
        with open(th_path, "r") as f:
            th = yaml.safe_load(f)

    phone_hand_th = th.get("phone_hand", 1.10)
    call_ear_th = th.get("call_ear", 0.60)
    call_hold_th = th.get("call_hold", 2.5)
    talk_motion_th = th.get("talk_motion", 0.25)
    kb_motion_th = th.get("kb_motion", 0.12)
    active_grace_th = th.get("active_grace", 6.0)
    walk_speed_th = th.get("walk_speed", 0.55)

    ambiguous_records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader):
            t = float(r["t"])
            student = r["student"]
            seat = r.get("seat", "")
            raw = r["raw"]

            wrist_phone = float(r.get("wrist_phone", 99.0))
            wrist_ear = float(r.get("wrist_ear", 99.0))
            head_down = r.get("head_down", "False").lower() == "true"
            turn_dir = int(r.get("turn_dir", 0))
            hands_low = r.get("hands_low", "False").lower() == "true"
            on_kb = r.get("on_kb", "False").lower() == "true"
            motion = float(r.get("motion", 0.0))
            speed = float(r.get("speed", 0.0))
            standing = r.get("standing", "False").lower() == "true"
            ear_hold = float(r.get("ear_hold", 0.0))
            since_motion = float(r.get("since_motion", 999.0))

            ambiguity_type = None
            reasons = []

            # 1. Borderline Phone
            if 0.85 <= wrist_phone <= 1.40:
                ambiguity_type = "borderline_phone_distance"
                reasons.append(f"wrist_phone={wrist_phone:.2f} near threshold {phone_hand_th:.2f}")
            elif wrist_phone < phone_hand_th and not head_down:
                ambiguity_type = "phone_without_head_down"
                reasons.append(f"wrist_phone={wrist_phone:.2f} but head_down=False")

            # 2. Borderline Call
            if 0.45 <= wrist_ear <= 0.80:
                ambiguity_type = "borderline_ear_proximity"
                reasons.append(f"wrist_ear={wrist_ear:.2f} near threshold {call_ear_th:.2f}")
            elif wrist_ear < call_ear_th and (1.0 <= ear_hold < call_hold_th):
                ambiguity_type = "short_call_hold"
                reasons.append(f"wrist_ear={wrist_ear:.2f} but ear_hold={ear_hold:.1f}s < {call_hold_th:.1f}s")

            # 3. Borderline Discussion vs Keyboard
            if turn_dir != 0 and (kb_motion_th <= motion <= talk_motion_th + 0.10):
                ambiguity_type = "turned_head_with_motion"
                reasons.append(f"turn_dir={turn_dir} with motion={motion:.2f} (near kb/talk boundary)")

            # 4. Borderline Active Work vs Idle
            if (hands_low or on_kb) and (0.08 <= motion <= 0.16):
                ambiguity_type = "low_motion_typing"
                reasons.append(f"hands_low/on_kb=True but subtle motion={motion:.2f} near {kb_motion_th:.2f}")
            elif (hands_low or on_kb) and (abs(since_motion - active_grace_th) < 1.5):
                ambiguity_type = "grace_period_expiry"
                reasons.append(f"since_motion={since_motion:.1f}s near grace limit {active_grace_th:.1f}s")

            # 5. Borderline Walking vs Seated Motion
            if abs(speed - walk_speed_th) < 0.12 and not standing:
                ambiguity_type = "seated_speed_spike"
                reasons.append(f"speed={speed:.2f} near walk threshold {walk_speed_th:.2f} while seated")

            if ambiguity_type:
                ambiguous_records.append({
                    "row_idx": idx,
                    "t": t,
                    "student": student,
                    "seat": seat,
                    "raw_label": raw,
                    "ambiguity_type": ambiguity_type,
                    "reason": "; ".join(reasons),
                    "features": {
                        "wrist_phone": wrist_phone, "wrist_ear": wrist_ear,
                        "motion": motion, "speed": speed, "turn_dir": turn_dir,
                        "ear_hold": ear_hold, "since_motion": since_motion
                    }
                })

    return ambiguous_records


def export_clips_and_template(
    video_path: str = "new lab video.mp4",
    features_csv: str = "test_features.csv",
    output_dir: str = "test_outputs/labeling_clips",
    template_csv: str = "test_outputs/ambiguous_segments_template.csv",
    max_clips: int = 15,
    half_window_s: float = 1.5
):
    """Generates trimmed MP4 video clips and corresponding CSV template for human labeling."""
    os.makedirs(output_dir, exist_ok=True)
    ambiguities = detect_ambiguities_from_features(features_csv)
    print(f"[INFO] Detected {len(ambiguities)} candidate ambiguous segments from {features_csv}.")

    if not ambiguities:
        print("[INFO] No ambiguous segments detected.")
        return

    # Filter/subsample so clips are spaced out across students and ambiguity types
    selected: List[Dict] = []
    seen_keys = set()
    for a in ambiguities:
        key = (a["student"], a["ambiguity_type"])
        if key not in seen_keys:
            seen_keys.add(key)
            selected.append(a)
            if len(selected) >= max_clips:
                break
    # If more clips allowed, add other time offsets
    if len(selected) < max_clips:
        last_t_per_stud = {}
        for a in ambiguities:
            stud = a["student"]
            if a["t"] - last_t_per_stud.get(stud, -999.0) >= 0.4:
                if a not in selected:
                    selected.append(a)
                    last_t_per_stud[stud] = a["t"]
                    if len(selected) >= max_clips:
                        break

    print(f"[INFO] Selected {len(selected)} diverse segments for clip generation.")

    if not os.path.exists(video_path):
        print(f"[WARNING] Video file {video_path} not found. Skipping video clip extraction, writing CSV template only.")
        video_available = False
    else:
        video_available = True

    cap = cv2.VideoCapture(video_path) if video_available else None
    fps = cap.get(cv2.CAP_PROP_FPS) if video_available else 30.0
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if video_available else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if video_available else 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if video_available else 720

    csv_rows = []
    for clip_idx, item in enumerate(selected, start=1):
        center_t = item["t"]
        student = item["student"]
        amb_type = item["ambiguity_type"]
        reason = item["reason"]
        raw_label = item["raw_label"]

        start_t = max(0.0, center_t - half_window_s)
        end_t = center_t + half_window_s

        clip_filename = f"clip_{clip_idx:02d}_{student}_{amb_type}.mp4"
        clip_filepath = os.path.join(output_dir, clip_filename)

        if video_available and cap is not None:
            start_frame = int(start_t * fps)
            end_frame = min(total_frames - 1, int(end_t * fps))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(clip_filepath, fourcc, fps, (w, h))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_f = start_frame
            while current_f <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                cur_t = current_f / fps
                # Overlay banner for human annotator
                banner = f"Clip #{clip_idx:02d} | {student} | t={cur_t:.2f}s | Rule: {raw_label} | Reason: {reason}"
                cv2.rectangle(frame, (10, 10), (w - 10, 50), (0, 0, 0), -1)
                cv2.putText(frame, banner, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                out.write(frame)
                current_f += 1

            out.release()
            print(f"  [SAVED] {clip_filepath} ({end_t - start_t:.1f}s)")

        csv_rows.append({
            "clip_id": f"CLIP-{clip_idx:03d}",
            "video_file": video_path,
            "clip_file": clip_filepath if video_available else "N/A",
            "start_time_s": f"{start_t:.2f}",
            "end_time_s": f"{end_t:.2f}",
            "target_student": student,
            "ambiguity_type": amb_type,
            "rule_prediction": raw_label,
            "ambiguity_evidence": reason,
            "ground_truth_label": "",   # User annotator fills this in
            "confidence_rating (1-5)": "", # User annotator fills this in
            "notes": ""                 # User annotator fills this in
        })

    if cap is not None:
        cap.release()

    # Write CSV template
    os.makedirs(os.path.dirname(template_csv) if os.path.dirname(template_csv) else ".", exist_ok=True)
    fieldnames = [
        "clip_id", "video_file", "clip_file", "start_time_s", "end_time_s",
        "target_student", "ambiguity_type", "rule_prediction", "ambiguity_evidence",
        "ground_truth_label", "confidence_rating (1-5)", "notes"
    ]
    with open(template_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[SUCCESS] Exported {len(csv_rows)} clips and generated CSV template at: {template_csv}")

if __name__ == "__main__":
    export_clips_and_template()
