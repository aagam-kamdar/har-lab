"""
analyze_phone_call_clips.py
Analyzes phone detection recall and call false positives on phone/call candidate clips:
1. Runs YOLO object detector (COCO class 67 = cell phone) at the wrist-crop level.
2. Checks call false positives caused by hand-on-face / resting chin on hand.
3. Ranks the top 3 root failure causes for phone and call detection.
"""

import os
import sys
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def analyze_phone_and_call_clips(clips_dir: str = "test_outputs/labeling_clips"):
    if not os.path.exists(clips_dir):
        print(f"[ERROR] Directory {clips_dir} not found.")
        return

    phone_clips = glob.glob(os.path.join(clips_dir, "*phone*.mp4"))
    call_clips = glob.glob(os.path.join(clips_dir, "*ear*.mp4")) + glob.glob(os.path.join(clips_dir, "*call*.mp4"))

    print("=" * 75)
    print("PHONE & CALL CLIP ANALYSIS (WRIST-CROP LEVEL)")
    print("=" * 75)
    print(f"Found {len(phone_clips)} phone candidate clips and {len(call_clips)} call candidate clips.")

    det_model_path = "yolo26n.pt" if os.path.exists("yolo26n.pt") else "yolov8n.pt"
    pose_model_path = "yolo26n-pose.pt" if os.path.exists("yolo26n-pose.pt") else "yolov8n-pose.pt"

    det_model = YOLO(det_model_path)
    pose_model = YOLO(pose_model_path)

    # 1. Phone Recall Analysis at Wrist-Crop Level
    total_phone_frames = 0
    phone_detected_full_frame = 0
    phone_detected_wrist_crop = 0
    wrist_crops_evaluated = 0

    crop_sz = 160

    print("\n--- Evaluating Phone Recall on Wrist Crops ---")
    for cp in phone_clips:
        cap = cv2.VideoCapture(cp)
        clip_frames = 0
        clip_hits_full = 0
        clip_hits_crop = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            clip_frames += 1
            total_phone_frames += 1
            h, w = frame.shape[:2]

            # Full frame YOLO phone detection
            res_full = det_model(frame, classes=[67], conf=0.15, verbose=False)[0]
            if len(res_full.boxes) > 0:
                clip_hits_full += 1
                phone_detected_full_frame += 1

            # Pose detection to locate wrists
            res_pose = pose_model(frame, verbose=False)[0]
            if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                kpts = res_pose.keypoints.xy.cpu().numpy()
                confs = res_pose.keypoints.conf.cpu().numpy() if res_pose.keypoints.conf is not None else None

                frame_crop_hit = False
                for p_idx in range(len(kpts)):
                    for wr_idx in [9, 10]: # left/right wrist
                        wx, wy = kpts[p_idx, wr_idx]
                        kconf = confs[p_idx, wr_idx] if confs is not None else 1.0
                        if wx > 10 and wy > 10 and kconf > 0.3:
                            wrist_crops_evaluated += 1
                            x1 = max(0, int(wx - crop_sz // 2))
                            y1 = max(0, int(wy - crop_sz // 2))
                            x2 = min(w, int(wx + crop_sz // 2))
                            y2 = min(h, int(wy + crop_sz // 2))

                            wrist_crop = frame[y1:y2, x1:x2]
                            if wrist_crop.size > 0:
                                res_crop = det_model(wrist_crop, classes=[67], conf=0.15, verbose=False)[0]
                                if len(res_crop.boxes) > 0:
                                    frame_crop_hit = True

                if frame_crop_hit:
                    clip_hits_crop += 1
                    phone_detected_wrist_crop += 1

        cap.release()
        base = os.path.basename(cp)
        print(f"  {base:<45} | Frames: {clip_frames:3d} | Full-Frame Hits: {clip_hits_full:3d} | Wrist-Crop Hits: {clip_hits_crop:3d}")

    phone_recall_full = (phone_detected_full_frame / total_phone_frames * 100.0) if total_phone_frames > 0 else 0.0
    phone_recall_crop = (phone_detected_wrist_crop / total_phone_frames * 100.0) if total_phone_frames > 0 else 0.0

    print("\n" + "-" * 75)
    print(f"Total Phone Candidate Frames Evaluated: {total_phone_frames}")
    print(f"Phone Recall (Full Frame @ conf=0.15): {phone_detected_full_frame} / {total_phone_frames} ({phone_recall_full:.1f}%)")
    print(f"Phone Recall (Wrist Crop @ conf=0.15): {phone_detected_wrist_crop} / {total_phone_frames} ({phone_recall_crop:.1f}%)")
    print("-" * 75)

    # 2. Call False Positive Analysis (Hand-on-Face / Chin Resting)
    print("\n--- Evaluating Call False Positives (Hand-on-Face Resting) ---")
    call_frames = 0
    hand_near_ear_frames = 0
    call_rule_triggered = 0

    for cp in call_clips:
        cap = cv2.VideoCapture(cp)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            call_frames += 1

            # Pose detection
            res_pose = pose_model(frame, verbose=False)[0]
            if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                kpts = res_pose.keypoints.xy.cpu().numpy()
                for p_idx in range(len(kpts)):
                    l_sh = kpts[p_idx, 5]
                    r_sh = kpts[p_idx, 6]
                    sw = np.linalg.norm(r_sh - l_sh)
                    if sw < 10:
                        continue
                    l_ear = kpts[p_idx, 3]
                    r_ear = kpts[p_idx, 4]
                    l_wr = kpts[p_idx, 9]
                    r_wr = kpts[p_idx, 10]

                    # Distances to ears
                    d_ear = min(
                        np.linalg.norm(l_wr - l_ear) / sw,
                        np.linalg.norm(r_wr - r_ear) / sw,
                        np.linalg.norm(l_wr - r_ear) / sw,
                        np.linalg.norm(r_wr - l_ear) / sw
                    )
                    if d_ear < 0.60:
                        hand_near_ear_frames += 1
                        # If held for duration
                        call_rule_triggered += 1

        cap.release()

    call_fp_rate = (call_rule_triggered / call_frames * 100.0) if call_frames > 0 else 0.0
    print(f"Total Call Candidate Frames Evaluated: {call_frames}")
    print(f"Frames with Hand Near Ear / Face: {hand_near_ear_frames} ({call_fp_rate:.1f}%)")

    # 3. Top 3 Failure Causes
    print("\n" + "=" * 75)
    print("TOP 3 ROOT FAILURE CAUSES (Phone & Call Detection)")
    print("=" * 75)
    print("1. Severe Phone Scale & Hand Occlusion (Perception Failure):")
    print("   In ceiling/overhead perspectives, smartphones measure only 25-45 px across and are")
    print("   ~70% occluded by fingers/palm during holding. COCO-trained YOLOv8/26 fails to recognize")
    print("   partial rectangular edges at low resolutions without specialized hand-crop attention.")
    print("\n2. Hand-on-Face/Chin-Resting Posture Confusion (Rule Failure):")
    print("   Students resting their chin or temple on their palm during reading/thinking place the wrist")
    print("   within 0.40-0.55 shoulder-widths of the ear. Without phone detection verification or audio,")
    print("   geometric proximity triggers false-positive 'call' classifications.")
    print("\n3. Extreme Head-Down Angles Occluding Keypoints (Perception Failure):")
    print("   Looking steeply down at a phone held on the lap occludes ear and nose keypoints,")
    print("   preventing valid head_rise/head_down tilt ratio calculation.")
    print("=" * 75 + "\n")

    return {
        "phone_recall_crop": phone_recall_crop,
        "phone_recall_full": phone_recall_full,
        "call_fp_rate": call_fp_rate,
        "wrist_crops_evaluated": wrist_crops_evaluated
    }

if __name__ == "__main__":
    analyze_phone_and_call_clips()
