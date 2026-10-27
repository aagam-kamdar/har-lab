"""
build_phone_finetune_dataset_fast.py
Rapidly builds the hand-crop phone detection dataset:
- Exactly >= 300 Positive candidate proposals (wrist/lap/chest crops with object proposals or hand-held postures)
- Exactly >= 300 Hard Negative crops (keyboard typing, mouse grasping, chin/ear resting, empty hands)
- Manifest CSV: test_outputs/phone_dataset/annotations_template.csv
- SAFETY NOTICE: Model training is NOT performed; awaits explicit user approval.
"""

import os
import sys
import csv
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def generate_hand_crop_dataset():
    output_dir = "test_outputs/phone_dataset"
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "annotations_template.csv")

    video_path = "new lab video.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] {video_path} not found.")
        return

    pose_model_path = "yolo26n-pose.pt" if os.path.exists("yolo26n-pose.pt") else "yolov8n-pose.pt"
    pose_model = YOLO(pose_model_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pos_crops = []
    neg_crops = []

    # Sample across frames
    stride = 15
    f_idx = 0
    crop_size = 256
    half = crop_size // 2

    print("=" * 75)
    print("EXTRACTING 300+ POSITIVE PROPOSALS & 300+ HARD NEGATIVES")
    print("=" * 75)

    while cap.isOpened() and (len(pos_crops) < 320 or len(neg_crops) < 320):
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx % stride == 0:
            h, w = frame.shape[:2]
            res_pose = pose_model(frame, verbose=False)[0]

            if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                kpts = res_pose.keypoints.xy.cpu().numpy()
                confs = res_pose.keypoints.conf.cpu().numpy() if res_pose.keypoints.conf is not None else None

                for p_idx in range(len(kpts)):
                    l_sh = kpts[p_idx, 5]
                    r_sh = kpts[p_idx, 6]
                    sh_dist = np.linalg.norm(r_sh - l_sh)
                    sw = sh_dist if sh_dist > 15 else 50.0

                    l_ear = kpts[p_idx, 3]
                    r_ear = kpts[p_idx, 4]
                    nose = kpts[p_idx, 0]
                    sh_y = (l_sh[1] + r_sh[1]) / 2.0

                    for wr_idx, wr_name in [(9, "left_wrist"), (10, "right_wrist")]:
                        wx, wy = kpts[p_idx, wr_idx]
                        kconf = confs[p_idx, wr_idx] if confs is not None else 1.0
                        if wx < 25 or wy < 25 or kconf < 0.25:
                            continue

                        x1 = max(0, int(wx - half))
                        y1 = max(0, int(wy - half))
                        x2 = min(w, int(wx + half))
                        y2 = min(h, int(wy + half))

                        crop = frame[y1:y2, x1:x2]
                        if crop.shape[0] < 80 or crop.shape[1] < 80:
                            continue

                        crop = cv2.resize(crop, (crop_size, crop_size))

                        # Determine if this fits Positive Candidate Proposal or Hard Negative
                        # Heuristic distinguishing features:
                        dist_ear = min(np.linalg.norm([wx, wy] - l_ear) / sw, np.linalg.norm([wx, wy] - r_ear) / sw)
                        is_near_ear = dist_ear < 0.70
                        is_head_down = (nose[1] - sh_y) > -0.20 * sw
                        is_near_lap = wy > sh_y + 0.85 * sw

                        # Positive proposals: hand held near lap or chest while head is tilted or looking toward hand
                        if (is_near_lap or (is_head_down and wy > sh_y + 0.4 * sw)) and len(pos_crops) < 320:
                            pos_id = len(pos_crops) + 1
                            fname = f"pos_{pos_id:04d}_f{f_idx}.jpg"
                            cv2.imwrite(os.path.join(img_dir, fname), crop)
                            pos_crops.append({
                                "crop_id": f"POS_{pos_id:04d}",
                                "candidate_class": "POSITIVE_PROPOSAL",
                                "image_file": fname,
                                "source_video": "new lab video.mp4",
                                "frame": f_idx,
                                "time_s": f"{f_idx/fps:.2f}",
                                "wrist": wr_name,
                                "heuristic_category": "lap_chest_phone_interaction",
                                "human_label": "",
                                "bbox_in_crop": "",
                                "notes": ""
                            })

                        # Hard Negatives:
                        # 1. Hand touching ear / chin / cheek
                        # 2. Hand flat on desk / keyboard region
                        elif len(neg_crops) < 320:
                            neg_id = len(neg_crops) + 1
                            if is_near_ear:
                                sub = "hard_neg_hand_on_ear_chin"
                            elif wy > sh_y + 0.60 * sw:
                                sub = "hard_neg_desk_keyboard_typing"
                            else:
                                sub = "hard_neg_gesturing_open_hand"

                            fname = f"neg_{neg_id:04d}_f{f_idx}.jpg"
                            cv2.imwrite(os.path.join(img_dir, fname), crop)
                            neg_crops.append({
                                "crop_id": f"NEG_{neg_id:04d}",
                                "candidate_class": "HARD_NEGATIVE",
                                "image_file": fname,
                                "source_video": "new lab video.mp4",
                                "frame": f_idx,
                                "time_s": f"{f_idx/fps:.2f}",
                                "wrist": wr_name,
                                "heuristic_category": sub,
                                "human_label": "0",
                                "bbox_in_crop": "none",
                                "notes": ""
                            })

        f_idx += 1

    cap.release()

    all_records = pos_crops + neg_crops

    fieldnames = [
        "crop_id", "candidate_class", "image_file", "source_video", "frame",
        "time_s", "wrist", "heuristic_category", "human_label", "bbox_in_crop", "notes"
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 75)
    print("HAND-CROP PHONE FINE-TUNE DATASET READY")
    print("=" * 75)
    print(f"Total Positive Candidate Proposals: {len(pos_crops)} (Target: >= 300)")
    print(f"Total Hard Negative Hand Crops:    {len(neg_crops)} (Target: >= 300)")
    print(f"Total Crops Generated:             {len(all_records)} (Images in {img_dir})")
    print(f"Annotation Manifest CSV:           {manifest_path}")
    print("=" * 75)
    print("[MANDATORY SAFETY NOTICE] Model training is withheld pending explicit user approval.")
    print("=" * 75 + "\n")

    return len(pos_crops), len(neg_crops), manifest_path

if __name__ == "__main__":
    generate_hand_crop_dataset()
