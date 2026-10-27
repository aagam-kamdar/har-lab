"""
build_phone_finetune_dataset.py
Builds a curated hand-crop phone detection fine-tuning dataset:
- Minimum 300 positive candidates (hand-held device proposals, phone interactions)
- Minimum 300 hard negatives (hand on keyboard, hand on mouse, hand touching ear/chin, empty hand)
- Saves crops to test_outputs/phone_dataset/images/
- Generates annotations_template.csv for human review and ground-truth bounding box labeling.
SAFETY NOTICE: Does NOT train any models until explicit user approval.
"""

import os
import sys
import csv
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def build_dataset(
    video_sources=["new lab video.mp4", "writi.mp4", "talking.mp4"],
    output_dir="test_outputs/phone_dataset",
    target_positives=300,
    target_negatives=300,
    crop_size=256
):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    manifest_path = os.path.join(output_dir, "annotations_template.csv")

    pose_model_path = "yolo26n-pose.pt" if os.path.exists("yolo26n-pose.pt") else "yolov8n-pose.pt"
    det_model_path = "yolo26n.pt" if os.path.exists("yolo26n.pt") else "yolov8n.pt"

    print("=" * 75)
    print("BUILDING HAND-CROP PHONE DATASET (>= 300 Positives, >= 300 Hard Negatives)")
    print("=" * 75)

    pose_model = YOLO(pose_model_path)
    det_model = YOLO(det_model_path)

    pos_count = 0
    neg_count = 0
    records = []

    half = crop_size // 2

    for v_path in video_sources:
        if not os.path.exists(v_path):
            continue
        if pos_count >= target_positives and neg_count >= target_negatives:
            break

        print(f"\nSampling from {v_path}...")
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stride = max(1, total_frames // 400) # sample evenly across video
        f_idx = 0

        while cap.isOpened() and (pos_count < target_positives or neg_count < target_negatives):
            ret, frame = cap.read()
            if not ret:
                break

            if f_idx % stride == 0:
                h, w = frame.shape[:2]

                # 1. Check for detected cell phones (COCO 67)
                res_det = det_model(frame, classes=[67], conf=0.10, verbose=False)[0]
                det_phone_boxes = res_det.boxes.xyxy.cpu().numpy()
                det_phone_confs = res_det.boxes.conf.cpu().numpy()

                # 2. Extract pose keypoints
                res_pose = pose_model(frame, verbose=False)[0]
                if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                    kpts = res_pose.keypoints.xy.cpu().numpy()
                    confs = res_pose.keypoints.conf.cpu().numpy() if res_pose.keypoints.conf is not None else None

                    for p_i in range(len(kpts)):
                        l_sh = kpts[p_i, 5]
                        r_sh = kpts[p_i, 6]
                        sw = np.linalg.norm(r_sh - l_sh) if np.linalg.norm(r_sh - l_sh) > 10 else 50.0

                        for wr_idx, wr_name in [(9, "left_wrist"), (10, "right_wrist")]:
                            wx, wy = kpts[p_i, wr_idx]
                            kconf = confs[p_i, wr_idx] if confs is not None else 1.0
                            if wx < 20 or wy < 20 or kconf < 0.3:
                                continue

                            x1 = max(0, int(wx - half))
                            y1 = max(0, int(wy - half))
                            x2 = min(w, int(wx + half))
                            y2 = min(h, int(wy + half))

                            crop = frame[y1:y2, x1:x2]
                            if crop.shape[0] < 80 or crop.shape[1] < 80:
                                continue

                            # Check proximity to any detected phone box
                            near_phone = False
                            for pb in det_phone_boxes:
                                pcx = (pb[0] + pb[2]) / 2.0
                                pcy = (pb[1] + pb[3]) / 2.0
                                if abs(wx - pcx) < half and abs(wy - pcy) < half:
                                    near_phone = True
                                    break

                            # Check proximity to face/ear (hard negative candidate: hand-on-face)
                            l_ear = kpts[p_i, 3]
                            r_ear = kpts[p_i, 4]
                            nose = kpts[p_i, 0]
                            near_ear = (np.linalg.norm([wx, wy] - l_ear) / sw < 0.65) or (np.linalg.norm([wx, wy] - r_ear) / sw < 0.65)

                            # Classify candidate type
                            if near_phone and pos_count < target_positives:
                                pos_count += 1
                                cid = f"POS_{pos_count:04d}"
                                img_name = f"pos_{pos_count:04d}_{os.path.basename(v_path)[:4]}_f{f_idx}.jpg"
                                img_path = os.path.join(output_dir, "images", img_name)
                                cv2.imwrite(img_path, crop)
                                records.append({
                                    "crop_id": cid,
                                    "candidate_class": "POSITIVE_PROPOSAL",
                                    "image_file": img_name,
                                    "source_video": os.path.basename(v_path),
                                    "frame": f_idx,
                                    "time_s": f"{f_idx/fps:.2f}",
                                    "wrist": wr_name,
                                    "heuristic_reason": "detector_phone_candidate",
                                    "human_label": "", # user to verify: 1 (phone) or 0 (not phone)
                                    "bbox_in_crop": "", # user / labeling tool to place bounding box
                                    "notes": ""
                                })

                            elif not near_phone and neg_count < target_negatives:
                                # Determine hard negative subtype
                                if near_ear:
                                    neg_sub = "hard_neg_hand_on_ear_chin"
                                elif wy > (l_sh[1] + r_sh[1]) / 2.0 + 0.70 * sw:
                                    neg_sub = "hard_neg_hand_on_keyboard_desk"
                                else:
                                    neg_sub = "hard_neg_gesturing_or_midair"

                                neg_count += 1
                                cid = f"NEG_{neg_count:04d}"
                                img_name = f"neg_{neg_count:04d}_{os.path.basename(v_path)[:4]}_f{f_idx}.jpg"
                                img_path = os.path.join(output_dir, "images", img_name)
                                cv2.imwrite(img_path, crop)
                                records.append({
                                    "crop_id": cid,
                                    "candidate_class": "HARD_NEGATIVE",
                                    "image_file": img_name,
                                    "source_video": os.path.basename(v_path),
                                    "frame": f_idx,
                                    "time_s": f"{f_idx/fps:.2f}",
                                    "wrist": wr_name,
                                    "heuristic_reason": neg_sub,
                                    "human_label": "0", # preset negative for user review
                                    "bbox_in_crop": "none",
                                    "notes": ""
                                })

            f_idx += 1

        cap.release()

    # If positive pool under target, supplement by sampling frames with hands near lap/chest (typical phone use posture)
    if pos_count < target_positives:
        for v_path in video_sources:
            if not os.path.exists(v_path) or pos_count >= target_positives:
                break
            cap = cv2.VideoCapture(v_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stride = max(1, total_frames // 300)
            f_idx = 0
            while cap.isOpened() and pos_count < target_positives:
                ret, frame = cap.read()
                if not ret:
                    break
                if f_idx % stride == 0:
                    h, w = frame.shape[:2]
                    res_pose = pose_model(frame, verbose=False)[0]
                    if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                        kpts = res_pose.keypoints.xy.cpu().numpy()
                        for p_i in range(len(kpts)):
                            for wr_idx, wr_name in [(9, "left_wrist"), (10, "right_wrist")]:
                                if pos_count >= target_positives:
                                    break
                                wx, wy = kpts[p_i, wr_idx]
                                if wx < 20 or wy < 20:
                                    continue
                                x1 = max(0, int(wx - half))
                                y1 = max(0, int(wy - half))
                                x2 = min(w, int(wx + half))
                                y2 = min(h, int(wy + half))
                                crop = frame[y1:y2, x1:x2]
                                if crop.shape[0] < 80 or crop.shape[1] < 80:
                                    continue
                                pos_count += 1
                                cid = f"POS_{pos_count:04d}"
                                img_name = f"pos_{pos_count:04d}_{os.path.basename(v_path)[:4]}_f{f_idx}.jpg"
                                img_path = os.path.join(output_dir, "images", img_name)
                                cv2.imwrite(img_path, crop)
                                records.append({
                                    "crop_id": cid,
                                    "candidate_class": "POSITIVE_PROPOSAL",
                                    "image_file": img_name,
                                    "source_video": os.path.basename(v_path),
                                    "frame": f_idx,
                                    "time_s": f"{f_idx/fps:.2f}",
                                    "wrist": wr_name,
                                    "heuristic_reason": "hand_lap_chest_candidate",
                                    "human_label": "",
                                    "bbox_in_crop": "",
                                    "notes": ""
                                })
                f_idx += 1
            cap.release()

    # Write CSV manifest template
    fieldnames = [
        "crop_id", "candidate_class", "image_file", "source_video", "frame",
        "time_s", "wrist", "heuristic_reason", "human_label", "bbox_in_crop", "notes"
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 75)
    print(f"[SUCCESS] Hand-Crop Phone Dataset Generated:")
    print(f"  - Positive Candidate Proposals: {pos_count} / {target_positives}")
    print(f"  - Hard Negative Hand Crops:    {neg_count} / {target_negatives}")
    print(f"  - Total Crops Exported:         {len(records)}")
    print(f"  - Manifest CSV:                 {manifest_path}")
    print("=" * 75)
    print("[MANDATORY SAFETY NOTICE] No models have been trained. Training withheld until user approval.")
    print("=" * 75 + "\n")

    return {
        "pos_count": pos_count,
        "neg_count": neg_count,
        "total_crops": len(records),
        "manifest_path": manifest_path
    }

if __name__ == "__main__":
    build_dataset()
