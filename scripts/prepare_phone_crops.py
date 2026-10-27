"""
prepare_phone_crops.py
Prepares a fine-tuning dataset for phone detection in lab environments:
1. Samples video frames from lab cameras (e.g. new lab video.mp4).
2. Extracts wrist/hand and desk-lap crops where phones are typically held.
3. Automatically proposes candidate labels with bounding boxes for user correction.
4. Generates an annotation manifest (phone_crops_manifest.csv).
IMPORTANT: Per safety constraint, this script ONLY prepares candidate crops for human review.
It DOES NOT train or fine-tune any model until explicit user approval.
"""

import os
import sys
import csv
import cv2
import numpy as np
from ultralytics import YOLO

def prepare_phone_crops(
    video_path: str = "new lab video.mp4",
    output_dir: str = "test_outputs/phone_finetune_candidates",
    max_crops: int = 50,
    sample_stride_frames: int = 45,
    crop_size: int = 256
):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    manifest_csv = os.path.join(output_dir, "phone_crops_manifest.csv")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video {video_path} not found.")
        return

    print(f"[INFO] Scanning {video_path} to prepare phone detection fine-tune candidates...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Load YOLO pose to locate wrists
    pose_model_path = "yolo26n-pose.pt" if os.path.exists("yolo26n-pose.pt") else "yolov8n-pose.pt"
    det_model_path = "yolo26n.pt" if os.path.exists("yolo26n.pt") else "yolov8n.pt"

    pose_model = YOLO(pose_model_path)
    det_model = YOLO(det_model_path)

    crops_saved = 0
    frame_idx = 0
    records = []

    # COCO cell phone class is 67
    PHONE_CLASS_ID = 67

    while cap.isOpened() and frame_idx < total_frames and crops_saved < max_crops:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_stride_frames == 0:
            h, w = frame.shape[:2]

            # 1. Detect cell phones with low threshold to catch weak proposals
            det_res = det_model(frame, classes=[PHONE_CLASS_ID], conf=0.15, verbose=False)[0]
            detected_phones = det_res.boxes.xyxy.cpu().numpy()
            det_confs = det_res.boxes.conf.cpu().numpy()

            # 2. Extract pose wrists (keypoints 9 and 10 in COCO 17-keypoint skeleton)
            pose_res = pose_model(frame, verbose=False)[0]

            candidate_centers = []
            # Add detected phone centers
            for box, conf in zip(detected_phones, det_confs):
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                candidate_centers.append((cx, cy, f"detector_conf_{conf:.2f}", float(conf), "phone"))

            # Add wrist locations as candidate regions
            if pose_res.keypoints is not None and len(pose_res.keypoints) > 0:
                kpts = pose_res.keypoints.xy.cpu().numpy() # [N, 17, 2]
                confs_kp = pose_res.keypoints.conf.cpu().numpy() if pose_res.keypoints.conf is not None else None

                for p_idx in range(len(kpts)):
                    for wrist_idx in [9, 10]: # left_wrist, right_wrist
                        wx, wy = kpts[p_idx, wrist_idx]
                        kconf = confs_kp[p_idx, wrist_idx] if confs_kp is not None else 1.0
                        if wx > 10 and wy > 10 and kconf > 0.3:
                            candidate_centers.append((int(wx), int(wy), f"wrist_anchor_kp{wrist_idx}", 0.0, "unknown_hand_region"))

            # Deduplicate close centers
            filtered_centers = []
            for c in candidate_centers:
                cx, cy, reason, conf, prop_label = c
                if not any(abs(cx - fc[0]) < 60 and abs(cy - fc[1]) < 60 for fc in filtered_centers):
                    filtered_centers.append(c)

            # Crop and save
            half = crop_size // 2
            for cx, cy, reason, conf, prop_label in filtered_centers:
                if crops_saved >= max_crops:
                    break

                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(w, cx + half)
                y2 = min(h, cy + half)

                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] < 50 or crop.shape[1] < 50:
                    continue

                crop_filename = f"crop_{crops_saved+1:04d}_f{frame_idx}.jpg"
                crop_path = os.path.join(output_dir, "images", crop_filename)
                cv2.imwrite(crop_path, crop)

                records.append({
                    "crop_id": f"PHONE-CROP-{crops_saved+1:04d}",
                    "crop_file": crop_path,
                    "source_frame": frame_idx,
                    "video_time_s": f"{frame_idx / fps:.2f}",
                    "proposal_type": reason,
                    "detector_confidence": f"{conf:.3f}",
                    "proposed_label": prop_label,
                    "user_approval (phone/not_phone/ambiguous)": "", # Human annotator
                    "user_corrected_bbox": "",                      # Human annotator
                    "notes": ""
                })
                crops_saved += 1

        frame_idx += 1

    cap.release()

    # Write manifest
    fieldnames = [
        "crop_id", "crop_file", "source_frame", "video_time_s", "proposal_type",
        "detector_confidence", "proposed_label", "user_approval (phone/not_phone/ambiguous)",
        "user_corrected_bbox", "notes"
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Prepared {crops_saved} candidate phone crops in: {output_dir}/images")
    print(f"[SUCCESS] Generated annotation manifest: {manifest_csv}")
    print("[MANDATORY SAFETY NOTICE] Model training is paused pending user review & approval.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    prepare_phone_crops()
