"""
test_phone_recall_native_res.py
Evaluates phone detection recall across three resolution modes:
1. Downscaled to 640px (pipeline default: imgsz=640)
2. Native Source Resolution (imgsz=1120, 1:1 full-frame pixels)
3. Native Wrist Crops (1:1 pixel scale, hand-centered)
Reports the exact delta and impact of resolution.
"""

import os
import sys
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def run_native_resolution_phone_test():
    clips_dir = "test_outputs/labeling_clips"
    phone_clips = glob.glob(os.path.join(clips_dir, "*phone*.mp4"))

    print("=" * 80)
    print("PHONE DETECTION RECALL: DOWNSCALED (640) VS NATIVE RESOLUTION")
    print("=" * 80)
    print(f"Evaluating on {len(phone_clips)} candidate phone clips...")

    det_model = YOLO("yolo26n.pt" if os.path.exists("yolo26n.pt") else "yolov8n.pt")
    pose_model = YOLO("yolo26n-pose.pt" if os.path.exists("yolo26n-pose.pt") else "yolov8n-pose.pt")

    total_frames = 0
    hits_downscaled_640 = 0
    hits_native_1120 = 0
    hits_native_wrist_crops = 0

    crop_sz = 160

    for cp in phone_clips:
        cap = cv2.VideoCapture(cp)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            h, w = frame.shape[:2]

            # 1. Pipeline default: imgsz=640 (downscales 1120x480 -> 640x288)
            res_640 = det_model(frame, classes=[67], conf=0.15, imgsz=640, verbose=False)[0]
            if len(res_640.boxes) > 0:
                hits_downscaled_640 += 1

            # 2. Native resolution: imgsz=max(h, w) (preserves 1:1 pixel grid)
            res_native = det_model(frame, classes=[67], conf=0.15, imgsz=max(h, w), verbose=False)[0]
            if len(res_native.boxes) > 0:
                hits_native_1120 += 1

            # 3. Native Wrist Crops
            res_pose = pose_model(frame, imgsz=max(h, w), verbose=False)[0]
            if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                kpts = res_pose.keypoints.xy.cpu().numpy()
                confs = res_pose.keypoints.conf.cpu().numpy() if res_pose.keypoints.conf is not None else None

                frame_crop_hit = False
                for p_idx in range(len(kpts)):
                    for wr_idx in [9, 10]:
                        wx, wy = kpts[p_idx, wr_idx]
                        kconf = confs[p_idx, wr_idx] if confs is not None else 1.0
                        if wx > 10 and wy > 10 and kconf > 0.25:
                            x1 = max(0, int(wx - crop_sz // 2))
                            y1 = max(0, int(wy - crop_sz // 2))
                            x2 = min(w, int(wx + crop_sz // 2))
                            y2 = min(h, int(wy + crop_sz // 2))
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                res_crop = det_model(crop, classes=[67], conf=0.15, imgsz=crop_sz, verbose=False)[0]
                                if len(res_crop.boxes) > 0:
                                    frame_crop_hit = True

                if frame_crop_hit:
                    hits_native_wrist_crops += 1

        cap.release()

    r_640 = (hits_downscaled_640 / total_frames * 100.0) if total_frames > 0 else 0.0
    r_native = (hits_native_1120 / total_frames * 100.0) if total_frames > 0 else 0.0
    r_crops = (hits_native_wrist_crops / total_frames * 100.0) if total_frames > 0 else 0.0

    print("\n" + "-" * 80)
    print(f"Total Frames Evaluated: {total_frames}")
    print(f"1. Downscaled Full-Frame (imgsz=640)   : {hits_downscaled_640:3d} / {total_frames} ({r_640:.1f}%)")
    print(f"2. Native Full-Frame (imgsz=1120)     : {hits_native_1120:3d} / {total_frames} ({r_native:.1f}%)")
    print(f"3. Native Wrist Crops (160x160 1:1)   : {hits_native_wrist_crops:3d} / {total_frames} ({r_crops:.1f}%)")
    print("-" * 80)
    print(f"Resolution Gain (Native vs Downscaled): {r_native - r_640:+.1f}% on Full Frame, {r_crops - r_640:+.1f}% on Wrist Crops")
    print("=" * 80 + "\n")

    return {
        "total_frames": total_frames,
        "hits_640": hits_downscaled_640,
        "hits_native": hits_native_1120,
        "hits_crops": hits_native_wrist_crops,
        "gain_native_pct": r_native - r_640,
        "gain_crops_pct": r_crops - r_640
    }

if __name__ == "__main__":
    run_native_resolution_phone_test()
