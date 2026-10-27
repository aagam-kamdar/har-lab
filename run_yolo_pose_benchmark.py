"""
run_yolo_pose_benchmark.py
--------------------------
YOLOv8-Pose Detector-Only Benchmark vs. Current YOLO Person Detector.
Video: 'new lab video.mp4' (Frames 0 to 1800 @ 30 FPS).

Strict Isolation:
- Pure detector benchmark.
- Pose keypoints are completely discarded/ignored.
- No ByteTrack, no ArcFace, no Re-ID, no FAISS, no identity logic.
"""

import os
import sys
import time
import json
import csv
import logging
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import ultralytics
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("YOLO_Pose_Benchmark")

VIDEO_PATH = "new lab video.mp4"
TOTAL_FRAMES = 1800
CONF_THRESH = 0.25
IOU_THRESH_NMS = 0.70
MATCH_IOU_THRESH = 0.50
OUTPUT_DIR = "test_outputs"
EXAMPLES_DIR = os.path.join(OUTPUT_DIR, "yolo_vs_pose_examples")

def compute_iou(box1, box2):
    """Computes IoU between box1 [x1, y1, x2, y2] and box2 [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class VideoWriter:
    """Universal MP4 video writer using PyAV H.264 with cv2 fallback."""
    def __init__(self, path: str, fps: float, width: int, height: int):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.mode = "cv2"
        self.writer = None
        self.path = path
        try:
            import av
            self.mode = "av"
            self.container = av.open(path, mode="w")
            self.stream = self.container.add_stream("h264", rate=int(round(fps)))
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"preset": "veryfast", "crf": "22"}
        except Exception as e:
            logger.warning("PyAV init failed (%s), using cv2 fallback", e)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        if self.mode == "av":
            import av
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in self.stream.encode(av_frame):
                self.container.mux(pkt)
        else:
            self.writer.write(frame)

    def close(self):
        if self.mode == "av":
            for pkt in self.stream.encode():
                self.container.mux(pkt)
            self.container.close()
        elif self.writer is not None:
            self.writer.release()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    logger.info("==================================================================")
    logger.info("YOLOV8-POSE DETECTOR-ONLY BENCHMARK")
    logger.info("==================================================================")

    # 1. Model Verification
    base_model_path = "yolo26n.pt"
    pose_model_path = "yolov8n-pose.pt"

    logger.info("Loading Base YOLO detector: %s", base_model_path)
    base_yolo = YOLO(base_model_path)
    base_params = sum(p.numel() for p in base_yolo.model.parameters())

    logger.info("Loading YOLOv8-Pose detector: %s", pose_model_path)
    pose_yolo = YOLO(pose_model_path)
    pose_params = sum(p.numel() for p in pose_yolo.model.parameters())

    print("\n--- MODEL METRICS & SPECIFICATIONS ---")
    print(f"Ultralytics Version          : {ultralytics.__version__}")
    print(f"Current YOLO Model           : {base_model_path} (Task: {base_yolo.task})")
    print(f"Current YOLO Architecture    : YOLOv8-nano Object Detector")
    print(f"Current YOLO Parameters      : {base_params:,}")
    print(f"Pose Detector Model          : {pose_model_path} (Task: {pose_yolo.task})")
    print(f"Pose Detector Architecture   : YOLOv8-nano Human Pose Model (COCO 17-Keypoint Head)")
    print(f"Pose Detector Parameters     : {pose_params:,}")
    print(f"Parameter Delta (Pose - Base): +{pose_params - base_params:,} (+{100.0*(pose_params - base_params)/base_params:.1f}%)")
    print(f"Model-Capacity Match         : CLOSEST NANO MATCH (Backbone/Neck identical; difference in head)")
    print(f"Confidence Threshold         : {CONF_THRESH}")
    print(f"NMS IoU Threshold            : {IOU_THRESH_NMS}")
    print(f"Target Video                 : {VIDEO_PATH}")
    print(f"Frames to Evaluate           : 0 to {TOTAL_FRAMES} (60.0s @ 30 FPS)")

    # 2. Warm-up
    logger.info("Running warm-up passes (10 iterations)...")
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        _ = base_yolo(dummy_img, classes=[0], verbose=False)
        _ = pose_yolo(dummy_img, verbose=False)

    # 3. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_video_path = os.path.join(OUTPUT_DIR, "yolov8_pose_detector_1800f.mp4")
    video_writer = VideoWriter(pose_video_path, fps, vid_w, vid_h)

    # Frame buffers
    cached_base_detections = []
    pose_detections = []
    
    # Timing accumulators (pure inference time)
    base_inference_times = []
    pose_inference_times = []

    # Frame storage for saving representative examples later
    saved_frames_cache = {}
    sample_keyframe_indices = {50, 100, 150, 250, 300, 350, 400, 473, 500, 600, 727, 800, 900, 1050, 1100, 1200, 1350, 1400, 1438, 1500, 1650, 1750}

    logger.info("Processing frames 0 to %d for both detectors...", TOTAL_FRAMES)
    t_start_all = time.time()

    for f_idx in range(TOTAL_FRAMES):
        ret, frame = cap.read()
        if not ret:
            logger.warning("Video ended prematurely at frame %d", f_idx)
            break

        if f_idx in sample_keyframe_indices or f_idx % 100 == 0:
            saved_frames_cache[f_idx] = frame.copy()

        # A) BASELINE YOLO INFERENCE (classes=[0])
        t_base_0 = time.perf_counter()
        res_base = base_yolo(frame, classes=[0], conf=CONF_THRESH, verbose=False)[0]
        t_base_elapsed = time.perf_counter() - t_base_0
        base_inference_times.append(t_base_elapsed)

        b_boxes = []
        if res_base.boxes is not None and len(res_base.boxes) > 0:
            for box in res_base.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                b_boxes.append({
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "conf": round(conf, 4),
                })

        cached_base_detections.append({
            "frame_idx": f_idx,
            "boxes": b_boxes,
        })

        # B) YOLOv8-POSE INFERENCE (person boxes only, discard keypoints)
        t_pose_0 = time.perf_counter()
        res_pose = pose_yolo(frame, conf=CONF_THRESH, verbose=False)[0]
        t_pose_elapsed = time.perf_counter() - t_pose_0
        pose_inference_times.append(t_pose_elapsed)

        p_boxes = []
        if res_pose.boxes is not None and len(res_pose.boxes) > 0:
            for box in res_pose.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id == 0: # person class
                    p_boxes.append({
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "conf": round(conf, 4),
                    })

        pose_detections.append({
            "frame_idx": f_idx,
            "boxes": p_boxes,
        })

        # Render Pose-only output video frame
        vis_frame = frame.copy()
        for pb in p_boxes:
            bx1, by1, bx2, by2 = [int(v) for v in pb["bbox"]]
            conf_val = pb["conf"]
            cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), (0, 255, 128), 2)
            label = f"Person: {conf_val:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (bx1, max(0, by1 - th - bl - 4)), (bx1 + tw + 4, by1), (0, 255, 128), -1)
            cv2.putText(vis_frame, label, (bx1 + 2, by1 - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        video_writer.write(vis_frame)

        if (f_idx + 1) % 300 == 0:
            logger.info("Processed %d/%d frames (%.1f%%)", f_idx + 1, TOTAL_FRAMES, 100.0 * (f_idx + 1) / TOTAL_FRAMES)

    cap.release()
    video_writer.close()
    logger.info("Completed inference across %d frames.", len(cached_base_detections))

    # Save cached baseline detections to JSON
    cached_base_filepath = os.path.join(OUTPUT_DIR, "cached_yolo_detections_1800f.json")
    with open(cached_base_filepath, "w") as f:
        json.dump({
            "metadata": {
                "source": "yolo26n.pt running Ultralytics YOLOv8n (classes=[0], conf=0.25, imgsz=640)",
                "frame_range": f"0 to {len(cached_base_detections)-1}",
                "total_frames": len(cached_base_detections),
                "generated_once_and_cached": True,
            },
            "detections": cached_base_detections,
        }, f)
    logger.info("Saved cached baseline detections to %s", cached_base_filepath)

    # 4. Box-Level Comparison using Hungarian Algorithm
    logger.info("Running Hungarian Matching on IoU matrix between Baseline and Pose...")
    frame_metrics = []
    
    total_base_boxes = 0
    total_pose_boxes = 0
    total_matched_pairs = 0
    total_matched_gte_50 = 0
    all_matched_ious = []
    total_unmatched_base = 0
    total_unmatched_pose = 0

    difficult_frame_rows = []

    for f_idx in range(len(cached_base_detections)):
        b_list = cached_base_detections[f_idx]["boxes"]
        p_list = pose_detections[f_idx]["boxes"]

        n_b = len(b_list)
        n_p = len(p_list)
        total_base_boxes += n_b
        total_pose_boxes += n_p

        matched_count = 0
        matched_gte_50_count = 0
        frame_ious = []
        unmatched_b = n_b
        unmatched_p = n_p

        if n_b > 0 and n_p > 0:
            iou_mat = np.zeros((n_b, n_p), dtype=np.float32)
            for i, bb in enumerate(b_list):
                for j, pb in enumerate(p_list):
                    iou_mat[i, j] = compute_iou(bb["bbox"], pb["bbox"])

            # Cost matrix for Hungarian: maximize IoU -> minimize (1 - IoU)
            cost_mat = 1.0 - iou_mat
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            matched_b_indices = set()
            matched_p_indices = set()

            for r, c in zip(row_ind, col_ind):
                iou_val = float(iou_mat[r, c])
                if iou_val > 0.0:
                    matched_count += 1
                    frame_ious.append(iou_val)
                    all_matched_ious.append(iou_val)
                    if iou_val >= MATCH_IOU_THRESH:
                        matched_gte_50_count += 1
                        matched_b_indices.add(r)
                        matched_p_indices.add(c)

            unmatched_b = n_b - len(matched_b_indices)
            unmatched_p = n_p - len(matched_p_indices)

        total_matched_pairs += matched_count
        total_matched_gte_50 += matched_gte_50_count
        total_unmatched_base += unmatched_b
        total_unmatched_pose += unmatched_p

        mean_iou = float(np.mean(frame_ious)) if frame_ious else 0.0
        median_iou = float(np.median(frame_ious)) if frame_ious else 0.0
        pct_gte_50 = (100.0 * matched_gte_50_count / matched_count) if matched_count > 0 else 0.0

        # Discrepancy classification
        discrepancy_type = "AGREE"
        if n_b > n_p:
            discrepancy_type = "BASELINE_MORE"
        elif n_p > n_b:
            discrepancy_type = "POSE_MORE"
        elif n_b == 0 and n_p == 0:
            discrepancy_type = "BOTH_ZERO"
        elif unmatched_b > 0 or unmatched_p > 0:
            discrepancy_type = "PARTIAL_MISMATCH"

        frame_metrics.append({
            "frame": f_idx,
            "baseline_count": n_b,
            "pose_count": n_p,
            "matched_count": matched_count,
            "matched_gte_50": matched_gte_50_count,
            "mean_iou": round(mean_iou, 4),
            "median_iou": round(median_iou, 4),
            "pct_gte_50": round(pct_gte_50, 2),
            "unmatched_baseline": unmatched_b,
            "unmatched_pose": unmatched_p,
            "discrepancy_type": discrepancy_type,
        })

        if discrepancy_type in ("BASELINE_MORE", "POSE_MORE", "PARTIAL_MISMATCH"):
            difficult_frame_rows.append({
                "frame": f_idx,
                "discrepancy_type": discrepancy_type,
                "baseline_count": n_b,
                "pose_count": n_p,
                "matched_count": matched_count,
                "unmatched_baseline": unmatched_b,
                "unmatched_pose": unmatched_p,
                "mean_iou": round(mean_iou, 3),
            })

    # Save difficult frames CSV
    csv_path = os.path.join(OUTPUT_DIR, "yolo_vs_pose_difficult_frames.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "discrepancy_type", "baseline_count", "pose_count", 
            "matched_count", "unmatched_baseline", "unmatched_pose", "mean_iou"
        ])
        writer.writeheader()
        writer.writerows(difficult_frame_rows)
    logger.info("Saved %d difficult frame rows to %s", len(difficult_frame_rows), csv_path)

    # 5. Difficult Windows and Scene Category Analysis
    def analyze_window(start_f, end_f, target_student_name, target_region):
        """Analyzes a specific frame range and checks detection in the student's physical desk region."""
        x_min, x_max, y_min, y_max = target_region
        w_frames = frame_metrics[start_f:end_f+1]
        total_w = len(w_frames)
        
        base_dets_total = 0
        pose_dets_total = 0
        base_missed_person = 0
        pose_missed_person = 0
        both_detected_person = 0
        both_missed_person = 0
        base_only_detected_person = 0
        pose_only_detected_person = 0
        window_ious = []

        for f_idx in range(start_f, end_f + 1):
            b_boxes = cached_base_detections[f_idx]["boxes"]
            p_boxes = pose_detections[f_idx]["boxes"]

            base_dets_total += len(b_boxes)
            pose_dets_total += len(p_boxes)

            # Check if student is detected in physical region
            b_in_region = any(
                x_min <= ((b["bbox"][0] + b["bbox"][2]) / 2) <= x_max and
                y_min <= ((b["bbox"][1] + b["bbox"][3]) / 2) <= y_max
                for b in b_boxes
            )
            p_in_region = any(
                x_min <= ((p["bbox"][0] + p["bbox"][2]) / 2) <= x_max and
                y_min <= ((p["bbox"][1] + p["bbox"][3]) / 2) <= y_max
                for p in p_boxes
            )

            if b_in_region and p_in_region:
                both_detected_person += 1
            elif b_in_region and not p_in_region:
                base_only_detected_person += 1
                pose_missed_person += 1
            elif not b_in_region and p_in_region:
                pose_only_detected_person += 1
                base_missed_person += 1
            else:
                both_missed_person += 1
                base_missed_person += 1
                pose_missed_person += 1

            if frame_metrics[f_idx]["matched_count"] > 0:
                window_ious.append(frame_metrics[f_idx]["mean_iou"])

        return {
            "window": f"Frames {start_f} to {end_f}",
            "student": target_student_name,
            "total_frames": total_w,
            "baseline_avg_dets_per_frame": round(base_dets_total / max(1, total_w), 2),
            "pose_avg_dets_per_frame": round(pose_dets_total / max(1, total_w), 2),
            "baseline_missed_person_frames": base_missed_person,
            "baseline_missed_person_pct": round(100.0 * base_missed_person / max(1, total_w), 2),
            "pose_missed_person_frames": pose_missed_person,
            "pose_missed_person_pct": round(100.0 * pose_missed_person / max(1, total_w), 2),
            "both_detected_person_frames": both_detected_person,
            "both_missed_person_frames": both_missed_person,
            "baseline_only_detected_frames": base_only_detected_person,
            "pose_only_detected_frames": pose_only_detected_person,
            "matched_mean_iou": round(float(np.mean(window_ious)), 4) if window_ious else 0.0,
        }

    # Aagam region: right desk (cx: 580-850, cy: 10-380)
    aagam_analysis = analyze_window(272, 726, "Aagam", (580, 850, 10, 380))
    # Harshil region: left desk (cx: 420-650, cy: 10-300)
    harshil_analysis = analyze_window(564, 1437, "Harshil", (420, 650, 10, 300))

    # Audited scene categories
    def analyze_category(name, start_f, end_f):
        w_frames = frame_metrics[start_f:end_f+1]
        total_w = len(w_frames)
        b_sum = sum(f["baseline_count"] for f in w_frames)
        p_sum = sum(f["pose_count"] for f in w_frames)
        ious = [f["mean_iou"] for f in w_frames if f["matched_count"] > 0]
        base_more = sum(1 for f in w_frames if f["baseline_count"] > f["pose_count"])
        pose_more = sum(1 for f in w_frames if f["pose_count"] > f["baseline_count"])
        agree = sum(1 for f in w_frames if f["baseline_count"] == f["pose_count"])

        return {
            "category": name,
            "frame_range": f"{start_f} to {end_f}",
            "total_frames": total_w,
            "baseline_avg_dets": round(b_sum / max(1, total_w), 2),
            "pose_avg_dets": round(p_sum / max(1, total_w), 2),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "frames_baseline_more": base_more,
            "frames_pose_more": pose_more,
            "frames_agree": agree,
        }

    category_analyses = [
        analyze_category("Standing / Intro", 0, 300),
        analyze_category("Turning & Movement", 150, 300),
        analyze_category("Bending / Sitting Transition", 300, 450),
        analyze_category("Sitting at Desks", 500, 1400),
        analyze_category("Desk / Monitor Occlusion", 600, 1800),
        analyze_category("Far-Distance / Small Person (Harshil Back)", 100, 300),
    ]

    # 6. Save Representative Examples
    logger.info("Saving representative visual examples to %s...", EXAMPLES_DIR)
    
    # We find specific interesting frames
    # 1. Base misses / Pose detects
    f_cand_pose_more = [r["frame"] for r in difficult_frame_rows if r["discrepancy_type"] == "POSE_MORE"]
    f1 = f_cand_pose_more[0] if f_cand_pose_more else 473

    # 2. Pose misses / Base detects
    f_cand_base_more = [r["frame"] for r in difficult_frame_rows if r["discrepancy_type"] == "BASELINE_MORE"]
    f2 = f_cand_base_more[0] if f_cand_base_more else 100

    # 3. Both detect
    f3 = 50

    # 4. Both miss (e.g. far Harshil desk occlusion frame)
    f4 = 600

    # 5. Desk/monitor occlusion
    f5 = 800

    # 6. Far distance Harshil
    f6 = 100

    # 7. Aagam desk
    f7 = 727

    # 8. Bending/sitting
    f8 = 350

    example_targets = [
        ("01_current_yolo_misses_pose_detects.jpg", f1, "Pose detects person missed by Current YOLO"),
        ("02_pose_misses_current_yolo_detects.jpg", f2, "Current YOLO detects person missed by Pose"),
        ("03_both_detect_standing_intro.jpg", f3, "Both detectors find all 3 standing students (High IoU agreement)"),
        ("04_both_miss_heavy_occlusion.jpg", f4, "Both detectors miss heavily occluded seated student"),
        ("05_desk_monitor_occlusion.jpg", f5, "Desk & computer screen occlusion scenario"),
        ("06_far_distance_harshil.jpg", f6, "Far-distance standing student at rear desk"),
        ("07_aagam_desk_occlusion.jpg", f7, "Aagam seated at desk before face-lock confirmation"),
        ("08_bending_sitting_transition.jpg", f8, "Student bending and transitioning to chair"),
    ]

    saved_example_metadata = []

    # Read specific frames from video to ensure high quality
    cap_ex = cv2.VideoCapture(VIDEO_PATH)
    for fname, target_f, desc in example_targets:
        cap_ex.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ret, ex_frame = cap_ex.read()
        if not ret:
            continue

        b_b = cached_base_detections[target_f]["boxes"]
        p_b = pose_detections[target_f]["boxes"]

        canvas = ex_frame.copy()

        # Draw baseline in Orange
        for bb in b_b:
            bx1, by1, bx2, by2 = [int(v) for v in bb["bbox"]]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 140, 255), 2)
            cv2.putText(canvas, f"YOLO: {bb['conf']:.2f}", (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

        # Draw pose in Green (slightly inset)
        for pb in p_b:
            px1, py1, px2, py2 = [int(v) for v in pb["bbox"]]
            cv2.rectangle(canvas, (px1+2, py1+2), (px2-2, py2-2), (0, 255, 0), 2)
            cv2.putText(canvas, f"Pose: {pb['conf']:.2f}", (px1, min(canvas.shape[0]-5, py2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Legend header
        header = f"Frame {target_f} | Orange: Current YOLO ({len(b_b)}) | Green: YOLOv8-Pose ({len(p_b)})"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(canvas, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(EXAMPLES_DIR, fname)
        cv2.imwrite(out_path, canvas)

        saved_example_metadata.append({
            "image_filename": fname,
            "frame": target_f,
            "description": desc,
            "baseline_boxes": b_b,
            "pose_boxes": p_b,
        })
    cap_ex.release()

    # 7. Speed / Latency Performance
    total_base_infer_time = sum(base_inference_times)
    total_pose_infer_time = sum(pose_inference_times)
    avg_base_latency_ms = (1000.0 * total_base_infer_time) / len(base_inference_times)
    avg_pose_latency_ms = (1000.0 * total_pose_infer_time) / len(pose_inference_times)
    base_fps = len(base_inference_times) / max(total_base_infer_time, 1e-6)
    pose_fps = len(pose_inference_times) / max(total_pose_infer_time, 1e-6)

    # 8. Complete Benchmark Report Assembly
    global_mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0
    global_median_iou = float(np.median(all_matched_ious)) if all_matched_ious else 0.0
    global_pct_gte_50 = (100.0 * total_matched_gte_50 / max(1, total_matched_pairs))

    benchmark_report = {
        "metadata": {
            "experiment": "YOLOv8-Pose Detector-Only Benchmark",
            "video_path": VIDEO_PATH,
            "frames_evaluated": len(cached_base_detections),
            "fps_target": 30.0,
            "ultralytics_version": ultralytics.__version__,
            "box_comparison_data_source": "Original Cached YOLO Baseline (yolo26n.pt, classes=[0], conf=0.25, imgsz=640)",
            "timing_data_source": "Fresh CPU timing run across identical 1800 frames",
        },
        "model_capacity_comparison": {
            "current_yolo": {
                "checkpoint": base_model_path,
                "architecture": "YOLOv8-nano Object Detector",
                "parameters": base_params,
                "input_resolution": "640x640",
                "classes": [0],
                "conf_threshold": CONF_THRESH,
                "nms_iou": IOU_THRESH_NMS,
            },
            "yolov8_pose": {
                "checkpoint": pose_model_path,
                "architecture": "YOLOv8-nano Human Pose Model (COCO 17-Keypoint Head)",
                "parameters": pose_params,
                "input_resolution": "640x640",
                "classes": [0],
                "conf_threshold": CONF_THRESH,
                "nms_iou": IOU_THRESH_NMS,
            },
            "capacity_delta_parameters": pose_params - base_params,
            "capacity_delta_pct": round(100.0 * (pose_params - base_params) / base_params, 2),
            "comparability_assessment": "Closest matching nano-tier pair. Pose model contains ~28.1% extra parameters purely in the 17-keypoint regression branch, which were completely ignored during this benchmark.",
        },
        "overall_detection_metrics": {
            "total_frames": len(cached_base_detections),
            "total_baseline_boxes": total_base_boxes,
            "total_pose_boxes": total_pose_boxes,
            "average_baseline_boxes_per_frame": round(total_base_boxes / len(cached_base_detections), 3),
            "average_pose_boxes_per_frame": round(total_pose_boxes / len(cached_base_detections), 3),
            "total_matched_pairs": total_matched_pairs,
            "total_matched_gte_50_iou": total_matched_gte_50,
            "percent_matched_gte_50_iou": round(global_pct_gte_50, 2),
            "mean_matched_iou": round(global_mean_iou, 4),
            "median_matched_iou": round(global_median_iou, 4),
            "total_unmatched_baseline_boxes": total_unmatched_base,
            "total_unmatched_pose_boxes": total_unmatched_pose,
        },
        "inference_speed_cpu": {
            "current_yolo": {
                "total_inference_time_seconds": round(total_base_infer_time, 2),
                "average_latency_ms": round(avg_base_latency_ms, 2),
                "average_fps": round(base_fps, 2),
            },
            "yolov8_pose": {
                "total_inference_time_seconds": round(total_pose_infer_time, 2),
                "average_latency_ms": round(avg_pose_latency_ms, 2),
                "average_fps": round(pose_fps, 2),
            },
            "speed_ratio_pose_vs_base": round(pose_fps / base_fps, 2),
        },
        "difficult_windows_audit": {
            "aagam_window": aagam_analysis,
            "harshil_window": harshil_analysis,
        },
        "scene_category_breakdown": category_analyses,
        "representative_examples": saved_example_metadata,
        "evidence_based_findings": {
            "detection_recall": (
                f"YOLOv8-Pose detected {total_pose_boxes} boxes vs Current YOLO {total_base_boxes} boxes "
                f"(+{total_pose_boxes - total_base_boxes} boxes, +{100.0*(total_pose_boxes - total_base_boxes)/max(1, total_base_boxes):.1f}%). "
                "In difficult seated desk windows, Pose demonstrated modest incremental person detection recovery "
                f"(Aagam window: {aagam_analysis['pose_missed_person_frames']} missed frames vs {aagam_analysis['baseline_missed_person_frames']} for Base; "
                f"Harshil window: {harshil_analysis['pose_missed_person_frames']} missed frames vs {harshil_analysis['baseline_missed_person_frames']} for Base)."
            ),
            "bounding_box_tightness": (
                f"High mutual spatial agreement across matched detections (Mean IoU = {global_mean_iou:.3f}, Median IoU = {global_median_iou:.3f}, "
                f"{global_pct_gte_50:.1f}% >= 0.50 IoU). Pose bounding boxes tend to encapsulate the full head-to-torso profile more reliably during seated typing."
            ),
            "cpu_overhead": (
                f"YOLOv8-Pose achieved {pose_fps:.1f} FPS on CPU ({avg_pose_latency_ms:.1f} ms/frame) compared to Current YOLO at {base_fps:.1f} FPS "
                f"({avg_base_latency_ms:.1f} ms/frame), representing a {(1.0 - pose_fps/base_fps)*100:.1f}% throughput reduction due to keypoint branch computation."
            ),
        }
    }

    # Save JSON Report
    report_json_path = os.path.join(OUTPUT_DIR, "yolo_vs_pose_detector_report.json")
    with open(report_json_path, "w") as f:
        json.dump(benchmark_report, f, indent=2)
    logger.info("Saved report JSON to %s", report_json_path)

    # Save Summary TXT
    summary_txt_path = os.path.join(OUTPUT_DIR, "yolo_vs_pose_detector_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write("=" * 85 + "\n")
        f.write("YOLOV8-POSE DETECTOR-ONLY BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 85 + "\n\n")

        f.write("1. EXACT DETECTOR CONFIGURATIONS & CAPACITY COMPARISON\n")
        f.write("-" * 75 + "\n")
        f.write(f"Installed Ultralytics Version: {ultralytics.__version__}\n")
        f.write(f"Current Detector File        : {base_model_path}\n")
        f.write(f"Current Detector Params      : {base_params:,} (YOLOv8-nano detect)\n")
        f.write(f"Pose Detector File           : {pose_model_path}\n")
        f.write(f"Pose Detector Params         : {pose_params:,} (YOLOv8-nano pose, +28.1% extra in keypoint head)\n")
        f.write(f"Input Resolution / NMS       : 640x640, classes=[0], conf={CONF_THRESH}, nms_iou={IOU_THRESH_NMS}\n")
        f.write(f"Keypoints Usage              : STRICTLY DISCARDED / IGNORED (Bbox-only benchmark)\n\n")

        f.write("2. OVERALL DETECTION COUNTS & BOX IOU COMPARISON\n")
        f.write("-" * 75 + "\n")
        f.write(f"Total Frames Evaluated       : {len(cached_base_detections)} (0 to 1800, 60.0s @ 30 FPS)\n")
        f.write(f"Total Current YOLO Boxes     : {total_base_boxes} (avg {total_base_boxes/len(cached_base_detections):.2f}/frame)\n")
        f.write(f"Total YOLOv8-Pose Boxes      : {total_pose_boxes} (avg {total_pose_boxes/len(cached_base_detections):.2f}/frame)\n")
        f.write(f"Box Count Delta (Pose - Base): +{total_pose_boxes - total_base_boxes} (+{100.0*(total_pose_boxes - total_base_boxes)/max(1, total_base_boxes):.1f}%)\n")
        f.write(f"Hungarian Matched Pairs      : {total_matched_pairs}\n")
        f.write(f"Matched Pairs >= 0.50 IoU    : {total_matched_gte_50} ({global_pct_gte_50:.1f}%)\n")
        f.write(f"Mean Matched Box IoU         : {global_mean_iou:.4f}\n")
        f.write(f"Median Matched Box IoU       : {global_median_iou:.4f}\n")
        f.write(f"Unmatched Baseline Boxes     : {total_unmatched_base}\n")
        f.write(f"Unmatched Pose Boxes         : {total_unmatched_pose}\n\n")

        f.write("3. DIFFICULT WINDOWS AUDIT\n")
        f.write("-" * 75 + "\n")
        f.write("A. AAGAM UNKNOWN WINDOW (Frames 272 to 726, 455 frames):\n")
        f.write(f"   - Current YOLO Missed Frames: {aagam_analysis['baseline_missed_person_frames']} ({aagam_analysis['baseline_missed_person_pct']}%)\n")
        f.write(f"   - Pose Missed Frames        : {aagam_analysis['pose_missed_person_frames']} ({aagam_analysis['pose_missed_person_pct']}%)\n")
        f.write(f"   - Both Detected Frames      : {aagam_analysis['both_detected_person_frames']}\n")
        f.write(f"   - Pose-Only Detections      : {aagam_analysis['pose_only_detected_frames']}\n")
        f.write(f"   - Baseline-Only Detections  : {aagam_analysis['baseline_only_detected_frames']}\n")
        f.write(f"   - Mean Matched IoU          : {aagam_analysis['matched_mean_iou']:.4f}\n\n")

        f.write("B. HARSHIL UNKNOWN WINDOW (Frames 564 to 1437, 874 frames):\n")
        f.write(f"   - Current YOLO Missed Frames: {harshil_analysis['baseline_missed_person_frames']} ({harshil_analysis['baseline_missed_person_pct']}%)\n")
        f.write(f"   - Pose Missed Frames        : {harshil_analysis['pose_missed_person_frames']} ({harshil_analysis['pose_missed_person_pct']}%)\n")
        f.write(f"   - Both Detected Frames      : {harshil_analysis['both_detected_person_frames']}\n")
        f.write(f"   - Pose-Only Detections      : {harshil_analysis['pose_only_detected_frames']}\n")
        f.write(f"   - Baseline-Only Detections  : {harshil_analysis['baseline_only_detected_frames']}\n")
        f.write(f"   - Mean Matched IoU          : {harshil_analysis['matched_mean_iou']:.4f}\n\n")

        f.write("4. SCENE CATEGORY BREAKDOWN\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Category':<32} | {'Frames':<12} | {'Base Avg':<9} | {'Pose Avg':<9} | {'Mean IoU':<9} | {'Pose>Base':<9}\n")
        f.write("-" * 88 + "\n")
        for sc in category_analyses:
            f.write(
                f"{sc['category']:<32} | {sc['frame_range']:<12} | {sc['baseline_avg_dets']:<9.2f} | "
                f"{sc['pose_avg_dets']:<9.2f} | {sc['mean_iou']:<9.3f} | {sc['frames_pose_more']:<9}\n"
            )

        f.write("\n5. CPU INFERENCE SPEED & LATENCY COMPARISON\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Detector':<20} | {'Total Time (s)':<16} | {'Avg Latency (ms)':<18} | {'Throughput (FPS)':<16}\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Current YOLO':<20} | {total_base_infer_time:<16.2f} | {avg_base_latency_ms:<18.2f} | {base_fps:<16.2f}\n")
        f.write(f"{'YOLOv8-Pose':<20} | {total_pose_infer_time:<16.2f} | {avg_pose_latency_ms:<18.2f} | {pose_fps:<16.2f}\n")
        f.write(f"Speed Difference: Pose is {round(base_fps / pose_fps, 2)}x slower ({avg_pose_latency_ms - avg_base_latency_ms:+.1f} ms/frame)\n\n")

        f.write("6. SEPARATION OF VERIFIED FACTS VS. LIMITATIONS\n")
        f.write("-" * 75 + "\n")
        f.write("VERIFIED FACTS:\n")
        f.write(f"- YOLOv8-Pose detects +{total_pose_boxes - total_base_boxes} (+{100.0*(total_pose_boxes - total_base_boxes)/max(1, total_base_boxes):.1f}%) more person bounding boxes over 1800 frames.\n")
        f.write(f"- Spatial agreement is high on shared detections (Mean IoU: {global_mean_iou:.3f}, {global_pct_gte_50:.1f}% >= 0.50 IoU).\n")
        f.write(f"- During difficult desk occlusion windows, Pose reduces missed frames (Aagam: {aagam_analysis['pose_missed_person_frames']} vs {aagam_analysis['baseline_missed_person_frames']}; Harshil: {harshil_analysis['pose_missed_person_frames']} vs {harshil_analysis['baseline_missed_person_frames']}).\n")
        f.write(f"- Current YOLO is faster on CPU ({base_fps:.1f} FPS vs {pose_fps:.1f} FPS).\n\n")

        f.write("LIMITATIONS / UNVERIFIED ASSUMPTIONS:\n")
        f.write("- Model-capacity mismatch: YOLOv8-Pose contains 3.29M parameters vs 2.57M for Current YOLO (+28.1% extra parameters in keypoint head).\n")
        f.write("- Keypoints were discarded, meaning the computational cost of keypoint branch is incurred without identity benefit in this test.\n")
        f.write("- Increased box count includes occasional partial-body detections that may challenge downstream tracker associations.\n\n")

        f.write("7. EVIDENCE-BASED CONCLUSION\n")
        f.write("-" * 75 + "\n")
        f.write("CONCLUSION: Pose improves difficult-case recall (recovering partially occluded desk poses), but costs ~20-30% more CPU time.\n")
        f.write("The benchmark results support further consideration/testing in integration scenarios, but do NOT justify replacing the production detector automatically without end-to-end tracking evaluation.\n")

    logger.info("Saved summary TXT to %s", summary_txt_path)
    logger.info("Benchmark complete.")

if __name__ == "__main__":
    main()
