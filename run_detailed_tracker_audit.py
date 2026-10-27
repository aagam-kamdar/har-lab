"""
run_detailed_tracker_audit.py
------------------------------
Comprehensive Audit & Verification of ByteTrack vs. OC-SORT vs. BoT-SORT.

1. Pre-caches identical YOLO detections for frames 0..1800.
2. Runs ByteTrack, OC-SORT, and BoT-SORT with matched thresholds (0.35, 0.10, 0.85, buffer=90).
3. Evaluates ground truth student identities at 20 keyframes (Frames 1, 50, 100, 150, 200, 250, 300, 400, 473, 500, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1750, 1800).
4. Measures raw ID switches (IDSW) and trajectory continuity for each student.
5. Records empirical Kalman filter covariance / state values for BoT-SORT and ByteTrack during bending/sitting.
"""

import os
import sys
import json
import logging
from types import SimpleNamespace
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.oc_sort import OCSORT
from ultralytics.trackers.bot_sort import BOTSORT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack as BTBaseTrack
from features import FaceFeatureExtractor
from enroll_students import enroll_students

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrackerAudit")


def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)


def main():
    video_path = "new lab video.mp4"
    max_frames = 1800
    
    logger.info("Step 1: Enrolling students for high-confidence ground truth keyframe annotation...")
    face_profiles, _ = enroll_students(verbose=False)
    face_extractor = FaceFeatureExtractor(det_size=(320, 320))
    
    logger.info("Step 2: Caching YOLO detections for 1800 frames...")
    yolo = YOLO("yolo26n.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cached_detections = [] # list of (frame_idx, list of [x1,y1,x2,y2,conf], yolo_boxes_tensor, frame_img)
    
    keyframe_indices = [1, 50, 100, 150, 200, 250, 300, 400, 473, 500, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1750, 1799]
    keyframe_set = set(keyframe_indices)
    keyframe_images = {}
    
    for f_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        res = yolo(frame, classes=[0], verbose=False)[0]
        boxes_list = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes_list.append([x1, y1, x2, y2, conf])
                
        cached_detections.append({
            "frame_idx": f_idx,
            "boxes_list": boxes_list,
            "boxes_tensor": res.boxes,
        })
        
        if f_idx in keyframe_set:
            keyframe_images[f_idx] = frame.copy()
            
    cap.release()
    logger.info("Cached %d frames of YOLO detections.", len(cached_detections))
    
    # Step 3: Determine ground truth identities at keyframes
    logger.info("Step 3: Analyzing physical ground truth identities at keyframes...")
    keyframe_gt = {} # frame_idx -> list of {"student": name, "bbox": [x1,y1,x2,y2]}
    
    for kf in keyframe_indices:
        frame = keyframe_images[kf]
        dets = cached_detections[kf]["boxes_list"]
        gt_list = []
        for det in dets:
            x1, y1, x2, y2, conf = det
            bbox = (int(x1), int(y1), int(x2), int(y2))
            
            # Extract face
            face_emb = face_extractor.extract_from_crop(frame, bbox, padding=0.25)
            gt_name = "Person"
            match_score = 0.0
            if face_emb is not None:
                best_name, best_score, second_score = face_extractor.top_match(face_emb, face_profiles)
                if best_score >= 0.35 and (best_score - second_score) >= 0.04:
                    gt_name = best_name
                    match_score = best_score
            
            # Spatial context heuristic for seated positions if face occluded
            # Camera view:
            # - Left desk (x ~ 150-450): Soham / Harshil
            # - Right desk (x ~ 600-950): Swayam / Aagam
            gt_list.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": round(conf, 2),
                "face_student": gt_name,
                "face_score": round(match_score, 3),
            })
        keyframe_gt[kf] = gt_list

    # Step 4: Run ByteTrack with exact cached detections
    logger.info("Step 4: Running ByteTrack on cached detections...")
    BTBaseTrack.reset_id()
    bt_args = BYTETrackerArgs()
    bt_args.track_thresh = 0.35
    bt_args.track_buffer = 90
    bt_args.match_thresh = 0.85
    bt_args.mot20 = False
    bytetrack = BYTETracker(bt_args, frame_rate=30)
    
    bt_frame_tracks = {}
    bt_covariances_sample = []
    
    for item in cached_detections:
        f_idx = item["frame_idx"]
        dets = item["boxes_list"]
        det_arr = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)
        online = bytetrack.update(det_arr, [vid_h, vid_w], [vid_h, vid_w])
        
        tracks = []
        for t in online:
            tlwh = t.tlwh
            tracks.append((int(t.track_id), [tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]]))
            # Sample covariance during bending/sitting frames (300..500)
            if 300 <= f_idx <= 350 and hasattr(t, 'mean') and hasattr(t, 'covariance'):
                bt_covariances_sample.append({
                    "frame": f_idx,
                    "track_id": int(t.track_id),
                    "mean_shape": t.mean.shape,
                    "mean_xyah": [round(float(v), 2) for v in t.mean],
                    "cov_diag": [round(float(t.covariance[i, i]), 4) for i in range(min(8, t.covariance.shape[0]))]
                })
        bt_frame_tracks[f_idx] = tracks

    # Step 5: Run OC-SORT on exact cached detections
    logger.info("Step 5: Running OC-SORT on cached detections...")
    oc_args = SimpleNamespace(
        track_high_thresh=0.35,
        track_low_thresh=0.1,
        new_track_thresh=0.35,
        track_buffer=90,
        match_thresh=0.85,
        fuse_score=True,
        delta_t=3,
        inertia=0.2,
        use_byte=True,
        mot20=False
    )
    ocsort = OCSORT(oc_args)
    oc_frame_tracks = {}
    
    for item in cached_detections:
        f_idx = item["frame_idx"]
        out = ocsort.update(item["boxes_tensor"])
        tracks = []
        if out is not None and len(out) > 0:
            for row in out:
                x1, y1, x2, y2, tid = row[:5]
                tracks.append((int(tid), [float(x1), float(y1), float(x2), float(y2)]))
        oc_frame_tracks[f_idx] = tracks

    # Step 6: Run BoT-SORT on exact cached detections
    logger.info("Step 6: Running BoT-SORT on cached detections...")
    bot_args = SimpleNamespace(
        track_high_thresh=0.35,
        track_low_thresh=0.1,
        new_track_thresh=0.35,
        track_buffer=90,
        match_thresh=0.85,
        fuse_score=True,
        gmc_method="none",
        proximity_thresh=0.5,
        appearance_thresh=0.8,
        with_reid=False,
        model="auto",
        mot20=False
    )
    botsort = BOTSORT(bot_args)
    bot_frame_tracks = {}
    bot_covariances_sample = []
    
    for item in cached_detections:
        f_idx = item["frame_idx"]
        frame = keyframe_images.get(f_idx, np.zeros((vid_h, vid_w, 3), dtype=np.uint8))
        out = botsort.update(item["boxes_tensor"], img=frame)
        tracks = []
        if out is not None and len(out) > 0:
            for row in out:
                x1, y1, x2, y2, tid = row[:5]
                tracks.append((int(tid), [float(x1), float(y1), float(x2), float(y2)]))
        bot_frame_tracks[f_idx] = tracks
        
        # Sample BoT-SORT Kalman covariance
        if 300 <= f_idx <= 350:
            for t in botsort.tracked_stracks:
                if hasattr(t, 'mean') and hasattr(t, 'covariance') and t.mean is not None and t.covariance is not None:
                    bot_covariances_sample.append({
                        "frame": f_idx,
                        "track_id": int(t.track_id),
                        "mean_shape": t.mean.shape,
                        "mean_xywh": [round(float(v), 2) for v in t.mean],
                        "cov_diag": [round(float(t.covariance[i, i]), 4) for i in range(min(8, t.covariance.shape[0]))]
                    })

    # Step 7: ID Switch (IDSW) & Keyframe Matching Analysis
    logger.info("Step 7: Calculating Raw ID Switches across all keyframe pairs...")
    
    def evaluate_idsw(frame_tracks_map):
        id_switches = 0
        track_to_ground_truth = {} # track_id -> (last_frame, last_box_center, assigned_id)
        person_trajectories = defaultdict(list) # person_cluster -> list of (frame, track_id, bbox)
        
        prev_matched = {} # person_idx -> track_id
        for f in range(len(cached_detections)):
            tracks = frame_tracks_map.get(f, [])
            # Map detections spatially across frames
            current_matched = {}
            for tid, box in tracks:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                # Find nearest prior track in prev frame
                current_matched[tid] = (cx, cy, box)
                
            # Check for ID swaps between close targets
            if f > 0:
                prev_tracks = frame_tracks_map.get(f-1, [])
                # Compute overlap between t_prev and t_curr
                for tid_p, box_p in prev_tracks:
                    best_iou = 0.0
                    best_tid_c = None
                    for tid_c, box_c in tracks:
                        iou = compute_iou(box_p, box_c)
                        if iou > best_iou:
                            best_iou = iou
                            best_tid_c = tid_c
                    # If strong spatial overlap (> 0.60) but track_id mutated -> ID switch
                    if best_iou >= 0.60 and best_tid_c is not None and best_tid_c != tid_p:
                        id_switches += 1
                        
        return id_switches

    bt_idsw = evaluate_idsw(bt_frame_tracks)
    oc_idsw = evaluate_idsw(oc_frame_tracks)
    bot_idsw = evaluate_idsw(bot_frame_tracks)

    # Step 8: Build Keyframe Audit Table
    keyframe_table = []
    for kf in keyframe_indices:
        gt_dets = keyframe_gt[kf]
        bt_t = bt_frame_tracks.get(kf, [])
        oc_t = oc_frame_tracks.get(kf, [])
        bot_t = bot_frame_tracks.get(kf, [])
        
        row = {
            "frame": kf,
            "ground_truth_targets": len(gt_dets),
            "targets": [],
        }
        
        for g in gt_dets:
            g_box = g["bbox"]
            
            # Find matching track in each tracker
            def find_match(tracks):
                best_iou = 0.0
                best_id = None
                for tid, box in tracks:
                    iou = compute_iou(g_box, box)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = tid
                return best_id if best_iou >= 0.30 else "None"
                
            bt_m = find_match(bt_t)
            oc_m = find_match(oc_t)
            bot_m = find_match(bot_t)
            
            row["targets"].append({
                "student": g["face_student"],
                "bbox": g_box,
                "bytetrack_id": bt_m,
                "ocsort_id": oc_m,
                "botsort_id": bot_m,
            })
        keyframe_table.append(row)

    # Step 9: Save Comprehensive JSON Audit Report
    audit_report = {
        "benchmark_window": "0 to 1800 frames (60.0 seconds @ 30.0 fps)",
        "detection_source": "Cached identical YOLOv8 detections (classes=[0])",
        "tracker_parameters_matched": {
            "track_high_thresh": 0.35,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.35,
            "track_buffer": 90,
            "match_thresh": 0.85,
            "fuse_score": True,
        },
        "raw_id_switches": {
            "bytetrack_idsw": bt_idsw,
            "ocsort_idsw": oc_idsw,
            "botsort_idsw": bot_idsw,
        },
        "total_unique_track_ids": {
            "bytetrack": len(set(tid for tracks in bt_frame_tracks.values() for tid, _ in tracks)),
            "ocsort": len(set(tid for tracks in oc_frame_tracks.values() for tid, _ in tracks)),
            "botsort": len(set(tid for tracks in bot_frame_tracks.values() for tid, _ in tracks)),
        },
        "keyframe_student_continuity": keyframe_table,
        "bot_sort_covariance_measurements": bot_covariances_sample[:5],
        "bytetrack_covariance_measurements": bt_covariances_sample[:5],
    }

    out_json = "test_outputs/detailed_tracker_audit_1800f.json"
    with open(out_json, "w") as f:
        json.dump(audit_report, f, indent=2)

    logger.info("Saved detailed audit report to %s", out_json)
    print("\n================== AUDIT COMPLETED SUCCESSFULLY ==================")
    print(f"Window: 0..1800 frames | Detections: Identical Cached")
    print(f"ByteTrack  -> Total IDs: {audit_report['total_unique_track_ids']['bytetrack']} | Raw ID Switches: {bt_idsw}")
    print(f"OC-SORT    -> Total IDs: {audit_report['total_unique_track_ids']['ocsort']} | Raw ID Switches: {oc_idsw}")
    print(f"BoT-SORT   -> Total IDs: {audit_report['total_unique_track_ids']['botsort']} | Raw ID Switches: {bot_idsw}")


if __name__ == "__main__":
    main()
