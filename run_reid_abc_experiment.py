"""
run_reid_abc_experiment.py
--------------------------
Controlled 1,800-frame A/B/C Body Re-ID Experiment on 'new lab video.mp4'.

Compares:
  A) BASELINE: MobileNetV3 + HSV (672-dim, thresh=0.78, margin=0.06)
  B) OSNet-x0.25 (512-dim, thresh=0.68, margin=0.05)
  C) ResNet-18-IBN-A (512-dim, thresh=0.80, margin=0.05)

Guarantees:
  - Same cached ByteTrack tracklets and identical person crops for A, B, C.
  - Same 6 reference images per student with identical augmentation pipeline.
  - Independent galleries and FAISS indexes for each model.
  - ArcFace is the sole identity-lock authority. Body Re-ID is recovery-only.
  - Full threshold evaluation: positive pairs below thresh, negative pairs above thresh, FPR, FNR.
"""

from __future__ import annotations

import os
import sys
import glob
import time
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack as BTBaseTrack
from features import FaceFeatureExtractor, BodyGallery
from identity_lock_manager import IdentityLockManager, FACE_FRAME_GAP_FOR_RECOVERY
from enroll_students import augment_image, KNOWN_STUDENTS, FACE_FOLDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReID_ABC")


# --------------------------------------------------------------------------- #
#  Model Extractors                                                           #
# --------------------------------------------------------------------------- #

class ExtractorA:
    """Variant A: MobileNetV3-Small (576-D) + HSV Spatial Histogram (96-D) = 672-D."""
    def __init__(self):
        from reid.embedding_model import PersonReIDExtractor
        self.extractor = PersonReIDExtractor()
        self.dim = 672

    def extract(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return None
        try:
            emb = self.extractor.extract(crop)
            if emb is None:
                return None
            norm = np.linalg.norm(emb)
            return (emb / max(norm, 1e-12)).astype(np.float32)
        except Exception:
            return None


class ExtractorB:
    """Variant B: OSNet-x0.25 (512-D)."""
    def __init__(self, weights_path: str = "reid/weights/osnet_x0_25_msmt17.pth"):
        import reid.osnet as osnet_mod
        self.model = osnet_mod.osnet_x0_25(num_classes=1000, pretrained=False)
        ckpt = torch.load(weights_path, map_location="cpu")
        matched = {k: v for k, v in ckpt.items() if k in self.model.state_dict() and v.shape == self.model.state_dict()[k].shape}
        self.model.load_state_dict(matched, strict=False)
        self.model.eval()
        self.dim = 512
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def extract(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return None
        try:
            res = cv2.resize(crop, (128, 256))
            rgb = res[:, :, ::-1].astype(np.float32) / 255.0
            norm = (rgb - self.mean) / self.std
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
            with torch.no_grad():
                out = self.model(tensor).squeeze().numpy()
            l2 = np.linalg.norm(out)
            return (out / max(l2, 1e-12)).astype(np.float32)
        except Exception:
            return None


class ExtractorC:
    """Variant C: ResNet-18-IBN-A (512-D)."""
    def __init__(self, weights_path: str = "reid/weights/resnet18_ibn_a-2f571257.pth"):
        import reid.resnet_ibn as ibn_mod
        self.model = ibn_mod.resnet18_ibn_a(pretrained=False)
        ckpt = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=True)
        self.model.eval()
        self.dim = 512
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def extract(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return None
        try:
            res = cv2.resize(crop, (128, 256))
            rgb = res[:, :, ::-1].astype(np.float32) / 255.0
            norm = (rgb - self.mean) / self.std
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
            with torch.no_grad():
                fmap = self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(tensor))))))))
                pooled = self.adaptive_pool(fmap)
                out = torch.flatten(pooled, 1).squeeze().numpy()
            l2 = np.linalg.norm(out)
            return (out / max(l2, 1e-12)).astype(np.float32)
        except Exception:
            return None


# --------------------------------------------------------------------------- #
#  Enrollment & Gallery Construction                                          #
# --------------------------------------------------------------------------- #

def enroll_model(extractor, name_prefix: str) -> Tuple[BodyGallery, Dict[str, List[np.ndarray]]]:
    """Enroll all 4 students into an independent BodyGallery using the same photos and augmentations."""
    gallery = BodyGallery(dim=extractor.dim)
    student_embeddings = defaultdict(list)

    for student in KNOWN_STUDENTS:
        folder = Path(f"face/{student}")
        image_files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # 1. Base embedding
            emb = extractor.extract(img)
            if emb is not None:
                gallery.add(student, emb)
                student_embeddings[student].append(emb)

            # 2. Augmented variants
            for aug_img in augment_image(img):
                aug_emb = extractor.extract(aug_img)
                if aug_emb is not None:
                    gallery.add(student, aug_emb)
                    student_embeddings[student].append(aug_emb)

    logger.info("Enrolled %s: Gallery sizes = %s", name_prefix, gallery.sizes())
    return gallery, student_embeddings


# --------------------------------------------------------------------------- #
#  Threshold Evaluation                                                       #
# --------------------------------------------------------------------------- #

def evaluate_threshold(student_embs: Dict[str, List[np.ndarray]], threshold: float) -> Dict[str, Any]:
    pos_sims = []
    neg_sims = []

    students = list(student_embs.keys())
    for i, s1 in enumerate(students):
        embs1 = student_embs[s1]
        # Positive pairs (same student across different images)
        for idx1 in range(len(embs1)):
            for idx2 in range(idx1 + 1, len(embs1)):
                sim = float(np.dot(embs1[idx1], embs1[idx2]))
                pos_sims.append(sim)

        # Negative pairs (different students)
        for s2 in students[i+1:]:
            embs2 = student_embs[s2]
            for e1 in embs1:
                for e2 in embs2:
                    sim = float(np.dot(e1, e2))
                    neg_sims.append(sim)

    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)

    fn = int(np.sum(pos_sims < threshold))
    fp = int(np.sum(neg_sims >= threshold))
    total_pos = len(pos_sims)
    total_neg = len(neg_sims)

    fnr = round(100.0 * fn / max(total_pos, 1), 2)
    fpr = round(100.0 * fp / max(total_neg, 1), 2)

    return {
        "threshold": threshold,
        "total_positive_pairs": total_pos,
        "positive_pairs_below_threshold_fn": fn,
        "false_negative_rate_pct": fnr,
        "total_negative_pairs": total_neg,
        "negative_pairs_above_threshold_fp": fp,
        "empirical_false_positive_rate_pct": fpr,
        "intra_mean": round(float(pos_sims.mean()), 3),
        "intra_min": round(float(pos_sims.min()), 3),
        "inter_mean": round(float(neg_sims.mean()), 3),
        "inter_max": round(float(neg_sims.max()), 3),
    }


# --------------------------------------------------------------------------- #
#  Main Experiment Execution                                                  #
# --------------------------------------------------------------------------- #

def run_experiment(video_path: str = "new lab video.mp4", max_frames: int = 1800, output_dir: str = "test_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("==================================================================")
    logger.info("STARTING BODY RE-ID A/B/C EXPERIMENT (%d frames)", max_frames)
    logger.info("==================================================================")

    # 1. Instantiate extractors
    ext_a = ExtractorA()
    ext_b = ExtractorB()
    ext_c = ExtractorC()

    # 2. Build independent galleries
    logger.info("Building independent galleries from reference photos...")
    gal_a, embs_a = enroll_model(ext_a, "Model A (MobileNetV3+HSV)")
    gal_b, embs_b = enroll_model(ext_b, "Model B (OSNet-x0.25)")
    gal_c, embs_c = enroll_model(ext_c, "Model C (ResNet-18-IBN-A)")

    # 3. Threshold evaluation
    thresh_a = evaluate_threshold(embs_a, threshold=0.78)
    thresh_b = evaluate_threshold(embs_b, threshold=0.68)
    thresh_c = evaluate_threshold(embs_c, threshold=0.80)

    # 4. ArcFace Face Profiles
    face_extractor = FaceFeatureExtractor(det_size=(320, 320))
    face_profiles: Dict[str, np.ndarray] = {}
    for student in KNOWN_STUDENTS:
        folder = Path(f"face/{student}")
        front_imgs = [cv2.imread(str(p)) for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and "back" not in p.name.lower()]
        embs = []
        for im in front_imgs:
            if im is not None:
                e = face_extractor.extract_from_image(im)
                if e is not None:
                    embs.append(e)
        if embs:
            avg_e = np.mean(embs, axis=0)
            face_profiles[student] = (avg_e / np.linalg.norm(avg_e)).astype(np.float32)

    logger.info("Enrolled ArcFace face profiles: %s", list(face_profiles.keys()))

    # 5. Step 1: Pre-cache ByteTrack tracklets, person crops, and face predictions
    logger.info("Caching ByteTrack tracklets and person crops from %s (first %d frames)...", video_path, max_frames)
    yolo = YOLO("yolo26n.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    BTBaseTrack.reset_id()
    bt_args = BYTETrackerArgs()
    bt_args.track_thresh = 0.35
    bt_args.track_buffer = 90
    bt_args.match_thresh = 0.85
    bt_args.mot20 = False
    bytetrack = BYTETracker(bt_args, frame_rate=30)

    cached_frames = [] # list of {frame_idx, online_tracks: [(tid, bbox, crop, face_pred)], tracked_ids, lost_ids, removed_ids}

    t0 = time.time()
    for f_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        res = yolo(frame, classes=[0], verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                dets.append([x1, y1, x2, y2, conf])

        det_arr = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)
        online = bytetrack.update(det_arr, [vid_h, vid_w], [vid_h, vid_w])

        tracked_ids = [int(t.track_id) for t in bytetrack.tracked_stracks if t.is_activated]
        lost_ids = [int(t.track_id) for t in bytetrack.lost_stracks]
        removed_ids = [int(t.track_id) for t in bytetrack.removed_stracks]

        online_tracks = []
        for t in online:
            tid = int(t.track_id)
            tlwh = t.tlwh
            x1, y1 = max(0, int(tlwh[0])), max(0, int(tlwh[1]))
            x2, y2 = min(vid_w, int(tlwh[0] + tlwh[2])), min(vid_h, int(tlwh[1] + tlwh[3]))
            crop = frame[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None

            # Face extraction
            face_pred = None
            face_emb = face_extractor.extract_from_crop(frame, (x1, y1, x2, y2), padding=0.25)
            if face_emb is not None:
                f_name, f_score, f_sec = face_extractor.top_match(face_emb, face_profiles)
                face_pred = (f_name, f_score, f_sec)

            online_tracks.append({
                "track_id": tid,
                "bbox": [x1, y1, x2, y2],
                "crop": crop,
                "face_pred": face_pred,
            })

        cached_frames.append({
            "frame_idx": f_idx,
            "online_tracks": online_tracks,
            "tracked_ids": tracked_ids,
            "lost_ids": lost_ids,
            "removed_ids": removed_ids,
        })

        if (f_idx + 1) % 300 == 0:
            logger.info("Cached %d/%d frames (%.1f FPS)", f_idx + 1, max_frames, (f_idx + 1) / (time.time() - t0))

    cap.release()
    logger.info("Completed tracklet and crop caching in %.2fs", time.time() - t0)

    # 6. Replay through Pipeline A, B, and C
    def simulate_variant(variant_name, extractor, gallery, threshold, margin):
        logger.info("Running simulation for %s (threshold=%.2f, margin=%.2f)...", variant_name, threshold, margin)
        lock_mgr = IdentityLockManager()
        track_last_active = {}
        last_body_emb = {}
        
        stats = {
            "variant": variant_name,
            "threshold": threshold,
            "margin": margin,
            "face_locks": 0,
            "gap_recovery_attempts": 0,
            "gap_recovery_successes": 0,
            "gap_recovery_events": [],
            "self_learned_embeddings": 0,
            "student_coverage_frames": defaultdict(int),
            "extraction_time_total_seconds": 0.0,
            "extractions_count": 0,
        }

        t_sim0 = time.time()

        for item in cached_frames:
            f_idx = item["frame_idx"]
            lock_mgr.update_lifecycle(item["tracked_ids"], item["lost_ids"], item["removed_ids"], f_idx)

            for tr in item["online_tracks"]:
                tid = tr["track_id"]
                crop = tr["crop"]
                face_pred = tr["face_pred"]
                
                state = lock_mgr.update_seen(tid, f_idx)
                was_locked = state.locked

                # 1. Gap Recovery check (when track was lost for >= FACE_FRAME_GAP_FOR_RECOVERY frames)
                frames_gap = f_idx - track_last_active.get(tid, f_idx)
                if frames_gap >= FACE_FRAME_GAP_FOR_RECOVERY and tid in last_body_emb:
                    stats["gap_recovery_attempts"] += 1
                    b_emb = last_body_emb[tid]
                    candidate_students = set(KNOWN_STUDENTS) - set(lock_mgr.student_owner.keys())
                    b_name, b_score, b_sec = gallery.top_match(b_emb, candidate_students=candidate_students)
                    
                    # Gap recovery matching check with model-specific threshold and margin
                    if state.locked and b_score >= threshold and (b_score - b_sec) >= margin:
                        if b_name == state.locked_name:
                            recovered = lock_mgr.attempt_gap_recovery(tid, f_idx, b_name, b_score, b_sec)
                            if recovered:
                                stats["gap_recovery_successes"] += 1
                                stats["gap_recovery_events"].append({
                                    "frame": f_idx,
                                    "track_id": tid,
                                    "recovered_student": recovered,
                                    "score": round(b_score, 3),
                                    "margin": round(b_score - b_sec, 3),
                                    "correct": True,
                                })
                        else:
                            # Recovery mismatch
                            stats["gap_recovery_events"].append({
                                "frame": f_idx,
                                "track_id": tid,
                                "locked": state.locked_name,
                                "body_candidate": b_name,
                                "score": round(b_score, 3),
                                "margin": round(b_score - b_sec, 3),
                                "correct": False,
                            })

                # 2. Face lock submission (sole authority for locking)
                if face_pred is not None:
                    f_name, f_score, f_sec = face_pred
                    lock_mgr.submit_face_match(tid, f_idx, f_name, f_score, f_sec)
                    if not was_locked and state.locked:
                        stats["face_locks"] += 1

                # 3. Body feature extraction
                if crop is not None:
                    t_ext0 = time.perf_counter()
                    body_emb = extractor.extract(crop)
                    t_ext_elapsed = time.perf_counter() - t_ext0
                    stats["extraction_time_total_seconds"] += t_ext_elapsed
                    stats["extractions_count"] += 1

                    if body_emb is not None:
                        last_body_emb[tid] = body_emb
                        if state.locked:
                            cand = {state.locked_name}
                        else:
                            cand = set(KNOWN_STUDENTS) - set(lock_mgr.student_owner.keys())
                        b_name, b_score, _ = gallery.top_match(body_emb, candidate_students=cand)
                        if b_name is not None:
                            lock_mgr.submit_body_evidence(tid, f_idx, b_name, b_score)

                        # Self-learning: add to gallery only if identity is locked and not near-duplicate
                        if state.locked and b_score < 0.94:
                            added = gallery.add(state.locked_name, body_emb)
                            if added:
                                stats["self_learned_embeddings"] += 1

                track_last_active[tid] = f_idx
                stats["student_coverage_frames"][state.identity_label()] += 1

        elapsed = time.time() - t_sim0
        avg_ext_fps = stats["extractions_count"] / max(stats["extraction_time_total_seconds"], 1e-6)

        # Coverage percentages
        coverage_pct = {}
        for s, frames in stats["student_coverage_frames"].items():
            coverage_pct[s] = {
                "frames": frames,
                "coverage_pct": round(100.0 * frames / max_frames, 1),
            }

        return {
            "variant": variant_name,
            "threshold": threshold,
            "margin": margin,
            "simulation_time_seconds": round(elapsed, 2),
            "face_locks": stats["face_locks"],
            "gap_recovery_attempts": stats["gap_recovery_attempts"],
            "gap_recovery_successes": stats["gap_recovery_successes"],
            "gap_recovery_events": stats["gap_recovery_events"],
            "self_learned_embeddings": stats["self_learned_embeddings"],
            "final_gallery_sizes": gallery.sizes(),
            "extraction_throughput_fps": round(avg_ext_fps, 1),
            "student_coverage": coverage_pct,
            "final_ownership": lock_mgr.ownership_snapshot(),
        }

    res_a = simulate_variant("Variant A (MobileNetV3 + HSV)", ext_a, gal_a, threshold=0.78, margin=0.06)
    res_b = simulate_variant("Variant B (OSNet-x0.25)", ext_b, gal_b, threshold=0.68, margin=0.05)
    res_c = simulate_variant("Variant C (ResNet-18-IBN-A)", ext_c, gal_c, threshold=0.80, margin=0.05)

    report = {
        "benchmark_window": f"1800 frames ({round(max_frames/fps, 1)} seconds @ 30.0 fps)",
        "threshold_evaluation": {
            "Variant_A_MobileNetV3": thresh_a,
            "Variant_B_OSNet_x0_25": thresh_b,
            "Variant_C_ResNet18_IBN_A": thresh_c,
        },
        "tracking_and_recovery_evaluation": {
            "Variant_A": res_a,
            "Variant_B": res_b,
            "Variant_C": res_c,
        },
    }

    out_path = os.path.join(output_dir, "reid_abc_experiment_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved complete A/B/C experiment report to %s", out_path)
    return report


if __name__ == "__main__":
    run_experiment(video_path="new lab video.mp4", max_frames=1800, output_dir="test_outputs")
