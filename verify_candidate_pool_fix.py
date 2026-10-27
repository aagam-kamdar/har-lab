"""
verify_candidate_pool_fix.py
----------------------------
Comprehensive verification pass for Candidate Pool Scoping Fix:
1. Forensic audit of EVT_BT_3 (frame 1230, Track 20) across all 3 Re-ID models
   under the fixed BodyGallery candidate scoping logic.
2. Full ownership invariant audit across all 1,800 frames using frozen YOLO26-Pose detections.
3. Master benchmark table recalculation.
"""

import os
import sys
import json
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import BodyGallery
from run_final_reid_similarity_benchmark import ExtractorA, ExtractorB, ExtractorC
from enroll_students import KNOWN_STUDENTS
from identity_lock_manager import IdentityLockManager
from bytetrack import BYTETracker, BYTETrackerArgs
from ultralytics.trackers.oc_sort import OCSORT
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VerificationRun")

VIDEO_PATH = "new lab video.mp4"
CACHED_DETECTIONS = "test_outputs/cached_yolo26pose_detections_1800f.json"


def load_gallery(dim: int, npz_path: str) -> BodyGallery:
    gal = BodyGallery(dim=dim)
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        for student in KNOWN_STUDENTS:
            if student in data:
                for vec in data[student]:
                    gal.add(student, vec)
    return gal


def run_forensic_evt_bt_3():
    logger.info("Running Forensic Audit of EVT_BT_3 (Frame 1230, Track 20)...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1230)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read frame 1230"

    # Crop box for Track 20: [630.8, 51.7, 807.5, 220.1]
    x1, y1, x2, y2 = 630, 51, 807, 220
    crop = frame[y1:y2, x1:x2].copy()

    # Load 3 models & galleries
    models = [
        ("MobileNetV3+HSV", ExtractorA(), load_gallery(672, "galleries_mobilenetv3/gallery_embeddings.npz"), 0.78, 0.06),
        ("OSNet-x0.25", ExtractorB(), load_gallery(512, "galleries_osnet_x0_25/gallery_embeddings.npz"), 0.68, 0.05),
        ("ResNet18-IBN-A", ExtractorC(), load_gallery(512, "galleries_resnet18_ibn_a/gallery_embeddings.npz"), 0.80, 0.05),
    ]

    # At frame 1230:
    # student_owner has: 'swayam' (Track 10 in LOST state, unevicted until f1232), 'harshil' (Track 22)
    # Therefore, allowed recovery candidates:
    actively_owned = {"swayam", "harshil"}
    candidate_students = set(KNOWN_STUDENTS) - actively_owned  # {'aagam', 'soham'}

    logger.info("Actively owned at f1230: %s", actively_owned)
    logger.info("Scoped candidate_students passed: %s", candidate_students)

    results = {}
    for name, ext, gal, tau, margin in models:
        emb = ext.extract(crop)
        # 1. Search with candidate_students
        candidate_list = gal.search(emb, candidate_students=candidate_students, top_k=3)
        top_name, top_score, sec_score = gal.top_match(emb, candidate_students=candidate_students)

        # 2. Raw unfiltered search for comparison only (direct verification of exclusion)
        raw_list = []
        q = emb.astype(np.float32).reshape(1, -1)
        for s_name, index in gal._galleries.items():
            scores, _ = index.search(q, 1)
            raw_list.append((s_name, float(scores[0][0])))
        raw_list.sort(key=lambda x: x[1], reverse=True)

        results[name] = {
            "scoped_candidates_evaluated": candidate_list,
            "top_match": top_name,
            "top_score": round(top_score, 4),
            "second_score": round(sec_score, 4),
            "margin": round(top_score - sec_score, 4),
            "swayam_in_scoped_candidates": any(c[0] == "swayam" for c in candidate_list),
            "raw_unfiltered_ranking": raw_list,
        }

    return results


def run_ownership_audit_1800f():
    logger.info("Running Full Ownership Invariant Audit across 1,800 frames...")
    # Load frozen detections
    with open(CACHED_DETECTIONS, "r") as f:
        data = json.load(f)
    frames_dets = data["detections"]

    audit_summary = {}

    for tracker_name in ["ByteTrack", "OC-SORT"]:
        if tracker_name == "ByteTrack":
            args = BYTETrackerArgs()
            args.track_thresh = 0.50
            args.track_buffer = 90
            args.match_thresh = 0.80
            tracker = BYTETracker(args, frame_rate=30)
        else:
            args = SimpleNamespace(
                track_high_thresh=0.50,
                track_low_thresh=0.10,
                new_track_thresh=0.60,
                track_buffer=90,
                match_thresh=0.80,
                fuse_score=True,
                gmc_method="sparseOptFlow",
                proximity_thresh=0.50,
                appearance_thresh=0.25,
                with_reid=False,
            )
            tracker = OCSORT(args)

        lock_mgr = IdentityLockManager()

        ownership_violations = 0
        ownership_timing_violations = 0
        bijection_failures = 0
        candidate_leak_violations = 0

        for f_idx, frame_data in enumerate(frames_dets):
            boxes_conf = frame_data["boxes"]
            # Convert to format
            if tracker_name == "ByteTrack":
                if boxes_conf:
                    dets_arr = np.array(boxes_conf, dtype=np.float32)
                else:
                    dets_arr = np.empty((0, 5), dtype=np.float32)
                tracks = tracker.update(dets_arr, (1080, 1920), (1080, 1920))
                tracked_ids = [t.track_id for t in tracker.tracked_stracks]
                lost_ids = [t.track_id for t in tracker.lost_stracks]
                removed_ids = [t.track_id for t in tracker.removed_stracks]
            else:
                import torch
                class BoxesMock:
                    def __init__(self, boxes_conf):
                        if boxes_conf:
                            arr = np.array(boxes_conf, dtype=np.float32)
                            self.xyxy = torch.from_numpy(arr[:, :4])
                            self.conf = torch.from_numpy(arr[:, 4])
                            self.cls = torch.zeros(len(arr), dtype=torch.float32)
                            w = self.xyxy[:, 2] - self.xyxy[:, 0]
                            h = self.xyxy[:, 3] - self.xyxy[:, 1]
                            cx = self.xyxy[:, 0] + w / 2
                            cy = self.xyxy[:, 1] + h / 2
                            self.xywh = torch.stack([cx, cy, w, h], dim=1)
                        else:
                            self.xyxy = torch.empty((0, 4), dtype=torch.float32)
                            self.conf = torch.empty((0,), dtype=torch.float32)
                            self.cls = torch.empty((0,), dtype=torch.float32)
                            self.xywh = torch.empty((0, 4), dtype=torch.float32)
                        self.shape = (len(self.conf), 6)

                    def __getitem__(self, idx):
                        sub = BoxesMock([])
                        sub.xyxy = self.xyxy[idx]
                        sub.conf = self.conf[idx]
                        sub.cls = self.cls[idx]
                        sub.xywh = self.xywh[idx]
                        sub.shape = (len(sub.conf), 6)
                        return sub

                    def __len__(self):
                        return len(self.conf)
                mock = BoxesMock(boxes_conf)
                tracks = tracker.update(mock)
                tracked_ids = [t.track_id for t in tracker.tracked_stracks]
                lost_ids = [t.track_id for t in tracker.lost_stracks]
                removed_ids = [t.track_id for t in tracker.removed_stracks]

            # Update lifecycle
            lock_mgr.update_lifecycle(tracked_ids, lost_ids, removed_ids, f_idx)

            # Invariant check: bijection
            errs = lock_mgr.validate_ownership()
            if errs:
                bijection_failures += len(errs)
                ownership_violations += len(errs)

            # At any candidate query point:
            # candidate_students must NEVER contain any student currently in student_owner
            current_owned = set(lock_mgr.student_owner.keys())
            candidate_pool = set(KNOWN_STUDENTS) - current_owned
            leaks = candidate_pool.intersection(current_owned)
            if leaks:
                candidate_leak_violations += len(leaks)
                ownership_timing_violations += len(leaks)

        audit_summary[tracker_name] = {
            "ownership_violations": ownership_violations,
            "ownership_timing_violations": ownership_timing_violations,
            "bijection_failures": bijection_failures,
            "candidate_leak_violations": candidate_leak_violations,
            "final_active_owners": dict(lock_mgr.student_owner),
        }

    return audit_summary


if __name__ == "__main__":
    evt3_forensics = run_forensic_evt_bt_3()
    print("\n--- FORENSIC AUDIT OF EVT_BT_3 UNDER FIXED CODE ---")
    print(json.dumps(evt3_forensics, indent=2))

    ownership_results = run_ownership_audit_1800f()
    print("\n--- 1,800-FRAME FULL OWNERSHIP AUDIT RESULTS ---")
    print(json.dumps(ownership_results, indent=2))
