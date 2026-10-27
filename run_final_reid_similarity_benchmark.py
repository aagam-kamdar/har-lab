"""
run_final_reid_similarity_benchmark.py
--------------------------------------
Executes Sections 4, 5, 6, 7 of the Final Integration Benchmark:
- Independent gallery generation for:
    A: galleries_mobilenetv3/
    B: galleries_osnet_x0_25/
    C: galleries_resnet18_ibn_a/
- Student-vs-Student Similarity/Dissimilarity Matrix on reference images.
- Threshold sweeps for A, B, and C.
- Real-video difficult crops similarity benchmark across A, B, and C.
Outputs:
  test_outputs/final_pipeline_similarity_matrices.csv
  test_outputs/final_pipeline_threshold_sweep.csv
  test_outputs/final_real_crop_similarity_report.json
"""

import os
import sys
import glob
import time
import json
import csv
import logging
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import BodyGallery, cosine_sim
from enroll_students import augment_image, KNOWN_STUDENTS, FACE_FOLDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Final_ReID_Benchmark")


# --------------------------------------------------------------------------- #
#  Extractors A, B, C                                                         #
# --------------------------------------------------------------------------- #

class ExtractorA:
    """Variant A: MobileNetV3-Small (576-D) + HSV Spatial Histogram (96-D) = 672-D."""
    def __init__(self):
        from reid.embedding_model import PersonReIDExtractor
        self.name = "MobileNetV3+HSV"
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
        self.name = "OSNet-x0.25"
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
        self.name = "ResNet18-IBN-A"
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
#  Gallery Enrollment & Persistence                                           #
# --------------------------------------------------------------------------- #

def build_and_save_gallery(extractor, gallery_dir: str) -> Tuple[BodyGallery, Dict[str, List[np.ndarray]], Dict[str, np.ndarray]]:
    os.makedirs(gallery_dir, exist_ok=True)
    gallery = BodyGallery(dim=extractor.dim)
    student_all_embs = defaultdict(list)
    student_centroid = {}

    for student in KNOWN_STUDENTS:
        folder = Path(f"face/{student}")
        image_files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        
        base_embs = []
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            emb = extractor.extract(img)
            if emb is not None:
                gallery.add(student, emb)
                student_all_embs[student].append(emb)
                base_embs.append(emb)

            for aug_img in augment_image(img):
                aug_emb = extractor.extract(aug_img)
                if aug_emb is not None:
                    gallery.add(student, aug_emb)
                    student_all_embs[student].append(aug_emb)

        # Compute centroid for clean student-to-student macro distance
        if base_embs:
            centroid = np.mean(base_embs, axis=0)
            student_centroid[student] = (centroid / np.linalg.norm(centroid)).astype(np.float32)

    # Save gallery embeddings as npz
    save_data = {}
    for student in KNOWN_STUDENTS:
        save_data[student] = np.array(student_all_embs[student], dtype=np.float32)
        if student in student_centroid:
            save_data[f"{student}_centroid"] = student_centroid[student]
    np.savez_compressed(os.path.join(gallery_dir, "gallery_embeddings.npz"), **save_data)
    logger.info("Enrolled %s -> %s (Sizes: %s)", extractor.name, gallery_dir, gallery.sizes())
    return gallery, student_all_embs, student_centroid


# --------------------------------------------------------------------------- #
#  Similarity & Dissimilarity Analysis                                        #
# --------------------------------------------------------------------------- #

def compute_similarity_analysis(name: str, student_embs: Dict[str, List[np.ndarray]], centroids: Dict[str, np.ndarray]):
    students = list(student_embs.keys())
    
    # 1. Macro Student-vs-Student matrix using centroids
    matrix = np.zeros((len(students), len(students)), dtype=np.float32)
    for i, s1 in enumerate(students):
        for j, s2 in enumerate(students):
            matrix[i, j] = float(np.dot(centroids[s1], centroids[s2]))

    # 2. Pairwise distributions: SAME-STUDENT vs DIFFERENT-STUDENT
    same_sims = defaultdict(list)
    diff_sims = defaultdict(list)
    all_same = []
    all_diff = []

    for i, s1 in enumerate(students):
        embs1 = student_embs[s1]
        for a_idx in range(len(embs1)):
            for b_idx in range(a_idx + 1, len(embs1)):
                sim = float(np.dot(embs1[a_idx], embs1[b_idx]))
                same_sims[f"{s1}-{s1}"].append(sim)
                all_same.append(sim)

        for j in range(i + 1, len(students)):
            s2 = students[j]
            embs2 = student_embs[s2]
            pair_name = f"{s1}-{s2}"
            for e1 in embs1:
                for e2 in embs2:
                    sim = float(np.dot(e1, e2))
                    diff_sims[pair_name].append(sim)
                    all_diff.append(sim)

    all_same_arr = np.array(all_same, dtype=np.float32)
    all_diff_arr = np.array(all_diff, dtype=np.float32)

    stats = {
        "model": name,
        "matrix_students": students,
        "centroid_matrix": matrix.tolist(),
        "same_pairs_count": len(all_same),
        "diff_pairs_count": len(all_diff),
        "same_student_mean": float(np.mean(all_same_arr)),
        "same_student_median": float(np.median(all_same_arr)),
        "same_student_min": float(np.min(all_same_arr)),
        "different_student_mean": float(np.mean(all_diff_arr)),
        "different_student_max": float(np.max(all_diff_arr)),
        "separation_gap": float(np.min(all_same_arr) - np.max(all_diff_arr)),
        "per_pair_same": {k: {"mean": float(np.mean(v)), "min": float(np.min(v))} for k, v in same_sims.items()},
        "per_pair_diff": {k: {"mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in diff_sims.items()},
    }
    return stats, all_same_arr, all_diff_arr


# --------------------------------------------------------------------------- #
#  Threshold Sweep                                                            #
# --------------------------------------------------------------------------- #

def sweep_thresholds(name: str, same_sims: np.ndarray, diff_sims: np.ndarray, thresholds: List[float]) -> List[Dict[str, Any]]:
    rows = []
    total_pos = len(same_sims)
    total_neg = len(diff_sims)

    for tau in thresholds:
        fn = int(np.sum(same_sims < tau))
        tp = total_pos - fn
        fp = int(np.sum(diff_sims >= tau))
        tn = total_neg - fp

        fnr = round(100.0 * fn / max(total_pos, 1), 2)
        fpr = round(100.0 * fp / max(total_neg, 1), 2)
        precision = round(100.0 * tp / max(tp + fp, 1), 2)
        recall = round(100.0 * tp / max(tp + fn, 1), 2)
        accuracy = round(100.0 * (tp + tn) / max(total_pos + total_neg, 1), 2)

        rows.append({
            "model": name,
            "threshold": tau,
            "total_positive": total_pos,
            "fn_count": fn,
            "fnr_pct": fnr,
            "total_negative": total_neg,
            "fp_count": fp,
            "fpr_pct": fpr,
            "precision_pct": precision,
            "recall_pct": recall,
            "accuracy_pct": accuracy,
        })
    return rows


# --------------------------------------------------------------------------- #
#  Real-Video Difficult Crops Benchmark                                       #
# --------------------------------------------------------------------------- #

DIFFICULT_CROPS_SPEC = [
    {"crop_id": 1, "frame": 50, "student": "soham", "cond": "standing_frontal", "bbox": [763, 92, 947, 374]},
    {"crop_id": 2, "frame": 150, "student": "soham", "cond": "walking_motion", "bbox": [602, 61, 818, 359]},
    {"crop_id": 3, "frame": 400, "student": "soham", "cond": "approaching_desk", "bbox": [800, 100, 950, 320]},
    {"crop_id": 4, "frame": 473, "student": "swayam", "cond": "standing_desk", "bbox": [650, 50, 770, 260]},
    {"crop_id": 5, "frame": 800, "student": "swayam", "cond": "seated_desk_side", "bbox": [680, 70, 840, 310]},
    {"crop_id": 6, "frame": 1400, "student": "swayam", "cond": "seated_typing", "bbox": [700, 80, 860, 320]},
    {"crop_id": 7, "frame": 100, "student": "harshil", "cond": "standing_back_far", "bbox": [513, 43, 626, 140]},
    {"crop_id": 8, "frame": 1438, "student": "harshil", "cond": "seated_monitor_occlusion", "bbox": [480, 50, 600, 250]},
    {"crop_id": 9, "frame": 727, "student": "aagam", "cond": "seated_desk_frontal", "bbox": [650, 70, 800, 280]},
    {"crop_id": 10, "frame": 1100, "student": "aagam", "cond": "seated_turning_back", "bbox": [670, 70, 820, 290]},
]

def evaluate_real_crops(extractors: Dict[str, Any], galleries: Dict[str, BodyGallery], thresholds: Dict[str, float], margins: Dict[str, float]):
    cap = cv2.VideoCapture("new lab video.mp4")
    crop_images = {}
    for spec in DIFFICULT_CROPS_SPEC:
        f_idx = spec["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            x1, y1, x2, y2 = spec["bbox"]
            crop = frame[y1:y2, x1:x2].copy()
            crop_images[spec["crop_id"]] = crop
    cap.release()

    results_by_model = {}

    for mod_key, ext in extractors.items():
        gallery = galleries[mod_key]
        tau = thresholds[mod_key]
        margin_req = margins[mod_key]
        
        crop_evals = []
        correct_count = 0
        false_count = 0
        unknown_count = 0

        for spec in DIFFICULT_CROPS_SPEC:
            cid = spec["crop_id"]
            gt = spec["student"]
            img = crop_images.get(cid)
            if img is None:
                continue

            emb = ext.extract(img)
            if emb is None:
                top_m, top_s, sec_s = "UNKNOWN", 0.0, 0.0
            else:
                top_m, top_s, sec_s = gallery.top_match(emb, candidate_students=set(KNOWN_STUDENTS))
                if top_m is None:
                    top_m = "UNKNOWN"

            margin = top_s - sec_s
            accepted = (top_s >= tau) and (margin >= margin_req)

            if not accepted:
                classification = "UNKNOWN"
                unknown_count += 1
            elif top_m == gt:
                classification = "CORRECT"
                correct_count += 1
            else:
                classification = "FALSE_RECOVERY"
                false_count += 1

            crop_evals.append({
                "crop_id": cid,
                "frame": spec["frame"],
                "ground_truth": gt,
                "condition": spec["cond"],
                "top_match": top_m,
                "top_similarity": round(top_s, 4),
                "second_similarity": round(sec_s, 4),
                "margin": round(margin, 4),
                "threshold_applied": tau,
                "margin_required": margin_req,
                "accepted": accepted,
                "classification": classification,
            })

        total = len(crop_evals)
        results_by_model[ext.name] = {
            "model": ext.name,
            "threshold": tau,
            "margin_required": margin_req,
            "total_crops": total,
            "correct_recoveries": correct_count,
            "false_recoveries": false_count,
            "unknown_rejected": unknown_count,
            "accuracy_pct": round(100.0 * correct_count / max(total, 1), 1),
            "false_recovery_rate_pct": round(100.0 * false_count / max(total, 1), 1),
            "rejection_rate_pct": round(100.0 * unknown_count / max(total, 1), 1),
            "crops": crop_evals,
        }

    return results_by_model


# --------------------------------------------------------------------------- #
#  Main Runner                                                                #
# --------------------------------------------------------------------------- #

def main():
    os.makedirs("test_outputs", exist_ok=True)
    logger.info("Initializing extractors A, B, C...")
    ext_a = ExtractorA()
    ext_b = ExtractorB()
    ext_c = ExtractorC()

    logger.info("Building independent galleries...")
    gal_a, embs_a, cent_a = build_and_save_gallery(ext_a, "galleries_mobilenetv3")
    gal_b, embs_b, cent_b = build_and_save_gallery(ext_b, "galleries_osnet_x0_25")
    gal_c, embs_c, cent_c = build_and_save_gallery(ext_c, "galleries_resnet18_ibn_a")

    logger.info("Computing similarity / dissimilarity distributions...")
    stats_a, same_a, diff_a = compute_similarity_analysis(ext_a.name, embs_a, cent_a)
    stats_b, same_b, diff_b = compute_similarity_analysis(ext_b.name, embs_b, cent_b)
    stats_c, same_c, diff_c = compute_similarity_analysis(ext_c.name, embs_c, cent_c)

    # Threshold Sweeps
    logger.info("Executing threshold sweeps...")
    sweep_rows = []
    # For MobileNet+HSV: test 0.65 to 0.85
    sweep_rows.extend(sweep_thresholds(ext_a.name, same_a, diff_a, [0.65, 0.70, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85]))
    # For OSNet-x0.25: test 0.55 to 0.75
    sweep_rows.extend(sweep_thresholds(ext_b.name, same_b, diff_b, [0.55, 0.60, 0.64, 0.66, 0.68, 0.70, 0.72, 0.75]))
    # For ResNet18-IBN-A: test 0.70 to 0.88
    sweep_rows.extend(sweep_thresholds(ext_c.name, same_c, diff_c, [0.70, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85, 0.88]))

    # Save threshold sweep CSV
    sweep_csv_path = "test_outputs/final_pipeline_threshold_sweep.csv"
    with open(sweep_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    logger.info("Saved threshold sweep to %s", sweep_csv_path)

    # Save Similarity Matrices CSV
    sim_csv_path = "test_outputs/final_pipeline_similarity_matrices.csv"
    with open(sim_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Student_Row", "Aagam", "Swayam", "Harshil", "Soham"])
        for stat in [stats_a, stats_b, stats_c]:
            st_list = stat["matrix_students"]
            mat = stat["centroid_matrix"]
            for i, s_row in enumerate(st_list):
                writer.writerow([stat["model"], s_row] + [round(mat[i][j], 4) for j in range(len(st_list))])
    logger.info("Saved similarity matrices to %s", sim_csv_path)

    # Real-Video Difficult Crops Benchmark
    logger.info("Evaluating real-video difficult crops...")
    extractors = {"A": ext_a, "B": ext_b, "C": ext_c}
    galleries = {"A": gal_a, "B": gal_b, "C": gal_c}
    # Optimal verified operating thresholds:
    thresholds = {"A": 0.78, "B": 0.68, "C": 0.80}
    margins = {"A": 0.06, "B": 0.05, "C": 0.05}

    real_crop_results = evaluate_real_crops(extractors, galleries, thresholds, margins)
    real_crop_json_path = "test_outputs/final_real_crop_similarity_report.json"
    with open(real_crop_json_path, "w") as f:
        json.dump(real_crop_results, f, indent=2)
    logger.info("Saved real-crop evaluation report to %s", real_crop_json_path)

    print("\n--- REAL-VIDEO DIFFICULT CROP RESULTS ---")
    for mod_name, res in real_crop_results.items():
        print(f"{mod_name:18s} | Accuracy: {res['accuracy_pct']:5.1f}% | False Rec: {res['false_recovery_rate_pct']:5.1f}% | Rejection: {res['rejection_rate_pct']:5.1f}%")

if __name__ == "__main__":
    main()
