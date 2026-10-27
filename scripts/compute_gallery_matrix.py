"""
scripts/compute_gallery_matrix.py
---------------------------------
Computes:
1. Inter-student body gallery similarity matrix (OSNet x0.25 512-dim):
   - Centroid-to-centroid cosine similarity
   - Mean pairwise embedding cosine similarity
   - Min / Max pairwise embedding cosine similarity
2. Per-pair mean/min similarity of runtime crops from 'new lab video.mp4'
   for each student against each student's enrolled gallery.
"""

import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_final_reid_similarity_benchmark import ExtractorB

STUDENTS = ["aagam", "harshil", "soham", "swayam"]
GALLERY_NPZ = "galleries_osnet_x0_25/gallery_embeddings.npz"
VIDEO_PATH = "new lab video.mp4"
CACHED_POSE_JSON = "test_outputs/cached_yolo26pose_detections_1800f.json"


def compute_inter_student_matrix():
    print("=" * 80)
    print("1. INTER-STUDENT BODY GALLERY SIMILARITY MATRIX (OSNet x0.25 512-dim)")
    print("=" * 80)

    data = np.load(GALLERY_NPZ)
    centroids = {s: data[f"{s}_centroid"] for s in STUDENTS}
    embeddings = {s: data[s] for s in STUDENTS}

    # Normalize centroids just in case
    for s in STUDENTS:
        c = centroids[s]
        centroids[s] = c / (np.linalg.norm(c) + 1e-8)
        # Normalize embeddings
        embs = embeddings[s]
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embeddings[s] = embs / norms

    # 1A. Centroid Cosine Similarity Matrix
    print("\n--- CENTROID-TO-CENTROID COSINE SIMILARITY MATRIX ---")
    header = f"{'Student':<10} | " + " | ".join(f"{s:<10}" for s in STUDENTS)
    print(header)
    print("-" * len(header))
    centroid_matrix = {}
    for s1 in STUDENTS:
        row = []
        centroid_matrix[s1] = {}
        for s2 in STUDENTS:
            sim = float(np.dot(centroids[s1], centroids[s2]))
            centroid_matrix[s1][s2] = sim
            row.append(f"{sim:10.4f}")
        print(f"{s1:<10} | " + " | ".join(row))

    # 1B. All-pairs Embedding Cosine Similarity (Mean / Min / Max)
    print("\n--- ALL-PAIRS EMBEDDING COSINE SIMILARITY (Mean +/- Std [Min, Max]) ---")
    all_pairs_matrix = {}
    for s1 in STUDENTS:
        all_pairs_matrix[s1] = {}
        for s2 in STUDENTS:
            sim_matrix = np.dot(embeddings[s1], embeddings[s2].T)
            mean_sim = float(np.mean(sim_matrix))
            min_sim = float(np.min(sim_matrix))
            max_sim = float(np.max(sim_matrix))
            std_sim = float(np.std(sim_matrix))
            all_pairs_matrix[s1][s2] = {
                "mean": mean_sim,
                "min": min_sim,
                "max": max_sim,
                "std": std_sim
            }

    print(f"{'Pair':<20} | {'Mean':<10} | {'Min':<10} | {'Max':<10} | {'Std':<10}")
    print("-" * 65)
    for s1 in STUDENTS:
        for s2 in STUDENTS:
            stats = all_pairs_matrix[s1][s2]
            print(f"{s1 + ' vs ' + s2:<20} | {stats['mean']:<10.4f} | {stats['min']:<10.4f} | {stats['max']:<10.4f} | {stats['std']:<10.4f}")

    return centroid_matrix, all_pairs_matrix


def compute_runtime_crops_similarity():
    print("\n" + "=" * 80)
    print("2. PER-PAIR MEAN/MIN SIMILARITY OF RUNTIME CROPS TO EACH GALLERY")
    print("=" * 80)

    # We extract high-confidence runtime crops for each student from new lab video.mp4
    # Ground truth desk locations from desk_zones.yaml:
    # D01: Soham: [450, 0, 570, 360]
    # D02: Swayam: [540, 20, 660, 360]
    # D03: Harshil: [640, 40, 780, 380]
    # D04: Aagam: [770, 50, 960, 420]

    DESK_ZONES = {
        "soham": (450, 570),
        "swayam": (540, 660),
        "harshil": (640, 780),
        "aagam": (770, 960)
    }

    extractor = ExtractorB()
    data = np.load(GALLERY_NPZ)
    galleries = {s: data[s] / (np.linalg.norm(data[s], axis=1, keepdims=True) + 1e-8) for s in STUDENTS}

    with open(CACHED_POSE_JSON, "r") as f:
        pose_data = json.load(f)
    dets_by_frame = {d["frame_idx"]: d["boxes"] for d in pose_data["detections"]}

    cap = cv2.VideoCapture(VIDEO_PATH)

    # Sample up to 50 clear crops per student across the first 1800 frames (sample every 15 frames)
    student_crops_embs = {s: [] for s in STUDENTS}

    f_idx = 0
    while f_idx < 1800:
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx % 15 == 0:
            boxes = dets_by_frame.get(f_idx, [])
            h, w = frame.shape[:2]

            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                cx = (x1 + x2) / 2.0
                bw, bh = x2 - x1, y2 - y1
                if bw < 30 or bh < 60:
                    continue

                for s, (zx1, zx2) in DESK_ZONES.items():
                    if zx1 <= cx <= zx2 and len(student_crops_embs[s]) < 50:
                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop.size > 0:
                            emb = extractor.extract(crop)
                            if emb is not None:
                                student_crops_embs[s].append(emb)

        f_idx += 1

    cap.release()

    for s, embs in student_crops_embs.items():
        print(f"Extracted {len(embs)} runtime crops for {s}")

    # Now compute similarity of each student's runtime crops to all 4 galleries
    # In BodyGallery, matching is top-1 cosine similarity against gallery embeddings.
    # We report:
    # 1. Top-1 match similarity per crop: Mean, Min, Max
    # 2. All-pairs similarity across all crops and gallery embeddings: Mean, Min, Max

    runtime_results = {}
    print("\n--- TOP-1 SIMILARITY OF RUNTIME CROPS TO EACH GALLERY ---")
    print(f"{'Crop Source':<15} | {'Gallery':<10} | {'Mean Top-1':<12} | {'Min Top-1':<12} | {'Max Top-1':<12} | {'Crops':<6}")
    print("-" * 75)

    for s_src in STUDENTS:
        embs = student_crops_embs[s_src]
        if not embs:
            continue
        embs_arr = np.array(embs)  # (N, 512)
        runtime_results[s_src] = {}

        for s_gal in STUDENTS:
            gal = galleries[s_gal]  # (84, 512)
            # Dot product (N, 84)
            sim_matrix = np.dot(embs_arr, gal.T)
            top1_per_crop = np.max(sim_matrix, axis=1)  # (N,)

            mean_top1 = float(np.mean(top1_per_crop))
            min_top1 = float(np.min(top1_per_crop))
            max_top1 = float(np.max(top1_per_crop))

            runtime_results[s_src][s_gal] = {
                "mean_top1": mean_top1,
                "min_top1": min_top1,
                "max_top1": max_top1,
                "num_crops": len(embs)
            }
            is_match = "*" if s_src == s_gal else " "
            print(f"{s_src + is_match:<15} | {s_gal:<10} | {mean_top1:<12.4f} | {min_top1:<12.4f} | {max_top1:<12.4f} | {len(embs):<6}")

    # Also compute all-pairs mean and min similarity between runtime crops and galleries
    print("\n--- ALL-PAIRS MEAN / MIN SIMILARITY (Every Crop to Every Gallery Vector) ---")
    print(f"{'Crop Source':<15} | {'Gallery':<10} | {'All Mean':<12} | {'All Min':<12} | {'All Max':<12}")
    print("-" * 65)
    for s_src in STUDENTS:
        embs = student_crops_embs[s_src]
        if not embs:
            continue
        embs_arr = np.array(embs)
        for s_gal in STUDENTS:
            gal = galleries[s_gal]
            sim_matrix = np.dot(embs_arr, gal.T)
            print(f"{s_src:<15} | {s_gal:<10} | {float(np.mean(sim_matrix)):<12.4f} | {float(np.min(sim_matrix)):<12.4f} | {float(np.max(sim_matrix)):<12.4f}")

    out_file = "test_outputs/inter_student_reid_similarity_matrix.json"
    with open(out_file, "w") as f:
        json.dump({
            "runtime_similarity": runtime_results
        }, f, indent=2)
    print(f"\n[INFO] Saved results to {out_file}")


if __name__ == "__main__":
    compute_inter_student_matrix()
    compute_runtime_crops_similarity()
