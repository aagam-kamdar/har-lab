"""
benchmark_person_scale.py - Scale Benchmark for Person-Centric Lab Monitor
Measures effective FPS and latency across 4, 20, and 40 simulated persons.
"""
import time
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from lab_monitor import PersonTracker, raw_label_person, find_dynamic_discussion, calc_productivity

def run_benchmark():
    print("=" * 65)
    print("PERSON-CENTRIC MONITOR SCALE BENCHMARK (4, 20, 40 Persons)")
    print("=" * 65)

    # 1. Load test frame
    cap = cv2.VideoCapture("new lab video.mp4")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        print("[WARN] Using synthetic 720p frame.")
    else:
        print(f"[INFO] Loaded real video frame: {frame.shape[1]}x{frame.shape[0]}")

    # 2. Load models
    print("[INFO] Loading YOLO-Pose and YOLO-Det models...")
    t0 = time.time()
    pose_model = YOLO("yolov8n-pose.pt")
    det_model = YOLO("yolov8n.pt")
    print(f"[INFO] Models loaded in {time.time() - t0:.2f}s")

    # Load thresholds
    with open("thresholds.yaml", "r") as f:
        th = yaml.safe_load(f)

    # Benchmark full-frame perception
    print("\n[STEP 1] Benchmarking Full-Frame YOLO Perception (1 camera):")
    # Warmup
    _ = pose_model(frame, imgsz=640, verbose=False)
    _ = det_model(frame, imgsz=640, verbose=False)

    n_runs = 10
    t_pose = 0.0
    t_det = 0.0
    for _ in range(n_runs):
        s = time.time()
        p_res = pose_model(frame, imgsz=640, verbose=False)[0]
        t_pose += time.time() - s

        s = time.time()
        d_res = det_model(frame, imgsz=640, verbose=False)[0]
        t_det += time.time() - s

    avg_pose = (t_pose / n_runs) * 1000
    avg_det = (t_det / n_runs) * 1000
    print(f"  - YOLO-Pose (640p): {avg_pose:.1f} ms")
    print(f"  - YOLO-Det  (640p): {avg_det:.1f} ms")
    print(f"  - Subtotal Perception: {avg_pose + avg_det:.1f} ms")

    # Benchmark across person counts: 4, 20, 40
    print("\n[STEP 2] Benchmarking Person Tracking, Feature Extraction, Rules & Dynamic Discussion:")
    scale_counts = [4, 20, 40]
    
    for count in scale_counts:
        # Create simulated persons & features
        trackers = {}
        all_feats = {}
        
        for i in range(count):
            pid = f"student_{i+1:02d}"
            p = PersonTracker(person_id=pid, display_name=f"Student {i+1}", vote_len=8)
            p.track_id = i + 1
            p.since = 0.0
            p.state = "idle"
            trackers[pid] = p

            gx = 200 + (i % 8) * 120
            gy = 200 + (i // 8) * 120
            sw = 55.0
            
            f_sim = {
                "speed": 0.05,
                "motion": 0.22 if i % 2 == 0 else 0.02,
                "hands_low": True,
                "standing": False,
                "wrist_ear": 1.2,
                "ear_hold": 0.0,
                "wrist_phone": 99.0,
                "head_down": True,
                "turn_dir": 1 if i % 4 == 0 else 0,
                "shoulder_mid": (gx, gy),
                "nose": (gx + 15, gy - 20),
                "sw": sw,
                "on_kb": True,
                "since_motion": 0.5,
                "dist_phone": 999.0,
                "dist_kb": 0.3,
                "dist_laptop": 0.4,
                "bbox": [gx - 30, gy - 40, gx + 30, gy + 80]
            }
            all_feats[pid] = f_sim

        # Benchmark feature classification & state machine updates
        n_iters = 50
        t_classify = 0.0
        t_discussion = 0.0

        for _ in range(n_iters):
            # Dynamic discussion check directly
            s_disc = time.time()
            for pid, f in all_feats.items():
                cand, reason = find_dynamic_discussion(pid, f, all_feats, trackers, th)
            t_discussion += time.time() - s_disc

            # Full Rule cascade & debounce state update
            s_cls = time.time()
            now_sim = time.time()
            for pid, f in all_feats.items():
                p = trackers[pid]
                label, reason = raw_label_person(p, f, all_feats, trackers, now_sim, th)
                p.votes.append(label)
                p.acc[label] += 0.125
                _ = calc_productivity(p.acc)
            t_classify += time.time() - s_cls

        avg_disc = (t_discussion / n_iters) * 1000
        avg_cls = (t_classify / n_iters) * 1000
        
        # Total per cycle latency (Perception + Classification + Discussion)
        total_cycle_ms = avg_pose + avg_det + avg_cls
        effective_fps = 1000.0 / total_cycle_ms

        print(f"\n  Scale: {count:2d} Persons")
        print(f"    - Dynamic Discussion Matching ($O(N^2)$): {avg_disc:6.2f} ms")
        print(f"    - Rule Cascade & Debounce Update:         {avg_cls:6.2f} ms")
        print(f"    - Total Logic & State (CPU):              {avg_cls:6.2f} ms")
        print(f"    - Total Cycle Latency (Perception+Logic): {total_cycle_ms:6.2f} ms")
        print(f"    - Effective Throughput:                   {effective_fps:6.1f} FPS")

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    run_benchmark()
