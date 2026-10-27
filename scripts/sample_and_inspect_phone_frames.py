"""
scripts/sample_and_inspect_phone_frames.py
Samples 30 random frames from the 253 candidate frames across the 5 phone candidate clips.
Saves high-resolution crops and full frames for visual inspection of whether a phone is visibly present.
"""

import os
import glob
import cv2
import numpy as np
import random

def sample_phone_candidate_frames(seed=42, n_samples=30):
    clips_dir = "test_outputs/labeling_clips"
    phone_clips = sorted(glob.glob(os.path.join(clips_dir, "*phone*.mp4")))
    
    out_dir = "test_outputs/phone_sample_30"
    os.makedirs(out_dir, exist_ok=True)

    all_frames = []
    for cp in phone_clips:
        cname = os.path.basename(cp)
        cap = cv2.VideoCapture(cp)
        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append((cname, f_idx, frame))
            f_idx += 1
        cap.release()

    total_candidates = len(all_frames)
    print(f"Total candidate frames across {len(phone_clips)} phone clips: {total_candidates}")

    random.seed(seed)
    sampled_indices = sorted(random.sample(range(total_candidates), n_samples))

    manifest = []
    print(f"\nSampled {n_samples} frames (seed={seed}):")
    for sample_i, idx in enumerate(sampled_indices):
        cname, f_idx, frame = all_frames[idx]
        h, w = frame.shape[:2]
        
        # Save full frame
        frame_name = f"sample_{sample_i+1:02d}_{cname}_f{f_idx:03d}.jpg"
        full_path = os.path.join(out_dir, frame_name)
        cv2.imwrite(full_path, frame)
        
        manifest.append({
            "sample_num": sample_i + 1,
            "candidate_index": idx,
            "clip": cname,
            "frame_idx": f_idx,
            "file": frame_name,
            "path": full_path
        })
        print(f"  #{sample_i+1:02d}: Clip={cname}, Frame={f_idx} -> {frame_name}")

    return manifest

if __name__ == "__main__":
    sample_phone_candidate_frames()
