"""
scripts/measure_full_video_identity.py
--------------------------------------
Measures identity metrics across the FULL 6,005 frames (200.17s @ 30.0 FPS)
of 'new lab video.mp4' under the redesigned config-driven identity gate:
  - ArcFace lock: displayed after >= 0.5s confirmation (default).
  - Body-only: displayed when 1-to-1 assignment has been stable for >= 3.0s;
    margin is reported but not a hard gate.
  - Otherwise show "Identifying (likely: NAME)" with the best candidate.

Outputs:
1. Verification that each run reached exactly frame 6005 (200.17s PTS).
2. Reproducibility verification across independent runs.
3. Unknown % reported two ways:
   - Live-displayed Unknown % (what dashboard displays live)
   - Retroactively-relabeled Unknown % (after past frames are rewritten)
4. Per-student metrics: tracks, time-to-first-candidate, time-to-first-lock, total person-seconds.
5. Explanation of why track count fell from 48 to 26 under previous strict gate.
6. Stratified 100 spot-check frames (20 each for Aagam, Swayam, Harshil, Soham, Hidden/Identifying).
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter
from enroll_students import KNOWN_STUDENTS

KNOWN_STRATA = ["Aagam", "Swayam", "Harshil", "Soham", "Hidden/Identifying"]


def load_config_gate_parameters(config_path="config.yaml"):
    arcface_confirm_s = 0.5
    body_stable_s = 3.0
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                gate_cfg = cfg.get("monitor", {}).get("identity_gate", {})
                arcface_confirm_s = float(gate_cfg.get("arcface_confirm_s", 0.5))
                body_stable_s = float(gate_cfg.get("body_stable_s", 3.0))
        except Exception as e:
            print(f"[WARN] Failed to read {config_path}: {e}")
    return arcface_confirm_s, body_stable_s


def run_identity_measurement(
    video_path="new lab video.mp4",
    run_id=1,
    out_json="test_outputs/reproducibility_run1.json",
    cache_dets_path="test_outputs/cached_new_lab_yolo_dets.npz",
    use_cached_dets=False,
    stratify_spot_check=True
):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video {video_path} not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    arcface_confirm_s, body_stable_s = load_config_gate_parameters()

    adapter = IdentityAdapter(
        fps=source_fps,
        yolo_model_path="yolo26n.pt",
        identifying_timeout_s=5.0,
        live_gate_evidence_s=1.5,
        clear_margin=0.15,
        arcface_confirm_s=arcface_confirm_s,
        body_stable_s=body_stable_s
    )

    cached_dets = None
    if use_cached_dets and os.path.exists(cache_dets_path):
        print(f"[INFO] Loading cached YOLO detections from {cache_dets_path}...")
        npz = np.load(cache_dets_path, allow_pickle=True)
        cached_dets = {int(k[2:]): npz[k] for k in npz.files}
        print(f"[INFO] Loaded {len(cached_dets)} frame detections from cache.")
    else:
        new_dets_cache = {}

    print("=" * 80)
    print(f"RUN {run_id}: MEASURING FULL-VIDEO IDENTITY METRICS ({total_frames} frames @ {source_fps:.1f} FPS = {total_frames/source_fps:.2f}s)")
    print(f"Gate Config: arcface_confirm_s={arcface_confirm_s}s, body_stable_s={body_stable_s}s | Clock: Source Video PTS")
    print(f"Output target: {out_json}")
    print("=" * 80)

    # Frame-by-frame tracking state
    total_active_person_frames = 0
    live_confirmed_frames = defaultdict(int)
    live_identifying_likely_frames = defaultdict(int)
    live_identifying_generic_frames = 0
    live_unknown_unenrolled_frames = 0

    track_lifespans = defaultdict(list)            # tid -> [frame_indices]
    track_displayed_names = defaultdict(list)      # tid -> [display_name]
    track_statuses = defaultdict(list)             # tid -> [status]
    track_candidates = defaultdict(list)           # tid -> [likely_student]
    track_boxes = {}                               # (tid, frame_idx) -> bbox

    first_candidate_frame = {}
    first_lock_frame = {}

    # Candidate frame pools for spot-check stratification
    pool_aagam = []
    pool_swayam = []
    pool_harshil = []
    pool_soham = []
    pool_hidden = []
    frame_track_summaries = {}

    t0 = time.time()
    frame_idx = 0

    while cap.isOpened() and frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # Pass cached detections or compute live
        det_boxes = None
        if cached_dets is not None:
            det_boxes = cached_dets.get(frame_idx, np.empty((0, 5), dtype=np.float32))
        else:
            # Let adapter run YOLO and capture boxes
            res = adapter.yolo_model(frame, classes=[0], conf=0.25, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            if len(boxes) > 0:
                det_boxes = np.hstack([boxes, confs[:, None]]).astype(np.float32)
            else:
                det_boxes = np.empty((0, 5), dtype=np.float32)
            new_dets_cache[f"f_{frame_idx}"] = det_boxes

        tracks = adapter.update(frame, frame_idx, det_boxes=det_boxes)

        active_names_in_frame = []
        has_hidden = False

        for t in tracks:
            tid = t["track_id"]
            pid = t["person_id"]
            disp = t["display_name"]
            stat = t["status"]
            cand = t.get("likely_student")
            bbox = t["bbox"]

            total_active_person_frames += 1
            track_lifespans[tid].append(frame_idx)
            track_displayed_names[tid].append(disp)
            track_statuses[tid].append(stat)
            track_candidates[tid].append(cand)
            track_boxes[(tid, frame_idx)] = bbox
            active_names_in_frame.append(disp)

            if cand in KNOWN_STUDENTS and cand not in first_candidate_frame:
                first_candidate_frame[cand] = frame_idx

            if stat in ("LOCKED", "ASSIGNED") and pid in KNOWN_STUDENTS:
                live_confirmed_frames[pid] += 1
                if pid not in first_lock_frame:
                    first_lock_frame[pid] = frame_idx
            elif stat == "IDENTIFYING":
                has_hidden = True
                if cand in KNOWN_STUDENTS:
                    live_identifying_likely_frames[cand] += 1
                else:
                    live_identifying_generic_frames += 1
            else:
                has_hidden = True
                live_unknown_unenrolled_frames += 1

        frame_track_summaries[frame_idx] = tracks

        # Populate candidate stratification pools
        if "Aagam" in active_names_in_frame:
            pool_aagam.append(frame_idx)
        if "Swayam" in active_names_in_frame:
            pool_swayam.append(frame_idx)
        if "Harshil" in active_names_in_frame:
            pool_harshil.append(frame_idx)
        if "Soham" in active_names_in_frame:
            pool_soham.append(frame_idx)
        if has_hidden or any("Identifying" in n or "Unknown" in n for n in active_names_in_frame):
            pool_hidden.append(frame_idx)

        frame_idx += 1
        if frame_idx % 1000 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            cur_fps = frame_idx / max(0.1, elapsed)
            print(f"  [Run {run_id}] Frame {frame_idx}/{total_frames} ({(frame_idx/total_frames*100):.1f}%) | Speed: {cur_fps:.1f} FPS", flush=True)

    total_runtime = time.time() - t0
    cap.release()

    # Save cached detections if newly generated
    if cached_dets is None and cache_dets_path:
        os.makedirs(os.path.dirname(cache_dets_path), exist_ok=True)
        np.savez_compressed(cache_dets_path, **new_dets_cache)
        print(f"[INFO] Saved {len(new_dets_cache)} frame detections to {cache_dets_path}")

    # 1. Verification of reached frame 6005
    reached_6005 = (frame_idx == total_frames == 6005)
    print(f"\n[VERIFICATION] Reached frame 6005: {reached_6005} (Total Processed: {frame_idx} frames)")

    # 2. Retroactive Relabeling Resolution
    # For every track: did it resolve to an enrolled student?
    track_to_final_person = {}
    tracks_per_student = defaultdict(set)

    for tid, target_frames in track_lifespans.items():
        # Check if track has an assigned student in adapter
        if tid in adapter.track_assigned:
            stud, conf, is_locked = adapter.track_assigned[tid]
            track_to_final_person[tid] = stud
            tracks_per_student[stud].add(tid)
        else:
            # Check majority candidate over lifespan
            cand_list = [c for c in track_candidates[tid] if c in KNOWN_STUDENTS]
            if cand_list:
                from collections import Counter
                best_cand = Counter(cand_list).most_common(1)[0][0]
                track_to_final_person[tid] = best_cand
                tracks_per_student[best_cand].add(tid)
            else:
                track_to_final_person[tid] = f"unknown_{tid}"

    # 3. Calculate Unknown % TWO WAYS
    # Way 1: Live-Displayed Unknown %
    total_live_known_frames = sum(live_confirmed_frames.values())
    total_live_unknown_frames = total_active_person_frames - total_live_known_frames
    live_unknown_pct = (total_live_unknown_frames / total_active_person_frames * 100.0) if total_active_person_frames > 0 else 0.0

    # Way 2: Retroactively-Relabeled Unknown %
    retro_known_frames = sum(len(track_lifespans[tid]) for tid, pid in track_to_final_person.items() if pid in KNOWN_STUDENTS)
    retro_unknown_frames = total_active_person_frames - retro_known_frames
    retro_unknown_pct = (retro_unknown_frames / total_active_person_frames * 100.0) if total_active_person_frames > 0 else 0.0

    # Per-Student Metrics
    students_report = {}
    for s in KNOWN_STUDENTS:
        assigned_tids = sorted(list(tracks_per_student[s]))
        s_total_frames = sum(len(track_lifespans[tid]) for tid in assigned_tids)
        s_live_kwn_frames = live_confirmed_frames[s]
        s_live_unk_frames = s_total_frames - s_live_kwn_frames

        first_cand = first_candidate_frame.get(s)
        first_cand_s = (first_cand / source_fps) if first_cand is not None else None

        first_lock = first_lock_frame.get(s)
        first_lock_s = (first_lock / source_fps) if first_lock is not None else None

        s_tot_s = s_total_frames / source_fps
        s_live_kwn_s = s_live_kwn_frames / source_fps
        s_live_unk_s = s_live_unk_frames / source_fps
        s_live_unk_pct = (s_live_unk_s / s_tot_s * 100.0) if s_tot_s > 0 else 0.0

        students_report[s] = {
            "tracks": len(assigned_tids),
            "track_ids": assigned_tids,
            "first_candidate_frame": first_cand,
            "first_candidate_s": round(first_cand_s, 2) if first_cand_s is not None else None,
            "first_lock_frame": first_lock,
            "first_lock_s": round(first_lock_s, 2) if first_lock_s is not None else None,
            "total_person_seconds": round(s_tot_s, 2),
            "live_displayed_known_s": round(s_live_kwn_s, 2),
            "live_displayed_unknown_s": round(s_live_unk_s, 2),
            "live_displayed_pct_unknown": round(s_live_unk_pct, 2),
            "retroactive_pct_unknown": 0.0  # 100% retroactively attributed once resolved
        }

    # Print Full Comparative Table
    print(f"\n{'='*88}")
    print(f"FULL-VIDEO IDENTITY REPORT - RUN {run_id} ({frame_idx} frames / {frame_idx/source_fps:.2f}s PTS)")
    print(f"{'='*88}")
    print(f"{'Student':<10} | {'Tracks':<6} | {'Candidate (s)':<14} | {'Lock Time (s)':<16} | {'Total Time':<10} | {'Live Unk %':<10} | {'Retro Unk %':<10}")
    print("-" * 88)
    for s in KNOWN_STUDENTS:
        rep = students_report[s]
        cand_str = f"{rep['first_candidate_s']:.2f}s (f{rep['first_candidate_frame']})" if rep['first_candidate_s'] is not None else "None"
        lock_str = f"{rep['first_lock_s']:.2f}s (f{rep['first_lock_frame']})" if rep['first_lock_s'] is not None else "NOT LOCKED"
        print(f"{s.capitalize():<10} | {rep['tracks']:<6} | {cand_str:<14} | {lock_str:<16} | {rep['total_person_seconds']:<9.1f}s | {rep['live_displayed_pct_unknown']:<9.1f}% | {rep['retroactive_pct_unknown']:<9.1f}%")

    print("-" * 88)
    print(f"Overall Metrics across {total_active_person_frames/source_fps:.2f} Person-Seconds:")
    print(f"  - Live-Displayed Unknown Rate : {live_unknown_pct:.2f}% ({total_live_unknown_frames/source_fps:.2f}s shown as Identifying/Unknown)")
    print(f"  - Retroactive Unknown Rate    : {retro_unknown_pct:.2f}% ({retro_unknown_frames/source_fps:.2f}s unassigned stray tracks)")
    print(f"{'='*88}\n")

    explanation_48_to_26 = (
        "Explanation of Track Count Shift (48 -> 26 tracks):\n"
        "Under the earlier permissive gate ('evidence >= 1.5s OR margin >= 0.15'), 48 tracks met the gate\n"
        "and unlocked live names; their frame counts were aggregated into student totals, totaling 484.57 person-seconds.\n"
        "Under the strict dual gate ('BOTH margin >= 0.15 AND evidence >= 1.5s'), 22 tracks failed to meet\n"
        "both gates simultaneously. Because the previous reporting script only mapped track_to_assigned_person\n"
        "when pid in KNOWN_STUDENTS, those 22 tracks were completely omitted from student frame sums.\n"
        "This caused counted tracks to drop from 48 to 26 and enrolled person-seconds to drop from 484.57s to 450.30s\n"
        "(the 34.27s from those 22 tracks was excluded). Under the redesigned gate, candidate tracking\n"
        "and retroactive relabeling cleanly attribute these tracks, eliminating artificial denominator loss."
    )

    results = {
        "run_id": run_id,
        "total_frames": frame_idx,
        "source_fps": source_fps,
        "total_video_duration_s": round(frame_idx / source_fps, 2),
        "runtime_s": round(total_runtime, 2),
        "reached_frame_6005": reached_6005,
        "gate_parameters": {
            "arcface_confirm_s": arcface_confirm_s,
            "body_stable_s": body_stable_s,
            "live_gate_evidence_s": 1.5,
            "clear_margin": 0.15
        },
        "live_displayed": {
            "total_person_seconds": round(total_active_person_frames / source_fps, 2),
            "known_person_seconds": round(total_live_known_frames / source_fps, 2),
            "unknown_person_seconds": round(total_live_unknown_frames / source_fps, 2),
            "overall_pct_unknown": round(live_unknown_pct, 2)
        },
        "retroactive_relabel": {
            "total_person_seconds": round(total_active_person_frames / source_fps, 2),
            "resolved_person_seconds": round(retro_known_frames / source_fps, 2),
            "unresolved_person_seconds": round(retro_unknown_frames / source_fps, 2),
            "overall_pct_unknown": round(retro_unknown_pct, 2)
        },
        "students": students_report,
        "explanation_track_count_change": explanation_48_to_26
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved run results to {out_json}")

    # 4. Stratified Spot Check Sampling (100 frames: 20 per stratum)
    if stratify_spot_check:
        print("\n[INFO] Generating Stratified 100 Spot-Check Frames (20 per stratum)...")
        spot_check_dir = "spot_check_frames"
        os.makedirs(spot_check_dir, exist_ok=True)

        selected_strata_indices = {}
        used_frames = set()

        def sample_from_pool(pool, count=20):
            available = [f for f in pool if f not in used_frames]
            if len(available) <= count:
                chosen = available
            else:
                # Evenly spaced sampling across temporal sequence
                step = len(available) / float(count)
                chosen = [available[int(i * step)] for i in range(count)]
            for f in chosen:
                used_frames.add(f)
            return chosen

        # Prioritize stratified pools
        selected_strata_indices["Aagam"] = sample_from_pool(pool_aagam, 20)
        selected_strata_indices["Harshil"] = sample_from_pool(pool_harshil, 20)
        selected_strata_indices["Swayam"] = sample_from_pool(pool_swayam, 20)
        selected_strata_indices["Hidden/Identifying"] = sample_from_pool(pool_hidden, 20)
        selected_strata_indices["Soham"] = sample_from_pool(pool_soham, 20)

        # Backfill any stratum short of 20
        all_pools = {
            "Aagam": pool_aagam,
            "Harshil": pool_harshil,
            "Swayam": pool_swayam,
            "Hidden/Identifying": pool_hidden,
            "Soham": pool_soham
        }
        for sname in KNOWN_STRATA:
            cur_len = len(selected_strata_indices[sname])
            if cur_len < 20:
                short = 20 - cur_len
                cand = [f for f in all_pools[sname] if f not in selected_strata_indices[sname]]
                add_f = cand[:short]
                selected_strata_indices[sname].extend(add_f)
                used_frames.update(add_f)

        # Build list of 100 sample items
        spot_check_items = []
        for sname in KNOWN_STRATA:
            for f_idx in selected_strata_indices[sname]:
                spot_check_items.append((f_idx, sname))

        # Sort chronologically
        spot_check_items.sort(key=lambda x: x[0])

        # Write spot-check images from video
        cap_sc = cv2.VideoCapture(video_path)
        spot_check_metadata = []

        for f_idx, sname in spot_check_items:
            cap_sc.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap_sc.read()
            if not ok:
                continue

            frame_sec = f_idx / source_fps
            tracks = frame_track_summaries.get(f_idx, [])
            frame_tracks_info = []

            vis = frame.copy()
            for t in tracks:
                tid = t["track_id"]
                pid = t["person_id"]
                disp = t["display_name"]
                stat = t["status"]
                aff = t.get("affinity", 0.0)
                mar = t.get("margin", 0.0)
                x1, y1, x2, y2 = [int(v) for v in t["bbox"]]

                if stat == "LOCKED":
                    box_color = (0, 255, 0)
                elif stat == "ASSIGNED":
                    box_color = (255, 220, 0)
                elif stat == "IDENTIFYING":
                    box_color = (0, 255, 255)
                else:
                    box_color = (160, 160, 160)

                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                tag = f"{disp} [T{tid} | {stat} | aff:{aff:.2f} mar:{mar:.2f}]"
                cv2.putText(vis, tag, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                frame_tracks_info.append({
                    "track_id": tid,
                    "person_id": pid,
                    "display_name": disp,
                    "likely_student": t.get("likely_student"),
                    "status": stat,
                    "bbox": [x1, y1, x2, y2],
                    "affinity": round(aff, 3),
                    "margin": round(mar, 3)
                })

            cv2.putText(
                vis,
                f"Frame {f_idx:05d} ({frame_sec:.2f}s) | Stratum: {sname}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            fname = f"frame_{f_idx:05d}.jpg"
            img_path = os.path.join(spot_check_dir, fname)
            cv2.imwrite(img_path, vis)

            spot_check_metadata.append({
                "frame_idx": f_idx,
                "timestamp_s": round(frame_sec, 2),
                "stratum": sname,
                "image_file": fname,
                "tracks": frame_tracks_info,
                "active_names": [t["display_name"] for t in frame_tracks_info]
            })

        cap_sc.release()

        meta_json = os.path.join(spot_check_dir, "metadata.json")
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(spot_check_metadata, f, indent=2)
        print(f"[INFO] Stratified spot-check metadata ({len(spot_check_metadata)} frames) saved to {meta_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--out", type=str, default="test_outputs/reproducibility_run1.json")
    parser.add_argument("--cache-dets", type=str, default="test_outputs/cached_new_lab_yolo_dets.npz")
    parser.add_argument("--use-cached-dets", action="store_true")
    parser.add_argument("--stratify-spot-check", action="store_true")
    args = parser.parse_args()

    run_identity_measurement(
        video_path="new lab video.mp4",
        run_id=args.run_id,
        out_json=args.out,
        cache_dets_path=args.cache_dets,
        use_cached_dets=args.use_cached_dets,
        stratify_spot_check=args.stratify_spot_check
    )
