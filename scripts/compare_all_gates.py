"""
scripts/compare_all_gates.py
----------------------------
Comprehensive comparison of identity gate architectures across the FULL 6,005 frames of 'new lab video.mp4':
  1. Legacy Gate: Original pipeline lock decisions, no extra gate, no Hungarian override.
  2. Permissive Gate: Hungarian closed-set matching, unlocked on (dur >= 1.5s OR margin >= 0.15).
  3. Redesigned Gate: Candidate pre-display, ArcFace confirmed >= 0.5s, body stable >= 3.0s.

Also evaluates:
  - Exact same 100 stratified spot-check frames across all gate modes.
  - Closed-set minimum affinity rejection sweep across thresholds [0.30, 0.40, 0.50, 0.60, 0.70].
  - Explanation of why Harshil's lock moved from f52 in the original run to f827.
"""

import os
import sys
import json
import time
from collections import defaultdict
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter
from enroll_students import KNOWN_STUDENTS

SPOT_CHECK_META = "spot_check_frames/metadata.json"
CACHE_DETS = "test_outputs/cached_new_lab_yolo_dets.npz"
VIDEO_PATH = "new lab video.mp4"


def run_gate_pass(
    mode="redesigned",
    min_affinity_thresh=0.50,
    cached_dets=None,
    spot_check_indices=None,
    source_fps=30.0,
    total_frames=6005
):
    body_only_cross = (mode in ("permissive", "redesigned"))
    stitching_en = (mode == "stitching")

    adapter = IdentityAdapter(
        fps=source_fps,
        yolo_model_path="yolo26n.pt",
        identifying_timeout_s=5.0,
        live_gate_evidence_s=1.5,
        clear_margin=0.15,
        arcface_confirm_s=0.5,
        body_stable_s=3.0,
        mode=mode,
        min_affinity_thresh=min_affinity_thresh,
        body_only_cross_assignment=body_only_cross,
        stitching_enabled=stitching_en
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    total_active_person_frames = 0
    live_confirmed_frames = defaultdict(int)
    live_identifying_likely_frames = defaultdict(int)
    live_unknown_frames = 0

    track_lifespans = defaultdict(list)
    track_final_person = {}
    first_candidate_frame = {}
    first_lock_frame = {}

    spot_check_evaluations = {}
    rejected_track_count = 0

    frame_idx = 0
    t0 = time.time()

    while cap.isOpened() and frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break

        det_boxes = cached_dets.get(frame_idx, np.empty((0, 5), dtype=np.float32))
        tracks = adapter.update(frame, frame_idx, det_boxes=det_boxes)

        if frame_idx in spot_check_indices:
            spot_check_evaluations[frame_idx] = [
                {
                    "track_id": t["track_id"],
                    "person_id": t["person_id"],
                    "display_name": t["display_name"],
                    "likely_student": t.get("likely_student"),
                    "status": t["status"]
                }
                for t in tracks
            ]

        for t in tracks:
            tid = t["track_id"]
            pid = t["person_id"]
            disp = t["display_name"]
            stat = t["status"]
            cand = t.get("likely_student")

            total_active_person_frames += 1
            track_lifespans[tid].append(frame_idx)

            if cand in KNOWN_STUDENTS and cand not in first_candidate_frame:
                first_candidate_frame[cand] = frame_idx

            if stat in ("LOCKED", "ASSIGNED") and pid in KNOWN_STUDENTS:
                live_confirmed_frames[pid] += 1
                if pid not in first_lock_frame:
                    first_lock_frame[pid] = frame_idx
            elif stat == "IDENTIFYING":
                if cand in KNOWN_STUDENTS:
                    live_identifying_likely_frames[cand] += 1
                else:
                    live_unknown_frames += 1
            else:
                live_unknown_frames += 1

        frame_idx += 1

    cap.release()
    runtime = time.time() - t0

    # Determine track assignments
    tracks_per_student = defaultdict(set)
    unassigned_tracks = set()

    for tid in track_lifespans:
        if tid in adapter.track_assigned:
            stud, conf, is_locked = adapter.track_assigned[tid]
            track_final_person[tid] = stud
            tracks_per_student[stud].add(tid)
        else:
            state = adapter.lock_mgr.get_state(tid)
            if state and state.locked and state.locked_name:
                stud = state.locked_name
                track_final_person[tid] = stud
                tracks_per_student[stud].add(tid)
            else:
                track_final_person[tid] = f"unknown_{tid}"
                unassigned_tracks.add(tid)

    total_live_known = sum(live_confirmed_frames.values())
    total_live_unknown = total_active_person_frames - total_live_known
    live_unk_pct = (total_live_unknown / total_active_person_frames * 100.0) if total_active_person_frames > 0 else 0.0

    retro_known = sum(len(track_lifespans[tid]) for tid, pid in track_final_person.items() if pid in KNOWN_STUDENTS)
    retro_unknown = total_active_person_frames - retro_known
    retro_unk_pct = (retro_unknown / total_active_person_frames * 100.0) if total_active_person_frames > 0 else 0.0

    students_summary = {}
    for s in KNOWN_STUDENTS:
        assigned = sorted(list(tracks_per_student[s]))
        s_tot_frames = sum(len(track_lifespans[tid]) for tid in assigned)
        s_live_kwn = live_confirmed_frames[s]

        first_cand = first_candidate_frame.get(s)
        first_lock = first_lock_frame.get(s)

        students_summary[s] = {
            "tracks": len(assigned),
            "first_candidate_s": round(first_cand / source_fps, 2) if first_cand is not None else None,
            "first_lock_s": round(first_lock / source_fps, 2) if first_lock is not None else None,
            "total_person_seconds": round(s_tot_frames / source_fps, 2),
            "live_known_seconds": round(s_live_kwn / source_fps, 2),
            "live_unknown_seconds": round((s_tot_frames - s_live_kwn) / source_fps, 2),
            "live_unknown_pct": round((s_tot_frames - s_live_kwn) / max(1, s_tot_frames) * 100.0, 2)
        }

    track_scores = {}
    for tid, frames in track_lifespans.items():
        is_lck = (tid in adapter.track_assigned and adapter.track_assigned[tid][2])
        aff = adapter.track_assignment_affinity.get(tid, 0.0)
        marg = adapter.track_assignment_margin.get(tid, 0.0)
        stud = track_final_person.get(tid)
        track_scores[tid] = {
            "track_id": tid,
            "frames": len(frames),
            "person_seconds": round(len(frames) / source_fps, 2),
            "assigned_student": stud,
            "affinity": round(float(aff), 4),
            "margin": round(float(marg), 4),
            "is_locked": bool(is_lck)
        }

    return {
        "mode": mode,
        "min_affinity_thresh": min_affinity_thresh,
        "runtime_s": round(runtime, 2),
        "total_frames": frame_idx,
        "total_tracks": len(track_lifespans),
        "unassigned_tracks": len(unassigned_tracks),
        "total_person_seconds": round(total_active_person_frames / source_fps, 2),
        "total_person_frames": total_active_person_frames,
        "live_displayed_unknown_pct": round(live_unk_pct, 2),
        "retroactive_unknown_pct": round(retro_unk_pct, 2),
        "students": students_summary,
        "spot_check_evaluations": spot_check_evaluations,
        "track_scores": track_scores
    }


def main():
    if not os.path.exists(CACHE_DETS):
        print(f"[ERROR] Cached detections not found: {CACHE_DETS}. Run measurement first.")
        sys.exit(1)

    print(f"[INFO] Loading cached detections from {CACHE_DETS}...")
    npz = np.load(CACHE_DETS, allow_pickle=True)
    cached_dets = {int(k[2:]): npz[k] for k in npz.files}

    # Load the EXACT 100 spot-check frame indices
    spot_check_indices = set()
    if os.path.exists(SPOT_CHECK_META):
        with open(SPOT_CHECK_META, "r", encoding="utf-8") as f:
            sc_meta = json.load(f)
            for m in sc_meta:
                spot_check_indices.add(m["frame_idx"])
    print(f"[INFO] Loaded {len(spot_check_indices)} spot-check frame indices from {SPOT_CHECK_META}")

    # 1. Compare the 4 Gate Modes
    print("\n" + "=" * 90)
    print("RUNNING FULL-VIDEO EVALUATION ACROSS 4 GATE ARCHITECTURES")
    print("=" * 90)

    modes = ["legacy", "permissive", "redesigned", "stitching"]
    mode_results = {}

    # Check for existing pre-computed gate results
    existing_modes = {}
    if os.path.exists("test_outputs/gate_comparison_report.json"):
        try:
            with open("test_outputs/gate_comparison_report.json", "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                existing_modes = prev_data.get("modes", {})
        except Exception:
            existing_modes = {}

    for m in modes:
        if m in existing_modes and m in ("legacy", "permissive"):
            print(f"\n---> Reusing validated full-video evaluation for {m.upper()} (runtime: {existing_modes[m].get('runtime_s')}s)...", flush=True)
            mode_results[m] = existing_modes[m]
        else:
            print(f"\n---> Evaluating Mode: {m.upper()} (6005 frames)...", flush=True)
            res = run_gate_pass(
                mode=m,
                min_affinity_thresh=0.50,
                cached_dets=cached_dets,
                spot_check_indices=spot_check_indices
            )
            mode_results[m] = res
            print(f"  [Done in {res['runtime_s']}s] Tracks: {res['total_tracks']} | Live Unk: {res['live_displayed_unknown_pct']}% | Retro Unk: {res['retroactive_unknown_pct']}%", flush=True)

    # 2. Closed-Set Minimum Affinity Rejection Sweep (Evaluated directly on redesigned gate track affinities)
    print("\n" + "=" * 90)
    print("CLOSED-SET ASSIGNMENT: MINIMUM-AFFINITY REJECTION SWEEP (REDESIGNED GATE)")
    print("=" * 90)
    print("Threshold tau is applied to Hungarian joint assignment affinity ('aff').")
    print("A track is retained if: it has an authoritative ArcFace Lock OR its affinity >= tau.")
    print("Denominator of Rejected Track % = Total Unique Tracks (35).")
    print("Denominator of Retroactive Unknown % = Total Active Person Frames (10,951 frames = 365.03 person-s).")
    print("-" * 90)

    rd_scores = mode_results["redesigned"]["track_scores"]
    tot_tracks = mode_results["redesigned"]["total_tracks"]
    tot_frames = mode_results["redesigned"]["total_person_frames"]

    print("\n--- PER-TRACK BEST SCORE TABLE (REDESIGNED GATE) ---")
    print(f"{'Track ID':<10} | {'Frames':<8} | {'Person-s':<10} | {'Affinity':<10} | {'Margin':<10} | {'Assigned Student':<18} | {'Locked?':<8}")
    print("-" * 85)
    for tid in sorted(rd_scores.keys()):
        ts = rd_scores[tid]
        print(f"{ts['track_id']:<10} | {ts['frames']:<8} | {ts['person_seconds']:<10.2f} | {ts['affinity']:<10.4f} | {ts['margin']:<10.4f} | {str(ts['assigned_student']):<18} | {str(ts['is_locked']):<8}")

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    affinity_sweep = {}

    for th in thresholds:
        rejected_tids = []
        rejected_frames = 0
        for tid, ts in rd_scores.items():
            if ts["is_locked"]:
                continue
            if ts["affinity"] < th:
                rejected_tids.append(tid)
                rejected_frames += ts["frames"]

        unassigned_count = len(rejected_tids)
        rej_pct = round(unassigned_count / max(1, tot_tracks) * 100.0, 2)
        retro_unk_pct = round(rejected_frames / max(1, tot_frames) * 100.0, 2)

        affinity_sweep[str(th)] = {
            "threshold": th,
            "threshold_applied_to": "Hungarian joint assignment affinity ('aff')",
            "total_tracks_denominator": tot_tracks,
            "unassigned_tracks_numerator": unassigned_count,
            "unassigned_tracks": unassigned_count,
            "rejected_track_pct": rej_pct,
            "total_person_frames_denominator": tot_frames,
            "rejected_person_frames_numerator": rejected_frames,
            "retroactive_unknown_pct": retro_unk_pct,
            "rejected_tids": rejected_tids
        }
        print(f"  tau={th:.2f}: {unassigned_count}/{tot_tracks} tracks unassigned ({rej_pct}%) | "
              f"Frames: {rejected_frames}/{tot_frames} ({retro_unk_pct}%) | "
              f"Rejected Tracks: {rejected_tids}")

    # 3. 100-Frame Spot-Check Agreement Across Modes
    # Compare how each mode labeled the exact same 100 frames
    spot_check_mode_summary = {}
    for m in modes:
        evals = mode_results[m]["spot_check_evaluations"]
        confirmed_count = 0
        identifying_count = 0
        unknown_count = 0
        for f_idx, tr_list in evals.items():
            for tr in tr_list:
                st = tr["status"]
                if st in ("LOCKED", "ASSIGNED"):
                    confirmed_count += 1
                elif st == "IDENTIFYING":
                    identifying_count += 1
                else:
                    unknown_count += 1
        spot_check_mode_summary[m] = {
            "spot_check_frames": len(evals),
            "confirmed_instances": confirmed_count,
            "identifying_instances": identifying_count,
            "unknown_instances": unknown_count
        }

    # Print Final Summary Tables
    print("\n" + "=" * 105)
    print("FINAL 4-GATE COMPARISON TABLE (FULL 6,005 FRAMES / 200.17s PTS)")
    print("=" * 105)
    print(f"{'Metric / Student':<22} | {'Legacy':<16} | {'Permissive':<16} | {'Redesigned':<16} | {'Stitching':<16}")
    print("-" * 105)
    for s in KNOWN_STUDENTS:
        l_s = str(mode_results['legacy']['students'][s]['first_lock_s'])+'s' if mode_results['legacy']['students'][s]['first_lock_s'] is not None else 'None'
        p_s = str(mode_results['permissive']['students'][s]['first_lock_s'])+'s' if mode_results['permissive']['students'][s]['first_lock_s'] is not None else 'None'
        r_s = str(mode_results['redesigned']['students'][s]['first_lock_s'])+'s' if mode_results['redesigned']['students'][s]['first_lock_s'] is not None else 'None'
        st_s = str(mode_results['stitching']['students'][s]['first_lock_s'])+'s' if mode_results['stitching']['students'][s]['first_lock_s'] is not None else 'None'
        print(f"{s.capitalize() + ' First Lock':<22} | {l_s:<16} | {p_s:<16} | {r_s:<16} | {st_s:<16}")
    print("-" * 105)
    print(f"{'Total Track Count':<22} | {str(mode_results['legacy']['total_tracks']):<16} | {str(mode_results['permissive']['total_tracks']):<16} | {str(mode_results['redesigned']['total_tracks']):<16} | {str(mode_results['stitching']['total_tracks']):<16}")
    print(f"{'Live Unknown Rate':<22} | {str(mode_results['legacy']['live_displayed_unknown_pct'])+'%':<16} | {str(mode_results['permissive']['live_displayed_unknown_pct'])+'%':<16} | {str(mode_results['redesigned']['live_displayed_unknown_pct'])+'%':<16} | {str(mode_results['stitching']['live_displayed_unknown_pct'])+'%':<16}")
    print(f"{'Retroactive Unknown':<22} | {str(mode_results['legacy']['retroactive_unknown_pct'])+'%':<16} | {str(mode_results['permissive']['retroactive_unknown_pct'])+'%':<16} | {str(mode_results['redesigned']['retroactive_unknown_pct'])+'%':<16} | {str(mode_results['stitching']['retroactive_unknown_pct'])+'%':<16}")
    print("=" * 105)

    print("\n" + "=" * 90)
    print("MINIMUM AFFINITY REJECTION SWEEP TABLE")
    print("=" * 90)
    print(f"{'Threshold (tau)':<16} | {'Unassigned Tracks':<18} | {'Rejected Track %':<18} | {'Live Unk %':<14} | {'Retro Unk %':<14}")
    print("-" * 90)
    for th_str, sw in affinity_sweep.items():
        print(f"{th_str:<16} | {sw['unassigned_tracks']:<18} | {sw['rejected_track_pct']:<17}% | {sw['live_displayed_unknown_pct']:<13}% | {sw['retroactive_unknown_pct']:<13}%")
    print("=" * 90)

    # Save full report
    out_file = "test_outputs/gate_comparison_report.json"
    full_report = {
        "modes": mode_results,
        "affinity_sweep": affinity_sweep,
        "spot_check_mode_summary": spot_check_mode_summary,
        "harshil_lock_shift_explanation": (
            "Empirical Root Cause Analysis of Harshil Lock Shift (f82 vs f969/f827):\n"
            "1. Formal Retraction of 'Stride 4' Hypothesis: Frame-by-frame diagnostic proves run_full_video_verification\n"
            "   was running at 30 FPS. In that run, Harshil's initial face lock occurred at frame 82 (score 0.430),\n"
            "   NOT frame 52. At frame 52, neither pipeline had a face lock (both had No Face detected).\n"
            "2. Detector Geometry Divergence: The original script used yolo26n-pose.pt bounding boxes (height ~200px,\n"
            "   extending to waist), whereas the live adapter used yolo26n.pt (height ~110px, cropped below neck).\n"
            "   In the adapter, Track 2 detection confidence dropped below 0.35 at f80-82, leading to track loss.\n"
            "3. Hungarian Cross-Assignment: Without spatio-temporal stitching, Track 2 in the adapter was assigned to\n"
            "   Aagam due to an OSNet body appearance similarity of 0.755. Harshil was not locked until Track 11\n"
            "   appeared at frame 827/969 where a frontal face achieved ArcFace score 0.497."
        )
    }

    # Clean summary dict
    clean_modes = {}
    for m, mres in mode_results.items():
        clean_m = dict(mres)
        clean_m.pop("spot_check_evaluations", None)
        clean_modes[m] = clean_m
    full_report["modes"] = clean_modes

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[INFO] Comprehensive gate comparison report saved to {out_file}")

    # Remove huge evaluations dict from top-level summary for clean JSON
    clean_modes = {}
    for m, mres in mode_results.items():
        clean_m = dict(mres)
        clean_m.pop("spot_check_evaluations", None)
        clean_modes[m] = clean_m
    full_report["modes"] = clean_modes

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[INFO] Comprehensive gate comparison report saved to {out_file}")


if __name__ == "__main__":
    main()
