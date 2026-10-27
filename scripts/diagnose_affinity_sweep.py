"""
scripts/diagnose_affinity_sweep.py
----------------------------------
Explains and fixes the flat rejection sweep:
1. Prints each track's best face EMA, body EMA, and Hungarian affinity score.
2. Explains which score the threshold is applied to.
3. Shows the exact numerator and denominator for every percentage.
4. Sweeps threshold tau properly (uncoupled from any hardcoded 0.52).
"""

import os
import sys
import json
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter
from enroll_students import KNOWN_STUDENTS

CACHE_DETS = "test_outputs/cached_new_lab_yolo_dets.npz"
VIDEO_PATH = "new lab video.mp4"


def run_track_affinity_audit():
    print("=" * 90)
    print("PER-TRACK BEST SCORE & REJECTION SWEEP AUDIT")
    print("=" * 90)

    npz = np.load(CACHE_DETS, allow_pickle=True)
    cached_dets = {int(k[2:]): npz[k] for k in npz.files}

    adapter = IdentityAdapter(
        fps=30.0,
        yolo_model_path="yolo26n.pt",
        identifying_timeout_s=5.0,
        live_gate_evidence_s=1.5,
        clear_margin=0.15,
        arcface_confirm_s=0.5,
        body_stable_s=3.0,
        mode="redesigned",
        min_affinity_thresh=0.0  # zero out rejection to capture ALL best raw scores
    )

    import cv2
    cap = cv2.VideoCapture(VIDEO_PATH)

    track_lifespans = defaultdict(list)
    track_max_face = defaultdict(float)
    track_max_body = defaultdict(float)
    track_max_aff = defaultdict(float)
    track_best_student = defaultdict(str)

    frame_idx = 0
    while cap.isOpened() and frame_idx < 6005:
        ok, frame = cap.read()
        if not ok:
            break
        det_boxes = cached_dets.get(frame_idx, np.empty((0, 5), dtype=np.float32))
        tracks = adapter.update(frame, frame_idx, det_boxes=det_boxes)

        for t in tracks:
            tid = t["track_id"]
            track_lifespans[tid].append(frame_idx)

            # Record max scores observed for this track
            f_ema = adapter.track_face_ema.get(tid, {})
            b_ema = adapter.track_body_ema.get(tid, {})

            for s in KNOWN_STUDENTS:
                fs = f_ema.get(s, 0.0)
                bs = b_ema.get(s, 0.0)
                if fs > track_max_face[tid]:
                    track_max_face[tid] = fs
                if bs > track_max_body[tid]:
                    track_max_body[tid] = bs

            aff = t.get("affinity", 0.0)
            if aff > track_max_aff[tid]:
                track_max_aff[tid] = aff
                if t.get("likely_student"):
                    track_best_student[tid] = t.get("likely_student")

        frame_idx += 1
    cap.release()

    total_tracks = len(track_lifespans)
    total_active_person_frames = sum(len(frames) for frames in track_lifespans.values())

    print(f"\nTotal Tracks: {total_tracks}")
    print(f"Total Active Person Frames: {total_active_person_frames} ({total_active_person_frames/30.0:.2f} person-seconds)")
    print("\n--- PER-TRACK BEST SCORE TABLE ---")
    print(f"{'Track ID':<10} | {'Frames':<8} | {'Best Face':<12} | {'Best Body':<12} | {'Best Aff':<12} | {'Best Match':<12} | {'Locked?':<8}")
    print("-" * 85)

    track_scores = []
    for tid in sorted(track_lifespans.keys()):
        state = adapter.lock_mgr.get_state(tid)
        is_locked = state.locked if state else False
        locked_name = state.locked_name if state else ""
        best_name = locked_name or track_best_student.get(tid, "None")
        n_f = len(track_lifespans[tid])
        mf = track_max_face[tid]
        mb = track_max_body[tid]
        ma = track_max_aff[tid]
        track_scores.append({
            "track_id": tid,
            "frames": n_f,
            "best_face": mf,
            "best_body": mb,
            "best_aff": ma,
            "best_name": best_name,
            "is_locked": is_locked
        })
        print(f"{tid:<10} | {n_f:<8} | {mf:<12.4f} | {mb:<12.4f} | {ma:<12.4f} | {best_name:<12} | {str(is_locked):<8}")

    # Rejection analysis across thresholds tau
    print("\n" + "=" * 90)
    print("DETAILED REJECTION SWEEP ANALYSIS")
    print("=" * 90)
    print("Threshold tau is applied to the Track Assignment Affinity ('aff').")
    print("A track is retained if: it has an authoritative Face Lock OR its best affinity >= tau.")
    print("Denominator of Rejected Track % = Total Unique Tracks (35).")
    print("Denominator of Person-Frame Unknown % = Total Active Person Frames (10,951 frames = 365.03 person-s).")
    print("-" * 90)

    sweep_results = {}
    for tau in [0.30, 0.40, 0.50, 0.60, 0.70]:
        retained = []
        rejected = []
        for ts in track_scores:
            if ts["is_locked"]:
                retained.append(ts)
            elif ts["best_aff"] >= tau:
                retained.append(ts)
            else:
                rejected.append(ts)

        n_rejected = len(rejected)
        rej_pct = (n_rejected / total_tracks) * 100.0

        rejected_frames = sum(ts["frames"] for ts in rejected)
        retro_unk_pct = (rejected_frames / total_active_person_frames) * 100.0

        sweep_results[tau] = {
            "tau": tau,
            "retained_count": len(retained),
            "rejected_count": n_rejected,
            "total_tracks_denom": total_tracks,
            "rejected_pct": round(rej_pct, 2),
            "rejected_frames": rejected_frames,
            "total_frames_denom": total_active_person_frames,
            "retro_unk_pct": round(retro_unk_pct, 2),
            "rejected_tids": [ts["track_id"] for ts in rejected]
        }
        print(f"tau={tau:.2f} | Rejected: {n_rejected:2d} / {total_tracks} tracks ({rej_pct:5.2f}%) | "
              f"Person-Frames: {rejected_frames:5d} / {total_active_person_frames} ({retro_unk_pct:5.2f}%) | "
              f"Rejected Tracks: {[ts['track_id'] for ts in rejected]}")

    out_file = "test_outputs/rejection_sweep_diagnostic.json"
    with open(out_file, "w") as f:
        json.dump({
            "track_scores": track_scores,
            "sweep_results": sweep_results
        }, f, indent=2)
    print(f"\n[INFO] Saved rejection sweep diagnostic to {out_file}")


if __name__ == "__main__":
    run_track_affinity_audit()
