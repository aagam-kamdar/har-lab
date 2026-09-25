"""
run_full_diagnostic_test_video.py
---------------------------------
Comprehensive diagnostic run on the user's exact test_video.mp4:
- Local YOLO person-only detection (classes=[0])
- Official BYTETracker
- Generates output_tracked.mp4
- Logs and computes all 10 summary metrics, per-track lifetime table,
  first 10 identity failure transition cases, root cause analysis,
  and distribution across the physical student locations.
"""

import os
import sys
import json
import logging
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack, STrack, TrackState
from video_processor import UniversalVideoWriter, draw_tracks_on_frame

logging.basicConfig(level=logging.WARNING)


def box_iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def box_distance(b1, b2):
    c1 = box_center(b1)
    c2 = box_center(b2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def main():
    video_path = "test_video.mp4"
    if not os.path.exists(video_path):
        print(f"ERROR: {video_path} not found!")
        return

    output_dir = os.path.join("static", "results", "diagnostic_run")
    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, "output_tracked.mp4")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {video_path}: {width}x{height} @ {fps:.2f} FPS ({total_frames} frames)...")

    # Load local YOLO
    model_path = "yolo26n.pt"
    yolo = YOLO(model_path)

    # Initialize official BYTETracker
    BaseTrack.reset_id()
    args = BYTETrackerArgs(
        track_thresh=0.50,
        track_buffer=30,
        match_thresh=0.80,
        mot20=False,
    )
    tracker = BYTETracker(args=args, frame_rate=int(round(fps)))
    writer = UniversalVideoWriter(out_video_path, fps, width, height)

    # Tracking & Diagnostic Data Structures
    track_meta = {}  # id -> {first_frame, last_frame, lost_count, reactivated, last_box, ...}
    total_person_dets = 0
    peak_simultaneous = 0
    total_reactivations = 0
    total_lost_events = 0
    total_removals = 0

    # For mapping identity transitions
    recent_removed_tracks = [] # list of (track_id, removed_frame, last_box, last_seen_frame)
    identity_transitions = []
    
    # Store lost track states before update
    prev_lost_set = set()
    prev_tracked_set = set()

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # 1. Local YOLO Person Detection (classes=[0])
        res = yolo(frame, conf=0.30, classes=[0], verbose=False)[0]
        num_dets = len(res.boxes)
        total_person_dets += num_dets

        if num_dets > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()[:, None]
            dets = np.hstack([xyxy, confs])
        else:
            dets = np.empty((0, 5), dtype=float)

        # Snapshot before update for diagnostic
        before_tracked = {t.track_id: t for t in tracker.tracked_stracks}
        before_lost = {t.track_id: t for t in tracker.lost_stracks}
        before_lost_boxes = {t.track_id: (t.tlbr.copy(), t.end_frame) for t in tracker.lost_stracks}

        # 2. Official BYTETracker update
        online_targets = tracker.update(dets, (height, width), (height, width))
        active_ids = [t.track_id for t in online_targets]
        peak_simultaneous = max(peak_simultaneous, len(active_ids))

        after_tracked = {t.track_id: t for t in tracker.tracked_stracks}
        after_lost = {t.track_id: t for t in tracker.lost_stracks}
        after_removed = {t.track_id: t for t in tracker.removed_stracks}

        # Check newly lost
        newly_lost = set(after_lost.keys()) - set(before_lost.keys())
        total_lost_events += len(newly_lost)
        for tid in newly_lost:
            if tid in track_meta:
                track_meta[tid]["lost_count"] += 1

        # Check reactivated
        reactivated = set(after_tracked.keys()) & set(before_lost.keys())
        total_reactivations += len(reactivated)
        for tid in reactivated:
            if tid in track_meta:
                track_meta[tid]["reactivated"] = True

        # Check newly removed
        newly_removed = [t for t in tracker.removed_stracks if t.track_id not in [r[0] for r in recent_removed_tracks]]
        total_removals += len(newly_removed)
        for t in newly_removed:
            recent_removed_tracks.append((t.track_id, frame_idx, t.tlbr.copy(), t.end_frame))
            if len(recent_removed_tracks) > 200:
                recent_removed_tracks.pop(0)

        # Check newly created tracks
        newly_created = set(after_tracked.keys()) - set(before_tracked.keys()) - set(before_lost.keys())
        for tid in newly_created:
            trk = after_tracked[tid]
            trk_box = trk.tlbr.copy()
            track_meta[tid] = {
                "id": tid,
                "first_frame": frame_idx,
                "last_frame": frame_idx,
                "lifetime": 1,
                "lost_count": 0,
                "reactivated": False,
                "last_box": trk_box,
                "boxes": [trk_box]
            }

            # Check if this new track corresponds to an old Lost or Removed track in approximately the same position
            # Find closest candidate in before_lost or recent_removed_tracks
            best_candidate = None
            best_iou = 0.0
            best_dist = 9999.0
            cand_status = ""
            cand_last_frame = 0

            # Check currently lost tracks
            for lid, (lbox, lend) in before_lost_boxes.items():
                iou = box_iou(trk_box, lbox)
                dist = box_distance(trk_box, lbox)
                if iou > best_iou or (iou == 0 and dist < best_dist and dist < 150):
                    best_iou = iou
                    best_dist = dist
                    best_candidate = lid
                    cand_status = "Lost"
                    cand_last_frame = lend

            # Check recently removed tracks
            for rid, rframe, rbox, rend in recent_removed_tracks:
                if rid == best_candidate: continue
                iou = box_iou(trk_box, rbox)
                dist = box_distance(trk_box, rbox)
                if (iou > best_iou and iou > 0.15) or (best_candidate is None and dist < 100):
                    best_iou = iou
                    best_dist = dist
                    best_candidate = rid
                    cand_status = "Removed"
                    cand_last_frame = rend

            # If there's a strong spatial match, record the transition case
            if best_candidate is not None and (best_iou > 0.10 or best_dist < 100):
                gap = frame_idx - cand_last_frame
                
                # Determine root cause
                if cand_status == "Removed" and gap > args.track_buffer:
                    cause_desc = f"Track buffer ({args.track_buffer} frames) expired after {gap} frame gap; old track was Removed (C/A)"
                    cat = "C"
                elif cand_status == "Lost" and best_iou > 0.10:
                    cause_desc = f"Lost track was present nearby (IoU={best_iou:.2f}) but ByteTrack high_thresh/match_thresh failed association (B/E)"
                    cat = "B"
                elif best_dist < 80:
                    cause_desc = f"Person detected nearby ({best_dist:.1f}px) after detection drop of {gap} frames (A/F)"
                    cat = "A"
                else:
                    cause_desc = f"Occlusion / spatial displacement (gap={gap} frames, dist={best_dist:.1f}px) (D/E)"
                    cat = "D"

                identity_transitions.append({
                    "old_id": best_candidate,
                    "new_id": tid,
                    "old_last_frame": cand_last_frame,
                    "new_first_frame": frame_idx,
                    "gap_frames": gap,
                    "iou": float(best_iou),
                    "dist_px": float(best_dist),
                    "det_conf": float(trk.score),
                    "old_status": cand_status,
                    "cause": cause_desc,
                    "category": cat
                })

        # Update last_seen and boxes for active tracks
        for t in online_targets:
            tid = t.track_id
            if tid in track_meta:
                track_meta[tid]["last_frame"] = frame_idx
                track_meta[tid]["lifetime"] = frame_idx - track_meta[tid]["first_frame"] + 1
                track_meta[tid]["last_box"] = t.tlbr.copy()
                if len(track_meta[tid]["boxes"]) < 50:
                    track_meta[tid]["boxes"].append(t.tlbr.copy())

        # 3. Draw annotations & write output video
        annotated_frame = draw_tracks_on_frame(frame, online_targets)
        writer.write(annotated_frame)

        if frame_idx % 500 == 0 or frame_idx == 1:
            print(f"  Frame {frame_idx:4d}/{total_frames} | active={len(active_ids)} | total unique={len(track_meta)}")

    cap.release()
    writer.release()
    print("Video processing and MP4 writing completed.")

    # -------------------------------------------------------------
    # CALCULATE METRICS & REPORT
    # -------------------------------------------------------------
    lifetimes = [m["lifetime"] for m in track_meta.values()]
    median_life = float(np.median(lifetimes)) if lifetimes else 0
    min_life = int(np.min(lifetimes)) if lifetimes else 0
    max_life = int(np.max(lifetimes)) if lifetimes else 0
    total_unique_ids = len(track_meta)
    avg_dets = total_person_dets / frame_idx if frame_idx > 0 else 0

    results_data = {
        "total_frames": frame_idx,
        "total_person_dets": total_person_dets,
        "avg_dets_per_frame": round(avg_dets, 2),
        "peak_simultaneous_tracks": peak_simultaneous,
        "total_unique_ids": total_unique_ids,
        "total_track_creations": total_unique_ids,
        "total_reactivations": total_reactivations,
        "total_lost_events": total_lost_events,
        "total_removals": total_removals,
        "lifetime_stats": {
            "median": median_life,
            "min": min_life,
            "max": max_life
        },
        "tracks": list(track_meta.values()),
        "transitions": identity_transitions
    }

    with open(os.path.join(output_dir, "diagnostic_data.json"), "w") as f:
        # serialize numpy objects cleanly
        json.dump(results_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    # Print the full comprehensive formatted report to stdout
    SEP = "=" * 80
    print("\n" + SEP)
    print("DIAGNOSTIC REPORT ON test_video.mp4")
    print(SEP)
    print(f"1. Total video frames                 : {frame_idx}")
    print(f"2. Total person detections            : {total_person_dets}")
    print(f"3. Average person detections/frame    : {avg_dets:.2f}")
    print(f"4. Peak simultaneous active tracks    : {peak_simultaneous}")
    print(f"5. Total unique track IDs created     : {total_unique_ids}")
    print(f"6. Total track creations              : {total_unique_ids}")
    print(f"7. Total reactivations                : {total_reactivations}")
    print(f"8. Total lost events                  : {total_lost_events}")
    print(f"9. Total removed tracks               : {total_removals}")
    print(f"10. Track lifetime statistics:")
    print(f"    - Median lifetime                 : {median_life:.1f} frames ({median_life/fps:.2f}s)")
    print(f"    - Minimum lifetime                : {min_life} frames ({min_life/fps:.2f}s)")
    print(f"    - Maximum lifetime                : {max_life} frames ({max_life/fps:.2f}s)")

    print("\n" + SEP)
    print("PER-TRACK SUMMARY TABLE (All Unique Track IDs)")
    print(SEP)
    print(f"{'ID':>4} | {'First Frame':>11} | {'Last Frame':>10} | {'Lifetime':>8} | {'Lost Count':>10} | {'Reactivated':>11}")
    print("-" * 68)
    for tid in sorted(track_meta.keys()):
        m = track_meta[tid]
        re_str = "Yes" if m["reactivated"] else "No"
        print(f"{tid:>4} | {m['first_frame']:>11} | {m['last_frame']:>10} | {m['lifetime']:>8} | {m['lost_count']:>10} | {re_str:>11}")

    print("\n" + SEP)
    print("FIRST 10 IDENTITY FAILURE TRANSITION CASES (Lost/Removed Track → New ID)")
    print(SEP)
    for i, t in enumerate(identity_transitions[:10], 1):
        print(f"Case {i:2d}:")
        print(f"  Old ID          : ID {t['old_id']}")
        print(f"  New ID          : ID {t['new_id']}")
        print(f"  Old Last Frame  : Frame {t['old_last_frame']}")
        print(f"  New First Frame : Frame {t['new_first_frame']}")
        print(f"  Gap in Frames   : {t['gap_frames']} frames ({t['gap_frames']/fps:.2f}s)")
        print(f"  IoU Overlap     : {t['iou']:.3f} (Spatial Distance: {t['dist_px']:.1f}px)")
        print(f"  Detection Conf  : {t['det_conf']:.2f}")
        print(f"  Old Track State : {t['old_status']}")
        print(f"  Root Cause      : {t['cause']}")
        print("-" * 60)

    # Cluster tracks spatially by centroid to show distribution across 10 people
    print("\n" + SEP)
    print("DISTRIBUTION OF TRACK IDs ACROSS ACTUAL PHYSICAL PERSON LOCATIONS")
    print(SEP)
    # Group tracks by approximate center coordinate
    locations = []
    for tid, m in track_meta.items():
        mean_box = np.mean(m["boxes"], axis=0)
        c = box_center(mean_box)
        placed = False
        for loc in locations:
            # If within 100 pixels of existing location cluster
            if np.sqrt((c[0] - loc["center"][0])**2 + (c[1] - loc["center"][1])**2) < 90:
                loc["ids"].append((tid, m["first_frame"], m["last_frame"], m["lifetime"]))
                placed = True
                break
        if not placed:
            locations.append({
                "center": c,
                "ids": [(tid, m["first_frame"], m["last_frame"], m["lifetime"])]
            })

    for i, loc in enumerate(locations, 1):
        loc_ids = loc["ids"]
        id_list = [f"ID {item[0]} (f{item[1]}-f{item[2]})" for item in loc_ids]
        print(f"Person Location #{i:2d} (approx ({int(loc['center'][0])}, {int(loc['center'][1])})):")
        print(f"  Total IDs spawned for this person : {len(loc_ids)}")
        print(f"  Sequence of IDs                   : {' -> '.join(id_list)}")
        print()

    print(SEP)
    print(f"Annotated video written to: {out_video_path}")
    print(SEP)

if __name__ == "__main__":
    main()
