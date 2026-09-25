"""
bytetrack_diagnostic.py
-----------------------
Deep instrumentation of the YOLO + BYTETracker pipeline.

For every frame this script records:
  - frame number
  - YOLO person detection count + bounding boxes
  - active track IDs
  - lost track IDs
  - removed track IDs
  - newly created track IDs (and WHY they were created)
  - track reactivations (lost → tracked again)

At the end it prints a full causal analysis:
  - Which of A–G caused the ID explosion
  - At least 3 identity failure examples with frame numbers
  - A summary table

Usage:
  python bytetrack_diagnostic.py [VIDEO_PATH] [--conf CONF] [--track_buffer BUF] [--match_thresh MT]
"""

import sys
import os
import argparse
import logging
import json
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack, TrackState, STrack
from bytetrack.matching import iou_distance

logging.basicConfig(level=logging.WARNING)   # suppress ultralytics noise

# ---------------------------------------------------------------------------
# IOU helper: overlap between two boxes [x1,y1,x2,y2]
# ---------------------------------------------------------------------------
def box_iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Instrumented BYTETracker wrapper – patches update() to capture internals
# ---------------------------------------------------------------------------
class DiagnosticTracker:
    def __init__(self, args, fps):
        self.tracker = BYTETracker(args=args, frame_rate=int(round(fps)))
        # ----- counters -----
        self.total_track_creations = 0
        self.total_reactivations   = 0
        self.total_removals        = 0
        self.frames_zero_dets      = 0
        self.peak_simultaneous     = 0
        # ----- event log -----
        self.events = []   # list of per-frame dicts
        # ----- creation reasons -----
        self.creation_reasons = defaultdict(int)
        # "no_prior"       – brand new person
        # "lost_expired"   – lost track already removed; person created fresh
        # "assoc_failure"  – lost track existed nearby but IoU/score didn't match
        # ----- id history -----
        self.track_history  = {}  # track_id → {first_frame, last_seen, state}
        self.identity_failures = []  # list of dicts describing each failure

        # Track which IDs existed at end of previous frame for delta comparison
        self._prev_tracked_ids   = set()
        self._prev_lost_ids      = set()
        self._prev_removed_ids   = set()

    def update(self, detections_np, img_info, img_size, frame_idx):
        """
        Wraps BYTETracker.update() and captures full state snapshot.
        """
        tracker = self.tracker

        # ---- Before-state snapshot ----
        before_tracked  = {t.track_id for t in tracker.tracked_stracks}
        before_lost     = {t.track_id for t in tracker.lost_stracks}
        before_removed  = {t.track_id for t in tracker.removed_stracks}

        # Keep a tlbr snapshot of lost tracks for IoU comparison later
        lost_tlbrs = {t.track_id: t.tlbr.tolist() for t in tracker.lost_stracks}

        # ---- Official BYTETracker update ----
        online_targets = tracker.update(detections_np, img_info, img_size)

        # ---- After-state snapshot ----
        after_tracked  = {t.track_id for t in tracker.tracked_stracks}
        after_lost     = {t.track_id for t in tracker.lost_stracks}
        after_removed  = {t.track_id for t in tracker.removed_stracks}
        active_ids     = [t.track_id for t in online_targets]

        # ---- Derived events ----
        new_ids        = after_tracked - before_tracked - before_lost   # brand-new activations
        reactivated    = after_tracked & before_lost                     # lost → tracked
        newly_lost     = after_lost - before_lost
        newly_removed  = after_removed - before_removed

        # ---- Update counters ----
        if len(detections_np) == 0:
            self.frames_zero_dets += 1

        self.total_track_creations += len(new_ids)
        self.total_reactivations   += len(reactivated)
        self.total_removals        += len(newly_removed)
        self.peak_simultaneous      = max(self.peak_simultaneous, len(active_ids))

        # ---- Classify creation reasons ----
        for tid in new_ids:
            # find the new STrack object
            new_trk = next((t for t in tracker.tracked_stracks if t.track_id == tid), None)
            new_box = new_trk.tlbr.tolist() if new_trk else None

            if not lost_tlbrs:
                reason = "no_prior"
            else:
                # Check if there was a lost track nearby (IoU > 0.1)
                best_iou = 0.0
                best_lost_id = None
                if new_box:
                    for lid, lb in lost_tlbrs.items():
                        iou = box_iou(new_box, lb)
                        if iou > best_iou:
                            best_iou = iou
                            best_lost_id = lid

                if best_iou > 0.1:
                    # Lost track was nearby but association failed
                    reason = "assoc_failure"
                    self.identity_failures.append({
                        "frame": frame_idx,
                        "new_id": tid,
                        "lost_id": best_lost_id,
                        "iou": round(best_iou, 3),
                        "reason": "Lost track existed nearby (IoU={:.3f}) but was not recovered".format(best_iou),
                    })
                elif best_lost_id and best_iou <= 0.1:
                    # Lost tracks exist but none overlapped with new detection
                    reason = "lost_expired_or_moved"
                else:
                    reason = "no_prior"

            self.creation_reasons[reason] += 1

            # Classify further: was there a *removed* track that covered this area?
            if new_box:
                removed_nearby = []
                for rid in before_removed:
                    # We don't have removed boxes easily, mark as unknown
                    removed_nearby.append(rid)

        # ---- Track history update ----
        for tid in new_ids:
            self.track_history[tid] = {"first_frame": frame_idx, "last_seen": frame_idx,
                                        "state": "Tracked", "lost_at": None, "removed_at": None}
        for tid in reactivated:
            if tid in self.track_history:
                self.track_history[tid]["state"] = "Reactivated"
                self.track_history[tid]["last_seen"] = frame_idx
        for tid in newly_lost:
            if tid in self.track_history:
                self.track_history[tid]["state"] = "Lost"
                self.track_history[tid]["lost_at"] = frame_idx
        for tid in newly_removed:
            if tid in self.track_history:
                self.track_history[tid]["state"] = "Removed"
                self.track_history[tid]["removed_at"] = frame_idx

        for tid in active_ids:
            if tid in self.track_history:
                self.track_history[tid]["last_seen"] = frame_idx

        # ---- Store per-frame event ----
        frame_event = {
            "frame": frame_idx,
            "det_count": len(detections_np),
            "det_boxes": [list(map(float, d[:4])) for d in detections_np],
            "det_scores": [float(d[4]) for d in detections_np],
            "active_ids": sorted(active_ids),
            "lost_ids": sorted(after_lost),
            "removed_ids_count": len(after_removed),
            "new_ids": sorted(new_ids),
            "reactivated_ids": sorted(reactivated),
            "newly_lost_ids": sorted(newly_lost),
            "newly_removed_ids": sorted(newly_removed),
        }
        self.events.append(frame_event)

        self._prev_tracked_ids  = after_tracked
        self._prev_lost_ids     = after_lost
        self._prev_removed_ids  = after_removed

        return online_targets


# ---------------------------------------------------------------------------
# Main diagnostic routine
# ---------------------------------------------------------------------------
def run_diagnostic(video_path, conf_thresh=0.30, track_thresh=0.50,
                   track_buffer=30, match_thresh=0.80, max_frames=None,
                   log_every=5):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n{'='*70}")
    print(f"VIDEO: {os.path.basename(video_path)}")
    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps:.1f}")
    print(f"  Frames     : {total}")
    print(f"PARAMETERS:")
    print(f"  conf_thresh  = {conf_thresh}")
    print(f"  track_thresh = {track_thresh}")
    print(f"  track_buffer = {track_buffer}")
    print(f"  match_thresh = {match_thresh}")
    print(f"{'='*70}\n")

    print("Loading YOLO model...")
    yolo = YOLO("yolo26n.pt")

    BaseTrack.reset_id()
    args = BYTETrackerArgs(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        mot20=False,
    )
    diag = DiagnosticTracker(args, fps)

    # ----- Per-frame console log header -----
    print(f"{'Frame':>6} | {'Dets':>4} | {'Active IDs':<30} | {'Lost IDs':<20} | {'New':>4} | {'Reactiv':>7}")
    print("-" * 90)

    frame_idx  = 0
    all_unique = set()
    det_counts = []  # for jitter analysis
    consecutive_zero = []  # track runs of zero-det frames

    zero_run = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        # ---- YOLO person detection ----
        res = yolo(frame, conf=conf_thresh, classes=[0], verbose=False)[0]
        if len(res.boxes) > 0:
            xyxy  = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()[:, None]
            dets  = np.hstack([xyxy, confs])
        else:
            dets = np.empty((0, 5), dtype=float)

        det_counts.append(len(dets))
        if len(dets) == 0:
            zero_run += 1
        else:
            if zero_run > 0:
                consecutive_zero.append(zero_run)
            zero_run = 0

        # ---- Tracker update ----
        targets = diag.update(dets, (height, width), (height, width), frame_idx)
        active_ids = [t.track_id for t in targets]
        all_unique.update(active_ids)

        ev = diag.events[-1]

        # Console log (every N frames, or when interesting events happen)
        if frame_idx % log_every == 0 or ev["new_ids"] or ev["reactivated_ids"] or ev["newly_removed_ids"]:
            lost_str   = str(sorted(ev["lost_ids"])[:6])  # cap at 6 for display
            active_str = str(sorted(active_ids)[:8])      # cap at 8 for display
            new_str    = f"+{ev['new_ids']}" if ev["new_ids"] else ""
            re_str     = f"~{ev['reactivated_ids']}" if ev["reactivated_ids"] else ""
            print(f"{frame_idx:>6} | {len(dets):>4} | {active_str:<30} | {lost_str:<20} | {new_str:<10} | {re_str}")

    if zero_run > 0:
        consecutive_zero.append(zero_run)
    cap.release()

    # -----------------------------------------------------------------------
    # SUMMARY SECTION
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")

    total_unique = max(diag.track_history.keys()) if diag.track_history else 0
    print(f"  Total frames processed          : {frame_idx}")
    print(f"  Total UNIQUE track IDs created  : {total_unique}")
    print(f"  Peak simultaneous active tracks : {diag.peak_simultaneous}")
    print(f"  Track creations (new IDs)       : {diag.total_track_creations}")
    print(f"  Track reactivations             : {diag.total_reactivations}")
    print(f"  Track removals                  : {diag.total_removals}")
    print(f"  Frames with ZERO detections     : {diag.frames_zero_dets}")

    # Detection jitter
    if det_counts:
        arr = np.array(det_counts)
        print(f"\nYOLO DETECTION STATISTICS (person class only):")
        print(f"  Mean dets/frame : {arr.mean():.2f}")
        print(f"  Std dets/frame  : {arr.std():.2f}")
        print(f"  Min dets/frame  : {int(arr.min())}")
        print(f"  Max dets/frame  : {int(arr.max())}")
        print(f"  Frames with 0   : {int((arr == 0).sum())}")
        print(f"  Frames with <5  : {int((arr < 5).sum())}")

        # Jitter metric: count frames where |dets[t] - dets[t-1]| > 2
        jitter_frames = int(np.sum(np.abs(np.diff(arr)) > 2))
        print(f"  Detection jitter frames (|delta|>2): {jitter_frames}")

    if consecutive_zero:
        print(f"\n  Longest zero-det run            : {max(consecutive_zero)} frames")
        print(f"  Zero-det runs of ≥3 frames      : {sum(1 for r in consecutive_zero if r >= 3)}")
        print(f"  Zero-det runs of ≥track_buffer  : {sum(1 for r in consecutive_zero if r >= track_buffer)}")

    # Creation reasons breakdown
    print(f"\nTRACK CREATION REASONS:")
    reason_labels = {
        "no_prior":              "Brand-new track (no prior track at all)",
        "assoc_failure":         "Association failure (lost track nearby, IoU>0.1 but not re-matched)",
        "lost_expired_or_moved": "Lost track expired or moved far (IoU≤0.1)",
    }
    total_reasons = sum(diag.creation_reasons.values())
    for k, label in reason_labels.items():
        cnt = diag.creation_reasons.get(k, 0)
        pct = 100 * cnt / total_reasons if total_reasons > 0 else 0
        print(f"  {label:<55} : {cnt:>4}  ({pct:.0f}%)")

    # Root cause table
    print(f"\n{'='*70}")
    print("ROOT CAUSE TABLE")
    print(f"{'='*70}")
    print(f"  {'Cause':<55} {'Count':>6}")
    print(f"  {'-'*62}")

    causes = {
        "New track created (total IDs issued)":               diag.total_track_creations,
        "Track reactivated (same ID recovered)":              diag.total_reactivations,
        "Track permanently removed":                          diag.total_removals,
        "Frames with 0 YOLO detections (A)":                 diag.frames_zero_dets,
        "Assoc failure – lost track nearby not re-matched (B/D)": diag.creation_reasons.get("assoc_failure", 0),
        "Detection appeared with no prior track (E/F)":       diag.creation_reasons.get("no_prior", 0) + diag.creation_reasons.get("lost_expired_or_moved", 0),
        "Detection jitter frames |delta|>2 (E)":             jitter_frames if det_counts else 0,
    }
    for k, v in causes.items():
        print(f"  {k:<55} {v:>6}")

    # -----------------------------------------------------------------------
    # IDENTITY FAILURE EXAMPLES
    # -----------------------------------------------------------------------
    if diag.identity_failures:
        print(f"\n{'='*70}")
        print("IDENTITY FAILURE EXAMPLES (lost track → new ID)")
        print(f"{'='*70}")
        for i, fail in enumerate(diag.identity_failures[:6]):
            print(f"\n  Example {i+1}:")
            print(f"    Frame            : {fail['frame']}")
            print(f"    Old (lost) ID    : {fail['lost_id']}")
            print(f"    New ID assigned  : {fail['new_id']}")
            print(f"    IoU overlap      : {fail['iou']:.3f}")
            print(f"    Explanation      : {fail['reason']}")

            # Find when old ID was last seen
            hist = diag.track_history.get(fail["lost_id"])
            if hist:
                lost_at  = hist.get("lost_at", "?")
                removed_at = hist.get("removed_at", "still alive")
                gap = (fail["frame"] - lost_at) if isinstance(lost_at, int) else "?"
                print(f"    Old ID last seen : frame {hist.get('last_seen','?')}")
                print(f"    Old ID went Lost : frame {lost_at}")
                print(f"    Old ID removed   : frame {removed_at}")
                print(f"    Gap (lost→new)   : {gap} frames")
                print(f"    track_buffer     : {track_buffer} frames")
                if isinstance(gap, int) and isinstance(lost_at, int) and gap > track_buffer:
                    print(f"    >> CAUSE: gap ({gap}) > track_buffer ({track_buffer}). "
                          f"Old track was DELETED before person reappeared. (C)")
                elif fail["iou"] > 0:
                    print(f"    >> CAUSE: Lost track existed (IoU={fail['iou']:.3f}) but "
                          f"match_thresh={match_thresh} or fuse_score caused rejection. (B/D)")
    else:
        print(f"\n  No association failures detected with IoU>0.1.")
        print(f"  This means ALL new IDs came from tracks that were either brand-new or already removed.")
        print(f"  PRIMARY CAUSE: track_buffer too small OR YOLO dropping detections for entire persons.")

    # -----------------------------------------------------------------------
    # PER-TRACK LIFETIME SUMMARY  (show tracks with short life = suspicious)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PER-TRACK LIFETIME (short-lived tracks indicate instability)")
    print(f"{'='*70}")
    print(f"  {'ID':>5} | {'First':>6} | {'Last':>6} | {'Life':>6} | State")
    print(f"  {'-'*50}")
    for tid in sorted(diag.track_history.keys()):
        h = diag.track_history[tid]
        life = h["last_seen"] - h["first_frame"]
        state = h["state"]
        flag = " <<< SHORT" if life < fps else ""  # flag if < 1 second
        print(f"  {tid:>5} | {h['first_frame']:>6} | {h['last_seen']:>6} | {life:>6} | {state}{flag}")

    # -----------------------------------------------------------------------
    # FINAL DIAGNOSIS
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FINAL DIAGNOSIS")
    print(f"{'='*70}")

    assoc_fail  = diag.creation_reasons.get("assoc_failure", 0)
    no_prior    = diag.creation_reasons.get("no_prior", 0)
    expired_mov = diag.creation_reasons.get("lost_expired_or_moved", 0)

    if diag.frames_zero_dets > frame_idx * 0.10:
        print("  >> A: YOLO is missing people frequently (>10% zero-det frames).")
        print("        This is the primary cause — ByteTrack cannot track what YOLO doesn't detect.")
        print("        FIX: Lower conf_thresh or use a more sensitive model.")
    if assoc_fail > 5:
        print(f"  >> B/D: Association failures = {assoc_fail}.")
        print(f"          Lost tracks had nearby detections but were not re-matched.")
        print(f"          FIX: Increase match_thresh or increase track_buffer.")
    if jitter_frames > frame_idx * 0.15 if det_counts else False:
        print("  >> E: Detection jitter is HIGH (>15% of frames have large det count swings).")
        print("        YOLO alternates between detecting and missing the same person.")
        print("        FIX: Apply temporal smoothing or lower conf_thresh slightly.")
    if diag.frames_zero_dets < 5 and assoc_fail < 5 and diag.total_track_creations > 15:
        print("  >> C: track_buffer may be too small for the video FPS.")
        print(f"        At {fps:.0f}fps, track_buffer={track_buffer} = {track_buffer/fps:.1f}s patience.")
        print(f"        If people are occluded >1s, increase track_buffer to 60–90.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BYTETracker Diagnostic Tool")
    parser.add_argument("video",           type=str,   default="classroom_sample.mp4", nargs="?")
    parser.add_argument("--conf",          type=float, default=0.30)
    parser.add_argument("--track_thresh",  type=float, default=0.50)
    parser.add_argument("--track_buffer",  type=int,   default=30)
    parser.add_argument("--match_thresh",  type=float, default=0.80)
    parser.add_argument("--max_frames",    type=int,   default=None)
    parser.add_argument("--log_every",     type=int,   default=30)
    args = parser.parse_args()

    run_diagnostic(
        video_path=args.video,
        conf_thresh=args.conf,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        max_frames=args.max_frames,
        log_every=args.log_every,
    )
