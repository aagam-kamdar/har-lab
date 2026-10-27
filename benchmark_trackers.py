"""
benchmark_trackers.py
---------------------
Controlled 3-Way Benchmark: ByteTrack vs. OC-SORT vs. BoT-SORT on 'new lab video.mp4' (1800 frames).

Identical conditions:
  - Exact same YOLO person detections fed to all three trackers on each frame.
  - Identity managers, ArcFace, body Re-ID, FAISS are completely excluded.
  - Output videos show only Track ID + bounding box.
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
from types import SimpleNamespace
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.oc_sort import OCSORT
from ultralytics.trackers.bot_sort import BOTSORT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack as BTBaseTrack

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrackerBenchmark")


# --------------------------------------------------------------------------- #
#  Video Writer                                                               #
# --------------------------------------------------------------------------- #

class VideoWriter:
    def __init__(self, path: str, fps: float, width: int, height: int):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.mode = "cv2"
        self.writer = None

        try:
            import av
            self.mode = "av"
            self.container = av.open(path, mode="w")
            self.stream = self.container.add_stream("h264", rate=int(round(fps)))
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"preset": "veryfast", "crf": "22"}
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        if self.mode == "av":
            import av
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in self.stream.encode(av_frame):
                self.container.mux(pkt)
        else:
            self.writer.write(frame)

    def close(self):
        if self.mode == "av":
            for pkt in self.stream.encode():
                self.container.mux(pkt)
            self.container.close()
        elif self.writer:
            self.writer.release()


# --------------------------------------------------------------------------- #
#  Drawing helper                                                             #
# --------------------------------------------------------------------------- #

def get_color(track_id: int) -> Tuple[int, int, int]:
    np.random.seed(int(track_id) * 37 + 11)
    color = np.random.randint(50, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))


def draw_boxes(frame: np.ndarray, tracks: List[Tuple[int, List[float]]], tracker_name: str, frame_idx: int) -> np.ndarray:
    annotated = frame.copy()
    for tid, bbox in tracks:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = get_color(tid)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID: {tid}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bg_y1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
        )

    # Header banner
    cv2.putText(
        annotated, f"{tracker_name} | Frame {frame_idx} | Active Tracks: {len(tracks)}",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    return annotated


# --------------------------------------------------------------------------- #
#  Tracker Statistics Monitor                                                 #
# --------------------------------------------------------------------------- #

class TrackerStatsMonitor:
    def __init__(self, name: str):
        self.name = name
        self.all_track_ids = set()
        self.active_tracks_per_frame = []
        self.frame_to_tracks: Dict[int, List[Tuple[int, List[float]]]] = {}
        self.track_first_seen: Dict[int, int] = {}
        self.track_last_seen: Dict[int, int] = {}
        self.track_frames_count: Dict[int, int] = defaultdict(int)
        
        # State transitions
        self.recoveries_count = 0  # LOST -> TRACKED
        self.removed_count = 0     # Permanently removed tracks
        self.lost_durations = []   # Duration of each lost spell before recovery
        self.track_lost_spells: Dict[int, int] = {} # track_id -> frame last seen before becoming lost
        
        # Timing
        self.total_track_time = 0.0

    def record_frame(
        self,
        frame_idx: int,
        tracked_tracks: List[Tuple[int, List[float]]],
        lost_ids: List[int],
        removed_ids: List[int],
        step_time: float,
    ):
        self.total_track_time += step_time
        active_ids = {tid for tid, _ in tracked_tracks}
        self.active_tracks_per_frame.append(len(active_ids))
        self.frame_to_tracks[frame_idx] = tracked_tracks

        for tid, _ in tracked_tracks:
            if tid not in self.all_track_ids:
                self.all_track_ids.add(tid)
                self.track_first_seen[tid] = frame_idx
            
            # Check if this track was previously lost and is now recovered
            if tid in self.track_lost_spells:
                lost_start = self.track_lost_spells.pop(tid)
                duration = frame_idx - lost_start
                self.lost_durations.append(duration)
                self.recoveries_count += 1
            
            self.track_last_seen[tid] = frame_idx
            self.track_frames_count[tid] += 1

        for tid in lost_ids:
            if tid not in self.track_lost_spells and tid in self.all_track_ids:
                self.track_lost_spells[tid] = self.track_last_seen.get(tid, frame_idx)

        for tid in removed_ids:
            self.removed_count += 1
            self.track_lost_spells.pop(tid, None)

    def compute_metrics(self, total_frames: int, fps: float) -> Dict[str, Any]:
        durations = [self.track_last_seen[tid] - self.track_first_seen[tid] + 1 for tid in self.all_track_ids]
        
        short_tracks = sum(1 for d in durations if d < 30)
        long_tracks = sum(1 for d in durations if d >= 150)

        avg_fps = total_frames / max(self.total_track_time, 1e-6)

        return {
            "tracker": self.name,
            "total_frames_processed": total_frames,
            "total_unique_track_ids": len(self.all_track_ids),
            "peak_simultaneous_tracks": max(self.active_tracks_per_frame) if self.active_tracks_per_frame else 0,
            "average_active_tracks_per_frame": round(float(np.mean(self.active_tracks_per_frame)), 2) if self.active_tracks_per_frame else 0,
            "max_continuous_track_duration_frames": int(max(durations)) if durations else 0,
            "max_continuous_track_duration_seconds": round(float(max(durations)) / fps, 2) if durations else 0,
            "median_track_duration_frames": float(np.median(durations)) if durations else 0,
            "mean_track_duration_frames": round(float(np.mean(durations)), 1) if durations else 0,
            "short_tracks_count_lt_1s": short_tracks,
            "long_tracks_count_gte_5s": long_tracks,
            "lost_to_tracked_recoveries": self.recoveries_count,
            "average_lost_recovery_duration_frames": round(float(np.mean(self.lost_durations)), 1) if self.lost_durations else 0,
            "max_lost_recovery_duration_frames": int(max(self.lost_durations)) if self.lost_durations else 0,
            "removed_tracks_count": self.removed_count,
            "tracking_only_cpu_fps": round(avg_fps, 1),
            "tracking_time_total_seconds": round(self.total_track_time, 3),
        }


# --------------------------------------------------------------------------- #
#  Benchmark Runner                                                           #
# --------------------------------------------------------------------------- #

def run_benchmark(
    video_path: str = "new lab video.mp4",
    max_frames: int = 1800,
    output_dir: str = "test_outputs",
):
    logger.info("==================================================================")
    logger.info("STARTING 3-WAY BENCHMARK: ByteTrack vs OC-SORT vs BoT-SORT (%d frames)", max_frames)
    logger.info("==================================================================")

    # 1. Load YOLO model
    yolo = YOLO("yolo26n.pt")

    # 2. Setup ByteTrack (current pipeline configuration)
    BTBaseTrack.reset_id()
    bt_args = BYTETrackerArgs()
    bt_args.track_thresh = 0.35
    bt_args.track_buffer = 90
    bt_args.match_thresh = 0.85
    bt_args.mot20 = False
    bytetrack = BYTETracker(bt_args, frame_rate=30)
    bt_monitor = TrackerStatsMonitor("ByteTrack")

    # 3. Setup OC-SORT
    ocsort_args = SimpleNamespace(
        track_high_thresh=0.35,
        track_low_thresh=0.1,
        new_track_thresh=0.35,
        track_buffer=90,
        match_thresh=0.85,
        fuse_score=True,
        delta_t=3,
        inertia=0.2,
        use_byte=True,
        mot20=False
    )
    ocsort = OCSORT(ocsort_args)
    oc_monitor = TrackerStatsMonitor("OC-SORT")

    # 4. Setup BoT-SORT (matched thresholds, gmc_method=sparseOptFlow / with_reid=False for pure tracking)
    botsort_args = SimpleNamespace(
        track_high_thresh=0.35,
        track_low_thresh=0.1,
        new_track_thresh=0.35,
        track_buffer=90,
        match_thresh=0.85,
        fuse_score=True,
        gmc_method="none", # static camera in lab scene
        proximity_thresh=0.5,
        appearance_thresh=0.8,
        with_reid=False,
        model="auto",
        mot20=False
    )
    botsort = BOTSORT(botsort_args)
    bot_monitor = TrackerStatsMonitor("BoT-SORT")

    # 5. Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_in_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_limit = min(max_frames, total_frames_in_vid)

    logger.info("Input video: %dx%d @ %.1f FPS (%d total frames)", vid_w, vid_h, fps, total_frames_in_vid)

    # 6. Writers
    os.makedirs(output_dir, exist_ok=True)
    bt_video_path = os.path.join(output_dir, "bytetrack_600f.mp4")
    oc_video_path = os.path.join(output_dir, "ocsort_600f.mp4")
    bot_video_path = os.path.join(output_dir, "botsort_600f.mp4")
    bt_writer = VideoWriter(bt_video_path, fps, vid_w, vid_h)
    oc_writer = VideoWriter(oc_video_path, fps, vid_w, vid_h)
    bot_writer = VideoWriter(bot_video_path, fps, vid_w, vid_h)

    scene_phase_stats = {
        "Phase_1_Standing_Turn_0_300": {"bytetrack_ids": set(), "ocsort_ids": set(), "botsort_ids": set()},
        "Phase_2_Bending_Sitting_300_600": {"bytetrack_ids": set(), "ocsort_ids": set(), "botsort_ids": set()},
        "Phase_3_Desk_Occlusion_600_1800": {"bytetrack_ids": set(), "ocsort_ids": set(), "botsort_ids": set()},
    }

    t0 = time.time()
    for frame_idx in range(process_limit):
        ret, frame = cap.read()
        if not ret:
            break

        # --- Detection: run YOLO once ---
        yolo_res = yolo(frame, classes=[0], verbose=False)[0]
        dets_xyxy_conf = []
        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            for box in yolo_res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                dets_xyxy_conf.append([x1, y1, x2, y2, conf])

        # --- A) Update ByteTrack ---
        t_bt0 = time.perf_counter()
        if dets_xyxy_conf:
            det_arr = np.array(dets_xyxy_conf, dtype=np.float32)
        else:
            det_arr = np.empty((0, 5), dtype=np.float32)

        bt_online = bytetrack.update(det_arr, [vid_h, vid_w], [vid_h, vid_w])
        t_bt_elapsed = time.perf_counter() - t_bt0

        bt_tracks = []
        for t in bt_online:
            tlwh = t.tlwh
            x1 = tlwh[0]
            y1 = tlwh[1]
            x2 = tlwh[0] + tlwh[2]
            y2 = tlwh[1] + tlwh[3]
            bt_tracks.append((int(t.track_id), [x1, y1, x2, y2]))

        bt_lost_ids = [int(t.track_id) for t in bytetrack.lost_stracks]
        bt_removed_ids = [int(t.track_id) for t in bytetrack.removed_stracks]
        bt_monitor.record_frame(frame_idx, bt_tracks, bt_lost_ids, bt_removed_ids, t_bt_elapsed)

        # --- B) Update OC-SORT ---
        t_oc0 = time.perf_counter()
        oc_out = ocsort.update(yolo_res.boxes)
        t_oc_elapsed = time.perf_counter() - t_oc0

        oc_tracks = []
        if oc_out is not None and len(oc_out) > 0:
            for row in oc_out:
                x1, y1, x2, y2, tid = row[:5]
                oc_tracks.append((int(tid), [float(x1), float(y1), float(x2), float(y2)]))

        oc_lost_ids = [int(t.track_id) for t in ocsort.lost_stracks]
        oc_removed_ids = [int(t.track_id) for t in ocsort.removed_stracks]
        oc_monitor.record_frame(frame_idx, oc_tracks, oc_lost_ids, oc_removed_ids, t_oc_elapsed)

        # --- C) Update BoT-SORT ---
        t_bot0 = time.perf_counter()
        bot_out = botsort.update(yolo_res.boxes, img=frame)
        t_bot_elapsed = time.perf_counter() - t_bot0

        bot_tracks = []
        if bot_out is not None and len(bot_out) > 0:
            for row in bot_out:
                x1, y1, x2, y2, tid = row[:5]
                bot_tracks.append((int(tid), [float(x1), float(y1), float(x2), float(y2)]))

        bot_lost_ids = [int(t.track_id) for t in botsort.lost_stracks]
        bot_removed_ids = [int(t.track_id) for t in botsort.removed_stracks]
        bot_monitor.record_frame(frame_idx, bot_tracks, bot_lost_ids, bot_removed_ids, t_bot_elapsed)

        # Phase tracking
        phase = "Phase_1_Standing_Turn_0_300" if frame_idx < 300 else ("Phase_2_Bending_Sitting_300_600" if frame_idx < 600 else "Phase_3_Desk_Occlusion_600_1800")
        for tid, _ in bt_tracks:
            scene_phase_stats[phase]["bytetrack_ids"].add(tid)
        for tid, _ in oc_tracks:
            scene_phase_stats[phase]["ocsort_ids"].add(tid)
        for tid, _ in bot_tracks:
            scene_phase_stats[phase]["botsort_ids"].add(tid)

        # --- Render Videos ---
        bt_frame_annotated = draw_boxes(frame, bt_tracks, "ByteTrack", frame_idx)
        oc_frame_annotated = draw_boxes(frame, oc_tracks, "OC-SORT", frame_idx)
        bot_frame_annotated = draw_boxes(frame, bot_tracks, "BoT-SORT", frame_idx)
        
        bt_writer.write(bt_frame_annotated)
        oc_writer.write(oc_frame_annotated)
        bot_writer.write(bot_frame_annotated)

        if frame_idx % 300 == 0 or frame_idx == process_limit - 1:
            logger.info(
                "Frame %d/%d (%.1f%%) | ByteTrack: %d (total: %d) | OC-SORT: %d (total: %d) | BoT-SORT: %d (total: %d)",
                frame_idx + 1, process_limit, 100.0 * (frame_idx + 1) / process_limit,
                len(bt_tracks), len(bt_monitor.all_track_ids),
                len(oc_tracks), len(oc_monitor.all_track_ids),
                len(bot_tracks), len(bot_monitor.all_track_ids),
            )

    cap.release()
    bt_writer.close()
    oc_writer.close()
    bot_writer.close()

    total_time = time.time() - t0
    logger.info("Processing finished in %.2f seconds (overall pipeline FPS: %.1f)", total_time, process_limit / total_time)

    # Compute metrics
    bt_metrics = bt_monitor.compute_metrics(process_limit, fps)
    oc_metrics = oc_monitor.compute_metrics(process_limit, fps)
    bot_metrics = bot_monitor.compute_metrics(process_limit, fps)

    phase_summary = {}
    for p, data in scene_phase_stats.items():
        phase_summary[p] = {
            "bytetrack_unique_ids": len(data["bytetrack_ids"]),
            "ocsort_unique_ids": len(data["ocsort_ids"]),
            "botsort_unique_ids": len(data["botsort_ids"]),
        }

    report = {
        "video": video_path,
        "frames_evaluated": process_limit,
        "fps": fps,
        "duration_evaluated_seconds": round(process_limit / fps, 2),
        "bytetrack_metrics": bt_metrics,
        "ocsort_metrics": oc_metrics,
        "botsort_metrics": bot_metrics,
        "phase_breakdown": phase_summary,
        "rankings_by_total_ids": sorted(
            [
                {"tracker": "ByteTrack", "total_unique_ids": bt_metrics["total_unique_track_ids"]},
                {"tracker": "OC-SORT", "total_unique_ids": oc_metrics["total_unique_track_ids"]},
                {"tracker": "BoT-SORT", "total_unique_ids": bot_metrics["total_unique_track_ids"]},
            ],
            key=lambda x: x["total_unique_ids"]
        )
    }

    report_path = os.path.join(output_dir, "tracker_benchmark_3way_1800f_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved 3-way benchmark report to %s", report_path)
    return report


if __name__ == "__main__":
    run_benchmark(
        video_path="new lab video.mp4",
        max_frames=1800,
        output_dir="test_outputs",
    )
