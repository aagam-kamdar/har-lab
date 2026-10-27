"""
run_reid_pipeline_test.py
-------------------------
Runs the complete integrated Video -> Local YOLO -> Official ByteTrack -> Re-ID -> FAISS
pipeline on the actual test_video.mp4.
Computes and prints all metrics from Requirement 13, recovery table from Requirement 14,
and outputs the final annotated MP4 video with "Track X | Student Y" banners.
"""

import os
import sys
import logging
from video_processor import process_video_local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    video_path = "new lab video.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    output_dir = os.path.join("static", "results", "reid_verified_run")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("STARTING FULL RE-ID INTEGRATED PIPELINE ON test_video.mp4")
    print("Video -> Local YOLO (person only) -> Official ByteTrack -> FAISS Re-ID -> output_tracked.mp4")
    print("=" * 80)

    result = process_video_local(
        video_path=video_path,
        output_dir=output_dir,
        conf_thresh=0.30,
        track_thresh=0.50,
        track_buffer=30,
        match_thresh=0.80,
        similarity_thresh=0.80,
        ambiguity_margin=0.04,
        min_observations=3,
        output_filename="output_tracked.mp4",
    )

    m = result.reid_metrics

    SEP = "=" * 80
    print("\n" + SEP)
    print("FINAL RE-ID INTEGRATION METRICS REPORT (Requirement 13)")
    print(SEP)
    print(f"Total Video Frames                       : {result.total_frames}")
    print(f"Total YOLO Person Detections             : {m.get('total_person_dets', 'N/A')}")
    print(f"Average Person Detections/Frame          : {m.get('avg_dets_per_frame', 'N/A')}")
    print(f"Total BYTETracker Track IDs (Before)     : {len(result.unique_track_ids)}")
    print(f"Total Persistent Student IDs (After)     : {len(result.persistent_student_ids)}")
    print(f"Number of Re-ID Inference Attempts       : {m.get('total_reid_attempts', 0)}")
    print(f"Number of Successful Re-ID Matches       : {m.get('successful_reid_matches', 0)}")
    print(f"Number of Unknown / Unconfirmed Tracks   : {m.get('unknown_identities', 0)}")
    print(f"Number of Short-Term ByteTrack Reactiv.  : {m.get('short_term_reactivations', 0)}")
    print(f"Number of Long-Term Identity Recoveries  : {m.get('long_term_recoveries', 0)}")
    print(f"Number of Unresolved Track Fragments     : {m.get('unresolved_track_fragments', 0)}")

    print("\n" + "-" * 80)
    print(f"KEY IMPROVEMENT MEASUREMENT:")
    print(f"  Before Re-ID : {len(result.unique_track_ids)} fragmented track IDs")
    print(f"  After Re-ID  : {len(result.persistent_student_ids)} persistent student identities ({', '.join(result.persistent_student_ids)})")
    print("-" * 80)

    # Requirement 14: Recovery log
    print("\n" + SEP)
    print("RECOVERED IDENTITY EVENT LOG (Requirement 14)")
    print(SEP)
    print(f"{'Old Track':>9} | {'New Track':>9} | {'Student ID':<12} | {'Similarity':>10} | {'Gap (Frames)':>12} | {'Gap (Sec)':>10} | {'Result':<10}")
    print("-" * 85)
    for ev in result.recovery_log:
        print(f"Track {ev['old_track']:4d} | Track {ev['new_track']:4d} | {ev['student_id']:<12} | {ev['similarity']:10.3f} | {ev['gap_frames']:12d} | {ev['gap_sec']:9.2f}s | {ev['result']:<10}")

    print("\n" + SEP)
    print(f"Annotated Output Video generated: {result.output_video_path}")
    print(f"File size: {os.path.getsize(result.output_video_path):,} bytes")
    print(SEP)


if __name__ == "__main__":
    main()
