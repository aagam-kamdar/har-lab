#!/usr/bin/env python3
"""
merge_videos.py
Sequential merger for lab monitoring source videos:
  1. new lab video.mp4 (200.17s)
  2. talking.mp4       (113.84s)
  3. writi.mp4         (249.47s)
Target output: new video.mp4 (1120x480 @ 30.0 fps, H.264 / AAC)
"""

import os
import sys
import subprocess
import cv2

FFMPEG_PATH = r"C:\Users\admin\anaconda3\envs\opencv_env\Library\bin\ffmpeg.exe"

CLIPS = [
    "new lab video.mp4",
    "talking.mp4",
    "writi.mp4"
]
OUTPUT = "new video.mp4"

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = cnt / fps if fps > 0 else 0
    cap.release()
    return {"width": w, "height": h, "fps": fps, "frames": cnt, "duration": dur}

def merge():
    print(f"=== Merging {len(CLIPS)} source videos into '{OUTPUT}' ===")
    total_expected_dur = 0.0
    for idx, clip in enumerate(CLIPS, 1):
        if not os.path.exists(clip):
            print(f"Error: {clip} not found!")
            sys.exit(1)
        info = get_video_info(clip)
        print(f"[{idx}] {clip}: {info['width']}x{info['height']} @ {info['fps']:.2f}fps, {info['frames']} frames, {info['duration']:.2f}s")
        total_expected_dur += info['duration']
    
    print(f"Total expected duration: {total_expected_dur:.2f}s (~{total_expected_dur/60:.2f} min)")
    
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", CLIPS[0],
        "-i", CLIPS[1],
        "-i", CLIPS[2],
        "-filter_complex",
        "[0:v]fps=30,scale=1120:480,setsar=1[v0]; "
        "[1:v]fps=30,scale=1120:480,setsar=1[v1]; "
        "[2:v]fps=30,scale=1120:480,setsar=1[v2]; "
        "[0:a]aformat=sample_rates=48000:channel_layouts=stereo[a0]; "
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo[a1]; "
        "[2:a]aformat=sample_rates=48000:channel_layouts=stereo[a2]; "
        "[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[v][a]",
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "h264_mf",
        "-b:v", "3500k",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        OUTPUT
    ]
    
    print("Running FFmpeg merge...")
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"FFmpeg failed with return code {ret.returncode}:")
        print(ret.stderr[-1000:])
        sys.exit(1)
        
    print("Merge complete! Verifying output...")
    out_info = get_video_info(OUTPUT)
    if not out_info:
        print(f"Error: failed to open generated {OUTPUT}")
        sys.exit(1)
        
    print(f"Result: {OUTPUT}")
    print(f"  Resolution: {out_info['width']}x{out_info['height']}")
    print(f"  FPS:        {out_info['fps']:.2f}")
    print(f"  Frames:     {out_info['frames']}")
    print(f"  Duration:   {out_info['duration']:.2f}s")
    print(f"  Size:       {os.path.getsize(OUTPUT) / (1024*1024):.2f} MB")
    
    diff = abs(out_info['duration'] - total_expected_dur)
    print(f"  Duration delta vs source sum: {diff:.2f}s")
    assert diff < 2.0, f"Duration discrepancy too large: {diff:.2f}s"
    print("SUCCESS: new video.mp4 verified successfully!")

if __name__ == "__main__":
    merge()
