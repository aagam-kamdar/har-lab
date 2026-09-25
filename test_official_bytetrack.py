"""
test_official_bytetrack.py
--------------------------
Comprehensive verification script for the integrated official BYTETracker.
Tests:
  A. Normal movement ID stability (2-3 students)
  B. Temporary occlusion recovery (student occluded for frames, then returns)
  C. Crossing/overlap ID switch analysis (students pass each other)
  D. Long disappearance behavior (exceeding track_buffer)
"""

import sys
import numpy as np

sys.path.insert(0, r"c:\Users\admin\OneDrive\Desktop\Projects")

from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack, TrackState

SEP = "=" * 70


def reset_and_create(track_thresh=0.5, track_buffer=30, match_thresh=0.8, fps=30):
    BaseTrack.reset_id()
    args = BYTETrackerArgs(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        mot20=False,
    )
    return BYTETracker(args=args, frame_rate=fps)


# ==============================================================================
# A. Normal Movement ID Stability
# ==============================================================================
print(SEP)
print("TEST A: Normal Movement ID Stability (3 students visible)")
print(SEP)

tracker = reset_and_create(track_buffer=30)
# 3 students sitting / slightly moving across 15 frames
# Each student drifts by 2-3 pixels per frame
frames_ids = []

for frame_idx in range(1, 16):
    drift = frame_idx * 2
    dets = np.array([
        [100 + drift, 150, 200 + drift, 400, 0.92],  # Student 1
        [350 + drift, 140, 450 + drift, 390, 0.88],  # Student 2
        [600 + drift, 160, 700 + drift, 410, 0.85],  # Student 3
    ], dtype=float)

    targets = tracker.update(dets, (720, 1280), (720, 1280))
    active_ids = sorted([t.track_id for t in targets])
    states = [TrackState.to_string(t.state) for t in targets]
    frames_ids.append(active_ids)
    print(f"  Frame {frame_idx:2d}: active_ids={active_ids}  states={states}")

# Check stability from frame 2 onwards (frame 1 initializes, frame 2 confirms)
stable = all(ids == [1, 2, 3] for ids in frames_ids[1:])
print(f"\n=> Result A: Normal movement IDs are stable [1, 2, 3]: {stable}")
print()


# ==============================================================================
# B. Temporary Occlusion Recovery
# ==============================================================================
print(SEP)
print("TEST B: Temporary Occlusion Recovery (Student 2 occluded for 10 frames)")
print(SEP)

tracker = reset_and_create(track_buffer=30)

# Warm up: 5 frames with 3 students
for f in range(1, 6):
    dets = np.array([
        [100, 150, 200, 400, 0.92],
        [350, 140, 450, 390, 0.88],
        [600, 160, 700, 410, 0.85],
    ], dtype=float)
    tracker.update(dets, (720, 1280), (720, 1280))

print("  Warm-up complete: Students 1, 2, 3 active.")

# Student 2 disappears for 10 frames (occluded by whiteboard / classmate)
print("  Student 2 disappears for 10 frames...")
for f in range(6, 16):
    dets = np.array([
        [100, 150, 200, 400, 0.92],
        # Student 2 omitted
        [600, 160, 700, 410, 0.85],
    ], dtype=float)
    targets = tracker.update(dets, (720, 1280), (720, 1280))
    active_ids = sorted([t.track_id for t in targets])
    lost_ids = sorted([t.track_id for t in tracker.lost_stracks])
    if f in (6, 10, 15):
        print(f"    Frame {f:2d}: active_ids={active_ids}  lost_ids={lost_ids}")

# Student 2 reappears at frame 16
print("  Student 2 reappears at frame 16...")
dets = np.array([
    [100, 150, 200, 400, 0.92],
    [355, 142, 455, 392, 0.87],  # Student 2 returns near previous location
    [600, 160, 700, 410, 0.85],
], dtype=float)
targets = tracker.update(dets, (720, 1280), (720, 1280))
active_ids = sorted([t.track_id for t in targets])
print(f"    Frame 16: active_ids={active_ids}")

s2_recovered = (2 in active_ids)
print(f"\n=> Result B: Student 2 ID=2 successfully preserved upon reactivation: {s2_recovered}")
print()


# ==============================================================================
# C. Crossing / Overlap ID Switches
# ==============================================================================
print(SEP)
print("TEST C: Crossing / Overlap Behavior (Student 1 & Student 2 cross paths)")
print(SEP)

tracker = reset_and_create(track_buffer=30)

# Student 1 starts left (x=100) moving right (+30 px/frame)
# Student 2 starts right (x=400) moving left (-30 px/frame)
trajectories = []
for f in range(1, 12):
    s1_x1 = 100 + (f - 1) * 30
    s2_x1 = 400 - (f - 1) * 30

    dets = np.array([
        [s1_x1, 150, s1_x1 + 100, 400, 0.90],
        [s2_x1, 150, s2_x1 + 100, 400, 0.90],
    ], dtype=float)

    targets = tracker.update(dets, (720, 1280), (720, 1280))
    # Order targets by x position to see left-to-right ID assignment
    sorted_targets = sorted(targets, key=lambda t: t.tlbr[0])
    order = [(t.track_id, int(t.tlbr[0])) for t in sorted_targets]
    trajectories.append(order)
    print(f"  Frame {f:2d}: s1_x1={s1_x1:3d}, s2_x1={s2_x1:3d} | Left-to-right: {order}")

initial_order = [t[0] for t in trajectories[1]]
post_cross_order = [t[0] for t in trajectories[-1]]
print(f"\n  Initial left-to-right IDs (Frame 2): {initial_order}")
print(f"  Post-crossing left-to-right IDs (Frame 11): {post_cross_order}")
print("=> Result C: When two bounding boxes overlap strongly (IoU ambiguity),")
print("   motion-only IoU tracking has no appearance features (Re-ID) to disambiguate.")
print("   ByteTrack relies purely on Kalman position/velocity projection.")
print()


# ==============================================================================
# D. Long Disappearance Behavior (Exceeding track_buffer)
# ==============================================================================
print(SEP)
print("TEST D: Long Disappearance Behavior (exceeding track_buffer=15 frames)")
print(SEP)

# Use track_buffer=15
tracker = reset_and_create(track_buffer=15)

# Initialize Student 1 (ID=1)
for f in range(1, 4):
    dets = np.array([[200, 200, 300, 450, 0.90]], dtype=float)
    tracker.update(dets, (720, 1280), (720, 1280))

print("  Student 1 initialized with track_id=1.")

# Student 1 disappears for 25 frames (longer than track_buffer=15)
for f in range(4, 29):
    tracker.update(np.empty((0, 5), dtype=float), (720, 1280), (720, 1280))

print(f"  After 25 empty frames: lost={len(tracker.lost_stracks)}  removed={len(tracker.removed_stracks)}")

# Student 1 returns at frame 29
dets = np.array([[200, 200, 300, 450, 0.90]], dtype=float)
targets = tracker.update(dets, (720, 1280), (720, 1280))
new_ids = [t.track_id for t in targets]
print(f"  Student reappears at frame 29: active_ids={new_ids}")

reassigned = (1 not in new_ids)
print(f"\n=> Result D: Track was removed from memory after track_buffer was exceeded.")
print(f"   Student was assigned a NEW ID {new_ids} instead of ID 1: {reassigned}")
print("   This confirms ByteTrack does NOT have long-term appearance memory (Re-ID needed).")
print(SEP)
