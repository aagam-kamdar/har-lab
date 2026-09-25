"""
tracker_diagnostic.py
Comprehensive diagnostic tests for the ByteTrack implementation after Kalman F-matrix fix.
Run with: python tracker_diagnostic.py
"""

import sys
import numpy as np
sys.path.insert(0, r'c:\Users\admin\OneDrive\Desktop\Projects')

from byte_tracker import ByteTracker, KalmanBoxTracker, _iou_batch


def make_tracker():
    KalmanBoxTracker._id_counter = 0
    return ByteTracker(
        high_thresh=0.6,
        low_thresh=0.1,
        max_age=30,
        min_hits=2,
        iou_threshold=0.3,
    )


SEP = "=" * 60


# ------------------------------------------------------------------ #
# TEST 1a  –  Stationary people (most common lab scenario)
# ------------------------------------------------------------------ #
print(SEP)
print("TEST 1a: Stationary people (1fps sampling, 30fps native)")
print("         max_age=30 means a track survives all 29 inter-sample frames")
print(SEP)

tracker = make_tracker()

DETS = [
    {"bbox": [50.0,  100.0, 150.0, 300.0], "conf": 0.91, "class_name": "person"},
    {"bbox": [300.0, 100.0, 400.0, 300.0], "conf": 0.87, "class_name": "person"},
    {"bbox": [550.0, 100.0, 650.0, 300.0], "conf": 0.83, "class_name": "person"},
]

for sample_i in range(8):
    for _ in range(29):   # 29 non-sampled between each 1fps sample
        tracker.update([])
    tracked = tracker.update(DETS)
    ids = sorted(obj.track_id for obj in tracked)
    print(f"  sample {sample_i}  confirmed_ids={ids}")

print()


# ------------------------------------------------------------------ #
# TEST 1b  –  Slow movement  (student shuffling in seat, ~2px/frame)
# ------------------------------------------------------------------ #
print(SEP)
print("TEST 1b: Slow movement (2px/native-frame, 10 non-sampled frames)")
print(SEP)

tracker = make_tracker()
total_native = 0

for sample_i in range(8):
    for _ in range(10):
        tracker.update([])
        total_native += 1
    d = total_native * 2   # cumulative pixel drift
    dets = [
        {"bbox": [50.0  + d, 100.0, 150.0 + d, 300.0], "conf": 0.91, "class_name": "person"},
        {"bbox": [300.0 + d, 100.0, 400.0 + d, 300.0], "conf": 0.87, "class_name": "person"},
        {"bbox": [550.0 + d, 100.0, 650.0 + d, 300.0], "conf": 0.83, "class_name": "person"},
    ]
    tracked = tracker.update(dets)
    total_native += 1
    ids = sorted(obj.track_id for obj in tracked)
    print(f"  sample {sample_i}  drift={d}px  confirmed_ids={ids}")

print()


# ------------------------------------------------------------------ #
# TEST 2  –  Temporary occlusion
# Count exactly how many frames before track disappears from output,
# and whether the same ID is restored when the person returns.
# ------------------------------------------------------------------ #
print(SEP)
print("TEST 2: Temporary occlusion – person 2 hidden, then returns")
print(SEP)

tracker = make_tracker()

p1 = {"bbox": [50.0,  100.0, 150.0, 300.0], "conf": 0.91, "class_name": "person"}
p2 = {"bbox": [300.0, 100.0, 400.0, 300.0], "conf": 0.87, "class_name": "person"}
p3 = {"bbox": [550.0, 100.0, 650.0, 300.0], "conf": 0.83, "class_name": "person"}

# Warmup: confirm all 3
for _ in range(4):
    tracker.update([])
    tracker.update([p1, p2, p3])

baseline = tracker.update([p1, p2, p3])
confirmed_ids = sorted(obj.track_id for obj in baseline)
p2_id = confirmed_ids[1]
print(f"  Baseline confirmed IDs: {confirmed_ids}   person-2 track_id={p2_id}")
print()

# Now p2 disappears (only p1, p3 detected)
disappeared_at = None
trk_state_by_frame = []

for occ in range(1, 36):
    tracker.update([])          # non-sampled frame
    result = tracker.update([p1, p3])
    visible = sorted(obj.track_id for obj in result)

    # Inspect raw tracker state for p2's track
    p2_trk = next((t for t in tracker.trackers if t.track_id == p2_id), None)
    tsu = p2_trk.time_since_update if p2_trk else "DEAD"
    trk_state_by_frame.append((occ, visible, tsu))

    if occ <= 6 or occ % 5 == 0:
        alive_str = "in_output" if p2_id in visible else f"hidden(tsu={tsu})"
        print(f"  occlusion frame {occ:2d}  visible_ids={visible}  p2={alive_str}")

    if p2_trk is None:
        disappeared_at = occ
        print(f"  ** Track {p2_id} DELETED at occlusion frame {occ} **")
        break

print()

# Return p2 at different gap lengths and check if ID is preserved
for return_after in [1, 5, 10, 25, 35]:
    tracker2 = make_tracker()
    for _ in range(4):
        tracker2.update([])
        tracker2.update([p1, p2, p3])
    bl = tracker2.update([p1, p2, p3])
    original_ids = sorted(obj.track_id for obj in bl)
    orig_p2_id = original_ids[1]

    # Hide p2 for return_after frames
    for i in range(return_after):
        tracker2.update([])
        tracker2.update([p1, p3])

    # Return p2
    tracker2.update([])
    result = tracker2.update([p1, p2, p3])
    returned_ids = sorted(obj.track_id for obj in result)
    p2_new_id = returned_ids[1] if len(returned_ids) >= 2 else "?"
    same = (p2_new_id == orig_p2_id)
    print(f"  Return after {return_after:2d} occ-frames: orig_id={orig_p2_id}  returned_id={p2_new_id}  same={same}")

print()


# ------------------------------------------------------------------ #
# TEST 3  –  ID swap when two people cross paths
# ------------------------------------------------------------------ #
print(SEP)
print("TEST 3: ID swap test – two people crossing paths")
print("        (expected: ByteTrack MAY swap IDs at the crossing point)")
print(SEP)

tracker = make_tracker()

# Person A starts left, Person B starts right; they move toward each other
# At the crossing frame the boxes maximally overlap
PA_START = [50.0, 100.0, 150.0, 300.0]   # leftmost, moving right
PB_START = [450.0, 100.0, 550.0, 300.0]  # rightmost, moving left

SPEED = 40   # px per sample step

id_log = []

for sample_i in range(8):
    tracker.update([])  # non-sampled
    drift = sample_i * SPEED
    # A moves right, B moves left
    ax1, ay1, ax2, ay2 = PA_START[0]+drift, PA_START[1], PA_START[2]+drift, PA_START[3]
    bx1, by1, bx2, by2 = PB_START[0]-drift, PB_START[1], PB_START[2]-drift, PB_START[3]

    dets = [
        {"bbox": [ax1, ay1, ax2, ay2], "conf": 0.90, "class_name": "person"},
        {"bbox": [bx1, by1, bx2, by2], "conf": 0.88, "class_name": "person"},
    ]

    # Skip if boxes would be inverted (fully crossed)
    if ax1 < 0 or bx1 < 0 or ax2 > bx2:
        tracked = tracker.update([])
    else:
        tracked = tracker.update(dets)

    sorted_tracked = sorted(tracked, key=lambda o: o.bbox[0])  # sort by x
    ids_by_x = [obj.track_id for obj in sorted_tracked]
    iou = _iou_batch(
        np.array([[ax1,ay1,ax2,ay2]]),
        np.array([[bx1,by1,bx2,by2]])
    )[0][0] if bx2 > bx1 else 0.0

    id_log.append(ids_by_x)
    print(f"  sample {sample_i:2d}  A=[{int(ax1)},{int(ax2)}]  B=[{int(bx1)},{int(bx2)}]"
          f"  iou={iou:.2f}  ids_left_to_right={ids_by_x}")

print()
# Detect if swap happened
if len([l for l in id_log if len(l) == 2]) >= 4:
    first_valid = next(l for l in id_log if len(l) == 2)
    last_valid  = next(l for l in reversed(id_log) if len(l) == 2)
    swap_happened = (first_valid[0] != last_valid[0])
    print(f"  First confirmed order (left->right): {first_valid}")
    print(f"  Final confirmed order (left->right): {last_valid}")
    print(f"  ID swap occurred: {swap_happened}")
    print()
    if swap_happened:
        print("  *** This is a known ByteTrack limitation. ***")
        print("  *** Re-ID (appearance embedding) would prevent this. ***")
print()


# ------------------------------------------------------------------ #
# TEST 4  –  Unmatched frames before reassignment
# Precisely count the frames a track survives after matching stops.
# ------------------------------------------------------------------ #
print(SEP)
print("TEST 4: Unmatched frames before reassignment (max_age boundary)")
print(SEP)

tracker = make_tracker()

# Warm up one track
for _ in range(5):
    tracker.update([])
    tracker.update([{"bbox": [50.0, 100.0, 150.0, 300.0], "conf": 0.91, "class_name": "person"}])

# Person permanently disappears
print("  Person disappears. Counting frames until track is deleted...")
frames_alive = 0
for frame in range(200):
    tracker.update([])
    alive = any(t.track_id == 1 for t in tracker.trackers)
    if not alive:
        frames_alive = frame
        break

print(f"  Track survived {frames_alive} frames after last detection")
print(f"  max_age setting = 30  (each non-sampled update([]) counts as 1 frame)")
print(f"  Assertion max_age=30: {frames_alive <= 32}")   # small tolerance for final delete
print()


# ------------------------------------------------------------------ #
# TEST 5  –  Verify sequential frame updates reach the tracker
# ------------------------------------------------------------------ #
print(SEP)
print("TEST 5: Verify every native frame calls tracker.update()")
print("        (checks the video_processor loop design)")
print(SEP)

import types

call_log = []

class InstrumentedTracker(ByteTracker):
    def update(self, detections):
        call_log.append(len(detections))
        return super().update(detections)

KalmanBoxTracker._id_counter = 0
itracker = InstrumentedTracker(high_thresh=0.6, low_thresh=0.1, max_age=30, min_hits=2, iou_threshold=0.3)

# Simulate video_processor: native=30fps, sample_fps=1
native_fps = 30
sample_fps = 1
frame_interval = round(native_fps / sample_fps)  # = 30

total_native = 60   # 2 seconds of video

dets_60fps = [{"bbox": [50.0, 100.0, 150.0, 300.0], "conf": 0.91, "class_name": "person"}]

for frame_index in range(total_native):
    is_sampled = (frame_index % frame_interval == 0)
    if is_sampled:
        itracker.update(dets_60fps)
    else:
        itracker.update([])

total_calls = len(call_log)
sampled_calls = sum(1 for c in call_log if c > 0)
non_sampled_calls = sum(1 for c in call_log if c == 0)

print(f"  Total native frames simulated : {total_native}")
print(f"  Total tracker.update() calls  : {total_calls}   (should equal {total_native})")
print(f"  Sampled calls (with dets)     : {sampled_calls}")
print(f"  Non-sampled calls (empty)     : {non_sampled_calls}")
print(f"  Every frame reached tracker   : {total_calls == total_native}")
print()

print(SEP)
print("ALL TESTS COMPLETE")
print(SEP)
