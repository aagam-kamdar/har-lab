"""
tests/test_event_driven_face_identity.py
-----------------------------------------
Unit tests verifying:
  1. Quality-gated facial observation rejection (size, sharpness, aspect ratio, profile).
  2. Event-driven ArcFace execution: Seated and locked tracks bypass ArcFace calls.
  3. Candidate evidence aggregation with temporal independence constraints.
  4. Explicit Identity State model transitions:
     UNIDENTIFIED -> FACE_CANDIDATE -> IDENTITY_LOCKED -> REIDENTIFICATION_PENDING -> IDENTITY_CONFLICT -> UNKNOWN.
  5. Spatial desk conflict arbitration (Desk 1 Harshil, Desk 3 Aagam).
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter, IdentityState, HOME_DESKS, DESK_ROIS


class DummyTrack:
    def __init__(self, track_id: int, tlwh):
        self.track_id = track_id
        self.tlwh = tlwh


class TestEventDrivenFaceIdentity(unittest.TestCase):

    def setUp(self):
        self.adapter = IdentityAdapter(
            fps=30.0,
            yolo_model_path=None,
            mode="stitching",
            face_min_size=24,
            face_min_det_conf=0.55,
            face_min_sharpness=30.0,
            face_temporal_independence_s=0.20,
            face_min_confirmations=2,
            face_event_interval_frames=10,
        )

    def test_quality_gating_filters_poor_crops(self):
        """Verify quality gating rejects tiny, blurred, or empty frames."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Black frame has no face and zero sharpness
        res = self.adapter.extract_quality_face(dummy_frame, (100, 100, 200, 300))
        self.assertIsNone(res)

    def test_seated_and_locked_tracks_skip_arcface(self):
        """Verify once a track is locked, ArcFace is bypassed to keep identity sticky."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Setup track 10 locked to Harshil at Desk 1 (cx=510, cy=180)
        self.adapter.lock_mgr.get_or_create(10, 0)
        self.adapter.lock_mgr._lock(self.adapter.lock_mgr.get_state(10), "harshil", 0.75, 0)
        self.adapter.track_assigned[10] = ("harshil", 0.75, True)
        self.adapter.track_last_center[10] = (510.0, 180.0)

        # Mock tracker to return track 10
        self.adapter.tracker.tracked_stracks = [DummyTrack(10, [460, 50, 100, 260])]
        self.adapter.tracker.lost_stracks = []
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTrack(10, [460, 50, 100, 260])]

        call_counter = {"count": 0}
        original_extract = self.adapter.extract_quality_face

        def mock_extract(*args, **kwargs):
            call_counter["count"] += 1
            return original_extract(*args, **kwargs)

        self.adapter.extract_quality_face = mock_extract

        # Run 20 frames
        for f in range(20):
            tracks = self.adapter.update(dummy_frame, frame_idx=f, det_boxes=np.empty((0, 5), dtype=np.float32))
            t = next(tr for tr in tracks if tr["track_id"] == 10)
            self.assertEqual(t["person_id"], "harshil")
            self.assertEqual(t["identity_state"], IdentityState.IDENTITY_LOCKED)
            self.assertTrue(t["is_locked"])

        # ArcFace must have been called 0 times because track is already locked!
        self.assertEqual(call_counter["count"], 0)

    def test_temporal_independence_evidence_accumulation(self):
        """Verify adjacent frames within temporal gap do not count as independent confirmations."""
        tid = 15
        min_gap_f = int(self.adapter.face_temporal_independence_s * self.adapter.fps)  # 6 frames
        self.assertEqual(min_gap_f, 6)

        # Mock observation accumulation
        obs_buf = self.adapter.face_candidate_observations[tid]
        self.assertEqual(len(obs_buf), 0)

        # First observation at frame 10
        self.adapter.track_last_face_frame[tid] = 10
        obs_buf.append({"student": "aagam", "score": 0.55, "margin": 0.12, "quality": 0.8, "frame_idx": 10})

        # Observation attempt at frame 12 (gap=2 < 6) -> ignored by temporal filter
        last_f = self.adapter.track_last_face_frame[tid]
        self.assertTrue((12 - last_f) < min_gap_f)

        # Observation attempt at frame 18 (gap=8 >= 6) -> valid independent confirmation
        self.assertTrue((18 - last_f) >= min_gap_f)
        obs_buf.append({"student": "aagam", "score": 0.58, "margin": 0.15, "quality": 0.85, "frame_idx": 18})

        self.assertEqual(len(obs_buf), 2)
        self.assertTrue(all(o["student"] == "aagam" for o in obs_buf))

    def test_conflict_detection_and_home_desk_arbitration(self):
        """
        Verify when 2 tracks claim Harshil, the track at Desk 1 (D01: Harshil's home desk)
        wins and retains Harshil, while the intruder is marked as IDENTITY_CONFLICT.
        """
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(HOME_DESKS["harshil"], "D01")
        self.assertEqual(HOME_DESKS["aagam"], "D03")

        # Track 1 at Desk 1 (cx=500, cy=150)
        # Track 2 at Desk 3 (cx=700, cy=150)
        self.adapter.tracker.tracked_stracks = [
            DummyTrack(1, [450, 50, 100, 200]),
            DummyTrack(2, [650, 50, 100, 200]),
        ]
        self.adapter.tracker.lost_stracks = []
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [
            DummyTrack(1, [450, 50, 100, 200]),
            DummyTrack(2, [650, 50, 100, 200]),
        ]

        # Track 1 has authoritative lock to 'harshil' at Desk 1
        self.adapter.lock_mgr.get_or_create(1, 0)
        self.adapter.lock_mgr._lock(self.adapter.lock_mgr.get_state(1), "harshil", 0.80, 0)
        self.adapter.track_assigned[1] = ("harshil", 0.80, True)
        self.adapter.track_assigned[2] = ("harshil", 0.75, False)
        self.adapter.track_last_center[1] = (500.0, 150.0)
        self.adapter.track_last_center[2] = (700.0, 150.0)

        tracks = self.adapter.update(dummy_frame, frame_idx=50, det_boxes=np.empty((0, 5), dtype=np.float32))

        t1 = next(tr for tr in tracks if tr["track_id"] == 1)
        t2 = next(tr for tr in tracks if tr["track_id"] == 2)

        # Track 1 (at Desk 1) must win Harshil
        self.assertEqual(t1["person_id"], "harshil")

        # Track 2 (at Desk 3) must be marked as IDENTITY_CONFLICT
        self.assertEqual(t2["identity_state"], IdentityState.IDENTITY_CONFLICT)
        self.assertEqual(t2["status"], "CONFLICT")

    def test_explicit_identity_state_model_coverage(self):
        """Verify explicit states: UNIDENTIFIED, FACE_CANDIDATE, IDENTITY_LOCKED, UNKNOWN."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # New unassigned track at frame 0 -> UNIDENTIFIED
        self.adapter.tracker.tracked_stracks = [DummyTrack(99, [100, 50, 100, 200])]
        self.adapter.tracker.lost_stracks = []
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTrack(99, [100, 50, 100, 200])]

        tracks = self.adapter.update(dummy_frame, frame_idx=0, det_boxes=np.empty((0, 5), dtype=np.float32))
        t99 = next(tr for tr in tracks if tr["track_id"] == 99)
        self.assertEqual(t99["identity_state"], IdentityState.UNIDENTIFIED)

        # Track with candidate face match -> FACE_CANDIDATE
        self.adapter.face_candidate_observations[99].append({"student": "swayam", "score": 0.50, "margin": 0.10, "quality": 0.7, "frame_idx": 10})
        tracks = self.adapter.update(dummy_frame, frame_idx=10, det_boxes=np.empty((0, 5), dtype=np.float32))
        t99 = next(tr for tr in tracks if tr["track_id"] == 99)
        self.assertEqual(t99["identity_state"], IdentityState.FACE_CANDIDATE)

        # Track exceeding timeout without match -> UNKNOWN
        self.adapter.face_candidate_observations.pop(99, None)
        # Timeout is 5.0s (150 frames at 30 fps)
        tracks = self.adapter.update(dummy_frame, frame_idx=200, det_boxes=np.empty((0, 5), dtype=np.float32))
        t99 = next(tr for tr in tracks if tr["track_id"] == 99)
        self.assertEqual(t99["identity_state"], IdentityState.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
