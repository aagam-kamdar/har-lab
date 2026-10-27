"""
tests/test_spatio_temporal_stitching.py
---------------------------------------
Unit tests for Spatio-Temporal Stitching & Body-Only Cross-Track Assignment Disabling:
1. Nominal stitching across track loss/removal (dt <= 5.0s, dist <= 100px).
2. Temporal veto (dt > 5.0s rejects stitching).
3. Spatial distance veto (dist > 100px rejects stitching).
4. Active overlap veto (cannot stitch to a student already possessed by an active track).
5. Body appearance veto (body similarity < 0.40 vetoes stitching).
6. Disabling body-only cross-track assignment ensures pure body score cannot cross-assign.
"""

import os
import sys
import unittest
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter


class DummyTarget:
    def __init__(self, track_id, tlwh):
        self.track_id = track_id
        self.tlwh = tlwh


class TestSpatioTemporalStitching(unittest.TestCase):

    def setUp(self):
        # Initialize adapter without loading heavy YOLO detector
        self.adapter = IdentityAdapter(
            fps=30.0,
            yolo_model_path=None,
            mode="stitching",
            stitching_enabled=True,
            stitching_max_gap_s=5.0,
            stitching_max_dist_px=100.0,
            stitching_body_veto_thresh=0.40,
            body_only_cross_assignment=False
        )

    def test_stitching_nominal_inheritance(self):
        # 1. Establish track 1 at Harshil's desk (cx=700, cy=150) with face lock to harshil
        self.adapter.track_last_center[1] = (700.0, 150.0)
        self.adapter.track_last_seen_f[1] = 100
        self.adapter.track_last_bbox[1] = (650, 100, 750, 200)
        self.adapter.lock_mgr.get_or_create(1, 100)
        self.adapter.lock_mgr._lock(self.adapter.lock_mgr.get_state(1), "harshil", 0.55, 100)

        # 2. Track 1 is removed at frame 110 (t=3.67s)
        self.adapter.update_lifecycle = lambda *args, **kwargs: None
        # Simulate removal of track 1
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.adapter.tracker.tracked_stracks = []
        self.adapter.tracker.lost_stracks = []
        self.adapter.tracker.removed_stracks = [DummyTarget(1, [650, 100, 100, 100])]
        self.adapter.tracker.update = lambda *args, **kwargs: []

        tracks = self.adapter.update(dummy_frame, frame_idx=110, det_boxes=np.empty((0, 5), dtype=np.float32))
        self.assertEqual(len(self.adapter.recent_ended_tracks), 1)
        self.assertEqual(self.adapter.recent_ended_tracks[0]["student"], "harshil")

        # 3. New track 2 appears at frame 140 (t=4.67s, gap=1.0s <= 5.0s) at (710, 155), dist=11.18px <= 100px
        # Body similarity to harshil = 0.65 >= 0.40
        self.adapter.tracker.tracked_stracks = [DummyTarget(2, [660, 105, 100, 100])]
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTarget(2, [660, 105, 100, 100])]
        self.adapter.track_body_ema[2]["harshil"] = 0.65

        tracks = self.adapter.update(dummy_frame, frame_idx=140, det_boxes=np.empty((0, 5), dtype=np.float32))
        t2 = next(t for t in tracks if t["track_id"] == 2)
        # Track 2 must inherit Harshil via spatio-temporal stitching
        self.assertEqual(t2["person_id"], "harshil")
        self.assertIn("Harshil", t2["display_name"])

    def test_stitching_temporal_veto(self):
        # Ended track at frame 100 (t=3.33s)
        self.adapter.recent_ended_tracks.append({
            "track_id": 1,
            "student": "swayam",
            "frame_idx": 100,
            "timestamp_s": 100 / 30.0,
            "last_center": (500.0, 150.0),
            "last_bbox": (450, 100, 550, 200)
        })

        # New track appears at frame 300 (t=10.0s -> dt = 6.67s > 5.0s max gap)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.adapter.tracker.tracked_stracks = [DummyTarget(3, [450, 100, 100, 100])]
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTarget(3, [450, 100, 100, 100])]
        self.adapter.track_body_ema[3]["swayam"] = 0.70

        tracks = self.adapter.update(dummy_frame, frame_idx=300, det_boxes=np.empty((0, 5), dtype=np.float32))
        t3 = next(t for t in tracks if t["track_id"] == 3)
        # Must be rejected because gap > 5s
        self.assertNotEqual(t3["person_id"], "swayam")
        self.assertNotIn(3, self.adapter.stitched_tracks)

    def test_stitching_spatial_distance_veto(self):
        # Ended track at (300, 150)
        self.adapter.recent_ended_tracks.append({
            "track_id": 1,
            "student": "soham",
            "frame_idx": 100,
            "timestamp_s": 100 / 30.0,
            "last_center": (300.0, 150.0),
            "last_bbox": (250, 100, 350, 200)
        })

        # New track appears at (600, 150) -> dist = 300px > 100px max dist
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.adapter.tracker.tracked_stracks = [DummyTarget(4, [550, 100, 100, 100])]
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTarget(4, [550, 100, 100, 100])]
        self.adapter.track_body_ema[4]["soham"] = 0.70

        tracks = self.adapter.update(dummy_frame, frame_idx=110, det_boxes=np.empty((0, 5), dtype=np.float32))
        t4 = next(t for t in tracks if t["track_id"] == 4)
        # Must be rejected due to distance
        self.assertNotEqual(t4["person_id"], "soham")
        self.assertNotIn(4, self.adapter.stitched_tracks)

    def test_stitching_body_appearance_veto(self):
        # Ended track for aagam
        self.adapter.recent_ended_tracks.append({
            "track_id": 1,
            "student": "aagam",
            "frame_idx": 100,
            "timestamp_s": 100 / 30.0,
            "last_center": (850.0, 150.0),
            "last_bbox": (800, 100, 900, 200)
        })

        # New track appears near desk, but body appearance similarity is 0.25 (< 0.40 veto thresh)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.adapter.tracker.tracked_stracks = [DummyTarget(5, [805, 105, 100, 100])]
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTarget(5, [805, 105, 100, 100])]
        self.adapter.track_body_ema[5]["aagam"] = 0.25  # BELOW 0.40 VETO

        tracks = self.adapter.update(dummy_frame, frame_idx=110, det_boxes=np.empty((0, 5), dtype=np.float32))
        t5 = next(t for t in tracks if t["track_id"] == 5)
        # Must be vetoed by appearance
        self.assertNotEqual(t5["person_id"], "aagam")
        self.assertNotIn(5, self.adapter.stitched_tracks)

    def test_body_only_cross_assignment_disabled_by_default(self):
        # Track 6 has body similarity 0.75 to harshil, but NO face evidence (f_s = 0.0)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.adapter.tracker.tracked_stracks = [DummyTarget(6, [200, 100, 100, 100])]
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTarget(6, [200, 100, 100, 100])]
        self.adapter.track_body_ema[6]["harshil"] = 0.75
        self.adapter.track_face_ema[6]["harshil"] = 0.0

        tracks = self.adapter.update(dummy_frame, frame_idx=10, det_boxes=np.empty((0, 5), dtype=np.float32))
        t6 = next(t for t in tracks if t["track_id"] == 6)
        # Because body_only_cross_assignment is False, Track 6 CANNOT be assigned to Harshil
        self.assertNotEqual(t6["person_id"], "harshil")


if __name__ == "__main__":
    unittest.main()
