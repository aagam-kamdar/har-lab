"""
tests/test_identity_reid_workstation.py
---------------------------------------
Comprehensive acceptance & regression test suite verifying:
  1. Duplicate identity prevention (overlapping tracks / track splits)
  2. Temporal co-existence invariant in spatio-temporal stitching
  3. Workstation boundary enforcement during track stitching
  4. Temporary occlusion preservation and recovery
  5. Invalidation and reassignment on contradictory face evidence
  6. Persistent desk assignment consistency during student relocation
  7. Pretrained VideoMAE top-k confidence & margin-qualified activity mapping
  8. Structured re-identification transition logging
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity_adapter import IdentityAdapter, IdentityState, HOME_DESKS, DESK_ROIS, PersistentIdentity
from har_classifier import HARClassifier, KINETICS_TO_APP_MAP


class DummyTrack:
    def __init__(self, track_id: int, tlwh):
        self.track_id = track_id
        self.tlwh = tlwh


class TestIdentityReIDWorkstation(unittest.TestCase):

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

    def test_duplicate_identity_prevention(self):
        """
        Verify two overlapping bounding boxes (IoU >= 0.45) for the same person
        do not create competing tracks or duplicate enrolled claims.
        """
        dummy_frame = np.zeros((480, 1120, 3), dtype=np.uint8)
        # Track 1 and Track 21 at Desk 4 with heavy overlap (IoU > 0.70)
        # Bboxes: (773, 65, 981, 240) and (776, 64, 984, 315)
        self.adapter.tracker.tracked_stracks = [
            DummyTrack(1, [773, 65, 208, 175]),
            DummyTrack(21, [776, 64, 208, 251]),
        ]
        self.adapter.tracker.lost_stracks = []
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [
            DummyTrack(1, [773, 65, 208, 175]),
            DummyTrack(21, [776, 64, 208, 251]),
        ]

        # Lock track 1 to Soham
        self.adapter.lock_mgr.get_or_create(1, 0)
        self.adapter.lock_mgr._lock(self.adapter.lock_mgr.get_state(1), "soham", 0.85, 0)
        self.adapter.track_assigned[1] = ("soham", 0.85, True)

        tracks = self.adapter.update(dummy_frame, frame_idx=0, det_boxes=np.empty((0, 5), dtype=np.float32))

        t1 = next(tr for tr in tracks if tr["track_id"] == 1)
        t21 = next(tr for tr in tracks if tr["track_id"] == 21)

        self.assertEqual(t1["person_id"], "soham")
        self.assertEqual(t1["status"], "LOCKED")

        # Track 21 must be marked as duplicate and suppressed from claiming an enrolled slot
        self.assertEqual(t21["status"], "DUPLICATE")
        self.assertIn(21, self.adapter.duplicate_of)
        self.assertEqual(self.adapter.duplicate_of[21], 1)

    def test_temporal_coexistence_invariant_in_stitching(self):
        """
        Verify a track that already co-existed with an ended track cannot be stitched to it.
        """
        # Track 10 (Swayam at Desk 2) ended at frame 100
        # Track 20 was born at frame 50 (co-existed for 50 frames!)
        curr_t_s = 101 / 30.0
        self.adapter.recent_ended_tracks.append({
            "track_id": 10,
            "student": "swayam",
            "frame_idx": 100,
            "timestamp_s": 100 / 30.0,
            "last_center": (600.0, 150.0),
            "last_bbox": (550, 50, 650, 250),
        })

        # Track 20 was first seen at frame 50
        self.adapter.track_first_seen_frame[20] = 50
        self.adapter.track_last_center[20] = (610.0, 150.0)

        dummy_frame = np.zeros((480, 1120, 3), dtype=np.uint8)
        self.adapter.tracker.tracked_stracks = [DummyTrack(20, [560, 50, 100, 200])]
        self.adapter.tracker.lost_stracks = []
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTrack(20, [560, 50, 100, 200])]

        tracks = self.adapter.update(dummy_frame, frame_idx=101, det_boxes=np.empty((0, 5), dtype=np.float32))
        t20 = next(tr for tr in tracks if tr["track_id"] == 20)

        # Track 20 cannot stitch to Swayam because it was already alive while Track 10 was active
        self.assertNotEqual(t20["person_id"], "swayam")
        self.assertNotIn(20, self.adapter.stitched_tracks)

    def test_workstation_boundary_stitching_veto(self):
        """
        Verify an ended track from Desk 2 (D02) is NOT stitched to an unassigned track at Desk 3 (D03).
        """
        # Swayam ended at Desk 2 (cx=600 in D02 [540, 660])
        self.adapter.recent_ended_tracks.append({
            "track_id": 10,
            "student": "swayam",
            "frame_idx": 100,
            "timestamp_s": 100 / 30.0,
            "last_center": (600.0, 150.0),
            "last_bbox": (550, 50, 650, 250),
        })

        # Track 25 starts at frame 102 at Desk 3 (cx=700 in D03 [640, 780])
        # Even though distance hypot(700-600) = 100px is near threshold, desk zones differ
        dummy_frame = np.zeros((480, 1120, 3), dtype=np.uint8)
        self.adapter.tracker.tracked_stracks = [DummyTrack(25, [650, 50, 100, 200])]
        self.adapter.tracker.lost_stracks = []
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTrack(25, [650, 50, 100, 200])]

        tracks = self.adapter.update(dummy_frame, frame_idx=102, det_boxes=np.empty((0, 5), dtype=np.float32))
        t25 = next(tr for tr in tracks if tr["track_id"] == 25)

        # Must not be stitched to Swayam
        self.assertNotEqual(t25["person_id"], "swayam")
        self.assertNotIn(25, self.adapter.stitched_tracks)

    def test_temporary_occlusion_and_preservation(self):
        """
        Verify a locked student's identity is retained in PersistentIdentity as TRACK_OCCLUDED
        when briefly lost, and recovered cleanly when reappearing at the same desk.
        """
        dummy_frame = np.zeros((480, 1120, 3), dtype=np.uint8)

        # 1. Lock Harshil at Desk 1 (cx=500)
        self.adapter.lock_mgr.get_or_create(5, 0)
        self.adapter.lock_mgr._lock(self.adapter.lock_mgr.get_state(5), "harshil", 0.85, 0)
        self.adapter.track_assigned[5] = ("harshil", 0.85, True)

        self.adapter.tracker.tracked_stracks = [DummyTrack(5, [450, 50, 100, 200])]
        self.adapter.tracker.lost_stracks = []
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTrack(5, [450, 50, 100, 200])]

        self.adapter.update(dummy_frame, frame_idx=0, det_boxes=np.empty((0, 5), dtype=np.float32))
        p_harshil = self.adapter.persistent_identities["harshil"]
        self.assertEqual(p_harshil.active_track_id, 5)
        self.assertEqual(p_harshil.identity_state, IdentityState.IDENTITY_LOCKED)

        # 2. Track 5 removed at frame 30 (brief occlusion)
        self.adapter.tracker.tracked_stracks = []
        self.adapter.tracker.lost_stracks = []
        self.adapter.tracker.removed_stracks = [DummyTrack(5, [450, 50, 100, 200])]
        self.adapter.tracker.update = lambda *args, **kwargs: []

        self.adapter.update(dummy_frame, frame_idx=30, det_boxes=np.empty((0, 5), dtype=np.float32))
        self.assertEqual(p_harshil.active_track_id, None)
        self.assertEqual(p_harshil.previous_track_id, 5)
        self.assertEqual(p_harshil.identity_state, IdentityState.TRACK_OCCLUDED)
        self.assertEqual(p_harshil.occluded_since_frame, 30)

        # 3. Track 30 appears at frame 40 at the same desk (Desk 1)
        self.adapter.tracker.tracked_stracks = [DummyTrack(30, [455, 52, 100, 200])]
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTrack(30, [455, 52, 100, 200])]
        self.adapter.track_body_ema[30]["harshil"] = 0.70

        tracks = self.adapter.update(dummy_frame, frame_idx=40, det_boxes=np.empty((0, 5), dtype=np.float32))
        t30 = next(tr for tr in tracks if tr["track_id"] == 30)

        # Track 30 smoothly stitches to Harshil
        self.assertEqual(t30["person_id"], "harshil")
        self.assertEqual(p_harshil.active_track_id, 30)

    def test_contradictory_face_match_invalidation(self):
        """
        Verify that if an active track is locked to Harshil, but repeatedly observes Aagam
        with high quality face matches, the previous lock is invalidated and reassigned.
        """
        dummy_frame = np.zeros((480, 1120, 3), dtype=np.uint8)

        # Track 16 locked to harshil at frame 0
        self.adapter.lock_mgr.get_or_create(16, 0)
        self.adapter.lock_mgr._lock(self.adapter.lock_mgr.get_state(16), "harshil", 0.75, 0)
        self.adapter.track_assigned[16] = ("harshil", 0.75, True)
        self.adapter.track_last_center[16] = (700.0, 150.0)  # Desk 3! Relocated away from D01

        self.adapter.tracker.tracked_stracks = [DummyTrack(16, [650, 50, 100, 200])]
        self.adapter.tracker.lost_stracks = []
        self.adapter.tracker.removed_stracks = []
        self.adapter.tracker.update = lambda *args, **kwargs: [DummyTrack(16, [650, 50, 100, 200])]

        # Mock extract_quality_face to return a valid embedding matching 'aagam'
        aagam_emb = self.adapter.face_profiles["aagam"]
        mock_meta = {"quality": 0.85, "det_score": 0.80, "sharpness": 55.0}

        def mock_extract(frame, bbox, *args, **kwargs):
            return aagam_emb, 0.85, mock_meta

        self.adapter.extract_quality_face = mock_extract

        # Run frame 10 (first confirmation of contradiction)
        self.adapter.update(dummy_frame, frame_idx=10, det_boxes=np.empty((0, 5), dtype=np.float32))
        self.assertEqual(len(self.adapter.contradictory_observations[16]), 1)

        # Run frame 20 (second independent confirmation -> lock invalidated & reassigned to aagam!)
        self.adapter.update(dummy_frame, frame_idx=20, det_boxes=np.empty((0, 5), dtype=np.float32))

        state = self.adapter.lock_mgr.get_state(16)
        self.assertEqual(state.locked_name, "aagam")
        self.assertTrue(any(ev["trigger_event"] == "contradictory_face_evidence" for ev in self.adapter.reid_events))

    def test_desk_assignment_consistency_during_relocation(self):
        """
        Verify that when a student is seated at Desk 1 for 50 frames and then
        temporarily walks past Desk 3, their cumulative verified workstation remains Desk 1.
        """
        dummy_frame = np.zeros((480, 1120, 3), dtype=np.uint8)

        # Harshil seated at D01 (cx=500) for 10 frames
        self.adapter.lock_mgr.get_or_create(2, 0)
        self.adapter.lock_mgr._lock(self.adapter.lock_mgr.get_state(2), "harshil", 0.90, 0)
        self.adapter.track_assigned[2] = ("harshil", 0.90, True)

        for f in range(10):
            self.adapter.tracker.update = lambda *args, **kwargs: [DummyTrack(2, [450, 50, 100, 200])]
            self.adapter.update(dummy_frame, frame_idx=f, det_boxes=np.empty((0, 5), dtype=np.float32))

        # Harshil walks past Desk 3 (cx=700) for 2 frames
        for f in range(10, 12):
            self.adapter.tracker.update = lambda *args, **kwargs: [DummyTrack(2, [650, 50, 100, 200])]
            self.adapter.update(dummy_frame, frame_idx=f, det_boxes=np.empty((0, 5), dtype=np.float32))

        p_harshil = self.adapter.persistent_identities["harshil"]
        self.assertEqual(p_harshil.home_desk, "D01")
        # D01 has 10 frames, D03 has 2 frames
        self.assertEqual(p_harshil.desk_frames["D01"], 10)
        self.assertEqual(p_harshil.desk_frames["D03"], 2)
        verified_desk = max(p_harshil.desk_frames.items(), key=lambda x: x[1])[0]
        self.assertEqual(verified_desk, "D01")

    def test_videomae_topk_and_margin_activity_mapping(self):
        """
        Verify VideoMAE label mapping logic:
          - 'using computer' with confidence >= 0.25 maps to active_work
          - 'using computer' with confidence 0.16 and margin >= 0.05 maps to active_work
          - Unmapped Kinetics classes (e.g. 'skydiving') map to uncertain
          - Low confidence without margin maps to uncertain
        """
        har = HARClassifier(confidence_threshold=0.25)

        # Standard threshold pass
        mapped, is_m = har.map_label("using computer", confidence=0.35, margin=0.10)
        self.assertEqual(mapped, "active_work")
        self.assertTrue(is_m)

        # Margin-qualified pass for 400-class probability diffusion
        mapped, is_m = har.map_label("using computer", confidence=0.16, margin=0.08)
        self.assertEqual(mapped, "active_work")
        self.assertTrue(is_m)

        # Low confidence without margin falls through to uncertain
        mapped, is_m = har.map_label("using computer", confidence=0.08, margin=0.01)
        self.assertEqual(mapped, "uncertain")
        self.assertFalse(is_m)

        # Out-of-domain Kinetics class
        mapped, is_m = har.map_label("skydiving", confidence=0.80, margin=0.50)
        self.assertEqual(mapped, "uncertain")
        self.assertFalse(is_m)

    def test_structured_reid_event_logging(self):
        """
        Verify structured re-identification transition logging captures:
          previous identity, confidence, tracker IDs, trigger event, face quality, and reason.
        """
        self.adapter.log_reid_event(
            track_id=22,
            prev_identity="harshil",
            prev_confidence=0.72,
            prev_track_id=16,
            trigger_event="occlusion_recovery",
            face_meta={"quality": 0.88, "det_score": 0.82, "sharpness": 62.4},
            candidates={"harshil": 0.72, "aagam": 0.18},
            decision="REIDENTIFICATION_PENDING",
            reason="Recovered from occlusion at home desk D01",
        )

        self.assertEqual(len(self.adapter.reid_events), 1)
        ev = self.adapter.reid_events[0]
        self.assertEqual(ev["track_id"], 22)
        self.assertEqual(ev["prev_identity"], "harshil")
        self.assertEqual(ev["prev_confidence"], 0.72)
        self.assertEqual(ev["prev_track_id"], 16)
        self.assertEqual(ev["trigger_event"], "occlusion_recovery")
        self.assertEqual(ev["face_quality"], 0.88)
        self.assertEqual(ev["decision"], "REIDENTIFICATION_PENDING")
        self.assertIn("home desk D01", ev["reason"])


if __name__ == "__main__":
    unittest.main()
