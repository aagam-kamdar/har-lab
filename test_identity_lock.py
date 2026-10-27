"""
test_identity_lock.py
---------------------
Unit tests for IdentityLockManager and Rendering Layer.
Tests all locking-policy rules, permanent stickiness, ByteTrack TRACKED/LOST/REMOVED
lifecycle handling, ownership invariants, and the rendering/output layer.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identity_lock_manager import (
    IdentityLockManager,
    FACE_LOCK_THRESHOLD,
    FACE_AMBIGUITY_MARGIN,
    FACE_MIN_CONFIRMATIONS,
    BODY_RECOVERY_THRESHOLD,
    BODY_RECOVERY_AMBIGUITY,
    FACE_FRAME_GAP_FOR_RECOVERY,
)
from robust_tracker import draw_track


class TestLockingPolicy(unittest.TestCase):

    def setUp(self):
        self.mgr = IdentityLockManager()

    # ------------------------------------------------------------------
    # Rule 1: New track starts UNKNOWN
    # ------------------------------------------------------------------
    def test_rule1_new_track_is_unknown(self):
        state = self.mgr.get_or_create(track_id=1, current_frame=0)
        self.assertFalse(state.locked)
        self.assertIsNone(state.locked_name)
        self.assertEqual(state.identity_label(), "UNKNOWN")
        self.assertEqual(self.mgr.get_label(1), "UNKNOWN")

    # ------------------------------------------------------------------
    # Rule 2: Body Re-ID alone can NEVER assign a name
    # ------------------------------------------------------------------
    def test_rule2_body_cannot_assign_name(self):
        self.mgr.get_or_create(1, 0)
        for i in range(20):
            self.mgr.submit_body_evidence(1, i, "aagam", 0.95)
        state = self.mgr.get_state(1)
        self.assertFalse(state.locked)
        self.assertEqual(state.identity_label(), "UNKNOWN")
        self.assertEqual(self.mgr.get_label(1), "UNKNOWN")

    # ------------------------------------------------------------------
    # Rule 3: Identity locked ONLY after confident face match
    # ------------------------------------------------------------------
    def test_rule3_weak_face_does_not_lock(self):
        score_below_threshold = FACE_LOCK_THRESHOLD - 0.05
        for frame in range(FACE_MIN_CONFIRMATIONS + 5):
            label = self.mgr.submit_face_match(
                1, frame, "aagam", score_below_threshold, 0.01
            )
        self.assertEqual(label, "UNKNOWN")
        self.assertFalse(self.mgr.get_state(1).locked)

    def test_rule3_ambiguous_face_does_not_lock(self):
        score = FACE_LOCK_THRESHOLD + 0.10
        second = score - (FACE_AMBIGUITY_MARGIN / 2)
        for frame in range(FACE_MIN_CONFIRMATIONS + 5):
            label = self.mgr.submit_face_match(1, frame, "aagam", score, second)
        self.assertEqual(label, "UNKNOWN")
        self.assertFalse(self.mgr.get_state(1).locked)

    def test_rule3_confident_face_locks_after_confirmations(self):
        score = FACE_LOCK_THRESHOLD + 0.15
        second = score - FACE_AMBIGUITY_MARGIN - 0.05
        label = None
        for frame in range(FACE_MIN_CONFIRMATIONS):
            label = self.mgr.submit_face_match(1, frame, "aagam", score, second)
        self.assertEqual(label, "aagam")
        state = self.mgr.get_state(1)
        self.assertTrue(state.locked)
        self.assertEqual(state.locked_name, "aagam")
        self.assertEqual(self.mgr.student_owner.get("aagam"), 1)

    def test_initial_lock_requires_320_640_agreement(self):
        """
        Dual-resolution agreement gate for initial locks:
        - Positive test: both 320 and 640 score >= 0.38 for same student -> lock succeeds.
        - Negative test 1: 320 scores >= 0.38, 640 scores < 0.38 -> lock rejected.
        - Negative test 2: 320 and 640 score >= 0.38 but for DIFFERENT students -> lock rejected.
        - Negative test 3: Swayam/Harshil exact case from frame 1946 (320=0.410, 640=0.2095) -> lock rejected.
        """
        # Positive test: both 320 and 640 score >= 0.38 for the same student -> lock succeeds
        label_pos = None
        for frame in range(FACE_MIN_CONFIRMATIONS):
            label_pos = self.mgr.submit_face_match(
                1, frame, "aagam", 0.50, 0.10,
                name_640="aagam", score_640=0.48, second_score_640=0.10
            )
        self.assertEqual(label_pos, "aagam")
        self.assertTrue(self.mgr.get_state(1).locked)
        self.assertEqual(self.mgr.student_owner.get("aagam"), 1)

        # Negative test 1: 320 scores >= 0.38, 640 scores < 0.38 -> lock rejected
        for frame in range(FACE_MIN_CONFIRMATIONS + 2):
            label_neg1 = self.mgr.submit_face_match(
                2, frame, "swayam", 0.45, 0.10,
                name_640="swayam", score_640=0.32, second_score_640=0.10
            )
            self.assertEqual(label_neg1, "UNKNOWN")
        self.assertFalse(self.mgr.get_state(2).locked)
        self.assertNotIn("swayam", self.mgr.student_owner)

        # Negative test 2: 320 and 640 score >= 0.38 but for DIFFERENT students -> lock rejected
        for frame in range(FACE_MIN_CONFIRMATIONS + 2):
            label_neg2 = self.mgr.submit_face_match(
                3, frame, "swayam", 0.45, 0.10,
                name_640="harshil", score_640=0.42, second_score_640=0.10
            )
            self.assertEqual(label_neg2, "UNKNOWN")
        self.assertFalse(self.mgr.get_state(3).locked)
        self.assertNotIn("swayam", self.mgr.student_owner)
        self.assertNotIn("harshil", self.mgr.student_owner)

        # Negative test 3: Swayam/Harshil exact case from frame 1946 (320=0.410, 640=0.2095) -> lock rejected
        for frame in range(FACE_MIN_CONFIRMATIONS + 2):
            label_neg3 = self.mgr.submit_face_match(
                11, 1946 + frame, "harshil", 0.410, 0.228,
                name_640="harshil", score_640=0.2095, second_score_640=0.1887
            )
            self.assertEqual(label_neg3, "UNKNOWN")
        self.assertFalse(self.mgr.get_state(11).locked)
        self.assertNotIn("harshil", self.mgr.student_owner)

    # ------------------------------------------------------------------
    # Track Lifecycle: LOST vs REMOVED Regression Test
    # ------------------------------------------------------------------
    def test_track_lifecycle_lost_survives_and_removed_releases(self):
        """
        1. Track 10 locks to Swayam.
        2. Track 10 becomes LOST for several frames.
        3. Assert ownership and locked_name remain Swayam.
        4. Track 10 becomes TRACKED again.
        5. Assert it is still Swayam.
        6. Then mark Track 10 REMOVED.
        7. Assert RELEASE occurs and ownership is cleared.
        """
        score = FACE_LOCK_THRESHOLD + 0.15
        sec = 0.05

        # 1. Track 10 locks to Swayam
        for f in range(FACE_MIN_CONFIRMATIONS):
            self.mgr.submit_face_match(10, f, "swayam", score, sec)
        self.mgr.update_lifecycle(tracked_ids=[10], lost_ids=[], removed_ids=[], current_frame=1)
        self.assertEqual(self.mgr.get_label(10), "swayam")
        self.assertEqual(self.mgr.student_owner.get("swayam"), 10)
        self.assertEqual(self.mgr.get_state(10).status, "TRACKED")

        # 2. Track 10 becomes LOST for several frames (occluded / undetected)
        for f in range(2, 20):
            self.mgr.update_lifecycle(tracked_ids=[], lost_ids=[10], removed_ids=[], current_frame=f)
            # 3. Assert ownership and locked_name remain Swayam
            self.assertEqual(self.mgr.student_owner.get("swayam"), 10, f"Frame {f}: lost track lost ownership prematurely!")
            self.assertEqual(self.mgr.get_label(10), "swayam", f"Frame {f}: lost track label changed!")
            self.assertEqual(self.mgr.get_state(10).status, "LOST")
            self.assertTrue(self.mgr.get_state(10).locked)

        # 4. Track 10 becomes TRACKED again (recovered with same track_id)
        self.mgr.update_lifecycle(tracked_ids=[10], lost_ids=[], removed_ids=[], current_frame=21)
        # 5. Assert it is still Swayam
        self.assertEqual(self.mgr.get_state(10).status, "TRACKED")
        self.assertEqual(self.mgr.get_label(10), "swayam")
        self.assertEqual(self.mgr.student_owner.get("swayam"), 10)

        # 6. Then mark Track 10 REMOVED (permanently deleted by ByteTracker)
        self.mgr.update_lifecycle(tracked_ids=[], lost_ids=[], removed_ids=[10], current_frame=100)

        # 7. Assert RELEASE occurs and ownership is cleared
        self.assertNotIn("swayam", self.mgr.student_owner)
        self.assertNotIn(10, self.mgr.track_to_student)
        self.assertEqual(self.mgr.get_state(10).status, "REMOVED")
        self.assertEqual(self.mgr.get_label(10), "INACTIVE")

    # ------------------------------------------------------------------
    # Permanent Sticky Lock Regression Test across 200 adverse frames
    # ------------------------------------------------------------------
    def test_locked_label_survives_every_kind_of_bad_evidence_every_frame(self):
        score = FACE_LOCK_THRESHOLD + 0.15
        sec = 0.05

        # Lock Track 5 to aagam
        for f in range(FACE_MIN_CONFIRMATIONS):
            self.mgr.submit_face_match(5, f, "aagam", score, sec)

        self.assertTrue(self.mgr.get_state(5).locked)
        self.assertEqual(self.mgr.get_label(5), "aagam")
        self.assertEqual(self.mgr.student_owner.get("aagam"), 5)

        students = ["soham", "swayam", "harshil"]

        for f in range(10, 210):
            adversity_type = f % 5

            if adversity_type == 0:
                self.mgr.update_seen(5, f)
            elif adversity_type == 1:
                cand = students[f % len(students)]
                self.mgr.submit_face_match(5, f, cand, 0.95, 0.10)
            elif adversity_type == 2:
                cand = students[f % len(students)]
                self.mgr.submit_body_evidence(5, f, cand, 0.98)
            elif adversity_type == 3:
                cand = students[f % len(students)]
                self.mgr.submit_face_match(5, f, cand, 0.45, 0.38)
            elif adversity_type == 4:
                cand = students[f % len(students)]
                self.mgr.submit_body_evidence(5, f, cand, 0.92)
                self.mgr.submit_face_match(5, f, cand, 0.88, 0.15)

            self.assertEqual(self.mgr.get_label(5), "aagam")
            state = self.mgr.get_state(5)
            self.assertTrue(state.locked)
            self.assertEqual(state.locked_name, "aagam")
            self.assertEqual(self.mgr.student_owner.get("aagam"), 5)

        # Mark Track 5 removed
        self.mgr.update_lifecycle(tracked_ids=[1, 2, 3], lost_ids=[], removed_ids=[5], current_frame=211)

        self.assertNotIn("aagam", self.mgr.student_owner)
        self.assertNotIn(5, self.mgr.track_to_student)
        self.assertEqual(self.mgr.get_label(5), "INACTIVE")

    # ------------------------------------------------------------------
    # Rendering Layer Test (isolated)
    # ------------------------------------------------------------------
    def test_render_layer_draws_locked_identity_despite_contradicting_face(self):
        score = FACE_LOCK_THRESHOLD + 0.15
        sec = 0.05

        # Lock Track 5 to aagam
        for f in range(FACE_MIN_CONFIRMATIONS):
            self.mgr.submit_face_match(5, f, "aagam", score, sec)

        # Raw face detector on this frame returns "soham" with high confidence (0.92)
        raw_face_name = "soham"
        raw_face_score = 0.92
        self.mgr.submit_face_match(5, 100, raw_face_name, raw_face_score, 0.10)

        # Authoritative rendering label
        state = self.mgr.get_state(5)
        label = state.locked_name if state.locked else "UNKNOWN"

        self.assertEqual(label, "aagam")

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_track(dummy_frame, (100, 100, 200, 300), 5, label)

        self.assertIsNotNone(annotated)
        self.assertEqual(label, "aagam")
        self.assertNotEqual(label, "soham")
        self.assertNotEqual(label, "UNKNOWN")

    # ------------------------------------------------------------------
    # Summary test
    # ------------------------------------------------------------------
    def test_summary(self):
        score = FACE_LOCK_THRESHOLD + 0.15
        sec = 0.05
        for f in range(FACE_MIN_CONFIRMATIONS):
            self.mgr.submit_face_match(20, f, "aagam", score, sec)
        self.mgr.get_or_create(21, 5)

        summary = self.mgr.summary()
        self.assertEqual(summary["total_tracks"], 2)
        self.assertEqual(summary["locked_tracks"], 1)
        self.assertEqual(summary["unknown_tracks"], 1)
        self.assertIn("aagam", summary["name_assignments"])


class TestRecoveryCandidatePoolScoping(unittest.TestCase):
    """
    Permanent regression tests for recovery candidate pool scoping.
    Guarantees actively-owned students (TRACKED or LOST) can NEVER appear
    in similarity comparison sets or win recovery.
    """

    def setUp(self):
        self.lock_mgr = IdentityLockManager()
        self.known_students = ["aagam", "swayam", "harshil", "soham"]

    def test_actively_owned_lost_student_excluded_from_recovery_candidates(self):
        # 1. Track 10 locks 'swayam'
        self.lock_mgr.get_or_create(10, current_frame=500)
        self.lock_mgr._lock(self.lock_mgr.get_state(10), "swayam", score=0.85, frame=510)
        self.assertEqual(self.lock_mgr.student_owner.get("swayam"), 10)

        # 2. Track 10 transitions to LOST state at frame 1142
        self.lock_mgr.update_lifecycle(tracked_ids=[], lost_ids=[10], removed_ids=[], current_frame=1142)
        state_10 = self.lock_mgr.get_state(10)
        self.assertEqual(state_10.status, "LOST")
        self.assertEqual(self.lock_mgr.student_owner.get("swayam"), 10)

        # 3. Track 8 ('aagam') is permanently REMOVED at frame 1034
        self.lock_mgr.get_or_create(8, current_frame=400)
        self.lock_mgr._lock(self.lock_mgr.get_state(8), "aagam", score=0.88, frame=410)
        self.lock_mgr.update_lifecycle(tracked_ids=[], lost_ids=[], removed_ids=[8], current_frame=1034)
        self.assertNotIn("aagam", self.lock_mgr.student_owner)

        # 4. Candidate pool at frame 1230:
        candidate_students = set(self.known_students) - set(self.lock_mgr.student_owner.keys())
        self.assertNotIn("swayam", candidate_students)
        self.assertIn("aagam", candidate_students)

    def test_candidate_pool_matches_unowned_students_exactly(self):
        self.lock_mgr.get_or_create(1, 0)
        self.lock_mgr._lock(self.lock_mgr.get_state(1), "harshil", 0.9, 5)
        self.lock_mgr.get_or_create(2, 0)
        self.lock_mgr._lock(self.lock_mgr.get_state(2), "swayam", 0.9, 5)

        candidates = set(self.known_students) - set(self.lock_mgr.student_owner.keys())
        expected = {"aagam", "soham"}
        self.assertEqual(candidates, expected)

    def test_owned_student_never_appears_in_similarity_results(self):
        from features import BodyGallery

        dim = 16
        gallery = BodyGallery(dim=dim)

        v_aagam = np.zeros(dim, dtype=np.float32); v_aagam[0] = 1.0
        v_swayam = np.zeros(dim, dtype=np.float32); v_swayam[1] = 1.0
        v_harshil = np.zeros(dim, dtype=np.float32); v_harshil[2] = 1.0
        v_soham = np.zeros(dim, dtype=np.float32); v_soham[3] = 1.0

        gallery.add("aagam", v_aagam)
        gallery.add("swayam", v_swayam)
        gallery.add("harshil", v_harshil)
        gallery.add("soham", v_soham)

        # 2 actively owned students: 'swayam' (Track 10) and 'harshil' (Track 12)
        self.lock_mgr.get_or_create(10, 0)
        self.lock_mgr._lock(self.lock_mgr.get_state(10), "swayam", 0.9, 1)
        self.lock_mgr.get_or_create(12, 0)
        self.lock_mgr._lock(self.lock_mgr.get_state(12), "harshil", 0.9, 1)

        candidate_students = set(self.known_students) - set(self.lock_mgr.student_owner.keys())
        self.assertEqual(candidate_students, {"aagam", "soham"})

        query = np.zeros(dim, dtype=np.float32)
        query[1] = 0.99
        query[0] = 0.10
        query = query / np.linalg.norm(query)

        # 1. Search full scoped candidates
        results = gallery.search(query, candidate_students=candidate_students, top_k=3)
        candidate_names = [name for name, score in results]

        self.assertNotIn("swayam", candidate_names,
                         "CRITICAL: Actively-owned student 'swayam' leaked into search candidate results!")
        self.assertNotIn("harshil", candidate_names,
                         "CRITICAL: Actively-owned student 'harshil' leaked into search candidate results!")

        # 2. top_match assertion
        top_name, top_score, sec_score = gallery.top_match(query, candidate_students=candidate_students)
        self.assertNotEqual(top_name, "swayam")
        self.assertEqual(top_name, "aagam")

    def test_empty_candidate_pool_returns_none_immediately(self):
        from features import BodyGallery

        dim = 16
        gallery = BodyGallery(dim=dim)
        v = np.zeros(dim, dtype=np.float32); v[0] = 1.0
        gallery.add("aagam", v)

        candidate_students = set()
        top_name, top_score, sec_score = gallery.top_match(v, candidate_students=candidate_students)
        self.assertIsNone(top_name)
        self.assertEqual(top_score, 0.0)
        self.assertEqual(sec_score, 0.0)

        results = gallery.search(v, candidate_students=candidate_students)
        self.assertEqual(results, [])

    def test_acquire_ownership_releases_prior_student_assigned_to_same_track(self):
        """
        Verify that if Track X previously acquired student A (e.g. unconfirmed Re-ID),
        and subsequently acquires student B (e.g. ArcFace lock), student A is cleanly released
        and bijective invariants hold with zero violations.
        """
        mgr = IdentityLockManager()
        # Step 1: Track 32 acquires harshil (e.g. Re-ID recovery)
        mgr.get_or_create(32, 100)
        mgr._acquire_ownership(32, "harshil", frame=100)
        self.assertEqual(mgr.student_owner.get("harshil"), 32)
        self.assertEqual(mgr.track_to_student.get(32), "harshil")
        self.assertEqual(mgr.validate_ownership(), [])

        # Step 2: Track 32 locks to swayam (ArcFace confirmation)
        mgr._lock(mgr.get_state(32), "swayam", score=0.60, frame=200)
        self.assertEqual(mgr.student_owner.get("swayam"), 32)
        self.assertEqual(mgr.track_to_student.get(32), "swayam")
        self.assertNotIn("harshil", mgr.student_owner, "harshil must be released back to unowned pool")
        self.assertEqual(mgr.validate_ownership(), [], "Ownership invariants must have 0 violations")

    def test_escalation_trigger_represents_one_real_second_across_fps(self):
        """
        Verify:
        - 30 FPS -> 30 frames
        - 12 FPS -> 12 frames
        And verify Option B escalation persistence across time.
        """
        from robust_tracker import calculate_escalation_frames, RobustTracker

        # Formula check
        self.assertEqual(calculate_escalation_frames(30.0), 30)
        self.assertEqual(calculate_escalation_frames(12.0), 12)
        self.assertEqual(calculate_escalation_frames(25.0), 25)
        self.assertEqual(calculate_escalation_frames(29.97), 30)

        # 30 FPS Tracker verification
        tracker_30 = RobustTracker({}, None, None, None, None, fps=30.0)
        self.assertEqual(tracker_30.unknown_escalation_frames, 30)
        self.assertFalse(tracker_30.should_escalate_to_640(tid=1, frame_idx=0, first_seen_frame=0, is_locked=False))
        self.assertFalse(tracker_30.should_escalate_to_640(tid=1, frame_idx=29, first_seen_frame=0, is_locked=False))
        # Frame 30 (1.0 second elapsed): triggers 640 escalation
        self.assertTrue(tracker_30.should_escalate_to_640(tid=1, frame_idx=30, first_seen_frame=0, is_locked=False))
        # Option B: Frame 31 returns immediately to 320
        self.assertFalse(tracker_30.should_escalate_to_640(tid=1, frame_idx=31, first_seen_frame=0, is_locked=False))
        # Frame 59 still returns False (< 30 frames since last escalation)
        self.assertFalse(tracker_30.should_escalate_to_640(tid=1, frame_idx=59, first_seen_frame=0, is_locked=False))
        # Frame 60 (another 1.0 second of continued UNKNOWN): re-escalates
        self.assertTrue(tracker_30.should_escalate_to_640(tid=1, frame_idx=60, first_seen_frame=0, is_locked=False))

        # 12 FPS Tracker verification
        tracker_12 = RobustTracker({}, None, None, None, None, fps=12.0)
        self.assertEqual(tracker_12.unknown_escalation_frames, 12)
        self.assertFalse(tracker_12.should_escalate_to_640(tid=2, frame_idx=0, first_seen_frame=0, is_locked=False))
        self.assertFalse(tracker_12.should_escalate_to_640(tid=2, frame_idx=11, first_seen_frame=0, is_locked=False))
        # Frame 12 (1.0 second elapsed at 12 FPS): triggers 640 escalation
        self.assertTrue(tracker_12.should_escalate_to_640(tid=2, frame_idx=12, first_seen_frame=0, is_locked=False))
        # Option B: Frame 13 returns immediately to 320
        self.assertFalse(tracker_12.should_escalate_to_640(tid=2, frame_idx=13, first_seen_frame=0, is_locked=False))
        # Frame 24 (another 1.0 second of continued UNKNOWN): re-escalates
        self.assertTrue(tracker_12.should_escalate_to_640(tid=2, frame_idx=24, first_seen_frame=0, is_locked=False))

        # Locked track: never escalates to 640 (always returns to 320)
        self.assertFalse(tracker_30.should_escalate_to_640(tid=1, frame_idx=90, first_seen_frame=0, is_locked=True))


if __name__ == "__main__":
    print("Running IdentityLockManager unit tests...\n")
    unittest.main(verbosity=2)
