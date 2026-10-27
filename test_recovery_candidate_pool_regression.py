"""
test_recovery_candidate_pool_regression.py
------------------------------------------
Regression test suite for recovery candidate pool scoping.

Scenario Tested (EVT_BT_3 Near-Miss Audit):
- Student A ('swayam') is locked to Track 10 and is currently in LOST state (age < track_buffer, unevicted).
  Therefore, Student A is actively owned in student_owner.
- Student B ('aagam') was previously locked to Track 8, but Track 8 was REMOVED (evicted).
  Therefore, Student B is genuinely unowned/missing.
- New Track 20 appears.
- If Track 20's body embedding scores HIGHER for actively-owned Student A than missing Student B,
  the recovery candidate search MUST NEVER return Student A in the candidate evaluation pool.
  It must strictly restrict candidates to genuinely unowned/missing students.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identity_lock_manager import IdentityLockManager
from features import BodyGallery


def get_recovery_candidate_pool(lock_mgr: IdentityLockManager, all_known_students: list) -> set:
    """
    Candidate pool specification for safe Re-ID recovery:
    Only students who are NOT currently owned by an active or LOST track
    can be included in the recovery candidate pool.
    """
    currently_owned = set(lock_mgr.student_owner.keys())
    return set(all_known_students) - currently_owned


class TestRecoveryCandidatePoolScoping(unittest.TestCase):

    def setUp(self):
        self.lock_mgr = IdentityLockManager()
        self.known_students = ["aagam", "swayam", "harshil", "soham"]

    def test_actively_owned_lost_student_excluded_from_recovery_candidates(self):
        """
        Verify that a student whose track is LOST (not yet REMOVED) is strictly
        excluded from the recovery candidate pool, preventing duplicate ownership collisions.
        """
        # 1. Establish initial ownership: Track 10 locks 'swayam'
        self.lock_mgr.get_or_create(10, current_frame=500)
        self.lock_mgr._lock(self.lock_mgr.get_state(10), "swayam", score=0.85, frame=510)
        self.assertEqual(self.lock_mgr.student_owner.get("swayam"), 10)

        # 2. Track 10 transitions to LOST state at frame 1142
        self.lock_mgr.update_lifecycle(tracked_ids=[], lost_ids=[10], removed_ids=[], current_frame=1142)
        state_10 = self.lock_mgr.get_state(10)
        self.assertEqual(state_10.status, "LOST")
        # Ownership MUST remain active while LOST
        self.assertEqual(self.lock_mgr.student_owner.get("swayam"), 10)

        # 3. Student B ('aagam') old Track 8 was permanently REMOVED at frame 1034
        self.lock_mgr.get_or_create(8, current_frame=400)
        self.lock_mgr._lock(self.lock_mgr.get_state(8), "aagam", score=0.88, frame=410)
        self.lock_mgr.update_lifecycle(tracked_ids=[], lost_ids=[], removed_ids=[8], current_frame=1034)
        self.assertNotIn("aagam", self.lock_mgr.student_owner)  # genuinely missing

        # 4. At frame 1230, new Track 20 appears while Track 10 is still LOST
        candidates = get_recovery_candidate_pool(self.lock_mgr, self.known_students)

        # Assertion: Actively-owned Student A ('swayam') MUST NOT appear in candidates
        self.assertNotIn("swayam", candidates, "DEFECT: Actively-owned student 'swayam' appeared in recovery candidate pool while Track 10 is LOST!")
        
        # Assertion: Genuinely missing Student B ('aagam') MUST appear in candidates
        self.assertIn("aagam", candidates, "Genuinely missing student 'aagam' should be available for recovery.")

    def test_candidate_pool_matches_unowned_students_exactly(self):
        """
        Assert that candidate pool generation output set exactly equals
        the set of unowned students (all_known - student_owner.keys()).
        """
        # Lock harshil and swayam
        self.lock_mgr.get_or_create(1, 0)
        self.lock_mgr._lock(self.lock_mgr.get_state(1), "harshil", 0.9, 5)
        self.lock_mgr.get_or_create(2, 0)
        self.lock_mgr._lock(self.lock_mgr.get_state(2), "swayam", 0.9, 5)

    def test_owned_student_never_appears_in_similarity_results(self):
        """
        Construct scenario with 2 locked/active students and 2 unowned students.
        Query top_match() with a body embedding closest to one of the ACTIVELY OWNED students.
        Assert:
          - The returned match is NOT the actively-owned student, even though it has the highest raw similarity.
          - The actively-owned student's score does not appear anywhere in the returned candidate list at all.
        """
        dim = 16
        gallery = BodyGallery(dim=dim)

        # 4 orthogonal base vectors
        np.random.seed(42)
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

        # Unowned students are: 'aagam' and 'soham'
        candidate_students = set(self.known_students) - set(self.lock_mgr.student_owner.keys())
        self.assertEqual(candidate_students, {"aagam", "soham"})

        # Query embedding is almost identical to actively owned 'swayam' (e.g. dot product 0.99 with swayam, 0.1 with aagam)
        query = np.zeros(dim, dtype=np.float32)
        query[1] = 0.99
        query[0] = 0.10
        query = query / np.linalg.norm(query)

        # 1. Search full scoped candidates
        results = gallery.search(query, candidate_students=candidate_students, top_k=3)
        candidate_names = [name for name, score in results]

        # Assert: actively-owned student 'swayam' does NOT appear ANYWHERE in similarity results
        self.assertNotIn("swayam", candidate_names,
                         "CRITICAL: Actively-owned student 'swayam' leaked into search candidate results!")
        self.assertNotIn("harshil", candidate_names,
                         "CRITICAL: Actively-owned student 'harshil' leaked into search candidate results!")

        # 2. top_match assertion
        top_name, top_score, sec_score = gallery.top_match(query, candidate_students=candidate_students)
        self.assertNotEqual(top_name, "swayam",
                            "CRITICAL: top_match returned actively-owned student 'swayam'!")
        self.assertEqual(top_name, "aagam",
                         "top_match should have matched the highest among unowned candidates ('aagam').")

    def test_empty_candidate_pool_returns_none_immediately(self):
        """
        If candidate_students is empty (every student is currently owned),
        top_match() must return no match / None immediately, without running
        any embedding comparison at all.
        """
        dim = 16
        gallery = BodyGallery(dim=dim)
        v = np.zeros(dim, dtype=np.float32); v[0] = 1.0
        gallery.add("aagam", v)

        # All students owned -> candidate pool empty
        candidate_students = set()
        top_name, top_score, sec_score = gallery.top_match(v, candidate_students=candidate_students)
        self.assertIsNone(top_name)
        self.assertEqual(top_score, 0.0)
        self.assertEqual(sec_score, 0.0)

        results = gallery.search(v, candidate_students=candidate_students)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
