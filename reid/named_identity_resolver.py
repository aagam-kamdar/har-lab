"""
reid/named_identity_resolver.py
--------------------------------
Named Student Identity Resolver.

Resolves ByteTrack temporary track_ids into persistent student names:
    Aagam / Swayam / Harshil / Soham

Key differences from the original identity_resolver.py:
  1. Works with named enrolled students (not auto-numbered Student 01, 02...).
  2. Does NOT create new identities during tracking. Only the 4 enrolled students exist.
  3. Uses multi-signal identity scoring: body Re-ID + face similarity when available.
  4. UNKNOWN protection: strict threshold, ambiguity margin, temporal voting.
  5. Adaptive gallery updates when confident.
  6. Tracks detailed metrics: gallery updates, face-assisted matches, body-only matches, etc.
  7. Full identity switch detection for evaluation report.

UNKNOWN behavior:
  - If body Re-ID score < similarity_thresh → UNKNOWN.
  - If top-1 vs top-2 margin < ambiguity_margin → UNKNOWN.
  - If not enough consistent observations → None (pending).
  - UNKNOWN is NOT a new student. It means identity is uncertain.
"""

import logging
from collections import defaultdict, Counter, deque
from typing import Dict, List, Optional, Tuple
import numpy as np

from .enrolled_gallery import EnrolledGallery

logger = logging.getLogger(__name__)


class NamedIdentityResolver:
    """
    Maps ByteTrack track_ids to persistent named student identities.
    Uses multi-signal scoring: body Re-ID (primary) + face matching (fallback/bonus).
    """

    def __init__(
        self,
        gallery: EnrolledGallery,
        body_extractor,
        face_embedder=None,
        similarity_thresh: float = 0.78,
        ambiguity_margin: float = 0.05,
        min_observations: int = 4,
        fps: float = 30.0,
        adaptive_update: bool = True,
        adaptive_confidence_thresh: float = 0.90,
    ):
        self.gallery = gallery
        self.body_extractor = body_extractor
        self.face_embedder = face_embedder
        self.similarity_thresh = similarity_thresh
        self.ambiguity_margin = ambiguity_margin
        self.min_observations = min_observations
        self.fps = fps
        self.adaptive_update = adaptive_update
        self.adaptive_confidence_thresh = adaptive_confidence_thresh

        # track_id -> confirmed student_name
        self.track_to_student: Dict[int, str] = {}

        # student_name -> list of track_ids assigned (for timeline)
        self.student_to_tracks: Dict[str, List[int]] = defaultdict(list)

        # Temporal voting buffer: track_id -> deque of (candidate_name, body_score, face_score)
        self.track_observations: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))

        # Last seen info: student_name -> (last_frame, last_track_id)
        self.student_last_seen: Dict[str, Tuple[int, int]] = {}

        # ---- METRICS ----
        self.total_reid_attempts: int = 0
        self.body_reid_matches: int = 0
        self.face_assisted_matches: int = 0
        self.unknown_periods: int = 0
        self.gallery_updates: int = 0
        self.recoveries: int = 0
        self.identity_switches: int = 0  # times a track changed student assignment

        # Similarity tracking for report
        self.successful_similarities: List[float] = []

        # Recovery log
        self.recovery_log: List[Dict] = []

        # Identity switch log
        self.switch_log: List[Dict] = []

        # Timeline: list of (frame_idx, track_id, student_name)
        self.identity_timeline: List[Tuple[int, int, str]] = []

        # Store last confirmed student for each track to detect switches
        self._prev_student: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Main Resolution Entry Point
    # ------------------------------------------------------------------

    def resolve_track(
        self,
        track_id: int,
        crop: np.ndarray,
        frame_idx: int,
        face_crop: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Resolve the persistent student_name for a ByteTrack track_id.

        If already confirmed: returns immediately.
        If new/unconfirmed: runs body Re-ID + optional face matching + temporal voting.
        Returns:
          - student name ("Aagam", "Swayam", etc.) if confirmed.
          - "UNKNOWN" if seen enough times but no match.
          - None if not yet enough observations.
        """
        # Fast path: already confirmed
        if track_id in self.track_to_student:
            sname = self.track_to_student[track_id]
            self.student_last_seen[sname] = (frame_idx, track_id)
            self.identity_timeline.append((frame_idx, track_id, sname))
            return sname

        if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return None

        self.total_reid_attempts += 1

        # 1. Body embedding
        body_emb = self.body_extractor.extract(crop)

        # 2. Body Re-ID search
        body_results = self.gallery.search_body(body_emb, top_k=min(8, self.gallery.faiss_index.ntotal or 1))
        body_candidate, body_score, is_ambiguous = self._evaluate_body_results(body_results)

        # 3. Face matching (optional bonus signal)
        face_candidate = None
        face_score = 0.0
        if self.face_embedder is not None and face_crop is not None:
            face_emb = self.face_embedder.extract_from_person_crop(face_crop)
            if face_emb is not None:
                face_result = self.gallery.search_face(face_emb, threshold=0.65)
                if face_result is not None:
                    face_candidate, face_score = face_result

        # 4. Multi-signal scoring
        final_candidate, final_score = self._combine_signals(
            body_candidate, body_score, is_ambiguous,
            face_candidate, face_score,
        )

        # 5. Accumulate temporal observations
        self.track_observations[track_id].append((final_candidate, final_score, body_emb, crop))

        obs = list(self.track_observations[track_id])
        if len(obs) < self.min_observations:
            return None  # Not enough observations yet

        # 6. Temporal voting
        confirmed_name = self._temporal_vote(track_id, obs, frame_idx, final_score, body_emb, crop, face_candidate)
        return confirmed_name

    def _evaluate_body_results(self, results: List[Dict]) -> Tuple[Optional[str], float, bool]:
        """
        Evaluate body Re-ID search results.
        Returns (candidate_name, score, is_ambiguous).
        """
        if not results:
            return None, 0.0, False

        top = results[0]
        top_name = top["student_name"]
        top_score = top["score"]

        if top_score < self.similarity_thresh:
            return None, top_score, False

        # Check ambiguity: is the top-1 clearly better than top-2 from a different student?
        for other in results[1:]:
            if other["student_name"] != top_name:
                margin = top_score - other["score"]
                if margin < self.ambiguity_margin:
                    return None, top_score, True  # Ambiguous
                break

        return top_name, top_score, False

    def _combine_signals(
        self,
        body_candidate: Optional[str],
        body_score: float,
        is_ambiguous: bool,
        face_candidate: Optional[str],
        face_score: float,
    ) -> Tuple[Optional[str], float]:
        """
        Combine body Re-ID and face matching signals.
        Body Re-ID is primary. Face is a bonus/tiebreaker.
        """
        # If body is clear and face agrees → boost confidence
        if body_candidate is not None and not is_ambiguous:
            if face_candidate is not None and face_candidate == body_candidate:
                # Signals agree → slightly boost score
                combined_score = min(1.0, body_score * 0.85 + face_score * 0.15)
                return body_candidate, combined_score
            # Body alone is sufficient
            return body_candidate, body_score

        # If body is ambiguous but face is confident → use face
        if is_ambiguous and face_candidate is not None and face_score >= 0.75:
            return face_candidate, face_score * 0.7  # Downscale since body didn't agree

        # If body failed but face is very confident
        if body_candidate is None and face_candidate is not None and face_score >= 0.80:
            return face_candidate, face_score * 0.65  # Face-only (weaker)

        return None, body_score

    def _temporal_vote(
        self,
        track_id: int,
        obs: list,
        frame_idx: int,
        latest_score: float,
        latest_body_emb: np.ndarray,
        latest_crop: np.ndarray,
        face_candidate: Optional[str],
    ) -> Optional[str]:
        """
        Vote over temporal observations to confirm an identity.
        Returns the confirmed student name, "UNKNOWN", or None if voting inconclusive.
        """
        # Collect valid (non-None) votes
        valid_votes = [(name, score) for name, score, _, _ in obs if name is not None]
        vote_counter = Counter(name for name, _ in valid_votes)

        if not vote_counter:
            self.unknown_periods += 1
            return "UNKNOWN"

        best_name, best_count = vote_counter.most_common(1)[0]
        majority_needed = (self.min_observations // 2) + 1

        if best_count >= majority_needed:
            # Confirmed!
            avg_score = np.mean([s for n, s in valid_votes if n == best_name])

            # Check for identity switch
            if track_id in self._prev_student and self._prev_student[track_id] != best_name:
                self.identity_switches += 1
                self.switch_log.append({
                    "track_id": track_id,
                    "frame": frame_idx,
                    "from": self._prev_student[track_id],
                    "to": best_name,
                })

            self._prev_student[track_id] = best_name
            self.track_to_student[track_id] = best_name
            self.student_to_tracks[best_name].append(track_id)
            self.successful_similarities.append(float(avg_score))

            # Determine if face assisted
            if face_candidate == best_name:
                self.face_assisted_matches += 1
            else:
                self.body_reid_matches += 1

            # Check if this is a recovery of a student seen before on a different track
            if best_name in self.student_last_seen:
                old_frame, old_track = self.student_last_seen[best_name]
                if old_track != track_id:
                    gap_frames = frame_idx - old_frame
                    self.recoveries += 1
                    self.recovery_log.append({
                        "event": "RECOVERY",
                        "student": best_name,
                        "old_track": old_track,
                        "new_track": track_id,
                        "gap_frames": gap_frames,
                        "gap_sec": round(gap_frames / self.fps, 2),
                        "score": round(float(avg_score), 3),
                    })
                    logger.info(
                        "RECOVERY: Track %d -> %s (was Track %d, gap=%d frames, score=%.3f)",
                        track_id, best_name, old_track, gap_frames, avg_score
                    )

            self.student_last_seen[best_name] = (frame_idx, track_id)
            self.identity_timeline.append((frame_idx, track_id, best_name))

            # Adaptive gallery update
            if self.adaptive_update and avg_score >= self.adaptive_confidence_thresh:
                updated = self.gallery.adaptive_update(
                    best_name, latest_crop, latest_body_emb,
                    avg_score, self.adaptive_confidence_thresh
                )
                if updated:
                    self.gallery_updates += 1
                    logger.debug("Gallery updated for %s (score=%.3f)", best_name, avg_score)

            logger.debug(
                "Track %d -> %s confirmed (score=%.3f, votes=%d/%d)",
                track_id, best_name, avg_score, best_count, len(obs)
            )
            return best_name

        # Not yet majority
        return None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_student_name(self, track_id: int) -> Optional[str]:
        """Return confirmed student name for track_id, or None."""
        return self.track_to_student.get(track_id)

    # ------------------------------------------------------------------
    # Metrics and Reporting
    # ------------------------------------------------------------------

    def finalize(self):
        """Compute final metrics."""
        pass

    def get_metrics(self) -> Dict:
        sims = self.successful_similarities
        return {
            "total_reid_attempts": self.total_reid_attempts,
            "body_reid_matches": self.body_reid_matches,
            "face_assisted_matches": self.face_assisted_matches,
            "total_successful_matches": self.body_reid_matches + self.face_assisted_matches,
            "unknown_periods": self.unknown_periods,
            "gallery_updates": self.gallery_updates,
            "recoveries": self.recoveries,
            "identity_switches": self.identity_switches,
            "total_bytetrack_ids": len(set(
                tid for tid in self.track_observations
            )),
            "enrolled_students_tracked": sorted(set(self.track_to_student.values()) - {"UNKNOWN"}),
            "avg_similarity": round(float(np.mean(sims)), 3) if sims else 0.0,
            "min_similarity": round(float(np.min(sims)), 3) if sims else 0.0,
            "max_similarity": round(float(np.max(sims)), 3) if sims else 0.0,
        }

    def get_timeline(self) -> List[Dict]:
        """Return list of identity timeline events."""
        result = []
        for frame, tid, sname in self.identity_timeline:
            result.append({"frame": frame, "track_id": tid, "student": sname})
        return result
