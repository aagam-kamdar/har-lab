"""
reid/identity_resolver.py
-------------------------
Core Identity Resolution Engine for Student Lab Productivity Tracker.

Bridges temporary ByteTrack track_id and persistent student_id:
  track_id != student_id
  track_id = temporary tracker identity (e.g. 1..42)
  student_id = persistent student identity (e.g. S01, S02, ...)

Implements:
  1. Triggering Re-ID only when necessary (new/unconfirmed tracks).
  2. FAISS vector search with similarity threshold and ambiguity margin.
  3. Temporal confirmation over multiple consecutive observations.
  4. Long-term identity recovery when a student returns after track buffer removal.
  5. Detailed metric tracking and recovery logs for evaluation.
"""

import logging
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from .embedding_model import BaseReIDModel, PersonReIDExtractor
from .faiss_manager import FAISSManager
from .gallery import StudentGallery

logger = logging.getLogger(__name__)


class IdentityResolver:
    """
    Resolves temporary ByteTrack track_ids into persistent student_ids
    using local FAISS vector search and temporal confirmation.
    """

    def __init__(
        self,
        gallery: StudentGallery,
        extractor: BaseReIDModel = None,
        similarity_thresh: float = 0.80,
        ambiguity_margin: float = 0.04,
        min_observations: int = 3,
        fps: float = 30.0,
    ):
        self.gallery = gallery
        self.extractor = extractor or self.gallery.extractor
        self.similarity_thresh = similarity_thresh
        self.ambiguity_margin = ambiguity_margin
        self.min_observations = min_observations
        self.fps = fps

        # Track -> Student mapping (e.g. {7: "Student 03", 19: "Student 03"})
        self.track_to_student: Dict[int, str] = {}

        # Student -> List of Track IDs
        self.student_to_tracks: Dict[str, List[int]] = defaultdict(list)

        # Temporal observation buffer: track_id -> list of (predicted_student_id, score, crop, emb)
        self.track_observations: Dict[int, List[Tuple[str, float, np.ndarray, np.ndarray]]] = defaultdict(list)

        # Track last frame seen per student_id for gap computation: student_id -> (last_frame, last_track_id)
        self.student_last_seen: Dict[str, Tuple[int, int]] = {}

        # Counter for auto-registering new persistent students
        self._next_student_num = 1

        # Metrics Tracking (Requirement 13)
        self.total_reid_attempts = 0
        self.successful_reid_matches = 0
        self.unknown_identities = 0
        self.short_term_reactivations = 0
        self.long_term_recoveries = 0
        self.unresolved_fragments = 0

        # Detailed Recovery Log (Requirement 14)
        self.recovery_log: List[Dict] = []

    def get_student_id(self, track_id: int) -> Optional[str]:
        """Return the confirmed student_id for a track_id, or None if unconfirmed."""
        return self.track_to_student.get(track_id, None)

    def record_short_term_reactivation(self, track_id: int):
        """Called when ByteTrack reactivates a lost track directly without Re-ID."""
        self.short_term_reactivations += 1

    def resolve_track(
        self,
        track_id: int,
        crop: np.ndarray,
        frame_idx: int,
        allow_new_registration: bool = True,
    ) -> Optional[str]:
        """
        Resolve or confirm the student_id for a given track.
        If track_id is already confirmed: returns student_id immediately without Re-ID.
        If track_id is new/unconfirmed: triggers Re-ID + temporal voting.
        """
        # 1. Already confirmed? No expensive Re-ID needed.
        if track_id in self.track_to_student:
            student_id = self.track_to_student[track_id]
            self.student_last_seen[student_id] = (frame_idx, track_id)
            return student_id

        if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return None

        # 2. Extract local Re-ID embedding
        self.total_reid_attempts += 1
        emb = self.extractor.extract(crop)

        # 3. FAISS nearest-neighbor search
        search_results = self.gallery.faiss_mgr.search(emb, top_k=5)

        candidate_sid = None
        candidate_score = 0.0

        if search_results:
            top_match = search_results[0]
            top_sid = top_match["student_id"]
            top_score = top_match["score"]

            # Ambiguity check: ensure top match is sufficiently distinct from other students
            is_ambiguous = False
            for other in search_results[1:]:
                if other["student_id"] != top_sid:
                    if (top_score - other["score"]) < self.ambiguity_margin:
                        is_ambiguous = True
                    break

            if top_score >= self.similarity_thresh and not is_ambiguous:
                candidate_sid = top_sid
                candidate_score = top_score
            else:
                candidate_sid = "UNKNOWN"
                candidate_score = top_score
        else:
            candidate_sid = "UNKNOWN"
            candidate_score = 0.0

        # 4. Temporal Confirmation (Requirement 7)
        self.track_observations[track_id].append((candidate_sid, candidate_score, crop, emb))
        obs_list = self.track_observations[track_id]

        if len(obs_list) >= self.min_observations:
            # Tally votes from recent observations
            valid_candidates = [sid for sid, score, _, _ in obs_list if sid != "UNKNOWN"]
            counts = Counter(valid_candidates)

            if counts:
                best_sid, vote_count = counts.most_common(1)[0]
                # Require majority consensus
                if vote_count >= (self.min_observations // 2 + 1):
                    # CONFIRMED MATCH to existing student
                    self.track_to_student[track_id] = best_sid
                    self.student_to_tracks[best_sid].append(track_id)
                    self.successful_reid_matches += 1

                    # Check if this was a long-term recovery of an existing student
                    if best_sid in self.student_last_seen:
                        old_last_frame, old_track_id = self.student_last_seen[best_sid]
                        if old_track_id != track_id:
                            gap_frames = frame_idx - old_last_frame
                            gap_sec = gap_frames / self.fps
                            self.long_term_recoveries += 1
                            self.recovery_log.append({
                                "old_track": old_track_id,
                                "new_track": track_id,
                                "student_id": best_sid,
                                "similarity": round(float(candidate_score), 3),
                                "gap_frames": gap_frames,
                                "gap_sec": round(gap_sec, 2),
                                "reason": f"Track buffer expired ({gap_frames} frame gap); recovered via FAISS Re-ID",
                                "result": "RECOVERED",
                            })
                            logger.info(
                                "LONG-TERM RECOVERY: Track %d -> %s (previously Track %d, gap=%.2fs, sim=%.3f)",
                                track_id, best_sid, old_track_id, gap_sec, candidate_score
                            )

                    self.student_last_seen[best_sid] = (frame_idx, track_id)
                    return best_sid

            # No confident match to existing student
            if allow_new_registration:
                # Discover and register a new persistent student identity
                new_sid = f"Student {self._next_student_num:02d}"
                self._next_student_num += 1

                # Gather best 3 crops from observation buffer as reference gallery
                ref_crops = [c for _, _, c, _ in obs_list[:5]]
                ref_embs = [e for _, _, _, e in obs_list[:5]]

                self.gallery.register_student(new_sid, ref_crops, ref_embs)
                self.track_to_student[track_id] = new_sid
                self.student_to_tracks[new_sid].append(track_id)
                self.student_last_seen[new_sid] = (frame_idx, track_id)
                logger.info("REGISTERED NEW IDENTITY: Track %d -> %s", track_id, new_sid)
                return new_sid
            else:
                self.unknown_identities += 1
                return "UNKNOWN"

        # Not yet reached minimum observations
        return None

    def finalize(self):
        """Clean up and compute final unresolved fragment count."""
        total_tracks = len(self.track_observations)
        assigned_tracks = len(self.track_to_student)
        self.unresolved_fragments = max(0, total_tracks - assigned_tracks)

    def get_metrics(self) -> Dict:
        """Return metrics dictionary (Requirement 13)."""
        unique_students = set(self.track_to_student.values())
        return {
            "total_byte_track_ids": len(self.track_observations),
            "total_persistent_student_ids": len(unique_students),
            "total_reid_attempts": self.total_reid_attempts,
            "successful_reid_matches": self.successful_reid_matches,
            "unknown_identities": self.unknown_identities,
            "short_term_reactivations": self.short_term_reactivations,
            "long_term_recoveries": self.long_term_recoveries,
            "unresolved_track_fragments": self.unresolved_fragments,
        }
