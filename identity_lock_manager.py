"""
identity_lock_manager.py
------------------------
Strict Identity Locking Policy with ByteTrack Track Lifecycle (TRACKED / LOST / REMOVED).

LIFECYCLE RULES:
  1. TRACKED:
     - Track is actively visible / tracked by ByteTrack.
     - New track starts UNKNOWN until face-confirmed (FACE_MIN_CONFIRMATIONS).
     - Once locked, identity is 100% sticky.
  2. LOST:
     - Track is temporarily occluded/undetected, kept in ByteTrack's lost buffer.
     - Keeps locked identity and ownership in student_owner.
     - NEVER triggers a RELEASE while LOST.
  3. TRACKED AGAIN (recovered with same track_id):
     - Continues same locked identity seamlessly.
  4. REMOVED:
     - ByteTrack permanently removes the track (exceeded max_time_lost).
     - ONLY at this point is ownership RELEASED and the track marked inactive.
     - student_owner entry is freed so another track can acquire the identity if needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set

logger = logging.getLogger(__name__)

# Constants / thresholds
FACE_LOCK_THRESHOLD         = 0.38   # Min ArcFace cosine similarity to lock
FACE_AMBIGUITY_MARGIN       = 0.06   # Top-2 must differ by at least this much
FACE_MIN_CONFIRMATIONS      = 2      # Consecutive strong face matches to lock (new track)
# Body Re-ID thresholds — calibrated for MobileNetV3-Small + HSV (672-dim):
#   Benchmark: threshold=0.78, margin=0.06
BODY_RECOVERY_THRESHOLD     = 0.78   # Body FAISS cosine similarity for gap-recovery (MobileNetV3+HSV)
BODY_RECOVERY_AMBIGUITY     = 0.06   # Gap-recovery top-2 margin (MobileNetV3+HSV)
BODY_GALLERY_MAX            = 80     # Max body embeddings per student
BODY_GALLERY_DEDUP_THRESH   = 0.97   # Near-duplicate removal threshold
FACE_FRAME_GAP_FOR_RECOVERY = 90     # Frames a track must be lost before body recovery applies

# Decision constants
DECISION_HOLD = "HOLD"
DECISION_CONTRADICTION_IGNORED = "CONTRADICTION_IGNORED"


@dataclass
class TrackState:
    """Per-track identity state with ByteTrack lifecycle tracking."""
    track_id: int

    # -- Lock state --
    locked: bool = False
    locked_name: Optional[str] = None
    locked_face_score: float = 0.0
    locked_at_frame: int = -1

    # -- Pending face evidence (for initial locking) --
    pending_name: Optional[str] = None
    pending_score: float = 0.0
    pending_count: int = 0

    # -- Lifecycle state: "TRACKED", "LOST", "REMOVED" --
    status: str = "TRACKED"
    is_active: bool = True

    # -- Contradiction tracking --
    contradiction_count: int = 0

    # -- Frame tracking --
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    last_face_frame: int = -1
    frames_since_face: int = 0

    # -- Body evidence (supporting only) --
    body_evidence: List[Tuple[str, float]] = field(default_factory=list)

    # -- Statistics --
    face_matches: int = 0
    body_matches: int = 0
    recovery_count: int = 0
    frames_total: int = 0

    def identity_label(self) -> str:
        return self.locked_name if self.locked else "UNKNOWN"


class IdentityLockManager:
    """
    Manages per-track identity locking with ByteTrack lifecycle integration.
    """

    def __init__(self, require_dual_agreement: bool = False):
        self.require_dual_agreement = require_dual_agreement
        self._tracks: Dict[int, TrackState] = {}

        # Ownership maps
        self.student_owner: Dict[str, int] = {}     # student -> owning track_id
        self.track_to_student: Dict[int, str] = {}  # track_id -> student name

        # Track status sets
        self._tracked_ids: Set[int] = set()
        self._lost_ids: Set[int] = set()
        self._removed_ids: Set[int] = set()

    # ------------------------------------------------------------------ #
    #  ByteTrack lifecycle integration                                   #
    # ------------------------------------------------------------------ #

    def update_lifecycle(
        self,
        tracked_ids: List[int],
        lost_ids: List[int],
        removed_ids: List[int],
        current_frame: int,
    ):
        """
        Update track lifecycle status based on ByteTrack internal lists.
        - tracked_ids: currently active / detected tracks -> status="TRACKED"
        - lost_ids: temporarily lost tracks (in buffer) -> status="LOST" (KEEPS OWNERSHIP)
        - removed_ids: permanently removed tracks -> status="REMOVED" (TRIGGERS RELEASE)
        """
        self._tracked_ids = set(tracked_ids)
        self._lost_ids = set(lost_ids)

        # 1. Update active/tracked tracks
        for tid in self._tracked_ids:
            if tid in self._tracks:
                state = self._tracks[tid]
                state.status = "TRACKED"
                state.is_active = True

        # 2. Update lost tracks (keep ownership and locked identity)
        for tid in self._lost_ids:
            if tid in self._tracks:
                state = self._tracks[tid]
                state.status = "LOST"
                state.is_active = True  # still alive in tracker buffer

        # 3. Handle newly removed tracks -> ONLY these trigger RELEASE
        for tid in removed_ids:
            if tid not in self._removed_ids:
                self._removed_ids.add(tid)
                if tid in self._tracks:
                    state = self._tracks[tid]
                    state.status = "REMOVED"
                    state.is_active = False

                student = self.track_to_student.get(tid)
                if student is not None:
                    self._release_ownership(tid, student, reason="track_removed", frame=current_frame)

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def get_or_create(self, track_id: int, current_frame: int) -> TrackState:
        if track_id not in self._tracks:
            state = TrackState(
                track_id=track_id,
                first_seen_frame=current_frame,
                last_seen_frame=current_frame,
                status="TRACKED",
                is_active=True,
            )
            self._tracks[track_id] = state
            self._tracked_ids.add(track_id)
            logger.debug("Track %d created at frame %d", track_id, current_frame)
        return self._tracks[track_id]

    def update_seen(self, track_id: int, current_frame: int) -> TrackState:
        state = self.get_or_create(track_id, current_frame)
        state.last_seen_frame = current_frame
        state.frames_total += 1
        state.frames_since_face += 1
        state.status = "TRACKED"
        state.is_active = True
        return state

    def submit_face_match(
        self,
        track_id: int,
        current_frame: int,
        name: str,
        score: float,
        second_score: float = 0.0,
        name_640: Optional[str] = None,
        score_640: Optional[float] = None,
        second_score_640: float = 0.0,
    ) -> str:
        """
        Submit a face recognition result.
        Returns the authoritative identity label (permanently sticky once locked).
        Requires dual-resolution agreement (both 320 and 640 >= 0.38 on same student)
        before finalizing any initial lock.
        """
        state = self.get_or_create(track_id, current_frame)
        state.last_face_frame = current_frame
        state.frames_since_face = 0

        # ======================================================
        # ALREADY LOCKED: PERMANENTLY STICKY — NO SWITCHES EVER
        # ======================================================
        if state.locked:
            if score >= FACE_LOCK_THRESHOLD and (score - second_score) >= FACE_AMBIGUITY_MARGIN:
                if state.locked_name == name:
                    # Confirms lock
                    if score > state.locked_face_score:
                        state.locked_face_score = score
                    state.face_matches += 1
                    logger.debug("Track %d face confirms locked %s (%.3f)", track_id, name, score)
                else:
                    # Contradicts lock -> LOG AND IGNORE
                    state.contradiction_count += 1
                    logger.info(
                        "DECISION=%s | track=%d | locked=%s | candidate=%s | score=%.3f | threshold=%.2f | frame=%d",
                        DECISION_CONTRADICTION_IGNORED, track_id, state.locked_name, name, score, FACE_LOCK_THRESHOLD, current_frame
                    )
            else:
                logger.debug("Track %d face weak/ambiguous: %s=%.3f -> HOLD", track_id, name, score)

            return state.locked_name

        # ======================================================
        # NOT YET LOCKED: ACCUMULATE CONFIRMATIONS
        # ======================================================
        if score < FACE_LOCK_THRESHOLD:
            logger.debug("Track %d face weak: %s=%.3f (threshold=%.2f) -> HOLD", track_id, name, score, FACE_LOCK_THRESHOLD)
            return "UNKNOWN"

        margin = score - second_score
        if margin < FACE_AMBIGUITY_MARGIN:
            logger.debug("Track %d face ambiguous: %s=%.3f margin=%.3f -> HOLD", track_id, name, score, margin)
            return "UNKNOWN"

        # Dual-resolution agreement requirement before finalizing initial lock
        dual_check_active = self.require_dual_agreement or (score_640 is not None or name_640 is not None)
        if dual_check_active:
            if score_640 is None or score_640 < FACE_LOCK_THRESHOLD:
                logger.debug(
                    "Track %d initial lock rejected: 640 score (%s) < threshold (%.2f)",
                    track_id, f"{score_640:.3f}" if score_640 is not None else "None", FACE_LOCK_THRESHOLD
                )
                return "UNKNOWN"
            if name_640 != name:
                logger.debug(
                    "Track %d initial lock rejected: resolution disagreement (320=%s, 640=%s)",
                    track_id, name, name_640
                )
                return "UNKNOWN"

        state.face_matches += 1

        # Check if candidate name is already owned by another active/lost track
        existing_owner = self.student_owner.get(name)
        if existing_owner is not None and existing_owner != track_id:
            owner_state = self._tracks.get(existing_owner)
            if owner_state and owner_state.is_active and owner_state.status != "REMOVED":
                logger.debug("Track %d cannot lock %s: already owned by %s Track %d", track_id, name, owner_state.status, existing_owner)
                state.pending_name = None
                state.pending_count = 0
                return "UNKNOWN"

        if state.pending_name == name:
            state.pending_score = max(state.pending_score, score)
            state.pending_count += 1
        else:
            state.pending_name = name
            state.pending_score = score
            state.pending_count = 1

        if state.pending_count >= FACE_MIN_CONFIRMATIONS:
            self._lock(state, name, state.pending_score, current_frame)

        return state.identity_label()

    def submit_body_evidence(
        self,
        track_id: int,
        current_frame: int,
        name: str,
        score: float,
    ) -> None:
        state = self.get_or_create(track_id, current_frame)
        state.body_evidence.append((name, score))
        state.body_matches += 1
        if len(state.body_evidence) > 30:
            state.body_evidence = state.body_evidence[-30:]

    def attempt_gap_recovery(
        self,
        track_id: int,
        current_frame: int,
        body_name: str,
        body_score: float,
        body_second_score: float = 0.0,
    ) -> Optional[str]:
        state = self.get_or_create(track_id, current_frame)
        frames_lost = current_frame - state.last_seen_frame

        if frames_lost < FACE_FRAME_GAP_FOR_RECOVERY:
            return None

        if not state.locked:
            return None

        if body_score < BODY_RECOVERY_THRESHOLD:
            return None
        if (body_score - body_second_score) < BODY_RECOVERY_AMBIGUITY:
            return None

        if body_name == state.locked_name:
            logger.info(
                "Track %d: gap recovery confirmed %s (gap=%d frames, body=%.3f)",
                track_id, body_name, frames_lost, body_score,
            )
            state.recovery_count += 1
            return body_name
        else:
            return state.locked_name

    def get_label(self, track_id: int) -> str:
        if track_id in self._tracks:
            state = self._tracks[track_id]
            if state.status == "REMOVED" or not state.is_active:
                return "INACTIVE"
            return state.identity_label()
        return "UNKNOWN"

    def get_state(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(track_id)

    def all_states(self) -> Dict[int, TrackState]:
        return dict(self._tracks)

    def ownership_snapshot(self) -> Dict:
        return {
            "student_owner": dict(self.student_owner),
            "track_to_student": dict(self.track_to_student),
        }

    def validate_ownership(self) -> List[str]:
        violations = []
        for student, tid in self.student_owner.items():
            reverse = self.track_to_student.get(tid)
            if reverse != student:
                violations.append(
                    f"student_owner[{student}]={tid} but track_to_student[{tid}]={reverse}"
                )
        for tid, student in self.track_to_student.items():
            reverse = self.student_owner.get(student)
            if reverse != tid:
                violations.append(
                    f"track_to_student[{tid}]={student} but student_owner[{student}]={reverse}"
                )
        return violations

    def summary(self) -> Dict:
        locked = {tid: s for tid, s in self._tracks.items() if s.locked}
        unknown = {tid: s for tid, s in self._tracks.items() if not s.locked}
        return {
            "total_tracks": len(self._tracks),
            "locked_tracks": len(locked),
            "unknown_tracks": len(unknown),
            "name_assignments": {
                name: [tid for tid, s in locked.items() if s.locked_name == name]
                for name in set(s.locked_name for s in locked.values())
            },
            "ownership": self.ownership_snapshot(),
        }

    # ------------------------------------------------------------------ #
    #  Ownership helpers                                                 #
    # ------------------------------------------------------------------ #

    def _acquire_ownership(self, track_id: int, student: str, frame: int):
        prior_owner = self.student_owner.get(student)
        if prior_owner is not None and prior_owner != track_id:
            self._release_ownership(prior_owner, student, reason="superseded_by_lock", frame=frame)

        # Ensure track_id does not leave a dangling reference to a previously owned student
        prior_student = self.track_to_student.get(track_id)
        if prior_student is not None and prior_student != student:
            self._release_ownership(track_id, prior_student, reason="superseded_by_lock", frame=frame)

        self.student_owner[student] = track_id
        self.track_to_student[track_id] = student

    def _release_ownership(self, track_id: int, student: str, reason: str, frame: int):
        logger.info(
            "RELEASE:\n"
            "  student=%s | track=%d | reason=%s | frame=%d",
            student, track_id, reason, frame,
        )
        if self.student_owner.get(student) == track_id:
            self.student_owner.pop(student, None)
        self.track_to_student.pop(track_id, None)

    # ------------------------------------------------------------------ #
    #  Lock helpers                                                      #
    # ------------------------------------------------------------------ #

    def _lock(self, state: TrackState, name: str, score: float, frame: int):
        state.locked = True
        state.locked_name = name
        state.locked_face_score = score
        state.locked_at_frame = frame
        state.pending_name = None
        state.pending_count = 0

        # Acquire ownership
        self._acquire_ownership(state.track_id, name, frame)

        logger.info(
            "Track %d LOCKED -> %s (face_score=%.3f, frame=%d)",
            state.track_id, name, score, frame,
        )
