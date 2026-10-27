"""
action_event_logger.py
-----------------------
Logs per-student action events for the dashboard.

Tracks:
  - Current action per student per frame
  - Duration of each action bout
  - Alerts (phone > threshold, sleeping > threshold)
  - Productivity score computation
  - Session summary
"""

from __future__ import annotations

import json
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

KNOWN_STUDENTS = ["Aagam", "Swayam", "Harshil", "Soham"]

# Actions that count as "productive"
PRODUCTIVE_ACTIONS = {"writing", "typing", "reading", "sitting", "discuss"}

# Actions that trigger alerts if sustained too long (seconds)
ALERT_THRESHOLDS_SEC = {
    "phone":    120,   # 2 min on phone → alert
    "sleeping": 60,    # 1 min sleeping → alert
    "discuss":  180,   # 3 min discussing → alert
}

# FPS assumption for duration calculation
DEFAULT_FPS = 30.0


# --------------------------------------------------------------------------- #
#  Data structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class ActionBout:
    student: str
    action: str
    start_frame: int
    end_frame: int
    start_time: float     # seconds
    end_time: float       # seconds

    @property
    def duration_sec(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        return {
            "student": self.student,
            "action": self.action,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration_sec": round(self.duration_sec, 2),
        }


@dataclass
class AlertEvent:
    student: str
    action: str
    duration_sec: float
    frame: int
    timestamp: float      # wall clock
    acknowledged: bool = False

    def to_dict(self) -> Dict:
        return {
            "student": self.student,
            "action": self.action,
            "duration_sec": round(self.duration_sec, 1),
            "frame": self.frame,
            "timestamp": round(self.timestamp, 1),
            "acknowledged": self.acknowledged,
        }


# --------------------------------------------------------------------------- #
#  Main Logger                                                                  #
# --------------------------------------------------------------------------- #

class ActionEventLogger:
    """
    Thread-safe logger for action events.

    Call update(student, action, frame_idx, fps) every frame.
    Query live_status() for dashboard SSE streaming.
    Query session_summary() at end.
    """

    def __init__(self, fps: float = DEFAULT_FPS):
        self._fps = fps
        self._lock = threading.Lock()

        # Per-student state
        self._current_action: Dict[str, str] = {s: "unknown" for s in KNOWN_STUDENTS}
        self._current_start_frame: Dict[str, int] = {s: 0 for s in KNOWN_STUDENTS}
        self._current_start_time: Dict[str, float] = {s: 0.0 for s in KNOWN_STUDENTS}

        # Completed bouts
        self._bouts: List[ActionBout] = []

        # Frame-level log (capped for memory)
        self._frame_log: deque = deque(maxlen=10000)

        # Accumulated time per action per student
        self._action_time: Dict[str, Dict[str, float]] = {
            s: defaultdict(float) for s in KNOWN_STUDENTS
        }

        # Alert tracking
        self._alerts: List[AlertEvent] = []
        self._alert_cooldown: Dict[str, float] = {}   # student+action → last alert wall time
        self._alert_cooldown_sec = 60.0

        # Frame counter
        self._last_frame: int = 0
        self._session_start_wall = time.time()

    # ---------------------------------------------------------------------- #
    #  Update (called every frame per student)                                #
    # ---------------------------------------------------------------------- #

    def update(
        self,
        student: str,
        action: str,
        frame_idx: int,
        confidence: float = 1.0,
        is_visible: bool = True,
    ):
        """Record one frame of action data for a student."""
        t = frame_idx / self._fps
        wall = time.time()

        with self._lock:
            self._last_frame = max(self._last_frame, frame_idx)

            if not is_visible:
                action = "unknown"

            prev = self._current_action.get(student, "unknown")

            # Transition detected
            if action != prev:
                if prev not in ("unknown",) and student in self._current_start_frame:
                    # Close previous bout
                    prev_start_f = self._current_start_frame[student]
                    prev_start_t = self._current_start_time[student]
                    dur = t - prev_start_t
                    if dur > 0:
                        bout = ActionBout(
                            student=student,
                            action=prev,
                            start_frame=prev_start_f,
                            end_frame=frame_idx - 1,
                            start_time=prev_start_t,
                            end_time=t,
                        )
                        self._bouts.append(bout)
                        self._action_time[student][prev] += dur

                # Start new bout
                self._current_action[student] = action
                self._current_start_frame[student] = frame_idx
                self._current_start_time[student] = t
            else:
                # Same action continues — accumulate time incrementally
                inc = 1.0 / self._fps
                if action not in ("unknown",):
                    self._action_time[student][action] += inc

            # Frame-level log entry
            self._frame_log.append({
                "frame": frame_idx,
                "student": student,
                "action": action,
                "confidence": round(confidence, 3),
            })

            # Alert check
            if action in ALERT_THRESHOLDS_SEC and action != "unknown":
                current_dur_sec = t - self._current_start_time.get(student, t)
                threshold = ALERT_THRESHOLDS_SEC[action]
                if current_dur_sec >= threshold:
                    key = f"{student}:{action}"
                    last = self._alert_cooldown.get(key, 0.0)
                    if wall - last > self._alert_cooldown_sec:
                        alert = AlertEvent(
                            student=student,
                            action=action,
                            duration_sec=current_dur_sec,
                            frame=frame_idx,
                            timestamp=wall - self._session_start_wall,
                        )
                        self._alerts.append(alert)
                        self._alert_cooldown[key] = wall

    # ---------------------------------------------------------------------- #
    #  Live status (for SSE streaming to dashboard)                           #
    # ---------------------------------------------------------------------- #

    def live_status(self) -> Dict[str, Any]:
        """Return snapshot of current state for all students."""
        wall_elapsed = time.time() - self._session_start_wall
        with self._lock:
            students_status = {}
            for s in KNOWN_STUDENTS:
                cur_action = self._current_action.get(s, "unknown")
                cur_start = self._current_start_time.get(s, 0.0)
                t_now = self._last_frame / self._fps
                cur_duration = max(0.0, t_now - cur_start)

                total_productive = sum(
                    self._action_time[s].get(a, 0.0) for a in PRODUCTIVE_ACTIONS
                )
                total_tracked = sum(self._action_time[s].values())
                productivity_pct = (
                    round(100.0 * total_productive / total_tracked, 1)
                    if total_tracked > 1 else 0.0
                )

                students_status[s] = {
                    "current_action": cur_action,
                    "current_action_duration_sec": round(cur_duration, 1),
                    "productivity_pct": productivity_pct,
                    "action_breakdown": {
                        k: round(v, 1) for k, v in self._action_time[s].items()
                    },
                }

            recent_alerts = [
                a.to_dict() for a in self._alerts[-5:]
            ]

            return {
                "session_elapsed_sec": round(wall_elapsed, 1),
                "current_frame": self._last_frame,
                "students": students_status,
                "recent_alerts": recent_alerts,
            }

    # ---------------------------------------------------------------------- #
    #  Session summary                                                         #
    # ---------------------------------------------------------------------- #

    def session_summary(self) -> Dict[str, Any]:
        """Return full session analytics."""
        with self._lock:
            total_sec = self._last_frame / self._fps
            summary = {}
            for s in KNOWN_STUDENTS:
                action_times = dict(self._action_time[s])
                total = sum(action_times.values())
                breakdown_pct = {
                    k: round(100.0 * v / total, 1) if total > 0 else 0.0
                    for k, v in action_times.items()
                }
                productive = sum(action_times.get(a, 0.0) for a in PRODUCTIVE_ACTIONS)
                productivity = round(100.0 * productive / total, 1) if total > 0 else 0.0

                summary[s] = {
                    "total_tracked_sec": round(total, 1),
                    "productivity_score": productivity,
                    "action_times_sec": {k: round(v, 1) for k, v in action_times.items()},
                    "action_breakdown_pct": breakdown_pct,
                }

            return {
                "session_duration_sec": round(total_sec, 1),
                "total_frames": self._last_frame,
                "students": summary,
                "total_alerts": len(self._alerts),
                "alert_events": [a.to_dict() for a in self._alerts],
                "bouts_count": len(self._bouts),
            }

    def get_bouts(self) -> List[Dict]:
        with self._lock:
            return [b.to_dict() for b in self._bouts]

    def get_all_alerts(self) -> List[Dict]:
        with self._lock:
            return [a.to_dict() for a in self._alerts]

    def acknowledge_alert(self, idx: int):
        with self._lock:
            if 0 <= idx < len(self._alerts):
                self._alerts[idx].acknowledged = True

    def get_timeline_data(self) -> Dict[str, List[Dict]]:
        """Return per-student list of bouts for Chart.js timeline."""
        with self._lock:
            result = {s: [] for s in KNOWN_STUDENTS}
            for b in self._bouts:
                if b.student in result:
                    result[b.student].append(b.to_dict())
            return result

    def export_json(self, path: str):
        data = {
            "session_summary": self.session_summary(),
            "bouts": self.get_bouts(),
            "alerts": self.get_all_alerts(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
