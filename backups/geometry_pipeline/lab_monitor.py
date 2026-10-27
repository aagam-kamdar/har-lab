"""
lab_monitor.py - Person-Centric Computer Lab Activity & Productivity Monitor.

Replaces fixed seat ROIs as the primary unit of analysis.
Activity and productivity follow the IDENTIFIED PERSON, using the verified
frozen identity pipeline (ByteTrack + ArcFace + OSNet Re-ID) via IdentityAdapter.

Architecture:
  frame -> IdentityAdapter (ByteTrack + ArcFace + OSNet) -> tracked persons
        -> YOLO-Pose (keypoint skeletons) + YOLO-Det (phones, desk devices)
        -> Person-to-Pose & Person-to-Object spatial association
        -> Dynamic runtime neighbors for discussion (mutual orientation)
        -> Optional desk_zones.yaml reporting (desk_1, desk_2, corridor)
        -> Per-person debounced state machine (70% majority gate)
        -> SQLite WAL upsert (person_status + events + alerts + heartbeat)
        -> Decoupled frame publish to static/live_frame.jpg

Usage:
  python lab_monitor.py
  python lab_monitor.py --replay --source "new lab video.mp4"
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import signal
import sqlite3
import sys
import time
from collections import Counter, deque
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import psutil
import yaml
from ultralytics import YOLO

from identity_adapter import IdentityAdapter

PHONE, KEYBOARD, LAPTOP, TV = 67, 66, 63, 62   # COCO class ids
LOCK_FILE = "monitor.lock"

# Default fallback thresholds (overridden by thresholds.yaml)
DEFAULT_TH = dict(
    kp_conf=0.35,        # min keypoint confidence
    det_conf=0.20,       # object detector confidence
    call_ear=0.60,       # wrist-to-ear distance for phone call (shoulder-widths)
    call_hold=2.5,       # seconds hand must stay at ear
    phone_hand=1.10,     # wrist-to-phone distance for phone use (shoulder-widths)
    head_down=0.35,      # (shoulder_y - nose_y)/sw below this = head tilted down
    turn=0.45,           # |nose_x - shoulder_mid_x|/sw above this = head turned
    talk_dist=5.0,       # max distance (in sw) to candidate neighbor for discussion
    talk_motion=0.25,    # max wrist motion while discussing
    hand_low=0.75,       # wrist below shoulder line = hands at desk level
    kb_motion=0.12,      # min wrist motion (sw/s) to count as typing/mouse use
    active_grace=6.0,    # seconds hands-at-desk still counts as work after last motion
    walk_speed=0.55,     # hip-centroid speed (sw/s)
    stand_knee=0.85,     # knee below hip = standing posture
    away_timeout=10.0,   # seconds missing before marking known person as 'away'
)

COLORS = {  # BGR
    "active_work": (60, 180, 75),   # Green
    "writing":     (80, 200, 100),  # Teal-Green
    "reading":     (220, 180, 50),  # Cyan
    "phone":       (0, 0, 230),     # Red
    "call":        (0, 140, 255),   # Orange-Red
    "discussing":  (200, 120, 0),   # Blue-Amber
    "idle":        (0, 200, 220),   # Yellow
    "walking":     (180, 60, 160),  # Purple
    "away":        (130, 130, 130), # Gray
    "uncertain":   (160, 160, 160), # Light Gray
}


# ---------------------------------------------------------------- process lock
def acquire_lock(lock_file=LOCK_FILE, force=False):
    """Ensure single-instance execution via lock file with PID checking."""
    pid = os.getpid()
    if os.path.exists(lock_file) and not force:
        try:
            with open(lock_file, "r") as f:
                data = json.load(f)
            old_pid = data.get("pid")
            if old_pid and psutil.pid_exists(old_pid):
                proc = psutil.Process(old_pid)
                if proc.is_running() and "python" in proc.name().lower():
                    print(f"[ERROR] Monitor already running with PID {old_pid}. Use --force to override.")
                    return False
        except Exception:
            pass

    try:
        with open(lock_file, "w") as f:
            json.dump({"pid": pid, "started_at": time.time(), "cwd": os.getcwd()}, f)
    except Exception as e:
        print(f"[WARN] Could not write lock file: {e}")
        return False

    return True


def release_lock(lock_file=LOCK_FILE):
    """Clean up lock file on exit."""
    try:
        if os.path.exists(lock_file):
            with open(lock_file, "r") as f:
                data = json.load(f)
            if data.get("pid") == os.getpid():
                os.remove(lock_file)
    except Exception:
        pass


# ---------------------------------------------------------------- database
def init_db(db_path):
    """Open SQLite in WAL mode and initialize person-centric and backward-compatible schemas."""
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA synchronous = NORMAL;")

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS person_status (
        person_id TEXT PRIMARY KEY,
        display_name TEXT,
        track_id INTEGER,
        state TEXT,
        since REAL,
        reason TEXT,
        productivity_pct REAL,
        seconds_per_state TEXT,
        desk_zone TEXT,
        camera_id TEXT,
        identity_confidence REAL,
        last_seen REAL,
        last_update REAL
    );

    CREATE TABLE IF NOT EXISTS seat_status (
        seat_id TEXT PRIMARY KEY,
        person_id TEXT,
        state TEXT,
        since REAL,
        reason TEXT,
        productivity_pct REAL,
        seconds_per_state TEXT,
        last_update REAL,
        camera_id TEXT
    );

    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT,
        track_id INTEGER,
        label TEXT,
        t_start REAL,
        t_end REAL,
        duration_sec REAL,
        reason TEXT,
        desk_zone TEXT,
        camera_id TEXT,
        seat TEXT
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        person_id TEXT,
        track_id INTEGER,
        kind TEXT,
        detail TEXT,
        desk_zone TEXT,
        camera_id TEXT,
        seat TEXT
    );

    CREATE TABLE IF NOT EXISTS monitor_heartbeat (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        last_update REAL,
        fps REAL,
        active_seats INTEGER,
        status TEXT
    );

    CREATE INDEX IF NOT EXISTS ix_events_person ON events(person_id, t_start);
    CREATE INDEX IF NOT EXISTS ix_alerts_ts ON alerts(ts);
    """)

    # Auto-migrate any pre-existing tables if columns are missing
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(events)")
    cols_ev = [c[1] for c in cur.fetchall()]
    if "track_id" not in cols_ev:
        cur.execute("ALTER TABLE events ADD COLUMN track_id INTEGER")
    if "desk_zone" not in cols_ev:
        cur.execute("ALTER TABLE events ADD COLUMN desk_zone TEXT")

    cur.execute("PRAGMA table_info(alerts)")
    cols_al = [c[1] for c in cur.fetchall()]
    if "track_id" not in cols_al:
        cur.execute("ALTER TABLE alerts ADD COLUMN track_id INTEGER")
    if "desk_zone" not in cols_al:
        cur.execute("ALTER TABLE alerts ADD COLUMN desk_zone TEXT")
    if "camera_id" not in cols_al:
        cur.execute("ALTER TABLE alerts ADD COLUMN camera_id TEXT DEFAULT 'cam_0'")
    if "seat" not in cols_al:
        cur.execute("ALTER TABLE alerts ADD COLUMN seat TEXT")

    conn.commit()
    return conn


# ---------------------------------------------------------------- geometry helpers
def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def in_box(p, b, pad=0.0):
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - pad * w <= p[0] <= b[2] + pad * w) and (b[1] - pad * h <= p[1] <= b[3] + pad * h)


def calc_productivity(acc: Counter, productive_classes: Optional[List[str]] = None) -> float:
    """Productivity formula: sum(productive_classes) / present, excluding away, walking, and uncertain."""
    if productive_classes is None:
        productive_classes = ["active_work", "writing"]
    present = sum(v for k, v in acc.items() if k not in ("away", "walking", "uncertain"))
    if present <= 1.0:
        return 0.0
    prod_sec = sum(acc.get(c, 0.0) for c in productive_classes)
    return float(np.clip(100.0 * prod_sec / present, 0.0, 100.0))


# ---------------------------------------------------------------- person tracker state
class PersonTracker:
    """Maintains continuous debounced state machine and history for an identified person."""

    def __init__(self, person_id: str, display_name: str | dict, vote_len: int):
        self.person_id = person_id
        if isinstance(display_name, dict):
            # Backward-compatibility for legacy SeatTracker(sid, cfg, vote_len)
            self.display_name = display_name.get("person_id") or display_name.get("student") or person_id
        else:
            self.display_name = display_name or person_id
        self.track_id: Optional[int] = None
        self.camera_id = "cam_0"
        self.desk_zone: Optional[str] = None
        self.identity_confidence: float = 0.0
        self.is_locked: bool = False
        self.first_seen_t: Optional[float] = None
        self.mean_pose_conf: float = 1.0

        self.wrist_hist = deque(maxlen=80)
        self.hip_hist = deque(maxlen=80)
        self.votes = deque(maxlen=vote_len)
        self.state: str = "walking"
        self.since: Optional[float] = None
        self.reason: str = "detecting"
        self.ear_since: Optional[float] = None
        self.turn_since: Optional[float] = None
        self.last_motion_t: float = -1e9
        self.last_seen_t: float = 0.0
        self.acc = Counter()
        self.last_alert: Dict[str, float] = {}

    def merge_from(self, other: PersonTracker):
        """Merge state when an unknown track is confirmed as this named person."""
        self.acc.update(other.acc)
        if other.wrist_hist:
            self.wrist_hist.extend(other.wrist_hist)
        if other.hip_hist:
            self.hip_hist.extend(other.hip_hist)
        if other.votes:
            self.votes.extend(other.votes)
        if other.first_seen_t and (self.first_seen_t is None or other.first_seen_t < self.first_seen_t):
            self.first_seen_t = other.first_seen_t
        self.last_motion_t = max(self.last_motion_t, other.last_motion_t)
        self.last_seen_t = max(self.last_seen_t, other.last_seen_t)
        if other.turn_since and (self.turn_since is None or other.turn_since < self.turn_since):
            self.turn_since = other.turn_since
        if other.ear_since and (self.ear_since is None or other.ear_since < self.ear_since):
            self.ear_since = other.ear_since
        if other.last_seen_t >= self.last_seen_t:
            self.state = other.state
            self.since = other.since
            self.reason = other.reason
        self.last_alert.update(other.last_alert)
        self.track_id = other.track_id
        self.camera_id = other.camera_id
        self.desk_zone = other.desk_zone or self.desk_zone


# Backward compatibility alias
SeatTracker = PersonTracker


def wrist_motion(p: PersonTracker, now: float, sw: float, win: float = 2.0) -> float:
    pts = [pt for pt in p.wrist_hist if now - pt[0] <= win]
    if len(pts) < 2:
        return 0.0
    tot, n = 0.0, 0
    for (_, l0, r0), (_, l1, r1) in zip(pts, pts[1:]):
        for a, b in ((l0, l1), (r0, r1)):
            if a is not None and b is not None:
                tot += dist(a, b)
                n += 1
    dur = pts[-1][0] - pts[0][0]
    if n == 0 or dur <= 0:
        return 0.0
    return tot / dur / 2.0 / sw


def hip_speed(p: PersonTracker, now: float, sw: float, win: float = 1.0) -> float:
    pts = [(t, c) for t, c in p.hip_hist if now - t <= win and c is not None]
    if len(pts) < 2:
        return 0.0
    dur = pts[-1][0] - pts[0][0]
    return dist(pts[0][1], pts[-1][1]) / max(dur, 1e-3) / sw


def extract_person_features(
    p: PersonTracker,
    kp: Optional[np.ndarray],
    kc: Optional[np.ndarray],
    objs: Dict[int, List[np.ndarray]],
    now: float,
    th: Dict
) -> Optional[Dict]:
    """Turn person keypoints + detected objects into explainable numbers."""
    ok = lambda i: (kc[i] > th["kp_conf"]) if (kc is not None and len(kc) > i) else False
    if kp is None or not (ok(5) and ok(6)):
        p.wrist_hist.append((now, None, None))
        p.hip_hist.append((now, None))
        return None

    sh_mid = (kp[5] + kp[6]) / 2.0
    sw = max(dist(kp[5], kp[6]), 20.0)
    wr = [kp[i] for i in (9, 10) if ok(i)]
    ears = [kp[i] for i in (3, 4) if ok(i)]
    phone_items = objs.get(PHONE, [])
    phone_boxes = []
    verified_phone_boxes = []
    min_verified_conf = float(th.get("verified_phone_conf", 0.60))
    for item in phone_items:
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (float, int)):
            b, c_val = item[0], float(item[1])
            phone_boxes.append(b)
            if c_val >= min_verified_conf:
                verified_phone_boxes.append(b)
        else:
            phone_boxes.append(item)

    phones = [center(b) for b in phone_boxes]

    p.wrist_hist.append((now, kp[9].copy() if ok(9) else None, kp[10].copy() if ok(10) else None))
    hips = [kp[i] for i in (11, 12) if ok(i)]
    hip_centroid = np.mean(hips, axis=0) if hips else None
    p.hip_hist.append((now, hip_centroid))

    f = {}
    f["sw"] = sw
    f["sh_mid"] = sh_mid
    f["hip_centroid"] = hip_centroid
    f["wrist_phone"] = min([dist(w, ph) for w in wr for ph in phones], default=9.0) / sw
    f["has_verified_phone"] = False
    if verified_phone_boxes:
        v_phones = [center(b) for b in verified_phone_boxes]
        f["has_verified_phone"] = (min([dist(w, ph) for w in wr for ph in v_phones], default=9.0) / sw) < th["phone_hand"]
    f["wrist_ear"] = min([dist(w, e) for w in wr for e in ears], default=9.0) / sw
    f["head_rise"] = (sh_mid[1] - kp[0][1]) / sw if ok(0) else 1.0
    f["head_down"] = f["head_rise"] < th["head_down"]

    turn = (kp[0][0] - sh_mid[0]) / sw if ok(0) else 0.0
    f["turn"] = float(turn)
    f["turn_dir"] = int(np.sign(turn)) if abs(turn) > th["turn"] else 0
    f["hands_low"] = any(w[1] > sh_mid[1] + th["hand_low"] * sw for w in wr)

    desk_devices = objs.get(KEYBOARD, []) + objs.get(LAPTOP, []) + objs.get(TV, [])
    f["on_kb"] = any(in_box(w, b, 0.30) for w in wr for b in desk_devices) if desk_devices else False

    f["motion"] = wrist_motion(p, now, sw)
    f["speed"] = hip_speed(p, now, sw)
    knees = [kp[i] for i in (13, 14) if ok(i)]
    f["standing"] = bool(hips and knees and
                         (np.mean(knees, axis=0)[1] - np.mean(hips, axis=0)[1]) > th["stand_knee"] * sw)

    if f["wrist_ear"] < th["call_ear"]:
        p.ear_since = p.ear_since or now
    else:
        p.ear_since = None
    f["ear_hold"] = (now - p.ear_since) if p.ear_since else 0.0

    if f["turn_dir"] != 0:
        p.turn_since = p.turn_since or now
    else:
        p.turn_since = None
    f["turn_hold"] = (now - p.turn_since) if p.turn_since else 0.0

    if f["motion"] > th["kb_motion"]:
        p.last_motion_t = now
    f["since_motion"] = now - p.last_motion_t

    # Check elbow and wrist height geometry for call vs chin-resting
    call_pose_valid = False
    chin_resting = False
    if f["wrist_ear"] < th["call_ear"]:
        for wr_i in (9, 10):
            if ok(wr_i):
                wr_pt = kp[wr_i]
                for ear_i in (3, 4):
                    if ok(ear_i):
                        ear_pt = kp[ear_i]
                        if dist(wr_pt, ear_pt) / sw < th["call_ear"]:
                            el_i = 7 if wr_i == 9 else 8
                            sh_i = 5 if wr_i == 9 else 6
                            el_pt = kp[el_i] if ok(el_i) else None
                            sh_pt = kp[sh_i] if ok(sh_i) else None
                            wrist_at_ear_level = wr_pt[1] <= ear_pt[1] + 0.15 * sw
                            elbow_lifted = True
                            if el_pt is not None and sh_pt is not None:
                                elbow_lifted = (el_pt[1] < sh_pt[1] + 0.70 * sw) or (abs(el_pt[0] - sh_pt[0]) > 0.40 * sw)
                            if wrist_at_ear_level and elbow_lifted:
                                call_pose_valid = True
                            else:
                                chin_resting = True

    f["call_pose_valid"] = call_pose_valid
    f["chin_resting"] = chin_resting
    f["mean_kp_conf"] = float(np.mean(kc)) if kc is not None and len(kc) > 0 else 1.0
    return f


def find_dynamic_discussion(
    pid: str,
    f: Dict,
    all_f: Dict[str, Dict],
    persons: Dict[str, PersonTracker],
    th: Dict
) -> Tuple[bool, str]:
    """
    Dynamic spatial neighbor resolution:
    Checks if person is turned toward another tracked person within talk_dist (in sw).
    """
    if f is None or "sh_mid" not in f:
        return False, ""

    sh_a = f["sh_mid"]
    sw_a = f["sw"]
    turn_dir = f["turn_dir"]
    is_standing = f["standing"]

    candidates = []
    for other_id, other_f in all_f.items():
        if other_id == pid or other_f is None or "sh_mid" not in other_f:
            continue
        sh_b = other_f["sh_mid"]
        d_sw = dist(sh_a, sh_b) / sw_a
        dx = sh_b[0] - sh_a[0]

        # 1. Check standing discussion (both standing close and facing each other)
        if is_standing and other_f.get("standing", False) and d_sw <= 3.5:
            other_p = persons.get(other_id)
            other_tid = getattr(other_p, "track_id", other_id)
            if other_p and other_p.person_id in ("aagam", "harshil", "soham", "swayam"):
                b_name = other_p.display_name
            else:
                b_name = f"track:{other_tid}"
            return True, f"standing discussion with {b_name}"

        # 2. Check seated / directional discussion
        if turn_dir != 0 and np.sign(dx) == turn_dir and d_sw <= th["talk_dist"]:
            other_turn = other_f.get("turn_dir", 0)
            is_mutual = (other_turn == -turn_dir)
            if is_mutual or d_sw <= 2.5:
                candidates.append((d_sw, other_id, other_f, is_mutual))

    if not candidates:
        return False, ""

    # Closest person in turn direction
    candidates.sort(key=lambda x: x[0])
    best_dist, best_id, best_f, mutual = candidates[0]
    best_p = persons.get(best_id)
    best_tid = getattr(best_p, "track_id", best_id)
    if best_p and best_p.person_id in ("aagam", "harshil", "soham", "swayam"):
        best_name = best_p.display_name
    else:
        best_name = f"track:{best_tid}"

    # Check mutual orientation (person B turning toward person A)
    return True, f"turned toward {best_name} (mutual={mutual})"


def raw_label_person(
    p: PersonTracker,
    f: Optional[Dict],
    all_f: Dict[str, Dict],
    persons: Dict[str, PersonTracker],
    now: float,
    th: Dict
) -> Tuple[str, str]:
    """Explainable rule cascade. Returns (label, reason)."""
    if f is None:
        missing_sec = now - p.last_seen_t
        if missing_sec >= th["away_timeout"]:
            return "away", f"not seen in camera for {missing_sec:.0f}s"
        return "away", "no keypoint detection in current frame"

    # Walking: high hip speed or standing upright (does not depend on leaving a seat!)
    if f["speed"] > th["walk_speed"] or f["standing"]:
        return "walking", f"hip speed {f['speed']:.2f} sw/s, standing={f['standing']}"

    # Phone Call: wrist held to ear + elbow lifted + wrist at/above ear level + stillness
    if f["wrist_ear"] < th["call_ear"] and f.get("call_pose_valid", False) and f["ear_hold"] >= th["call_hold"]:
        return "call", f"wrist-ear {f['wrist_ear']:.2f} (elbow lifted, ear-level), held {f['ear_hold']:.1f}s"

    # Phone Use: wrist near detected phone with head tilted down
    if f["wrist_phone"] < th["phone_hand"] and f["head_down"]:
        return "phone", f"wrist-phone {f['wrist_phone']:.2f} sw, head down"

    # Discussing: dynamic neighbor check (reverted default: 0.0s, no duration gate before discussing)
    discuss_min_turn_s = th.get("discuss_min_turn_s", 0.0)
    if f["turn_dir"] != 0 and f["motion"] < th["talk_motion"] and f.get("turn_hold", 0.0) >= discuss_min_turn_s:
        is_disc, disc_reason = find_dynamic_discussion(p.person_id, f, all_f, persons, th)
        if is_disc:
            return "discussing", f"{disc_reason} (held {f.get('turn_hold', 0.0):.1f}s)"

    # Writing: hands low at desk, head tilted down toward notebook/paper, micro-motion present
    if f["hands_low"] and f["head_down"] and (0.04 <= f["motion"] <= 0.45) and not f["on_kb"]:
        return "writing", f"handwriting on desk, motion {f['motion']:.2f} sw/s, head down"

    # Active Work (digital work / typing): hands low or on keyboard/laptop with motion or grace period
    if (f["hands_low"] or f["on_kb"]) and \
            (f["motion"] > th["kb_motion"] or f["since_motion"] < th["active_grace"]):
        return "active_work", f"hands at desk/keyboard, motion {f['motion']:.2f} sw/s"

    # Reading: seated steadily facing monitor/desk without motion (or resting chin while reading)
    if (f["hands_low"] or f.get("chin_resting", False)) and f["motion"] <= th["kb_motion"]:
        return "reading", f"reading/monitoring (stillness {f['since_motion']:.1f}s)"

    # Uncertain: noisy/low confidence keypoints
    if f.get("mean_kp_conf", 1.0) < 0.30:
        return "uncertain", f"low keypoint confidence ({f.get('mean_kp_conf', 0.0):.2f})"

    return "idle", f"motion {f['motion']:.2f} sw/s, no desk interaction"


# ---------------------------------------------------------------- monitor runner
class LabMonitor:
    def __init__(
        self,
        config_path="config.yaml",
        th_path=None,
        db_path=None,
        desk_zones_path=None,
        source_override=None,
        replay=False,
        fps_override=None,
        show=False,
        device=None,
        seats_path=None,
    ):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        app_cfg = self.cfg.get("app", {})
        mon_cfg = self.cfg.get("monitor", {})
        cam_cfg = self.cfg.get("cameras", {})

        self.db_path = db_path or app_cfg.get("db_path", "lab_events.db")
        self.frame_dump_path = app_cfg.get("frame_dump_path", "static/live_frame.jpg")
        os.makedirs(os.path.dirname(self.frame_dump_path) or ".", exist_ok=True)

        # Init DB immediately so canonical tables exist before loading models
        self.db = init_db(self.db_path)

        self.th_path = th_path or mon_cfg.get("thresholds_path", "thresholds.yaml")
        self.desk_zones_path = desk_zones_path or mon_cfg.get("desk_zones_path", "desk_zones.yaml")
        self.target_fps = fps_override or float(mon_cfg.get("fps", 8.0))
        self.debounce_s = float(mon_cfg.get("debounce_window_s", 2.0))
        self.away_timeout_s = float(mon_cfg.get("away_timeout_s", 10.0))
        self.imgsz = int(mon_cfg.get("imgsz", 640))
        self.device = device or mon_cfg.get("device", "cpu")
        self.publish_fps = float(mon_cfg.get("frame_publish_fps", 5.0))
        self.publish_scale = float(mon_cfg.get("frame_publish_scale", 0.6))
        self.phone_alert_s = float(mon_cfg.get("phone_alert_s", 60.0))
        self.idle_alert_s = float(mon_cfg.get("idle_alert_s", 180.0))
        self.alert_cooldown_s = float(mon_cfg.get("alert_cooldown_s", 300.0))
        self.min_track_duration_s = float(mon_cfg.get("min_track_duration_s", 1.5))
        self.min_pose_conf = float(mon_cfg.get("min_pose_conf", 0.30))
        self.identifying_timeout_s = float(mon_cfg.get("identifying_timeout_s", 5.0))
        self.productive_classes = mon_cfg.get("productive_classes", ["active_work", "writing"])
        # Read validation registry (authoritative validation status)
        self.validated_classes = {
            "active_work": False, "writing": False, "reading": False, "discussing": False,
            "idle": False, "walking": False, "phone": False, "call": False, "away": False, "uncertain": False
        }
        if os.path.exists("validation_registry.yaml"):
            try:
                with open("validation_registry.yaml", "r", encoding="utf-8") as f:
                    reg = yaml.safe_load(f)
                    if reg and "validated_classes" in reg:
                        self.validated_classes.update(reg["validated_classes"])
            except Exception:
                pass
        elif "validated_classes" in mon_cfg:
            self.validated_classes.update(mon_cfg["validated_classes"])

        self.replay = replay
        self.show = show
        self.source_override = source_override
        self.cam_cfg = cam_cfg

        # Load thresholds
        self.th = dict(DEFAULT_TH)
        self.th["away_timeout"] = self.away_timeout_s
        if os.path.exists(self.th_path):
            with open(self.th_path, "r") as f:
                loaded_th = yaml.safe_load(f)
                if loaded_th:
                    self.th.update(loaded_th)

        # Person tracker state & voting configuration
        self.vote_len = max(3, int(self.debounce_s * self.target_fps))
        self.persons: Dict[str, PersonTracker] = {}
        self.last_publish_t = 0.0
        self.running = True

        # Load optional desk zones
        self.desk_zones = {}
        if os.path.exists(self.desk_zones_path):
            try:
                with open(self.desk_zones_path, "r") as f:
                    z_data = yaml.safe_load(f)
                    self.desk_zones = z_data.get("zones", {}) or {}
                print(f"[INFO] Loaded {len(self.desk_zones)} optional desk zones from '{self.desk_zones_path}'.")
            except Exception as e:
                print(f"[WARN] Could not parse desk zones: {e}")

        # Backward compatibility for seats attribute
        self.seats = {}
        if seats_path and os.path.exists(seats_path):
            try:
                with open(seats_path, "r") as f:
                    s_data = yaml.safe_load(f)
                    seats_raw = s_data.get("seats", {}) or {}
                    for sid, scfg in seats_raw.items():
                        st = SeatTracker(sid, scfg, self.vote_len)
                        st.roi = scfg.get("roi", [0, 0, 100, 100])
                        self.seats[sid] = st
            except Exception as e:
                print(f"[WARN] Could not parse seats: {e}")

        # Open video capture(s)
        self.caps = {}
        for cid, cinfo in self.cam_cfg.items():
            src = self.source_override or cinfo.get("source", 0)
            if isinstance(src, str) and src.isdigit():
                src = int(src)
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source for {cid}: {src}")
            self.caps[cid] = cap

        self.primary_cam = "cam_0" if "cam_0" in self.caps else next(iter(self.caps.keys()))

        # Initialize Models & Identity Adapter
        print(f"[INFO] Initializing IdentityAdapter & YOLO models on {self.device}...")
        gate_cfg = mon_cfg.get("identity_gate", {})
        self.arcface_confirm_s = float(gate_cfg.get("arcface_confirm_s", 0.5))
        self.body_stable_s = float(gate_cfg.get("body_stable_s", 3.0))
        self.live_gate_evidence_s = float(mon_cfg.get("live_gate_evidence_s", 1.5))
        self.clear_margin = float(mon_cfg.get("clear_margin", 0.15))
        self.gate_mode = str(gate_cfg.get("mode", "stitching"))
        self.body_only_cross_assignment = bool(gate_cfg.get("body_only_cross_assignment", False))
        self.min_affinity_thresh = float(gate_cfg.get("min_affinity_thresh", 0.50))
        self.stitching_max_gap_s = float(gate_cfg.get("stitching_max_gap_s", 5.0))
        self.stitching_max_dist_px = float(gate_cfg.get("stitching_max_dist_px", 100.0))
        self.stitching_body_veto_thresh = float(gate_cfg.get("stitching_body_veto_thresh", 0.40))
        self.identity_adapters = {
            cid: IdentityAdapter(
                fps=float(self.caps[cid].get(cv2.CAP_PROP_FPS) or self.target_fps),
                yolo_model_path="yolo26n.pt",
                identifying_timeout_s=self.identifying_timeout_s,
                live_gate_evidence_s=self.live_gate_evidence_s,
                clear_margin=self.clear_margin,
                arcface_confirm_s=self.arcface_confirm_s,
                body_stable_s=self.body_stable_s,
                mode=self.gate_mode,
                min_affinity_thresh=self.min_affinity_thresh,
                body_only_cross_assignment=self.body_only_cross_assignment,
                stitching_max_gap_s=self.stitching_max_gap_s,
                stitching_max_dist_px=self.stitching_max_dist_px,
                stitching_body_veto_thresh=self.stitching_body_veto_thresh,
            )
            for cid in self.caps
        }
        self.pose = YOLO("yolov8n-pose.pt")
        self.det = YOLO("yolov8n.pt")

    def match_desk_zone(self, bbox: List[int]) -> Optional[str]:
        """Check if person bbox centroid falls inside an optional desk zone."""
        if not self.desk_zones:
            return None
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        for zid, zinfo in self.desk_zones.items():
            rx1, ry1, rx2, ry2 = zinfo["roi"]
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return zid
        return None

    def retroactive_relabel(self, track_id: int, resolved_person_id: str, display_name: str, desk_zone: Optional[str] = None):
        """
        Retroactively rewrite historical rows in events, alerts, and person_status
        when a track resolves to an enrolled person.
        """
        old_ids = (f"unknown_{track_id}", f"identifying_{track_id}")

        # 1. Update SQLite events table
        self.db.execute(
            """
            UPDATE events
            SET person_id = ?, seat = COALESCE(?, seat)
            WHERE (track_id = ? OR person_id = ? OR person_id = ?)
              AND person_id NOT IN ('aagam', 'harshil', 'soham', 'swayam')
            """,
            (resolved_person_id, desk_zone, track_id, old_ids[0], old_ids[1])
        )

        # 2. Update SQLite alerts table
        self.db.execute(
            """
            UPDATE alerts
            SET person_id = ?, seat = COALESCE(?, seat)
            WHERE (track_id = ? OR person_id = ? OR person_id = ?)
              AND person_id NOT IN ('aagam', 'harshil', 'soham', 'swayam')
            """,
            (resolved_person_id, desk_zone, track_id, old_ids[0], old_ids[1])
        )

        # 3. Clean up temporary person_status entries
        self.db.execute(
            """
            DELETE FROM person_status
            WHERE person_id IN (?, ?)
            """,
            old_ids
        )

        # 4. In-memory tracker merge
        for old_k in old_ids:
            if old_k in self.persons:
                old_p = self.persons.pop(old_k)
                if resolved_person_id in self.persons:
                    self.persons[resolved_person_id].merge_from(old_p)
                else:
                    old_p.person_id = resolved_person_id
                    old_p.display_name = display_name
                    self.persons[resolved_person_id] = old_p

    def close_interval(self, p: PersonTracker, end_time: float):
        """Append closed interval to events table."""
        # Transient filter for unenrolled tracks
        track_dur = (end_time - p.first_seen_t) if (p.first_seen_t is not None) else (end_time - (p.since or end_time))
        is_enrolled = p.person_id in ("aagam", "harshil", "soham", "swayam")
        if not is_enrolled and (track_dur < self.min_track_duration_s or p.mean_pose_conf < self.min_pose_conf):
            return

        if p.since is not None and end_time > p.since:
            dur = end_time - p.since
            self.db.execute(
                """
                INSERT INTO events (person_id, track_id, label, t_start, t_end, duration_sec, reason, desk_zone, camera_id, seat)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (p.person_id, p.track_id, p.state, p.since, end_time, dur, p.reason, p.desk_zone, p.camera_id, p.desk_zone)
            )

    def publish_frame(self, frame: np.ndarray):
        """Save downscaled annotated frame atomically."""
        try:
            h, w = frame.shape[:2]
            scaled = cv2.resize(frame, (int(w * self.publish_scale), int(h * self.publish_scale)))
            tmp_path = self.frame_dump_path + ".tmp.jpg"
            cv2.imwrite(tmp_path, scaled, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            os.replace(tmp_path, self.frame_dump_path)
        except Exception:
            pass

    def run(self, dump_csv: Optional[str] = None, max_frames: Optional[int] = None):
        dump_f, dump_w = None, None
        if dump_csv:
            dump_f = open(dump_csv, "w", newline="")

        print(f"[START] Person-Centric Monitor running | Replay={self.replay} | Target FPS={self.target_fps}")
        frame_idx = 0
        last_cycle_t = None
        cycle_count = 0

        try:
            while self.running:
                if max_frames is not None and cycle_count >= max_frames:
                    print(f"[INFO] Reached max_frames={max_frames}, stopping run.")
                    break
                loop_start_wall = time.time()
                frames = {}

                for cid, cap in self.caps.items():
                    ok, frame = cap.read()
                    if not ok:
                        if self.replay:
                            self.running = False
                            break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            self.running = False
                            break
                    frames[cid] = frame

                if not self.running or not frames:
                    break

                primary_frame = frames.get(self.primary_cam)
                now_wall = time.time()

                if self.replay and primary_frame is not None:
                    pos_msec = self.caps[self.primary_cam].get(cv2.CAP_PROP_POS_MSEC)
                    now_sim = pos_msec / 1000.0
                else:
                    now_sim = now_wall

                dt = (now_sim - last_cycle_t) if (last_cycle_t is not None) else (1.0 / self.target_fps)
                if dt <= 0:
                    dt = 1.0 / self.target_fps
                last_cycle_t = now_sim

                # Process per camera: Identity + YOLO Pose + YOLO Det
                per_camera_tracks = {}
                per_camera_poses = {}
                per_camera_objs = {}

                for cid, cam_frame in frames.items():
                    # 1. Identity tracking pass
                    tracks = self.identity_adapters[cid].update(cam_frame, frame_idx)
                    per_camera_tracks[cid] = tracks

                    # 2. YOLO pose pass
                    pose_res = self.pose(cam_frame, imgsz=self.imgsz, verbose=False, device=self.device)[0]
                    per_camera_poses[cid] = pose_res

                    # 3. YOLO object detection pass (phone, keyboard, laptop, tv)
                    det_res = self.det(
                        cam_frame,
                        classes=[PHONE, KEYBOARD, LAPTOP, TV],
                        conf=self.th["det_conf"],
                        imgsz=self.imgsz,
                        verbose=False,
                        device=self.device
                    )[0]

                    objs = {PHONE: [], KEYBOARD: [], LAPTOP: [], TV: []}
                    if det_res.boxes is not None and len(det_res.boxes) > 0:
                        b_xyxy = det_res.boxes.xyxy.cpu().numpy()
                        b_cls = det_res.boxes.cls.cpu().numpy()
                        b_confs = det_res.boxes.conf.cpu().numpy() if det_res.boxes.conf is not None else np.ones(len(b_cls))
                        for box, cls_id, b_c in zip(b_xyxy, b_cls, b_confs):
                            cid_int = int(cls_id)
                            if cid_int in objs:
                                if cid_int == PHONE:
                                    objs[cid_int].append((box, float(b_c)))
                                else:
                                    objs[cid_int].append(box)
                    per_camera_objs[cid] = objs

                # Associate each track to nearest pose skeleton in that camera
                active_observations = {}
                for cid, tracks in per_camera_tracks.items():
                    pose_res = per_camera_poses[cid]
                    objs = per_camera_objs[cid]

                    p_boxes = pose_res.boxes.xyxy.cpu().numpy() if (pose_res.boxes is not None and len(pose_res.boxes) > 0) else np.empty((0, 4))
                    p_kps = pose_res.keypoints.xy.cpu().numpy() if pose_res.keypoints is not None else None
                    p_confs = pose_res.keypoints.conf.cpu().numpy() if pose_res.keypoints is not None else None

                    for t in tracks:
                        tid = t["track_id"]
                        pid = t["person_id"]
                        tb = t["bbox"]

                        # Keyed by person_id (or unknown_<tid>)
                        state_key = pid if pid != "unknown" else f"unknown_{tid}"

                        # Best matching pose skeleton
                        matched_kp, matched_kc = None, None
                        best_iou = 0.0

                        for idx_p, pb in enumerate(p_boxes):
                            iou = bbox_iou(tb, pb)
                            if iou > best_iou and iou >= 0.25:
                                best_iou = iou
                                if p_kps is not None and len(p_kps) > idx_p:
                                    matched_kp = p_kps[idx_p]
                                    matched_kc = p_confs[idx_p] if p_confs is not None else None

                        # Multi-camera merge: prefer view with higher keypoint confidence
                        mean_conf = float(np.mean(matched_kc)) if (matched_kc is not None and len(matched_kc) > 0) else 0.0

                        if state_key not in active_observations or mean_conf > active_observations[state_key]["conf"]:
                            active_observations[state_key] = {
                                "track": t,
                                "camera_id": cid,
                                "kp": matched_kp,
                                "kc": matched_kc,
                                "objs": objs,
                                "conf": mean_conf,
                            }

                # Manage PersonTrackers and handle identity promotions
                # (e.g. unknown_1 promoted to soham)
                for state_key, obs in active_observations.items():
                    t = obs["track"]
                    pid = t["person_id"]
                    disp = t["display_name"]

                    if pid in ("aagam", "harshil", "soham", "swayam"):
                        # Retroactively relabel earlier rows if this track was previously unknown or identifying
                        self.retroactive_relabel(t["track_id"], pid, disp, self.match_desk_zone(t["bbox"]))

                    if state_key not in self.persons:
                        self.persons[state_key] = PersonTracker(state_key, disp, self.vote_len)
                        self.persons[state_key].first_seen_t = now_sim

                    p = self.persons[state_key]
                    if p.first_seen_t is None:
                        p.first_seen_t = now_sim
                    p.mean_pose_conf = obs["conf"]
                    p.display_name = disp
                    p.track_id = t["track_id"]
                    p.camera_id = obs["camera_id"]
                    p.identity_confidence = t["identity_confidence"]
                    p.is_locked = t["is_locked"]
                    p.desk_zone = self.match_desk_zone(t["bbox"])
                    p.last_seen_t = now_sim

                # Extract features for all persons
                feats = {}
                for p_key, p in self.persons.items():
                    if p_key in active_observations:
                        obs = active_observations[p_key]
                        f = extract_person_features(p, obs["kp"], obs["kc"], obs["objs"], now_sim, self.th)
                    else:
                        f = None
                    feats[p_key] = f

                # Evaluate state machine, debounce, and alerts
                for p_key, p in list(self.persons.items()):
                    f = feats[p_key]
                    label, reason = raw_label_person(p, f, feats, self.persons, now_sim, self.th)

                    p.votes.append(label)
                    cand, count = Counter(p.votes).most_common(1)[0]

                    if p.since is None:
                        p.since, p.state, p.reason = now_sim, cand, reason
                    elif cand != p.state and count >= 0.7 * p.votes.maxlen:
                        self.close_interval(p, now_sim)
                        p.state, p.since, p.reason = cand, now_sim, reason

                    p.acc[p.state] += dt
                    prod_pct = calc_productivity(p.acc, self.productive_classes)
                    seconds_json = json.dumps(dict(p.acc))

                    # Ignore transient unenrolled tracks from cluttering person_status
                    track_dur = (now_sim - p.first_seen_t) if (p.first_seen_t is not None) else (now_sim - (p.since or now_sim))
                    is_enrolled = p.person_id in ("aagam", "harshil", "soham", "swayam")
                    if not is_enrolled and (track_dur < self.min_track_duration_s or p.mean_pose_conf < self.min_pose_conf):
                        continue

                    # Upsert person_status
                    self.db.execute(
                        """
                        INSERT INTO person_status (
                            person_id, display_name, track_id, state, since, reason,
                            productivity_pct, seconds_per_state, desk_zone, camera_id,
                            identity_confidence, last_seen, last_update
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(person_id) DO UPDATE SET
                            display_name=excluded.display_name,
                            track_id=excluded.track_id,
                            state=excluded.state,
                            since=excluded.since,
                            reason=excluded.reason,
                            productivity_pct=excluded.productivity_pct,
                            seconds_per_state=excluded.seconds_per_state,
                            desk_zone=excluded.desk_zone,
                            camera_id=excluded.camera_id,
                            identity_confidence=excluded.identity_confidence,
                            last_seen=excluded.last_seen,
                            last_update=excluded.last_update;
                        """,
                        (
                            p.person_id, p.display_name, p.track_id, p.state, p.since, p.reason,
                            prod_pct, seconds_json, p.desk_zone, p.camera_id,
                            p.identity_confidence, p.last_seen_t, now_wall
                        )
                    )

                    # Backward-compatibility: if desk_zone matches a known seat, upsert seat_status
                    if p.desk_zone and p.desk_zone.startswith("D"):
                        self.db.execute(
                            """
                            INSERT INTO seat_status (
                                seat_id, person_id, state, since, reason, productivity_pct,
                                seconds_per_state, last_update, camera_id
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(seat_id) DO UPDATE SET
                                person_id=excluded.person_id,
                                state=excluded.state,
                                since=excluded.since,
                                reason=excluded.reason,
                                productivity_pct=excluded.productivity_pct,
                                seconds_per_state=excluded.seconds_per_state,
                                last_update=excluded.last_update,
                                camera_id=excluded.camera_id;
                            """,
                            (p.desk_zone, p.display_name, p.state, p.since, p.reason, prod_pct, seconds_json, now_wall, p.camera_id)
                        )

                    # Distraction Alerts
                    held = now_sim - p.since
                    for kind, limit in (("phone", self.phone_alert_s), ("idle", self.idle_alert_s)):
                        # Disable distraction alerts for any label marked unvalidated
                        if not self.validated_classes.get(kind, False):
                            continue

                        # Phone alerts must also require a verified phone detection, not merely a nearby wrist
                        if kind == "phone":
                            if f is None or not f.get("has_verified_phone", False):
                                continue

                        if p.state == kind and held >= limit and (now_sim - p.last_alert.get(kind, -1e9)) > self.alert_cooldown_s:
                            track_dur = (now_sim - p.first_seen_t) if (p.first_seen_t is not None) else held
                            is_enrolled = p.person_id in ("aagam", "harshil", "soham", "swayam")
                            if not is_enrolled and (track_dur < self.min_track_duration_s or p.mean_pose_conf < self.min_pose_conf):
                                continue
                            p.last_alert[kind] = now_sim
                            detail = json.dumps({"held_s": round(held, 1), "reason": p.reason})
                            self.db.execute(
                                """
                                INSERT INTO alerts (ts, person_id, track_id, kind, detail, desk_zone, camera_id, seat)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (now_wall, p.person_id, p.track_id, kind, detail, p.desk_zone, p.camera_id, p.desk_zone)
                            )
                            print(f"[ALERT] {p.display_name} - {kind} for {held:.0f}s ({p.reason})")

                    # CSV dump for threshold tuning if requested
                    if dump_f and f is not None:
                        row = {
                            "t": round(now_sim, 3), "person_id": p.person_id, "state": p.state,
                            "desk": p.desk_zone or "",
                            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items() if not isinstance(v, np.ndarray)}
                        }
                        if dump_w is None:
                            dump_w = csv.DictWriter(dump_f, fieldnames=list(row))
                            dump_w.writeheader()
                        dump_w.writerow(row)

                # Upsert heartbeat
                self.db.execute(
                    """
                    INSERT INTO monitor_heartbeat (id, last_update, fps, active_seats, status)
                    VALUES (1, ?, ?, ?, 'running')
                    ON CONFLICT(id) DO UPDATE SET
                        last_update=excluded.last_update,
                        fps=excluded.fps,
                        active_seats=excluded.active_seats,
                        status=excluded.status;
                    """,
                    (now_wall, self.target_fps, len(active_observations))
                )
                self.db.commit()

                # Visual overlay and frame publishing (throttled)
                if primary_frame is not None and (now_wall - self.last_publish_t >= 1.0 / self.publish_fps or self.show):
                    vis = primary_frame.copy()

                    # Draw optional desk zones
                    for zid, zinfo in self.desk_zones.items():
                        zx1, zy1, zx2, zy2 = zinfo["roi"]
                        cv2.rectangle(vis, (zx1, zy1), (zx2, zy2), (70, 70, 70), 1)
                        cv2.putText(vis, zinfo.get("name", zid), (zx1 + 4, zy1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

                    # Draw tracked person annotations
                    for p_key, obs in active_observations.items():
                        if obs["camera_id"] != self.primary_cam:
                            continue
                        p = self.persons.get(p_key)
                        if not p:
                            continue
                        t = obs["track"]
                        x1, y1, x2, y2 = t["bbox"]
                        col = COLORS.get(p.state, (130, 130, 130))

                        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                        tag = f"{p.display_name} | {p.state} ({calc_productivity(p.acc, self.productive_classes):.0f}%)"
                        if p.desk_zone:
                            tag += f" [{p.desk_zone}]"

                        tw = len(tag) * 9
                        cv2.rectangle(vis, (x1, max(0, y1 - 22)), (x1 + tw, y1), col, -1)
                        cv2.putText(vis, tag, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    if now_wall - self.last_publish_t >= 1.0 / self.publish_fps:
                        self.publish_frame(vis)
                        self.last_publish_t = now_wall

                    if self.show:
                        cv2.imshow("Person-Centric Lab Monitor", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                frame_idx += 1
                cycle_count += 1
                if not self.replay:
                    elapsed = time.time() - loop_start_wall
                    target_interval = 1.0 / self.target_fps
                    if elapsed < target_interval:
                        time.sleep(target_interval - elapsed)

        finally:
            end_wall = time.time()
            for p in self.persons.values():
                self.close_interval(p, last_cycle_t or end_wall)

            try:
                self.db.execute("UPDATE monitor_heartbeat SET status='stopped', last_update=? WHERE id=1", (time.time(),))
                self.db.commit()
                self.db.close()
            except Exception:
                pass

            for cap in self.caps.values():
                cap.release()

            if self.show:
                cv2.destroyAllWindows()
            if dump_f:
                dump_f.close()

            print(f"[FINISH] Monitor stopped. Processed {cycle_count} cycles.")


def main():
    ap = argparse.ArgumentParser(description="Person-Centric Lab Monitor")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--th", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--zones", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None, help="Stop after processing N frames")
    a = ap.parse_args()

    if not acquire_lock(force=a.force):
        sys.exit(1)

    atexit.register(release_lock)

    def handle_signal(sig, frame):
        print("\nStopping monitor...")
        release_lock()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    monitor = LabMonitor(
        config_path=a.config,
        th_path=a.th,
        db_path=a.db,
        desk_zones_path=a.zones,
        source_override=a.source,
        replay=a.replay,
        fps_override=a.fps,
        show=a.show,
        device=a.device,
    )
    monitor.run(dump_csv=a.dump, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
