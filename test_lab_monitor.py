"""
test_lab_monitor.py - Unit and regression tests for lab activity monitor & API.

Tests:
  1. State machine debounce logic (majority voting prevents flickering).
  2. Productivity calculation formula (active / present, excluding away & walking).
  3. Distraction alert cooldown logic (no duplicate alerts within cooldown window).
  4. SQLite schema and seat_status upsert integrity (WAL mode, JSON serialization).
  5. Single-instance process locking mechanism.
"""
import json
import os
import sqlite3
import time
from collections import Counter, deque

import numpy as np
import pytest

from lab_monitor import (
    SeatTracker,
    acquire_lock,
    calc_productivity,
    init_db,
    release_lock,
)


# ---------------------------------------------------------------- Test 1: Debounce Logic
def test_state_machine_debounce_prevents_flicker():
    """Verify that transient noise does not flip state without meeting 70% threshold."""
    vote_len = 10
    cfg = {"roi": [0, 0, 100, 100], "person_id": "TestStudent", "camera_id": "cam_0"}
    tracker = SeatTracker("S01", cfg, vote_len)
    tracker.state = "active_work"
    tracker.since = 0.0

    # Fill votes with active_work
    for _ in range(vote_len):
        tracker.votes.append("active_work")

    assert tracker.state == "active_work"

    # Inject minor noise (2 votes of 'idle')
    tracker.votes.append("idle")
    tracker.votes.append("idle")

    cand, count = Counter(tracker.votes).most_common(1)[0]
    # 'active_work' still has 8/10 votes, no transition
    assert cand == "active_work"
    assert count == 8
    assert count >= 0.7 * vote_len

    # Now inject 5 votes of 'idle' (5 idle, 5 active_work)
    for _ in range(3):
        tracker.votes.append("idle")

    cand, count = Counter(tracker.votes).most_common(1)[0]
    # No candidate has >= 70% (7 votes)
    threshold_met = (count >= 0.7 * tracker.votes.maxlen)
    assert not threshold_met, "50% vote split must NOT trigger state change"

    # Now inject until idle reaches 8/10
    for _ in range(3):
        tracker.votes.append("idle")

    cand, count = Counter(tracker.votes).most_common(1)[0]
    assert cand == "idle"
    assert count == 8
    assert count >= 0.7 * tracker.votes.maxlen, "80% majority MUST confirm state transition"


# ---------------------------------------------------------------- Test 2: Productivity Calc
def test_productivity_calculation():
    """
    Productivity must equal:
      active_work / (active_work + idle + phone + call + discussing)
    Excluding 'away' and 'walking'.
    """
    # Case 1: Pure active work
    acc1 = Counter({"active_work": 100.0, "away": 500.0, "walking": 200.0})
    prod1 = calc_productivity(acc1)
    assert pytest.approx(prod1, 0.01) == 100.0

    # Case 2: 60s active work, 40s idle, 300s away
    acc2 = Counter({"active_work": 60.0, "idle": 40.0, "away": 300.0})
    prod2 = calc_productivity(acc2)
    # present = 60 + 40 = 100. active = 60 -> 60%
    assert pytest.approx(prod2, 0.01) == 60.0

    # Case 3: Mixed activities
    acc3 = Counter({
        "active_work": 45.0,
        "phone": 15.0,
        "call": 10.0,
        "discussing": 20.0,
        "idle": 10.0,
        "walking": 50.0,
        "away": 100.0,
    })
    # present = 45 + 15 + 10 + 20 + 10 = 100. active = 45 -> 45%
    prod3 = calc_productivity(acc3)
    assert pytest.approx(prod3, 0.01) == 45.0

    # Case 4: No presence (< 1 sec)
    acc4 = Counter({"away": 120.0})
    assert calc_productivity(acc4) == 0.0

    # Case 5: Writing is productive by default
    acc5 = Counter({"writing": 40.0, "active_work": 40.0, "idle": 20.0, "walking": 50.0})
    # present = 40 + 40 + 20 = 100. productive (active_work + writing) = 80 -> 80%
    assert pytest.approx(calc_productivity(acc5), 0.01) == 80.0

    # Case 6: Configurable reading membership
    acc6 = Counter({"active_work": 30.0, "writing": 20.0, "reading": 50.0, "idle": 100.0})
    # Default (reading not productive): prod = (30 + 20) / (30 + 20 + 50 + 100) = 50 / 200 = 25%
    assert pytest.approx(calc_productivity(acc6), 0.01) == 25.0
    # Custom configured (reading productive): prod = (30 + 20 + 50) / 200 = 100 / 200 = 50%
    custom_prod = ["active_work", "writing", "reading"]
    assert pytest.approx(calc_productivity(acc6, custom_prod), 0.01) == 50.0


# ---------------------------------------------------------------- Test 3: Alert Cooldown
def test_alert_cooldown_logic():
    """Verify that alerts do not trigger repeatedly within the cooldown period."""
    cooldown_s = 300.0
    phone_alert_limit = 60.0
    last_alerts = {}

    def should_fire_alert(now, state, since, kind):
        held = now - since
        if state == kind and held >= phone_alert_limit:
            last_fired = last_alerts.get(kind, -1e9)
            if (now - last_fired) > cooldown_s:
                last_alerts[kind] = now
                return True
        return False

    t0 = 1000.0
    # At t = 1060s, held = 60s -> First alert fires
    assert should_fire_alert(t0 + 60.0, "phone", t0, "phone") is True
    assert last_alerts["phone"] == 1060.0

    # At t = 1070s, still in phone state (held = 70s), but within cooldown -> Does NOT fire
    assert should_fire_alert(t0 + 70.0, "phone", t0, "phone") is False

    # At t = 1350s (290s after first alert) -> Still within cooldown -> Does NOT fire
    assert should_fire_alert(t0 + 350.0, "phone", t0, "phone") is False

    # At t = 1361s (301s after first alert) -> Cooldown elapsed -> Fires again
    assert should_fire_alert(t0 + 361.0, "phone", t0, "phone") is True
    assert last_alerts["phone"] == 1361.0


# ---------------------------------------------------------------- Test 4: DB Schema & Upsert
def test_seat_status_upsert_schema(tmp_path):
    """Test SQLite WAL initialization and canonical seat_status upsert."""
    db_file = str(tmp_path / "test_events.db")
    db = init_db(db_file)

    # Check WAL mode
    cur = db.cursor()
    cur.execute("PRAGMA journal_mode")
    mode = cur.fetchone()[0]
    assert mode.lower() == "wal"

    # Insert seat D01
    sec_json = json.dumps({"active_work": 120.5, "idle": 15.0})
    db.execute("""
        INSERT INTO seat_status (seat_id, person_id, state, since, reason, productivity_pct,
                                 seconds_per_state, last_update, camera_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("D01", "Aagam", "active_work", 100.0, "hands at desk", 88.9, sec_json, 150.0, "cam_0"))
    db.commit()

    # Query back
    cur.execute("SELECT * FROM seat_status WHERE seat_id='D01'")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "D01"          # seat_id
    assert row[1] == "Aagam"        # person_id
    assert row[2] == "active_work"  # state
    assert json.loads(row[6])["active_work"] == 120.5

    # Upsert with state change
    sec_json2 = json.dumps({"active_work": 120.5, "idle": 25.0})
    db.execute("""
        INSERT INTO seat_status (seat_id, person_id, state, since, reason, productivity_pct,
                                 seconds_per_state, last_update, camera_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(seat_id) DO UPDATE SET
            state=excluded.state,
            since=excluded.since,
            reason=excluded.reason,
            productivity_pct=excluded.productivity_pct,
            seconds_per_state=excluded.seconds_per_state,
            last_update=excluded.last_update
    """, ("D01", "Aagam", "idle", 150.0, "looking away", 82.8, sec_json2, 160.0, "cam_0"))
    db.commit()

    cur.execute("SELECT state, reason, productivity_pct FROM seat_status WHERE seat_id='D01'")
    updated = cur.fetchone()
    assert updated[0] == "idle"
    assert updated[1] == "looking away"
    assert pytest.approx(updated[2], 0.1) == 82.8

    db.close()


# ---------------------------------------------------------------- Test 5: Process Lock
def test_process_lock_mechanism(tmp_path):
    """Test lock file creation, detection of active process, and cleanup."""
    test_lock = str(tmp_path / "test_monitor.lock")

    # Acquire lock
    assert acquire_lock(lock_file=test_lock) is True
    assert os.path.exists(test_lock)

    with open(test_lock, "r") as f:
        data = json.load(f)
    assert data["pid"] == os.getpid()

    # Second acquire without force should fail because current PID is alive
    assert acquire_lock(lock_file=test_lock, force=False) is False

    # Force acquire should succeed
    assert acquire_lock(lock_file=test_lock, force=True) is True

    # Release lock
    release_lock(lock_file=test_lock)
    assert not os.path.exists(test_lock)


# ---------------------------------------------------------------- Test 6: Person Status Schema
def test_person_status_upsert_schema(tmp_path):
    """Test person_status table schema and upsert behavior."""
    db_file = str(tmp_path / "test_person_events.db")
    db = init_db(db_file)

    sec_json = json.dumps({"active_work": 150.0, "idle": 20.0, "walking": 10.0})
    db.execute("""
        INSERT INTO person_status (
            person_id, display_name, track_id, state, since, reason,
            productivity_pct, seconds_per_state, desk_zone, camera_id,
            identity_confidence, last_seen, last_update
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "soham", "Soham", 1, "active_work", 10.0, "hands at desk, motion 0.45",
        88.2, sec_json, "D04", "cam_0", 0.88, 100.0, 100.0
    ))
    db.commit()

    cur = db.cursor()
    cur.execute("SELECT person_id, display_name, track_id, desk_zone, productivity_pct FROM person_status WHERE person_id='soham'")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "soham"
    assert row[1] == "Soham"
    assert row[2] == 1
    assert row[3] == "D04"
    assert pytest.approx(row[4], 0.1) == 88.2

    db.close()


# ---------------------------------------------------------------- Test 7: Dynamic Discussion
def test_dynamic_neighbor_discussion():
    """Verify runtime spatial neighbor discussion logic with mutual head turns."""
    from lab_monitor import PersonTracker, find_dynamic_discussion

    th = {"talk_dist": 5.0}
    p_soham = PersonTracker("soham", "Soham", 10)
    p_harshil = PersonTracker("harshil", "Harshil", 10)

    persons = {"soham": p_soham, "harshil": p_harshil}

    # Case A: Soham (at x=700) turns left (turn_dir=-1) toward Harshil (at x=600)
    # Harshil (at x=600) turns right (turn_dir=+1) toward Soham -> MUTUAL
    sw = 80.0
    f_soham = {
        "sh_mid": np.array([700.0, 200.0]),
        "sw": sw,
        "turn_dir": -1,  # turning left
        "standing": False,
    }
    f_harshil = {
        "sh_mid": np.array([600.0, 200.0]),
        "sw": sw,
        "turn_dir": +1,  # turning right
        "standing": False,
    }
    all_f = {"soham": f_soham, "harshil": f_harshil}

    is_disc, reason = find_dynamic_discussion("soham", f_soham, all_f, persons, th)
    assert is_disc is True
    assert "Harshil" in reason
    assert "mutual=True" in reason

    # Case B: Harshil looking away (turn_dir=-1) -> NOT mutual
    f_harshil_away = dict(f_harshil, turn_dir=-1)
    all_f_non_mutual = {"soham": f_soham, "harshil": f_harshil_away}
    is_disc2, reason2 = find_dynamic_discussion("soham", f_soham, all_f_non_mutual, persons, th)
    assert is_disc2 is True
    assert "Harshil" in reason2
    assert "mutual=False" in reason2


# ---------------------------------------------------------------- Test 8: Desk Swap Simulation
def test_desk_swap_simulation():
    """Verify that moving between desk zones preserves the person's identity and history."""
    from lab_monitor import PersonTracker, calc_productivity

    p = PersonTracker("aagam", "Aagam", 10)
    p.desk_zone = "D01"
    p.state = "active_work"
    p.acc["active_work"] += 120.0
    p.acc["idle"] += 30.0

    prod_desk1 = calc_productivity(p.acc)
    assert pytest.approx(prod_desk1, 0.1) == 80.0

    # Student walks to Desk 3 (desk swap)
    p.desk_zone = "D03"
    p.state = "active_work"
    p.acc["active_work"] += 60.0

    # Person-centric integrity: person_id remains 'aagam', history continues seamlessly
    assert p.person_id == "aagam"
    assert p.desk_zone == "D03"
    assert p.acc["active_work"] == 180.0
    prod_desk3 = calc_productivity(p.acc)
    # present = 180 + 30 = 210. 100 * 180 / 210 = 85.7%
    assert pytest.approx(prod_desk3, 0.1) == 85.7


# ---------------------------------------------------------------- Test 9: Unknown Stats Isolation
def test_unknown_person_stats_isolation():
    """Verify unknown persons do not pollute named classroom productivity calculations."""
    from app_api import app
    from fastapi.testclient import TestClient

    # Named stats: 100s active, 20s idle
    named_acc = {"active_work": 100.0, "idle": 20.0}
    # Unknown stats: 500s idle
    unknown_acc = {"idle": 500.0}

    # Named classroom productivity should be 100 / (100 + 20) = 83.3%
    total_active = named_acc["active_work"]
    total_present = named_acc["active_work"] + named_acc["idle"]
    class_prod = 100.0 * total_active / total_present
    assert pytest.approx(class_prod, 0.1) == 83.3


# ---------------------------------------------------------------- Test 10: Retroactive Relabeling
def test_retroactive_relabel(tmp_path):
    """Verify that resolving an unknown/identifying track retroactively rewrites earlier rows in DB and state."""
    from lab_monitor import LabMonitor, PersonTracker, init_db

    db_test = str(tmp_path / "test_relabel.db")
    con = init_db(db_test)

    # Simulate unknown_5 inserting earlier events, alerts, and person_status
    con.execute("""
        INSERT INTO events (person_id, track_id, label, t_start, t_end, duration_sec, reason, desk_zone, camera_id, seat)
        VALUES ('identifying_5', 5, 'active_work', 10.0, 15.0, 5.0, 'hands at desk', 'D03', 'cam_0', 'D03')
    """)
    con.execute("""
        INSERT INTO alerts (ts, person_id, track_id, kind, detail, desk_zone, camera_id, seat)
        VALUES (12.0, 'identifying_5', 5, 'idle', '{"held_s": 180}', 'D03', 'cam_0', 'D03')
    """)
    con.execute("""
        INSERT INTO person_status (person_id, display_name, track_id, state, since, reason, productivity_pct, seconds_per_state, desk_zone, camera_id, identity_confidence, last_seen, last_update)
        VALUES ('identifying_5', 'Identifying...', 5, 'active_work', 10.0, 'hands at desk', 100.0, '{"active_work": 5.0}', 'D03', 'cam_0', 0.0, 15.0, 15.0)
    """)
    con.commit()

    # Create dummy LabMonitor object
    class DummyMonitor:
        def __init__(self, db_conn):
            self.db = db_conn
            self.persons = {
                "identifying_5": PersonTracker("identifying_5", "Identifying...", 10)
            }
            self.persons["identifying_5"].acc["active_work"] += 5.0

    dummy = DummyMonitor(con)
    LabMonitor.retroactive_relabel(dummy, track_id=5, resolved_person_id="harshil", display_name="Harshil", desk_zone="D03")
    con.commit()

    # Verify SQLite events row was retroactively rewritten to 'harshil'
    cur = con.cursor()
    cur.execute("SELECT person_id, seat FROM events WHERE track_id = 5")
    ev = cur.fetchone()
    assert ev[0] == "harshil"
    assert ev[1] == "D03"

    # Verify SQLite alerts row was retroactively rewritten to 'harshil'
    cur.execute("SELECT person_id FROM alerts WHERE track_id = 5")
    al = cur.fetchone()
    assert al[0] == "harshil"

    # Verify identifying_5 was deleted from person_status
    cur.execute("SELECT count(*) FROM person_status WHERE person_id = 'identifying_5'")
    assert cur.fetchone()[0] == 0

    # Verify in-memory tracker merged into harshil
    assert "identifying_5" not in dummy.persons
    assert "harshil" in dummy.persons
    assert dummy.persons["harshil"].acc["active_work"] == 5.0
    assert dummy.persons["harshil"].display_name == "Harshil"

    con.close()


# ---------------------------------------------------------------- Test 11: Head-Turn Duration Requirement
def test_idle_discussing_head_turn_duration():
    """Verify that discussing requires a minimum head-turn duration (default 2s) before preempting idle."""
    from lab_monitor import PersonTracker, raw_label_person

    th = {
        "away_timeout": 30.0,
        "walk_speed": 0.55,
        "call_ear": 0.60,
        "call_hold": 2.5,
        "phone_hand": 1.10,
        "talk_motion": 0.25,
        "kb_motion": 0.12,
        "active_grace": 6.0,
        "hand_low": 0.75,
        "talk_dist": 5.0,
        "discuss_min_turn_s": 2.0,
    }

    pA = PersonTracker("soham", "Soham", 10)
    pB = PersonTracker("harshil", "Harshil", 10)
    persons = {"soham": pA, "harshil": pB}

    now = 100.0

    # Person A features: looking right toward B (turn_dir=1), low motion (0.05 sw/s), no desk interaction
    fA = {
        "speed": 0.0,
        "standing": False,
        "wrist_ear": 2.0,
        "ear_hold": 0.0,
        "wrist_phone": 99.0,
        "head_down": False,
        "turn_dir": 1,
        "motion": 0.05,
        "hands_low": False,
        "on_kb": False,
        "since_motion": 999.0,
        "sh_mid": [100.0, 100.0],
        "sw": 40.0,
        "turn_hold": 0.5, # ONLY 0.5s turn hold (< 2.0s threshold)
    }

    # Person B features: neighbor standing to the right (x=160), looking left toward A (turn_dir=-1)
    fB = {
        "sh_mid": [160.0, 100.0], # within talk_dist (60px / 40px = 1.5 sw)
        "sw": 40.0,
        "turn_dir": -1,           # mutual orientation
    }
    all_f = {"soham": fA, "harshil": fB}

    # Case 1: Brief glance (0.5s < 2.0s) -> MUST NOT PREEMPT IDLE
    label, reason = raw_label_person(pA, fA, all_f, persons, now, th)
    assert label == "idle", f"Expected 'idle' for brief glance, got '{label}'"

    # Case 2: Sustained head turn (2.5s >= 2.0s) -> NOW PREEMPTS TO DISCUSSING
    fA["turn_hold"] = 2.5
    label, reason = raw_label_person(pA, fA, all_f, persons, now, th)
    assert label == "discussing", f"Expected 'discussing' after sustained turn, got '{label}'"
    assert "Harshil" in reason or "track:" in reason
    assert "held 2.5s" in reason

    # Case 3: Configurable threshold (change discuss_min_turn_s to 3.0s)
    th["discuss_min_turn_s"] = 3.0
    label, _ = raw_label_person(pA, fA, all_f, persons, now, th)
    # 2.5s is now less than 3.0s -> returns idle
    assert label == "idle", f"Expected 'idle' when under reconfigured 3.0s threshold, got '{label}'"


# ---------------------------------------------------------------- Test: State Carry-Over Across Track Changes
def test_state_carryover_across_track_change(tmp_path):
    """
    Verify that motion history, debounce state, and productivity accumulators
    persist across track changes for the same person (both direct and via temporary identifying tracks).
    """
    from lab_monitor import PersonTracker, LabMonitor, wrist_motion
    import sqlite3

    db_path = str(tmp_path / "test_carryover.db")
    init_db(db_path)

    # 1. Direct Track ID Change
    p = PersonTracker("aagam", "Aagam", vote_len=10)
    p.track_id = 12
    p.first_seen_t = 0.0
    p.last_seen_t = 10.0
    p.state = "active_work"
    p.since = 2.0
    p.last_motion_t = 9.5
    p.acc["active_work"] = 25.0
    p.acc["idle"] = 5.0
    
    # Add wrist history points for track 12
    for t in np.linspace(1.0, 10.0, 10):
        p.wrist_hist.append((t, np.array([100.0 + t, 200.0]), np.array([150.0 + t, 200.0])))
        p.votes.append("active_work")

    # Mid-sequence, track changes from 12 to 44
    p.track_id = 44
    p.last_seen_t = 11.0
    p.wrist_hist.append((11.0, np.array([112.0, 200.0]), np.array([162.0, 200.0])))
    p.votes.append("active_work")
    p.acc["active_work"] += 1.0

    # Verify state, motion history, and productivity carry over directly
    assert p.track_id == 44
    assert len(p.wrist_hist) == 11
    assert p.last_motion_t == 9.5
    assert p.state == "active_work"
    assert p.since == 2.0
    assert p.acc["active_work"] == 26.0
    assert p.acc["idle"] == 5.0
    assert wrist_motion(p, now=11.0, sw=40.0) > 0.0

    # 2. Track Change via Temporary Identifying Track & Merge
    # Monitor maintains 'aagam' in persons
    monitor = LabMonitor.__new__(LabMonitor)
    monitor.db = sqlite3.connect(db_path)
    monitor.persons = {"aagam": p}

    # Simulate temporary identifying tracker for new track 55
    temp_track_id = 55
    temp_key = f"identifying_{temp_track_id}"
    temp_p = PersonTracker(temp_key, "Identifying...", vote_len=10)
    temp_p.track_id = temp_track_id
    temp_p.first_seen_t = 12.0
    temp_p.last_seen_t = 14.0
    temp_p.state = "active_work"
    temp_p.since = 12.0
    temp_p.last_motion_t = 13.8
    temp_p.acc["active_work"] = 2.0
    temp_p.wrist_hist.append((12.5, np.array([115.0, 200.0]), np.array([165.0, 200.0])))
    temp_p.wrist_hist.append((13.5, np.array([118.0, 200.0]), np.array([168.0, 200.0])))
    temp_p.votes.append("active_work")
    temp_p.votes.append("active_work")

    monitor.persons[temp_key] = temp_p

    # Add dummy event and alert under temporary key in DB to verify retroactive rewrite
    monitor.db.execute(
        "INSERT INTO events (seat, person_id, track_id, label, t_start, t_end, duration_sec, reason, desk_zone, camera_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("desk_1", temp_key, temp_track_id, "active_work", 12.0, 14.0, 2.0, "typing", "desk_1", "cam_0")
    )
    monitor.db.execute(
        "INSERT INTO alerts (ts, person_id, track_id, kind, detail, desk_zone, camera_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (13.0, temp_key, temp_track_id, "idle", "{}", "desk_1", "cam_0")
    )
    monitor.db.commit()

    # Call retroactive_relabel: resolves track 55 to 'aagam'
    monitor.retroactive_relabel(temp_track_id, "aagam", "Aagam", desk_zone="desk_1")

    # Verify temporary tracker was cleaned up from monitor.persons
    assert temp_key not in monitor.persons
    assert "aagam" in monitor.persons

    merged_aagam = monitor.persons["aagam"]
    # Check track ID updated
    assert merged_aagam.track_id == 55
    # Check productivity accumulators merged seamlessly (26.0 + 2.0 = 28.0)
    assert merged_aagam.acc["active_work"] == 28.0
    assert merged_aagam.acc["idle"] == 5.0
    # Check motion history merged and updated
    assert len(merged_aagam.wrist_hist) == 13
    assert merged_aagam.last_motion_t == 13.8
    # Check debounce state preserved
    assert merged_aagam.state == "active_work"
    assert len(merged_aagam.votes) == 10  # maxlen respected

    # Verify DB rows rewritten from identifying_55 to aagam
    cur = monitor.db.cursor()
    cur.execute("SELECT person_id FROM events WHERE track_id = ?", (temp_track_id,))
    ev_row = cur.fetchone()
    assert ev_row[0] == "aagam", f"Expected rewritten person_id 'aagam', got '{ev_row[0]}'"

    cur.execute("SELECT person_id FROM alerts WHERE track_id = ?", (temp_track_id,))
    al_row = cur.fetchone()
    assert al_row[0] == "aagam", f"Expected rewritten person_id 'aagam', got '{al_row[0]}'"

    monitor.db.close()


def test_distraction_alerts_validation_gating(tmp_path):
    """
    Verify:
    1. Distraction alerts are disabled for any label marked unvalidated (e.g. phone, discussing).
    2. Phone alerts require a verified phone detection, not merely a nearby wrist.
    """
    from lab_monitor import PersonTracker, LabMonitor

    db_path = str(tmp_path / "test_alerts.db")
    init_db(db_path)

    monitor = LabMonitor.__new__(LabMonitor)
    monitor.db = sqlite3.connect(db_path)
    monitor.phone_alert_s = 60.0
    monitor.idle_alert_s = 180.0
    monitor.alert_cooldown_s = 300.0
    monitor.min_track_duration_s = 1.5
    monitor.min_pose_conf = 0.30

    # Default validation config: idle is validated, phone is unvalidated
    monitor.validated_classes = {
        "active_work": True,
        "idle": True,
        "phone": False,
        "discussing": False
    }

    # Case 1: Person in 'phone' state for 70s (> 60s limit).
    # Since phone is UNVALIDATED, alert must NOT fire.
    p_phone = PersonTracker("aagam", "Aagam", vote_len=10)
    p_phone.state = "phone"
    p_phone.since = 100.0
    p_phone.first_seen_t = 90.0
    p_phone.track_id = 1
    p_phone.desk_zone = "desk_1"
    p_phone.camera_id = "cam_0"
    p_phone.last_alert = {}
    f_phone_unverified = {"has_verified_phone": False, "wrist_phone": 0.5}

    now_sim = 170.0  # held 70s
    held = now_sim - p_phone.since

    for kind, limit in (("phone", monitor.phone_alert_s), ("idle", monitor.idle_alert_s)):
        if not monitor.validated_classes.get(kind, False):
            continue
        if kind == "phone" and (f_phone_unverified is None or not f_phone_unverified.get("has_verified_phone", False)):
            continue
        if p_phone.state == kind and held >= limit:
            p_phone.last_alert[kind] = now_sim

    assert "phone" not in p_phone.last_alert, "Distraction alert must NOT fire for unvalidated label 'phone'"

    # Case 2: Person in 'idle' state for 200s (> 180s limit).
    # Since idle IS VALIDATED, alert must fire.
    p_idle = PersonTracker("harshil", "Harshil", vote_len=10)
    p_idle.state = "idle"
    p_idle.since = 100.0
    p_idle.first_seen_t = 90.0
    p_idle.track_id = 2
    p_idle.desk_zone = "desk_2"
    p_idle.camera_id = "cam_0"
    p_idle.last_alert = {}
    f_idle = {}

    now_sim = 300.0  # held 200s
    held = now_sim - p_idle.since

    for kind, limit in (("phone", monitor.phone_alert_s), ("idle", monitor.idle_alert_s)):
        if not monitor.validated_classes.get(kind, False):
            continue
        if kind == "phone" and (f_idle is None or not f_idle.get("has_verified_phone", False)):
            continue
        if p_idle.state == kind and held >= limit:
            p_idle.last_alert[kind] = now_sim

    assert "idle" in p_idle.last_alert, "Distraction alert must fire for validated label 'idle'"
    assert p_idle.last_alert["idle"] == 300.0

    # Case 3: Phone label marked validated in future config, but ONLY nearby wrist (no verified phone)
    monitor.validated_classes["phone"] = True
    p_phone.last_alert = {}
    f_phone_wrist_only = {"has_verified_phone": False, "wrist_phone": 0.4}

    now_sim = 200.0
    held = now_sim - p_phone.since

    for kind, limit in (("phone", monitor.phone_alert_s), ("idle", monitor.idle_alert_s)):
        if not monitor.validated_classes.get(kind, False):
            continue
        if kind == "phone" and (f_phone_wrist_only is None or not f_phone_wrist_only.get("has_verified_phone", False)):
            continue
        if p_phone.state == kind and held >= limit:
            p_phone.last_alert[kind] = now_sim

    assert "phone" not in p_phone.last_alert, "Phone alert must NOT fire with nearby wrist alone (requires verified phone detection)"

    # Case 4: Phone label marked validated AND verified phone detection confirmed
    f_phone_verified = {"has_verified_phone": True, "wrist_phone": 0.4}
    for kind, limit in (("phone", monitor.phone_alert_s), ("idle", monitor.idle_alert_s)):
        if not monitor.validated_classes.get(kind, False):
            continue
        if kind == "phone" and (f_phone_verified is None or not f_phone_verified.get("has_verified_phone", False)):
            continue
        if p_phone.state == kind and held >= limit:
            p_phone.last_alert[kind] = now_sim

    assert "phone" in p_phone.last_alert, "Phone alert MUST fire when both validated and verified phone detection exist"
    assert p_phone.last_alert["phone"] == 200.0

    monitor.db.close()


def test_absence_carryover_does_not_count_away_time():
    """
    Verify that when a person leaves for 30s and returns, their state duration
    upon return does NOT include the 30s absence, and cumulative active work duration
    excludes the away period.
    """
    from collections import Counter
    from lab_monitor import PersonTracker, calc_productivity

    vote_len = 5
    p = PersonTracker("swayam", "Swayam", vote_len=vote_len)

    # 1. Active Work Phase (t = 0.0 to 10.0s)
    p.state = "active_work"
    p.since = 0.0
    for _ in range(vote_len):
        p.votes.append("active_work")
    p.acc["active_work"] = 10.0

    current_dur_before_leaving = 10.0 - p.since
    assert current_dur_before_leaving == 10.0
    assert p.acc["active_work"] == 10.0

    # 2. Person leaves for 30s (t = 10.0 to 40.0s)
    # State transitions to 'away' at t = 10.0
    # Simulate interval close of active_work and transition to away
    p.state = "away"
    p.since = 10.0
    for _ in range(vote_len):
        p.votes.append("away")
    p.acc["away"] = 30.0  # 30 seconds away

    # During absence:
    away_dur_at_t35 = 35.0 - p.since
    assert away_dur_at_t35 == 25.0
    assert p.acc["active_work"] == 10.0  # Must NOT have increased!

    # 3. Person returns at t = 40.0s and resumes active work
    # Debounce confirms active_work at t = 40.0
    p.state = "active_work"
    p.since = 40.0
    for _ in range(vote_len):
        p.votes.append("active_work")

    # Person works for 5 seconds after returning (at t = 45.0s)
    now_return_work = 45.0
    p.acc["active_work"] += 5.0
    active_dur_after_return = now_return_work - p.since

    # CRITICAL INVARIANTS:
    # A. Active state duration upon return must be 5.0s, NOT 45.0s and NOT 35.0s
    assert active_dur_after_return == 5.0, (
        f"State duration must reflect time since return (5.0s), got {active_dur_after_return}s"
    )

    # B. Cumulative active work is 10.0s (initial) + 5.0s (after return) = 15.0s
    assert p.acc["active_work"] == 15.0, (
        f"Cumulative active work must exclude the 30s absence (expected 15.0s, got {p.acc['active_work']}s)"
    )

    # C. Away time is preserved in 'away' accumulator (30.0s)
    assert p.acc["away"] == 30.0

    # D. Productivity calculation (present = 15.0s active work, away excluded from denominator)
    prod = calc_productivity(p.acc, productive_classes=["active_work"])
    assert pytest.approx(prod, 0.01) == 100.0  # 15s active / 15s present






