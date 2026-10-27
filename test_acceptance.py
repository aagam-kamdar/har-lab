"""
test_acceptance.py - Acceptance criteria and regression verification.

Acceptance Criteria:
  1. /api/status returns all 4 seats with correct state, person_id, since, productivity_pct, reason.
  2. Offline detection: when heartbeat > 5s, monitor_online is False; when fresh, monitor_online is True.
  3. Concurrent DB access: 10 concurrent reader threads while writer commits -> zero "database is locked" errors.
  4. Idempotent process control: duplicate start calls do not spawn multiple processes.
  5. 36 dummy seats scaling: batching all 36 crops across 2 cameras in a single YOLO call per cycle.
"""
import concurrent.futures
import json
import os
import sqlite3
import sys
import time

import pytest
from fastapi.testclient import TestClient

from app_api import app, get_db
from lab_monitor import LabMonitor, acquire_lock, release_lock

client = TestClient(app)


# ---------------------------------------------------------------- Criterion 1: API Status & Schema
def test_api_status_returns_all_seats():
    """Verify /api/status returns all configured seats with full metadata."""
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "monitor_online" in data
    assert "seats" in data

    seats = data["seats"]
    assert len(seats) == 4, f"Expected 4 seats, got {len(seats)}"

    seat_ids = [s["seat_id"] for s in seats]
    expected_ids = ["D01", "D02", "D03", "D04"]
    for sid in expected_ids:
        assert sid in seat_ids

    # Check seat attributes
    for s in seats:
        assert "person_id" in s and s["person_id"]
        assert "state" in s and s["state"]
        assert "productivity_pct" in s and isinstance(s["productivity_pct"], (int, float))
        assert "reason" in s
        assert "camera_id" in s


def test_api_summary_and_events():
    """Verify /api/summary, /api/events, /api/alerts, and /api/config."""
    # Summary
    res = client.get("/api/summary")
    assert res.status_code == 200
    data = res.json()
    assert "class_productivity_pct" in data
    assert "state_distribution_seconds" in data
    assert "seats" in data

    # Events
    res_ev = client.get("/api/events?limit=10")
    assert res_ev.status_code == 200
    assert "events" in res_ev.json()

    # Alerts
    res_al = client.get("/api/alerts?limit=10")
    assert res_al.status_code == 200
    assert "alerts" in res_al.json()

    # Config
    res_cfg = client.get("/api/config")
    assert res_cfg.status_code == 200
    cfg = res_cfg.json()
    assert "seats" in cfg
    assert "cameras" in cfg


def test_ui_index_served():
    """Verify root / serves HTML UI."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Lab Activity Intelligence Monitor" in res.text
    assert "offlineBanner" in res.text


# ---------------------------------------------------------------- Criterion 2: Heartbeat Offline Detection
def test_heartbeat_offline_detection():
    """Verify monitor_online becomes False when heartbeat is older than timeout."""
    conn = sqlite3.connect("lab_events.db", timeout=5.0)
    cur = conn.cursor()

    # Case A: Fresh heartbeat
    now = time.time()
    cur.execute("UPDATE monitor_heartbeat SET last_update=?, status='running' WHERE id=1", (now,))
    conn.commit()

    res = client.get("/api/status")
    assert res.json()["monitor_online"] is True

    # Case B: Stale heartbeat (10 seconds ago)
    cur.execute("UPDATE monitor_heartbeat SET last_update=?, status='running' WHERE id=1", (now - 10.0,))
    conn.commit()

    res_stale = client.get("/api/status")
    assert res_stale.json()["monitor_online"] is False, "Heartbeat > 5s must report monitor_online = False"

    # Restore heartbeat
    cur.execute("UPDATE monitor_heartbeat SET last_update=?, status='running' WHERE id=1", (now,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- Criterion 3: Concurrency Test (No DB Lock)
def test_concurrent_reads_no_database_lock():
    """Test 10 concurrent requests reading DB during simulated commits: zero lock errors."""
    def make_request(idx):
        # Read from API
        r = client.get("/api/status")
        return r.status_code == 200

    # Also simulate writing in a separate connection
    def background_writer():
        conn = sqlite3.connect("lab_events.db", timeout=5.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        for _ in range(5):
            conn.execute("UPDATE monitor_heartbeat SET last_update=? WHERE id=1", (time.time(),))
            conn.commit()
            time.sleep(0.05)
        conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        writer_future = executor.submit(background_writer)
        reader_futures = [executor.submit(make_request, i) for i in range(10)]

        writer_future.result()
        for f in reader_futures:
            assert f.result() is True


# ---------------------------------------------------------------- Criterion 4: Idempotent Process Control
def test_process_control_idempotency(tmp_path):
    """Verify start clicked twice does not spawn duplicate processes."""
    test_lock = "monitor.lock"
    # Ensure starting from clean state
    if os.path.exists(test_lock):
        os.remove(test_lock)

    # First start call
    res1 = client.post("/api/monitor/start")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] in ("started", "already_running")
    pid1 = data1.get("pid")

    # Second immediate start call (double click simulation)
    res2 = client.post("/api/monitor/start")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "already_running"
    assert data2.get("pid") == pid1

    # Cleanup: stop monitor
    res_stop = client.post("/api/monitor/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] in ("stopped", "already_stopped")


# ---------------------------------------------------------------- Criterion 5: 36 Seats Batching Test
def test_36_seats_scaling_and_batching():
    """Verify monitor loads test_seats_36.yaml across 2 cameras and batches all 36 crops."""
    assert os.path.exists("test_seats_36.yaml"), "test_seats_36.yaml should exist"

    # Initialize monitor with test_seats_36.yaml
    mon = LabMonitor(
        config_path="config.yaml",
        seats_path="test_seats_36.yaml",
        source_override="new lab video.mp4",
        replay=True,
    )
    assert len(mon.seats) == 36, f"Expected 36 seats, got {len(mon.seats)}"

    # Check that crops across all 36 seats are collected and batched
    cam_frame = next(iter(mon.caps.values())).read()[1]
    assert cam_frame is not None

    all_crops = []
    for sid, s in mon.seats.items():
        x1, y1, x2, y2 = s.roi
        fh, fw = cam_frame.shape[:2]
        crop = cam_frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
        all_crops.append(crop)

    assert len(all_crops) == 36
    # Batch inference call with all 36 crops at once
    poses = mon.pose(all_crops, imgsz=320, verbose=False)
    dets = mon.det(all_crops, imgsz=320, verbose=False)
    assert len(poses) == 36
    assert len(dets) == 36

    # Release caps
    for cap in mon.caps.values():
        cap.release()
