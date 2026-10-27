"""
app_api.py - Unified FastAPI service for Computer Lab Productivity Monitor.

Replaces duplicate Flask servers with a single, decoupled, read-only service:
  - Reads SQLite (WAL mode, PRAGMA query_only = ON)
  - Serves REST API (/api/status, /api/events, /api/alerts, /api/summary, /api/config)
  - Real-time updates via Server-Sent Events (/api/stream)
  - Decoupled MJPEG video feed (/video) reading static/live_frame.jpg without running YOLO
  - Idempotent monitor process management (/api/monitor/start, /api/monitor/stop)
  - Single-page responsive UI (4 to 40 seats)
"""
import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from typing import Optional, Dict, List, Set, Tuple

import cv2
import numpy as np
import psutil
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

CONFIG_FILE = "config.yaml"
LOCK_FILE = "monitor.lock"

# Load initial config
with open(CONFIG_FILE, "r") as f:
    CONFIG = yaml.safe_load(f)

APP_CFG = CONFIG.get("app", {})
DB_PATH = APP_CFG.get("db_path", "lab_events.db")
FRAME_PATH = APP_CFG.get("frame_dump_path", "static/live_frame.jpg")
HEARTBEAT_TIMEOUT = float(APP_CFG.get("heartbeat_timeout_s", 5.0))
PORT = int(APP_CFG.get("port", 5002))
HOST = APP_CFG.get("host", "0.0.0.0")

app = FastAPI(title="Lab Productivity Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Global validation status mapping (loaded from validation_registry.yaml)
VALIDATED_CLASSES = {
    "active_work": False,
    "writing": False,
    "reading": False,
    "discussing": False,
    "idle": False,
    "walking": False,
    "phone": False,
    "call": False,
    "away": False,
    "uncertain": False
}
if os.path.exists("validation_registry.yaml"):
    try:
        with open("validation_registry.yaml", "r", encoding="utf-8") as f:
            reg = yaml.safe_load(f)
            if reg and "validated_classes" in reg:
                VALIDATED_CLASSES.update(reg["validated_classes"])
    except Exception:
        pass
elif os.path.exists("config.yaml"):
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
            VALIDATED_CLASSES.update(c.get("monitor", {}).get("validated_classes", {}))
    except Exception:
        pass


def get_db():
    """Strictly read-only database connection with WAL mode and busy timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA query_only = ON;")
    return conn


CURRENT_MONITOR_PROC = None


def get_monitor_process():
    """Check if monitor is currently running via active process ref or lock file."""
    global CURRENT_MONITOR_PROC
    if CURRENT_MONITOR_PROC is not None and CURRENT_MONITOR_PROC.poll() is None:
        try:
            return psutil.Process(CURRENT_MONITOR_PROC.pid)
        except Exception:
            pass

    if not os.path.exists(LOCK_FILE):
        return None
    try:
        with open(LOCK_FILE, "r") as f:
            data = json.load(f)
        pid = data.get("pid")
        if pid and psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            if proc.is_running() and "python" in proc.name().lower():
                return proc
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- REST Endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join("templates", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>UI template not found. Please create templates/index.html</h1>", status_code=404)


@app.get("/api/config")
async def get_config():
    with open(CONFIG_FILE, "r") as f:
        cfg = yaml.safe_load(f)

    desk_zones_file = cfg.get("monitor", {}).get("desk_zones_path", "desk_zones.yaml")
    zones_data = {}
    if os.path.exists(desk_zones_file):
        with open(desk_zones_file, "r") as f:
            zones_data = yaml.safe_load(f).get("zones", {})

    seats_file = cfg.get("monitor", {}).get("seats_path", "seats.yaml")
    seats_data = {}
    if os.path.exists(seats_file):
        with open(seats_file, "r") as f:
            seats_data = yaml.safe_load(f).get("seats", {})

    return {
        "app": cfg.get("app", {}),
        "monitor": cfg.get("monitor", {}),
        "cameras": cfg.get("cameras", {}),
        "desk_zones": zones_data,
        "seats": seats_data or zones_data,
    }


# ---------------------------------------------------------------- Helper: Resolve Evidence Text
def resolve_evidence_text(text: str, track_to_name: Dict[str, str]) -> str:
    """Resolve track:{tid} in evidence reasons to the student/person's display name."""
    if not text:
        return text
    def replace_fn(match):
        tid = match.group(1)
        return track_to_name.get(tid, f"Track #{tid}")
    return re.sub(r"track:(\d+)", replace_fn, text)


@app.get("/api/status")
async def get_status():
    now = time.time()
    monitor_online = False
    last_heartbeat = None
    persons_list = []
    seats_list = []

    try:
        conn = get_db()
        cur = conn.cursor()

        # Check heartbeat
        cur.execute("SELECT last_update, fps, active_seats, status FROM monitor_heartbeat WHERE id=1")
        hb = cur.fetchone()
        if hb:
            last_heartbeat = hb["last_update"]
            if (now - last_heartbeat) <= HEARTBEAT_TIMEOUT and hb["status"] == "running":
                monitor_online = True

        # Fetch person status (person-centric)
        cur.execute("""
            SELECT person_id, display_name, track_id, state, since, reason, productivity_pct,
                   seconds_per_state, desk_zone, camera_id, identity_confidence, last_seen, last_update
            FROM person_status
            ORDER BY
                CASE WHEN person_id LIKE 'unknown%' OR person_id LIKE 'identifying%' THEN 1 ELSE 0 END,
                display_name ASC
        """)
        rows_p = cur.fetchall()

        # Build track_to_name mapping
        track_to_name = {str(r["track_id"]): r["display_name"] for r in rows_p if r["track_id"] is not None}

        for r in rows_p:
            sec_dict = {}
            if r["seconds_per_state"]:
                try:
                    sec_dict = json.loads(r["seconds_per_state"])
                except Exception:
                    pass

            duration_s = max(0.0, round(now - r["since"], 1)) if r["since"] else 0.0
            resolved_reason = resolve_evidence_text(r["reason"] or "", track_to_name)

            persons_list.append({
                "person_id": r["person_id"],
                "display_name": r["display_name"] or r["person_id"],
                "track_id": r["track_id"],
                "state": r["state"],
                "since": r["since"],
                "duration_s": duration_s,
                "is_validated": bool(VALIDATED_CLASSES.get(r["state"], False)),
                "validation_status": "validated" if VALIDATED_CLASSES.get(r["state"], False) else "unvalidated",
                "reason": resolved_reason,
                "productivity_pct": round(r["productivity_pct"] or 0.0, 1),
                "seconds_per_state": sec_dict,
                "desk_zone": r["desk_zone"],
                "camera_id": r["camera_id"] or "cam_0",
                "identity_confidence": r["identity_confidence"],
                "last_seen": r["last_seen"],
                "last_update": r["last_update"],
            })

        # Fetch legacy seat_status for backward-compatibility
        cur.execute("""
            SELECT seat_id, person_id, state, since, reason, productivity_pct,
                   seconds_per_state, last_update, camera_id
            FROM seat_status
            ORDER BY seat_id ASC
        """)
        rows_s = cur.fetchall()
        for r in rows_s:
            sec_dict = {}
            if r["seconds_per_state"]:
                try:
                    sec_dict = json.loads(r["seconds_per_state"])
                except Exception:
                    pass
            duration_s = max(0.0, round(now - r["since"], 1)) if r["since"] else 0.0
            seats_list.append({
                "seat_id": r["seat_id"],
                "person_id": r["person_id"],
                "state": r["state"],
                "since": r["since"],
                "duration_s": duration_s,
                "reason": r["reason"] or "",
                "productivity_pct": round(r["productivity_pct"] or 0.0, 1),
                "seconds_per_state": sec_dict,
                "last_update": r["last_update"],
                "camera_id": r["camera_id"] or "cam_0",
            })

        conn.close()
    except Exception as e:
        return JSONResponse({"monitor_online": False, "error": str(e), "persons": [], "seats": []}, status_code=500)

    return {
        "monitor_online": monitor_online,
        "last_heartbeat": last_heartbeat,
        "active_persons_count": len(persons_list),
        "label_validation": VALIDATED_CLASSES,
        "persons": persons_list,
        "seats": seats_list or persons_list,
        "timestamp": now,
    }


@app.get("/api/labels")
async def get_label_validation():
    """Return per-label validation status and taxonomy metadata."""
    return {
        "validation": VALIDATED_CLASSES,
        "labels": [
            {"id": k, "name": k.replace("_", " ").title(), "validated": v}
            for k, v in VALIDATED_CLASSES.items()
        ]
    }


@app.get("/api/live")
async def get_live_alias():
    """Backward-compatibility alias for older dashboard views requesting /api/live."""
    status = await get_status()
    if isinstance(status, JSONResponse):
        return status

    alerts_res = await get_alerts(limit=8)
    alerts_list = alerts_res.get("alerts", []) if isinstance(alerts_res, dict) else []

    seats_dict = {}
    for s in status.get("seats", []):
        seats_dict[s["seat_id"]] = {
            "seat": s["seat_id"],
            "student": s["person_id"],
            "state": s["state"],
            "since": s["since"],
            "productivity": s["productivity_pct"],
            "reason": s["reason"],
            "last_updated": s["last_update"],
        }

    return {
        "running": status.get("monitor_online", False),
        "monitor_online": status.get("monitor_online", False),
        "seats": seats_dict,
        "seats_list": status.get("seats", []),
        "alerts": alerts_list,
    }


@app.get("/api/events")
async def get_events(since: Optional[float] = None, limit: int = 50):
    try:
        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT track_id, display_name FROM person_status WHERE track_id IS NOT NULL")
        track_to_name = {str(r["track_id"]): r["display_name"] for r in cur.fetchall()}

        if since is not None:
            cur.execute("""
                SELECT id, person_id, track_id, label, t_start, t_end, duration_sec, reason, desk_zone, camera_id, seat
                FROM events
                WHERE t_start >= ?
                ORDER BY t_start DESC LIMIT ?
            """, (since, limit))
        else:
            cur.execute("""
                SELECT id, person_id, track_id, label, t_start, t_end, duration_sec, reason, desk_zone, camera_id, seat
                FROM events
                ORDER BY t_start DESC LIMIT ?
            """, (limit,))

        rows = cur.fetchall()
        events = []
        for r in rows:
            ev = dict(r)
            ev["reason"] = resolve_evidence_text(ev.get("reason") or "", track_to_name)
            events.append(ev)
        conn.close()
        return {"events": events}
    except Exception as e:
        return JSONResponse({"error": str(e), "events": []}, status_code=500)


@app.get("/api/alerts")
async def get_alerts(since: Optional[float] = None, limit: int = 30):
    try:
        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT track_id, display_name FROM person_status WHERE track_id IS NOT NULL")
        track_to_name = {str(r["track_id"]): r["display_name"] for r in cur.fetchall()}

        if since is not None:
            cur.execute("""
                SELECT id, ts, person_id, track_id, kind, detail, desk_zone, camera_id, seat
                FROM alerts
                WHERE ts >= ?
                ORDER BY ts DESC LIMIT ?
            """, (since, limit))
        else:
            cur.execute("""
                SELECT id, ts, person_id, track_id, kind, detail, desk_zone, camera_id, seat
                FROM alerts
                ORDER BY ts DESC LIMIT ?
            """, (limit,))

        rows = cur.fetchall()
        alerts = []
        for r in rows:
            detail_obj = {}
            if r["detail"]:
                try:
                    detail_obj = json.loads(r["detail"])
                except Exception:
                    detail_obj = {"text": r["detail"]}
            if detail_obj.get("reason"):
                detail_obj["reason"] = resolve_evidence_text(detail_obj["reason"], track_to_name)
            alerts.append({
                "id": r["id"],
                "ts": r["ts"],
                "person_id": r["person_id"],
                "track_id": r["track_id"],
                "seat": r["seat"] or r["desk_zone"] or "",
                "desk_zone": r["desk_zone"],
                "kind": r["kind"],
                "detail": detail_obj,
            })
        conn.close()
        return {"alerts": alerts}
    except Exception as e:
        return JSONResponse({"error": str(e), "alerts": []}, status_code=500)


@app.get("/api/summary")
async def get_summary():
    """Aggregate per-person and class-wide productivity statistics."""
    try:
        conn = get_db()
        cur = conn.cursor()

        # Query person_status
        cur.execute("SELECT person_id, display_name, productivity_pct, seconds_per_state, desk_zone FROM person_status")
        rows = cur.fetchall()

        # Fallback to seat_status if person_status is empty
        if not rows:
            cur.execute("SELECT seat_id AS person_id, person_id AS display_name, productivity_pct, seconds_per_state, seat_id AS desk_zone FROM seat_status")
            rows = cur.fetchall()

        conn.close()

        class_seconds = {
            "active_work": 0.0, "phone": 0.0, "call": 0.0,
            "discussing": 0.0, "idle": 0.0, "walking": 0.0, "away": 0.0
        }
        person_summaries = []

        for r in rows:
            s_dict = {}
            if r["seconds_per_state"]:
                try:
                    s_dict = json.loads(r["seconds_per_state"])
                except Exception:
                    pass

            # Only aggregate into classroom stats if not an unknown person to avoid polluting named stats
            pid = r["person_id"]
            if not pid.startswith("unknown"):
                for k in class_seconds:
                    class_seconds[k] += s_dict.get(k, 0.0)

            person_summaries.append({
                "person_id": r["person_id"],
                "display_name": r["display_name"] or r["person_id"],
                "productivity_pct": round(r["productivity_pct"] or 0.0, 1),
                "seconds_per_state": s_dict,
                "desk_zone": r["desk_zone"],
                "seat_id": r["desk_zone"] or r["person_id"],
            })

        total_present = sum(v for k, v in class_seconds.items() if k not in ("away", "walking"))
        class_productivity = (100.0 * class_seconds["active_work"] / total_present) if total_present > 1.0 else 0.0

        return {
            "class_productivity_pct": round(class_productivity, 1),
            "total_present_seconds": round(total_present, 1),
            "state_distribution_seconds": {k: round(v, 1) for k, v in class_seconds.items()},
            "persons": person_summaries,
            "seats": person_summaries,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------- SSE Real-Time Stream
@app.get("/api/stream")
async def event_stream(request: Request):
    """Server-Sent Events pushing status every 1 second."""
    async def sse_generator():
        while True:
            if await request.is_disconnected():
                break

            status_data = await get_status()
            if isinstance(status_data, dict):
                yield f"data: {json.dumps(status_data)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------- Decoupled Video Stream
async def frame_generator():
    """MJPEG stream generator reading static/live_frame.jpg without executing YOLO."""
    placeholder = None
    while True:
        frame_bytes = None
        if os.path.exists(FRAME_PATH):
            try:
                with open(FRAME_PATH, "rb") as f:
                    frame_bytes = f.read()
            except Exception:
                pass

        if not frame_bytes:
            if placeholder is None:
                img = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(img, "Waiting for monitor camera feed...", (120, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
                _, buf = cv2.imencode(".jpg", img)
                placeholder = buf.tobytes()
            frame_bytes = placeholder

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        await asyncio.sleep(0.125)  # ~8 FPS


@app.get("/video")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------- Process Control
@app.post("/api/monitor/start")
async def start_monitor():
    global CURRENT_MONITOR_PROC
    proc = get_monitor_process()
    if proc is not None:
        return {"status": "already_running", "pid": proc.pid}

    python_exe = sys.executable
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    try:
        CURRENT_MONITOR_PROC = subprocess.Popen(
            [python_exe, "lab_monitor.py"],
            cwd=os.getcwd(),
            creationflags=creationflags
        )
        await asyncio.sleep(0.3)
        return {"status": "started", "pid": CURRENT_MONITOR_PROC.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitor: {e}")


@app.post("/api/monitor/stop")
async def stop_monitor():
    global CURRENT_MONITOR_PROC
    proc = get_monitor_process()
    CURRENT_MONITOR_PROC = None
    if proc is None:
        # Also clean up stale lock file if any
        if os.path.exists(LOCK_FILE):
            try:
                os.remove(LOCK_FILE)
            except Exception:
                pass
        return {"status": "already_stopped"}

    try:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except psutil.TimeoutExpired:
            proc.kill()
    except Exception:
        pass

    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass

    return {"status": "stopped"}


def main():
    display_host = "127.0.0.1" if HOST == "0.0.0.0" else HOST
    print(f"Starting Lab Monitor API: open your browser at http://{display_host}:{PORT}")
    uvicorn.run("app_api:app", host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
