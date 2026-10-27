# Person-Centric Computer Lab Productivity Intelligence Monitor

A decoupled, scalable computer vision system for real-time workstation productivity monitoring and classroom activity analytics powered by ByteTrack + OSNet Re-ID + ArcFace identity tracking and YOLOv8 pose and object perception.

Activity, productivity, and state **follow the identified person**, not static seat rectangles.

---

## 🏛️ System Architecture

The system is decoupled into three independent layers communicating via SQLite in WAL mode:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   HEADLESS MONITOR (Writer Process)                    │
│  - Video source / RTSP camera feeds                                    │
│  - Frozen Identity Pipeline via `identity_adapter.py`:                 │
│      * ByteTrack multi-object tracking                                 │
│      * ArcFace face recognition & feature extraction                   │
│      * OSNet Re-ID appearance matching & gallery management            │
│      * IdentityLockManager with permanent stickiness                   │
│  - Full-Frame YOLOv8-Pose + YOLOv8-Det (classes: phone, kb, laptop, tv)│
│  - Shoulder-width ($sw$) scale-invariant keypoint geometry             │
│  - Dynamic Neighbor Discussion matching (runtime heading & mutual gaze)│
│  - Walking detection: upright posture or hip speed, independent of ROI │
│  - Debounced state machine (70% majority gate across rolling window)   │
│  - Sole writer to SQLite (WAL mode):                                   │
│      * Upserts canonical `person_status` with JSON seconds_per_state   │
│      * Backwards-compatible `seat_status` upsert for desk zones        │
│      * Appends closed intervals to `events`                            │
│      * Appends threshold breaches to `alerts`                          │
│      * Writes `monitor_heartbeat` every cycle                          │
│  - Atomic downscaled JPEG publish (`static/live_frame.jpg`)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ SQLite (WAL Mode, query_only API)
┌───────────────────────────────────▼────────────────────────────────────┐
│                         FASTAPI BACKEND (API)                          │
│  - Strictly read-only against DB (PRAGMA query_only)                   │
│  - GET /api/status (Current person & seat states, heartbeat)           │
│  - GET /api/summary (Classroom aggregates & charts, isolates unknowns) │
│  - GET /api/events & GET /api/alerts (Filterable by person/zone)       │
│  - GET /api/config (Desk zones, cameras, thresholds)                   │
│  - GET /api/stream (Server-Sent Events real-time push)                 │
│  - GET /video (MJPEG feed decoupled from inference)                    │
│  - POST /api/monitor/start & stop (Idempotent process control)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP & SSE
┌───────────────────────────────────▼────────────────────────────────────┐
│                  REACTIVE SINGLE-PAGE UI (Dashboard)                   │
│  - Responsive grid supporting 4 to 40+ student cards                   │
│  - Displays Student Name, Track ID, ArcFace Confidence, Desk Zone      │
│  - Dynamic top banner for "Monitor Offline" (> 5s stale heartbeat)     │
│  - Live state badges, productivity bars, human-readable evidence text  │
│  - Chart.js time distribution doughnut chart                           │
│  - Live alerts feed with dismissal                                     │
│  - Live video stream toggle                                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Environment Setup
Activate the Python virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Start the FastAPI Web Service & Dashboard
Launch the unified web API (serves UI on port 5002):
```powershell
python app_api.py
```
Open your browser to: **`http://localhost:5002`**

### 3. Start the Headless Activity Monitor
Run in a separate terminal or use the web dashboard "Start Monitor" button:

```powershell
# Live monitoring on default camera / video:
python lab_monitor.py

# High-speed replay mode to rapidly populate database:
python lab_monitor.py --replay --source "new lab video.mp4" --force

# Stop after processing N frames:
python lab_monitor.py --replay --source "new lab video.mp4" --max-frames 200
```

Alternatively, use the process control scripts:
```powershell
python scripts/start_monitor.py
python scripts/stop_monitor.py
```

---

## ⚙️ Configuration Files

All parameters, mappings, thresholds, and camera definitions are externalized in YAML:

| File | Purpose |
|---|---|
| [`config.yaml`](config.yaml) | Application ports, database paths, frame publish rates, and camera sources. |
| [`desk_zones.yaml`](desk_zones.yaml) | Optional workstation ROI coordinates (D01–D04) for contextual reporting only. |
| [`thresholds.yaml`](thresholds.yaml) | Shoulder-width normalized geometric thresholds for activity classification. |
| [`FROZEN.md`](FROZEN.md) | Checksum manifest of all 149 frozen identity pipeline files. |

---

## 🏷️ Person-Centric Activity Labels & Logic

Activities are classified per tracked person using explainable geometric rules normalized by shoulder-width ($sw$):

- **`active_work`**: Hands positioned at desk/keyboard level with active wrist motion or within the grace window ($6.0\text{s}$). If a student changes desks, productivity follows the person.
- **`phone`**: Hand within $1.10\ sw$ of detected phone box and head tilted downward.
- **`call`**: Wrist within $0.60\ sw$ of ear keypoint held for $\ge 2.5\text{s}$.
- **`discussing`**: Dynamic neighbor check: computes line-of-sight heading toward nearby individuals and flags mutual vs non-mutual conversation (e.g. `turned toward Harshil (mutual=True)`). No fixed neighbor graphs required.
- **`idle`**: Seated with minimal hand motion and no object interaction.
- **`walking`**: Hip centroid speed $> 0.55\ sw/\text{s}$ or upright standing posture. Works dynamically throughout the room without seat ROI boundaries.
- **`away`**: Person not observed by camera for $\ge 10\text{s}$, distinct from students who have not yet arrived.

---

## ⚡ Performance & Scale Benchmark

Benchmark measured on full-frame perception + person tracking, dynamic discussion matching, and debounce state machine (`benchmark_person_scale.py`):

| Scale | Full-Frame Perception | Dynamic Discussion ($O(N^2)$) | Rule Cascade & Debounce | Total Cycle Latency | Effective Throughput |
|---|---|---|---|---|---|
| **4 Persons** | 119.0 ms | < 0.01 ms | 0.02 ms | 119.0 ms | **8.4 FPS** |
| **20 Persons** | 119.0 ms | < 0.01 ms | 0.16 ms | 119.1 ms | **8.4 FPS** |
| **40 Persons** | 119.0 ms | < 0.01 ms | 0.32 ms | 119.3 ms | **8.4 FPS** |

*Note: Because YOLO perception runs on the full frame once per cycle rather than per seat crop, increasing tracked persons from 4 to 40 adds less than 0.35 ms of CPU logic overhead.*

---

## 🧪 Testing & Verification

Run the full automated test suite covering unit logic and acceptance criteria:

```powershell
pytest test_lab_monitor.py test_acceptance.py -v
```

### Verified Acceptance Criteria:
1. **Person-Centric State**: `person_status` tracks state, identity confidence, desk zone, duration, productivity %, and human-readable evidence reasons.
2. **Offline Detection**: Stale heartbeat ($> 5\text{s}$) immediately flags `monitor_online: false` and displays the top alert banner in the UI; recovers automatically on restart.
3. **Database Concurrency**: Zero "database is locked" errors under concurrent readers/writers (SQLite WAL mode + busy timeout).
4. **Idempotent Control**: Duplicate calls to `/api/monitor/start` spawn exactly one process instance.
5. **Frozen Identity Integrity**: All 149 frozen files verified byte-for-byte with identical SHA-256 checksums; identity regression passed with 0 ID switches.
6. **Desk Swap Invariance**: Simulated desk swaps preserve individual student productivity without cross-contamination.
7. **Unknown Person Isolation**: Unidentified individuals are tracked as `Unknown #N` on dashboard cards but isolated from class-wide aggregate productivity metrics.
