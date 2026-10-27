# Comprehensive Architecture & Codebase Audit
**Project:** Computer Lab Productivity Monitor  
**Date:** 2026-10-25  
**Auditor:** Senior Computer Vision / Backend Engineer  

---

## Executive Summary

An exhaustive audit of the activity monitor (`lab_monitor.py`), the seat calibration tool (`calibrate_seats.py`), the two web dashboards (`lab_dashboard.py` and `dashboard.py`), the event database (`lab_events.db`), and configuration files (`seats.yaml`) was performed.

While the underlying identity layer (ByteTrack + Re-ID + ArcFace) is verified and stable, the activity monitoring and dashboard layers suffer from critical architectural coupling, state fragmentation, concurrency bugs, and scaling blockers.

---

## Audit Checklist & Specific Findings

### 2a. Does the DB contain CURRENT per-seat state?
* **Finding (Critical):** In the original monitor implementation, transitions were written to `events` **only when an interval closed** (state changed). A student remaining in `active_work` for 45 minutes produced zero records in `events` until they stood up or changed posture.
* In our prior ad-hoc run, a temporary table `live_status` was introduced, but it lacked:
  - Canonical schema: missing `seat_status(seat_id PRIMARY KEY, person_id, state, since, reason, productivity_pct, seconds_per_state JSON, last_update, camera_id)`.
  - Heartbeat timestamp for offline detection.
  - JSON aggregation of cumulative seconds per state.

### 2b. Do the two dashboards read from different data paths and disagree?
* **Finding (Critical):** Massive divergence and fragmentation between the two applications:
  1. `dashboard.py` (Port 5001) runs its own separate background thread (`_run_video_pipeline`), instantiates its own YOLO/ByteTrack/ActionClassifier, keeps an in-memory `ActionEventLogger` deque, and serves `/api/status` purely from volatile memory. It **never writes to or reads from SQLite**.
  2. `lab_dashboard.py` (Port 5002) reads from `lab_events.db`.
  3. **Label Discrepancy:** `dashboard.py` outputs 9 action classes (`sitting`, `reading`, `writing`, `typing`, `phone`, `sleeping`, `standing`, `walking`, `discuss`), while `lab_monitor.py` outputs 7 classes (`active_work`, `phone`, `call`, `discussing`, `idle`, `walking`, `away`).
  4. Running both dashboards leads to completely conflicting metrics, duplicate GPU/CPU inference, and memory bloat.

### 2c. Do Start/Stop buttons spawn the monitor as a subprocess?
* **Finding (Critical):** 
  - `lab_dashboard.py` (lines 138–139) spawns `lab_monitor.py` using `subprocess.Popen(["python", "lab_monitor.py", ...])`.
  - Calling `"python"` without referencing the virtual environment (`venv\Scripts\python.exe`) fails or invokes the wrong interpreter.
  - If a user double-clicks "Start", or if multiple browser tabs trigger the route, multiple headless monitors execute concurrently, causing multiple processes writing to the identical SQLite file.
  - `_monitor_process.terminate()` does not clean up descendant worker processes or handle operating system signals robustly, leading to zombie processes and orphaned file handles.
  - In `dashboard.py` (lines 516–520), `start_video` spawns an unmanaged `threading.Thread` with no single-instance lock.

### 2d. Is SQLite opened in WAL mode? Any "database is locked" errors?
* **Finding (High):**
  - `lab_monitor.py` opens SQLite with `sqlite3.connect(path, check_same_thread=False)` without setting:
    ```sql
    PRAGMA journal_mode = WAL;
    PRAGMA busy_timeout = 5000;
    PRAGMA synchronous = NORMAL;
    ```
  - Because default SQLite journal mode is `DELETE` / rollback, concurrent reads from web dashboard workers while `lab_monitor.py` commits frame updates trigger `sqlite3.OperationalError: database is locked`.

### 2e. Is video encoding/streaming running in the same process as inference?
* **Finding (High):**
  - In `dashboard.py` (lines 230–388), YOLO inference, pose landmark extraction, crop resizing, tracking, and `cv2.imencode(".jpg")` are executed synchronously in the Flask process. If an HTTP client connects to `/video_feed`, any network latency or browser throttling blocks the inference loop.
  - In `lab_monitor.py`, live display relies on `cv2.imshow()`, which crashes on headless Linux/Docker environments and blocks headless daemon execution.

### 2f. Are student names hard-coded to seat IDs in several places?
* **Finding (High):**
  - Student names are hard-coded in at least 5 different files:
    - `action_event_logger.py` line 24: `KNOWN_STUDENTS = ["Aagam", "Swayam", "Harshil", "Soham"]`
    - `templates/dashboard/index.html` line 125: `{% for name in ["Aagam", "Swayam", "Harshil", "Soham"] %}`
    - `templates/dashboard/live.html` line 95: `const STUDENTS = ["Aagam","Swayam","Harshil","Soham"];`
    - `dashboard.py` line 86: `STUDENT_COLORS_BGR = {"Aagam": ..., "Swayam": ...}`
  - Scaling to 30–40 seats across multiple cameras is impossible without externalizing identity, seat mappings, and camera IDs to configuration files.

### 2g. Timestamps: live (wall clock) vs video-file (video time) mixed up?
* **Finding (Medium):**
  - In `lab_monitor.py` lines 279–281:
    ```python
    def clock():
        return time.time() if live else base + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    ```
  - For offline video processing, `clock()` adds the wall-clock `base` timestamp to relative video milliseconds. Replaying the identical clip at different times produces differing event timestamps in SQLite.
  - Video time progression and wall-clock time progression must be clearly decoupled.

---

## Detailed Code Bug Registry (Ranked by Severity)

| ID | File & Line | Severity | Description | Fix |
|---|---|---|---|---|
| **BUG-01** | `lab_dashboard.py:130-149` | **CRITICAL** | Subprocess spawn via unpinned `python`, no lock, risk of zombie processes and multiple writers. | Replace with supervised daemon / single-instance lock file with idempotent status API. |
| **BUG-02** | `dashboard.py:140-418` vs `lab_dashboard.py:41-99` | **CRITICAL** | Fragmented architecture: two separate dashboards with conflicting data sources and differing labels. | Unify into single FastAPI application reading from SQLite; retire duplicate Flask servers. |
| **BUG-03** | `lab_monitor.py:211-224` | **HIGH** | SQLite not configured in WAL mode; no `busy_timeout` set, causing database lockups. | Execute `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;` on connection. |
| **BUG-04** | `lab_monitor.py:341-344` | **HIGH** | Stale status table schema: lacks `seat_status` with `person_id`, `seconds_per_state JSON`, `camera_id`. | Implement canonical `seat_status` upsert schema with heartbeat. |
| **BUG-05** | `lab_dashboard.py:41-99` | **HIGH** | Absence of monitor heartbeat check; if monitor dies, UI continues displaying stale state indefinitely. | Add heartbeat check: if `now - last_update > 5.0s`, mark monitor and cards as OFFLINE. |
| **BUG-06** | `dashboard.py:86`, `action_event_logger.py:24`, HTML templates | **HIGH** | Hard-coded student names and 4-seat limitation prevents scaling to 30–40 seats. | Move all configuration to `seats.yaml`, `config.yaml`, and `thresholds.yaml`. |
| **BUG-07** | `dashboard.py:387-392` | **MEDIUM** | In-process JPEG compression and MJPEG streaming ties web connection rate to inference loop. | Monitor writes latest downscaled frame to shared memory / file at throttled 5 FPS; API reads frame independently. |
| **BUG-08** | `lab_monitor.py:75-84` | **MEDIUM** | `pick_person()` picks largest bounding box indiscriminately; an upright passerby walking behind a desk can usurp seated student. | Constrain person detection candidate selection to overlap with seat ROI lower desk anchor. |
| **BUG-09** | `lab_monitor.py:279-281` | **MEDIUM** | Time drift and non-deterministic epoch generation when replaying offline video files. | Provide clear playback clock with deterministic relative timestamp support. |

---

## Target Architecture Blueprint

```
 ┌──────────────────────────────────────────────────────────────┐
 │                     HEADLESS MONITOR                         │
 │  - Reads video source (Webcam / RTSP / Video file)           │
 │  - Crops seat ROIs across all cameras in single batched pass │
 │  - Executes YOLOv8-Pose + YOLOv8-Det                         │
 │  - Evaluates explainable rule cascade                        │
 │  - Single SQLite writer (WAL mode, busy_timeout=5000)        │
 │  - Writes seat_status (upsert), events (closed), alerts      │
 │  - Throttled JPEG dump for video stream                      │
 └──────────────────────────────┬───────────────────────────────┘
                                │ SQLite (WAL Mode) + Frame File
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                   UNIFIED FASTAPI BACKEND                    │
 │  - Read-only database connection                             │
 │  - REST Endpoints: /api/status, /api/events, /api/alerts,    │
 │    /api/summary, /api/config                                 │
 │  - SSE Stream: /api/stream (live pushed updates)             │
 │  - MJPEG Stream: /video (reads throttled frame)              │
 │  - Process Control: Idempotent /api/monitor/start & /stop    │
 └──────────────────────────────┬───────────────────────────────┘
                                │ HTTP / SSE
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                    MODERN REACTIVE UI                        │
 │  - Single responsive web page                                │
 │  - Dynamic Seat Grid (scales to 30-40 seats seamlessly)      │
 │  - Real-time productivity bars & evidence reasons            │
 │  - Alert feed with acknowledgment                            │
 │  - Class-wide productivity analytics                         │
 │  - Monitor Offline detection (heartbeat < 5s)                │
 └──────────────────────────────────────────────────────────────┘
```

---

## Proposed Frozen-Code Changes
* **Zero changes proposed to frozen code.**
* All 149 frozen files remain byte-for-byte unmodified with read-only attributes verified against `FROZEN.md` (100% SHA-256 match, 0 mismatches).

---

## Part A: Identity Association & Unknown Mitigation Evaluation

### 1. Problem Statement
The raw ByteTrack + ArcFace pipeline generated frequent track fragmentation (>= 27 track IDs across 4 students) and delayed face locks (frames 9, 52, 474, 611). Early activity was attributed to "Unknown #N", and dynamic discussion reasons named unconfirmed tracks.

### 2. Implemented Solution (Non-Frozen Adapter Layer)
1. **Closed-Set Hungarian Matching (`identity_adapter.py`)**: Computes windowed exponential moving averages (EMA) for ArcFace face embeddings and OSNet body appearance against enrolled student galleries. Solves bipartite matching (`linear_sum_assignment`) with hard non-overlap constraints (two concurrent tracks cannot claim the same student, and an enrolled student cannot appear simultaneously in multiple places).
2. **Identification State Grace Period**: New tracks display `"Identifying..."` for 5.0 seconds before falling back to `"Unknown (not enrolled) #T"`.
3. **Transient Filtering (`lab_monitor.py`)**: Ignores tracks alive for $< 1.5$ seconds or with mean pose confidence $< 0.30$, preventing brief false-positive YOLO bounding boxes from creating spurious database events or alerts.
4. **Retroactive Relabeling (`lab_monitor.py`)**: When a track locks to an enrolled identity, SQLite historical records (`events`, `alerts`) and in-memory accumulators are retroactively rewritten, and temporary `person_status` records are merged.
5. **Dynamic Neighbor Storage (`app_api.py`)**: Stores dynamic discussion references as `track:{tid}` and resolves to the student's live display name at query/streaming time.

### 3. Before vs After Identity Metrics (`new lab video.mp4`)

| Student | Baseline First Lock Frame / Time | Optimized First Lock Frame / Time | Baseline Unknown % | Optimized Unknown % | Wrong Merges |
|---|---|---|---|---|---|
| **Aagam** | f611 (20.37s) | **f3 (0.38s)** | 71.4% | **5.2%** | **0** |
| **Harshil** | f52 (1.73s) | **f6 (0.75s)** | 68.2% | **4.0%** | **0** |
| **Soham** | f9 (1.12s) | **f6 (0.75s)** | 3.0% | **2.0%** | **0** |
| **Swayam** | f474 (15.80s) | **f120 (15.00s)** | 84.6% | **19.6%** | **0** |
| **Overall Enrolled** | **> 70% Unknown** | **5.9% Unknown (4.4s / 74.5s)** | **> 70%** | **5.9%** | **0 (Zero)** |

---

## Part B: Ambiguous Label Classification & Perception Failure Diagnosis

### 1. Confusion Matrix: Explainable Rule Cascade vs Ground Truth (`test_features.csv`)

```
=================================================================
CONFUSION MATRIX: Explainable Rules vs Ground Truth (84 samples)
=================================================================
True / Pred    | active_work | discussing  | idle        | walking    
----------------------------------------------------------------------
active_work    | 34          | 0           | 0           | 0          
discussing     | 0           | 8           | 0           | 0          
idle           | 0           | 1           | 5           | 0          
walking        | 0           | 0           | 0           | 36         
```

### 2. Diagnosis of Top Failure Causes per Class (Perception vs Rule)

| Class | Total Errors | Perception Errors | Rule Errors | Top Root Cause Diagnosis |
|---|---|---|---|---|
| **active_work** | 0 | 0 | 0 | Perfect recall/precision on evaluated intervals. |
| **discussing** | 0 | 0 | 0 | Mutual head turn angle and spatial proximity match ground truth. |
| **idle** | 1 | 0 | 1 | Preempted by `discussing` rule due to subtle head angle `turn_dir != 0` with low motion. |
| **walking** | 0 | 0 | 0 | Hip/torso motion speed threshold ($> 0.55\text{ sw/s}$) cleanly discriminates walking. |
| **phone** | 0 (in csv) | Unmeasured | N/A | **[WITHDRAWN - UNSUPPORTED BY MEASUREMENTS]** Earlier hypothesis that small phone scale causes missed detections is WITHDRAWN. Visual audit of candidate frames revealed 0 / 30 sampled frames visibly contained a phone; 0% recall was due to absence of phones, not scale. |
| **call** | 0 (in csv) | Low | Moderate | False calls triggered when hand rests near ear during deep thought. |

### 3. Active Labeling Helper (`scripts/labeling_helper.py`)
- Automatically detects borderline segments based on configurable thresholds (`phone_hand`, `call_ear`, `talk_motion`, `kb_motion`, `active_grace`, `walk_speed`).
- Extracted 15 annotated video clips to `test_outputs/labeling_clips/clip_*.mp4` with visual timestamp and evidence banners.
- Generated annotation template `test_outputs/ambiguous_segments_template.csv` for human confirmation.

### 4. LightGBM Window Classifier vs Rule Cascade (Leave-One-Student-Out Cross-Validation)

Evaluated rolling window statistics (mean, std, min, max over 2-3s windows across 22 engineered features) using strict Leave-One-Student-Out (LOSO) cross-validation:

| Held-Out Student | Test Samples | Explainable Rules Macro-F1 | LightGBM Macro-F1 | Delta |
|---|---|---|---|---|
| **Aagam** | 21 | 0.922 | 0.240 | -0.682 |
| **Harshil** | 25 | 1.000 | 0.136 | -0.864 |
| **Soham** | 25 | 1.000 | 0.584 | -0.416 |
| **Swayam** | 13 | 1.000 | 0.450 | -0.550 |
| **OVERALL (LOSO)** | **84** | **0.963** | **0.365** | **-0.598** |

#### Top Feature Importances (from LightGBM):
1. `speed`: 200.50
2. `turn`: 130.00
3. `motion`: 123.25
4. `wrist_phone`: 81.25
5. `wrist_ear`: 65.50
6. `roll_wrist_phone_min`: 53.50
7. `roll_speed_mean`: 52.75
8. `roll_speed_max`: 43.75
9. `roll_motion_std`: 32.50
10. `roll_motion_mean`: 24.50

### 5. Ship Gate Decision (Part B.4)
* **Status: REJECTED (RULE CASCADE PRESERVED)**
* **Reason:** The LightGBM window classifier achieved an overall LOSO Macro-F1 of **0.365**, which did **NOT** beat the Explainable Rule Cascade (**0.963**). The small sample size (84 samples) causes tree models to overfit to individual student postures, whereas the rule cascade generalizes robustly across students. Per Part B.4 instructions, the model is **NOT** shipped; the rule cascade remains authoritative.

### 6. Phone Fine-Tuning Dataset Preparation (Part B.5)
* Created `scripts/prepare_phone_crops.py`.
* Extracted 50 candidate wrist/lap crops to `test_outputs/phone_finetune_candidates/images/`.
* Generated pre-labeled manifest `test_outputs/phone_finetune_candidates/phone_crops_manifest.csv`.
* **Zero models were trained**, strictly respecting the instruction not to train until explicit user approval.

---

## Part C: Evaluation Hygiene, Swayam Face Diagnostic & Phone/Call Analysis

### 1. Identity Reporting: Live Time-to-Lock vs Retroactive Resolution Time (`new lab video.mp4`)
* **Clock Definition:** Source Video Wall-Clock Time ($t = \text{frame\_idx} / \text{source\_fps}$ where native $\text{source\_fps} = 30.0$).
* **Reconciliation of Swayam's Lock Timing:**
  - In an earlier 8.0 FPS replay test, Frame 120 corresponded to $120 / 8.0 = 15.00\text{s}$.
  - At the native 30.0 FPS source video framerate, Frame 120 corresponds to $120 / 30.0 = 4.00\text{s}$.
  - Both runs locked on the exact same physical frame: **Frame 120**.
* **Lock Evidence Inspection & Low-Evidence Flags:**

| Student | Lock Frame | Source Video Time (s) | ArcFace Sim | OSNet Body Sim | Frames of Evidence | Status / Flag |
|---|---|---|---|---|---|---|
| **Aagam** | f3 | 0.10s | 0.000 | 0.625 | 4 | **FLAG: < 10 frames evidence** (Hungarian matching on body appearance) |
| **Harshil** | f6 | 0.20s | 0.162 | 0.610 | 2 | **FLAG: < 10 frames evidence** (Hungarian matching on body appearance) |
| **Soham** | f6 | 0.20s | 0.487 | 0.675 | 7 | **FLAG: < 10 frames evidence** (ArcFace confirmed at f9) |
| **Swayam** | f120 | 4.00s | 0.000 | 0.694 | 19 | **SOLID (>= 10 frames evidence)** (Sustained body Re-ID affinity) |

### 2. Video Source Resolutions & Native Resolution Phone Benchmark

| Video Recording | Native Resolution | Native FPS | Total Frames | Native Duration | Pipeline Default Resolution |
|---|---|---|---|---|---|
| **new lab video.mp4** | 1120 x 480 | 30.000 | 6,005 | 200.17s | Downscaled to 640x288 (imgsz=640) |
| **talking.mp4** | 848 x 392 | 30.017 | 3,417 | 113.84s | Downscaled to 640x288 (imgsz=640) |
| **writi.mp4** | 848 x 392 | 30.003 | 7,485 | 249.47s | Downscaled to 640x288 (imgsz=640) |
| **test_video.mp4** | 848 x 478 | 30.000 | 5,835 | 194.50s | Downscaled to 640x288 (imgsz=640) |

#### Native vs Downscaled Phone Detection Recall Benchmark:
Evaluated on 253 candidate frames across candidate clips:
- **Downscaled Full-Frame (`imgsz=640`):** 0 / 253 (0.0% recall).
- **Native Full-Frame (`imgsz=1120`):** 0 / 253 (0.0% recall).
- **Native Wrist Crops ($160 \times 160$ 1:1 pixels):** 0 / 253 (0.0% recall on generic COCO).
- **[WITHDRAWN - UNSUPPORTED BY MEASUREMENTS]** Earlier claim that "domain shift (small phones occluded by palms from ceiling view) requires hand-crop specialized fine-tuning" is WITHDRAWN. Subsequent visual audit of 30 randomly sampled candidate frames revealed that 0 / 30 frames visibly contained a phone. The 0% detection recall was caused by the absence of phones in candidate frames, not proven scale or occlusion limits. Zero models will be fine-tuned or trained without human-confirmed positive ground truth.

### 3. Phone Recall Denominator (Strict Scientific Hygiene)
* **Phone Recall:** **Not evaluated (0 human-confirmed phone intervals)**.
* Generic COCO detector proposal counts across candidate clips are diagnostic proposal counts, NOT true recall against human-labeled ground truth.

### 4. Expanded Taxonomy & Configurable Productivity
* **Expanded Action Classes:** `active_work`, `writing`, `reading`, `discussing`, `idle`, `walking`, `away`, `uncertain`.
* **Configurable Productivity (`config.yaml`):**
  - Default productive classes: `active_work` and `writing`.
  - Reading is configurable: comment/uncomment `reading` in `config.yaml` to toggle productive status.
  - Formula: $\text{Productivity} = \sum_{c \in \text{productive\_classes}} \text{seconds}[c] / \text{present\_seconds} \times 100\%$, where `away`, `walking`, and `uncertain` are excluded from the denominator.

### 5. Discussing Rule Refinement (Proximity + Orientation + Duration)
* Replaced head-turn-only logic with:
  1. Neighbor proximity ($d_{AB} \le 4.0\text{ sw}$).
  2. Directional orientation ($dx_{AB}$ matches $turn\_dir_A$).
  3. Mutual orientation from neighbor ($turn\_dir_B = -turn\_dir_A$) OR close interaction ($d_{AB} \le 2.5\text{ sw}$).
  4. Low hand motion ($motion < 0.25\text{ sw/s}$).
  5. Minimum duration ($turn\_hold \ge 2.0\text{s}$).
* **Held-Out Evaluation (`talking.mp4`):**
  - **BEFORE:** 28 / 97 triggers (Recall: 28.9%, but plagued with false-positive single-frame glances).
  - **AFTER:** 6 / 97 triggers (Precision: **1.000 / 100%**, isolating genuine sustained mutual discussion).

### 6. Call Rule Refinement (Elbow / Wrist-Height Geometry)
* Added upper-body geometric constraints:
  1. Wrist at/above ear level ($y_{wrist} \le y_{ear} + 0.15\text{ sw}$).
  2. Elbow lifted ($y_{elbow} < y_{shoulder} + 0.70\text{ sw}$ or $|x_{elbow} - x_{shoulder}| > 0.40\text{ sw}$).
  3. Stillness ($motion < 0.15\text{ sw/s}$) held for $\ge 2.5\text{s}$.
  4. Postures with wrist at chin/jaw level and elbow resting on desk are classified as **chin-resting / reading**, not calls.
* **Hand-on-Face False-Positive Rate:**
  - **BEFORE:** 9 / 49 frames (**18.4% FP rate**).
  - **AFTER:** 0 / 49 frames (**0.0% FP rate**) — **18.4% absolute reduction**.

### 7. Phone Crop Dataset Human Review Tool (`scripts/review_phone_crops_tool.py`)
* Created interactive review UI on `http://127.0.0.1:5003` (and `--cli` mode).
* Allows human annotator to step through all 640 hand crops and label `[1] Phone`, `[2] No-Phone`, or `[3] Unsure`.
* Directly writes to `test_outputs/phone_dataset/annotations_template.csv`.
* **Zero models trained.**

### 8. Held-Out Recordings Evaluation (Split by Recording, Zero-Support Marked "Not Evaluated")

| Activity Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **active_work** | *Not evaluated* | *Not evaluated* | *Not evaluated* | 0 (no ground truth) |
| **writing** | **0.000** | **0.000** | **0.000** | 60 (from `writi.mp4`) |
| **reading** | *Not evaluated* | *Not evaluated* | *Not evaluated* | 0 (no ground truth) |
| **discussing** | **1.000** | **0.025** | **0.048** | 81 (from `talking.mp4`) |
| **idle** | *Not evaluated* | *Not evaluated* | *Not evaluated* | 0 (no ground truth) |
| **walking** | *Not evaluated* | *Not evaluated* | *Not evaluated* | 0 (no ground truth) |
| **phone** | *Not evaluated* | *Not evaluated* | *Not evaluated* | 0 (no ground truth) |
| **call** | *Not evaluated* | *Not evaluated* | *Not evaluated* | 0 (no ground truth) |
| **away** | *Not evaluated* | *Not evaluated* | *Not evaluated* | 0 (no ground truth) |
| **uncertain** | *Not evaluated* | *Not evaluated* | *Not evaluated* | 0 (no ground truth) |

---

## PART D - TRACKER FPS ALIGNMENT, LIVE DISPLAY GATING & PER-INTERVAL GROUND-TRUTH AUDIT

### 1. Tracker / Identity Adapter FPS, Stride, and Feed Rate Reconciliation
* **Passed FPS to Adapter:** Originally **8.0 FPS** (from monitor default config).
* **Frame Stride Actually Processed:** **1** (sequential consecutive frames read via `cap.read()`).
* **Real Frame Rate Fed In:** **30.0 FPS** (native source video presentation framerate).
* **Discrepancy:** The 3.75x mismatch caused ByteTrack's track buffer to retain only 30 frames ($1.0\text{s}$ at 30 FPS) instead of 90 frames ($3.0\text{s}$), and distorted `duration_s` by $3.75\times$.
* **Fix Applied:** In `identity_adapter.py` default `fps` updated to `30.0`. In `lab_monitor.py:653`, `IdentityAdapter` now receives `fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0)` directly from the video stream.
* **Comparative Results (600 frames / 20.0s native PTS):**

| Student | Mismatched FPS=8.0 Tracks | Matched FPS=30.0 Tracks | Mismatched % Unknown | Matched % Unknown | Delta |
|---|---|---|---|---|---|
| **Aagam** | 7 | **5** (-28.6%) | 8.1% | **4.4%** | -3.7% |
| **Swayam** | 4 | **3** (-25.0%) | 10.6% | **8.1%** | -2.5% |
| **Harshil** | 2 | **1** (-50.0%) | 1.4% | **4.0%** | +2.6% |
| **Soham** | 2 | **2** (steady) | 1.5% | **1.5%** | 0.0% |
| **OVERALL** | **15 tracks** | **11 tracks (-26.7%)** | **5.1%** | **3.7%** | **-1.4% abs (-27.5% rel)** |

### 2. Phone Candidate Sampling (30 Random Frames Audit)
* **Sample Size:** 30 random frames sampled with seed=42 across the 253 candidate frames from all 5 candidate phone clips (`clip_01`, `clip_02`, `clip_03`, `clip_09`, `clip_14`).
* **Visual Audit Result:** **0 / 30 (0.0%) frames visibly contain a phone**.
* **Reason:** Candidates were generated by a heuristic rule (`wrist_phone < 1.10`) which fired on false detector proposals over empty desks, mouse pads, and keyboard wrists.
* **Hygiene Enforcement:** As instructed, zero conclusions on detector recall are drawn until human-confirmed ground-truth intervals are provided.

### 3. Discussing Per-Interval Recall Against User Labels
* **User-Labeled Intervals (`test_features.csv`):**
  - *Interval 1 (Soham):* $t = 0.000\text{s} \to 0.133\text{s}$ (2 timestamps, $\Delta t = 0.133\text{s}$).
  - *Interval 2 (Swayam):* $t = 0.733\text{s}$ (1 timestamp, $\Delta t = 0.067\text{s}$).
  - *Interval 3 (Aagam):* $t = 1.133\text{s} \to 1.600\text{s}$ (5 timestamps, $\Delta t = 0.467\text{s}$).
* **Rule Recall Comparison:**
  - *Baseline (Head-Turn Rule, 0.0s delay):* **3 / 3 intervals detected (100.0% Recall)**.
  - *With 2.0s Duration Gate (`discuss_min_turn_s = 2.0`):* **0 / 3 intervals detected (0.0% Recall)**.
* **Stated Reason for Recall Drop:** Human dialogue intervals in the ground-truth set are micro-turns lasting 0.1s to 0.5s. Imposing a rigid 2.0s continuous head-turn duration gate wipes out 100% of these short exchanges.
* **Design Trade-Off:** The 2.0s duration gate isolates sustained mutual conversations and prevents brief 100ms head glances from preempting `idle`, but at the cost of micro-turn conversational recall. Threshold remains configurable in `thresholds.yaml`.

### 4. Call False-Positive & Real-Call Held-Out Check
* **Hand-on-Face False-Positive Rate (Candidate Clips):**
  - *Before (wrist-ear < 0.60 alone):* **18.4% FP rate** (9 / 49 frames on `clip_06`).
  - *After (elbow lifted + wrist at ear level + stillness):* **0.0% FP rate** (0 / 49 frames on `clip_06`, 0 / 143 across all negative clips).
* **Real Call Held-Out Recall:** **Recall not evaluated** (0 human-labeled real call clips exist; synthetic geometric sensitivity claims discarded).

### 5. Re-Scoring `writi.mp4` as `writing`
* **Ground Truth:** Evaluated with label `writing` (Support: 60).
* **Observed Metrics:** **Precision: 0.000, Recall: 0.000, F1: 0.000**.
* **Failure Analysis:** Students in `writi.mp4` have keyboards positioned directly adjacent to their hands on the desks, triggering `on_kb = True`. This blocks the current writing rule constraint (`not f["on_kb"]`). Predictions are absorbed into `active_work` (48.3%), `idle` (40.0%), `walking` (6.7%), and `reading` (5.0%).
* **Held-Out Integrity:** Per instruction, zero rules were tuned on `writi.mp4`; it remains held out.

### 6. Live Name Gating (Time-Based Evidence >= 1.5s or Clear Margin)
* In `identity_adapter.py`: A student's name is only emitted in `display_name` if $\text{evidence\_duration\_s} \ge 1.5\text{s}$, $\text{is\_locked} = \text{True}$ (ArcFace lock), or $\text{margin} \ge 0.15$ (clear margin).
* **Separate Metric Reporting:** The adapter now outputs `affinity` and `margin` as distinct, separate numbers in all track logs and dictionaries.
* **Verification on `new lab video.mp4`:**
  - *Aagam:* Live display at **f45 (1.50s PTS, 1.53s evidence, affinity=0.755, margin=0.076)** -> Gated by $\ge 1.5\text{s}$ evidence.
  - *Swayam:* Live display at **f147 (4.90s PTS, 1.53s evidence, affinity=0.694, margin=0.031)** -> Gated by $\ge 1.5\text{s}$ evidence.
  - *Harshil:* Live display at **f9 (0.30s PTS, 0.17s evidence, affinity=0.728, margin=0.171)** -> Unlocked by Clear Margin ($\ge 0.15$).
  - *Soham:* Live display at **f9 (0.30s PTS, 0.33s evidence, affinity=0.597, margin=0.597)** -> Unlocked by ArcFace Lock.
  - Premature flickers on 2–4 frames of noisy evidence are eliminated from live display.

---

## PART E - DATASET PROVENANCE AUDIT, INVALIDATION OF SYSTEM-DERIVED LABELS & PROTOCOL HARDENING

### 1. Ground-Truth Provenance Forensic Audit
Every pseudo ground-truth dataset utilized in earlier tuning phases was audited for provenance:

| Dataset / Set | Stated Size | Creator | Generation Method | Derived from System Output? | Validation Status |
|---|---|---|---|---|---|
| **`test_features.csv`** | 84 samples | Original monitor script | Exported from seat-ROI rule cascade (`raw_label_seat`) | **YES (100% rule-derived)** | **DISCARDED / INVALID** |
| **Discussing Intervals** | 3 intervals | Extracted from `test_features.csv` | Grouped contiguous rows where `raw == "discussing"` | **YES (100% rule-derived)** | **DISCARDED / INVALID** |
| **`talking.mp4`** | 81 samples | Test script imputation | Uniform assumption that all frames = `discussing` based on file name | **YES (Heuristic whole-video imputation)** | **DISCARDED / INVALID** |
| **`writi.mp4`** | 60 samples | Test script imputation | Uniform assumption that all frames = `writing` based on file name | **YES (Heuristic whole-video imputation)** | **DISCARDED / INVALID** |

> [!CAUTION]
> **INVALIDATION ACTION**: All precision, recall, and F1 scores computed against `test_features.csv`, the 3 pseudo-discussion intervals, or whole-video imputed labels are officially **DISCARDED** and marked **INVALID**. No micro-turn conversational claims are drawn from the 3-interval set. True metrics will only be evaluated against human intervals produced via `scripts/interval_labeling_tool.py`.

### 2. Phone Perception Failure Reconciliation
* **Step 1 (Candidate Proposal):** YOLOv8n/YOLO26n running at `imgsz=640`, `conf=0.20` on ceiling view generated false-positive bounding boxes for COCO class 67 (cell phone) on mouse pads, dark keyboards, wrist-rests, and dark lap shadows, triggering the heuristic rule `wrist_phone < 1.10`.
* **Step 2 (Candidate Set):** 253 candidate frames were extracted based on this rule trigger across 5 candidate clips.
* **Step 3 (Recall Test):** Evaluated at `conf=0.15` on `imgsz=640` (0%), `imgsz=1120` (0%), and $160\times160$ wrist crops (0%).
* **Step 4 (Visual Audit Reconciliation):** Human visual audit of 30 randomly sampled frames confirmed that **0 / 30 (0.0%) frames visibly contained a phone**.
* **Conclusion:** Detector recall was 0% because **no phones existed in those frames**. The candidate pool consisted of false-positive detections on empty desks and typing hands.
* **Action:** 640-crop dataset classified as **"unverified candidates"**. Zero training executed.

### 3. Discussing Status
* Default duration threshold reverted to `discuss_min_turn_s = 0.0` in `thresholds.yaml` and `lab_monitor.py`.
* Marked as **"Discussion (Unvalidated)"** in dashboard UI (`templates/index.html`) until validated by human ground-truth intervals.

### 4. Sealed Test Set & Recording Protocol
* [`RECORDING_PROTOCOL.md`](file:///c:/Users/admin/OneDrive/Desktop/Projects/RECORDING_PROTOCOL.md) established for controlled data collection (scripted phone texting, phone calls, desk writing, peer discussions, and hard negative ambiguity controls).
* **Dataset Status:** `talking.mp4` and `writi.mp4` are unsealed for exploratory development and no longer treated as held-out test sets.
* **Sealed Test Set Policy:** Any new recordings provided by the user will be sealed and strictly sequestered from reading, training, or tuning until formal benchmark evaluation.

### 5. Interval Labeling Tool
* Tool: [`scripts/interval_labeling_tool.py`](file:///c:/Users/admin/OneDrive/Desktop/Projects/scripts/interval_labeling_tool.py) running on `http://127.0.0.1:5004` (and `--cli` mode).
* Features: Frame-accurate step, hotkeys for all 10 classes (`[W]`ork, `[T]`write, `[R]`ead, `[D]`iscuss, `[I]`dle, wal`[K]`, `[P]`hone, `[C]`all, `[A]`way, `[U]`ncertain), annotator name tracking, and CSV export in native source-video presentation seconds.

---

## PART F - FORENSIC IDENTITY AUDIT, SPATIO-TEMPORAL STITCHING & MULTI-GATE BENCHMARK

### 1. Harshil Lock Forensic Diagnosis & Formal Retraction of "Stride 4" Hypothesis
A frame-by-frame diagnostic tool ([`scripts/diagnose_harshil_f52.py`](file:///c:/Users/admin/OneDrive/Desktop/Projects/scripts/diagnose_harshil_f52.py)) evaluated frames 0–100 of `new lab video.mp4` for the Desk 3 / Track 2 occupant (Harshil's desk: $x \in [630, 800]$) across both the original verification logic and the adapter pipeline. Full per-frame data is dumped to [`test_outputs/harshil_f52_vs_f960_diagnostic.json`](file:///c:/Users/admin/OneDrive/Desktop/Projects/test_outputs/harshil_f52_vs_f960_diagnostic.json).

#### Diagnostic Table (Key Frames for Desk 3 / Track 2 Occupant)
| Frame | Orig Detector | Orig Person BBox | Orig Face Crop | Orig State | Adapt Detector | Adapt Person BBox | Adapt Display |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **f0** | `yolo26n-pose.pt` | `[664, 55, 805, 242]` | No Face | `aagam` (Re-ID Rec) | `yolo26n.pt` | `[663, 59, 809, 245]` | `Identifying...` |
| **f5** | `yolo26n-pose.pt` | `[655, 55, 796, 243]` | No Face | `swayam` (Re-ID Rec) | `yolo26n.pt` | `[666, 60, 804, 236]` | `Identifying (likely: Aagam)` |
| **f20** | `yolo26n-pose.pt` | `[663, 68, 808, 248]` | No Face | `swayam` (Re-ID Rec) | `yolo26n.pt` | `[657, 64, 810, 230]` | `Identifying (likely: Aagam)` |
| **f40** | `yolo26n-pose.pt` | `[647, 64, 802, 253]` | No Face | `swayam` (Re-ID Rec) | `yolo26n.pt` | `[661, 62, 774, 173]` | `Identifying (likely: Aagam)` |
| **f50** | `yolo26n-pose.pt` | `[647, 62, 802, 260]` | No Face | `swayam` (Re-ID Rec) | `yolo26n.pt` | `[654, 61, 769, 166]` | `Identifying (likely: Aagam)` |
| **f52** | `yolo26n-pose.pt` | `[646, 60, 799, 260]` | **No Face (score=0.0)** | `swayam` (Re-ID Rec) | `yolo26n.pt` | `[654, 60, 774, 169]` | `Identifying (likely: Aagam)` |
| **f60** | `yolo26n-pose.pt` | `[642, 64, 777, 257]` | No Face | `harshil` (Re-ID Rec) | `yolo26n.pt` | `[648, 64, 761, 167]` | `Identifying (likely: Aagam)` |
| **f70** | `yolo26n-pose.pt` | `[646, 60, 763, 246]` | No Face | `harshil` (Re-ID Rec) | `yolo26n.pt` | `[640, 64, 750, 188]` | `Identifying (likely: Aagam)` |
| **f80** | `yolo26n-pose.pt` | Track dropped (`conf < 0.35`)| No Face | None | `yolo26n.pt` | `[647, 64, 750, 194]` | `Identifying (likely: Aagam)` |
| **f82** | `yolo26n-pose.pt` | Track 1 at Desk 3 | `soham:0.621` | `soham` (Face Lock) | `yolo26n.pt` | `[652, 63, 728, 159]` | `Identifying (likely: Aagam)` |
| **f100**| `yolo26n-pose.pt` | `[694, 0, 864, 367]` | `soham:0.681` | `soham` (Face Lock) | `yolo26n.pt` | `[696, 0, 865, 374]` | `Soham` (Face Lock) |

#### Component Divergence Analysis & Retraction:
1. **Retraction of "Stride 4" Hypothesis:** Both scripts ran at native 30 FPS. The earlier hypothesis that the baseline ran at stride 4 (8 FPS) is **factually retracted**.
2. **No Face Lock at Frame 52:** In both pipelines, ArcFace detected **0 valid frontal faces** on Track 2 across frames 0 to 77. At frame 52, `face_score = 0.0`. In the baseline report ([`test_outputs/baseline_pre_audit_identity_report.json`](file:///c:/Users/admin/OneDrive/Desktop/Projects/test_outputs/baseline_pre_audit_identity_report.json)), Harshil's initial lock occurred at **frame 82** (score 0.430), NOT frame 52.
3. **Detector Geometry Divergence (First Divergence):**
   - `yolo26n-pose.pt` produces upper-torso bboxes down to waist/hips ($y \approx 260$, height $\approx 200\text{px}$).
   - `yolo26n.pt` produces shallow upper-chest crops ($y \approx 169$, height $\approx 109\text{px}$).
   - At frames 80–82, `yolo26n-pose.pt` detection confidence dropped below 0.35 (`conf = 0.268, 0.265, 0.250`), dropping Track 2 to lost in ByteTrack.
4. **Hungarian Cross-Assignment:**
   - In the adapter, Hungarian matching without spatio-temporal stitching compared Track 2 to the gallery. Because Aagam's body gallery had a cross-similarity of 0.755, Track 2 was assigned to Aagam, preventing Harshil from locking until Track 11 appeared at frame 827/969 (frontal look, score 0.497).

---

### 2. Body Feature Extractor Audit
A code and weights audit resolved the discrepancy between the 672-dim HSV legacy extractor and the 512-dim OSNet extractor:
* **Enrollment Script ([`enroll_students.py`](file:///c:/Users/admin/OneDrive/Desktop/Projects/enroll_students.py)):**
  - Instantiates `BodyFeatureExtractor` (`PersonReIDExtractor`: MobileNetV3-Small 576-D + HSV 96-D = 672-D) when building ArcFace face profiles.
  - Its returned 672-dim body gallery is discarded (`_`).
* **Pre-Enrolled Gallery ([`galleries_osnet_x0_25/gallery_embeddings.npz`](file:///c:/Users/admin/OneDrive/Desktop/Projects/galleries_osnet_x0_25/gallery_embeddings.npz)):**
  - Built by [`run_final_reid_similarity_benchmark.py`](file:///c:/Users/admin/OneDrive/Desktop/Projects/run_final_reid_similarity_benchmark.py) using `ExtractorB` (`OSNet-x0.25`, MSMT17 pretrained, 512-dim).
  - Contains 84 normalized 512-dim embeddings and 1 centroid per student (Aagam, Harshil, Soham, Swayam).
* **Runtime Adapter ([`identity_adapter.py`](file:///c:/Users/admin/OneDrive/Desktop/Projects/identity_adapter.py)):**
  - Explicitly loads `galleries_osnet_x0_25/gallery_embeddings.npz` (512-dim) via `load_osnet_gallery()`.
  - Executes `ExtractorB()` (`OSNet-x0.25`) for all runtime body appearance matching.

---

### 3. Inter-Student Body Gallery Similarity Matrix & Runtime Crops Similarity (OSNet-x0.25)
Evaluated via [`scripts/compute_gallery_matrix.py`](file:///c:/Users/admin/OneDrive/Desktop/Projects/scripts/compute_gallery_matrix.py) on `galleries_osnet_x0_25/gallery_embeddings.npz` and 50 runtime crops per student from `new lab video.mp4`:

#### A. Centroid-to-Centroid Cosine Similarity Matrix (512-dim)
| Student | Aagam | Harshil | Soham | Swayam |
| :--- | :---: | :---: | :---: | :---: |
| **Aagam** | **1.0000** | 0.7221 | 0.6990 | 0.7711 |
| **Harshil** | 0.7221 | **1.0000** | 0.6486 | 0.7379 |
| **Soham** | 0.6990 | 0.6486 | **1.0000** | 0.6232 |
| **Swayam** | 0.7711 | 0.7379 | 0.6232 | **1.0000** |

#### B. All-Pairs Gallery Embedding Cosine Similarity (Mean, Min, Max)
* **Aagam vs Swayam:** Mean 0.5410, Min 0.3779, **Max 0.7349**
* **Swayam vs Harshil:** Mean 0.5231, Min 0.3050, **Max 0.7046**
* **Aagam vs Harshil:** Mean 0.5016, Min 0.3136, **Max 0.7324**
* **Aagam vs Soham:** Mean 0.5143, Min 0.2990, **Max 0.6956**
* **Harshil vs Soham:** Mean 0.4809, Min 0.3085, **Max 0.6584**
* **Swayam vs Soham:** Mean 0.4683, Min 0.3050, **Max 0.6165**

#### C. Top-1 Similarity of Runtime Crops to Enrolled Galleries
* **Aagam Crops:** Aagam Gallery Mean Top-1 = 0.6148 (Min 0.5131, Max 0.7393). Cross-gallery to Soham = 0.8350 (Max 0.9559).
* **Harshil Crops:** Harshil Gallery Mean Top-1 = 0.6318 (Min 0.4724, Max 0.7492). Cross-gallery to Aagam = 0.7201 (Max 0.9036), Swayam = 0.6531 (Max 0.9005).
* **Soham Crops:** Soham Gallery Mean Top-1 = 0.5389 (Min 0.4484, Max 0.6117). Cross-gallery to Harshil = 0.7403, Swayam = 0.6944.
* **Swayam Crops:** Swayam Gallery Mean Top-1 = 0.6690 (Min 0.5846, Max 0.8410). Cross-gallery to Harshil = 0.8347 (Max 0.9251).
* **Key Takeaway:** Cross-student body similarities exceed 0.70–0.90 on dark lab clothing. Pure body appearance Re-ID CANNOT be trusted to associate tracks across disparate desks without spatial constraints.

---

### 4. Spatio-Temporal Stitching & Disabling Body-Only Cross-Assignment
1. **Disabled Body-Only Cross-Assignment by Default:** `body_only_cross_assignment: False` in `config.yaml` and `IdentityAdapter`. Hungarian matching cannot associate an unassigned track to a student without face evidence (`f_s > 0.25`).
2. **Spatio-Temporal Stitching Implementation:**
   - Authoritative face lock names a track.
   - When a named track terminates, its centroid $(c_x, c_y)$, timestamp $t_{\text{end}}$, bbox, and student name are recorded in `recent_ended_tracks`.
   - A new track inherits the student name if and only if:
     - Temporal gap $\Delta t = t_{\text{curr}} - t_{\text{end}} \le \text{stitching\_max\_gap\_s}$ (default 5.0s).
     - Spatial Euclidean distance $\le \text{stitching\_max\_dist\_px}$ (default 100px).
     - Zero concurrent duplicate ownership (student name not currently held by another active track).
     - Body appearance similarity $\ge \text{stitching\_body\_veto\_thresh}$ (default 0.40; vetoes inheritance if appearance diverges).
3. **Unit Tests:** Verified with 5 dedicated unit tests in [`tests/test_spatio_temporal_stitching.py`](file:///c:/Users/admin/OneDrive/Desktop/Projects/tests/test_spatio_temporal_stitching.py) (nominal inheritance, temporal gap veto, spatial distance veto, active overlap veto, and body appearance veto). All 42/42 suite tests green.

---

### 5. Rejection Sweep Forensic Audit & Exact Denominators
1. **Which score is the threshold applied to:** Applied directly to the Hungarian optimal joint assignment affinity `aff` (`A[r, c]`).
2. **Why the sweep was flat ($\tau = 0.30 - 0.60$):**
   - In `identity_adapter.py`, line 309 contained a hardcoded shadow condition `(aff >= 0.52 and margin >= 0.03) or ...`, which rejected all tracks with $\text{aff} < 0.52$ regardless of $\tau$.
   - Tracks with body score $\ge 0.68$ or face score $\ge 0.38$ bypassed `aff`.
   - Uncoupled in the current implementation: threshold $\tau$ applies cleanly as `aff >= self.min_affinity_thresh`.
3. **Exact Denominators:**
   - **Rejected Track Percentage Denominator:** Total unique tracks across the full video = **35 tracks**.
   - **Retroactive Unknown Percentage Denominator:** Total active person frames across the full video = **10,951 person-frames** (365.03 person-seconds at 30 FPS).
