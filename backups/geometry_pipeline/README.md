# Geometry-Based Activity Recognition Backup

This directory contains the untouched backup of the geometry-based activity recognition pipeline created prior to migrating to the pretrained Human Activity Recognition (HAR) model (`MCG-NJU/videomae-small-finetuned-kinetics`).

## Backed-Up Files

| File | Purpose in Original Geometry Pipeline |
|------|---------------------------------------|
| `lab_monitor.py` | Complete geometry-based activity monitor (YOLO-Pose keypoints, shoulder-width heuristics, wrist-to-ear, wrist-to-phone, dynamic discussion, writing/reading/calling/idle rules, state machine, SQLite events, and alerts) |
| `config.yaml` | Application & monitor configuration with thresholds path, debounce window, identity gate, and validation registry flags |
| `thresholds.yaml` | Geometric thresholds (kp_conf, call_ear, call_hold, phone_hand, head_down, turn, talk_motion, kb_motion, etc.) |
| `action_classifier.py` | Standalone 17-keypoint COCO geometry action classifier |
| `action_event_logger.py` | Frame-level action bout and alert logger |
| `test_lab_monitor.py` | Unit tests for geometry activity recognition rules |
| `MANIFEST.json` | SHA-256 hashes and file metadata for verification |

## How to Roll Back to Geometry Pipeline

### Method 1: Configuration Switch (Instant Zero-Code Rollback)
The active application supports configuration-driven rollback. In `config.yaml`:
```yaml
activity_recognition:
  mode: "geometry"
```
When `mode` is set to `"geometry"`, the monitor activates the preserved geometry engine and runs exactly as it did before.

### Method 2: Physical File Restoration
To completely restore the repository to the pre-HAR state:
```powershell
Copy-Item backups/geometry_pipeline/lab_monitor.py lab_monitor.py -Force
Copy-Item backups/geometry_pipeline/config.yaml config.yaml -Force
Copy-Item backups/geometry_pipeline/thresholds.yaml thresholds.yaml -Force
Copy-Item backups/geometry_pipeline/action_classifier.py action_classifier.py -Force
Copy-Item backups/geometry_pipeline/action_event_logger.py action_event_logger.py -Force
Copy-Item backups/geometry_pipeline/test_lab_monitor.py test_lab_monitor.py -Force
```
After restoration, verify the test suite:
```powershell
.\venv\Scripts\python.exe -m pytest tests/ test_identity_lock.py test_lab_monitor.py test_acceptance.py
```
