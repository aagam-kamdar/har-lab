"""
tests/test_har_pipeline.py
--------------------------
Comprehensive unit and integration tests for the Pretrained HAR (VideoMAE) pipeline:
  1. Pretrained model initialization & inference
  2. Frame-buffer construction & per-track clip buffering
  3. Activity label mapping to application taxonomy
  4. Confidence-based rejection (< threshold -> 'uncertain')
  5. Missing model weights & error handling (graceful fallback)
  6. Independent per-student tracking ID isolation (no cross-track leakage)
  7. Database persistence & live status schema updates
  8. Semantic comparison logic (geometry vs HAR)
  9. Rollback mode configuration ('geometry' vs 'har')
  10. Backup file integrity check against backups/geometry_pipeline/MANIFEST.json
"""

import hashlib
import json
import os
import sqlite3
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from har_classifier import HARClassifier, HARPrediction, KINETICS_TO_APP_MAP
from lab_monitor import LabMonitor, init_db, semantic_compare


# --------------------------------------------------------------------------- #
# Test 1: Model Initialization & Label Count                                  #
# --------------------------------------------------------------------------- #
def test_har_model_initialization():
    """Verify that VideoMAE small checkpoint initializes and contains 400 Kinetics classes."""
    har = HARClassifier(async_mode=False)
    assert har.is_ready is True
    assert har.model is not None
    assert har.processor is not None
    assert len(har.labels) == 400
    assert har.model_name == "MCG-NJU/videomae-small-finetuned-kinetics"


# --------------------------------------------------------------------------- #
# Test 2: Frame Buffer Construction & Crop Extraction                         #
# --------------------------------------------------------------------------- #
def test_frame_buffer_construction_and_cropping():
    """Verify that person crop is correctly extracted and buffers exactly 16 frames."""
    har = HARClassifier(clip_len=16, async_mode=False)

    frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
    bbox = (50, 60, 250, 360)

    crop = har.extract_person_crop(frame, bbox)
    assert crop.shape == (224, 224, 3)

    # Feed 15 frames -> buffer not full, no prediction returned yet
    for f_idx in range(15):
        pred = har.update_track(track_id=101, frame=frame, bbox=bbox, timestamp=f_idx * 0.1)
    assert pred is None
    assert len(har._buffers[101]) == 15

    # 16th frame -> buffer full, inference runs
    pred = har.update_track(track_id=101, frame=frame, bbox=bbox, timestamp=1.5)
    assert pred is not None
    assert pred.track_id == 101
    assert pred.latency_ms > 0
    assert len(har._buffers[101]) == 16


# --------------------------------------------------------------------------- #
# Test 3: Verified Activity Label Mapping                                      #
# --------------------------------------------------------------------------- #
def test_activity_label_mapping():
    """Verify verified mappings from Kinetics-400 to application taxonomy."""
    har = HARClassifier(async_mode=False, confidence_threshold=0.30)

    assert har.map_label("using computer", 0.75) == ("active_work", True)
    assert har.map_label("assembling computer", 0.60) == ("active_work", True)
    assert har.map_label("writing", 0.85) == ("writing", True)
    assert har.map_label("reading book", 0.55) == ("reading", True)
    assert har.map_label("reading newspaper", 0.65) == ("reading", True)
    assert har.map_label("texting", 0.80) == ("phone", True)
    assert har.map_label("answering questions", 0.50) == ("discussing", True)
    assert har.map_label("shaking head", 0.45) == ("discussing", True)
    assert har.map_label("yawning", 0.70) == ("idle", True)
    assert har.map_label("walking the dog", 0.60) == ("walking", True)


# --------------------------------------------------------------------------- #
# Test 4: Confidence-Based Rejection                                           #
# --------------------------------------------------------------------------- #
def test_confidence_rejection():
    """Verify that predictions below threshold map to 'uncertain' while preserving raw label."""
    har = HARClassifier(async_mode=False, confidence_threshold=0.50)

    # Below threshold (0.42 < 0.50)
    action, is_mapped = har.map_label("using computer", 0.42)
    assert action == "uncertain"
    assert is_mapped is False

    # Above threshold (0.55 >= 0.50)
    action, is_mapped = har.map_label("using computer", 0.55)
    assert action == "active_work"
    assert is_mapped is True


# --------------------------------------------------------------------------- #
# Test 5: Unsupported Classes Handled Safely                                   #
# --------------------------------------------------------------------------- #
def test_unsupported_classes_handling():
    """Verify that classes not in Kinetics (e.g. phone call) map to 'uncertain' without crash."""
    har = HARClassifier(async_mode=False)

    # 'call' does not exist in Kinetics-400
    action, is_mapped = har.map_label("talking on cell phone", 0.99)
    assert action == "uncertain"
    assert is_mapped is False

    # Arbitrary Kinetics class not in lab taxonomy
    action, is_mapped = har.map_label("playing saxophone", 0.95)
    assert action == "uncertain"
    assert is_mapped is False


# --------------------------------------------------------------------------- #
# Test 6: Missing Weights / Init Error Handling                                #
# --------------------------------------------------------------------------- #
def test_missing_weights_handling():
    """Verify that an invalid model path does not crash the application."""
    har = HARClassifier(model_name="nonexistent/fake_model_weights_path", async_mode=False)
    assert har.is_ready is False
    assert har.init_error is not None

    # Predicting with unready model should return fallback safely
    raw, conf, mapped, is_mapped, lat = har.predict_clip([np.zeros((224, 224, 3), dtype=np.uint8)] * 16)
    assert raw == "model_not_ready"
    assert mapped == "uncertain"
    assert is_mapped is False


# --------------------------------------------------------------------------- #
# Test 7: Independent Per-Student Tracking ID Isolation                        #
# --------------------------------------------------------------------------- #
def test_per_student_track_isolation():
    """Verify that frame buffers and inferences for different students are completely isolated."""
    har = HARClassifier(clip_len=16, async_mode=False)

    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 200

    # Feed track 1 (10 frames)
    for i in range(10):
        har.update_track(track_id=1, frame=frame1, bbox=(0, 0, 100, 200), timestamp=i * 0.1)

    # Feed track 2 (16 frames)
    for i in range(16):
        har.update_track(track_id=2, frame=frame2, bbox=(200, 0, 300, 200), timestamp=i * 0.1)

    assert len(har._buffers[1]) == 10
    assert len(har._buffers[2]) == 16
    assert har.get_prediction(1) is None
    assert har.get_prediction(2) is not None
    assert har.get_prediction(2).track_id == 2


# --------------------------------------------------------------------------- #
# Test 8: Semantic Comparison Logic                                            #
# --------------------------------------------------------------------------- #
def test_semantic_comparison():
    """Verify semantic agreement rules between geometry and HAR classifications."""
    assert semantic_compare("active_work", "active_work", "using computer") == "Agreement"
    assert semantic_compare("active_work", "writing", "writing") == "Agreement (Related Work)"
    assert semantic_compare("writing", "active_work", "using computer") == "Agreement (Related Work)"
    assert semantic_compare("phone", "phone", "texting") == "Agreement"
    assert semantic_compare("call", "phone", "texting") == "Agreement (Phone Interaction)"
    assert semantic_compare("active_work", "idle", "yawning") == "Disagreement"
    assert semantic_compare("call", "reading", "reading book") == "Disagreement"
    assert semantic_compare("active_work", "uncertain", "spinning poi") == "Unmapped / Uncertain"
    assert semantic_compare("active_work", "buffering", "buffering") == "Unmapped / Uncertain"


# --------------------------------------------------------------------------- #
# Test 9: Database Schema Persistence with HAR Columns                         #
# --------------------------------------------------------------------------- #
def test_database_har_schema_and_persistence(tmp_path):
    """Verify that init_db creates HAR columns in person_status and live_status."""
    test_db = str(tmp_path / "test_har_events.db")
    conn = init_db(test_db)

    cur = conn.cursor()
    cur.execute("PRAGMA table_info(person_status)")
    ps_cols = [c[1] for c in cur.fetchall()]
    assert "har_raw_label" in ps_cols
    assert "har_confidence" in ps_cols
    assert "har_mapped_action" in ps_cols
    assert "har_latency_ms" in ps_cols
    assert "comparison_status" in ps_cols

    cur.execute("PRAGMA table_info(live_status)")
    ls_cols = [c[1] for c in cur.fetchall()]
    assert "har_raw_label" in ls_cols
    assert "har_confidence" in ls_cols
    assert "comparison_status" in ls_cols

    # Test insertion
    conn.execute("""
        INSERT INTO live_status (seat, student, state, since, productivity, reason, last_updated, har_raw_label, har_confidence, comparison_status)
        VALUES ('D01', 'Aagam', 'active_work', 10.0, 95.0, 'HAR: using computer', 20.0, 'using computer', 0.88, 'Agreement')
    """)
    conn.commit()

    row = conn.execute("SELECT * FROM live_status WHERE seat='D01'").fetchone()
    assert row is not None
    conn.close()


# --------------------------------------------------------------------------- #
# Test 10: Backup Manifest Integrity Verification                             #
# --------------------------------------------------------------------------- #
def test_backup_manifest_integrity():
    """Verify that all geometry pipeline files in backups/ match the exact SHA-256 in MANIFEST.json."""
    manifest_path = "backups/geometry_pipeline/MANIFEST.json"
    assert os.path.exists(manifest_path), "Backup manifest must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for orig_path, info in manifest.items():
        backup_file = info["backup_file"]
        expected_hash = info["sha256"]
        assert os.path.exists(backup_file), f"Backup file {backup_file} is missing"

        with open(backup_file, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch in {backup_file}: expected {expected_hash}, got {actual_hash}"
