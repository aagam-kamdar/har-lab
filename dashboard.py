"""
dashboard.py
------------
Flask dashboard for Student Productivity Tracker with Action Detection.

Provides:
  GET  /              → Live dashboard (video + student cards + charts)
  GET  /stream        → Server-Sent Events stream for real-time updates
  GET  /api/status    → JSON snapshot of current state
  GET  /api/summary   → JSON full session summary
  GET  /api/bouts     → JSON list of all action bouts
  GET  /api/alerts    → JSON list of all alert events
  POST /api/acknowledge_alert?idx=N  → Acknowledge an alert
  GET  /analyze_video → Form to run action detection on uploaded/lab video
  POST /analyze_video → Start analysis job
  GET  /webcam        → Live webcam mode

Run:
    python dashboard.py
"""

from __future__ import annotations

import os
import sys
import json
import time
import queue
import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for, stream_with_context

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from action_classifier import ActionClassifier, find_nearby_students, ACTION_COLORS_BGR, ACTION_EMOJI
from action_event_logger import ActionEventLogger, KNOWN_STUDENTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Dashboard")

# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #

VIDEO_PATH = os.environ.get("VIDEO_PATH", "new lab video.mp4")
YOLO_POSE_MODEL = os.environ.get("YOLO_POSE_MODEL", "yolov8n-pose.pt")
YOLO_DETECT_MODEL = os.environ.get("YOLO_DETECT_MODEL", "yolo26n-pose.pt")
WEBCAM_INDEX = int(os.environ.get("WEBCAM_INDEX", "0"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))   # 0 = unlimited
FPS_CAP = float(os.environ.get("FPS_CAP", "30"))
PROXIMITY_PX = float(os.environ.get("PROXIMITY_PX", "150"))

OUTPUT_DIR = "static/results/dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("templates/dashboard", exist_ok=True)

app = Flask(__name__)

# Global state
_state = {
    "running": False,
    "mode": "idle",           # "idle" | "video" | "webcam"
    "frame_idx": 0,
    "fps": 30.0,
    "last_jpeg": None,        # latest JPEG bytes for MJPEG stream
    "event_logger": None,
    "action_classifier": None,
    "error": None,
    "thread": None,
}
_state_lock = threading.Lock()

# SSE event queue
_sse_queue: queue.Queue = queue.Queue(maxsize=50)


# --------------------------------------------------------------------------- #
#  Annotation helpers                                                           #
# --------------------------------------------------------------------------- #

STUDENT_COLORS_BGR = {
    "Aagam":   (94, 234, 212),
    "Swayam":  (244, 114, 182),
    "Harshil": (129, 140, 248),
    "Soham":   (251, 191, 36),
    "UNKNOWN": (128, 128, 128),
}

def annotate_frame(
    frame: np.ndarray,
    tracks: list,   # [{student, action, conf, bbox, track_id}]
) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        student = t.get("student", "UNKNOWN")
        action = t.get("action", "sitting")
        conf = t.get("confidence", 0.0)
        x1, y1, x2, y2 = t.get("bbox", (0, 0, 100, 100))
        tid = t.get("track_id", 0)

        s_color = STUDENT_COLORS_BGR.get(student, (128, 128, 128))
        a_color = ACTION_COLORS_BGR.get(action, (180, 180, 180))

        # BBox
        cv2.rectangle(out, (x1, y1), (x2, y2), s_color, 2)

        # Top label: "Track N | StudentName"
        label1 = f"#{tid} {student}"
        # Bottom label: "action emoji + action name"
        emoji = ACTION_EMOJI.get(action, "")
        label2 = f"{action} ({int(conf*100)}%)"

        lh = 22
        cv2.rectangle(out, (x1, y1 - lh*2), (x2, y1), s_color, -1)
        cv2.putText(out, label1, (x1 + 4, y1 - lh - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10,10,10), 1, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y2), (x2, y2 + lh), a_color, -1)
        cv2.putText(out, label2, (x1 + 4, y2 + lh - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (10,10,10), 1, cv2.LINE_AA)

    return out


# --------------------------------------------------------------------------- #
#  Core processing pipeline                                                     #
# --------------------------------------------------------------------------- #

def _load_identity_pipeline():
    """Load the existing ByteTrack + Re-ID pipeline components (read-only use)."""
    from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack
    from features import FaceFeatureExtractor, BodyGallery
    from identity_lock_manager import IdentityLockManager
    from enroll_students import KNOWN_STUDENTS as KS, FACE_FOLDER, augment_image

    return BYTETracker, BYTETrackerArgs, BaseTrack, FaceFeatureExtractor, BodyGallery, IdentityLockManager

def _run_video_pipeline(video_path: str, use_webcam: bool = False):
    """
    Main processing loop: runs YOLO-Pose + ByteTrack + action classification.
    Writes annotated JPEG frames to _state["last_jpeg"] for MJPEG stream.
    Updates _state["event_logger"] with live action data.
    """
    from ultralytics import YOLO
    from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack

    logger.info("Loading YOLO-Pose model: %s", YOLO_POSE_MODEL)
    pose_model = YOLO(YOLO_POSE_MODEL)

    # Load identity components (non-destructive)
    try:
        from features import FaceFeatureExtractor, BodyGallery
        from identity_lock_manager import IdentityLockManager
        from enroll_students import KNOWN_STUDENTS as KS, FACE_FOLDER
        from reid.embedding_model import PersonReIDExtractor
        import faiss
        import pathlib

        face_extractor = FaceFeatureExtractor(det_size=(320, 320))
        face_profiles = {}
        for s in KS:
            folder = pathlib.Path(f"face/{s}")
            imgs = [cv2.imread(str(p)) for p in folder.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and "back" not in p.name.lower()]
            embs = []
            for im in imgs:
                if im is not None:
                    e = face_extractor.extract_from_image(im)
                    if e is not None:
                        embs.append(e)
            if embs:
                avg_e = np.mean(embs, axis=0)
                face_profiles[s] = (avg_e / np.linalg.norm(avg_e)).astype(np.float32)

        lock_mgr = IdentityLockManager()
        body_extractor = PersonReIDExtractor()

        # Load body gallery
        from reid.enrolled_gallery import EnrolledGallery
        body_gallery = EnrolledGallery(data_dir="reid/data")
        has_identity = True
        logger.info("Identity pipeline loaded. Face profiles: %s", list(face_profiles.keys()))
    except Exception as e:
        logger.warning("Identity pipeline not available (%s), using track IDs only.", e)
        has_identity = False
        face_profiles = {}
        lock_mgr = None
        body_gallery = None
        body_extractor = None
        face_extractor = None

    # ByteTrack
    BaseTrack.reset_id()
    bt_args = BYTETrackerArgs()
    bt_args.track_thresh = 0.35
    bt_args.track_buffer = 90
    bt_args.match_thresh = 0.85
    bt_args.mot20 = False
    tracker = BYTETracker(bt_args, frame_rate=30)

    # Action components
    action_clf = ActionClassifier(smooth_window=15)
    fps_val = 30.0

    if use_webcam:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        logger.info("Opening webcam index %d", WEBCAM_INDEX)
    else:
        cap = cv2.VideoCapture(video_path)
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        logger.info("Opening video: %s (%.1f fps)", video_path, fps_val)

    event_logger = ActionEventLogger(fps=fps_val)

    with _state_lock:
        _state["event_logger"] = event_logger
        _state["action_classifier"] = action_clf
        _state["fps"] = fps_val

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    t_loop_start = time.time()
    last_sse_push = 0.0

    while _state["running"]:
        ret, frame = cap.read()
        if not ret:
            if use_webcam:
                continue
            break

        # --- YOLO Pose detection ---
        results = pose_model(frame, classes=[0], verbose=False)[0]

        dets = []
        all_kps = {}    # tid → keypoints array after tracking

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                dets.append([x1, y1, x2, y2, conf])

        det_arr = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)
        online_tracks = tracker.update(det_arr, [vid_h, vid_w], [vid_h, vid_w])

        # Associate keypoints with tracks (by bbox IoU)
        pose_kps_list = []
        if results.keypoints is not None:
            try:
                kps_data = results.keypoints.data  # (N, 17, 3)
                pose_kps_list = kps_data.cpu().numpy() if hasattr(kps_data, 'cpu') else np.array(kps_data)
            except Exception:
                pose_kps_list = []

        def _match_kps_to_track(track_bbox, det_boxes_np, kps_list):
            """Find best-matching keypoint set for a track bbox."""
            if len(kps_list) == 0 or len(det_boxes_np) == 0:
                return None
            tx1, ty1, tx2, ty2 = track_bbox
            best_iou, best_kps = 0.0, None
            for i, db in enumerate(det_boxes_np):
                dx1, dy1, dx2, dy2 = db[:4]
                ix1, iy1 = max(tx1, dx1), max(ty1, dy1)
                ix2, iy2 = min(tx2, dx2), min(ty2, dy2)
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                union = (tx2-tx1)*(ty2-ty1) + (dx2-dx1)*(dy2-dy1) - inter
                iou = inter / max(union, 1e-6)
                if iou > best_iou and i < len(kps_list):
                    best_iou = iou
                    best_kps = kps_list[i]
            return best_kps if best_iou > 0.3 else None

        # Build named identity map
        track_to_student: Dict[int, str] = {}
        current_bboxes: Dict[str, tuple] = {}

        track_annotations = []

        for t in online_tracks:
            tid = int(t.track_id)
            tlwh = t.tlwh
            x1 = max(0, int(tlwh[0]))
            y1 = max(0, int(tlwh[1]))
            x2 = min(vid_w, int(tlwh[0] + tlwh[2]))
            y2 = min(vid_h, int(tlwh[1] + tlwh[3]))
            bbox = (x1, y1, x2, y2)

            # Named identity
            student = "UNKNOWN"
            if has_identity and lock_mgr is not None:
                tracked_ids = [int(t2.track_id) for t2 in tracker.tracked_stracks if t2.is_activated]
                lost_ids = [int(t2.track_id) for t2 in tracker.lost_stracks]
                removed_ids = [int(t2.track_id) for t2 in tracker.removed_stracks]
                lock_mgr.update_lifecycle(tracked_ids, lost_ids, removed_ids, frame_idx)
                state_obj = lock_mgr.update_seen(tid, frame_idx)
                student = state_obj.identity_label() if state_obj.locked else "UNKNOWN"

                # Face matching
                if face_extractor and face_profiles:
                    face_emb = face_extractor.extract_from_crop(frame, (x1, y1, x2, y2), padding=0.25)
                    if face_emb is not None:
                        f_name, f_score, f_sec = face_extractor.top_match(face_emb, face_profiles)
                        lock_mgr.submit_face_match(tid, frame_idx, f_name, f_score, f_sec)
                        if not state_obj.locked and state_obj.locked:
                            student = f_name

            track_to_student[tid] = student
            if student != "UNKNOWN":
                current_bboxes[student] = bbox

            track_annotations.append({
                "track_id": tid,
                "student": student,
                "bbox": bbox,
                "_tlwh": tlwh,
            })

        # Action classification per track
        annotated_tracks = []
        for ann in track_annotations:
            tid = ann["track_id"]
            student = ann["student"]
            bbox = ann["bbox"]
            x1, y1, x2, y2 = bbox

            crop = frame[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None

            # Match keypoints
            kps = _match_kps_to_track(
                (x1, y1, x2, y2),
                np.array(dets) if dets else np.empty((0,5)),
                pose_kps_list,
            )

            # Find nearby students for talking detection
            nearby = find_nearby_students(student, current_bboxes, proximity_px=PROXIMITY_PX)

            # Classify action
            action_result = action_clf.update(
                student_id=student if student != "UNKNOWN" else f"track_{tid}",
                track_id=tid,
                frame_idx=frame_idx,
                keypoints=kps,
                bbox=bbox,
                nearby_students=nearby,
                crop=crop,
            )

            # Log to event logger
            if student in KNOWN_STUDENTS:
                event_logger.update(
                    student=student,
                    action=action_result.action,
                    frame_idx=frame_idx,
                    confidence=action_result.confidence,
                    is_visible=True,
                )

            annotated_tracks.append({
                "track_id": tid,
                "student": student,
                "action": action_result.action,
                "confidence": action_result.confidence,
                "bbox": (x1, y1, x2, y2),
            })

        # Mark missing known students as unknown
        visible_students = {ann["student"] for ann in track_annotations if ann["student"] != "UNKNOWN"}
        for s in KNOWN_STUDENTS:
            if s not in visible_students:
                event_logger.update(student=s, action="unknown", frame_idx=frame_idx, is_visible=False)

        # Annotate frame
        annotated = annotate_frame(frame, annotated_tracks)

        # Add FPS overlay
        elapsed = time.time() - t_loop_start
        live_fps = (frame_idx + 1) / max(elapsed, 0.001)
        cv2.putText(annotated, f"Frame {frame_idx} | {live_fps:.1f} FPS",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Encode as JPEG for streaming
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _state_lock:
            _state["last_jpeg"] = jpeg.tobytes()
            _state["frame_idx"] = frame_idx

        # Push SSE event every 0.5s
        now = time.time()
        if now - last_sse_push > 0.5:
            try:
                status = event_logger.live_status()
                _sse_queue.put_nowait(json.dumps(status))
                last_sse_push = now
            except queue.Full:
                pass

        frame_idx += 1
        if MAX_FRAMES > 0 and frame_idx >= MAX_FRAMES:
            break

    cap.release()

    # Final export
    summary = event_logger.session_summary()
    out_path = os.path.join(OUTPUT_DIR, "session_summary.json")
    event_logger.export_json(out_path)
    logger.info("Session saved to %s", out_path)

    with _state_lock:
        _state["running"] = False
        _state["mode"] = "idle"


# --------------------------------------------------------------------------- #
#  Flask routes                                                                 #
# --------------------------------------------------------------------------- #

@app.route("/")
def index():
    return render_template("dashboard/index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard/live.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of annotated frames."""
    def generate():
        while True:
            jpeg = _state.get("last_jpeg")
            if jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.033)   # ~30 fps cap
    return Response(stream_with_context(generate()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stream")
def sse_stream():
    """Server-Sent Events for live student status."""
    def generate():
        yield "data: {}\n\n"
        while True:
            try:
                data = _sse_queue.get(timeout=2.0)
                yield f"data: {data}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/status")
def api_status():
    logger = _state.get("event_logger")
    if logger is None:
        return jsonify({"mode": _state["mode"], "running": _state["running"], "students": {}})
    return jsonify(logger.live_status())


@app.route("/api/summary")
def api_summary():
    logger = _state.get("event_logger")
    if logger is None:
        return jsonify({"error": "No session running"})
    return jsonify(logger.session_summary())


@app.route("/api/bouts")
def api_bouts():
    logger = _state.get("event_logger")
    if logger is None:
        return jsonify([])
    return jsonify(logger.get_bouts())


@app.route("/api/alerts")
def api_alerts():
    logger = _state.get("event_logger")
    if logger is None:
        return jsonify([])
    return jsonify(logger.get_all_alerts())


@app.route("/api/acknowledge_alert", methods=["POST"])
def acknowledge_alert():
    idx = int(request.args.get("idx", -1))
    logger = _state.get("event_logger")
    if logger and idx >= 0:
        logger.acknowledge_alert(idx)
    return jsonify({"ok": True})


@app.route("/api/start_video", methods=["POST"])
def start_video():
    if _state["running"]:
        return jsonify({"error": "Already running"}), 400
    video_path = request.json.get("video_path", VIDEO_PATH) if request.is_json else VIDEO_PATH
    if not os.path.exists(video_path):
        return jsonify({"error": f"Video not found: {video_path}"}), 404

    with _state_lock:
        _state["running"] = True
        _state["mode"] = "video"

    t = threading.Thread(target=_run_video_pipeline, args=(video_path, False), daemon=True)
    t.start()
    with _state_lock:
        _state["thread"] = t
    return jsonify({"ok": True, "mode": "video", "video": video_path})


@app.route("/api/start_webcam", methods=["POST"])
def start_webcam():
    if _state["running"]:
        return jsonify({"error": "Already running"}), 400
    with _state_lock:
        _state["running"] = True
        _state["mode"] = "webcam"
    t = threading.Thread(target=_run_video_pipeline, args=("", True), daemon=True)
    t.start()
    with _state_lock:
        _state["thread"] = t
    return jsonify({"ok": True, "mode": "webcam"})


@app.route("/api/stop", methods=["POST"])
def stop():
    with _state_lock:
        _state["running"] = False
    return jsonify({"ok": True})


@app.route("/api/state")
def api_state():
    return jsonify({
        "running": _state["running"],
        "mode": _state["mode"],
        "frame_idx": _state["frame_idx"],
        "fps": _state["fps"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
