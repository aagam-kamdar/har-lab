import logging
import os
import shutil
import tempfile
import uuid

from flask import Flask, render_template, request, send_from_directory
from video_processor import process_video_local, LocalTrackingResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(APP_ROOT, "static", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    video_file = request.files.get("video")

    if not video_file or video_file.filename == "":
        return render_template("index.html", error="Please choose a video file first.")

    ext = os.path.splitext(video_file.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return render_template(
            "index.html",
            error=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    conf_thresh = float(request.form.get("conf_thresh", 0.30))
    track_buffer = int(request.form.get("track_buffer", 30))

    job_id = uuid.uuid4().hex[:12]
    output_dir = os.path.join(RESULTS_DIR, job_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, f"upload{ext}")
        video_file.save(video_path)

        try:
            logger.info("Starting local tracking for job %s: %s", job_id, video_file.filename)
            tracking_result: LocalTrackingResult = process_video_local(
                video_path=video_path,
                output_dir=output_dir,
                conf_thresh=conf_thresh,
                track_buffer=track_buffer,
            )

        except Exception as exc:
            shutil.rmtree(output_dir, ignore_errors=True)
            logger.exception("Local video tracking failed")
            return render_template("index.html", error=str(exc))

    return render_template(
        "results.html",
        job_id=job_id,
        result=tracking_result,
        video_url=f"/static/results/{job_id}/{tracking_result.output_video_filename}",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)