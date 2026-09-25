# Fix app.py Unable to Run - The Implementation Plan (Decomposed and Prioritized Task List)

## [x] Task 1: Fix roboflow_client.py — API key env var, remove hardcoded key, add Detection and WorkflowResult dataclasses, add run_workflow_on_image function
- **Priority**: high
- **Depends On**: None
- **Description**:
  - Remove the hardcoded `API_KEY = "eAH3a7KkChusddboikWe"` on line 4
  - Change `os.getenv("API_KEY")` to `os.getenv("ROBOFLOW_API_KEY")` so env var name matches the error message
  - Add `from dataclasses import dataclass, field` import
  - Add `from typing import Any, Optional, List` import
  - Add `import io` and `import cv2` or use `PIL`/`numpy` to encode an OpenCV frame to JPEG bytes; minimal: use `cv2.imencode('.jpg', frame)` to get bytes then wrap in `io.BytesIO` (or pass bytes directly with a filename)
  - Add `import numpy as np` if needed for isinstance checks
  - Define `@dataclass class Detection: x: float; y: float; width: float; height: float; class_name: str; confidence: float`
  - Define `@dataclass class WorkflowResult: detections: List[Detection] = field(default_factory=list); raw: Optional[dict] = None`
  - Implement `run_workflow_on_image(frame, parameters=None)`:
    - Accept an OpenCV BGR image (numpy ndarray, HWC uint8) and optional parameters dict
    - Encode to JPEG bytes via `ok, buf = cv2.imencode('.jpg', frame); if not ok: raise RoboflowWorkflowError(...)`
    - Build an in-memory file object from `buf.tobytes()` (or wrap `BytesIO(buf.tobytes())`)
    - POST to `WORKFLOW_URL` with `params={"api_key": API_KEY}`, `files={"image": ("frame.jpg", jpeg_bytes_or_bytesio, "image/jpeg")}`, timeout=60 (mirror existing `run_workflow` request shape)
    - Parse JSON response into `WorkflowResult` by iterating over the detections list (assumed shape: list of objects with `x`, `y`, `width`, `height`, `class`/`class_name`, `confidence` fields — map `class` → `class_name` if needed)
    - On non-200 or request error, raise `RoboflowWorkflowError` with HTTP status + text (mirror existing `run_workflow` error handling)
    - Attach `raw=response.json()` to the WorkflowResult for debugging
- **Acceptance Criteria Addressed**: AC-2, AC-3, AC-4, AC-8
- **Test Requirements**:
  - `programmatic` TR-1.1: With ROBOFLOW_API_KEY unset, importing roboflow_client raises ValueError whose message contains "ROBOFLOW_API_KEY is not set"
  - `programmatic` TR-1.2: With ROBOFLOW_API_KEY set, `dir(roboflow_client)` includes `RoboflowWorkflowError`, `WorkflowResult`, `Detection`, `run_workflow`, `run_workflow_on_image`
  - `programmatic` TR-1.3: `Detection` and `WorkflowResult` are dataclass instances with correct field names/types (instantiate one, check fields)
  - `programmatic` TR-1.4: `run_workflow_on_image` — mock `requests.post` to return a JSON payload containing detections; verify returned `WorkflowResult.detections` is a list of `Detection` with correct field values parsed from JSON; verify `requests.post` was called once with `files=` containing an `image` entry
  - `programmatic` TR-1.5: No occurrence of the old hardcoded key string `eAH3a7KkChusddboikWe` or any other 15+ char alphanumeric secret-looking literal in roboflow_client.py, app.py, video_processor.py (grep)
- **Notes**: Roboflow JSON shape may nest detections under a key like `predictions`; if unsure, make the parser flexible: try `data.get('predictions') or data.get('detections') or data`, and tolerate both `class` and `class_name` keys.

## [x] Task 2: Fix app.py — ensure RESULTS_DIR exists at import time
- **Priority**: high
- **Depends On**: None (can be parallel with Task 1)
- **Description**:
  - Move `os.makedirs(RESULTS_DIR, exist_ok=True)` from inside the `if __name__ == "__main__":` block to module level, right after the `RESULTS_DIR` constant is defined (around line 17)
- **Acceptance Criteria Addressed**: AC-6
- **Test Requirements**:
  - `programmatic` TR-2.1: Delete `static/results` if it exists, then run a script that imports `app`; afterwards assert `os.path.isdir(RESULTS_DIR)` is True
- **Notes**: `RESULTS_DIR` is already defined; move the `os.makedirs` call to immediately after the constants block. Avoid duplicating; remove it from the `__main__` block or keep both (exist_ok=True makes it idempotent either way — keeping both is fine but cleaner to only have it once at module level).

## [x] Task 3: Create templates/results.html content to render FrameResult data
- **Priority**: high
- **Depends On**: None (can be parallel)
- **Description**:
  - Write a valid HTML5 page that accepts `job_id`, `frames: list[FrameResult]`, `total_detections: int`
  - Header shows job ID and "Total detections: X" summary
  - For each frame in `frames`, render a card/row containing:
    - `<img src="{{ url_for('static', filename='results/' + job_id + '/' + frame.annotated_path) }}" alt="frame {{ frame.frame_index }}">` (only if `frame.annotated_path` is truthy; otherwise show a placeholder "annotated image not available" box)
    - Metadata: Frame index `{{ frame.frame_index }}`, Timestamp `{{ "%.2f"|format(frame.timestamp_sec) }}s`, Detections `{{ frame.detection_count }}`
    - Top confidence: `{{ "%.2f"|format(frame.top_confidence) }}` if not None, else "N/A"
    - Classes: comma-separated `{{ frame.classes | join(', ') }}`
    - If `frame.error` is set, show it in a red/alert box
  - Use Jinja2 `{% for frame in frames %}{% endfor %}` loop
  - Add a link back to home (`<a href="/">Upload another video</a>`)
  - No external CSS required; basic styling via inline `<style>` tag is acceptable (e.g., grid layout, borders, padding)
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `programmatic` TR-3.1: Using Flask's test client and app context, `render_template('results.html', job_id='abc', frames=[...sample FrameResult dicts/objects...], total_detections=5)` returns a 200-equivalent (no TemplateSyntaxError) and the rendered output contains: 'abc', 'Total detections', at least one `<img` tag, the string '0.50' (formatted top confidence), class names joined, and an error section when error is set
  - `human-judgement` TR-3.2: Reviewer opens rendered HTML in a browser — layout is readable, images appear where expected, no broken layout/obvious glitches
- **Notes**: Use minimal but sensible inline CSS (grid, max-width on imgs, padding). FrameResult fields are: `frame_index: int`, `timestamp_sec: float`, `annotated_path: str`, `detection_count: int`, `top_confidence: float | None`, `classes: list[str]`, `error: str | None`.

## [x] Task 4: Add sample_fps and max_frames form inputs to index.html
- **Priority**: medium
- **Depends On**: None
- **Description**:
  - The `/analyze` route already reads `sample_fps` (default 1.0) and `max_frames` (default 24) from `request.form`; however the form in `index.html` has no inputs for them.
  - Add two labeled number inputs inside the existing `<form>`:
    - `<label>Sample FPS: <input type="number" name="sample_fps" step="0.1" min="0.1" value="1.0"></label>`
    - `<label>Max frames: <input type="number" name="max_frames" step="1" min="1" max="1000" value="24"></label>`
  - Keep the file input and Analyze button
- **Acceptance Criteria Addressed**: No AC directly (improvement), but supports AC-7 flow completeness
- **Test Requirements**:
  - `programmatic` TR-4.1: GET `/` → response body contains `<input` for both `sample_fps` and `max_frames` with those exact `name` attributes
- **Notes**: Small UX polish. Not strictly blocking startup, but ensures the user can actually control these parameters that the backend already consumes.

## [x] Task 5: Integration smoke test — app startup + render routes with mocks
- **Priority**: high
- **Depends On**: Task 1, Task 2, Task 3, Task 4
- **Description**:
  - Write a small inline check script (via RunCommand, no new files unless absolutely needed) that:
    - Sets `ROBOFLOW_API_KEY=test_key` env var
    - (Optional) Temporarily patches `requests.post` via `unittest.mock.patch` to return a fake 200 response with a sample detections JSON list: e.g. `[{"x":100,"y":100,"width":50,"height":50,"class":"phone","confidence":0.95}]`
    - Imports `app` from app.py (verifies Task 1 + Task 2: no import error, RESULTS_DIR created)
    - Uses Flask's test client: `client = app.app.test_client()`
    - GET `/` — assert 200, form present (verifies index.html loads, Task 4 fields present)
    - Renders results template directly via `with app.app.app_context(): render_template('results.html', ...)` to verify Task 3 template is valid Jinja
    - For a full `/analyze` hit: programmatically craft a tiny one-frame video in memory using cv2 (write a single 64x64 black frame to a temp .mp4 via cv2.VideoWriter_fourcc(*'mp4v')), POST it with `files={'video': (name, bytes, 'video/mp4')}`, `data={'sample_fps':'1','max_frames':'1'}`, with requests.post mocked — assert 200, results.html renders, at least one detection string appears.
- **Acceptance Criteria Addressed**: AC-1, AC-5, AC-6, AC-7
- **Test Requirements**:
  - `programmatic` TR-5.1: Import succeeds (exit code 0, no traceback)
  - `programmatic` TR-5.2: GET `/` → status 200, body contains `sample_fps` and `max_frames` inputs
  - `programmatic` TR-5.3: Render `results.html` with a sample frame list — contains expected fields
  - `programmatic` TR-5.4: Mocked `/analyze` POST returns status 200 and body contains `Total detections` (or the configured job_id string)
- **Notes**: This task is the integration pass. If any sub-task fails, loop back and fix before marking it done.

## [x] Task 6: requirements.txt completeness audit
- **Priority**: medium
- **Depends On**: None
- **Description**:
  - `requests` is used in `roboflow_client.py` but not listed in requirements.txt
  - `numpy` is implicitly required (OpenCV returns numpy arrays; `cv2.imencode` works without explicit numpy import but code in new `run_workflow_on_image` may do isinstance checks; safe to list it)
  - Verify the file contains: `flask`, `opencv-python` (or `opencv-python-headless`), `requests`, `inference-sdk`, `numpy`
  - Append missing entries to requirements.txt (one per line, keep existing order)
- **Acceptance Criteria Addressed**: Supports AC-1/NFR-1 by ensuring `pip install -r requirements.txt` installs everything
- **Test Requirements**:
  - `programmatic` TR-6.1: `cat requirements.txt` output contains lines exactly matching `flask`, `opencv-python`, `requests`, `inference-sdk`, `numpy` (order independent)
- **Notes**: Do not upgrade or change versions — just add missing package names without version pins to match existing style.
