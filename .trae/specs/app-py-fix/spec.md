# Fix app.py Unable to Run - Product Requirement Document

## Overview
- **Summary**: Fix multiple import, configuration, and template issues that prevent the Flask video analysis application (`app.py`) from starting and operating correctly. The application uploads videos, samples frames, runs Roboflow object detection workflow, and displays annotated results.
- **Purpose**: The app currently crashes on startup due to missing symbols in `roboflow_client.py`, inconsistent API key environment variable handling, and missing template content. This prevents users from accessing the video analysis functionality.
- **Target Users**: End users who need to analyze videos for student productivity monitoring via the web UI.

## Goals
- `app.py` starts successfully without import errors
- API key configuration works correctly (env var name matches error message)
- `roboflow_client.py` exports all symbols imported by `video_processor.py`
- `results.html` template renders frame analysis results properly
- Static results directory exists before any route is accessed
- Full upload → analyze → results display flow works end-to-end

## Non-Goals (Out of Scope)
- Rewriting the video analysis algorithm
- Changing the Roboflow API endpoint or workflow
- Adding authentication or user accounts
- Refactoring the overall architecture
- Adding new features beyond what the existing code intends

## Background & Context
The project is a Flask web app for "Student Productivity Monitoring". Users upload a video (`/` route → index.html), which gets processed (`/analyze` route): frames are sampled via OpenCV, each frame is sent to a Roboflow workflow for object detection, detections are drawn on frames and saved as JPEGs, then a results page shows everything.

Current files:
- `app.py` — Flask routes, file upload handling
- `roboflow_client.py` — Roboflow HTTP client (broken: wrong env var, missing classes/functions)
- `video_processor.py` — frame sampling + annotation (imports missing symbols from roboflow_client)
- `templates/index.html` — upload form (works but missing optional FPS/frames inputs)
- `templates/results.html` — **empty file** (needs content to display results)
- `requirements.txt` — flask, opencv-python, inference-sdk

## Functional Requirements
- **FR-1**: `roboflow_client.py` shall define `WorkflowResult` dataclass with `detections` attribute
- **FR-2**: `roboflow_client.py` shall define `run_workflow_on_image(frame, parameters=None)` that accepts an OpenCV BGR image (numpy array), encodes it, calls the workflow, and returns a `WorkflowResult`
- **FR-3**: `roboflow_client.py` shall check env var `ROBOFLOW_API_KEY` (not `API_KEY`) and raise a matching error message
- **FR-4**: `roboflow_client.py` shall NOT hardcode an API key string in the source
- **FR-5**: `templates/results.html` shall render: job_id, total detection count, and a grid/table of each frame's annotated image, timestamp, detection count, top confidence, class names, and error (if any)
- **FR-6**: `app.py` shall ensure `RESULTS_DIR` (`static/results`) exists at app initialization time (not only inside `__main__`)
- **FR-7**: `roboflow_client.py` shall define a `Detection` dataclass used inside `WorkflowResult` with fields: `x`, `y`, `width`, `height`, `class_name`, `confidence`

## Non-Functional Requirements
- **NFR-1**: Startup must complete in <5 seconds on a typical dev machine (excluding pip install)
- **NFR-2**: No hardcoded secrets in source code (security requirement)
- **NFR-3**: `results.html` must be valid HTML5 that renders in modern browsers
- **NFR-4**: Error messages in roboflow_client must be descriptive (include HTTP status + response body on failure)

## Constraints
- **Technical**: Python 3.x, Flask, OpenCV (`cv2`), `requests` library, standard library (`dataclasses`, `typing`, `io`, `base64`, `PIL`/numpy-compatible image encoding)
- **Business**: Roboflow API key must come from environment variable `ROBOFLOW_API_KEY` (consistency with error message)
- **Dependencies**: `requests`, `opencv-python-headless` (or opencv-python), `numpy`, `flask` — `requests` is used in roboflow_client already but not listed in requirements.txt (it should be present transitively via inference-sdk, but should be explicit)

## Assumptions
- The Roboflow endpoint returns JSON in a format compatible with the existing `run_workflow` function (the function already reads `response.json()`)
- Each detection in the Roboflow JSON response includes: bounding box center (`x`, `y`), size (`width`, `height`), class name (`class` or `class_name`), and confidence (`confidence`)
- OpenCV frames are BGR numpy uint8 arrays; `run_workflow_on_image` must convert to bytes (e.g., JPEG-encode via `cv2.imencode`) before POSTing
- The hardcoded API key on line 4 of roboflow_client.py is a placeholder/leftover and should be removed entirely (env var only)

## Acceptance Criteria

### AC-1: App starts without import errors
- **Given**: ROBOFLOW_API_KEY environment variable is set (any non-empty string for import check)
- **When**: Running `python app.py` or importing the app module
- **Then**: Process does not raise `ValueError`, `ImportError`, or `NameError` during module loading
- **Verification**: `programmatic` (exit code 0 from a startup probe script)
- **Notes**: Import-only check; server does not need to serve requests for this AC

### AC-2: roboflow_client raises correct error when API key missing
- **Given**: `ROBOFLOW_API_KEY` environment variable is unset or empty
- **When**: Importing `roboflow_client` module
- **Then**: Raises `ValueError` with message containing "ROBOFLOW_API_KEY is not set"
- **Verification**: `programmatic`

### AC-3: roboflow_client exports required symbols
- **Given**: Module loads successfully
- **When**: Inspecting `roboflow_client` namespace
- **Then**: The following symbols exist and are importable: `RoboflowWorkflowError`, `WorkflowResult`, `Detection`, `run_workflow`, `run_workflow_on_image`
- **Verification**: `programmatic`

### AC-4: run_workflow_on_image returns WorkflowResult with Detections
- **Given**: A mock image (numpy BGR array) and a mocked requests.post returning a known JSON payload
- **When**: Calling `run_workflow_on_image(frame)`
- **Then**: Returns a `WorkflowResult` instance; each item in `.detections` is a `Detection` with `x`, `y`, `width`, `height`, `class_name`, `confidence` populated from the JSON response
- **Verification**: `programmatic`

### AC-5: results.html template renders frame data
- **Given**: A Flask test client and sample FrameResult data
- **When**: Rendering `results.html` with `job_id`, `frames`, `total_detections`
- **Then**: Response body contains the job_id, total_detections, and for each frame: annotated image reference, detection count, and class list
- **Verification**: `programmatic` (Flask test client + assertIn on HTML)

### AC-6: static/results directory exists on import
- **Given**: The `static/results` directory does not exist
- **When**: Importing `app.py`
- **Then**: Directory `static/results` is created (via `os.makedirs(..., exist_ok=True)`)
- **Verification**: `programmatic`

### AC-7: End-to-end smoke test with mocked Roboflow
- **Given**: ROBOFLOW_API_KEY set, `app.py` importable, `requests.post` mocked to return a valid detections payload
- **When**: POST `/analyze` with a small synthetic valid video (e.g., 1-frame .mp4), sample_fps=1, max_frames=1
- **Then**: HTTP 200 response, `results.html` renders, at least 1 frame result shown, static JPEG saved under `static/results/<job_id>/`
- **Verification**: `programmatic`

### AC-8: No hardcoded API key in source
- **Given**: Source files `app.py`, `roboflow_client.py`, `video_processor.py`
- **When**: Searching for the previous hardcoded key string or any obvious API-key literal
- **Then**: No API-key literal is found in the three source files
- **Verification**: `programmatic` (grep / text search)

## Open Questions
- [ ] What exact JSON structure does the Roboflow `WORKFLOW_URL` return for detections? (Assumed standard fields: `x`, `y`, `width`, `height`, `class`/`class_name`, `confidence` — may need field mapping in `run_workflow_on_image`.)
- [ ] Is `inference-sdk` (in requirements.txt) intended to replace the custom `roboflow_client.py` calls, or should we keep the custom `requests.post` approach? (Keeping custom approach since code is already written; just adding missing pieces.)
