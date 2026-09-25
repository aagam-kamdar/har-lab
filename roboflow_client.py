# roboflow_client.py

from dataclasses import dataclass
from typing import List, Optional
import logging
import os

logger = logging.getLogger(__name__)


class RoboflowWorkflowError(Exception):
    pass


@dataclass
class Detection:
    x: float
    y: float
    width: float
    height: float
    class_name: str
    confidence: float


@dataclass
class WorkflowResult:
    detections: List[Detection]
    raw: dict | None = None


ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
USE_LOCAL_DETECTOR = os.getenv("USE_LOCAL_DETECTOR", "").lower() in ("1", "true", "yes")

# Initialize client only if API key is present
client = None
try:
    if ROBOFLOW_API_KEY:
        from inference_sdk import InferenceHTTPClient
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=ROBOFLOW_API_KEY,
        )
except Exception as e:
    logger.warning("Could not initialize Roboflow client: %s", e)

# Cached local YOLO model instance
_local_yolo_model = None


def get_local_yolo_model(weights_path: str = "yolo26n.pt"):
    """
    Lazy loader for local YOLO model (bypassing Roboflow cloud API).
    Uses yolo26n.pt if present in project root, or falls back to yolo11n.pt / yolov8n.pt.
    """
    global _local_yolo_model
    if _local_yolo_model is None:
        from ultralytics import YOLO

        target_weights = weights_path
        if not os.path.exists(target_weights):
            for candidate in ("yolo26n.pt", "yolo11n.pt", "yolov8n.pt"):
                if os.path.exists(candidate):
                    target_weights = candidate
                    break

        logger.info("Initializing local YOLO detector with weights: %s", target_weights)
        _local_yolo_model = YOLO(target_weights)
    return _local_yolo_model


def run_local_yolo(frame, conf_threshold: float = 0.25, weights_path: str = "yolo26n.pt") -> WorkflowResult:
    """
    Local YOLO inference on CPU/GPU.
    Converts results to the exact same WorkflowResult / Detection format.
    Zero cloud calls, zero API keys required.
    """
    try:
        model = get_local_yolo_model(weights_path)
        results = model(frame, conf=conf_threshold, verbose=False)[0]

        detections: List[Detection] = []
        for box in results.boxes:
            x, y, w, h = box.xywh[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            detections.append(
                Detection(
                    x=float(x),
                    y=float(y),
                    width=float(w),
                    height=float(h),
                    class_name=str(class_name),
                    confidence=conf,
                )
            )

        return WorkflowResult(
            detections=detections,
            raw={"source": "local_yolo", "model": weights_path, "count": len(detections)},
        )
    except Exception as exc:
        raise RoboflowWorkflowError(f"Local YOLO inference failed: {exc}") from exc


def run_workflow_on_image(frame, parameters=None) -> WorkflowResult:
    """
    Runs detection on a frame.
    Supports:
      - mode='local' (runs local yolo26n.pt offline)
      - mode='roboflow' (calls Roboflow cloud API)
      - mode='auto' (default: uses local if ROBOFLOW_API_KEY is not set or if Roboflow fails)
    """
    params = parameters or {}
    mode = params.get("detector_mode", "auto")

    # 1. Force local if requested or if no API key is available
    if mode == "local" or USE_LOCAL_DETECTOR or not ROBOFLOW_API_KEY:
        return run_local_yolo(frame)

    # 2. Attempt Roboflow cloud workflow
    try:
        if client is None or not ROBOFLOW_API_KEY:
            raise RoboflowWorkflowError("ROBOFLOW_API_KEY is not set.")

        payload = client.run_workflow(
            workspace_name="aagams-workspace-2pogv",
            workflow_id="ipd-vipd-ejtth-2-yolo11n-t1-logic",
            images={"image": frame},
            parameters=params,
        )

        predictions = payload[0].get("predictions", {}).get("predictions", [])

        detections = [
            Detection(
                x=float(p.get("x", 0)),
                y=float(p.get("y", 0)),
                width=float(p.get("width", 0)),
                height=float(p.get("height", 0)),
                class_name=p.get("class", "object"),
                confidence=float(p.get("confidence", 0)),
            )
            for p in predictions
        ]

        return WorkflowResult(detections=detections, raw=payload)

    except Exception as exc:
        if mode == "auto":
            logger.warning(
                "Roboflow cloud API call failed (%s). Automatically falling back to local YOLO.",
                exc,
            )
            return run_local_yolo(frame)
        raise RoboflowWorkflowError(str(exc)) from exc


def run_inference(frame):
    return run_workflow_on_image(frame)