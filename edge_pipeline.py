"""
IBVAP - Intelligent Border Video Analytics Platform
Phase 2: Edge AI Inference Pipeline

Design:
  - Captures frames from the local webcam (cv2.VideoCapture(0)).
  - Runs YOLOv8n inference every INFERENCE_EVERY_N_FRAMES frames (smart skipping).
  - Detects humans (COCO class 0) and vehicles (classes 2,3,5,7).
  - On a detection above CONFIDENCE_THRESHOLD:
      * Draws a bounding box + label on the frame.
      * Encodes the annotated frame as Base64 JPEG.
      * POSTs a JSON payload to the IBVAP backend API.
      * Enters a COOLDOWN_SECONDS cooldown to prevent API flooding.
  - Displays the live annotated feed in a local OpenCV window for debugging.
  - Press 'q' to quit cleanly.
"""

import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ibvap.edge")

# ---------------------------------------------------------------------------
# Configuration  (edit these constants to tune the pipeline)
# ---------------------------------------------------------------------------

# Backend API endpoint
API_ENDPOINT: str = "http://localhost:8000/api/events"

# Camera index (0 = default system webcam)
CAMERA_INDEX: int = 0

# Camera label sent with every alert
CAMERA_ID: str = "CAM-EDGE-01"

# YOLOv8 model weights — 'yolov8n.pt' is downloaded automatically on first run
MODEL_WEIGHTS: str = "yolov8n.pt"

# Run inference only every N-th frame (reduces CPU load; skipped frames
# still display the last-known bounding boxes for visual continuity)
INFERENCE_EVERY_N_FRAMES: int = 5

# Minimum detection confidence to trigger an alert (0–1)
CONFIDENCE_THRESHOLD: float = 0.60

# After sending an alert, suppress further alerts for this many seconds
COOLDOWN_SECONDS: float = 5.0

# COCO class IDs to monitor
# 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
WATCHED_CLASSES: dict[int, str] = {
    0: "PERSON_DETECTED",
    2: "VEHICLE_DETECTED",
    3: "VEHICLE_DETECTED",
    5: "VEHICLE_DETECTED",
    7: "VEHICLE_DETECTED",
}

# Annotation colours per class group (BGR)
COLOUR_PERSON: tuple[int, int, int] = (0, 255, 80)     # neon green
COLOUR_VEHICLE: tuple[int, int, int] = (0, 160, 255)   # amber-orange
COLOUR_TEXT_BG: tuple[int, int, int] = (0, 0, 0)

# JPEG encoding quality for the snapshot (0-100)
SNAPSHOT_JPEG_QUALITY: int = 85

# ---------------------------------------------------------------------------
# Detection result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """Represents a single object detection from the model."""

    class_id: int
    event_type: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bounding_box(self) -> dict:
        return {
            "x": self.x1,
            "y": self.y1,
            "width": self.x2 - self.x1,
            "height": self.y2 - self.y1,
        }

    @property
    def colour(self) -> tuple[int, int, int]:
        return COLOUR_PERSON if self.class_id == 0 else COLOUR_VEHICLE


# ---------------------------------------------------------------------------
# Core pipeline components
# ---------------------------------------------------------------------------


def load_model(weights: str) -> YOLO:
    """Load (or download) the YOLOv8 model weights."""
    log.info("Loading model: %s", weights)
    model = YOLO(weights)
    log.info("Model loaded — %d classes available", len(model.names))
    return model


def open_camera(index: int) -> cv2.VideoCapture:
    """Open the webcam and verify it is accessible."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {index}. "
            "Check that the webcam is connected and not in use by another app."
        )
    # Set a sensible resolution for the local prototype
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Camera %d opened: %dx%d", index, w, h)
    return cap


def run_inference(model: YOLO, frame: np.ndarray) -> list[Detection]:
    """
    Run YOLOv8 inference on a single frame.
    Returns only watched-class detections above CONFIDENCE_THRESHOLD.
    """
    results = model(frame, verbose=False)[0]
    detections: list[Detection] = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in WATCHED_CLASSES:
            continue
        conf = float(box.conf[0].item())
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append(
            Detection(
                class_id=cls_id,
                event_type=WATCHED_CLASSES[cls_id],
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
            )
        )

    return detections


def annotate_frame(
    frame: np.ndarray, detections: list[Detection]
) -> np.ndarray:
    """
    Draw bounding boxes and confidence labels on the frame (in-place).
    Returns the same frame for chaining.
    """
    for det in detections:
        colour = det.colour
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), colour, 2)

        label = f"{det.event_type.replace('_DETECTED','')}  {det.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        # Label background pill
        cv2.rectangle(
            frame,
            (det.x1, det.y1 - th - baseline - 6),
            (det.x1 + tw + 8, det.y1),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (det.x1 + 4, det.y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return frame


def encode_snapshot(frame: np.ndarray) -> str:
    """Encode the annotated frame as a Base64 JPEG string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_JPEG_QUALITY]
    ok, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        raise RuntimeError("cv2.imencode failed — cannot create snapshot")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def post_alert(
    session: requests.Session,
    detection: Detection,
    snapshot_b64: str,
) -> bool:
    """
    POST the alert JSON payload to the IBVAP backend API.
    Returns True on HTTP 201, False on any error (non-blocking).
    """
    payload = {
        "camera_id": CAMERA_ID,
        "event_type": detection.event_type,
        "confidence": round(detection.confidence, 6),
        "bounding_box": detection.bounding_box,
        "snapshot_b64": snapshot_b64,
    }
    try:
        resp = session.post(API_ENDPOINT, json=payload, timeout=5)
        if resp.status_code == 201:
            log.info(
                "Alert sent  →  event_id=%s  type=%s  conf=%.1f%%",
                resp.json().get("id", "?"),
                detection.event_type,
                detection.confidence * 100,
            )
            return True
        else:
            log.warning(
                "API returned %d: %s", resp.status_code, resp.text[:200]
            )
    except requests.exceptions.ConnectionError:
        log.error("Cannot reach API at %s — is the backend running?", API_ENDPOINT)
    except requests.exceptions.Timeout:
        log.error("API request timed out after 5 s")
    except Exception as exc:
        log.error("Unexpected error posting alert: %s", exc)
    return False


def draw_hud(
    frame: np.ndarray,
    frame_count: int,
    in_cooldown: bool,
    cooldown_remaining: float,
    fps: float,
) -> None:
    """Overlay a minimal HUD (status bar) on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    status = (
        f"COOLDOWN  {cooldown_remaining:.1f}s"
        if in_cooldown
        else "MONITORING"
    )
    status_colour = (0, 140, 255) if in_cooldown else (0, 220, 80)

    cv2.putText(
        frame, f"IBVAP  |  {CAMERA_ID}  |  FPS: {fps:.1f}  |  Frame: {frame_count}",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, status,
        (w - 220, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_colour, 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Entry point — runs until the user presses 'q'."""
    model = load_model(MODEL_WEIGHTS)
    cap = open_camera(CAMERA_INDEX)
    session = requests.Session()  # reuse TCP connections for efficiency

    frame_count: int = 0     # total frames — used only for inference modulo
    fps_frame_count: int = 0  # frames within the current 1-second FPS window
    last_alert_time: float = 0.0
    last_detections: list[Detection] = []
    fps_timer: float = time.monotonic()
    fps: float = 0.0

    log.info(
        "Pipeline running — inference every %d frames, "
        "threshold %.0f%%, cooldown %.0fs",
        INFERENCE_EVERY_N_FRAMES,
        CONFIDENCE_THRESHOLD * 100,
        COOLDOWN_SECONDS,
    )
    log.info("Press 'q' in the OpenCV window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            log.error("Failed to read frame from camera — exiting.")
            break

        frame_count += 1
        fps_frame_count += 1

        # ── Smart frame skipping ────────────────────────────────────────────
        if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
            last_detections = run_inference(model, frame)

        # ── Annotate with the most recent detections ────────────────────────
        annotate_frame(frame, last_detections)

        # ── Cooldown & alert logic ──────────────────────────────────────────
        now = time.monotonic()
        in_cooldown = (now - last_alert_time) < COOLDOWN_SECONDS
        cooldown_remaining = max(0.0, COOLDOWN_SECONDS - (now - last_alert_time))

        if last_detections and not in_cooldown:
            # Pick the highest-confidence detection to report
            best = max(last_detections, key=lambda d: d.confidence)
            snapshot_b64 = encode_snapshot(frame)
            success = post_alert(session, best, snapshot_b64)
            if success:
                last_alert_time = now
                in_cooldown = True
                cooldown_remaining = COOLDOWN_SECONDS

        # ── FPS calculation (per-window counter resets each second) ─────────
        elapsed = now - fps_timer
        if elapsed >= 1.0:
            fps = fps_frame_count / elapsed  # frames THIS window / seconds
            fps_timer = now
            fps_frame_count = 0              # reset for the next window

        # ── HUD overlay ────────────────────────────────────────────────────
        draw_hud(frame, frame_count, in_cooldown, cooldown_remaining, fps)

        # ── Display ────────────────────────────────────────────────────────
        cv2.imshow("IBVAP — Edge Pipeline  [q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("Quit key pressed — shutting down.")
            break

    cap.release()
    session.close()
    cv2.destroyAllWindows()
    log.info("Pipeline stopped cleanly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
