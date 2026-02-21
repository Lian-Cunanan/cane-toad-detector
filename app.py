# =============================================================================
# CANE TOAD DETECTOR — Local Web App
# =============================================================================
#
# Streams your webcam feed with live YOLOv8 detections to a browser page.
#
# SETUP (first time only):
#   venv\Scripts\activate          (Windows)
#   pip install -r requirements.txt
#
# RUN:
#   python app.py
#   Then open http://localhost:5000 in your browser.
#
# Press Ctrl+C in the terminal to stop the server.
# =============================================================================

import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config — adjust these if needed
# ---------------------------------------------------------------------------
MODEL_PATH       = "best.pt"
CAMERA_INDEX     = 0          # 0 = built-in webcam, try 1 or 2 for USB cameras
CONF_THRESHOLD   = 0.5
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
INFERENCE_SKIP   = 1          # Run inference every N frames (increase if too slow)
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Shared state (thread-safe enough for single-user local use)
latest_detections = []        # List of dicts for the sidebar
detection_lock    = threading.Lock()


def load_model() -> YOLO:
    """Load YOLOv8 model with a warm-up inference."""
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Class names: {model.names}")
    dummy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)
    print("[INFO] Model ready.")
    return model


def generate_frames(model: YOLO):
    """
    MJPEG generator — yields annotated frames as a multipart HTTP stream.
    This is what the <img> tag on the webpage reads from continuously.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}. Try changing CAMERA_INDEX in app.py.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_index  = 0
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % INFERENCE_SKIP == 0:
            results      = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            last_results = results

            # Update shared detection list for the sidebar
            boxes = results[0].boxes
            dets  = []
            if boxes is not None:
                for box in boxes:
                    dets.append({
                        "label": model.names[int(box.cls[0])],
                        "conf":  round(float(box.conf[0]), 3),
                        "bbox":  list(map(int, box.xyxy[0].tolist())),
                    })
            with detection_lock:
                latest_detections.clear()
                latest_detections.extend(dets)

        # Use YOLOv8's built-in annotator for clean boxes + labels
        if last_results is not None:
            annotated = last_results[0].plot()
        else:
            annotated = frame

        # Encode as JPEG and yield as MJPEG chunk
        success, buffer = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not success:
            continue

        frame_index += 1
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    cap.release()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream consumed by the <img> tag on the page."""
    return Response(
        generate_frames(model),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/detections")
def detections():
    """JSON endpoint polled by the page to update the detection sidebar."""
    with detection_lock:
        data = list(latest_detections)
    return jsonify(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = load_model()
    print("\n[INFO] Server starting...")
    print("[INFO] Open your browser and go to:  http://localhost:5000\n")
    # use_reloader=False is required — reloader would load the model twice
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
