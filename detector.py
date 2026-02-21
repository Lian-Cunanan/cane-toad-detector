# =============================================================================
# CANE TOAD DETECTOR — Raspberry Pi 4 + A4Tech USB Camera
# =============================================================================
#
# Requires: Raspberry Pi OS 64-bit (Bookworm), Pi 4 Model B
#
# SETUP (run on the Raspberry Pi):
#
#   1. Install system dependencies:
#         sudo apt update && sudo apt upgrade -y
#         sudo apt install -y python3-pip python3-venv \
#                            libgl1-mesa-glx libglib2.0-0 \
#                            libsm6 libxext6 libxrender-dev
#
#   2. Create and activate a virtual environment:
#         python3 -m venv ~/cane_toad_env
#         source ~/cane_toad_env/bin/activate
#
#   3. Install Python packages:
#         pip install --upgrade pip
#         pip install ultralytics opencv-python numpy
#
#      NOTE: If PyTorch ARM64 wheel fails, install manually first:
#         pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#      Then re-run: pip install ultralytics
#
#   4. Copy model to Pi (run from Windows terminal):
#         scp "c:\Users\Carlo\Desktop\AI Components\best.pt" \
#             pi@<PI_IP_ADDRESS>:/home/pi/cane_toad_detector/best.pt
#
#   5. Confirm class names used in your model (run once before deploying):
#         python -c "from ultralytics import YOLO; m = YOLO('best.pt'); print(m.names)"
#      Then update the CLASS_NAME_FILTER set below if needed.
#
# RUN:
#   source ~/cane_toad_env/bin/activate
#   cd /home/pi/cane_toad_detector
#   python detector.py
#
#   Optional arguments:
#   python detector.py --model best.pt --camera 0 --width 640 --height 480 \
#                      --conf 0.5 --skip 2
#
# CONTROLS:
#   Press Q in the video window to quit cleanly.
#
# =============================================================================

import argparse
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Class name filter for the GPIO stub trigger.
# After the first run, check terminal output to confirm the exact class label
# your model uses, then update this set if needed.
# ---------------------------------------------------------------------------
CLASS_NAME_FILTER = {"cane toad", "canetoad", "toad", "rhinella marina"}


# =============================================================================
# FUTURE GPIO STUB — Servo Motor Trigger
# =============================================================================
# This function is called whenever a cane toad is detected above the
# confidence threshold. Currently it does nothing.
#
# TODO (GPIO implementation):
#   1. Import RPi.GPIO or gpiozero at the top of this file
#   2. Define servo pins, e.g.: SERVO_PINS = [17, 27, 22]  (BCM numbering)
#   3. Add a setup_gpio() function:
#         def setup_gpio():
#             import RPi.GPIO as GPIO
#             GPIO.setmode(GPIO.BCM)
#             for pin in SERVO_PINS:
#                 GPIO.setup(pin, GPIO.OUT)
#             # configure PWM on each pin at 50 Hz
#   4. Call setup_gpio() once before the main loop in main()
#   5. In this function: activate each servo to open position, hold,
#      then return to neutral (closed) position to control the mesh gate
#   6. Add GPIO.cleanup() inside the finally block in main()
#
# Example future call site (already in place in main loop):
#   on_cane_toad_detected(count=2, bbox_xyxy=[[x1,y1,x2,y2]], confidence=0.92)
# =============================================================================
def on_cane_toad_detected(detection_count: int, bbox_xyxy: list, confidence: float) -> None:
    """
    Placeholder called each time cane toad detections occur in a frame.

    Args:
        detection_count: Number of cane toads detected in this frame.
        bbox_xyxy:       List of bounding boxes as [[x1,y1,x2,y2], ...].
        confidence:      Highest confidence score among detections this frame.
    """
    pass  # TODO: implement servo gate activation here


# =============================================================================
# CLI argument parsing
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cane Toad Detector — YOLOv8 on Raspberry Pi 4"
    )
    parser.add_argument(
        "--model", default="best.pt",
        help="Path to YOLOv8 .pt model file (default: best.pt)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="USB camera device index (default: 0). Run 'ls /dev/video*' on Pi to list devices."
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Capture width in pixels (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Capture height in pixels (default: 480)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Minimum confidence threshold for detections (default: 0.5)"
    )
    parser.add_argument(
        "--skip", type=int, default=2,
        help=(
            "Run inference every N frames (default: 2). "
            "1 = every frame (slowest), 3 = every 3rd frame (fastest). "
            "Cached results are drawn on skipped frames so the display stays smooth."
        )
    )
    return parser.parse_args()


# =============================================================================
# Model loading with warm-up
# =============================================================================
def load_model(model_path: str) -> YOLO:
    """
    Load YOLOv8 model and run one warm-up inference.
    The warm-up eliminates the JIT compile freeze on the first real camera frame.
    Exits with a clear error message if the model file is not found.
    """
    try:
        model = YOLO(model_path)
        print(f"[INFO] Model loaded: {model_path}")
        print(f"[INFO] Class names: {model.names}")
    except Exception as e:
        print(f"[ERROR] Failed to load model '{model_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Running warm-up inference...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(dummy_frame, verbose=False)
    print("[INFO] Warm-up complete.")
    return model


# =============================================================================
# Camera initialisation
# =============================================================================
def open_camera(device_index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    Open the A4Tech USB webcam using the V4L2 backend (Linux/Pi OS).
    Exits with helpful diagnostics if the camera is not accessible.
    """
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(
            f"[ERROR] Camera device {device_index} not found or not accessible.",
            file=sys.stderr
        )
        print("[HINT]  Run 'ls /dev/video*' on the Pi to list available video devices.",
              file=sys.stderr)
        print("[HINT]  Try --camera 1 or --camera 2 if index 0 does not work.",
              file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Buffer size 1: always read the newest frame, avoids stale detection lag on Pi
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: device={device_index}, resolution={actual_w}x{actual_h}")
    return cap


# =============================================================================
# Frame annotation
# =============================================================================
def draw_detections(frame: np.ndarray, boxes, class_names: dict) -> np.ndarray:
    """
    Draw bounding boxes and confidence labels onto the frame in-place.
    Returns the annotated frame for convenience.
    """
    BOX_COLOUR    = (0, 220, 0)       # Green
    TEXT_COLOUR   = (0, 0, 0)         # Black text on filled background
    BOX_THICKNESS = 2
    FONT          = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE    = 0.6
    FONT_THICK    = 2

    if boxes is None:
        return frame

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = f"{class_names[cls]}: {conf:.2f}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOUR, BOX_THICKNESS)

        # Filled label background for readability
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            BOX_COLOUR,
            -1  # filled
        )
        cv2.putText(frame, label, (x1, y1 - 4), FONT, FONT_SCALE, TEXT_COLOUR, FONT_THICK)

    return frame


# =============================================================================
# Terminal detection logging
# =============================================================================
def print_detections(results, class_names: dict, frame_index: int) -> None:
    """Print a summary of detections to the terminal. Silent if no detections."""
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return

    print(f"\n[FRAME {frame_index}] {len(boxes)} detection(s):")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        name = class_names[cls]
        print(f"  [{i + 1}] class='{name}'  conf={conf:.3f}  bbox=[{x1},{y1},{x2},{y2}]")


# =============================================================================
# Main detection loop
# =============================================================================
def main() -> None:
    args = parse_args()

    model       = load_model(args.model)
    cap         = open_camera(args.camera, args.width, args.height)
    class_names = model.names  # dict {int: str}

    WINDOW_NAME = "Cane Toad Detector  |  Press Q to quit"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_index    = 0
    last_results   = None   # Cache last inference output for skipped frames
    fps_start      = time.time()
    fps_counter    = 0
    fps_display    = 0.0

    print("[INFO] Detection loop started. Press 'Q' in the window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(
                    "[WARN] Failed to grab frame — camera may have been disconnected.",
                    file=sys.stderr
                )
                break

            # --- Inference (every N-th frame) ---
            if frame_index % args.skip == 0:
                results      = model.predict(frame, conf=args.conf, verbose=False)
                last_results = results
                print_detections(results, class_names, frame_index)

                # GPIO stub: trigger only for confirmed cane toad class names
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    toad_boxes = [
                        list(map(int, b.xyxy[0]))
                        for b in boxes
                        if class_names[int(b.cls[0])].lower() in CLASS_NAME_FILTER
                    ]
                    if toad_boxes:
                        best_conf = max(float(b.conf[0]) for b in boxes)
                        on_cane_toad_detected(len(toad_boxes), toad_boxes, best_conf)

            # --- Annotation (every frame using cached results) ---
            annotated = frame.copy()
            if last_results is not None:
                draw_detections(annotated, last_results[0].boxes, class_names)

            # --- FPS overlay ---
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display  = fps_counter / elapsed
                fps_counter  = 0
                fps_start    = time.time()

            cv2.putText(
                annotated,
                f"FPS: {fps_display:.1f}  |  Inf every {args.skip}f",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2
            )

            cv2.imshow(WINDOW_NAME, annotated)
            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'Q' pressed — exiting.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — exiting.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Goodbye.")


if __name__ == "__main__":
    main()
