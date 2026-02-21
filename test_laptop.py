# =============================================================================
# CANE TOAD DETECTOR — Laptop / Windows Test Script
# =============================================================================
#
# Use this to test best.pt using your laptop's built-in or USB webcam.
# This is the Windows-compatible version (no Raspberry Pi GPIO or V4L2).
#
# SETUP:
#   1. Create and activate a virtual environment (recommended):
#         python -m venv venv
#         venv\Scripts\activate
#
#   2. Install dependencies:
#         pip install -r requirements.txt
#
#   3. Run:
#         python test_laptop.py
#
#   Optional arguments:
#         python test_laptop.py --model best.pt --camera 0 --conf 0.5 --skip 1
#
# CONTROLS:
#   Press Q in the video window to quit.
# =============================================================================

import argparse
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cane Toad Detector — Laptop Test")
    parser.add_argument(
        "--model", default="best.pt",
        help="Path to YOLOv8 .pt model file (default: best.pt)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0 = built-in webcam)"
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
        help="Minimum confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--skip", type=int, default=1,
        help="Run inference every N frames (default: 1 = every frame). Laptops are faster than Pi."
    )
    return parser.parse_args()


def load_model(model_path: str) -> YOLO:
    """Load YOLOv8 model and run a warm-up inference to avoid a freeze on frame 1."""
    try:
        model = YOLO(model_path)
        print(f"[INFO] Model loaded: {model_path}")
        print(f"[INFO] Class names: {model.names}")
    except Exception as e:
        print(f"[ERROR] Failed to load model '{model_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Running warm-up inference...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)
    print("[INFO] Warm-up complete.")
    return model


def open_camera(device_index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open webcam. Works with built-in laptop cameras and USB webcams on Windows."""
    # No CAP_V4L2 here — that is Linux-only. Let OpenCV auto-detect on Windows.
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        print(
            f"[ERROR] Camera device {device_index} not found.",
            file=sys.stderr
        )
        print("[HINT]  Try --camera 1 if the built-in webcam is not at index 0.",
              file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: device={device_index}, resolution={actual_w}x{actual_h}")
    return cap


def draw_detections(frame: np.ndarray, boxes, class_names: dict) -> np.ndarray:
    """Draw bounding boxes and labels on the frame."""
    if boxes is None:
        return frame

    BOX_COLOUR = (0, 220, 0)
    TEXT_COLOUR = (0, 0, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = f"{class_names[cls]}: {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOUR, 2)

        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.6, 2)
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            BOX_COLOUR, -1
        )
        cv2.putText(frame, label, (x1, y1 - 4), FONT, 0.6, TEXT_COLOUR, 2)

    return frame


def print_detections(results, class_names: dict, frame_index: int) -> None:
    """Print detection info to terminal. Silent when nothing is detected."""
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return

    print(f"\n[FRAME {frame_index}] {len(boxes)} detection(s):")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        print(f"  [{i + 1}] class='{class_names[cls]}'  conf={conf:.3f}  bbox=[{x1},{y1},{x2},{y2}]")


def main() -> None:
    args = parse_args()

    model       = load_model(args.model)
    cap         = open_camera(args.camera, args.width, args.height)
    class_names = model.names

    WINDOW_NAME = "Cane Toad Detector — Laptop Test  |  Press Q to quit"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_index  = 0
    last_results = None
    fps_start    = time.time()
    fps_counter  = 0
    fps_display  = 0.0

    print("[INFO] Detection loop started. Press 'Q' in the window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame.", file=sys.stderr)
                break

            if frame_index % args.skip == 0:
                results      = model.predict(frame, conf=args.conf, verbose=False)
                last_results = results
                print_detections(results, class_names, frame_index)

            annotated = frame.copy()
            if last_results is not None:
                draw_detections(annotated, last_results[0].boxes, class_names)

            # FPS counter
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_start   = time.time()

            cv2.putText(
                annotated,
                f"FPS: {fps_display:.1f}  |  conf>={args.conf}",
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
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
