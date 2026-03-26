# =============================================================================
# CANE TOAD DETECTOR — Raspberry Pi 4 + A4Tech USB Camera (ONNX VERSION)
# =============================================================================
#
# This version uses ONNX Runtime instead of PyTorch/Ultralytics to avoid
# illegal instruction errors on older ARM processors.
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
#         pip install opencv-python numpy onnxruntime
#
#   4. Copy model and class names to Pi (run from Windows terminal):
#         scp "c:\Users\Carlo\Desktop\AI Components\docs\best.onnx" \
#             pi@<PI_IP_ADDRESS>:/home/pi/cane_toad_detector/best.onnx
#         scp "c:\Users\Carlo\Desktop\AI Components\docs\class_names.json" \
#             pi@<PI_IP_ADDRESS>:/home/pi/cane_toad_detector/class_names.json
#
# RUN:
#   source ~/cane_toad_env/bin/activate
#   cd /home/pi/cane_toad_detector
#   python detector_onnx.py
#
#   Optional arguments:
#   python detector_onnx.py --model best.onnx --camera 0 --width 640 --height 480 \
#                           --conf 0.5 --skip 2
#
# CONTROLS:
#   Press Q in the video window to quit cleanly.
#
# =============================================================================

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Class name filter for the GPIO stub trigger.
# After the first run, check terminal output to confirm the exact class label
# your model uses, then update this set if needed.
# ---------------------------------------------------------------------------
CLASS_NAME_FILTER = {"cane toad", "canetoad", "toad", "cane_toad", "rhinella marina"}


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
        description="Cane Toad Detector — YOLOv8 ONNX on Raspberry Pi 4"
    )
    parser.add_argument(
        "--model", default="best.onnx",
        help="Path to YOLOv8 .onnx model file (default: best.onnx)"
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
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
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


def load_class_names(model_path: str) -> dict:
    """Load class names from class_names.json in the same directory as the model."""
    model_dir = Path(model_path).parent
    class_file = model_dir / "class_names.json"
    
    if not class_file.exists():
        print(f"[WARN] Class names file not found: {class_file}")
        print("[WARN] Using default class name: 'object'")
        return {0: "object"}
    
    try:
        with open(class_file, "r") as f:
            class_names = json.load(f)
            # Convert string keys to integers
            return {int(k): v for k, v in class_names.items()}
    except Exception as e:
        print(f"[WARN] Failed to load class names: {e}")
        return {0: "object"}


# =============================================================================
# Model loading with warm-up
# =============================================================================
def load_model(model_path: str):
    """
    Load ONNX model and run one warm-up inference.
    The warm-up eliminates the first-frame delay.
    Exits with a clear error message if the model file is not found.
    """
    try:
        print(f"[INFO] Loading ONNX model: {model_path}")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get input details
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"[INFO] Model input: {input_name}, shape: {input_shape}")
        
        # Load class names
        class_names = load_class_names(model_path)
        print(f"[INFO] Class names: {class_names}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model '{model_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Running warm-up inference...")
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    session.run(None, {input_name: dummy})
    print("[INFO] Warm-up complete.")
    
    return session, input_name, class_names


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while maintaining aspect ratio."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess(image):
    """Preprocess image for YOLO inference."""
    # Letterbox resize to 640x640
    img, ratio, (dw, dh) = letterbox(image, new_shape=(640, 640))
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # HWC to CHW format
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img, ratio, (dw, dh)


def xywh2xyxy(x):
    """Convert bounding box from [x_center, y_center, width, height] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def nms(boxes, scores, iou_threshold):
    """Apply Non-Maximum Suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def postprocess(output, conf_threshold, iou_threshold, ratio, pad):
    """Process YOLO output and return detections."""
    # output shape: (1, 84, 8400) for YOLOv8
    # First 4 values are box coordinates, rest are class scores
    
    predictions = output[0]
    predictions = np.transpose(predictions, (0, 2, 1))  # (1, 8400, 84)
    predictions = predictions[0]  # (8400, 84)
    
    # Extract boxes and scores
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    
    # Get class predictions
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(class_ids)), class_ids]
    
    # Filter by confidence
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return [], [], []
    
    # Convert boxes from xywh to xyxy
    boxes = xywh2xyxy(boxes)
    
    # Apply NMS
    indices = nms(boxes, confidences, iou_threshold)
    
    boxes = boxes[indices]
    confidences = confidences[indices]
    class_ids = class_ids[indices]
    
    # Scale boxes back to original image
    dw, dh = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    
    return boxes, confidences, class_ids


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
def draw_detections(frame: np.ndarray, boxes, confidences, class_ids, class_names: dict) -> np.ndarray:
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

    if len(boxes) == 0:
        return frame

    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names.get(cls_id, 'unknown')}: {conf:.2f}"

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
def print_detections(boxes, confidences, class_ids, class_names: dict, frame_index: int) -> None:
    """Print a summary of detections to the terminal. Silent if no detections."""
    if len(boxes) == 0:
        return

    print(f"\n[FRAME {frame_index}] {len(boxes)} detection(s):")
    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        name = class_names.get(cls_id, 'unknown')
        print(f"  [{i + 1}] class='{name}'  conf={conf:.3f}  bbox=[{x1},{y1},{x2},{y2}]")


# =============================================================================
# Main detection loop
# =============================================================================
def main() -> None:
    args = parse_args()

    session, input_name, class_names = load_model(args.model)
    cap = open_camera(args.camera, args.width, args.height)

    WINDOW_NAME = "Cane Toad Detector (ONNX)  |  Press Q to quit"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_index = 0
    last_boxes = []
    last_confidences = []
    last_class_ids = []
    fps_start = time.time()
    fps_counter = 0
    fps_display = 0.0

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
                # Preprocess
                input_tensor, ratio, pad = preprocess(frame)
                
                # Inference
                outputs = session.run(None, {input_name: input_tensor})
                
                # Postprocess
                boxes, confidences, class_ids = postprocess(
                    outputs, args.conf, args.iou, ratio, pad
                )
                
                last_boxes = boxes
                last_confidences = confidences
                last_class_ids = class_ids
                
                print_detections(boxes, confidences, class_ids, class_names, frame_index)

                # GPIO stub: trigger only for confirmed cane toad class names
                if len(boxes) > 0:
                    toad_boxes = []
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        class_name = class_names.get(cls_id, 'unknown').lower()
                        if class_name in CLASS_NAME_FILTER:
                            toad_boxes.append(list(map(int, box)))
                    
                    if toad_boxes:
                        best_conf = max(confidences)
                        on_cane_toad_detected(len(toad_boxes), toad_boxes, best_conf)

            # --- Annotation (every frame using cached results) ---
            annotated = frame.copy()
            if len(last_boxes) > 0:
                draw_detections(annotated, last_boxes, last_confidences, last_class_ids, class_names)

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
