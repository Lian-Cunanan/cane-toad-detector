# =============================================================================
# CANE TOAD DETECTOR — Laptop / Windows Test Script (ONNX VERSION)
# =============================================================================
#
# Use this to test best.onnx using your laptop's built-in or USB webcam.
# This version uses ONNX Runtime instead of PyTorch/Ultralytics to avoid
# illegal instruction errors on older CPUs.
#
# SETUP:
#   1. Create and activate a virtual environment (recommended):
#         python -m venv venv
#         venv\Scripts\activate
#
#   2. Install dependencies:
#         pip install opencv-python numpy onnxruntime
#
#   3. Run:
#         python test_laptop_onnx.py
#
#   Optional arguments:
#         python test_laptop_onnx.py --model docs/best.onnx --camera 0 --conf 0.5 --skip 1
#
# CONTROLS:
#   Press Q in the video window to quit.
# =============================================================================

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import requests


def list_available_cameras(max_test: int = 10) -> list:
    """Detect available camera devices by testing indices 0 through max_test-1."""
    available = []
    print("[INFO] Scanning for available cameras...")
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
                print(f"  ✓ Camera {i} found")
            cap.release()
    return available


def select_camera_interactive() -> int:
    """Let user interactively choose from available cameras."""
    cameras = list_available_cameras()
    
    if not cameras:
        print("[ERROR] No cameras found!", file=sys.stderr)
        sys.exit(1)
    
    if len(cameras) == 1:
        print(f"[INFO] Only one camera found. Using camera {cameras[0]}.")
        return cameras[0]
    
    print(f"\n[INFO] Found {len(cameras)} camera(s): {cameras}")
    while True:
        try:
            choice = input(f"Select camera index {cameras}: ")
            camera_index = int(choice)
            if camera_index in cameras:
                return camera_index
            else:
                print(f"[ERROR] Invalid choice. Please select from {cameras}")
        except (ValueError, KeyboardInterrupt):
            print("\n[INFO] Selection cancelled.")
            sys.exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cane Toad Detector — Laptop Test (ONNX)")
    parser.add_argument(
        "--model", default="docs/best.onnx",
        help="Path to YOLOv8 .onnx model file (default: docs/best.onnx)"
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera device index (default: auto-detect and select)"
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="List all available cameras and exit"
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
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    parser.add_argument(
        "--skip", type=int, default=1,
        help="Run inference every N frames (default: 1 = every frame)"
    )
    parser.add_argument(
        "--api-url", type=str, default="http://localhost:5000",
        help="Backend API URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--username", type=str, default=None,
        help="Username for API authentication (enables live feed)"
    )
    parser.add_argument(
        "--password", type=str, default=None,
        help="Password for API authentication"
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


def api_login(api_url: str, username: str, password: str) -> str:
    """Login to backend API and return access token."""
    try:
        response = requests.post(
            f"{api_url}/api/auth/login",
            json={"username": username, "password": password},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            token = data.get('access_token')
            print(f"[INFO] Logged in as '{username}' - Live feed enabled!")
            return token
        else:
            print(f"[WARN] Login failed: {response.text}")
            return None
    except Exception as e:
        print(f"[WARN] API connection failed: {e}")
        print("[INFO] Continuing without live feed...")
        return None


def send_detection_to_api(api_url: str, token: str, box: list, confidence: float, class_name: str):
    """Send detection to backend API for live feed."""
    if not token:
        return
    
    try:
        x1, y1, x2, y2 = map(int, box)
        response = requests.post(
            f"{api_url}/api/detections",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "class_name": class_name,
                "confidence": float(confidence),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2
            },
            timeout=2
        )
        if response.status_code != 201:
            print(f"[WARN] Failed to send detection: {response.text}")
    except Exception as e:
        # Silent fail to avoid cluttering output
        pass


def load_model(model_path: str):
    """Load ONNX model and run a warm-up inference to avoid a freeze on frame 1."""
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


def open_camera(device_index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open webcam. Works with built-in laptop cameras and USB webcams on Windows."""
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


def draw_detections(frame: np.ndarray, boxes, confidences, class_ids, class_names: dict) -> np.ndarray:
    """Draw bounding boxes and labels on the frame."""
    if len(boxes) == 0:
        return frame

    BOX_COLOUR = (0, 220, 0)
    TEXT_COLOUR = (0, 0, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names.get(cls_id, 'unknown')}: {conf:.2f}"

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


def print_detections(boxes, confidences, class_ids, class_names: dict, frame_index: int) -> None:
    """Print detection info to terminal. Silent when nothing is detected."""
    if len(boxes) == 0:
        return

    print(f"\n[FRAME {frame_index}] {len(boxes)} detection(s):")
    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        name = class_names.get(cls_id, 'unknown')
        print(f"  [{i + 1}] class='{name}'  conf={conf:.3f}  bbox=[{x1},{y1},{x2},{y2}]")


def main() -> None:
    args = parse_args()

    # Handle --list-cameras flag
    if args.list_cameras:
        cameras = list_available_cameras()
        if cameras:
            print(f"\n[INFO] Available cameras: {cameras}")
            print("[INFO] Use --camera <index> to select a specific camera.")
        else:
            print("\n[ERROR] No cameras found.")
        sys.exit(0)

    # Auto-select camera if not specified
    if args.camera is None:
        args.camera = select_camera_interactive()
        print(f"[INFO] Selected camera {args.camera}\n")

    # Login to API if credentials provided
    api_token = None
    if args.username and args.password:
        print(f"[INFO] Connecting to backend: {args.api_url}")
        api_token = api_login(args.api_url, args.username, args.password)
    else:
        print("[INFO] No API credentials provided - running in local-only mode")
        print("[INFO] Use --username and --password to enable live feed to mobile app")

    session, input_name, class_names = load_model(args.model)
    cap = open_camera(args.camera, args.width, args.height)

    WINDOW_NAME = "Cane Toad Detector — Laptop Test (ONNX)  |  Press Q to quit"
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
                print("[WARN] Failed to grab frame.", file=sys.stderr)
                break

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
                
                # Send detections to API if connected
                if api_token and len(boxes) > 0:
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        class_name = class_names.get(cls_id, 'unknown')
                        send_detection_to_api(args.api_url, api_token, box, conf, class_name)

            annotated = frame.copy()
            if len(last_boxes) > 0:
                draw_detections(annotated, last_boxes, last_confidences, last_class_ids, class_names)

            # FPS counter
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

            # Status display
            status_text = f"FPS: {fps_display:.1f}  |  conf>={args.conf}"
            if api_token:
                status_text += "  |  LIVE FEED ON"
            
            cv2.putText(
                annotated,
                status_text,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if api_token else (0, 200, 255), 2
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
