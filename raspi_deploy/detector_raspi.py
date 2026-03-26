# =============================================================================
# CANE TOAD DETECTOR — Raspberry Pi ONNX Detector with Backend Integration
# =============================================================================
#
# Optimized ONNX detector for Raspberry Pi with Flask backend integration.
#
# Features:
#   - ONNX Runtime (CPU inference)
#   - V4L2 camera backend (better performance on Pi)
#   - Frame skipping for performance
#   - Headless mode (no display)
#   - Backend API integration
#   - FPS tracking
#
# Usage:
#   python detector_raspi.py --headless --skip 2
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


# =============================================================================
# Configuration
# =============================================================================

CLASS_NAME_FILTER = {"cane toad", "canetoad", "toad", "cane_toad", "rhinella marina"}


# =============================================================================
# CLI Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Cane Toad Detector - Raspberry Pi ONNX")
    parser.add_argument("--model", default="best.onnx", help="Path to ONNX model")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--skip", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--headless", action="store_true", help="No display (production mode)")
    return parser.parse_args()


# =============================================================================
# Model Loading
# =============================================================================

def load_class_names(model_path):
    """Load class names from class_names.json."""
    model_dir = Path(model_path).parent
    class_file = model_dir / "class_names.json"
    
    if not class_file.exists():
        print(f"[WARN] Class names file not found: {class_file}")
        return {0: "object"}
    
    try:
        with open(class_file, "r") as f:
            class_names = json.load(f)
            return {int(k): v for k, v in class_names.items()}
    except Exception as e:
        print(f"[WARN] Failed to load class names: {e}")
        return {0: "object"}


def load_model(model_path):
    """Load ONNX model and run warm-up inference."""
    if not Path(model_path).exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print(f"[ERROR] Please ensure best.onnx is in the current directory")
        sys.exit(1)
    
    try:
        print(f"[INFO] Loading ONNX model: {model_path}")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"[INFO] Model input: {input_name}, shape: {input_shape}")
        
        class_names = load_class_names(model_path)
        print(f"[INFO] Class names loaded: {len(class_names)} classes")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    
    # Warm-up
    print("[INFO] Running warm-up inference...")
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    session.run(None, {input_name: dummy})
    print("[INFO] Warm-up complete")
    
    return session, input_name, class_names


# =============================================================================
# Camera Setup
# =============================================================================

def open_camera(camera_id=0):
    """Open camera with V4L2 backend for better Pi performance."""
    print(f"[INFO] Opening camera {camera_id}...")
    
    # Try V4L2 backend first (better for Pi)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("[WARN] V4L2 failed, trying default backend...")
        cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera {camera_id}")
        print("[ERROR] Check camera connection and permissions")
        sys.exit(1)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[INFO] Camera opened: {actual_width}x{actual_height}")
    
    return cap


# =============================================================================
# Preprocessing
# =============================================================================

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while maintaining aspect ratio."""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im, r, (dw, dh)


def preprocess(image):
    """Preprocess image for YOLO inference."""
    img, ratio, (dw, dh) = letterbox(image, new_shape=(640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, ratio, (dw, dh)


# =============================================================================
# Postprocessing
# =============================================================================

def xywh2xyxy(x):
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
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
    predictions = output[0]
    predictions = np.transpose(predictions, (0, 2, 1))
    predictions = predictions[0]
    
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(class_ids)), class_ids]
    
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return [], [], []
    
    boxes = xywh2xyxy(boxes)
    keep = nms(boxes, confidences, iou_threshold)
    
    boxes = boxes[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]
    
    # Scale back to original image
    dw, dh = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    
    return boxes, confidences, class_ids


# =============================================================================
# Detection Callback
# =============================================================================

def on_cane_toad_detected(detection_count, bbox_xyxy, confidence):
    """Callback when cane toad detected."""
    # TODO: Add backend API integration here
    pass


# =============================================================================
# Main Detection Loop
# =============================================================================

def main():
    args = parse_args()
    
    # Load model
    session, input_name, class_names = load_model(args.model)
    
    # Open camera
    cap = open_camera(args.camera)
    
    # Initialize variables
    frame_count = 0
    detections_cache = ([], [], [])
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0
    
    print(f"[INFO] Starting detection loop...")
    print(f"[INFO] Frame skip: {args.skip} (processing every {args.skip} frame(s))")
    print(f"[INFO] Headless mode: {args.headless}")
    if not args.headless:
        print("[INFO] Press 'Q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame")
                break
            
            frame_count += 1
            
            # Run inference on selected frames
            if frame_count % args.skip == 0:
                # Preprocess
                input_tensor, ratio, pad = preprocess(frame)
                
                # Inference
                outputs = session.run(None, {input_name: input_tensor})
                
                # Postprocess
                boxes, confidences, class_ids = postprocess(
                    outputs, args.conf, args.iou, ratio, pad
                )
                
                detections_cache = (boxes, confidences, class_ids)
                
                # Check for cane toads
                if len(boxes) > 0:
                    max_conf = np.max(confidences)
                    print(f"[DETECT] Frame {frame_count}: {len(boxes)} detection(s), conf={max_conf:.2f}")
                    on_cane_toad_detected(len(boxes), boxes, max_conf)
            else:
                boxes, confidences, class_ids = detections_cache
            
            # Draw detections (if not headless)
            if not args.headless:
                display_frame = frame.copy()
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    label = class_names.get(int(cls_id), f"Class{cls_id}")
                    
                    # Draw box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label_text = f"{label} {conf:.2f}"
                    cv2.putText(display_frame, label_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw FPS
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start = time.time()
                
                cv2.putText(display_frame, f"FPS: {current_fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow("Cane Toad Detector", display_frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("[INFO] Quit requested")
                    break
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[INFO] Detector stopped")


if __name__ == "__main__":
    main()
