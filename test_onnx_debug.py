# =============================================================================
# ONNX Model Diagnostic + Live Test
# =============================================================================
#
# This script diagnoses exactly WHY the ONNX model may not be detecting
# cane toads, and lets you watch live detections on webcam or a static image.
#
# Usage:
#   venv\Scripts\activate
#   python test_onnx_debug.py                                         <- webcam, auto-detect model
#   python test_onnx_debug.py --model best_1.onnx                    <- specific model
#   python test_onnx_debug.py --model best_1.onnx --conf 0.25        <- lower threshold
#   python test_onnx_debug.py --model best_1.onnx --image photo.jpg  <- test on image
#   python test_onnx_debug.py --model best_1.onnx --info-only        <- just print model info, no video
#
# The --info-only flag prints:
#   - input tensor name + shape
#   - output tensor name + shape
#   - class names embedded in model
#   - a raw-score dump of one inference pass (what scores cane_toad is actually getting)
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
# Helpers
# ---------------------------------------------------------------------------

def load_class_names(classes_arg: str, model_path: str) -> dict:
    """
    Load class names from an explicit JSON file, or auto-discover one next to the model.
    Falls back to {0: 'object'} if nothing is found.
    """
    candidates = []
    if classes_arg:
        candidates.append(Path(classes_arg))

    model_dir = Path(model_path).parent
    model_stem = Path(model_path).stem
    candidates += [
        model_dir / f"{model_stem}_classes.json",
        model_dir / "class_names.json",
        model_dir / "classes.json",
        Path("docs") / "class_names.json",
    ]

    for p in candidates:
        if p.exists():
            try:
                with open(p) as f:
                    raw = json.load(f)
                names = {int(k): v for k, v in raw.items()}
                print(f"[INFO] Loaded class names from: {p}")
                return names
            except Exception as e:
                print(f"[WARN] Failed to read {p}: {e}")

    print("[WARN] No class names file found. Using {0: 'object'}")
    return {0: "object"}


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess(image):
    img, ratio, (dw, dh) = letterbox(image, new_shape=(640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, ratio, (dw, dh)


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms(boxes, scores, iou_threshold=0.45):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def postprocess(raw_output, conf_threshold, iou_threshold, ratio, pad):
    """
    Works for both:
      YOLOv8 export: output shape (1, num_classes+4, 8400)
      Older YOLO:    output shape (1, 8400, num_classes+5)  <-- has objectness column
    Returns (boxes_xyxy, confidences, class_ids)  — all in original-image pixel coords.
    """
    pred = raw_output[0]             # shape (1, A, B)

    # ---- detect layout --------------------------------------------------
    # YOLOv8 exports: (1,  4+nc, 8400)  — no objectness
    # YOLOv5 exports: (1, 8400, 4+1+nc) — has objectness at index 4
    if pred.shape[1] < pred.shape[2]:
        # YOLOv8 style: (1, 4+nc, 8400) → transpose to (8400, 4+nc)
        pred = pred[0].T
        boxes_xywh  = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids   = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]
    else:
        # YOLOv5 style: (1, 8400, cols)
        pred = pred[0]
        obj_conf    = pred[:, 4]
        class_scores = pred[:, 5:]
        class_ids   = np.argmax(class_scores, axis=1)
        confidences = obj_conf * class_scores[np.arange(len(class_ids)), class_ids]
        boxes_xywh  = pred[:, :4]

    # ---- filter by confidence -------------------------------------------
    mask = confidences > conf_threshold
    boxes_xywh  = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    if len(boxes_xywh) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])

    boxes_xyxy = xywh2xyxy(boxes_xywh)

    keep = nms(boxes_xyxy, confidences, iou_threshold)
    boxes_xyxy  = boxes_xyxy[keep]
    confidences = confidences[keep]
    class_ids   = class_ids[keep]

    # ---- scale back to original image -----------------------------------
    dw, dh = pad
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dw) / ratio
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dh) / ratio

    return boxes_xyxy, confidences, class_ids


# ---------------------------------------------------------------------------
# Raw-score dump — the key diagnostic tool
# ---------------------------------------------------------------------------

def dump_raw_scores(session, input_name, class_names, image_bgr):
    """
    Run inference once and print the top-10 per-class scores for every
    high-confidence anchor box.  This tells you EXACTLY what score the
    model is assigning to each class so you can spot class-name mismatches.
    """
    img, ratio, pad = preprocess(image_bgr)
    raw = session.run(None, {input_name: img})
    pred = raw[0]

    # normalise to (N, 4+nc)
    if pred.shape[1] < pred.shape[2]:          # YOLOv8
        pred = pred[0].T
        class_scores = pred[:, 4:]
    else:                                       # YOLOv5
        pred = pred[0]
        obj  = pred[:, 4:5]
        class_scores = pred[:, 5:] * obj

    # For each anchor, get its best score
    best_per_anchor = class_scores.max(axis=1)
    top_anchors = best_per_anchor.argsort()[::-1][:20]   # top 20 anchors

    print("\n[DIAG] Top-20 anchor boxes by best class score:")
    print(f"       {'anchor':>8}  {'best_score':>12}  {'class_id':>10}  class_name")
    for i in top_anchors:
        best_class = class_scores[i].argmax()
        best_score = class_scores[i, best_class]
        cname = class_names.get(int(best_class), f"class_{best_class}")
        print(f"       {i:>8}  {best_score:>12.4f}  {best_class:>10}  {cname}")

    print(f"\n[DIAG] All class names in the loaded JSON ({len(class_names)} classes):")
    for idx, name in sorted(class_names.items()):
        max_score = class_scores[:, idx].max() if idx < class_scores.shape[1] else -1
        print(f"         class {idx:3d}: {name:<30}  max score across all anchors: {max_score:.4f}")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_boxes(image, boxes, confidences, class_ids, class_names):
    for box, conf, cls in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names.get(int(cls), str(cls))}  {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ONNX Cane-Toad detector — diagnostic + live test")
    p.add_argument("--model",     default=None,
                   help="Path to .onnx model (default: auto-finds best_1.onnx or docs/best.onnx)")
    p.add_argument("--classes",   default=None,
                   help="Path to class-names JSON (auto-discovered if omitted)")
    p.add_argument("--image",     default=None,
                   help="Test on a single image file instead of webcam")
    p.add_argument("--camera",    type=int, default=0,
                   help="Webcam device index (default: 0)")
    p.add_argument("--conf",      type=float, default=0.25,
                   help="Confidence threshold (default: 0.25 — intentionally low for debugging)")
    p.add_argument("--iou",       type=float, default=0.45,
                   help="NMS IoU threshold (default: 0.45)")
    p.add_argument("--info-only", action="store_true",
                   help="Print model info + raw score dump on first frame then exit (no video window)")
    return p.parse_args()


def find_model():
    """Auto-find the ONNX model from known locations."""
    candidates = [
        "best_1.onnx",
        "docs/best.onnx",
        "best.onnx",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- find model ---------------------------------------------------------
    model_path = args.model or find_model()
    if not model_path or not Path(model_path).exists():
        print("[ERROR] No ONNX model found.")
        print("        Run:  python convert_best1.py   first to generate best_1.onnx")
        sys.exit(1)

    # ---- load model ---------------------------------------------------------
    print(f"\n[INFO] Loading model: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    inp  = session.get_inputs()[0]
    outp = session.get_outputs()[0]
    input_name = inp.name

    print(f"[INFO] Input  tensor: name={inp.name}  shape={inp.shape}  dtype={inp.type}")
    print(f"[INFO] Output tensor: name={outp.name}  shape={outp.shape}  dtype={outp.type}")

    # ---- load class names ---------------------------------------------------
    class_names = load_class_names(args.classes, model_path)

    print(f"\n[INFO] ---- Class names loaded ({len(class_names)}) ----")
    for idx, name in sorted(class_names.items()):
        print(f"         {idx:3d}: {name}")

    # ---- warm-up ------------------------------------------------------------
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    session.run(None, {input_name: dummy})
    print("\n[INFO] Warm-up done.")

    # =========================================================================
    # IMAGE mode
    # =========================================================================
    if args.image:
        img_path = args.image
        if not Path(img_path).exists():
            print(f"[ERROR] Image not found: {img_path}")
            sys.exit(1)

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[ERROR] Could not read image: {img_path}")
            sys.exit(1)

        # Always dump raw scores for image mode
        dump_raw_scores(session, input_name, class_names, frame)

        blob, ratio, pad = preprocess(frame)
        raw = session.run(None, {input_name: blob})
        boxes, confs, cls_ids = postprocess(raw, args.conf, args.iou, ratio, pad)

        print(f"\n[RESULT] Detections at conf>={args.conf}: {len(boxes)}")
        for i, (box, conf, cls) in enumerate(zip(boxes, confs, cls_ids)):
            name = class_names.get(int(cls), str(cls))
            print(f"   #{i+1}  class={name}  conf={conf:.4f}  box={list(map(int, box))}")

        annotated = draw_boxes(frame.copy(), boxes, confs, cls_ids, class_names)
        cv2.imshow("ONNX Test — press any key to close", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # =========================================================================
    # INFO-ONLY mode — grab one frame from webcam, dump scores, exit
    # =========================================================================
    if args.info_only:
        print(f"\n[INFO] Opening camera {args.camera} for one-frame score dump ...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {args.camera}")
            sys.exit(1)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("[ERROR] Failed to read frame from camera")
            sys.exit(1)
        dump_raw_scores(session, input_name, class_names, frame)
        print("\n[INFO-ONLY] Done.")
        return

    # =========================================================================
    # LIVE WEBCAM mode
    # =========================================================================
    print(f"\n[INFO] Opening camera {args.camera} ...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    print(f"[INFO] Confidence threshold: {args.conf}  (use --conf 0.1 to see even weak detections)")
    print("[INFO] Live feed started. Press  Q  to quit,  D  for a raw-score dump.\n")

    frame_i        = 0
    fps_timer      = time.time()
    fps            = 0.0
    last_boxes     = np.empty((0, 4))
    last_confs     = np.array([])
    last_cls_ids   = np.array([])
    dump_next      = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame read failed, retrying ...")
                time.sleep(0.1)
                continue

            frame_i += 1

            # ---- inference every frame (throttle with --skip if needed) -----
            blob, ratio, pad = preprocess(frame)
            raw = session.run(None, {input_name: blob})
            last_boxes, last_confs, last_cls_ids = postprocess(
                raw, args.conf, args.iou, ratio, pad
            )

            if dump_next:
                dump_raw_scores(session, input_name, class_names, frame)
                dump_next = False

            # ---- FPS --------------------------------------------------------
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = frame_i / (now - fps_timer + 1e-6)
                frame_i = 0
                fps_timer = now

            # ---- annotate ---------------------------------------------------
            display = draw_boxes(frame.copy(), last_boxes, last_confs, last_cls_ids, class_names)

            det_count = len(last_boxes)
            status_color = (0, 255, 0) if det_count > 0 else (0, 100, 255)
            status_txt   = f"Detections: {det_count}  |  conf>={args.conf}  |  FPS={fps:.1f}"
            cv2.putText(display, status_txt, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(display, "Q=quit  D=score dump", (8, display.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Cane Toad Detector — ONNX debug", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            if key == ord('d'):
                print("\n[USER] Score dump requested for next frame...")
                dump_next = True

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released.")


if __name__ == "__main__":
    main()
