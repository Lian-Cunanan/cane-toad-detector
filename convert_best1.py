# =============================================================================
# Convert  "best (1).pt"  ->  best_1.onnx  +  best_1_classes.json
# =============================================================================
#
# Usage (activate your venv first):
#
#   venv\Scripts\activate
#   python convert_best1.py
#
# Output files:
#   best_1.onnx            — ONNX model for ONNX Runtime / Raspberry Pi
#   best_1_classes.json    — class-name map  {"0": "cane_toad", ...}
#
# After confirming detections work, copy to the Pi:
#   scp best_1.onnx            aldrich@192.168.4.1:~/Desktop/RASPI_DEPLOY/
#   scp best_1_classes.json    aldrich@192.168.4.1:~/Desktop/RASPI_DEPLOY/
# =============================================================================

import json
import os
import shutil
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed.")
    print("        Run:  pip install ultralytics")
    sys.exit(1)

# ---- paths -------------------------------------------------------------------
SRC_PT   = r"C:\Users\Carlo\Desktop\AI Components\best (1).pt"
OUT_ONNX = r"C:\Users\Carlo\Desktop\AI Components\best_1.onnx"
OUT_JSON = r"C:\Users\Carlo\Desktop\AI Components\best_1_classes.json"
# ------------------------------------------------------------------------------

if not Path(SRC_PT).exists():
    print(f"[ERROR] Model file not found: {SRC_PT}")
    print("        Make sure 'best (1).pt' is in the same folder as this script.")
    sys.exit(1)

print(f"[INFO] Loading model: {SRC_PT}")
model = YOLO(SRC_PT)

# Print class names so you can verify them before testing
print(f"\n[INFO] ---- Class names inside this model ----")
for idx, name in model.names.items():
    print(f"         {idx}: {name}")
print()

print("[INFO] Exporting to ONNX ...")
exported = model.export(
    format   = "onnx",
    imgsz    = 640,
    simplify = True,
    dynamic  = False,
    opset    = 12,      # broad ONNX Runtime compatibility
)
print(f"[INFO] Raw export path: {exported}")

# Ultralytics saves the ONNX next to the .pt file; move it to our target name
exported_path = Path(str(exported))
if exported_path.exists() and str(exported_path) != OUT_ONNX:
    shutil.move(str(exported_path), OUT_ONNX)

print(f"[INFO] ONNX model saved: {OUT_ONNX}")

# Save class-name JSON alongside ONNX
names_dict = {str(k): v for k, v in model.names.items()}
with open(OUT_JSON, "w") as f:
    json.dump(names_dict, f, indent=2)
print(f"[INFO] Class names saved: {OUT_JSON}")

print("\n[DONE]")
print("="*60)
print("Next step — run the diagnostic test:")
print()
print("  python test_onnx_debug.py --model best_1.onnx --classes best_1_classes.json")
print()
print("If you want to test on a saved image instead of webcam:")
print("  python test_onnx_debug.py --model best_1.onnx --classes best_1_classes.json --image path/to/photo.jpg")
print("="*60)
