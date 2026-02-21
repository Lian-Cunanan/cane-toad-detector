# =============================================================================
# Export best.pt → best.onnx for browser-based inference (GitHub Pages)
# =============================================================================
#
# Run this ONCE on your laptop before pushing to GitHub:
#
#   venv\Scripts\activate
#   python export_model.py
#
# This will create:
#   docs/best.onnx        — model for the browser
#   docs/class_names.json — class labels used by the web page
#
# After running, commit and push:
#   git add docs/
#   git commit -m "Add ONNX model and web detector"
#   git push
#
# Then enable GitHub Pages:
#   GitHub repo → Settings → Pages → Branch: main → Folder: /docs → Save
#
# Your live URL will be:
#   https://lian-cunanan.github.io/cane-toad-detector/
# =============================================================================

import json
import os
import shutil

from ultralytics import YOLO

MODEL_PATH = "best.pt"
DOCS_DIR   = "docs"

print(f"[INFO] Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print(f"[INFO] Class names: {model.names}")

print("[INFO] Exporting to ONNX (this may take a minute)...")
exported_path = model.export(
    format   = "onnx",
    imgsz    = 640,
    simplify = True,    # Simplify graph for smaller file size
    dynamic  = False,   # Fixed input shape required for browser inference
)
print(f"[INFO] Exported: {exported_path}")

# Move ONNX file into docs/
os.makedirs(DOCS_DIR, exist_ok=True)
dest_onnx = os.path.join(DOCS_DIR, "best.onnx")
shutil.move(str(exported_path), dest_onnx)
print(f"[INFO] Moved to: {dest_onnx}")

# Save class names as JSON for the web page to load
names_dict = {str(k): v for k, v in model.names.items()}
dest_names = os.path.join(DOCS_DIR, "class_names.json")
with open(dest_names, "w") as f:
    json.dump(names_dict, f, indent=2)
print(f"[INFO] Class names saved: {dest_names}")

print("\n[DONE] ✓")
print("=" * 60)
print("Next steps:")
print()
print("  1. Commit and push:")
print("       git add docs/")
print('       git commit -m "Add ONNX model and web detector"')
print("       git push")
print()
print("  2. Enable GitHub Pages:")
print("       GitHub repo → Settings → Pages")
print("       Branch: main   Folder: /docs   → Save")
print()
print("  3. Your live site will be at:")
print("       https://lian-cunanan.github.io/cane-toad-detector/")
print("=" * 60)
