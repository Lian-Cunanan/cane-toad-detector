#!/usr/bin/env python3
# =============================================================================
# PyTorch to ONNX Converter for YOLOv8 Models
# =============================================================================
#
# Converts best.pt (PyTorch) to best.onnx for use with ONNX Runtime.
#
# Usage:
#   python convert_to_onnx.py
#
# This script:
#   1. Loads best.pt using ultralytics
#   2. Exports to best.onnx
#   3. Simplifies the model
#   4. Verifies the conversion
#
# Requirements:
#   pip install ultralytics onnx onnxsim
#
# Note: After conversion, you can uninstall PyTorch to save space:
#   pip uninstall torch torchvision torchaudio ultralytics
#
# =============================================================================

import sys
from pathlib import Path

def convert_to_onnx():
    """Convert YOLOv8 PyTorch model to ONNX format."""
    
    # Check if best.pt exists
    model_path = Path("best.pt")
    if not model_path.exists():
        print("[ERROR] best.pt not found in current directory!")
        print("[ERROR] Please copy your trained model here first:")
        print(f"[ERROR]   scp best.pt pi@raspberrypi.local:~/raspi_deploy/")
        sys.exit(1)
    
    print("[INFO] Found best.pt")
    print("[INFO] Starting conversion to ONNX...")
    print()
    
    try:
        # Import ultralytics (this also imports PyTorch)
        print("[INFO] Importing ultralytics...")
        from ultralytics import YOLO
        
        # Load PyTorch model
        print(f"[INFO] Loading PyTorch model: {model_path}")
        model = YOLO(str(model_path))
        
        # Export to ONNX
        print("[INFO] Exporting to ONNX format...")
        print("[INFO] This may take a few minutes...")
        model.export(
            format='onnx',
            imgsz=640,
            simplify=True,
            opset=12,
            dynamic=False
        )
        
        # Check if conversion succeeded
        onnx_path = Path("best.onnx")
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print()
            print(f"[SUCCESS] Conversion complete!")
            print(f"[SUCCESS] ONNX model created: {onnx_path}")
            print(f"[SUCCESS] File size: {size_mb:.2f} MB")
            print()
            print("[INFO] You can now run the detector:")
            print("[INFO]   python detector_raspi.py --headless --skip 2")
            print()
            print("[OPTIONAL] To save space, uninstall PyTorch:")
            print("[OPTIONAL]   pip uninstall -y torch torchvision torchaudio ultralytics")
            print("[OPTIONAL] (ONNX Runtime is much smaller and faster on Pi)")
        else:
            print()
            print("[ERROR] Conversion failed - best.onnx not created")
            sys.exit(1)
    
    except ImportError as e:
        print()
        print("[ERROR] ultralytics not installed!")
        print("[ERROR] Install it with:")
        print("[ERROR]   pip install ultralytics")
        print()
        print(f"[ERROR] Details: {e}")
        sys.exit(1)
    
    except Exception as e:
        print()
        print(f"[ERROR] Conversion failed: {e}")
        print()
        print("[HELP] Common issues:")
        print("[HELP]   - Corrupted best.pt file")
        print("[HELP]   - Incompatible ultralytics version")
        print("[HELP]   - Out of memory (Pi needs ~2GB free)")
        sys.exit(1)


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("YOLOv8 PyTorch to ONNX Converter")
    print("=" * 70)
    print()
    convert_to_onnx()
