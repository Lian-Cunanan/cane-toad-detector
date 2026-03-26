#!/usr/bin/env python3
"""
Test script to identify which Python package causes "Illegal instruction" error.
Run this on the Raspberry Pi to diagnose import issues.
"""

import sys
import subprocess

# List of packages to test
PACKAGES = [
    "numpy",
    "cv2",
    "torch",
    "torchvision",
    "ultralytics",
    "PIL",
]

def test_import(package_name):
    """Test importing a package in a subprocess to catch illegal instruction errors."""
    print(f"Testing {package_name:15s} ... ", end="", flush=True)
    
    # Run import in subprocess so illegal instruction doesn't kill this script
    cmd = [sys.executable, "-c", f"import {package_name}; print('OK')"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✓ OK")
            return True
        elif result.returncode == 132:  # SIGILL (Illegal instruction)
            print("✗ ILLEGAL INSTRUCTION")
            return False
        else:
            print(f"✗ ERROR (code {result.returncode})")
            if result.stderr:
                print(f"    {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ {e}")
        return False


def main():
    print("=" * 60)
    print("Python Import Diagnostic Tool for Raspberry Pi")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("=" * 60)
    print()
    
    results = {}
    for package in PACKAGES:
        results[package] = test_import(package)
    
    print()
    print("=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    working = [pkg for pkg, ok in results.items() if ok]
    broken = [pkg for pkg, ok in results.items() if not ok]
    
    if working:
        print(f"✓ Working packages ({len(working)}):")
        for pkg in working:
            print(f"  - {pkg}")
    
    if broken:
        print(f"\n✗ Problematic packages ({len(broken)}):")
        for pkg in broken:
            print(f"  - {pkg}")
        print("\nRECOMMENDATION:")
        print("  Reinstall the problematic packages with:")
        for pkg in broken:
            pypi_name = "opencv-python" if pkg == "cv2" else pkg
            print(f"    pip uninstall -y {pypi_name}")
            print(f"    pip install {pypi_name}")
    else:
        print("\n✓ All packages working correctly!")


if __name__ == "__main__":
    main()
