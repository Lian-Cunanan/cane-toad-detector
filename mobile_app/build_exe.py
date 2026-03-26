#!/usr/bin/env python
"""
Build a Windows desktop executable (.exe) with PyInstaller.

Important:
    This output is for Windows PCs only.
    Android phones cannot run .exe files.
    Use Buildozer to create an APK/AAB for phones.
"""
import PyInstaller.__main__
import os
import sys

# Get the mobile_app directory
app_dir = os.path.dirname(os.path.abspath(__file__))

# PyInstaller arguments
args = [
    'main.py',
    '--onefile',
    '--name=Cane_Toad_Detector',
    f'--distpath={os.path.join(app_dir, "dist")}',
    f'--workpath={os.path.join(app_dir, "build")}',
    '--clean',
    
    # Hidden imports for KivyMD and Kivy
    '--hidden-import=kivymd.icon_definitions',
    '--hidden-import=kivymd.icon_definitions.md_icons',
    '--hidden-import=kivymd.uix.hero',
    '--hidden-import=screens.login_screen',
    '--hidden-import=screens.register_screen',
    '--hidden-import=screens.dashboard_screen',
    '--hidden-import=screens.camera_screen',
    '--hidden-import=screens.settings_screen',
    '--hidden-import=widgets.mjpeg_viewer',
    '--hidden-import=api_client',
    '--hidden-import=PIL',
    '--hidden-import=PIL._imaging',
    
    # Collect all KivyMD data
    '--collect-all=kivymd',
    '--collect-all=kivy',
    
    # Use console for debugging
    '-c',
]

print("Building Cane Toad Detector.exe (Windows desktop only)...")
print(f"Working directory: {app_dir}")

# Run PyInstaller
PyInstaller.__main__.run(args)
