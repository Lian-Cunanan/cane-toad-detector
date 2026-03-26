[app]

# Application title
title = Cane Toad Detector

# Package name (must be unique)
package.name = canetoaddetector

# Package domain (used for Android package)
package.domain = com.agriculture

# Source code directory
source.dir = .

# Source files to include
source.include_exts = py,png,jpg,kv,atlas,json

# Application version
version = 1.0.0

# Application requirements
# Keep this in sync with imports used by the app (Pillow is required by MJPEGViewer).
requirements = python3,kivy,kivymd,requests,python-socketio,websocket-client,pillow

# Application permissions
android.permissions = INTERNET,CAMERA,VIBRATE

# Android API level
android.api = 31
android.minapi = 21
android.ndk = 25b

# Orientation (landscape, portrait, or all)
orientation = portrait

# Application icon
icon.filename = %(source.dir)s/assets/icon.png

# Presplash background color
#presplash.filename = %(source.dir)s/assets/presplash.png

# Android architecture
android.archs = arm64-v8a

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

[buildozer]

# Log level (0 = error only, 1 = info, 2 = debug)
log_level = 2

# Display warning if buildozer is run as root
warn_on_root = 1
