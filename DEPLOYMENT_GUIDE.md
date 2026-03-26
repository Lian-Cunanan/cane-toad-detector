# 🐸 Cane Toad Detector — Complete System Documentation

An intelligent agricultural pest management system for detecting and monitoring cane toads using YOLOv8 ONNX, Raspberry Pi, and cross-platform mobile application.

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Components](#components)
- [Quick Start](#quick-start)
- [Deployment Guide](#deployment-guide)
- [API Documentation](#api-documentation)
- [Mobile App Features](#mobile-app-features)
- [Troubleshooting](#troubleshooting)

---

## 🎯 System Overview

The Cane Toad Detector system consists of three main components:

1. **ONNX Detector** - Real-time object detection using YOLOv8 ONNX model
2. **Flask Backend API** - RESTful API with WebSocket support for real-time updates
3. **Kivy Mobile App** - Cross-platform mobile application for monitoring and control

### Key Features

✅ **Real-time Detection** - ONNX-based YOLOv8 inference (no PyTorch required)  
✅ **Live Monitoring** - Dual camera feeds (Cage View & Trap View)  
✅ **Progress Tracking** - Visual batch progress with target counts  
✅ **Phase Management** - Capturing → Euthanizing → Disposing → Heat Sealing  
✅ **System Status** - Battery level, WiFi, camera status monitoring  
✅ **Real-time Alerts** - WebSocket notifications for detection events  
✅ **Secure Authentication** - JWT-based login system  
✅ **Mobile Control** - Batch reset, configuration from Android device  

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Raspberry Pi 4                         │
│                                                          │
│  ┌──────────────┐         ┌──────────────────────────┐ │
│  │   ONNX       │         │   Flask Backend API      │ │
│  │  Detector    │◄────────┤  - RESTful endpoints     │ │
│  │ (detector_   │         │  - WebSocket server      │ │
│  │  onnx.py)    │         │  - SQLite database       │ │
│  └──────────────┘         │  - JWT authentication    │ │
│         │                 └──────────────────────────┘ │
│         │                            │                  │
│  ┌──────▼──────┐                    │                  │
│  │  USB Camera  │                    │                  │
│  │  (A4Tech)    │                    │                  │
│  └──────────────┘                    │                  │
│                                      │                  │
│  ┌──────────────┐                    │                  │
│  │  ESP8266     │────────────────────┘                  │
│  │  (Sensors)   │  HTTP POSTs                          │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
                            │
                            │ WiFi/Network
                            │ HTTP + WebSocket
                            ▼
                  ┌──────────────────┐
                  │  Android Device  │
                  │                  │
                  │  Kivy Mobile App │
                  │  - Dashboard     │
                  │  - Camera feeds  │
                  │  - Control       │
                  └──────────────────┘
```

---

## 📦 Components

### 1. ONNX Detector

**Location:** `/detector_onnx.py` (Raspberry Pi) or `/test_laptop_onnx.py` (Windows)

**Purpose:** Real-time cane toad detection using ONNX Runtime

**Features:**
- YOLOv8 ONNX inference (lightweight, no PyTorch)
- Configurable confidence thresholds
- Frame skipping for performance optimization
- Bounding box visualization
- Detection logging

**Usage:**
```bash
# On Raspberry Pi
python detector_onnx.py --model best.onnx --camera 0 --conf 0.5 --skip 2

# On Windows (testing)
python test_laptop_onnx.py --model docs/best.onnx --camera 0 --conf 0.5
```

### 2. Flask Backend API

**Location:** `/backend/`

**Files:**
- `app.py` - Main Flask application with API routes
- `database.py` - SQLAlchemy models (User, Detection, SystemStatus, etc.)
- `camera_stream.py` - MJPEG camera streaming
- `requirements.txt` - Python dependencies

**Endpoints:**

#### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and get JWT token
- `POST /api/auth/logout` - Logout

#### Detections
- `GET /api/detections/current` - Current batch status
- `GET /api/detections/history?page=1` - Detection history
- `GET /api/detections/stats` - Statistics

#### System Status
- `GET /api/status` - Current system status
- `POST /api/status/update` - Update status (ESP8266)

#### Batch Control
- `POST /api/batch/reset` - Reset batch count
- `GET /api/batch/settings` - Get settings
- `POST /api/batch/settings` - Update settings

#### Camera Streaming
- `GET /api/camera/cage/stream` - Cage camera MJPEG stream
- `GET /api/camera/trap/stream` - Trap camera MJPEG stream
- `GET /api/camera/status` - Camera availability

#### WebSocket Events (namespace: `/ws`)
- `detection_alert` - New toad detected
- `status_update` - System status changed
- `phase_update` - Operational phase changed
- `batch_reset` - Batch count reset

### 3. Kivy Mobile App

**Location:** `/mobile_app/`

**Files:**
- `main.py` - Main application entry point
- `api_client.py` - Backend API communication
- `screens/login_screen.py` - Login UI
- `screens/register_screen.py` - Registration UI
- `screens/dashboard_screen.py` - Main dashboard with monitoring
- `screens/camera_screen.py` - Dual camera feed viewer
- `buildozer.spec` - Android build configuration

**Screens:**

1. **Login Screen**
   - Username/password authentication
   - Error handling with alerts
   - Navigation to registration

2. **Registration Screen**
   - New user account creation
   - Password confirmation
   - Validation with error messages

3. **Dashboard Screen**
   - Real-time detection count (current/target)
   - Animated progress bar
   - Operational phase indicators
   - System status (battery, WiFi, camera)
   - Last detection timestamp
   - Batch reset button

4. **Camera Screen**
   - Cage View live feed
   - Trap View live feed
   - Camera status indicators
   - Refresh controls

---

## 🚀 Quick Start

### Prerequisites

**Hardware:**
- Raspberry Pi 4 (2GB+ RAM recommended)
- USB webcam (A4Tech or compatible)
- Android smartphone/tablet
- WiFi network

**Software:**
- Raspberry Pi OS 64-bit (Bookworm)
- Python 3.9+
- Windows PC (for development/testing)

### Step 1: Set Up Raspberry Pi Backend

```bash
# SSH into Raspberry Pi
ssh pi@your_pi_ip_address

# Install system dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv \
                   libgl1-mesa-glx libglib2.0-0 \
                   libsm6 libxext6 libxrender-dev

# Create project directory
mkdir ~/cane_toad_detector
cd ~/cane_toad_detector

# Copy backend files from Windows
# (Use scp or WinSCP to transfer files)

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt

# Copy ONNX model and class names
cp docs/best.onnx ~/cane_toad_detector/
cp docs/class_names.json ~/cane_toad_detector/

# Run backend
cd backend
python app.py
```

Backend will start on `http://0.0.0.0:5000`

### Step 2: Test on Windows (Optional)

```powershell
# In Windows PowerShell
cd "C:\Users\Carlo\Desktop\AI Components"

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements_onnx.txt

# Test detector
python test_laptop_onnx.py --model docs/best.onnx

# Test backend
cd backend
pip install -r requirements.txt
python app.py
```

### Step 3: Set Up Mobile App

**On Windows (Development):**

```powershell
cd mobile_app

# Install dependencies
pip install -r requirements.txt

# Update backend URL in main.py
# Change: backend_url = 'http://192.168.1.100:5000'
# To your Pi's IP address

# Run app
python main.py
```

**For Android Deployment:**

```bash
# On Linux (Ubuntu recommended)
cd mobile_app

# Install buildozer
pip install buildozer

# Build APK (first time takes ~30 minutes)
buildozer android debug

# APK will be in: bin/canetoaddetector-1.0.0-armeabi-v7a-debug.apk

# Transfer to Android device and install
adb install bin/canetoaddetector-1.0.0-armeabi-v7a-debug.apk
```

Important:
- A `.exe` file is a Windows desktop binary and will not run on Android phones.
- For Windows hosts, use WSL and run `mobile_app/build_android_wsl.ps1` to produce an APK.

### Step 4: Connect Everything

1. **Start Backend on Pi:**
   ```bash
   cd ~/cane_toad_detector/backend
   python app.py
   ```

2. **Start Detector on Pi:**
   ```bash
   cd ~/cane_toad_detector
   python detector_onnx.py --model best.onnx
   ```

3. **Open Mobile App on Android:**
   - Enter Pi's IP address in settings (or hardcode in `main.py`)
   - Register a new account
   - Login and start monitoring!

---

## 🔧 Configuration

### Backend Configuration

Edit `backend/app.py`:

```python
# Change these in production
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['JWT_SECRET_KEY'] = 'jwt-secret-key-change-this'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
```

### Mobile App Configuration

Edit `mobile_app/main.py`:

```python
# Set your Raspberry Pi's IP address
backend_url = StringProperty('http://192.168.1.100:5000')
```

### Detector Configuration

```bash
# Confidence threshold (0.0 to 1.0)
--conf 0.5

# Frame skipping (1 = every frame, 2 = every other frame)
--skip 2

# Camera index
--camera 0

# Target batch size (modify in app)
# Default: 10 toads per batch
```

---

## 📱 Mobile App Features

### Dashboard

- **Live Detection Counter** - Shows current batch progress (e.g., 7/10)
- **Progress Bar** - Visual indicator with percentage
- **Phase Indicators** - Shows current operation (Capturing, Euthanizing, etc.)
- **System Status**
  - Battery level with color coding (green > 20%, red < 20%)
  - WiFi connection status
  - Camera active/inactive
  - Last detection timestamp

### Controls

- **Next Batch / Reset** - Resets counter to 0, starts new batch
- **View Camera Feeds** - Opens dual camera view
- **Logout** - Securely logout

### Real-time Updates

- **Detection Alerts** - Popup when new toad detected
- **Status Changes** - Automatic refresh when status changes
- **Phase Transitions** - Visual feedback for workflow stages

---

## 🐛 Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError: No module named 'flask'`
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install -r backend/requirements.txt
```

**Problem:** Camera not accessible
```bash
# Check camera devices
ls /dev/video*

# Try different camera index
python detector_onnx.py --camera 1
```

**Problem:** `[ONNXRuntimeError] : 3 : NO_SUCHFILE`
```bash
# Ensure model file exists
ls best.onnx

# Use full path
python detector_onnx.py --model /home/pi/cane_toad_detector/best.onnx
```

### Mobile App Issues

**Problem:** Cannot connect to backend
- Check Pi's IP address: `hostname -I`
- Update `backend_url` in `main.py`
- Ensure Pi and phone are on same WiFi network
- Check firewall: `sudo ufw allow 5000`

**Problem:** Camera feeds not loading
- Verify backend is running
- Check `/api/camera/status` endpoint
- Ensure JWT token is valid (re-login)

**Problem:** Kivy import errors
```bash
pip install --upgrade pip
pip install kivy[base] kivymd
```

### Detector Issues

**Problem:** Low FPS on Raspberry Pi
```bash
# Increase frame skipping
python detector_onnx.py --skip 3

# Reduce resolution (edit detector_onnx.py)
--width 320 --height 240
```

**Problem:** No detections
- Lower confidence threshold: `--conf 0.3`
- Check if model is loaded correctly
- Verify class names match training data

---

## 📝 Development Notes

### Adding New API Endpoints

1. Add route in `backend/app.py`
2. Implement database logic if needed
3. Add corresponding method in `mobile_app/api_client.py`
4. Call from Kivy screen

### Modifying UI

Edit Kivy Builder strings in `screens/*.py` files. Use KivyMD components for consistent Material Design.

### Database Schema Changes

1. Modify models in `backend/database.py`
2. Delete `cane_toad_detector.db` to recreate (dev only)
3. For production, use Flask-Migrate for migrations

---

## 📄 License

This project is for agricultural research and pest management purposes.

---

## 👥 Support

For issues or questions:
1. Check this documentation
2. Review error logs in terminal
3. Test individual components separately

---

## 🎉 Final Notes

**Default Login (after first setup):**
- You must register a new account first
- No default credentials for security

**Production Deployment:**
- Change all secret keys
- Use HTTPS (reverse proxy with nginx)
- Set up proper authentication
- Consider adding user roles/permissions
- Implement proper logging
- Set up auto-start services (systemd)

**Performance Tips:**
- Use `--skip 2` or higher for Pi
- Consider using lighter ONNX model (quantized)
- Limit WebSocket broadcast frequency
- Use Redis for session storage (optional)

---

**System Ready! 🚀**

Start the backend, run the detector, and open the mobile app to begin monitoring your cane toad capture system!
