# 🐸 Cane Toad Detector — Complete System

A complete IoT agricultural pest management system with AI detection, Flask backend API, and Kivy mobile app.

## 🎯 What's Included

### ✅ ONNX Detectors (No PyTorch Needed!)
- `detector_onnx.py` - Raspberry Pi detector
- `test_laptop_onnx.py` - Windows testing version
- `requirements_onnx.txt` - Minimal dependencies

### ✅ Flask Backend API (`/backend/`)
- RESTful API with authentication (JWT)
- WebSocket for real-time updates
- SQLite database
- Camera streaming (MJPEG)
- Complete detection logging
- ESP8266 integration endpoints

### ✅ Kivy Mobile App (`/mobile_app/`)
- Login/Registration screens
- Real-time dashboard with progress tracking
- Dual camera feeds (Cage + Trap view)
- Operational phase monitoring
- System status display (battery, WiFi, etc.)
- Batch control and alerts
- Android APK buildable with Buildozer

## 🚀 Quick Test (Windows)

### Test Backend:
```powershell
cd backend
pip install -r requirements.txt
python test_backend.py
```
Backend runs at: `http://localhost:5000`

### Test Mobile App:
```powershell
cd mobile_app
pip install -r requirements.txt
# Edit main.py line 41: backend_url = 'http://localhost:5000'
python main.py
```

### Test Detector (with your webcam):
```powershell
pip install -r requirements_onnx.txt
python test_laptop_onnx.py --model docs/best.onnx
```

## 📱 Mobile App Quick Start

1. **Registration:**
   - Open app
   - Click "Don't have an account? Register"
   - Create username/password (min 3/6 chars)

2. **Dashboard Features:**
   - See live detection count (current/target)
   - Monitor operational phases
   - Check battery, WiFi, camera status
   - Reset batch with confirmation dialog

3. **Camera Feeds:**
   - Click "VIEW CAMERA FEEDS"
   - See Cage View and Trap View
   - Live MJPEG streams

## 🔧 For Raspberry Pi Deployment

See **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** for complete setup instructions.

Quick steps:
```bash
# On Pi
cd backend
pip install -r requirements.txt
python app.py

# In another terminal
python ../detector_onnx.py --model best.onnx
```

## 📁 Project Structure

```
AI Components/
├── backend/
│   ├── app.py                  # Flask API server
│   ├── database.py             # SQLAlchemy models
│   ├── camera_stream.py        # MJPEG streaming
│   ├── test_backend.py         # Simulator for testing
│   └── requirements.txt
├── mobile_app/
│   ├── main.py                 # Kivy app entry point
│   ├── api_client.py           # Backend communication
│   ├── screens/
│   │   ├── login_screen.py
│   │   ├── register_screen.py
│   │   ├── dashboard_screen.py
│   │   └── camera_screen.py
│   ├── buildozer.spec          # Android build config
│   └── requirements.txt
├── docs/
│   ├── best.onnx              # YOLOv8 model
│   └── class_names.json       # Model classes
├── detector_onnx.py           # Raspberry Pi detector
├── test_laptop_onnx.py        # Windows test detector
├── requirements_onnx.txt      # Detector dependencies
└── DEPLOYMENT_GUIDE.md        # Full documentation

```

## 🎨 Tech Stack

- **Backend:** Flask, Flask-SocketIO, SQLAlchemy, JWT
- **Mobile:** Kivy, KivyMD, python-socketio
- **AI:** ONNX Runtime, YOLOv8
- **Computer Vision:** OpenCV
- **Database:** SQLite

## 📖 API Endpoints

- `POST /api/auth/login` - Login
- `GET /api/detections/current` - Current batch status
- `GET /api/status` - System status
- `POST /api/batch/reset` - Reset batch
- `GET /api/camera/cage/stream` - Camera feed
- WebSocket: `/ws` namespace for real-time updates

## 🔐 Default Credentials

**No default credentials** - Register first user in mobile app.

## 🐛 Common Issues

1. **"Illegal instruction" error?** ✅ Fixed! Use ONNX versions
2. **Can't connect to backend?** Change `backend_url` in `main.py` to your Pi's IP
3. **Camera not found?** Try `--camera 1` or check `/dev/video*`
4. **No module named 'kivy'?** Run `pip install -r mobile_app/requirements.txt`

## 📚 Documentation

Full deployment guide: **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**

Includes:
- Complete setup instructions
- Architecture diagrams
- API documentation
- Troubleshooting guide
- Production deployment tips

## 🎉 Ready to Use!

Everything is set up and ready to deploy. Start with testing on Windows, then move to Raspberry Pi when ready!

---

**Made with ❤️ for agricultural pest management**
