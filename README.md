# 🐸 Cane Toad Detector

> Real-time cane toad detection using **YOLOv8** — runs on a laptop webcam or deploys to a **Raspberry Pi 4** with a one-way mesh gate trigger.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Raspberry%20Pi%204-green?style=flat-square)

---

## What It Does

Captures a live camera feed, runs a custom-trained YOLOv8 model (`best.pt`), and displays annotated detections in real time — either in a desktop window or a browser-based web UI.

When deployed on Raspberry Pi 4, a GPIO stub is ready to trigger **3 servo motors** that open a one-way mesh gate upon detection.

---

## Features

- **Live detection** with bounding boxes and confidence scores
- **Browser UI** (Flask) — dark-themed, live sidebar with per-detection confidence bars
- **Raspberry Pi 4 ready** — optimised for ARM64 with frame-skip and buffer controls
- **Servo motor stub** — GPIO integration point already in place for hardware deployment
- **Clean terminal output** — frame-by-frame detection logs with class, confidence and bbox

---

## Project Structure

```
cane-toad-detector/
├── best.pt              # Trained YOLOv8 model
├── app.py               # Flask web app (browser UI)
├── detector.py          # Raspberry Pi 4 standalone script
├── test_laptop.py       # Quick webcam test (no server needed)
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Web UI
└── .gitignore
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Lian-Cunanan/cane-toad-detector.git
cd cane-toad-detector

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux / Pi

pip install -r requirements.txt
```

### 2. Run options

| Mode | Command | Best for |
|---|---|---|
| Browser web app | `python app.py` | Laptop demo / testing |
| Standalone window | `python test_laptop.py` | Quick local test |
| Raspberry Pi | `python detector.py` | Hardware deployment |

> **Web app:** After running `python app.py`, open **http://localhost:5000** in your browser.

---

## Web UI Preview

```
┌─────────────────────────────────────┬──────────────────┐
│                                     │  Live Detections │
│                                     │                  │
│        [ Live Camera Feed ]         │  2 detected      │
│        [ Bounding Boxes   ]         │                  │
│        [ Confidence Labels]         │  cane toad 91%  │
│                                     │  ████████████░░  │
│                                     │  cane toad 78%  │
│  FPS: 24.3  |  conf >= 0.50         │  ████████░░░░░░  │
└─────────────────────────────────────┴──────────────────┘
```

---

## Raspberry Pi 4 Setup

```bash
# System dependencies
sudo apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0

# Virtual environment
python3 -m venv ~/cane_toad_env
source ~/cane_toad_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python detector.py
```

> **Performance tip:** Use `--skip 2` to run inference every other frame — keeps the display smooth at ~24 FPS on Pi 4.

```bash
python detector.py --skip 2 --conf 0.5 --width 640 --height 480
```

---

## Hardware Deployment (Pi 4 + Servo Gate)

The `on_cane_toad_detected()` function in `detector.py` is a ready-made stub for GPIO servo control. When a toad is detected, it receives:

```python
on_cane_toad_detected(
    detection_count = 2,
    bbox_xyxy       = [[x1, y1, x2, y2], ...],
    confidence      = 0.91
)
```

Fill in the stub with your `RPi.GPIO` or `gpiozero` servo logic to activate the mesh gate.

---

## Requirements

```
ultralytics >= 8.0.0
opencv-python >= 4.8.0
numpy >= 1.24.0
flask >= 3.0.0
```

> Raspberry Pi 4 note: if PyTorch fails to install via pip, run:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
> then re-run `pip install -r requirements.txt`

---

## Roadmap

- [x] YOLOv8 live detection on laptop webcam
- [x] Flask browser UI with live sidebar
- [x] Raspberry Pi 4 optimised script
- [x] GPIO servo stub
- [ ] Servo motor gate implementation (3x servos via RPi.GPIO)
- [ ] Night vision / IR camera support
- [ ] Detection logging to CSV with timestamps
- [ ] Telegram / email alert on detection

---

## License

MIT — free to use, modify and distribute.
