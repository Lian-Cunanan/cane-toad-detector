# Raspberry Pi Deployment Guide

Complete guide for deploying the Cane Toad Detector system on Raspberry Pi.

---

## 📦 What's Included

This deployment package contains:
- `detector_raspi.py` - Optimized ONNX detector for Pi
- `convert_to_onnx.py` - Converts PyTorch model to ONNX
- `setup.sh` - Automated dependency installation
- `setup_hotspot.sh` - WiFi hotspot configuration
- `diagnostics.sh` - System health checks
- `requirements.txt` - Python dependencies
- `README.md` - Quick reference
- `HOTSPOT_SETUP.md` - Detailed hotspot guide
- `QUICK_START.txt` - Fastest deployment path

---

## 🎯 Deployment Options

Choose the deployment method that fits your needs:

### Option 1: Field Deployment (WiFi Hotspot) ⭐ Recommended

Pi creates its own WiFi network. Mobile device connects directly.

**Pros:**
- No external WiFi needed
- Works anywhere (fields, remote areas)
- Simple mobile app setup
- Complete standalone system

**Setup:** Follow [HOTSPOT_SETUP.md](HOTSPOT_SETUP.md)

### Option 2: Local Network Deployment

Pi and mobile device connect to same WiFi router.

**Pros:**
- Shared internet access
- Connect multiple devices
- Easy for home/office testing

**Setup:** Standard setup with network IP address

### Option 3: Direct Ethernet Connection

Mobile device tethered via USB/Ethernet, Pi connected via Ethernet.

**Pros:**
- Stable wired connection
- No WiFi interference
- Good for stationary installations

**Setup:** Configure static IP on mobile device

---

## 🚀 Quick Start (Hotspot Mode)

For fastest deployment with WiFi hotspot:

### 1. Copy Files to Pi

```bash
# From your computer
scp -r raspi_deploy pi@raspberrypi.local:~/

# SSH into Pi
ssh pi@raspberrypi.local
cd ~/raspi_deploy
```

### 2. Run Automated Setup

```bash
# Install dependencies and convert model
chmod +x setup.sh
./setup.sh

# Configure WiFi hotspot
chmod +x setup_hotspot.sh
sudo ./setup_hotspot.sh
```

Follow prompts. Reboot when finished.

### 3. Configure Mobile App

After Pi reboots:
1. Connect phone to WiFi: **CaneToadDetector** (password: canetoad2024)
2. Open mobile app
3. Tap **⚙️ Connection Settings**
4. Select preset: **Raspberry Pi Hotspot (192.168.4.1:5000)**
5. Tap **TEST CONNECTION**
6. Tap **SAVE**

### 4. Start Detecting!

- Login or register account
- Navigate to Dashboard
- View live camera feeds
- Monitor detections in real-time

**Done! Your system is ready. ✅**

---

## 📋 Detailed Setup Instructions

### Prerequisites

**Hardware:**
- Raspberry Pi 3, 4, 400, or Zero W (with WiFi)
- Raspberry Pi Camera or USB webcam
- MicroSD card (16GB+ recommended)
- Power supply (5V 3A for Pi 4)
- SD card reader

**Software:**
- Raspberry Pi OS Lite or Desktop (latest)
- SSH enabled (for headless setup)
- Internet connection (for initial setup)

**Files:**
- `best.pt` (PyTorch YOLOv8 model) OR `best.onnx` (converted model)

---

### Step 1: Prepare Raspberry Pi

#### Flash Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Flash Raspberry Pi OS Lite (64-bit) to SD card
3. Enable SSH: Create empty file named `ssh` on boot partition
4. Configure WiFi (optional): Create `wpa_supplicant.conf` on boot partition:

```ini
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="YourWiFiName"
    psk="YourPassword"
    key_mgmt=WPA-PSK
}
```

5. Insert SD card in Pi and power on
6. Wait 1-2 minutes for boot

#### Connect via SSH

```bash
# Find Pi IP address
ping raspberrypi.local

# Or scan network
nmap -sn 192.168.1.0/24

# Connect
ssh pi@raspberrypi.local
# Default password: raspberry
```

#### Update System

```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

---

### Step 2: Copy Files to Pi

#### Option A: Using SCP (from computer)

```bash
# Copy entire folder
scp -r raspi_deploy pi@raspberrypi.local:~/

# Or individual files
scp best.pt pi@raspberrypi.local:~/raspi_deploy/
ssh pi@raspberrypi.local
```

#### Option B: Using Git

```bash
# Clone your repository
git clone https://github.com/yourusername/cane-toad-detector.git
cd cane-toad-detector/raspi_deploy
```

#### Option C: Using USB Drive

```bash
# Insert USB drive
sudo mkdir /mnt/usb
sudo mount /dev/sda1 /mnt/usb
cp -r /mnt/usb/raspi_deploy ~/
```

---

### Step 3: Install Dependencies

#### Automated Installation

```bash
cd ~/raspi_deploy
chmod +x setup.sh
./setup.sh
```

This will:
- Create Python virtual environment
- Install all dependencies
- Convert `best.pt` to `best.onnx` (if needed)
- Set up directories
- Check camera

#### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Convert model
python convert_to_onnx.py
```

---

### Step 4: Configure WiFi Hotspot (Recommended)

#### Automated Configuration

```bash
chmod +x setup_hotspot.sh
sudo ./setup_hotspot.sh
```

Follow prompts. System will configure:
- WiFi hotspot (SSID: CaneToadDetector)
- Static IP (192.168.4.1)
- Backend auto-start service
- DHCP for mobile devices

**See [HOTSPOT_SETUP.md](HOTSPOT_SETUP.md) for detailed instructions.**

#### Manual Configuration

See [HOTSPOT_SETUP.md](HOTSPOT_SETUP.md) for step-by-step manual setup.

---

### Step 5: Test the System

#### Check System Status

```bash
# Run diagnostics
chmod +x diagnostics.sh
./diagnostics.sh
```

#### Test Backend Manually

```bash
cd ~/raspi_deploy
source venv/bin/activate
python detector_raspi.py --help
```

Options:
- `--headless` - No display output (for production)
- `--skip N` - Process every Nth frame (default: 2)
- `--camera N` - Use camera device N (default: 0)
- `--conf FLOAT` - Confidence threshold (default: 0.25)

Example:
```bash
python detector_raspi.py --headless --skip 2
```

#### Test Camera

```bash
# List cameras
ls -l /dev/video*

# Test with OpenCV
python -c "import cv2; print('OpenCV version:', cv2.__version__); \
           cap = cv2.VideoCapture(0); \
           print('Camera opened:', cap.isOpened())"
```

---

### Step 6: Configure Mobile App

#### Connect to Pi Hotspot

1. Open phone WiFi settings
2. Connect to: **CaneToadDetector**
3. Password: **canetoad2024**
4. Wait for connection

#### Configure Backend URL

1. Open Cane Toad Detector app
2. Tap **⚙️ Connection Settings** on login screen
3. Choose preset or enter custom URL:
   - **Hotspot mode**: `http://192.168.4.1:5000`
   - **Local network**: `http://[Pi-IP]:5000` (find with `hostname -I`)
4. Tap **TEST CONNECTION** (should show ✅)
5. Tap **SAVE**

#### Register Account

1. Return to login screen
2. Tap **Register**
3. Create username and password
4. Login

---

### Step 7: Enable Auto-Start on Boot

Backend service was created by `setup_hotspot.sh`. To verify:

```bash
# Check service status
sudo systemctl status cane-toad-backend

# Enable auto-start
sudo systemctl enable cane-toad-backend

# Start now
sudo systemctl start cane-toad-backend

# View logs
sudo journalctl -u cane-toad-backend -f
```

To disable:
```bash
sudo systemctl stop cane-toad-backend
sudo systemctl disable cane-toad-backend
```

---

## 🔧 Configuration

### Camera Settings

Edit `detector_raspi.py` camera initialization (line ~98):

```python
def open_camera(camera_id=0):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    
    # Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS
    cap.set(cv2.CAP_PROP_FPS, 15)
```

### Backend Settings

Edit backend/app.py:

```python
# Port
app.run(host='0.0.0.0', port=5000)

# CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# JWT expiration
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
```

### Hotspot Settings

```bash
# Change SSID
sudo nmcli connection modify CaneToadHotspot \
  802-11-wireless.ssid "NewNetworkName"

# Change password
sudo nmcli connection modify CaneToadHotspot \
  802-11-wireless-security.psk "NewPassword123"

# Change IP
sudo nmcli connection modify CaneToadHotspot \
  ipv4.addresses 10.42.0.1/24

# Apply changes
sudo nmcli connection down CaneToadHotspot
sudo nmcli connection up CaneToadHotspot
```

---

## 📊 Monitoring

### View Logs

```bash
# Backend logs
sudo journalctl -u cane-toad-backend -f

# All system logs
sudo journalctl -f

# Last 100 lines
sudo journalctl -u cane-toad-backend -n 100
```

### System Status

```bash
# CPU/Memory
htop

# Temperature
vcgencmd measure_temp

# Network
ifconfig wlan0

# Disk space
df -h
```

### Performance Monitoring

```bash
# FPS counter built into detector
# Check logs for:
# [INFO] FPS: 12.34

# Network traffic
sudo iftop -i wlan0

# Process stats
top -p $(pgrep -f detector_raspi)
```

---

## 🔍 Troubleshooting

### Run Diagnostics

```bash
./diagnostics.sh
```

### Common Issues

#### Camera Not Working

```bash
# Check camera
ls -l /dev/video*
vcgencmd get_camera

# Enable camera
sudo raspi-config
# Interface Options → Camera → Enable

# Test
raspistill -o test.jpg
```

#### Hotspot Not Starting

```bash
# Check status
nmcli connection show

# Restart
sudo nmcli connection down CaneToadHotspot
sudo nmcli connection up CaneToadHotspot

# Unblock WiFi
sudo rfkill unblock wifi
```

#### Backend Not Starting

```bash
# Check service
sudo systemctl status cane-toad-backend

# View errors
sudo journalctl -u cane-toad-backend -n 50

# Test manually
cd ~/raspi_deploy
source venv/bin/activate
python detector_raspi.py --headless
```

#### Mobile App Can't Connect

```bash
# Check Pi IP
hostname -I

# Test from phone browser
# Open: http://192.168.4.1:5000

# Check firewall
sudo ufw status
sudo ufw allow 5000
```

#### Low FPS / Performance Issues

```python
# Increase frame skip in detector_raspi.py
python detector_raspi.py --skip 4

# Lower camera resolution (detector_raspi.py line ~98)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Reduce detection confidence (fewer false positives)
python detector_raspi.py --conf 0.4
```

---

## 🛡️ Security

### Change Default Credentials

```bash
# Change Pi password
passwd

# Change hotspot password
sudo nmcli connection modify CaneToadHotspot \
  802-11-wireless-security.psk "SecurePassword123!"

# Restart hotspot
sudo nmcli connection down CaneToadHotspot
sudo nmcli connection up CaneToadHotspot
```

### Firewall Configuration

```bash
# Install UFW
sudo apt install ufw

# Allow SSH
sudo ufw allow 22

# Allow backend
sudo ufw allow 5000

# Enable
sudo ufw enable
```

### SSL/HTTPS (Optional)

For production deployments, consider adding SSL:

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Update backend to use HTTPS
# (requires code modification)
```

---

## 📚 Additional Resources

- **Quick Start**: [QUICK_START.txt](QUICK_START.txt)
- **Hotspot Setup**: [HOTSPOT_SETUP.md](HOTSPOT_SETUP.md)
- **Backend API**: See backend/app.py for endpoints
- **Mobile App**: See mobile_app/README.md

---

## 🆘 Getting Help

If you encounter issues:

1. Run diagnostics: `./diagnostics.sh`
2. Check logs: `sudo journalctl -u cane-toad-backend -f`
3. Test manually: `python detector_raspi.py --headless`
4. Verify camera: `ls -l /dev/video*`
5. Check network: `ifconfig wlan0`

For persistent issues, collect:
- Output of `./diagnostics.sh`
- Backend logs (last 100 lines)
- System info: `uname -a`, `python --version`

---

## ✅ Deployment Checklist

Before going to the field:

- [ ] Pi OS installed and updated
- [ ] Files copied to ~/raspi_deploy
- [ ] Dependencies installed (`./setup.sh`)
- [ ] Model converted to ONNX
- [ ] Hotspot configured (`sudo ./setup_hotspot.sh`)
- [ ] Backend service enabled and running
- [ ] Camera connected and detected
- [ ] Mobile app configured with Pi IP
- [ ] Test account created successfully
- [ ] Camera feeds visible in mobile app
- [ ] Detections showing up in dashboard
- [ ] Default passwords changed
- [ ] Full system test completed
- [ ] Backup power solution ready
- [ ] Documentation printed or saved offline

**Your system is ready for deployment! 🎉**
