# Raspberry Pi WiFi Hotspot Setup Guide

This guide explains how to configure your Raspberry Pi as a WiFi hotspot so your mobile device can connect directly to it without requiring an external WiFi network. Perfect for field deployment!

---

## 📋 Prerequisites

- Raspberry Pi with built-in WiFi (Pi 3, Pi 4, Pi 400, Pi Zero W)
- Raspberry Pi OS Lite or Desktop installed
- SSH access or monitor/keyboard connected
- Mobile device with the Cane Toad Detector app installed

---

## 🚀 Quick Setup (Automated)

### Option 1: Run the Automated Script

```bash
cd raspi_deploy
chmod +x setup_hotspot.sh
sudo ./setup_hotspot.sh
```

The script will:
- Install required packages (NetworkManager, dnsmasq)
- Configure WiFi hotspot with default settings
- Set up IP address (192.168.4.1)
- Create SSID: `CaneToadDetector`
- Password: `canetoad2024`
- Start backend on boot

After reboot, your Pi will broadcast its own WiFi network!

---

## 🔧 Manual Setup (Step by Step)

### Step 1: Install Required Packages

```bash
sudo apt update
sudo apt install -y network-manager dnsmasq hostapd
```

### Step 2: Configure NetworkManager Hotspot

Create a hotspot using nmcli (easiest method):

```bash
sudo nmcli device wifi hotspot \
  ifname wlan0 \
  con-name CaneToadHotspot \
  ssid CaneToadDetector \
  band bg \
  channel 6 \
  password canetoad2024
```

**Hotspot Settings:**
- **SSID**: CaneToadDetector (visible network name)
- **Password**: canetoad2024 (change this for security!)
- **Channel**: 6 (2.4GHz - best compatibility)
- **IP Address**: 192.168.4.1 (default for Pi hotspots)

### Step 3: Configure Static IP Address

```bash
sudo nmcli connection modify CaneToadHotspot \
  ipv4.addresses 192.168.4.1/24 \
  ipv4.method shared
```

### Step 4: Enable and Start Hotspot

```bash
# Activate the hotspot
sudo nmcli connection up CaneToadHotspot

# Enable on boot
sudo nmcli connection modify CaneToadHotspot connection.autoconnect yes
```

### Step 5: Verify Hotspot is Running

```bash
# Check connection status
nmcli connection show --active

# Check IP address
ip addr show wlan0
```

You should see:
- Connection: CaneToadHotspot (active)
- IP: 192.168.4.1

---

## 📱 Configure Mobile App

### Update Backend URL

When you open the mobile app for the first time:

1. **Tap "⚙️ Connection Settings"** on login screen
2. **Select Preset**: "Raspberry Pi Hotspot (192.168.4.1:5000)"
3. **Tap "TEST CONNECTION"** to verify
4. **Tap "SAVE"**

Or enter manually:
```
http://192.168.4.1:5000
```

---

## 🔄 Backend Auto-Start on Boot

### Create Systemd Service

Create service file:
```bash
sudo nano /etc/systemd/system/cane-toad-backend.service
```

Add this content:
```ini
[Unit]
Description=Cane Toad Detector Backend
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/raspi_deploy
Environment="PATH=/home/pi/raspi_deploy/venv/bin"
ExecStart=/home/pi/raspi_deploy/venv/bin/python /home/pi/raspi_deploy/detector_raspi.py --headless
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable cane-toad-backend.service
sudo systemctl start cane-toad-backend.service

# Check status
sudo systemctl status cane-toad-backend.service
```

---

## 🌐 Testing the Setup

### From Raspberry Pi

Test backend is running:
```bash
curl http://localhost:5000/api/auth/register
```

### From Mobile Device

1. **Connect to WiFi**:
   - SSID: `CaneToadDetector`
   - Password: `canetoad2024`

2. **Open Mobile App**:
   - Login or register
   - Navigate to Dashboard
   - Check camera feeds

3. **Test Connection**:
   ```
   Open phone browser → http://192.168.4.1:5000
   ```
   Should show backend response

---

## 🔍 Troubleshooting

### Hotspot Not Visible

```bash
# Check WiFi is enabled
sudo rfkill unblock wifi

# Restart NetworkManager
sudo systemctl restart NetworkManager

# Reactivate hotspot
sudo nmcli connection down CaneToadHotspot
sudo nmcli connection up CaneToadHotspot
```

### Mobile App Can't Connect

1. **Verify Phone is Connected**:
   - Check WiFi settings on phone
   - Should show "CaneToadDetector"
   - Phone IP should be 192.168.4.x (2-255)

2. **Test Pi from Phone**:
   ```
   Open browser → http://192.168.4.1:5000
   ```

3. **Check Backend Running**:
   ```bash
   # On Pi
   ps aux | grep detector_raspi
   curl http://localhost:5000/api/auth/register
   ```

4. **Check Firewall**:
   ```bash
   # Disable firewall temporarily to test
   sudo ufw disable
   ```

### Backend Not Starting

```bash
# Check logs
sudo journalctl -u cane-toad-backend.service -f

# Restart service
sudo systemctl restart cane-toad-backend.service

# Test manually
cd /home/pi/raspi_deploy
source venv/bin/activate
python detector_raspi.py --headless
```

### Wrong IP Address

If your Pi uses a different IP (e.g., 10.42.0.1):

1. **Check IP**:
   ```bash
   ip addr show wlan0
   ```

2. **Update Mobile App**:
   - Open Settings → Enter custom URL
   - Example: `http://10.42.0.1:5000`

---

## 🔒 Security Considerations

### Change Default Password

```bash
sudo nmcli connection modify CaneToadHotspot \
  802-11-wireless-security.psk "YourNewSecurePassword123"

sudo nmcli connection down CaneToadHotspot
sudo nmcli connection up CaneToadHotspot
```

### Limit Connected Devices

```bash
# Maximum 4 devices
sudo nmcli connection modify CaneToadHotspot \
  802-11-wireless.max-connections 4
```

### MAC Address Filtering (Optional)

```bash
# Allow only specific devices
sudo nmcli connection modify CaneToadHotspot \
  802-11-wireless.mac-address-blacklist ""

sudo nmcli connection modify CaneToadHotspot \
  802-11-wireless.mac-address-whitelist "AA:BB:CC:DD:EE:FF,11:22:33:44:55:66"
```

---

## 📊 Monitoring Hotspot

### View Connected Devices

```bash
# Show all connections
sudo iw dev wlan0 station dump

# Or using arp
arp -a
```

### Check Network Traffic

```bash
# Install iftop
sudo apt install -y iftop

# Monitor traffic
sudo iftop -i wlan0
```

### View Logs

```bash
# NetworkManager logs
sudo journalctl -u NetworkManager -f

# Backend logs
sudo journalctl -u cane-toad-backend.service -f

# System logs
dmesg | grep wlan0
```

---

## 🌟 Advanced Configuration

### Dual Mode: Hotspot + Internet

Set up Pi to connect to WiFi AND create hotspot simultaneously:

1. **Use Ethernet for Internet**:
   - Connect Pi to router via Ethernet
   - WiFi creates hotspot
   - Internet shared to mobile devices

2. **Configure Routing**:
   ```bash
   sudo sysctl -w net.ipv4.ip_forward=1
   sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
   ```

### Change IP Range

```bash
# Use 10.42.0.1 instead of 192.168.4.1
sudo nmcli connection modify CaneToadHotspot \
  ipv4.addresses 10.42.0.1/24
```

Update mobile app settings to match!

---

## 📝 Default Settings Summary

| Setting | Value |
|---------|-------|
| **SSID** | CaneToadDetector |
| **Password** | canetoad2024 |
| **IP Address** | 192.168.4.1 |
| **DHCP Range** | 192.168.4.2 - 192.168.4.20 |
| **Backend URL** | http://192.168.4.1:5000 |
| **Channel** | 6 (2.4GHz) |
| **Max Devices** | 4 |

---

## 🎯 Quick Reference Commands

```bash
# Start hotspot
sudo nmcli connection up CaneToadHotspot

# Stop hotspot
sudo nmcli connection down CaneToadHotspot

# View hotspot status
nmcli connection show CaneToadHotspot

# Change password
sudo nmcli connection modify CaneToadHotspot \
  802-11-wireless-security.psk "NewPassword"

# Start backend manually
cd /home/pi/raspi_deploy
source venv/bin/activate
python detector_raspi.py --headless

# View backend logs
sudo journalctl -u cane-toad-backend.service -f

# Restart everything
sudo systemctl restart NetworkManager
sudo systemctl restart cane-toad-backend.service
```

---

## 🆘 Getting Help

If you encounter issues:

1. **Check Pi Status**:
   ```bash
   ./check_status.sh
   ```

2. **Run Diagnostics**:
   ```bash
   ./diagnostics.sh
   ```

3. **Reset to Defaults**:
   ```bash
   sudo ./reset_hotspot.sh
   ```

For more help, check the main [DEPLOY.md](DEPLOY.md) documentation.

---

## ✅ Deployment Checklist

- [ ] Pi hotspot configured and broadcasting
- [ ] Pi IP is 192.168.4.1 (or known alternative)
- [ ] Backend running on port 5000
- [ ] Backend auto-starts on boot
- [ ] Mobile app settings updated with Pi IP
- [ ] Camera connected and working
- [ ] best.onnx model present
- [ ] Tested connection from phone
- [ ] Tested camera feeds in mobile app
- [ ] Changed default WiFi password

**Your system is ready for field deployment! 🎉**
