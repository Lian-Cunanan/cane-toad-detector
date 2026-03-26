#!/bin/bash

# =============================================================================
# Cane Toad Detector - Diagnostics Script
# =============================================================================
#
# Checks system status and helps troubleshoot common issues
#
# Usage:
#   ./diagnostics.sh
# =============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_check() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# =============================================================================
# System Checks
# =============================================================================

print_header "System Diagnostics"
echo ""

# Check WiFi interface
print_header "1. WiFi Interface"
if iwconfig 2>&1 | grep -q "wlan0"; then
    print_check 0 "WiFi interface wlan0 exists"
    iwconfig wlan0 2>&1 | grep -E "ESSID|Mode|Frequency"
else
    print_check 1 "WiFi interface wlan0 not found"
fi
echo ""

# Check hotspot connection
print_header "2. Hotspot Status"
if nmcli connection show --active | grep -q "CaneToadHotspot"; then
    print_check 0 "Hotspot is active"
    nmcli connection show CaneToadHotspot | grep -E "connection.id|ipv4.addresses|802-11-wireless.ssid"
else
    print_check 1 "Hotspot is not active"
    print_info "To start: sudo nmcli connection up CaneToadHotspot"
fi
echo ""

# Check IP address
print_header "3. Network Configuration"
IP=$(ip addr show wlan0 2>/dev/null | grep "inet " | awk '{print $2}' | cut -d/ -f1)
if [ -n "$IP" ]; then
    print_check 0 "wlan0 has IP address: $IP"
else
    print_check 1 "wlan0 has no IP address"
fi
echo ""

# Check backend service
print_header "4. Backend Service"
if systemctl is-active --quiet cane-toad-backend.service; then
    print_check 0 "Backend service is running"
else
    print_check 1 "Backend service is not running"
    print_info "To start: sudo systemctl start cane-toad-backend"
fi

if systemctl is-enabled --quiet cane-toad-backend.service; then
    print_check 0 "Backend service is enabled (starts on boot)"
else
    print_check 1 "Backend service is not enabled"
    print_info "To enable: sudo systemctl enable cane-toad-backend"
fi
echo ""

# Check backend process
print_header "5. Backend Process"
if pgrep -f "detector_raspi.py" > /dev/null; then
    print_check 0 "detector_raspi.py is running"
    ps aux | grep detector_raspi.py | grep -v grep
else
    print_check 1 "detector_raspi.py is not running"
fi
echo ""

# Check camera
print_header "6. Camera"
if [ -e /dev/video0 ]; then
    print_check 0 "Camera device /dev/video0 exists"
    v4l2-ctl --list-devices 2>/dev/null || print_info "Install v4l-utils for camera info"
else
    print_check 1 "Camera device /dev/video0 not found"
    print_info "Check camera connection"
fi
echo ""

# Check ONNX model
print_header "7. ONNX Model"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/best.onnx" ]; then
    print_check 0 "best.onnx model exists"
    ls -lh "$SCRIPT_DIR/best.onnx"
else
    print_check 1 "best.onnx not found"
    if [ -f "$SCRIPT_DIR/best.pt" ]; then
        print_info "best.pt exists - run: python convert_to_onnx.py"
    else
        print_info "No model file found - please copy best.pt or best.onnx"
    fi
fi
echo ""

# Check Python environment
print_header "8. Python Environment"
if [ -d "$SCRIPT_DIR/venv" ]; then
    print_check 0 "Virtual environment exists"
    "$SCRIPT_DIR/venv/bin/python" --version
    
    # Check key packages
    if "$SCRIPT_DIR/venv/bin/python" -c "import onnxruntime" 2>/dev/null; then
        print_check 0 "onnxruntime installed"
    else
        print_check 1 "onnxruntime not installed"
    fi
    
    if "$SCRIPT_DIR/venv/bin/python" -c "import flask" 2>/dev/null; then
        print_check 0 "flask installed"
    else
        print_check 1 "flask not installed"
    fi
    
    if "$SCRIPT_DIR/venv/bin/python" -c "import cv2" 2>/dev/null; then
        print_check 0 "opencv installed"
    else
        print_check 1 "opencv not installed"
    fi
else
    print_check 1 "Virtual environment not found"
    print_info "Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
fi
echo ""

# Check port 5000
print_header "9. Backend Port"
if netstat -tuln 2>/dev/null | grep -q ":5000"; then
    print_check 0 "Port 5000 is listening"
    netstat -tuln | grep ":5000"
else
    print_check 1 "Port 5000 is not listening"
    print_info "Backend may not be running"
fi
echo ""

# Check connected devices (if hotspot is active)
print_header "10. Connected Devices"
if nmcli connection show --active | grep -q "CaneToadHotspot"; then
    CONNECTED=$(arp -a 2>/dev/null | grep -c "192.168.4." || echo "0")
    print_info "$CONNECTED devices connected"
    arp -a 2>/dev/null | grep "192.168.4." || echo "(none)"
else
    print_info "Hotspot not active - no devices connected"
fi
echo ""

# Test backend API
print_header "11. Backend API Test"
if curl -s --max-time 3 http://localhost:5000/api/auth/register > /dev/null 2>&1; then
    print_check 0 "Backend API responding on localhost"
else
    print_check 1 "Backend API not responding"
fi

if [ -n "$IP" ]; then
    if curl -s --max-time 3 http://$IP:5000/api/auth/register > /dev/null 2>&1; then
        print_check 0 "Backend API responding on $IP"
    else
        print_check 1 "Backend API not responding on $IP"
    fi
fi
echo ""

# =============================================================================
# Summary and Recommendations
# =============================================================================

print_header "Diagnostic Summary"
echo ""

# Count issues
ISSUES=0

if ! nmcli connection show --active | grep -q "CaneToadHotspot"; then
    ((ISSUES++))
    echo -e "${YELLOW}⚠${NC} Hotspot not active"
    echo "   Fix: sudo nmcli connection up CaneToadHotspot"
    echo ""
fi

if ! systemctl is-active --quiet cane-toad-backend.service; then
    ((ISSUES++))
    echo -e "${YELLOW}⚠${NC} Backend service not running"
    echo "   Fix: sudo systemctl start cane-toad-backend"
    echo "   Logs: sudo journalctl -u cane-toad-backend -f"
    echo ""
fi

if [ ! -f "$SCRIPT_DIR/best.onnx" ]; then
    ((ISSUES++))
    echo -e "${YELLOW}⚠${NC} ONNX model missing"
    echo "   Fix: python convert_to_onnx.py (if you have best.pt)"
    echo ""
fi

if [ ! -e /dev/video0 ]; then
    ((ISSUES++))
    echo -e "${YELLOW}⚠${NC} Camera not detected"
    echo "   Fix: Check camera connection and cable"
    echo "   Enable: sudo raspi-config → Interface Options → Camera"
    echo ""
fi

if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✓ No issues detected!${NC}"
    echo ""
    echo "Your system appears to be configured correctly."
    echo ""
    if [ -n "$IP" ]; then
        echo -e "${BLUE}Mobile App Settings:${NC}"
        echo "  Backend URL: http://$IP:5000"
    fi
else
    echo -e "${YELLOW}Found $ISSUES issue(s) that need attention.${NC}"
    echo "Please review the recommendations above."
fi

echo ""
print_header "Useful Commands"
echo ""
echo "View backend logs:"
echo "  sudo journalctl -u cane-toad-backend -f"
echo ""
echo "Restart backend:"
echo "  sudo systemctl restart cane-toad-backend"
echo ""
echo "Restart hotspot:"
echo "  sudo nmcli connection down CaneToadHotspot"
echo "  sudo nmcli connection up CaneToadHotspot"
echo ""
echo "Test backend manually:"
echo "  cd $(pwd)"
echo "  source venv/bin/activate"
echo "  python detector_raspi.py --headless"
echo ""
