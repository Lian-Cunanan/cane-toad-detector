#!/bin/bash

# =============================================================================
# Raspberry Pi WiFi Hotspot Setup Script
# =============================================================================
#
# Automatically configures your Raspberry Pi as a WiFi hotspot for the
# Cane Toad Detector system.
#
# Usage:
#   sudo ./setup_hotspot.sh
#
# What this script does:
#   1. Installs required packages (NetworkManager, dnsmasq)
#   2. Configures WiFi hotspot with SSID: CaneToadDetector
#   3. Sets static IP: 192.168.4.1
#   4. Creates systemd service for backend auto-start
#   5. Enables services on boot
#
# After reboot, your Pi will:
#   - Broadcast WiFi: CaneToadDetector (password: canetoad2024)
#   - Run backend on http://192.168.4.1:5000
#   - Start camera detection automatically
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
SSID="CaneToadDetector"
PASSWORD="canetoad2024"
IP_ADDRESS="192.168.4.1"
BACKEND_PORT="5000"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then 
        print_error "Please run as root (use sudo)"
        exit 1
    fi
}

check_wifi() {
    if ! command -v iwconfig &> /dev/null; then
        print_warning "WiFi tools not found, installing..."
        apt install -y wireless-tools
    fi
    
    if ! iwconfig 2>&1 | grep -q "wlan0"; then
        print_error "WiFi interface wlan0 not found!"
        print_info "This script requires built-in WiFi (Pi 3/4/Zero W)"
        exit 1
    fi
    print_success "WiFi interface detected"
}

# =============================================================================
# Installation Steps
# =============================================================================

install_packages() {
    print_header "Installing Required Packages"
    
    apt update || true
    apt install -y network-manager dnsmasq hostapd rfkill
    
    print_success "Packages installed"
}

configure_hotspot() {
    print_header "Configuring WiFi Hotspot"
    
    # Unblock WiFi
    rfkill unblock wifi
    print_success "WiFi unblocked"
    
    # Stop existing connection if any
    nmcli connection down CaneToadHotspot 2>/dev/null || true
    nmcli connection delete CaneToadHotspot 2>/dev/null || true
    
    # Create hotspot
    print_info "Creating hotspot..."
    print_info "  SSID: $SSID"
    print_info "  Password: $PASSWORD"
    print_info "  IP: $IP_ADDRESS"
    
    nmcli device wifi hotspot \
        ifname wlan0 \
        con-name CaneToadHotspot \
        ssid "$SSID" \
        band bg \
        channel 6 \
        password "$PASSWORD"
    
    print_success "Hotspot created"
    
    # Configure static IP
    nmcli connection modify CaneToadHotspot \
        ipv4.addresses "$IP_ADDRESS/24" \
        ipv4.method shared
    
    print_success "IP address configured"
    
    # Enable auto-connect on boot
    nmcli connection modify CaneToadHotspot \
        connection.autoconnect yes
    
    print_success "Auto-connect enabled"
    
    # Bring up connection
    nmcli connection up CaneToadHotspot
    
    print_success "Hotspot activated"
}

create_backend_service() {
    print_header "Creating Backend Service"
    
    # Get current user
    CURRENT_USER=${SUDO_USER:-$USER}
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    
    # Create systemd service file
    cat > /etc/systemd/system/cane-toad-backend.service <<EOF
[Unit]
Description=Cane Toad Detector Backend
After=network.target multi-user.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$SCRIPT_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$SCRIPT_DIR/venv/bin/python $SCRIPT_DIR/detector_raspi.py --headless --skip 2
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    print_success "Service file created"
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable cane-toad-backend.service
    print_success "Service enabled (will start on boot)"
}

test_setup() {
    print_header "Testing Setup"
    
    # Check hotspot status
    if nmcli connection show --active | grep -q "CaneToadHotspot"; then
        print_success "Hotspot is active"
    else
        print_error "Hotspot is not active"
    fi
    
    # Check IP address
    if ip addr show wlan0 | grep -q "$IP_ADDRESS"; then
        print_success "IP address configured correctly"
    else
        print_warning "IP address may not be set correctly"
    fi
    
    # Check service
    if systemctl is-enabled cane-toad-backend.service &> /dev/null; then
        print_success "Backend service is enabled"
    else
        print_warning "Backend service is not enabled"
    fi
}

print_summary() {
    print_header "Setup Complete!"
    
    echo ""
    echo -e "${GREEN}Your Raspberry Pi is now configured as a WiFi hotspot!${NC}"
    echo ""
    echo -e "${BLUE}Hotspot Details:${NC}"
    echo "  • SSID: $SSID"
    echo "  • Password: $PASSWORD"
    echo "  • IP Address: $IP_ADDRESS"
    echo "  • Backend URL: http://$IP_ADDRESS:$BACKEND_PORT"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "  1. Reboot your Pi: sudo reboot"
    echo "  2. Connect your mobile device to WiFi: $SSID"
    echo "  3. Open the mobile app"
    echo "  4. Go to Settings → Enter Backend URL: http://$IP_ADDRESS:$BACKEND_PORT"
    echo "  5. Login or register"
    echo ""
    echo -e "${YELLOW}Security Warning:${NC}"
    echo "  The default password is 'canetoad2024'"
    echo "  To change it, run:"
    echo "    sudo nmcli connection modify CaneToadHotspot \\"
    echo "      802-11-wireless-security.psk \"YourNewPassword\""
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  • Check hotspot status: nmcli connection show --active"
    echo "  • Check backend status: sudo systemctl status cane-toad-backend"
    echo "  • View backend logs: sudo journalctl -u cane-toad-backend -f"
    echo "  • Restart backend: sudo systemctl restart cane-toad-backend"
    echo ""
    echo -e "${GREEN}Setup completed successfully!${NC}"
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    print_header "Raspberry Pi Hotspot Setup"
    print_info "Starting automated setup..."
    echo ""
    
    # Pre-checks
    check_root
    check_wifi
    
    # Installation
    install_packages
    configure_hotspot
    create_backend_service
    
    # Verification
    test_setup
    
    # Summary
    print_summary
    
    # Prompt for reboot
    echo -e "${YELLOW}Do you want to reboot now? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Rebooting..."
        reboot
    else
        print_info "Please reboot manually when ready: sudo reboot"
    fi
}

# Run main function
main
