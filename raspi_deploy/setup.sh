#!/bin/bash

# =============================================================================
# Raspberry Pi Setup Script for Cane Toad Detector
# =============================================================================
#
# This script installs all dependencies and prepares the system.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# What this script does:
#   1. Installs system dependencies (OpenCV, etc.)
#   2. Creates Python virtual environment
#   3. Installs Python packages
#   4. Converts model to ONNX (if needed)
#   5. Verifies camera detection
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# =============================================================================
# System Dependencies
# =============================================================================

install_system_deps() {
    print_header "Installing System Dependencies"
    
    print_info "Updating package lists..."
    sudo apt update
    
    print_info "Installing required packages..."
    # Debian/Raspberry Pi OS package names changed across releases.
    # Trixie uses libgl1 and libglib2.0-0t64, while older releases use
    # libgl1-mesa-glx and libglib2.0-0.
    GL_PACKAGE="libgl1-mesa-glx"
    GLIB_PACKAGE="libglib2.0-0"

    if apt-cache show libgl1 >/dev/null 2>&1; then
        GL_PACKAGE="libgl1"
    fi

    if apt-cache show libglib2.0-0t64 >/dev/null 2>&1; then
        GLIB_PACKAGE="libglib2.0-0t64"
    fi

    sudo apt install -y \
        python3-pip \
        python3-venv \
        "$GL_PACKAGE" \
        "$GLIB_PACKAGE" \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        v4l-utils
    
    print_success "System dependencies installed"
}

# =============================================================================
# Python Environment
# =============================================================================

create_venv() {
    print_header "Creating Python Virtual Environment"
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Remove and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
            print_info "Removed old environment"
        else
            print_info "Using existing environment"
            return
        fi
    fi
    
    print_info "Creating virtual environment..."
    python3 -m venv venv
    
    print_success "Virtual environment created"
}

install_python_deps() {
    print_header "Installing Python Packages"
    
    source venv/bin/activate
    
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    print_info "Installing packages from requirements.txt..."
    pip install -r requirements.txt
    
    print_success "Python packages installed"
}

# =============================================================================
# Model Conversion
# =============================================================================

convert_model() {
    print_header "Model Conversion"
    
    if [ -f "best.onnx" ]; then
        print_success "ONNX model already exists: best.onnx"
        return
    fi
    
    if [ ! -f "best.pt" ]; then
        print_warning "No model file found (best.pt or best.onnx)"
        print_info "Please copy your model file:"
        print_info "  scp best.pt pi@raspberrypi.local:~/raspi_deploy/"
        return
    fi
    
    print_info "Found best.pt - converting to ONNX..."
    print_warning "This requires PyTorch and may take 5-10 minutes..."
    
    source venv/bin/activate
    
    # Install ultralytics for conversion
    print_info "Installing ultralytics (temporary)..."
    pip install ultralytics
    
    # Convert
    python convert_to_onnx.py
    
    if [ -f "best.onnx" ]; then
        print_success "Model converted successfully"
        
        # Optionally remove PyTorch to save space
        print_info ""
        print_info "PyTorch is large (~500MB). Remove it to save space?"
        print_info "(ONNX Runtime will still work)"
        read -p "Uninstall PyTorch? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip uninstall -y torch torchvision torchaudio ultralytics
            print_success "PyTorch uninstalled"
        fi
    else
        print_error "Model conversion failed"
    fi
}

# =============================================================================
# Camera Check
# =============================================================================

check_camera() {
    print_header "Camera Check"
    
    if [ -e /dev/video0 ]; then
        print_success "Camera detected: /dev/video0"
        v4l2-ctl --list-devices 2>/dev/null || true
    else
        print_warning "Camera not detected at /dev/video0"
        print_info "Enable camera with: sudo raspi-config"
        print_info "Interface Options → Camera → Enable"
    fi
}

# =============================================================================
# Summary
# =============================================================================

print_summary() {
    print_header "Setup Complete!"
    
    echo ""
    echo -e "${GREEN}✓ System dependencies installed${NC}"
    echo -e "${GREEN}✓ Python virtual environment created${NC}"
    echo -e "${GREEN}✓ Python packages installed${NC}"
    
    if [ -f "best.onnx" ]; then
        echo -e "${GREEN}✓ ONNX model ready${NC}"
    else
        echo -e "${YELLOW}⚠ ONNX model not found${NC}"
    fi
    
    if [ -e /dev/video0 ]; then
        echo -e "${GREEN}✓ Camera detected${NC}"
    else
        echo -e "${YELLOW}⚠ Camera not detected${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo ""
    echo "1. Test the detector:"
    echo "   source venv/bin/activate"
    echo "   python detector_raspi.py --headless --skip 2"
    echo ""
    echo "2. Set up WiFi hotspot (for mobile app):"
    echo "   chmod +x setup_hotspot.sh"
    echo "   sudo ./setup_hotspot.sh"
    echo ""
    echo "3. Check system status:"
    echo "   ./diagnostics.sh"
    echo ""
    echo -e "${GREEN}Setup successful!${NC}"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    print_header "Cane Toad Detector - Raspberry Pi Setup"
    echo ""
    
    install_system_deps
    create_venv
    install_python_deps
    convert_model
    check_camera
    print_summary
}

# Run
main
