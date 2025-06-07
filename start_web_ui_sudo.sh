#!/bin/bash

# VocalGem Web UI Startup Script (with sudo support)
# This script starts the web interface with proper virtual environment and sudo privileges

set -e

PROJECT_DIR="/home/yannis.achour/dev2/vocalgem"

echo "🚀 Starting VocalGem Web UI with sudo privileges..."
echo "================================================="

# Check if we're in the right directory
if [ ! -f "web_ui.py" ]; then
    echo "Error: web_ui.py not found. Please run this script from the VocalGem project directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Setting up..."
    ./setup_venv.sh
fi

# Install web UI dependencies in venv (as regular user first)
if [ "$EUID" -eq 0 ]; then
    # We're running as root, need to install as the actual user
    ACTUAL_USER=${SUDO_USER:-yannis.achour}
    echo "📦 Installing web UI dependencies as user $ACTUAL_USER..."
    sudo -u $ACTUAL_USER bash -c "cd $PROJECT_DIR && source venv/bin/activate && pip install -r requirements_web.txt"
else
    echo "📦 Installing web UI dependencies..."
    source venv/bin/activate
    pip install -r requirements_web.txt
fi

# Get the Raspberry Pi's IP address
PI_IP=$(hostname -I | cut -d' ' -f1)

echo ""
echo "🌐 VocalGem Web UI Information:"
echo "==============================="
echo "Local access:    http://localhost:5000"
echo "Network access:  http://$PI_IP:5000"
echo ""
echo "📱 Access from any device on your WiFi network!"
echo ""
echo "🔐 Running with sudo privileges - all commands available"
echo ""
echo "Press Ctrl+C to stop the web server"
echo ""

# Start the web UI with proper virtual environment
cd "$PROJECT_DIR"

if [ "$EUID" -eq 0 ]; then
    # Running as root via sudo
    ACTUAL_USER=${SUDO_USER:-yannis.achour}
    VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
    
    # Ensure the virtual environment python exists and is executable
    if [ ! -f "$VENV_PYTHON" ]; then
        echo "❌ Virtual environment Python not found at $VENV_PYTHON"
        exit 1
    fi
    
    # Run with the virtual environment python but keep sudo privileges
    exec "$VENV_PYTHON" web_ui.py
else
    # Running as regular user
    source venv/bin/activate
    python3 web_ui.py
fi 