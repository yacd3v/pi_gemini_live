#!/bin/bash

# VocalGem Web UI Startup Script
# This script starts the web interface for managing VocalGem service

set -e

PROJECT_DIR="/home/yannis.achour/dev2/vocalgem"

echo "🚀 Starting VocalGem Web UI..."
echo "================================"

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

# Install web UI dependencies
echo "📦 Installing web UI dependencies..."
source venv/bin/activate
pip install -r requirements_web.txt

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
echo "⚠️  Note: Some commands require sudo privileges."
echo "   If commands fail, you can run with: sudo python3 web_ui.py"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "🔐 Running with sudo privileges - all commands available"
    echo ""
fi

echo "Press Ctrl+C to stop the web server"
echo ""

# Start the web UI
cd "$PROJECT_DIR"
source venv/bin/activate
python3 web_ui.py 