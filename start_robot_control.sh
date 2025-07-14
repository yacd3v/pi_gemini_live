#!/bin/bash

# VocalGem Robot Control Service Startup Script
# This script starts the robot control service with BNO085 absolute positioning

set -e

PROJECT_DIR="/home/yannis.achour/dev2/vocalgem"

echo "🤖 Starting VocalGem Robot Control Service..."
echo "============================================="

# Check if we're in the right directory
if [ ! -f "robot_control_service.py" ]; then
    echo "Error: robot_control_service.py not found. Please run this script from the VocalGem project directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Setting up..."
    ./setup_venv.sh
fi

# Install robot control dependencies
echo "📦 Installing robot control dependencies..."
source venv/bin/activate
pip install -r requirements_robot_control.txt

# Check I2C is enabled
echo "🔍 Checking I2C configuration..."
if ! lsmod | grep -q i2c_dev; then
    echo "⚠️  I2C not enabled. Enabling I2C..."
    echo "Please ensure I2C is enabled in raspi-config:"
    echo "  sudo raspi-config"
    echo "  Interface Options > I2C > Enable"
    echo ""
    echo "Or add these lines to /boot/config.txt:"
    echo "  dtparam=i2c_arm=on"
    echo "  dtparam=i2c1=on"
    echo ""
fi

# Check if user is in i2c group
if ! groups $USER | grep -q i2c; then
    echo "⚠️  User not in i2c group. Adding user to i2c group..."
    echo "You may need to run: sudo usermod -a -G i2c $USER"
    echo "Then log out and log back in."
fi

# Get the Raspberry Pi's IP address
PI_IP=$(hostname -I | cut -d' ' -f1)

echo ""
echo "🌐 VocalGem Robot Control Service Information:"
echo "=============================================="
echo "Local access:    http://localhost:5001"
echo "Network access:  http://$PI_IP:5001"
echo ""
echo "🧭 Features:"
echo "- BNO085 absolute positioning sensor"
echo "- Precision movement control (move X meters, turn Y degrees)"
echo "- Live camera streaming"
echo "- Real-time sensor data display"
echo "- High-level movement commands"
echo "- Mobile-friendly responsive interface"
echo ""
echo "📱 Access from any device on your WiFi network!"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "🔐 Running with sudo privileges - all hardware access available"
    echo ""
fi

echo "💡 Quick Start:"
echo "1. Open http://$PI_IP:5001 in your browser"
echo "2. Click 'Initialize Robot' to start the BNO085 sensor"
echo "3. Use movement controls or enter high-level commands"
echo "4. Try: 'Move 1 meter forward' or 'Turn 90 degrees left'"
echo ""
echo "⚠️  Make sure your BNO085 is connected to I2C (detected at 0x4A)"
echo ""

echo "Press Ctrl+C to stop the service"
echo ""

# Start the robot control service
cd "$PROJECT_DIR"
source venv/bin/activate
python3 robot_control_service.py 