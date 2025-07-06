#!/bin/bash

# BNO085 Dependencies Installation Script
# ======================================
# This script installs all required dependencies for the BNO085 test script

echo "BNO085 Dependencies Installation Script"
echo "======================================"
echo

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "This script should be run with sudo privileges"
    echo "Usage: sudo bash install_dependencies.sh"
    exit 1
fi

# Update package list
echo "Updating package list..."
apt-get update -y

# Install system dependencies
echo "Installing system dependencies..."
apt-get install -y python3-pip python3-dev python3-venv i2c-tools

# Install Python dependencies
echo "Installing Python dependencies..."

# First try to install via apt (Debian packages)
echo "Trying to install via apt packages first..."
apt-get install -y python3-pip python3-setuptools python3-wheel

# For Raspberry Pi OS Bookworm, we need to handle externally-managed-environment
echo "Installing CircuitPython libraries..."

# Check if we're in a virtual environment first
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual environment detected, installing normally..."
    pip3 install --upgrade pip setuptools wheel
    pip3 install adafruit-circuitpython-bno08x
    pip3 install adafruit-blinka
else
    echo "System-wide installation needed..."
    echo "Note: Using --break-system-packages for CircuitPython libraries"
    echo "This is safe for embedded/robotics applications on Raspberry Pi"
    
    # Install CircuitPython libraries with --break-system-packages
    pip3 install --break-system-packages --upgrade pip setuptools wheel
    pip3 install --break-system-packages adafruit-circuitpython-bno08x
    pip3 install --break-system-packages adafruit-blinka
fi

# Enable I2C if not already enabled
echo "Configuring I2C..."
if ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt; then
    echo "dtparam=i2c_arm=on" >> /boot/firmware/config.txt
    echo "I2C enabled in config.txt"
else
    echo "I2C already enabled"
fi

# Check if i2c-dev is in modules
if ! grep -q "^i2c-dev" /etc/modules; then
    echo "i2c-dev" >> /etc/modules
    echo "i2c-dev added to modules"
else
    echo "i2c-dev already in modules"
fi

# Make test script executable
if [ -f "test_IMU.py" ]; then
    chmod +x test_IMU.py
    echo "Made test_IMU.py executable"
fi

echo
echo "Installation completed!"
echo "======================"
echo
echo "Next steps:"
echo "1. Reboot your Raspberry Pi: sudo reboot"
echo "2. After reboot, connect your BNO085 sensor:"
echo "   - VIN -> 3.3V or 5V"
echo "   - GND -> GND"
echo "   - SCL -> GPIO 3 (Pin 5)"
echo "   - SDA -> GPIO 2 (Pin 3)"
echo "3. Test I2C connection: sudo i2cdetect -y 1"
echo "4. Run the test script: python3 test_IMU.py"
echo
echo "If you encounter any issues, check the troubleshooting section in the script." 