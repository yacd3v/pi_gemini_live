#!/bin/bash

# Quick Fix for BNO085 Installation
# =================================
# This fixes the externally-managed-environment error

echo "BNO085 Installation Fix"
echo "======================="
echo

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "This script should be run with sudo privileges"
    echo "Usage: sudo bash fix_install.sh"
    exit 1
fi

echo "Installing CircuitPython libraries with --break-system-packages flag..."
echo "This is safe for Raspberry Pi robotics applications."
echo

# Install the required packages
pip3 install --break-system-packages --upgrade pip setuptools wheel
pip3 install --break-system-packages adafruit-circuitpython-bno08x
pip3 install --break-system-packages adafruit-blinka

echo
echo "Installation completed!"
echo "======================"
echo
echo "You can now run: python3 test_IMU.py"
echo
echo "Note: The --break-system-packages flag is safe to use on Raspberry Pi"
echo "for embedded/robotics applications. It's needed due to new Python security"
echo "features in Raspberry Pi OS Bookworm." 