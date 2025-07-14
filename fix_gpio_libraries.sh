#!/bin/bash

# GPIO Libraries Fix Script
# =========================
# Fixes the lgpio/gpiozero compatibility issue

echo "GPIO Libraries Compatibility Fix"
echo "================================"
echo

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "This script should be run with sudo privileges"
    echo "Usage: sudo bash fix_gpio_libraries.sh"
    exit 1
fi

echo "The error 'module lgpio has no attribute SET_BIAS_DISABLE' is a"
echo "compatibility issue between lgpio and gpiozero versions."
echo

# Option 1: Update both libraries to latest compatible versions
echo "Fixing GPIO library compatibility..."

echo "Step 1: Updating lgpio library..."
pip3 install --break-system-packages --upgrade lgpio

echo "Step 2: Updating gpiozero library..."
pip3 install --break-system-packages --upgrade gpiozero

echo "Step 3: Installing RPi.GPIO as fallback..."
pip3 install --break-system-packages --upgrade RPi.GPIO

echo
echo "Library fix completed!"
echo "====================="
echo

echo "Testing which libraries are available:"
python3 -c "
try:
    import lgpio
    print('✓ lgpio available')
except ImportError:
    print('❌ lgpio not available')

try:
    import gpiozero
    print('✓ gpiozero available')
except ImportError:
    print('❌ gpiozero not available')

try:
    import RPi.GPIO
    print('✓ RPi.GPIO available')
except ImportError:
    print('❌ RPi.GPIO not available')
"

echo
echo "Now try the fixed test script:"
echo "  python3 test_display_fixed.py"
echo
echo "If you still get errors, the test_display_fixed.py script"
echo "uses RPi.GPIO directly and should work around the issue." 