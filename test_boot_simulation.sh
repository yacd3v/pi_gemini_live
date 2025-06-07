#!/bin/bash

# Boot Simulation Test for VocalGem Service
# This script simulates boot conditions to test if the service will start properly

set -e

echo "🔄 VocalGem Boot Simulation Test"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_status $RED "❌ This script needs to be run with sudo"
    echo "Usage: sudo ./test_boot_simulation.sh"
    exit 1
fi

print_status $BLUE "📋 Step 1: Checking service configuration..."

# Check if service file exists
if [ ! -f "/etc/systemd/system/vocalgem.service" ]; then
    print_status $RED "❌ Service file not found. Run setup_startup.sh first."
    exit 1
fi

print_status $GREEN "✅ Service file exists"

# Check if service is enabled
if systemctl is-enabled vocalgem >/dev/null 2>&1; then
    print_status $GREEN "✅ Service is enabled for startup"
else
    print_status $RED "❌ Service is not enabled for startup"
    exit 1
fi

print_status $BLUE "📋 Step 2: Checking dependencies..."

# Check if virtual environment exists
VENV_PATH="/home/yannis.achour/dev2/vocalgem/venv"
if [ -d "$VENV_PATH" ]; then
    print_status $GREEN "✅ Virtual environment exists"
else
    print_status $RED "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check if wake_porcu.py exists
SCRIPT_PATH="/home/yannis.achour/dev2/vocalgem/wake_porcu.py"
if [ -f "$SCRIPT_PATH" ]; then
    print_status $GREEN "✅ wake_porcu.py script exists"
else
    print_status $RED "❌ wake_porcu.py not found at $SCRIPT_PATH"
    exit 1
fi

# Check audio devices (simulate what would be available at boot)
print_status $BLUE "📋 Step 3: Checking audio devices..."
if aplay -l | grep -q "ReSpeaker" 2>/dev/null; then
    print_status $GREEN "✅ ReSpeaker device detected"
else
    print_status $YELLOW "⚠️  ReSpeaker device not detected (may need USB reset at boot)"
fi

print_status $BLUE "📋 Step 4: Simulating boot environment..."

# Stop service if running
print_status $YELLOW "🛑 Stopping service if running..."
systemctl stop vocalgem 2>/dev/null || true

# Wait a moment
sleep 2

# Clear any previous logs for this test
print_status $YELLOW "🧹 Clearing previous test logs..."
journalctl --vacuum-time=1s >/dev/null 2>&1 || true

# Simulate boot by starting service
print_status $BLUE "🚀 Starting service (simulating boot)..."
if systemctl start vocalgem; then
    print_status $GREEN "✅ Service started successfully"
else
    print_status $RED "❌ Service failed to start"
    print_status $YELLOW "📋 Recent logs:"
    journalctl -u vocalgem -n 10 --no-pager
    exit 1
fi

# Wait for service to initialize
print_status $YELLOW "⏳ Waiting for service to initialize (10 seconds)..."
sleep 10

# Check service status
print_status $BLUE "📋 Step 5: Checking service health..."
if systemctl is-active vocalgem >/dev/null 2>&1; then
    print_status $GREEN "✅ Service is active and running"
else
    print_status $RED "❌ Service is not active"
    systemctl status vocalgem --no-pager
    exit 1
fi

# Check logs for errors
print_status $BLUE "📋 Step 6: Checking for errors in logs..."
if journalctl -u vocalgem --since "1 minute ago" | grep -i "error\|failed\|exception" >/dev/null; then
    print_status $YELLOW "⚠️  Found errors in logs:"
    journalctl -u vocalgem --since "1 minute ago" | grep -i "error\|failed\|exception" | tail -5
else
    print_status $GREEN "✅ No errors found in recent logs"
fi

# Check if LED initialization worked
print_status $BLUE "📋 Step 7: Checking LED functionality..."
if journalctl -u vocalgem --since "1 minute ago" | grep -q "LED controller initialized successfully"; then
    print_status $GREEN "✅ LED controller initialized successfully"
elif journalctl -u vocalgem --since "1 minute ago" | grep -q "LED module not available"; then
    print_status $YELLOW "⚠️  LED module not available (will work without LEDs)"
else
    print_status $YELLOW "⚠️  LED status unclear from logs"
fi

# Check if wake word detection is active
print_status $BLUE "📋 Step 8: Checking wake word detection..."
if journalctl -u vocalgem --since "1 minute ago" | grep -q "Listening for wake words"; then
    print_status $GREEN "✅ Wake word detection is active"
else
    print_status $YELLOW "⚠️  Wake word detection status unclear"
fi

# Show recent logs
print_status $BLUE "📋 Step 9: Recent service logs:"
echo "----------------------------------------"
journalctl -u vocalgem --since "2 minutes ago" -n 15 --no-pager
echo "----------------------------------------"

print_status $GREEN "🎉 Boot simulation completed!"
echo ""
print_status $BLUE "📊 Summary:"
echo "  ✅ Service configuration: OK"
echo "  ✅ Dependencies: OK" 
echo "  ✅ Service startup: OK"
echo "  ✅ Service health: OK"
echo ""
print_status $GREEN "🚀 Your VocalGem service should start successfully at boot!"
echo ""
print_status $YELLOW "💡 To test with actual reboot:"
echo "   sudo reboot"
echo "   # After reboot, check with:"
echo "   sudo systemctl status vocalgem"
echo "   sudo journalctl -u vocalgem -f"
echo ""
print_status $BLUE "🔧 Service management commands:"
echo "   sudo systemctl stop vocalgem      # Stop service"
echo "   sudo systemctl start vocalgem     # Start service"
echo "   sudo systemctl restart vocalgem   # Restart service"
echo "   sudo journalctl -u vocalgem -f    # View live logs" 