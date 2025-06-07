#!/bin/bash

# Quick Service Setup Check
echo "🔍 VocalGem Service Setup Check"
echo "==============================="

# Check if service file exists
if [ -f "/etc/systemd/system/vocalgem.service" ]; then
    echo "✅ Service file exists"
else
    echo "❌ Service file missing - run: sudo ./setup_startup.sh"
    exit 1
fi

# Check if service is enabled
if sudo systemctl is-enabled vocalgem >/dev/null 2>&1; then
    echo "✅ Service is enabled for startup"
else
    echo "❌ Service not enabled - run: sudo systemctl enable vocalgem"
fi

# Check current status
echo ""
echo "📊 Current service status:"
sudo systemctl status vocalgem --no-pager -l

echo ""
echo "🔧 Ready to test! Run one of these:"
echo "   sudo ./test_boot_simulation.sh  # Comprehensive test"
echo "   sudo systemctl restart vocalgem # Quick restart test"
echo "   sudo reboot                     # Full reboot test" 