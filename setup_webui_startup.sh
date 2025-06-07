#!/bin/bash

# VocalGem Web UI Auto-Startup Setup Script
# This script sets up the VocalGem web UI to run automatically at startup

set -e

echo "Setting up VocalGem Web UI to auto-start at boot..."
echo "=================================================="

# Check if running as root for systemd operations
if [ "$EUID" -ne 0 ]; then
    echo "This script needs to be run with sudo for systemd setup"
    echo "Usage: sudo ./setup_webui_startup.sh"
    exit 1
fi

# Get the actual user (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(eval echo ~$ACTUAL_USER)
PROJECT_DIR="$USER_HOME/dev2/vocalgem"

echo "Project directory: $PROJECT_DIR"
echo "User: $ACTUAL_USER"

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory $PROJECT_DIR does not exist"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Error: Virtual environment not found at $PROJECT_DIR/venv"
    echo "Please run the web UI setup first:"
    echo "  cd $PROJECT_DIR"
    echo "  ./start_web_ui.sh"
    exit 1
fi

# Check if web_ui.py exists
if [ ! -f "$PROJECT_DIR/web_ui.py" ]; then
    echo "Error: web_ui.py not found in $PROJECT_DIR"
    exit 1
fi

# Install web dependencies as the actual user first
echo "Installing web UI dependencies..."
sudo -u $ACTUAL_USER bash -c "cd $PROJECT_DIR && source venv/bin/activate && pip install -r requirements_web.txt"

# Copy service file to systemd directory
echo "Installing systemd service for web UI..."
cp "$PROJECT_DIR/vocalgem-webui.service" /etc/systemd/system/

# Update the service file with correct paths and user
sed -i "s|WorkingDirectory=/home/yannis.achour/dev2/vocalgem|WorkingDirectory=$PROJECT_DIR|g" /etc/systemd/system/vocalgem-webui.service
sed -i "s|Environment=PATH=/home/yannis.achour/dev2/vocalgem/venv/bin|Environment=PATH=$PROJECT_DIR/venv/bin|g" /etc/systemd/system/vocalgem-webui.service
sed -i "s|Environment=SUDO_USER=yannis.achour|Environment=SUDO_USER=$ACTUAL_USER|g" /etc/systemd/system/vocalgem-webui.service
sed -i "s|ExecStart=/home/yannis.achour/dev2/vocalgem/venv/bin/python /home/yannis.achour/dev2/vocalgem/web_ui.py|ExecStart=$PROJECT_DIR/venv/bin/python $PROJECT_DIR/web_ui.py|g" /etc/systemd/system/vocalgem-webui.service

# Set proper permissions
chmod 644 /etc/systemd/system/vocalgem-webui.service

# Reload systemd and enable the service
echo "Enabling VocalGem Web UI service..."
systemctl daemon-reload
systemctl enable vocalgem-webui.service

# Get Pi IP for display
PI_IP=$(hostname -I | cut -d' ' -f1)

echo ""
echo "✅ VocalGem Web UI auto-startup service has been installed and enabled!"
echo ""
echo "🌐 Your Web UI will be available at:"
echo "   Local:   http://localhost:5000"
echo "   Network: http://$PI_IP:5000"
echo ""
echo "📱 Accessible from any device on your WiFi network!"
echo ""
echo "Available commands:"
echo "  sudo systemctl start vocalgem-webui     # Start the web UI now"
echo "  sudo systemctl stop vocalgem-webui      # Stop the web UI"
echo "  sudo systemctl status vocalgem-webui    # Check web UI status"
echo "  sudo systemctl restart vocalgem-webui   # Restart the web UI"
echo "  sudo journalctl -u vocalgem-webui -f    # View web UI logs"
echo "  sudo systemctl disable vocalgem-webui   # Disable auto-start (if needed)"
echo ""
echo "🎯 Both services will now start automatically on boot:"
echo "   ✅ VocalGem service (wake word detection)"
echo "   ✅ VocalGem Web UI (control panel)"
echo ""
echo "To start the web UI now, run: sudo systemctl start vocalgem-webui"
echo "Then visit: http://$PI_IP:5000" 