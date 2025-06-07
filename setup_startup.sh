#!/bin/bash

# VocalGem Startup Setup Script
# This script sets up the VocalGem wake word detection to run at startup

set -e

echo "Setting up VocalGem to run at startup..."

# Check if running as root for systemd operations
if [ "$EUID" -ne 0 ]; then
    echo "This script needs to be run with sudo for systemd setup"
    echo "Usage: sudo ./setup_startup.sh"
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
    echo "Please create the virtual environment first:"
    echo "  cd $PROJECT_DIR"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if wake_porcu.py exists
if [ ! -f "$PROJECT_DIR/wake_porcu.py" ]; then
    echo "Error: wake_porcu.py not found in $PROJECT_DIR"
    exit 1
fi

# Copy service file to systemd directory
echo "Installing systemd service..."
cp "$PROJECT_DIR/vocalgem.service" /etc/systemd/system/

# Update the service file with correct paths and user
sed -i "s|User=yannis.achour|User=$ACTUAL_USER|g" /etc/systemd/system/vocalgem.service
sed -i "s|Group=yannis.achour|Group=$ACTUAL_USER|g" /etc/systemd/system/vocalgem.service
sed -i "s|WorkingDirectory=/home/yannis.achour/dev2/vocalgem|WorkingDirectory=$PROJECT_DIR|g" /etc/systemd/system/vocalgem.service
sed -i "s|Environment=PATH=/home/yannis.achour/dev2/vocalgem/venv/bin|Environment=PATH=$PROJECT_DIR/venv/bin|g" /etc/systemd/system/vocalgem.service
sed -i "s|ExecStart=/home/yannis.achour/dev2/vocalgem/venv/bin/python /home/yannis.achour/dev2/vocalgem/wake_porcu.py|ExecStart=$PROJECT_DIR/venv/bin/python $PROJECT_DIR/wake_porcu.py|g" /etc/systemd/system/vocalgem.service

# Set proper permissions
chmod 644 /etc/systemd/system/vocalgem.service

# Add user to audio and plugdev groups if not already added
usermod -a -G audio,plugdev $ACTUAL_USER

# Reload systemd and enable the service
echo "Enabling VocalGem service..."
systemctl daemon-reload
systemctl enable vocalgem.service

echo ""
echo "✅ VocalGem startup service has been installed and enabled!"
echo ""
echo "Available commands:"
echo "  sudo systemctl start vocalgem     # Start the service now"
echo "  sudo systemctl stop vocalgem      # Stop the service"
echo "  sudo systemctl status vocalgem    # Check service status"
echo "  sudo systemctl restart vocalgem   # Restart the service"
echo "  sudo journalctl -u vocalgem -f    # View live logs"
echo "  sudo systemctl disable vocalgem   # Disable startup (if needed)"
echo ""
echo "The service will automatically start on next boot."
echo "To start it now, run: sudo systemctl start vocalgem" 