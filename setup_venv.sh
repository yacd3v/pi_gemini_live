#!/bin/bash

# VocalGem Virtual Environment Setup Script
# This script creates and configures the virtual environment for VocalGem

set -e

PROJECT_DIR="/home/yannis.achour/dev2/vocalgem"

echo "Setting up VocalGem virtual environment..."

# Check if we're in the right directory
if [ ! -f "wake_porcu.py" ]; then
    echo "Error: wake_porcu.py not found. Please run this script from the VocalGem project directory."
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 first:"
    echo "  sudo apt update"
    echo "  sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Install system dependencies for audio and other requirements
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install additional dependencies that might be needed
echo "Installing additional dependencies..."
pip install \
    pvporcupine \
    python-dotenv \
    gpiozero \
    RPi.GPIO

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment manually:"
echo "  source venv/bin/activate"
echo ""
echo "To test the setup:"
echo "  source venv/bin/activate"
echo "  python wake_porcu.py"
echo ""
echo "Next step: Run the startup setup script:"
echo "  sudo ./setup_startup.sh" 