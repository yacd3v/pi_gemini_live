# VocalGem Startup Configuration

This guide will help you set up your VocalGem wake word detection script to run automatically at Raspberry Pi 5 startup.

## Prerequisites

- Raspberry Pi 5 with Raspberry Pi OS
- ReSpeaker USB microphone array
- Python 3 installed
- Your VocalGem project in `/home/yannis.achour/dev2/vocalgem`

## Quick Setup

### Step 1: Set up the virtual environment (if not already done)

```bash
cd /home/yannis.achour/dev2/vocalgem
chmod +x setup_venv.sh
./setup_venv.sh
```

### Step 2: Install the startup service

```bash
chmod +x setup_startup.sh
sudo ./setup_startup.sh
```

### Step 3: Start the service

```bash
sudo systemctl start vocalgem
```

## Manual Setup (Alternative)

If you prefer to set things up manually:

### 1. Create Virtual Environment

```bash
cd /home/yannis.achour/dev2/vocalgem
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pvporcupine python-dotenv gpiozero RPi.GPIO
```

### 2. Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3-dev portaudio19-dev libasound2-dev alsa-utils
```

### 3. Copy Service File

```bash
sudo cp vocalgem.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable vocalgem
```

## Service Management

### Start the service
```bash
sudo systemctl start vocalgem
```

### Stop the service
```bash
sudo systemctl stop vocalgem
```

### Check service status
```bash
sudo systemctl status vocalgem
```

### View live logs
```bash
sudo journalctl -u vocalgem -f
```

### Restart the service
```bash
sudo systemctl restart vocalgem
```

### Disable startup (if needed)
```bash
sudo systemctl disable vocalgem
```

## Troubleshooting

### Check if service is running
```bash
sudo systemctl status vocalgem
```

### View recent logs
```bash
sudo journalctl -u vocalgem -n 50
```

### Check audio devices
```bash
aplay -l
arecord -l
```

### Test ReSpeaker manually
```bash
# Test recording
arecord -D plughw:1,0 -c 6 -r 16000 -f S16_LE test.wav
```

### Common Issues

1. **Audio device not found**: Make sure ReSpeaker is connected and recognized
2. **Permission errors**: Ensure user is in `audio` and `plugdev` groups
3. **Virtual environment issues**: Recreate venv with `setup_venv.sh`
4. **GPIO errors**: The script includes GPIO cleanup, but you may need to reboot

### Manual GPIO cleanup (if needed)
```bash
sudo python3 -c "import RPi.GPIO as GPIO; GPIO.cleanup()"
```

## File Structure

- `vocalgem.service` - Systemd service configuration
- `setup_startup.sh` - Automated setup script
- `setup_venv.sh` - Virtual environment setup script
- `wake_porcu.py` - Main application script

## Service Configuration

The service is configured to:
- Start after network and sound systems are ready
- Run as your user account
- Restart automatically if it crashes
- Log to system journal
- Use the virtual environment
- Have proper audio device permissions

## Environment Variables

The service sets these environment variables:
- `ALSA_PCM_CARD=0`
- `ALSA_PCM_DEVICE=0`
- `PULSE_RUNTIME_PATH=/run/user/1000/pulse`

## Next Steps

After setup, your VocalGem service will:
1. Start automatically on boot
2. Listen for wake words
3. Launch Gemini when detected
4. Return to wake word listening after Gemini sessions
5. Restart automatically if it encounters errors

The service includes robust error handling and USB device reset functionality as implemented in your original script. 