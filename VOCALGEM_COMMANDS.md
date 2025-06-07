# VocalGem Service Commands Reference

Quick reference guide for managing your VocalGem wake word detection service on Raspberry Pi 5.

## 🚀 Service Management Commands

### **Basic Service Control**
```bash
# Start the service
sudo systemctl start vocalgem

# Stop the service
sudo systemctl stop vocalgem

# Restart the service
sudo systemctl restart vocalgem

# Check service status
sudo systemctl status vocalgem

# Enable service to start at boot (already done)
sudo systemctl enable vocalgem

# Disable service from starting at boot
sudo systemctl disable vocalgem
```

### **Service Information**
```bash
# Check if service is enabled for startup
sudo systemctl is-enabled vocalgem

# Check if service is currently active
sudo systemctl is-active vocalgem

# List all services (find vocalgem)
sudo systemctl list-unit-files | grep vocalgem
```

## 📋 Log Management Commands

### **View Logs**
```bash
# View live logs (real-time)
sudo journalctl -u vocalgem -f

# View last 50 log entries
sudo journalctl -u vocalgem -n 50

# View logs from last 10 minutes
sudo journalctl -u vocalgem --since "10 minutes ago"

# View logs from today
sudo journalctl -u vocalgem --since today

# View logs without pager (all at once)
sudo journalctl -u vocalgem --no-pager

# Search for specific terms in logs
sudo journalctl -u vocalgem | grep "ERROR"
sudo journalctl -u vocalgem | grep "LED"
sudo journalctl -u vocalgem | grep "wake word"
```

### **Log Cleanup**
```bash
# Clear old logs (keep last 1 day)
sudo journalctl --vacuum-time=1d

# Clear old logs (keep last 100MB)
sudo journalctl --vacuum-size=100M
```

## 🧪 Testing Commands

### **Service Testing**
```bash
# Quick setup check
./check_service_setup.sh

# Comprehensive boot simulation test
sudo ./test_boot_simulation.sh

# Test LED functionality
python3 test_wake_led.py

# Test wake word detection manually (in venv)
source venv/bin/activate
python3 wake_porcu.py
```

### **Audio Testing**
```bash
# List audio devices
aplay -l
arecord -l

# Test ReSpeaker recording
arecord -D plughw:1,0 -c 6 -r 16000 -f S16_LE test.wav

# Test audio playback
aplay test.wav
```

## 🔧 Troubleshooting Commands

### **Service Issues**
```bash
# If service fails to start
sudo journalctl -u vocalgem -n 50  # Check logs
sudo systemctl daemon-reload       # Reload service files
sudo ./setup_startup.sh           # Reinstall service

# If service keeps restarting
sudo systemctl stop vocalgem       # Stop it
sudo journalctl -u vocalgem -n 20  # Check what's wrong
# Fix the issue, then:
sudo systemctl start vocalgem      # Start again
```

### **Audio Device Issues**
```bash
# Check if ReSpeaker is detected
lsusb | grep 2886

# Reset USB audio devices
sudo modprobe -r snd_usb_audio
sudo modprobe snd_usb_audio

# Check audio groups
groups $USER  # Should include 'audio'

# Add user to audio group (if missing)
sudo usermod -a -G audio,plugdev $USER
```

### **Virtual Environment Issues**
```bash
# Recreate virtual environment
./setup_venv.sh

# Check if venv exists
ls -la venv/

# Test venv manually
source venv/bin/activate
python3 -c "import pvporcupine; print('Porcupine OK')"
python3 -c "import pyaudio; print('PyAudio OK')"
```

### **GPIO/LED Issues**
```bash
# Clean up GPIO resources
sudo python3 -c "import RPi.GPIO as GPIO; GPIO.cleanup()"

# Check if LED hardware is detected
python3 test_wake_led.py

# Manual GPIO cleanup (if needed)
sudo pkill -f python  # Kill any Python processes using GPIO
```

## 🔄 Reboot Testing

### **Before Reboot**
```bash
# Ensure service is enabled
sudo systemctl is-enabled vocalgem

# Run boot simulation
sudo ./test_boot_simulation.sh
```

### **Reboot and Check**
```bash
# Reboot the system
sudo reboot

# After reboot, check immediately:
sudo systemctl status vocalgem
sudo journalctl -u vocalgem -f
```

## 📁 File Locations

### **Important Files**
```bash
# Service configuration
/etc/systemd/system/vocalgem.service

# Main script
/home/yannis.achour/dev2/vocalgem/wake_porcu.py

# Virtual environment
/home/yannis.achour/dev2/vocalgem/venv/

# Setup scripts
/home/yannis.achour/dev2/vocalgem/setup_startup.sh
/home/yannis.achour/dev2/vocalgem/setup_venv.sh
```

### **Log Files**
```bash
# System logs (journalctl)
sudo journalctl -u vocalgem

# Audio debug file (created by script)
/home/yannis.achour/dev2/vocalgem/outputtest.wav
```

## 🎯 Quick Diagnostics

### **Is Everything Working?**
```bash
# One-liner health check
sudo systemctl is-active vocalgem && echo "✅ Service running" || echo "❌ Service not running"

# Check if listening for wake words
sudo journalctl -u vocalgem --since "1 minute ago" | grep -q "Listening for wake words" && echo "✅ Listening" || echo "❌ Not listening"

# Check LED status
sudo journalctl -u vocalgem --since "5 minutes ago" | grep -i led | tail -1
```

### **Common Status Checks**
```bash
# Full system check
echo "=== VocalGem Status ==="
echo "Service: $(sudo systemctl is-active vocalgem)"
echo "Enabled: $(sudo systemctl is-enabled vocalgem)"
echo "Audio: $(aplay -l | grep -q ReSpeaker && echo "✅ ReSpeaker detected" || echo "❌ ReSpeaker not found")"
echo "Venv: $([ -d venv ] && echo "✅ Virtual env exists" || echo "❌ Virtual env missing")"
```

## 🆘 Emergency Commands

### **If Service Won't Stop**
```bash
# Force kill the process
sudo pkill -f wake_porcu.py

# Force stop service
sudo systemctl kill vocalgem
sudo systemctl stop vocalgem
```

### **Complete Reset**
```bash
# Stop everything
sudo systemctl stop vocalgem
sudo pkill -f wake_porcu.py

# Clean up GPIO
sudo python3 -c "import RPi.GPIO as GPIO; GPIO.cleanup()" 2>/dev/null || true

# Reinstall service
sudo ./setup_startup.sh

# Start fresh
sudo systemctl start vocalgem
```

## 💡 Useful Aliases

Add these to your `~/.bashrc` for quick access:

```bash
# VocalGem aliases
alias vg-status='sudo systemctl status vocalgem'
alias vg-logs='sudo journalctl -u vocalgem -f'
alias vg-restart='sudo systemctl restart vocalgem'
alias vg-stop='sudo systemctl stop vocalgem'
alias vg-start='sudo systemctl start vocalgem'
alias vg-test='sudo ./test_boot_simulation.sh'
```

After adding, run: `source ~/.bashrc`

## 🎵 Wake Word Testing

### **Test Wake Word Detection**
1. Make sure service is running: `sudo systemctl status vocalgem`
2. Watch logs: `sudo journalctl -u vocalgem -f`
3. Say the wake word: **"salut karl"**
4. Look for: `WAKE WORD: Wake word detected!` in logs
5. LEDs should: Blue → Green flash → Off (during Gemini) → Blue

---

**💡 Tip**: Bookmark this file for quick reference! All commands are tested and ready to use. 