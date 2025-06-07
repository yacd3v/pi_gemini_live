# VocalGem Web UI

A modern, responsive web interface for managing your VocalGem wake word detection service on Raspberry Pi 5.

## 🌟 Features

### 📊 Real-time Monitoring
- **Live service status** with visual indicators
- **Real-time log streaming** via WebSocket
- **Comprehensive diagnostics** to check system health
- **Auto-refresh** status every 30 seconds

### 🎛️ Service Management
- **Start/Stop/Restart** the VocalGem service
- **Enable/Disable** auto-start at boot
- **Check service status** and configuration
- **View service information** in real-time

### 📋 Log Management
- **Live log streaming** with real-time updates
- **Filtered logs** (errors, wake words, LED events)
- **Historical logs** (recent, today, specific timeframes)
- **Log cleanup** to free up disk space

### 🧪 Testing & Diagnostics
- **Boot simulation test** to verify startup configuration
- **LED functionality test** to check hardware
- **Audio device testing** and verification
- **Comprehensive system diagnostics**

### 🔧 Troubleshooting
- **Force kill/stop** unresponsive processes
- **GPIO cleanup** for hardware conflicts
- **USB device reset** for audio issues
- **Complete system reset** when needed

### 🎵 Audio Management
- **List audio devices** to verify hardware
- **Test recording** functionality
- **Reset USB audio** devices
- **Check user permissions** for audio access

## 🚀 Quick Start

### 1. Install and Start
```bash
# Make the startup script executable (if not already done)
chmod +x start_web_ui.sh

# Start the web UI
./start_web_ui.sh
```

### 2. Access the Interface
- **Local access:** http://localhost:5000
- **Network access:** http://[your-pi-ip]:5000
- **Find Pi IP:** `hostname -I | cut -d' ' -f1`

### 3. For Full Functionality (Recommended)
```bash
# Run with sudo for all commands to work
sudo python3 web_ui.py
```

## 🌐 Network Access

The web UI is configured to be accessible from any device on your local WiFi network:

### From Your Computer/Phone/Tablet:
1. Connect to the same WiFi as your Raspberry Pi
2. Find your Pi's IP address (shown when starting the web UI)
3. Open your browser and go to: `http://[pi-ip-address]:5000`

### Example:
If your Pi's IP is `192.168.1.100`, access via: `http://192.168.1.100:5000`

## 📱 Mobile Friendly

The interface is fully responsive and works great on:
- 📱 **Smartphones** (iOS, Android)
- 📱 **Tablets** (iPad, Android tablets)
- 💻 **Laptops/Desktops** (Windows, Mac, Linux)
- 🖥️ **Any modern web browser**

## 🎯 Usage Guide

### Dashboard Overview
- **Service Status**: Shows if VocalGem is active/inactive
- **Auto-start Status**: Shows if service starts at boot
- **Quick Actions**: Refresh status and run diagnostics

### Service Management
- **Start**: Begin the VocalGem service
- **Stop**: Stop the VocalGem service
- **Restart**: Restart the service (useful after changes)
- **Status**: Get detailed service information
- **Enable Auto-start**: Service starts automatically at boot
- **Disable Auto-start**: Manual start required

### Log Monitoring
- **Live Logs**: Real-time streaming logs (WebSocket)
- **Recent (50)**: Last 50 log entries
- **Today**: All logs from today
- **Errors**: Filter for error messages only
- **Wake Words**: Filter for wake word detections
- **LED Events**: Filter for LED status changes

### Testing Features
- **Boot Test**: Simulate boot conditions to verify startup
- **LED Test**: Check LED hardware functionality
- **Audio Devices**: List all audio input/output devices

### Audio Controls
- **List Devices**: Show all audio devices
- **Reset USB**: Reset USB audio devices (fixes common issues)
- **Test Record**: Quick recording test with ReSpeaker

### Troubleshooting Tools
- **Force Kill**: Emergency stop of all VocalGem processes
- **Force Stop**: Forcefully stop the systemd service
- **GPIO Cleanup**: Clear GPIO conflicts
- **Complete Reset**: Full system reset and restart
- **Check Virtual Env**: Verify Python environment

## 🔒 Security Considerations

### Permissions
- Some commands require **sudo privileges**
- The web UI warns when running without sufficient permissions
- For full functionality, run with: `sudo python3 web_ui.py`

### Network Security
- The web UI runs on **port 5000**
- Accessible on **local network only** (not internet)
- No authentication required (local network assumed safe)
- Use **firewall rules** if needed for additional security

## 🛠️ Technical Details

### Dependencies
- **Flask**: Web framework
- **Flask-SocketIO**: Real-time WebSocket communication
- **Bootstrap 5**: Modern, responsive UI framework
- **Font Awesome**: Beautiful icons

### Architecture
- **Backend**: Python Flask application
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Real-time**: WebSocket for live log streaming
- **Responsive**: Mobile-first design approach

### File Structure
```
vocalgem/
├── web_ui.py              # Main Flask application
├── templates/
│   └── index.html         # Web interface template
├── requirements_web.txt   # Web UI dependencies
├── start_web_ui.sh       # Startup script
└── WEB_UI_README.md      # This documentation
```

## 🎨 Interface Features

### Visual Indicators
- 🟢 **Green dot**: Service active/healthy
- 🔴 **Red dot**: Service inactive/error
- 🟡 **Yellow dot**: Unknown/checking status
- ✅ **Green checkmark**: Success/enabled
- ❌ **Red X**: Failed/disabled

### Color Coding
- **Blue**: Primary actions and information
- **Green**: Success states and start actions
- **Yellow/Orange**: Warning states and stop actions
- **Red**: Error states and dangerous actions
- **Gray**: Secondary actions and disabled states

### Responsive Design
- **Desktop**: Full feature layout with side-by-side panels
- **Tablet**: Stacked layout with optimized spacing
- **Mobile**: Single column with collapsible sections

## 🔧 Troubleshooting Web UI

### Common Issues

#### "Commands failing" / "Permission denied"
**Solution**: Run with sudo privileges
```bash
sudo python3 web_ui.py
```

#### "Connection refused" from other devices
**Solution**: Check firewall and ensure Pi is accessible
```bash
# Test connectivity
ping [pi-ip-address]

# Check if port 5000 is open
nmap -p 5000 [pi-ip-address]
```

#### "ModuleNotFoundError" when starting
**Solution**: Install web dependencies
```bash
source venv/bin/activate
pip install -r requirements_web.txt
```

#### Web UI won't start
**Solution**: Check if port 5000 is already in use
```bash
# Check what's using port 5000
sudo netstat -tulpn | grep :5000

# Kill any process using port 5000 if needed
sudo pkill -f "web_ui.py"
```

### Debug Mode
For development/debugging, edit `web_ui.py` and change:
```python
socketio.run(app, host='0.0.0.0', port=5000, debug=True)
```

## 📞 Support

### Getting Help
1. Check the **diagnostics** in the web UI
2. Review **live logs** for error messages
3. Try the **troubleshooting tools** in the interface
4. Refer to the main `VOCALGEM_COMMANDS.md` for detailed command reference

### Useful Commands
```bash
# Check if web UI is running
ps aux | grep web_ui.py

# Stop web UI
pkill -f web_ui.py

# Check port usage
sudo netstat -tulpn | grep :5000

# Test network connectivity
curl http://localhost:5000
```

## 🎉 Enjoy!

Your VocalGem service is now fully manageable through a beautiful, modern web interface accessible from any device on your network. The interface provides everything you need to monitor, control, and troubleshoot your wake word detection system.

**Happy voice controlling! 🎙️✨** 