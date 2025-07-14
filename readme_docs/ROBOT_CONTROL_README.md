# 🤖 VocalGem Robot Control Mode

## Overview

Robot Control Mode is a new feature added to the VocalGem Web UI that temporarily disables voice services and enables direct manual control of your robot. This mode provides:

- **🚗 Robot Movement Controls** - Forward, backward, rotation, and continuous movement
- **📹 Camera Head Controls** - Pan, tilt, and preset positions
- **📺 Live Video Streaming** - Real-time camera feed from the robot
- **🎵 Live Audio Monitoring** - Real-time audio from the microphone array

## 🌟 Features

### 🎛️ **Dual Mode Operation**
- **Voice Mode**: Normal VocalGem wake word detection and AI conversation
- **Robot Mode**: Manual control with live streaming (voice services stopped)
- **Easy Switching**: One-click mode switching through the web interface

### 🚗 **Movement Controls**
- **Directional Buttons**: Intuitive arrow keys for movement
- **Continuous Movement**: Hold buttons for continuous motion
- **Preset Distances**: Quick buttons for 10cm, 20cm, 50cm moves
- **Rotation Controls**: 45°, 90°, 180° rotation presets
- **Speed Control**: Adjustable speed slider (500-2000)
- **Emergency Stop**: Instant stop button and spacebar hotkey

### 📹 **Camera Controls**
- **Manual Pan/Tilt**: Directional buttons and sliders
- **Servo Limits**: Automatically enforced safe movement ranges
- **Camera Presets**: Center, Left, Right, Up, Down positions
- **Real-time Feedback**: Live position updates

### 📺 **Live Streaming**
- **Video Stream**: Real-time 640x480 video at ~10 FPS
- **Audio Monitoring**: Live microphone array audio
- **Stream Controls**: Start/stop streaming independently
- **Low Latency**: Optimized for responsive control

### ⌨️ **Keyboard Controls**
- **WASD or Arrow Keys**: Robot movement
- **Spacebar**: Emergency stop
- **Mouse Controls**: Hold buttons for continuous movement

## 🚀 Getting Started

### 1. Access Robot Mode

1. Open the VocalGem Web UI: `http://your-pi-ip:5000`
2. Scroll to the "Robot Control Mode" section
3. Click "**Activate Robot Mode**"
4. Wait for hardware initialization
5. The robot control interface will appear

### 2. Basic Movement

**Continuous Movement (Recommended)**:
- Hold the directional arrow buttons to move
- Release to stop
- Use the red stop button for emergency stop

**Preset Distance Movement**:
- Click preset buttons (10cm, 20cm, 50cm)
- Use rotation buttons for turning

**Speed Control**:
- Adjust the speed slider (500-2000)
- Higher values = faster movement

### 3. Camera Control

**Quick Controls**:
- Use directional buttons around the center crosshair
- Click "Center" to reset camera position

**Manual Positioning**:
- Use Pan and Tilt sliders for precise control
- Use preset buttons (Left, Right, Up, Down)

**Live Video**:
- Click "Start Video" to begin streaming
- Video appears in the interface
- Click "Stop Video" when done

### 4. Return to Voice Mode

1. Click "**Return to Voice Mode**"
2. All streams will stop automatically
3. Robot hardware will be cleaned up
4. Voice services will restart
5. Normal VocalGem operation resumes

## 🔧 Technical Details

### Hardware Requirements

- **✅ Required**: Raspberry Pi 5 with VocalGem setup
- **✅ Required**: Freenove chassis with motor controller
- **✅ Required**: Servo motors for camera pan/tilt
- **✅ Required**: Picamera2-compatible camera
- **✅ Required**: ReSpeaker microphone array
- **⚠️ Optional**: LED indicators (graceful fallback)

### Network Configuration

- **Web UI Port**: 5000 (HTTP)
- **Video Streaming**: HTTP multipart stream
- **WebSocket**: Real-time robot control
- **Local Network**: Accessible from any WiFi device

### Performance Specifications

- **Video**: 640x480 @ ~10 FPS, JPEG compression
- **Audio**: 16kHz mono, live monitoring
- **Movement**: Variable speed 500-2000, distance-calibrated
- **Latency**: <100ms for control commands
- **Safety**: Emergency stop, servo limits, obstacle detection

## 🎮 Control Reference

### Web Interface Controls

| Control | Action |
|---------|--------|
| **Movement Arrows** | Continuous forward/backward/left/right |
| **Distance Buttons** | Move preset distances (10/20/50cm) |
| **Rotation Buttons** | Rotate by preset angles (45/90/180°) |
| **Red Stop Button** | Emergency stop all movement |
| **Speed Slider** | Adjust movement speed |
| **Camera Arrows** | Pan/tilt camera head |
| **Camera Presets** | Quick camera positions |
| **Pan/Tilt Sliders** | Manual servo positioning |

### Keyboard Controls

| Key | Action |
|-----|--------|
| **W / Arrow Up** | Move forward (hold) |
| **S / Arrow Down** | Move backward (hold) |
| **A / Arrow Left** | Rotate left (hold) |
| **D / Arrow Right** | Rotate right (hold) |
| **Spacebar** | Emergency stop |

## 🔒 Safety Features

### Movement Safety
- **Obstacle Detection**: Ultrasonic sensor integration
- **Emergency Stop**: Multiple stop methods available
- **Speed Limiting**: Configurable speed ranges
- **Servo Limits**: Hardware-enforced angle limits

### System Safety
- **Clean Shutdown**: Proper hardware cleanup on mode switch
- **Resource Management**: Prevents conflicts between modes
- **Error Handling**: Graceful failure recovery
- **Status Monitoring**: Real-time system status display

## 🛠️ Troubleshooting

### Common Issues

#### "Robot mode not active" errors
**Solution**: Make sure to activate robot mode first
```bash
# Check if robot mode activated successfully
curl http://localhost:5000/robot/status
```

#### Camera stream not working
**Solution**: Check camera hardware and permissions
```bash
# Test camera directly
rpicam-hello --timeout 2000

# Check camera permissions
groups $USER | grep video
```

#### Movement commands not working
**Solution**: Verify hardware connections and initialization
```bash
# Check if chassis functions work
cd /home/yannis.achour/dev2/vocalgem
python3 chassis_functions.py 10  # Test 10cm movement
```

#### "Hardware not available" errors
**Solution**: Install missing dependencies
```bash
# Reinstall robot dependencies
source venv/bin/activate
pip install picamera2 opencv-python numpy
```

### Hardware Debugging

#### Test Robot Movement
```bash
cd /home/yannis.achour/dev2/vocalgem
python3 chassis_functions.py 20        # Move 20cm forward
python3 chassis_functions.py rotate 90 # Rotate 90 degrees
```

#### Test Camera System
```bash
# Test camera capture
rpicam-still -o test.jpg

# Test servo controls
python3 -c "from freenove_examples.servo import Servo; s=Servo(); s.set_servo_pwm('0', 90)"
```

#### Check Audio System
```bash
# List audio devices
aplay -l
arecord -l

# Test ReSpeaker
arecord -D plughw:1,0 -c 6 -r 16000 -f S16_LE test.wav
```

### Log Analysis

#### Web UI Logs
```bash
# Check web UI logs
sudo journalctl -u vocalgem-webui -f

# Check for robot mode errors
sudo journalctl -u vocalgem-webui | grep -i robot
```

#### System Logs
```bash
# Check system messages
sudo dmesg | tail -20

# Check USB devices
lsusb | grep -i respeaker
```

## 🔄 Mode Switching Details

### What Happens During Mode Switch

**Activating Robot Mode**:
1. ✋ **Stop** VocalGem voice service (`systemctl stop vocalgem`)
2. 🤖 **Initialize** chassis controller and servo motors
3. 📹 **Setup** camera for streaming
4. 🌐 **Enable** robot control web interface
5. ✅ **Ready** for manual control

**Deactivating Robot Mode**:
1. 📺 **Stop** all active video/audio streams
2. 🧹 **Cleanup** robot hardware resources
3. 🔄 **Restart** VocalGem voice service (`systemctl start vocalgem`)
4. 🎙️ **Resume** normal wake word detection
5. ✅ **Ready** for voice control

### Resource Management

- **Exclusive Access**: Only one mode active at a time
- **Clean Transitions**: Proper resource cleanup between modes
- **Error Recovery**: Automatic cleanup on failures
- **Hardware Protection**: Safe servo limits and emergency stops

## 🚀 Advanced Usage

### Custom Movement Patterns
```javascript
// Example: Custom movement sequence via browser console
async function squarePattern() {
    await moveRobot('forward', 50);
    await rotateRobot('right', 90);
    await moveRobot('forward', 50);
    await rotateRobot('right', 90);
    await moveRobot('forward', 50);
    await rotateRobot('right', 90);
    await moveRobot('forward', 50);
    await rotateRobot('right', 90);
}
```

### Camera Patrol Sequence
```javascript
// Example: Camera patrol pattern
async function cameraPatrol() {
    await setCameraPosition(45, 60);   // Left
    await new Promise(r => setTimeout(r, 2000));
    await setCameraPosition(135, 60);  // Right
    await new Promise(r => setTimeout(r, 2000));
    await setCameraPosition(90, 45);   // Up
    await new Promise(r => setTimeout(r, 2000));
    await setCameraPosition(90, 75);   // Down
    await new Promise(r => setTimeout(r, 2000));
    await setCameraPosition(90, 60);   // Center
}
```

## 📈 Future Enhancements

### Planned Features
- **🎙️ Hybrid Mode**: Voice + manual control simultaneously
- **🗺️ Mapping**: SLAM and autonomous navigation
- **📱 Mobile App**: Dedicated mobile control interface
- **🎮 Gamepad Support**: Xbox/PlayStation controller support
- **🔄 Automation**: Scripted movement sequences
- **📊 Telemetry**: Real-time sensor data display

### Development Roadmap
1. **Phase 1**: ✅ Basic robot control mode (Current)
2. **Phase 2**: 🔄 Enhanced streaming (audio, higher resolution)
3. **Phase 3**: 🎙️ Hybrid voice+manual mode
4. **Phase 4**: 🗺️ Autonomous navigation features

## 🎉 Enjoy Your Robot!

You now have full manual control over your VocalGem robot through a beautiful web interface! This opens up endless possibilities for:

- **🎮 Remote Control**: Control your robot from anywhere on your network
- **👁️ Surveillance**: Use as a mobile security camera
- **🔍 Exploration**: Explore your environment remotely
- **🧪 Development**: Test and calibrate robot behaviors
- **🎓 Learning**: Understand robotics and control systems

**Access your robot control at: http://192.168.1.5:5000** 🚀

---

*Combined with the existing voice capabilities, your VocalGem robot is now a versatile platform for both AI interaction and manual exploration!* 