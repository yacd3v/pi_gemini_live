# 🤖 VocalGem Robot Control with BNO085 Absolute Positioning

A dedicated robot control service that integrates the BNO085 9-axis absolute orientation sensor for precise movement control. This system provides accurate positioning feedback and supports high-level movement commands like "move 1 meter forward" and "turn 25 degrees left".

## 🌟 Features

### 🧭 **BNO085 Absolute Positioning**
- **9-axis sensor fusion** (accelerometer, gyroscope, magnetometer)
- **Absolute orientation** tracking with quaternion and Euler angles
- **Position tracking** in 2D space (X, Y coordinates)
- **Real-time calibration** status monitoring
- **Drift compensation** for long-term accuracy

### 🎯 **Precision Movement Control**
- **Exact distance movements** (e.g., move 0.5 meters)
- **Precise angle turns** (e.g., turn 90 degrees)
- **Position-based navigation** (go to specific X, Y coordinates)
- **Movement verification** with actual vs. target feedback
- **Obstacle detection** integration

### 🕹️ **Intuitive Web Interface**
- **Live camera feed** with WebRTC streaming
- **Touch-friendly controls** for mobile devices
- **Real-time sensor data** display
- **Movement statistics** tracking
- **High-level command interface**

### 🗣️ **Natural Language Commands**
- "Move 1 meter forward"
- "Turn 90 degrees left"
- "Go to position (1.5, 2.0)"
- "Turn around 180 degrees"

## 🚀 Quick Start

### 1. **Hardware Setup**

Connect your BNO085 sensor to the Raspberry Pi I2C bus:
```
BNO085    Raspberry Pi
------    ------------
VCC   ->  3.3V (Pin 1)
GND   ->  GND (Pin 6)
SDA   ->  GPIO 2 (Pin 3)
SCL   ->  GPIO 3 (Pin 5)
```

Verify the sensor is detected:
```bash
i2cdetect -y 1
# Should show device at address 0x4A
```

### 2. **Software Installation**

```bash
# Clone/navigate to your VocalGem project
cd /path/to/vocalgem

# Run the setup script
./start_robot_control.sh
```

### 3. **Access the Interface**

Open your browser and go to:
- **Local**: http://localhost:5001
- **Network**: http://[your-pi-ip]:5001

## 📋 Installation Details

### **Prerequisites**
- Raspberry Pi with GPIO access
- I2C enabled (`sudo raspi-config` → Interface Options → I2C)
- Python 3.7+ with virtual environment
- Camera module (optional, for video streaming)

### **Manual Installation**
```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip python3-venv i2c-tools

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements_robot_control.txt

# Add user to i2c group
sudo usermod -a -G i2c $USER
# Log out and log back in for group changes to take effect
```

## 🎮 Usage Guide

### **Web Interface Overview**

#### **Left Panel: Camera & Controls**
- **Camera Feed**: Live video stream from robot camera
- **Movement Controls**: Arrow buttons for directional movement
- **Speed/Distance Sliders**: Fine-tune movement parameters
- **Quick Presets**: 25cm, 50cm, 1m movements and angle turns
- **High-Level Commands**: Natural language input box

#### **Right Panel: Status & Sensors**
- **Robot Status**: Initialize/shutdown controls
- **Position Display**: Current X, Y coordinates and heading compass
- **BNO085 Sensor Data**: Calibration status and orientation
- **Movement Statistics**: Distance traveled, rotation, command count

### **Movement Commands**

#### **Basic Controls**
- **Forward/Backward**: Precise distance-based movement
- **Turn Left/Right**: Exact angle-based rotation
- **Emergency Stop**: Immediate halt of all movement

#### **High-Level Commands**
Type natural language commands in the command box:

```
Move 1 meter forward
Turn 90 degrees left
Go to position (1.5, 2.0)
Turn around
Move 50 centimeters backward
Rotate 45 degrees clockwise
```

#### **Keyboard Controls**
- **W/Arrow Up**: Move forward
- **S/Arrow Down**: Move backward  
- **A/Arrow Left**: Turn left
- **D/Arrow Right**: Turn right
- **Spacebar**: Emergency stop

### **Calibration Process**

The BNO085 requires calibration for optimal accuracy:

1. **System Calibration**: Move the robot around in all directions
2. **Gyroscope**: Keep the robot stationary for a few seconds
3. **Accelerometer**: Place robot in different orientations
4. **Magnetometer**: Rotate robot in figure-8 patterns

**Calibration levels** (0-3, where 3 is fully calibrated):
- **System**: Overall calibration quality
- **Gyro**: Gyroscope calibration
- **Accel**: Accelerometer calibration  
- **Mag**: Magnetometer calibration

## 🔧 Configuration

### **Movement Parameters**
Adjust these in `precision_robot_control.py`:

```python
# Distance precision (meters)
precision = 0.05  # 5cm tolerance

# Angle precision (degrees)  
precision = 2.0   # 2° tolerance

# Motor speed range
speed = 1000      # 200-2000 range
```

### **Calibration Factors**
Fine-tune movement accuracy:

```python
# Adjust based on actual vs. expected movement
distance_calibration_factor = 1.0
rotation_calibration_factor = 1.0
```

## 🛠️ API Reference

### **Robot Control Endpoints**

#### **Initialize Robot**
```http
POST /api/robot/initialize
```

#### **Move Distance**
```http
POST /api/robot/move
Content-Type: application/json

{
  "distance": 1.0,    // meters (positive = forward)
  "speed": 1000,      // motor speed
  "precision": 0.05   // tolerance in meters
}
```

#### **Turn Angle**
```http
POST /api/robot/turn
Content-Type: application/json

{
  "angle": 90,        // degrees (positive = clockwise)
  "speed": 1000,      // motor speed
  "precision": 2.0    // tolerance in degrees
}
```

#### **Go to Position**
```http
POST /api/robot/goto
Content-Type: application/json

{
  "x": 1.5,           // target X coordinate
  "y": 2.0,           // target Y coordinate
  "speed": 1000,      // motor speed
  "precision": 0.1    // tolerance in meters
}
```

#### **Get Status**
```http
GET /api/robot/status

// Response:
{
  "initialized": true,
  "pose": {
    "position": {"x": 0.5, "y": 1.2},
    "heading": 45.5,
    "orientation": {"roll": 0.1, "pitch": -0.2, "yaw": 45.5}
  },
  "bno085_status": {...},
  "movement_stats": {...}
}
```

### **WebSocket Events**

#### **Real-time Sensor Data**
```javascript
socket.emit('request_sensor_data');

socket.on('sensor_data', function(data) {
  // Real-time pose, BNO085 status, movement stats
});
```

#### **High-Level Commands**
```javascript
socket.emit('high_level_command', {
  command: 'Move 1 meter forward'
});

socket.on('command_result', function(data) {
  // Command execution result
});
```

#### **Camera Feed**
```javascript
socket.on('camera_frame', function(data) {
  // Base64 encoded JPEG frame
  image.src = 'data:image/jpeg;base64,' + data.frame;
});
```

## 🐛 Troubleshooting

### **BNO085 Sensor Issues**

#### **Sensor Not Detected**
```bash
# Check I2C connection
i2cdetect -y 1
# Should show device at 0x4A

# Check wiring connections
# Verify 3.3V power supply
# Ensure SDA/SCL are connected correctly
```

#### **Poor Calibration**
```
# Move robot through calibration sequence:
1. Place on flat surface for 10 seconds
2. Slowly rotate in place 360°
3. Move in figure-8 pattern
4. Tilt and rotate in different orientations
```

#### **Drift Over Time**
```python
# Reset position periodically
robot_controller.reset_position()

# Recalibrate sensor
# Wait for full calibration (all levels = 3)
```

### **Movement Accuracy Issues**

#### **Distance Errors**
```python
# Measure actual vs. commanded distance
# Adjust calibration factor
robot_controller.set_calibration_factors(
    distance_factor=1.1,  # Increase if robot moves too little
    rotation_factor=1.0
)
```

#### **Angle Errors**
```python
# Test 360° rotation and measure actual angle
# Adjust rotation calibration factor
robot_controller.set_calibration_factors(
    distance_factor=1.0,
    rotation_factor=0.95  # Decrease if robot rotates too much
)
```

### **Camera Issues**

#### **No Camera Feed**
```bash
# Test camera directly
rpicam-hello --timeout 2000

# Check camera module connection
# Verify camera is enabled in raspi-config
sudo raspi-config  # Interface Options → Camera
```

#### **Poor Video Quality**
```python
# Adjust camera settings in robot_control_service.py
config = camera.create_preview_configuration(
    main={"size": (640, 480)},  # Increase resolution
    lores={"size": (320, 240)}  # Adjust preview size
)
```

### **Network Issues**

#### **Can't Access Web Interface**
```bash
# Check service is running
ps aux | grep robot_control_service

# Check firewall settings
sudo ufw status

# Find Pi IP address
hostname -I

# Test local access first
curl http://localhost:5001/api/robot/status
```

### **Permission Issues**

#### **I2C Access Denied**
```bash
# Add user to i2c group
sudo usermod -a -G i2c $USER

# Check group membership
groups $USER

# May need to log out and back in
```

#### **GPIO Access Denied**
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER

# Run with sudo if needed
sudo python3 robot_control_service.py
```

## 📊 Performance Optimization

### **Movement Accuracy**
- Run calibration sequence regularly
- Use slower speeds for higher precision
- Account for floor surface variations
- Consider wheel slippage factors

### **Sensor Performance**
- Keep BNO085 away from magnetic interference
- Mount sensor rigidly to robot chassis
- Use proper power supply filtering
- Regular recalibration in new environments

### **Network Performance**
- Use 5GHz WiFi for better streaming
- Adjust camera resolution based on network capacity
- Enable QoS for real-time applications

## 🔬 Advanced Usage

### **Custom Movement Sequences**
```python
from precision_robot_control import MovementCommand

# Create command sequence
commands = [
    MovementCommand("move", target_distance=1.0),
    MovementCommand("turn", target_angle=90),
    MovementCommand("move", target_distance=0.5),
    MovementCommand("goto", target_position=(2.0, 2.0))
]

# Execute sequence
results = robot_controller.execute_command_sequence(commands)
```

### **Integration with Existing VocalGem**
```python
# Use alongside voice control
# Different ports: VocalGem on 5000, Robot Control on 5001
# Share robot hardware between services
```

### **Data Logging**
```python
# Access movement history
stats = robot_controller.get_movement_statistics()
history = stats['recent_movements']

# Log to file for analysis
import json
with open('movement_log.json', 'w') as f:
    json.dump(history, f, indent=2)
```

## 📁 File Structure

```
vocalgem/
├── bno085_sensor.py              # BNO085 sensor interface
├── precision_robot_control.py    # High-level movement controller
├── robot_control_service.py      # Flask web service
├── start_robot_control.sh        # Startup script
├── requirements_robot_control.txt # Python dependencies
├── templates/
│   └── robot_control.html        # Web interface
└── ROBOT_CONTROL_BNO085_README.md # This file
```

## 🆘 Support

### **Getting Help**
1. Check the troubleshooting section above
2. Verify hardware connections and I2C setup
3. Test individual components (BNO085, camera, motors)
4. Check system logs for error messages

### **Common Solutions**
- **Restart the service**: `./start_robot_control.sh`
- **Recalibrate sensor**: Reset position and run calibration
- **Check connections**: Verify all hardware connections
- **Update software**: Pull latest changes and reinstall dependencies

## 🎯 Example Use Cases

### **Precision Navigation**
```
1. "Reset position"           # Set origin
2. "Move 2 meters forward"    # Go to (2, 0)
3. "Turn 90 degrees left"     # Face north
4. "Move 1.5 meters forward"  # Go to (2, 1.5)
5. "Go to position (0, 0)"    # Return to origin
```

### **Calibration Test**
```
1. "Move 1 meter forward"     # Test distance accuracy
2. "Turn 360 degrees"         # Test rotation accuracy
3. "Go to position (1, 1)"    # Test position navigation
4. "Go to position (0, 0)"    # Return to start
```

### **Remote Exploration**
```
1. Start camera feed
2. Use directional controls for exploration
3. Set waypoints with "Go to position"
4. Monitor position and orientation in real-time
```

---

**🤖 Happy robot controlling with precision positioning!** 