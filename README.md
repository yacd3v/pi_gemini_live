# 🤖 Robot Control Dashboard

Enhanced low-latency web stream with IMX500 camera + BNO085 IMU + Motor Controls

## 📁 Project Structure

```
vocalgem/
├── robot_dashboard.py          # Main application entry point
├── camera_imu_motor_stream.py  # Original monolithic script (legacy)
├── modules/                    # Modular components
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── camera.py              # Camera management
│   ├── imu.py                 # IMU sensor management
│   ├── motors.py              # Motor control management
│   ├── sensors.py             # Sensors (ultrasonic, battery, ADC, servo)
│   └── web_routes.py          # Flask web routes and API
├── templates/                  # HTML templates
│   └── index.html             # Main dashboard interface
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Raspberry Pi with Python 3.7+
- Required hardware:
  - IMX500 camera
  - BNO085 IMU sensor
  - Mecanum wheel motors
  - Ultrasonic sensor
  - Battery monitoring system
  - Servo motors (for camera pan/tilt)

### Installation

1. **Install Python dependencies:**
   ```bash
   pip3 install flask picamera2 opencv-python numpy adafruit-circuitpython-bno08x
   ```

2. **Install hardware libraries:**
   ```bash
   # Clone Freenove examples (if not already present)
   git clone https://github.com/Freenove/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi.git freenove_examples
   ```

3. **Run the application:**
   ```bash
   python3 robot_dashboard.py
   ```

4. **Access the dashboard:**
   - Local: http://localhost:5001
   - Network: http://YOUR_PI_IP:5001

## 🏗️ Architecture

### Modular Design

The application has been refactored into a clean, modular architecture:

#### **Core Modules**

- **`config.py`** - Centralized configuration management
- **`camera.py`** - IMX500 camera operations and streaming
- **`imu.py`** - BNO085 IMU sensor data processing
- **`motors.py`** - Mecanum car motor control and safety
- **`sensors.py`** - All sensor management (ultrasonic, battery, ADC, servo)
- **`web_routes.py`** - Flask web interface and API endpoints

#### **Main Application**

- **`robot_dashboard.py`** - Clean main script that orchestrates all modules
- **`templates/index.html`** - Separated HTML template for easy maintenance

### Key Benefits

✅ **Maintainable** - Each component is isolated and focused  
✅ **Testable** - Individual modules can be tested independently  
✅ **Extensible** - Easy to add new features or modify existing ones  
✅ **Readable** - Clear separation of concerns  
✅ **Reusable** - Modules can be used in other projects  

## 🎯 Features

### Camera System
- **Ultra-low latency** streaming (optimized for speed)
- **IMX500** camera support with fallback
- **Configurable** resolution and quality settings
- **Frame dropping** for minimal delay

### IMU Integration
- **BNO085** sensor support
- **Real-time** orientation data (roll, pitch, yaw)
- **3D visualization** with Three.js
- **Acceleration** monitoring

### Motor Control
- **Mecanum wheel** support
- **Safety features** (disabled by default)
- **Multiple movement** patterns (forward, backward, strafe, spin)
- **Speed control** with PWM management

### Sensor Suite
- **Ultrasonic** distance measurement
- **Battery monitoring** (main and chassis)
- **ADC voltage** monitoring
- **Servo control** for camera pan/tilt

### Web Interface
- **Real-time** data visualization
- **Keyboard controls** (WASD, arrow keys)
- **Touch-friendly** mobile interface
- **Safety warnings** and status indicators

## 🎮 Controls

### Keyboard Shortcuts

| Key | Function |
|-----|----------|
| `W/S` | Forward/Backward |
| `A/D` | Spin Left/Right |
| `Q/E` | Strafe Left/Right |
| `Space` | Emergency Stop |
| `←/→` | Camera Pan |
| `↑/↓` | Camera Tilt |
| `Home` | Center Camera |
| `F1` | Toggle Motors |
| `F2` | Emergency Stop |
| `F3` | Center Camera |
| `H` | Show Help |

### Web Interface
- **Motor Controls** - Enable/disable, movement buttons
- **Camera Controls** - Pan/tilt sliders and buttons
- **Status Monitoring** - Real-time sensor data
- **Configuration** - Overlay toggles and debug info

## ⚙️ Configuration

All settings are centralized in `modules/config.py`:

```python
# Performance settings
STREAM_CONFIG = {
    'resolution': (320, 240),
    'jpeg_quality': 60,
    'target_fps': 30,
    'max_frame_age': 0.05,
    'frame_buffer_size': 1,
    'skip_overlays': True
}

# Motor settings
MOTOR_CONFIG = {
    'base_speed': 2000,
    'max_pwm': 4095,
    'enabled': False  # Safety default
}
```

## 🔧 Development

### Adding New Features

1. **Create a new module** in `modules/`
2. **Add configuration** in `modules/config.py`
3. **Update main script** to initialize the new module
4. **Add web routes** if needed in `modules/web_routes.py`

### Example: Adding a New Sensor

```python
# modules/new_sensor.py
class NewSensorManager:
    def __init__(self):
        # Initialize sensor
        pass
    
    def initialize(self):
        # Setup hardware
        pass
    
    def get_data(self):
        # Return sensor data
        pass

# robot_dashboard.py
self.managers['new_sensor'] = NewSensorManager()
self.new_sensor_success = self.managers['new_sensor'].initialize()
```

## 🛡️ Safety Features

- **Motors disabled by default** - Must be explicitly enabled
- **Emergency stop** - Multiple ways to stop motors
- **Automatic cleanup** - Motors stop on exit
- **Safety warnings** - Clear UI indicators
- **Input validation** - All commands are validated

## 📊 Performance

### Optimizations
- **Ultra-low latency** camera streaming
- **Minimal frame buffering** (1 frame)
- **Aggressive frame dropping** for speed
- **Threaded operations** for non-blocking I/O
- **Optimized JPEG encoding** settings

### Latency Targets
- **Camera stream**: <50ms end-to-end
- **Motor control**: <10ms response
- **IMU data**: 20Hz update rate
- **Web interface**: Real-time updates

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check IMX500 connections
   - Verify camera permissions
   - Try fallback camera mode

2. **IMU not working**
   - Check I2C connections
   - Install adafruit-circuitpython-bno08x
   - Verify I2C address (0x4A or 0x4B)

3. **Motors not responding**
   - Ensure motors are enabled via web interface
   - Check PCA9685 connections
   - Verify PWM frequency settings

4. **High latency**
   - Disable overlays in web interface
   - Use wired network connection
   - Close other browser tabs

### Debug Mode

Enable debug output by modifying `FLASK_CONFIG`:
```python
FLASK_CONFIG = {
    'debug': True,
    # ... other settings
}
```

## 📝 License

This project is part of the Freenove 4WD Smart Car Kit for Raspberry Pi.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**⚠️ Safety Warning**: Motor controls are powerful and can cause injury or damage. Always test in a safe environment with adequate clearance. Motors are disabled by default for safety. 