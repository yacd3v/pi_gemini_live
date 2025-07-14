# 🤖 Robot Camera + IMU Live Stream

A comprehensive web interface that combines live video streaming from your IMX500 camera with real-time IMU sensor visualization for your Raspberry Pi 5 robot.

## ✨ Features

- **📷 Live Camera Stream**: IMX500 camera feed with timestamp overlay
- **📡 Real-time IMU Data**: BNO085 sensor readings (acceleration, orientation)
- **🎮 3D Visualization**: Interactive 3D robot representation using Three.js
- **📊 Live Data Display**: Real-time roll, pitch, yaw, and acceleration values
- **🌐 Network Accessible**: View from any device on your local network
- **🔄 Auto-fallback**: Graceful handling if IMU is not available

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install IMU library
sudo pip3 install adafruit-circuitpython-bno08x

# Install other dependencies
pip install flask opencv-python
```

### 2. Test IMU Integration (Optional but Recommended)

```bash
python3 test_imu_integration.py
```

This will test your BNO085 sensor and verify it's working correctly.

### 3. Run the Enhanced Stream

```bash
python3 camera_imu_stream.py
```

### 4. Access the Interface

- **On the Pi**: `http://localhost:5000`
- **From other devices**: `http://YOUR_PI_IP:5000`

## 🎯 What You'll See

### Camera Section
- Live video feed from IMX500 camera
- Timestamp overlay
- Camera status information

### IMU Section
- **Real-time Data Cards**:
  - Roll angle (X-axis rotation)
  - Pitch angle (Y-axis rotation) 
  - Yaw angle (Z-axis rotation)
  - Acceleration magnitude
- **3D Visualization**: Interactive wireframe cube showing robot orientation
- **Status Indicator**: Shows if IMU is online/offline

## 🔧 Hardware Requirements

### Camera
- IMX500 camera module (same as your main robot script)
- Connected to CSI port

### IMU Sensor
- BNO085 (or BNO080/BNO086) sensor
- **I2C Connection**:
  - VIN → 3.3V or 5V
  - GND → GND
  - SCL → GPIO 3 (Pin 5)
  - SDA → GPIO 2 (Pin 3)

## 📊 Understanding the Data

### Euler Angles
- **Roll**: Rotation around X-axis (left/right tilt)
- **Pitch**: Rotation around Y-axis (forward/backward tilt)
- **Yaw**: Rotation around Z-axis (left/right turn)

### Acceleration
- Shows the magnitude of acceleration in m/s²
- Useful for detecting movement and impacts

### 3D Visualization
- The wireframe cube represents your robot's orientation
- Red axis = X (Roll)
- Green axis = Y (Pitch)
- Blue axis = Z (Yaw)

## 🛠️ Troubleshooting

### IMU Not Working
1. **Check connections**: Ensure I2C wires are properly connected
2. **Enable I2C**: Make sure I2C is enabled in `raspi-config`
3. **Check address**: Run `i2cdetect -y 1` to see available devices
4. **Test separately**: Run `test_imu_integration.py` first

### Camera Issues
- The script will automatically fall back to regular camera if IMU500 fails
- Check camera connections and permissions

### Web Interface Not Loading
- Ensure port 5000 is not blocked by firewall
- Try accessing from the Pi itself first: `http://localhost:5000`

## 🔍 Technical Details

- **Camera Resolution**: 640x480
- **IMU Update Rate**: 20 Hz
- **Web Interface**: Flask + Three.js
- **Streaming Format**: MJPEG for video, JSON for IMU data
- **Threading**: Separate threads for camera capture and IMU reading

## 📁 Files

- `camera_imu_stream.py` - Main streaming server
- `test_imu_integration.py` - IMU testing script
- `requirements_imu_stream.txt` - Dependencies
- `README_camera_imu_stream.md` - This file

## 🎮 Usage Tips

1. **Calibration**: Keep the robot still for a few seconds to let the IMU calibrate
2. **Movement Detection**: Watch the acceleration value to detect when the robot moves
3. **Orientation**: Use the 3D visualization to understand the robot's current pose
4. **Network Access**: Share your Pi's IP address with others to let them view the stream

## 🔄 Integration with Main Robot

This stream can run alongside your main `vocal_gemini.py` script, but be aware:
- Both scripts use the camera, so run them separately
- I2C bus is shared, but the IMU library handles this gracefully
- Consider running this as a monitoring tool while testing your robot

## 🛑 Stopping the Server

Press `Ctrl+C` in the terminal running the script.

---

**Enjoy monitoring your robot's movements in real-time! 🤖📡** 