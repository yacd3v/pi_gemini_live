#!/usr/bin/env python3
"""
Enhanced web stream for IMX500 camera + BNO085 IMU visualization
Access via: http://raspberry_pi_ip:5000
"""

import time
import io
import threading
import json
import math
from flask import Flask, render_template, Response, jsonify
from picamera2 import Picamera2
from picamera2.devices import IMX500
from libcamera import Transform
import cv2
import numpy as np

# IMU imports
try:
    import board
    import busio
    from adafruit_bno08x.i2c import BNO08X_I2C
    from adafruit_bno08x import (
        BNO_REPORT_ACCELEROMETER,
        BNO_REPORT_GYROSCOPE,
        BNO_REPORT_MAGNETOMETER,
        BNO_REPORT_ROTATION_VECTOR,
        BNO_REPORT_LINEAR_ACCELERATION,
        BNO_REPORT_GRAVITY
    )
    IMU_AVAILABLE = True
except ImportError as e:
    print(f"IMU libraries not available: {e}")
    print("Install with: sudo pip3 install adafruit-circuitpython-bno08x")
    IMU_AVAILABLE = False

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
imu_data = {
    'acceleration': [0, 0, 0],
    'quaternion': [1, 0, 0, 0],
    'euler': [0, 0, 0],
    'timestamp': time.time()
}
lock = threading.Lock()
imu_lock = threading.Lock()

# IMU sensor instance
bno = None
i2c = None

def initialize_imu():
    """Initialize the BNO085 IMU sensor"""
    global bno, i2c
    
    if not IMU_AVAILABLE:
        print("⚠ IMU libraries not available - IMU features disabled")
        return False
    
    try:
        print("📡 Initializing BNO085 IMU sensor...")
        
        # Create I2C bus
        i2c = busio.I2C(board.SCL, board.SDA)
        
        # Scan for devices
        try:
            while not i2c.try_lock():
                time.sleep(0.01)
            
            devices = i2c.scan()
            print(f"Found I2C devices: {[hex(addr) for addr in devices]}")
            
            # Check for BNO085
            bno_addresses = [0x4A, 0x4B]
            found_bno = [addr for addr in devices if addr in bno_addresses]
            
            if found_bno:
                print(f"✓ BNO085 detected at {hex(found_bno[0])}")
            else:
                print("⚠ BNO085 not found at expected addresses")
                
        finally:
            i2c.unlock()
        
        # Create sensor instance
        bno = BNO08X_I2C(i2c)
        
        # Enable features
        bno.enable_feature(BNO_REPORT_ACCELEROMETER)
        bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
        
        print("✓ BNO085 IMU initialized successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize IMU: {e}")
        return False

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
    if q is None or len(q) < 4:
        return 0, 0, 0
        
    w, x, y, z = q
    
    # Convert to Euler angles
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    # Convert to degrees
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def read_imu_data():
    """Read IMU data and update global state"""
    global imu_data, bno
    
    if not bno:
        return
    
    try:
        # Read sensor data
        if bno.acceleration is not None:
            acc = bno.acceleration
        else:
            acc = [0, 0, 0]
            
        if bno.quaternion is not None:
            quat = bno.quaternion
            euler = quaternion_to_euler(quat)
        else:
            quat = [1, 0, 0, 0]
            euler = [0, 0, 0]
        
        # Update global data with thread safety
        with imu_lock:
            imu_data = {
                'acceleration': acc,
                'quaternion': quat,
                'euler': euler,
                'timestamp': time.time()
            }
            
    except Exception as e:
        print(f"Error reading IMU data: {e}")

def imu_thread():
    """Background thread for continuous IMU reading"""
    while True:
        read_imu_data()
        time.sleep(0.05)  # 20 Hz update rate

def initialize_camera():
    """Initialize the IMX500 camera"""
    global camera
    
    try:
        # Initialize IMX500 
        imx500 = IMX500()
        camera = Picamera2(imx500.camera_num)
        print(f"📷 Camera initialized with IMX500 on camera {imx500.camera_num}")
        
        # Configure camera
        config = camera.create_preview_configuration(
            main={"size": (640, 480)},
            raw={"size": (2028, 1520)},
            encode="main",
            buffer_count=6
        )
        
        # Set transform if supported
        try:
            config["transform"] = Transform()
        except Exception as e:
            print(f"Could not set transform: {e}")
        
        camera.configure(config)
        
        # Set up network intrinsics for face detection
        if imx500.network_intrinsics:
            ni = imx500.network_intrinsics
            ni.task = "pose estimation"
            ni.inference_rate = 30.0
            print("IMX500 network intrinsics configured")
            
        camera.start()
        print("✓ Camera started successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing IMX500 camera: {e}")
        # Fallback to regular camera
        try:
            camera = Picamera2()
            camera.configure(camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            ))
            camera.start()
            print("✓ Fallback to regular camera successful")
            return True
        except Exception as fallback_e:
            print(f"Fallback camera initialization also failed: {fallback_e}")
            return False

def capture_frames():
    """Continuously capture frames from the camera"""
    global output_frame, lock
    
    while True:
        try:
            # Capture frame
            frame = camera.capture_array()
            
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add timestamp overlay
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Add camera info overlay
            cam_info = "IMX500 + IMU Stream"
            cv2.putText(frame, cam_info, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if ret:
                with lock:
                    output_frame = buffer.tobytes()
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            time.sleep(0.1)
        
        time.sleep(1/30)  # ~30 FPS

def generate_frames():
    """Generate frames for streaming"""
    global output_frame, lock
    
    while True:
        while output_frame is None:
            time.sleep(0.01)
        
        with lock:
            frame = output_frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Main page with video stream and IMU visualization"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Robot Camera + IMU Stream</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .content {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .video-section {
            flex: 1;
            min-width: 400px;
        }
        .imu-section {
            flex: 1;
            min-width: 400px;
        }
        .video-container {
            border: 2px solid #444;
            border-radius: 10px;
            overflow: hidden;
            background: #000;
        }
        .imu-container {
            border: 2px solid #444;
            border-radius: 10px;
            background: #2a2a2a;
            padding: 20px;
            height: 400px;
        }
        .imu-data {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        .data-card {
            background: #333;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .data-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .data-label {
            font-size: 12px;
            color: #ccc;
            margin-top: 5px;
        }
        #imu-visualization {
            width: 100%;
            height: 300px;
            border-radius: 8px;
            background: #000;
        }
        .status {
            text-align: center;
            padding: 10px;
            background: #333;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .status.online { background: #2d5a2d; }
        .status.offline { background: #5a2d2d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Robot Camera + IMU Live Stream</h1>
            <p>Real-time camera feed and IMU sensor visualization</p>
        </div>
        
        <div class="content">
            <div class="video-section">
                <h3>📷 Camera Feed</h3>
                <div class="video-container">
                    <img src="/video_feed" alt="Camera Stream" style="width: 100%; height: auto;">
                </div>
            </div>
            
            <div class="imu-section">
                <h3>📡 IMU Sensor Data</h3>
                <div class="imu-container">
                    <div class="status" id="imu-status">
                        <span id="status-text">Connecting to IMU...</span>
                    </div>
                    
                    <div class="imu-data">
                        <div class="data-card">
                            <div class="data-value" id="roll">0.0°</div>
                            <div class="data-label">Roll</div>
                        </div>
                        <div class="data-card">
                            <div class="data-value" id="pitch">0.0°</div>
                            <div class="data-label">Pitch</div>
                        </div>
                        <div class="data-card">
                            <div class="data-value" id="yaw">0.0°</div>
                            <div class="data-label">Yaw</div>
                        </div>
                        <div class="data-card">
                            <div class="data-value" id="accel">0.0</div>
                            <div class="data-label">Accel (m/s²)</div>
                        </div>
                    </div>
                    
                    <div id="imu-visualization"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Three.js IMU visualization
        let scene, camera, renderer, cube;
        let isInitialized = false;
        
        function initVisualization() {
            const container = document.getElementById('imu-visualization');
            
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            
            // Camera setup
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.z = 5;
            
            // Renderer setup
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);
            
            // Create robot representation (simple cube)
            const geometry = new THREE.BoxGeometry(2, 1, 1);
            const material = new THREE.MeshBasicMaterial({ 
                color: 0x4CAF50,
                wireframe: true,
                transparent: true,
                opacity: 0.8
            });
            cube = new THREE.Mesh(geometry, material);
            scene.add(cube);
            
            // Add coordinate axes
            const axesHelper = new THREE.AxesHelper(3);
            scene.add(axesHelper);
            
            isInitialized = true;
            animate();
        }
        
        function animate() {
            if (!isInitialized) return;
            
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }
        
        function updateVisualization(roll, pitch, yaw) {
            if (!isInitialized || !cube) return;
            
            // Convert degrees to radians and apply rotations
            const rollRad = THREE.MathUtils.degToRad(roll);
            const pitchRad = THREE.MathUtils.degToRad(pitch);
            const yawRad = THREE.MathUtils.degToRad(yaw);
            
            // Apply rotations (order: yaw, pitch, roll)
            cube.rotation.set(pitchRad, yawRad, rollRad);
        }
        
        // IMU data polling
        function updateIMUData() {
            fetch('/imu_data')
                .then(response => response.json())
                .then(data => {
                    // Update status
                    const statusEl = document.getElementById('imu-status');
                    const statusText = document.getElementById('status-text');
                    
                    if (data.status === 'online') {
                        statusEl.className = 'status online';
                        statusText.textContent = 'IMU Online - Live Data';
                        
                        // Update data displays
                        document.getElementById('roll').textContent = data.euler[0].toFixed(1) + '°';
                        document.getElementById('pitch').textContent = data.euler[1].toFixed(1) + '°';
                        document.getElementById('yaw').textContent = data.euler[2].toFixed(1) + '°';
                        
                        // Calculate acceleration magnitude
                        const accel = Math.sqrt(
                            data.acceleration[0]**2 + 
                            data.acceleration[1]**2 + 
                            data.acceleration[2]**2
                        );
                        document.getElementById('accel').textContent = accel.toFixed(2);
                        
                        // Update 3D visualization
                        updateVisualization(data.euler[0], data.euler[1], data.euler[2]);
                        
                    } else {
                        statusEl.className = 'status offline';
                        statusText.textContent = 'IMU Offline - No Data';
                    }
                })
                .catch(error => {
                    console.error('Error fetching IMU data:', error);
                    document.getElementById('imu-status').className = 'status offline';
                    document.getElementById('status-text').textContent = 'IMU Error - Check Connection';
                });
        }
        
        // Initialize visualization when page loads
        window.addEventListener('load', () => {
            initVisualization();
            
            // Start IMU data polling
            updateIMUData();
            setInterval(updateIMUData, 100); // 10 Hz update rate
        });
        
        // Handle window resize
        window.addEventListener('resize', () => {
            if (renderer && camera) {
                const container = document.getElementById('imu-visualization');
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }
        });
    </script>
</body>
</html>
'''

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/imu_data')
def imu_data_endpoint():
    """IMU data endpoint for AJAX requests"""
    global imu_data, bno
    
    with imu_lock:
        data = imu_data.copy()
    
    # Add status information
    data['status'] = 'online' if bno else 'offline'
    data['imu_available'] = IMU_AVAILABLE
    
    return jsonify(data)

@app.route('/status')
def status():
    """Status endpoint"""
    return {
        'status': 'running',
        'camera': 'IMX500' if camera else 'not initialized',
        'imu': 'BNO085' if bno else 'not initialized',
        'imu_available': IMU_AVAILABLE,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

def main():
    """Main function"""
    print("=" * 70)
    print("🤖 Enhanced Robot Camera + IMU Web Stream Server")
    print("=" * 70)
    
    # Initialize camera
    print("📷 Initializing camera...")
    if not initialize_camera():
        print("❌ Failed to initialize camera. Exiting.")
        return
    
    # Initialize IMU
    print("📡 Initializing IMU sensor...")
    imu_success = initialize_imu()
    if not imu_success:
        print("⚠ IMU initialization failed - continuing without IMU features")
    
    # Start background threads
    print("🎬 Starting background threads...")
    
    # Camera capture thread
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    
    # IMU reading thread
    if imu_success:
        imu_thread_obj = threading.Thread(target=imu_thread, daemon=True)
        imu_thread_obj.start()
        print("✓ IMU thread started")
    
    # Wait for first frame
    print("⏳ Waiting for first frame...")
    while output_frame is None:
        time.sleep(0.1)
    
    print("✅ Stream ready!")
    print(f"🌐 Access at: http://localhost:5000")
    print("📱 Or from network: http://YOUR_PI_IP:5000")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 70)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
    finally:
        # Cleanup
        if camera:
            camera.stop()
            camera.close()
        print("✅ Cleanup completed")

if __name__ == '__main__':
    main() 