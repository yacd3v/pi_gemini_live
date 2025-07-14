#!/usr/bin/env python3
"""
Enhanced low-latency web stream with IMX500 camera + BNO085 IMU + Motor Controls
Access via: http://raspberry_pi_ip:5000

⚠️  SAFETY: Motor controls included but not auto-tested. Test manually first!
"""

import time
import io
import threading
import json
import math
import sys
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
from picamera2 import Picamera2
from picamera2.devices import IMX500
from libcamera import Transform
import cv2
import numpy as np

# Motor control imports
try:
    sys.path.append("freenove_examples")
    from pca9685 import PCA9685
    MOTOR_AVAILABLE = True
except ImportError as e:
    print(f"Motor libraries not available: {e}")
    MOTOR_AVAILABLE = False

# Ultrasonic sensor imports
try:
    from freenove_examples.ultrasonic import Ultrasonic
    ULTRASONIC_AVAILABLE = True
except ImportError as e:
    print(f"Ultrasonic libraries not available: {e}")
    ULTRASONIC_AVAILABLE = False

# Battery monitoring imports
try:
    from battery_monitor import UPSMonitor
    BATTERY_AVAILABLE = True
except ImportError as e:
    print(f"Battery monitoring libraries not available: {e}")
    BATTERY_AVAILABLE = False

# ADC monitoring imports (for chassis motor battery)
try:
    sys.path.append("freenove_examples")
    from adc import ADC
    ADC_AVAILABLE = True
except ImportError as e:
    print(f"ADC libraries not available: {e}")
    ADC_AVAILABLE = False

# Servo control imports (for camera pan/tilt)
try:
    sys.path.append("freenove_examples")
    from servo import Servo
    SERVO_AVAILABLE = True
except ImportError as e:
    print(f"Servo libraries not available: {e}")
    SERVO_AVAILABLE = False

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

# Performance configuration
STREAM_CONFIG = {
    'resolution': (320, 240),  # Further reduced resolution for lowest latency
    'jpeg_quality': 60,        # Lower quality for faster encoding
    'target_fps': 30,          # Higher FPS but with aggressive dropping
    'max_frame_age': 0.05,     # Much more aggressive - drop frames older than 50ms
    'frame_buffer_size': 1,    # Keep only 1 latest frame
    'skip_overlays': True      # Skip text overlays for speed
}

# Motor configuration
MOTOR_CONFIG = {
    'base_speed': 2000,        # Default motor speed
    'max_pwm': 4095,           # Maximum PWM value
    'enabled': False           # Safety: disabled by default
}

# Global variables
camera = None
frame_buffer = deque(maxlen=STREAM_CONFIG['frame_buffer_size'])
frame_lock = threading.Lock()
frame_stats = {'dropped': 0, 'served': 0, 'avg_age': 0}

imu_data = {
    'acceleration': [0, 0, 0],
    'quaternion': [1, 0, 0, 0],
    'euler': [0, 0, 0],
    'ultrasonic_distance': 0.0,  # Distance in meters
    'battery_percent': 0,
    'battery_voltage': 0,
    'battery_current': 0,
    'charging_state': 'Unknown',
    'chassis_battery_voltage': 0.0,  # Chassis motor battery voltage
    'chassis_battery_current': 0.0,  # Chassis motor battery current
    'servo_pan': 90,  # Camera pan position (0-180 degrees)
    'servo_tilt': 90,  # Camera tilt position (0-180 degrees)
    'timestamp': time.time()
}
imu_lock = threading.Lock()

# Ultrasonic sensor instance
ultrasonic = None

# Battery monitor instance
battery_monitor = None

# ADC instance (for chassis motor battery)
adc_monitor = None

# Servo instance (for camera pan/tilt)
servo_controller = None

# Motor and IMU instances
bno = None
i2c = None
motor_car = None

class MecanumCar:
    """Motor control class based on customMotor.py"""
    def __init__(self, addr=0x40, base_speed=2000, max_pwm=4095):
        if not MOTOR_AVAILABLE:
            raise Exception("Motor libraries not available")
        
        self.pwm = PCA9685(addr)
        self.pwm.set_pwm_freq(50)
        self.base = base_speed
        self.max = max_pwm
        self.is_enabled = False  # Safety feature

    def _limit(self, v):
        """Keep duty in range"""
        return max(min(v, self.max), -self.max)

    def _wheel(self, fwd_ch, rev_ch, duty):
        """Control one motor"""
        if not self.is_enabled:
            return  # Safety check
            
        if duty > 0:
            self.pwm.set_motor_pwm(rev_ch, 0)
            self.pwm.set_motor_pwm(fwd_ch, duty)
        elif duty < 0:
            self.pwm.set_motor_pwm(fwd_ch, 0)
            self.pwm.set_motor_pwm(rev_ch, -duty)
        else:  # brake/stop
            self.pwm.set_motor_pwm(fwd_ch, 4095)
            self.pwm.set_motor_pwm(rev_ch, 4095)

    def drive(self, fl, rl, fr, rr):
        """Control four motors: front-left, rear-left, front-right, rear-right"""
        if not self.is_enabled:
            return  # Safety check
            
        fl, rl, fr, rr = map(self._limit, (fl, rl, fr, rr))
        self._wheel(0, 1, fl)   # front-left
        self._wheel(3, 2, rl)   # rear-left
        self._wheel(6, 7, fr)   # front-right
        self._wheel(4, 5, rr)   # rear-right

    def stop(self):
        """Stop all motors"""
        self.drive(0, 0, 0, 0)

    def enable(self):
        """Enable motor control (safety feature)"""
        self.is_enabled = True

    def disable(self):
        """Disable motor control and stop"""
        self.stop()
        self.is_enabled = False

    def close(self):
        """Cleanup"""
        self.stop()
        if hasattr(self, 'pwm'):
            self.pwm.close()

def initialize_motors():
    """Initialize motor control system"""
    global motor_car
    
    if not MOTOR_AVAILABLE:
        print("⚠ Motor libraries not available - motor features disabled")
        return False
    
    try:
        print("🚗 Initializing motor control system...")
        motor_car = MecanumCar(
            base_speed=MOTOR_CONFIG['base_speed'],
            max_pwm=MOTOR_CONFIG['max_pwm']
        )
        print("✓ Motor control system initialized (DISABLED for safety)")
        print("⚠️  Use web interface to enable motors before testing")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize motors: {e}")
        return False

def initialize_ultrasonic():
    """Initialize the ultrasonic sensor"""
    global ultrasonic
    
    if not ULTRASONIC_AVAILABLE:
        print("⚠ Ultrasonic libraries not available - ultrasonic features disabled")
        return False
    
    try:
        print("📐 Initializing ultrasonic sensor...")
        ultrasonic = Ultrasonic()
        print("✓ Ultrasonic sensor initialized")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize ultrasonic sensor: {e}")
        return False

def initialize_battery_monitor():
    """Initialize the battery monitor"""
    global battery_monitor
    
    if not BATTERY_AVAILABLE:
        print("⚠ Battery monitoring libraries not available - battery features disabled")
        return False
    
    try:
        print("🔋 Initializing battery monitor...")
        battery_monitor = UPSMonitor()
        print("✓ Battery monitor initialized")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize battery monitor: {e}")
        return False

def initialize_adc():
    """Initialize the ADC for chassis motor battery monitoring"""
    global adc_monitor
    
    if not ADC_AVAILABLE:
        print("⚠ ADC libraries not available - chassis battery features disabled")
        return False
    
    try:
        print("🔋 Initializing ADC for chassis battery monitoring...")
        adc_monitor = ADC()
        print("✓ ADC initialized")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize ADC: {e}")
        return False

def initialize_servos():
    """Initialize the servo controller for camera pan/tilt"""
    global servo_controller
    
    if not SERVO_AVAILABLE:
        print("⚠ Servo libraries not available - camera pan/tilt features disabled")
        return False
    
    try:
        print("🎥 Initializing servo controller for camera pan/tilt...")
        servo_controller = Servo()
        
        # Initialize servos to center position
        servo_controller.set_servo_pwm('0', 90)  # Pan servo
        servo_controller.set_servo_pwm('1', 90)  # Tilt servo
        
        print("✓ Servo controller initialized - camera centered")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize servo controller: {e}")
        return False

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

def read_battery_data():
    """Read battery data and update global state"""
    global battery_monitor
    
    if not battery_monitor:
        return
    
    try:
        status = battery_monitor.get_battery_status()
        if status:
            # Update global data with thread safety
            with imu_lock:
                imu_data['battery_percent'] = status['battery_percent']
                imu_data['battery_voltage'] = status['battery_voltage']
                imu_data['battery_current'] = status['battery_current']
                imu_data['charging_state'] = status['charging_state']
                imu_data['timestamp'] = time.time()
                
    except Exception as e:
        print(f"Error reading battery data: {e}")

def read_adc_data():
    """Read ADC data for chassis motor battery and update global state"""
    global adc_monitor
    
    if not adc_monitor:
        return
    
    try:
        # Read power voltage from channel 2 (as shown in the ADC example)
        # The power value is calculated based on PCB version
        power_voltage = adc_monitor.read_adc(2) * (3 if adc_monitor.pcb_version == 1 else 2)
        
        # For now, we'll estimate current based on voltage drop or set to 0
        # You might need to add a current sensor or calculate based on motor usage
        estimated_current = 0.0  # Placeholder - could be calculated from motor usage
        
        # Update global data with thread safety
        with imu_lock:
            imu_data['chassis_battery_voltage'] = power_voltage
            imu_data['chassis_battery_current'] = estimated_current
            imu_data['timestamp'] = time.time()
            
    except Exception as e:
        print(f"Error reading ADC data: {e}")

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
        
        # Read ultrasonic data
        ultrasonic_distance = 0.0
        if ultrasonic:
            try:
                distance_cm = ultrasonic.get_distance()
                if distance_cm is not None:
                    ultrasonic_distance = distance_cm / 100.0  # Convert to meters
            except Exception as e:
                print(f"Error reading ultrasonic: {e}")
        
        # Update global data with thread safety
        with imu_lock:
            imu_data = {
                'acceleration': acc,
                'quaternion': quat,
                'euler': euler,
                'ultrasonic_distance': ultrasonic_distance,
                'battery_percent': imu_data.get('battery_percent', 0),
                'battery_voltage': imu_data.get('battery_voltage', 0),
                'battery_current': imu_data.get('battery_current', 0),
                'charging_state': imu_data.get('charging_state', 'Unknown'),
                'chassis_battery_voltage': imu_data.get('chassis_battery_voltage', 0.0),
                'chassis_battery_current': imu_data.get('chassis_battery_current', 0.0),
                'servo_pan': imu_data.get('servo_pan', 90),
                'servo_tilt': imu_data.get('servo_tilt', 90),
                'timestamp': time.time()
            }
            
    except Exception as e:
        print(f"Error reading IMU data: {e}")

def read_ultrasonic_data():
    """Read ultrasonic data and update global state"""
    global ultrasonic
    
    if not ultrasonic:
        return
    
    try:
        distance = ultrasonic.get_distance()
        # Assuming distance is in cm, convert to meters for consistency
        distance_m = distance / 100.0
        
        # Update global data with thread safety
        with imu_lock:
            # Add ultrasonic data to imu_data
            imu_data['ultrasonic_distance'] = distance_m
            imu_data['timestamp'] = time.time()
            
    except Exception as e:
        print(f"Error reading ultrasonic data: {e}")

def imu_thread():
    """Background thread for continuous IMU reading"""
    while True:
        read_imu_data()
        time.sleep(0.05)  # 20 Hz update rate

def ultrasonic_thread():
    """Background thread for continuous ultrasonic reading"""
    while True:
        read_ultrasonic_data()
        time.sleep(0.1) # Read every 100ms

def battery_thread():
    """Background thread for battery monitoring (low frequency)"""
    while True:
        read_battery_data()
        time.sleep(5.0)  # Read every 5 seconds - low frequency to avoid impact

def adc_thread():
    """Background thread for ADC monitoring (moderate frequency)"""
    while True:
        read_adc_data()
        time.sleep(1.0)  # Read every 1 second - moderate frequency

def initialize_camera():
    """Initialize the IMX500 camera with ultra-low latency settings"""
    global camera
    
    try:
        # Initialize IMX500 
        imx500 = IMX500()
        camera = Picamera2(imx500.camera_num)
        print(f"📷 Camera initialized with IMX500 on camera {imx500.camera_num}")
        
        # Configure camera for minimal latency - use MJPEG directly
        config = camera.create_video_configuration(
            main={"size": STREAM_CONFIG['resolution'], "format": "XRGB8888"},
            buffer_count=1,  # Absolute minimum buffering
            queue=False      # No frame queuing
        )
        
        # Set transform if supported
        try:
            config["transform"] = Transform()
        except Exception as e:
            print(f"Could not set transform: {e}")
        
        camera.configure(config)
        
        # Disable network intrinsics for lowest latency
        # (comment out AI processing to reduce overhead)
        
        camera.start()
        print(f"✓ Camera started at {STREAM_CONFIG['resolution']} (ultra-low latency mode)")
        return True
        
    except Exception as e:
        print(f"Error initializing IMX500 camera: {e}")
        # Fallback to regular camera with minimal latency config
        try:
            camera = Picamera2()
            config = camera.create_video_configuration(
                main={"size": STREAM_CONFIG['resolution'], "format": "XRGB8888"},
                buffer_count=1,
                queue=False
            )
            camera.configure(config)
            camera.start()
            print("✓ Fallback to regular camera (ultra-low latency mode)")
            return True
        except Exception as fallback_e:
            print(f"Fallback camera initialization also failed: {fallback_e}")
            return False

def capture_frames():
    """Ultra-low latency frame capture with minimal processing"""
    global frame_buffer, frame_lock, frame_stats
    
    frame_interval = 1.0 / STREAM_CONFIG['target_fps']
    last_capture = 0
    frame_count = 0
    
    while True:
        try:
            current_time = time.time()
            
            # More aggressive frame rate control
            if current_time - last_capture < frame_interval:
                continue  # No sleep - just continue immediately
                
            last_capture = current_time
            frame_count += 1
            
            # Skip every other frame for even lower latency
            if frame_count % 2 == 0:
                continue
            
            # Capture frame directly
            frame = camera.capture_array()
            capture_timestamp = time.time()
            
            # Skip all OpenCV processing if overlays disabled
            if STREAM_CONFIG.get('skip_overlays', False):
                # Direct encoding without any processing
                # Convert XRGB to RGB for JPEG encoding
                if frame.shape[2] == 4:  # XRGB format
                    frame_rgb = frame[:, :, :3]  # Drop alpha channel
                else:
                    frame_rgb = frame
                
                # Ultra-fast JPEG encoding
                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY, STREAM_CONFIG['jpeg_quality'],
                    cv2.IMWRITE_JPEG_OPTIMIZE, 0,  # Disable optimization for speed
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 0
                ]
                ret, buffer = cv2.imencode('.jpg', frame_rgb, encode_params)
            else:
                # Minimal processing version (for debugging)
                if len(frame.shape) == 3 and frame.shape[2] >= 3:
                    if frame.shape[2] == 4:  # XRGB
                        frame = frame[:, :, :3]  # Convert to RGB
                    
                    # Ensure frame is contiguous for OpenCV
                    frame = np.ascontiguousarray(frame)
                    
                    # Very minimal overlay
                    motor_status = "ON" if (motor_car and motor_car.is_enabled) else "OFF"
                    cv2.putText(frame, f"M:{motor_status}", (5, 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Fast encoding
                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY, STREAM_CONFIG['jpeg_quality'],
                    cv2.IMWRITE_JPEG_OPTIMIZE, 0,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 0
                ]
                ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if ret:
                frame_data = {
                    'data': buffer.tobytes(),
                    'timestamp': capture_timestamp,
                    'size': len(buffer.tobytes())
                }
                
                # Immediate buffer update - no locking delay
                with frame_lock:
                    # Clear old frames and add new one
                    frame_buffer.clear()
                    frame_buffer.append(frame_data)
                    
        except Exception as e:
            print(f"Error capturing frame: {e}")
            # Don't sleep on error - continue immediately

def get_latest_frame():
    """Get the most recent frame with minimal overhead"""
    global frame_buffer, frame_lock, frame_stats
    
    with frame_lock:
        if not frame_buffer:
            return None
        
        # Since we only keep 1 frame now, just check age and return
        current_time = time.time()
        latest_frame = frame_buffer[0]
        age = current_time - latest_frame['timestamp']
        
        # Drop frame if too old
        if age > STREAM_CONFIG['max_frame_age']:
            frame_buffer.clear()
            frame_stats['dropped'] += 1
            return None
        
        # Return frame immediately
        frame_stats['served'] += 1
        frame_stats['avg_age'] = age
        return latest_frame

def generate_frames():
    """Ultra-low latency frame generator"""
    frame_count = 0
    
    while True:
        frame_data = get_latest_frame()
        
        if frame_data is None:
            continue  # Don't sleep - just continue immediately
        
        frame_count += 1
        
        # Less frequent stats logging to reduce overhead
        if frame_count % 200 == 0:
            print(f"Stream stats - Served: {frame_stats['served']}, "
                  f"Dropped: {frame_stats['dropped']}, "
                  f"Avg age: {frame_stats['avg_age']*1000:.1f}ms")
        
        # Simplified multipart frame with minimal headers
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_data['data'])).encode() + b'\r\n'
               b'\r\n' + frame_data['data'] + b'\r\n')

@app.route('/')
def index():
    """Main page with video stream, IMU visualization, and motor controls"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Ultra-low latency video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
            'Connection': 'keep-alive'
        }
    )

@app.route('/imu_data')
def imu_data_endpoint():
    """IMU data endpoint for AJAX requests"""
    global imu_data, bno
    
    with imu_lock:
        data = imu_data.copy()
    
    data['status'] = 'online' if bno else 'offline'
    data['imu_available'] = IMU_AVAILABLE
    data['ultrasonic_available'] = ULTRASONIC_AVAILABLE
    data['battery_available'] = BATTERY_AVAILABLE
    data['adc_available'] = ADC_AVAILABLE
    data['servo_available'] = SERVO_AVAILABLE
    
    return jsonify(data)

@app.route('/motor_control', methods=['POST'])
def motor_control():
    """Motor control endpoint for web interface"""
    global motor_car
    
    if not motor_car:
        return jsonify({'success': False, 'error': 'Motors not initialized'})
    
    try:
        data = request.get_json()
        action = data.get('action')
        
        if action == 'enable':
            motor_car.enable()
            print("🟢 Motors ENABLED via web interface")
            return jsonify({'success': True, 'message': 'Motors enabled'})
            
        elif action == 'disable':
            motor_car.disable()
            print("⚫ Motors DISABLED via web interface")
            return jsonify({'success': True, 'message': 'Motors disabled'})
            
        elif action == 'stop':
            motor_car.stop()
            print("🛑 Emergency stop triggered")
            return jsonify({'success': True, 'message': 'Motors stopped'})
            
        elif action == 'move':
            if not motor_car.is_enabled:
                return jsonify({'success': False, 'error': 'Motors disabled'})
                
            direction = data.get('direction')
            speed = data.get('speed', 2000)
            
            # Movement mappings - CORRECTED for proper directions
            # Format: (front_left, rear_left, front_right, rear_right)
            movements = {
                'forward': (speed, speed, speed, speed),
                'backward': (-speed, -speed, -speed, -speed),
                'left': (speed, speed, -speed, -speed),      # spin left - CORRECTED
                'right': (-speed, -speed, speed, speed),     # spin right - CORRECTED
                'strafe_left': (-speed, speed, speed, -speed),
                'strafe_right': (speed, -speed, -speed, speed),
                'spin_left': (speed, speed, -speed, -speed),
                'spin_right': (-speed, -speed, speed, speed)
            }
            
            if direction in movements:
                fl, rl, fr, rr = movements[direction]
                motor_car.drive(fl, rl, fr, rr)
                return jsonify({'success': True, 'message': f'Moving {direction}'})
            else:
                return jsonify({'success': False, 'error': 'Invalid direction'})
                
        else:
            return jsonify({'success': False, 'error': 'Invalid action'})
            
    except Exception as e:
        print(f"Motor control error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/toggle_overlays', methods=['POST'])
def toggle_overlays():
    """Toggle overlay processing for debugging latency"""
    STREAM_CONFIG['skip_overlays'] = not STREAM_CONFIG.get('skip_overlays', True)
    mode = "disabled" if STREAM_CONFIG['skip_overlays'] else "enabled"
    print(f"🎥 Overlays {mode} (lower latency when disabled)")
    return jsonify({'success': True, 'overlays_enabled': not STREAM_CONFIG['skip_overlays']})

@app.route('/servo_control', methods=['POST'])
def servo_control():
    """Servo control endpoint for camera pan/tilt"""
    global servo_controller
    
    if not servo_controller:
        return jsonify({'success': False, 'error': 'Servo controller not initialized'})
    
    try:
        data = request.get_json()
        action = data.get('action')
        
        if action == 'pan':
            angle = data.get('angle', 90)
            angle = max(0, min(180, int(angle)))  # Clamp to 0-180 degrees
            servo_controller.set_servo_pwm('0', angle)
            
            # Update global data
            with imu_lock:
                imu_data['servo_pan'] = angle
            
            print(f"🎥 Camera pan set to {angle}°")
            return jsonify({'success': True, 'message': f'Pan set to {angle}°', 'angle': angle})
            
        elif action == 'tilt':
            angle = data.get('angle', 90)
            angle = max(0, min(180, int(angle)))  # Clamp to 0-180 degrees
            servo_controller.set_servo_pwm('1', angle)
            
            # Update global data
            with imu_lock:
                imu_data['servo_tilt'] = angle
            
            print(f"🎥 Camera tilt set to {angle}°")
            return jsonify({'success': True, 'message': f'Tilt set to {angle}°', 'angle': angle})
            
        elif action == 'center':
            # Center both servos
            servo_controller.set_servo_pwm('0', 90)
            servo_controller.set_servo_pwm('1', 90)
            
            # Update global data
            with imu_lock:
                imu_data['servo_pan'] = 90
                imu_data['servo_tilt'] = 90
            
            print("🎥 Camera centered")
            return jsonify({'success': True, 'message': 'Camera centered', 'pan': 90, 'tilt': 90})
            
        else:
            return jsonify({'success': False, 'error': 'Invalid action'})
            
    except Exception as e:
        print(f"Servo control error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/status')
def status():
    """Status endpoint with performance and motor info"""
    return {
        'status': 'running',
        'camera': 'IMX500' if camera else 'not initialized',
        'imu': 'BNO085' if bno else 'not initialized',
        'ultrasonic': 'Connected' if ultrasonic else 'not initialized',
        'battery': 'Connected' if battery_monitor else 'not initialized',
        'adc': 'Connected' if adc_monitor else 'not initialized',
        'servos': 'Connected' if servo_controller else 'not initialized',
        'motors': 'enabled' if (motor_car and motor_car.is_enabled) else 'disabled',
        'motor_available': MOTOR_AVAILABLE,
        'imu_available': IMU_AVAILABLE,
        'ultrasonic_available': ULTRASONIC_AVAILABLE,
        'battery_available': BATTERY_AVAILABLE,
        'adc_available': ADC_AVAILABLE,
        'servo_available': SERVO_AVAILABLE,
        'config': STREAM_CONFIG,
        'motor_config': MOTOR_CONFIG,
        'frame_stats': frame_stats,
        'optimization_mode': 'ultra-low-latency',
        'overlays_enabled': not STREAM_CONFIG.get('skip_overlays', True),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

def main():
    """Main function"""
    print("=" * 80)
    print("🤖 Robot Control Dashboard - Camera + IMU + Motors (ULTRA-LOW LATENCY)")
    print("=" * 80)
    print(f"📊 Stream Configuration (Optimized for Speed):")
    print(f"   Resolution: {STREAM_CONFIG['resolution']} (reduced for speed)")
    print(f"   JPEG Quality: {STREAM_CONFIG['jpeg_quality']}% (optimized for latency)")
    print(f"   Target FPS: {STREAM_CONFIG['target_fps']} (with frame skipping)")
    print(f"   Max Frame Age: {STREAM_CONFIG['max_frame_age']*1000:.0f}ms (aggressive dropping)")
    print(f"   Overlays: {'Disabled' if STREAM_CONFIG.get('skip_overlays') else 'Minimal'}")
    print("-" * 80)
    
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
    
    # Initialize motors
    print("🚗 Initializing motor control...")
    motor_success = initialize_motors()
    if not motor_success:
        print("⚠ Motor initialization failed - continuing without motor features")
    
    # Initialize ultrasonic sensor
    print("📐 Initializing ultrasonic sensor...")
    ultrasonic_success = initialize_ultrasonic()
    if not ultrasonic_success:
        print("⚠ Ultrasonic initialization failed - continuing without ultrasonic features")
    
    # Initialize battery monitor
    print("🔋 Initializing battery monitor...")
    battery_success = initialize_battery_monitor()
    if not battery_success:
        print("⚠ Battery monitor initialization failed - continuing without battery features")
    
    # Initialize ADC for chassis battery
    print("🔋 Initializing ADC for chassis battery monitoring...")
    adc_success = initialize_adc()
    if not adc_success:
        print("⚠ ADC initialization failed - continuing without chassis battery features")
    
    # Initialize servos for camera pan/tilt
    print("🎥 Initializing servo controller for camera pan/tilt...")
    servo_success = initialize_servos()
    if not servo_success:
        print("⚠ Servo initialization failed - continuing without camera pan/tilt features")
    
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
    
    # Ultrasonic reading thread
    if ultrasonic_success:
        ultrasonic_thread_obj = threading.Thread(target=ultrasonic_thread, daemon=True)
        ultrasonic_thread_obj.start()
        print("✓ Ultrasonic thread started")

    # Battery reading thread
    if battery_success:
        battery_thread_obj = threading.Thread(target=battery_thread, daemon=True)
        battery_thread_obj.start()
        print("✓ Battery thread started")

    # ADC reading thread
    if adc_success:
        adc_thread_obj = threading.Thread(target=adc_thread, daemon=True)
        adc_thread_obj.start()
        print("✓ ADC thread started")
    
    # Wait for first frame
    print("⏳ Waiting for first frame...")
    start_wait = time.time()
    while len(frame_buffer) == 0:
        time.sleep(0.01)
        if time.time() - start_wait > 10:
            print("❌ Timeout waiting for first frame")
            return
    
    print("✅ Robot Control Dashboard ready!")
    print(f"🌐 Access at: http://localhost:5001")
    print("📱 Or from network: http://YOUR_PI_IP:5001")
    print("🎯 Features: Camera + IMU + Ultrasonic + Battery + Chassis Battery + Camera Pan/Tilt + Motor Controls")
    print("⚠️  SAFETY: Motors are DISABLED by default!")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 80)
    
    try:
        app.run(
            host='0.0.0.0', 
            port=5001, 
            debug=False, 
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
    finally:
        # Cleanup
        if motor_car:
            motor_car.close()
        if ultrasonic:
            ultrasonic.close()
        if battery_monitor:
            # Battery monitor doesn't have a close method, but we can clean up the I2C bus
            try:
                battery_monitor.bus.close()
            except:
                pass
        if adc_monitor:
            # ADC monitor cleanup
            try:
                adc_monitor.close_i2c()
            except:
                pass
        if servo_controller:
            # Servo controller cleanup - center servos before exiting
            try:
                servo_controller.set_servo_pwm('0', 90)
                servo_controller.set_servo_pwm('1', 90)
            except:
                pass
        if camera:
            camera.stop()
            camera.close()
        print("✅ Cleanup completed")

if __name__ == '__main__':
    main() 