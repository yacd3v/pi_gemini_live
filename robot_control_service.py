#!/usr/bin/env python3
"""
VocalGem Robot Control Service
A dedicated Flask-based web interface for precise robot control with BNO085 absolute positioning
"""

import os
import json
import time
import threading
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from queue import Queue
import base64

# Import our custom modules
from bno085_sensor import BNO085Controller
from precision_robot_control import PrecisionRobotController, MovementCommand, MovementStatus

# Try to import picamera2 for camera streaming
try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
    import io
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("⚠️  Camera modules not available. Camera streaming will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'robot-control-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
robot_controller = None
camera = None
camera_streaming = False
camera_thread = None
sensor_thread = None
sensor_streaming = False

class CameraStream:
    """Camera streaming handler"""
    
    def __init__(self):
        self.camera = None
        self.streaming = False
        self.frame_queue = Queue(maxsize=2)
        self.last_frame = None
        
    def initialize(self):
        """Initialize camera"""
        if not CAMERA_AVAILABLE:
            logger.warning("Camera not available")
            return False
        
        try:
            self.camera = Picamera2()
            
            # Get camera configuration that works with IMX500
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                lores={"size": (320, 240), "format": "RGB888"}
            )
            
            logger.info(f"Camera configuration: {config}")
            self.camera.configure(config)
            
            # Start camera
            self.camera.start()
            time.sleep(3)  # Let camera warm up longer for IMX500
            
            logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            logger.info("Camera will be disabled, continuing without video feed")
            return False
    
    def start_streaming(self):
        """Start camera streaming"""
        if not self.camera:
            return False
        
        self.streaming = True
        threading.Thread(target=self._stream_frames, daemon=True).start()
        logger.info("Camera streaming started")
        return True
    
    def stop_streaming(self):
        """Stop camera streaming"""
        self.streaming = False
        logger.info("Camera streaming stopped")
    
    def _stream_frames(self):
        """Stream frames in separate thread"""
        while self.streaming:
            try:
                # Capture frame
                frame = self.camera.capture_array("main")
                
                # Handle different frame formats
                if len(frame.shape) == 3:
                    # If RGB format, convert to BGR for OpenCV
                    if frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    # If grayscale, convert to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                # Resize if too large
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    # Convert to base64 for web streaming
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send to web clients
                    socketio.emit('camera_frame', {
                        'frame': frame_base64,
                        'timestamp': time.time()
                    })
                    
                    # Store latest frame
                    self.last_frame = frame_base64
                    
                    # Debug: Log frame emission occasionally
                    if hasattr(self, '_frame_count'):
                        self._frame_count += 1
                    else:
                        self._frame_count = 1
                    
                    if self._frame_count % 50 == 0:  # Log every 50 frames
                        logger.info(f"Camera frame {self._frame_count} emitted, size: {len(frame_base64)} chars")
                
                time.sleep(0.1)  # ~10 FPS
                
            except Exception as e:
                logger.error(f"Camera streaming error: {e}")
                time.sleep(1)
    
    def get_frame_mjpeg(self):
        """Get current frame in MJPEG format"""
        if self.last_frame:
            frame_data = base64.b64decode(self.last_frame)
            return (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        return b''
    
    def cleanup(self):
        """Cleanup camera resources"""
        self.streaming = False
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except:
                pass
        logger.info("Camera cleanup completed")

# Initialize camera stream
camera_stream = CameraStream()

def sensor_data_broadcaster():
    """Broadcast sensor data to connected clients"""
    global sensor_streaming
    
    while sensor_streaming:
        try:
            if robot_controller and robot_controller.is_initialized:
                # Get current pose
                pose = robot_controller.get_current_pose()
                if pose:
                    # Get BNO085 status
                    bno085_status = robot_controller.bno085.get_status_info()
                    
                    # Get movement statistics
                    movement_stats = robot_controller.get_movement_statistics()
                    
                    # Add simulation mode indicator
                    system_status = {
                        'simulation_mode': not robot_controller.bno085.is_connected,
                        'chassis_available': robot_controller.chassis is not None,
                        'servo_available': robot_controller.servo is not None
                    }
                    
                    # Broadcast to all connected clients
                    socketio.emit('sensor_data', {
                        'pose': pose,
                        'bno085_status': bno085_status,
                        'movement_stats': movement_stats,
                        'system_status': system_status,
                        'timestamp': time.time()
                    })
            
            time.sleep(0.5)  # 2 Hz update rate
            
        except Exception as e:
            logger.error(f"Sensor broadcasting error: {e}")
            time.sleep(1)

@app.route('/')
def index():
    """Main robot control interface"""
    return render_template('robot_control.html')

@app.route('/camera_test')
def camera_test():
    """Camera frame test page"""
    with open('test_camera_frames.html', 'r') as f:
        return f.read()

@app.route('/api/robot/initialize', methods=['POST'])
def initialize_robot():
    """Initialize robot controller"""
    global robot_controller
    
    try:
        if not robot_controller:
            robot_controller = PrecisionRobotController()
        
        if robot_controller.initialize():
            return jsonify({
                'success': True,
                'message': 'Robot controller initialized successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to initialize robot controller'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/robot/shutdown', methods=['POST'])
def shutdown_robot():
    """Shutdown robot controller"""
    global robot_controller
    
    try:
        if robot_controller:
            robot_controller.shutdown()
            robot_controller = None
        
        return jsonify({
            'success': True,
            'message': 'Robot controller shutdown successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/robot/status')
def get_robot_status():
    """Get current robot status"""
    try:
        if not robot_controller or not robot_controller.is_initialized:
            return jsonify({
                'initialized': False,
                'error': 'Robot not initialized'
            })
        
        # Get current pose
        pose = robot_controller.get_current_pose()
        
        # Get BNO085 status
        bno085_status = robot_controller.bno085.get_status_info()
        
        # Get movement statistics
        movement_stats = robot_controller.get_movement_statistics()
        
        return jsonify({
            'initialized': True,
            'pose': pose,
            'bno085_status': bno085_status,
            'movement_stats': movement_stats,
            'current_status': robot_controller.current_status.value
        })
    except Exception as e:
        return jsonify({
            'initialized': False,
            'error': str(e)
        })

@app.route('/api/robot/move', methods=['POST'])
def move_robot():
    """Move robot with precision control"""
    try:
        if not robot_controller or not robot_controller.is_initialized:
            return jsonify({
                'success': False,
                'error': 'Robot not initialized'
            })
        
        data = request.json
        distance = data.get('distance', 0.0)  # meters
        speed = data.get('speed', 1000)
        precision = data.get('precision', 0.05)  # 5cm default
        
        result = robot_controller.move_distance(distance, speed, precision)
        
        return jsonify({
            'success': result.success,
            'status': result.status.value,
            'actual_distance': result.actual_distance,
            'final_position': result.final_position,
            'final_heading': result.final_heading,
            'error_message': result.error_message,
            'execution_time': result.execution_time
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/robot/turn', methods=['POST'])
def turn_robot():
    """Turn robot with precision control"""
    try:
        if not robot_controller or not robot_controller.is_initialized:
            return jsonify({
                'success': False,
                'error': 'Robot not initialized'
            })
        
        data = request.json
        angle = data.get('angle', 0.0)  # degrees
        speed = data.get('speed', 1000)
        precision = data.get('precision', 2.0)  # 2° default
        
        result = robot_controller.turn_angle(angle, speed, precision)
        
        return jsonify({
            'success': result.success,
            'status': result.status.value,
            'actual_angle': result.actual_angle,
            'final_heading': result.final_heading,
            'error_message': result.error_message,
            'execution_time': result.execution_time
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/robot/goto', methods=['POST'])
def goto_position():
    """Move robot to specific position"""
    try:
        if not robot_controller or not robot_controller.is_initialized:
            return jsonify({
                'success': False,
                'error': 'Robot not initialized'
            })
        
        data = request.json
        x = data.get('x', 0.0)  # meters
        y = data.get('y', 0.0)  # meters
        speed = data.get('speed', 1000)
        precision = data.get('precision', 0.1)  # 10cm default
        
        result = robot_controller.goto_position(x, y, speed, precision)
        
        return jsonify({
            'success': result.success,
            'status': result.status.value,
            'final_position': result.final_position,
            'final_heading': result.final_heading,
            'error_message': result.error_message,
            'execution_time': result.execution_time
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/robot/execute_sequence', methods=['POST'])
def execute_command_sequence():
    """Execute a sequence of movement commands"""
    try:
        if not robot_controller or not robot_controller.is_initialized:
            return jsonify({
                'success': False,
                'error': 'Robot not initialized'
            })
        
        data = request.json
        commands_data = data.get('commands', [])
        
        # Convert to MovementCommand objects
        commands = []
        for cmd_data in commands_data:
            cmd = MovementCommand(
                command_type=cmd_data.get('type', 'move'),
                target_distance=cmd_data.get('distance', 0.0),
                target_angle=cmd_data.get('angle', 0.0),
                target_position=(cmd_data.get('x', 0.0), cmd_data.get('y', 0.0)),
                speed=cmd_data.get('speed', 1000),
                precision=cmd_data.get('precision', 0.05),
                timeout=cmd_data.get('timeout', 30.0)
            )
            commands.append(cmd)
        
        results = robot_controller.execute_command_sequence(commands)
        
        # Convert results to JSON-serializable format
        results_data = []
        for result in results:
            results_data.append({
                'success': result.success,
                'status': result.status.value,
                'actual_distance': result.actual_distance,
                'actual_angle': result.actual_angle,
                'final_position': result.final_position,
                'final_heading': result.final_heading,
                'error_message': result.error_message,
                'execution_time': result.execution_time
            })
        
        return jsonify({
            'success': True,
            'results': results_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/robot/stop', methods=['POST'])
def emergency_stop():
    """Emergency stop all robot movement"""
    try:
        if robot_controller:
            robot_controller.emergency_stop()
        
        return jsonify({
            'success': True,
            'message': 'Emergency stop executed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/robot/reset_position', methods=['POST'])
def reset_position():
    """Reset robot position to origin"""
    try:
        if robot_controller:
            robot_controller.reset_position()
        
        return jsonify({
            'success': True,
            'message': 'Position reset to origin'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/robot/calibrate', methods=['POST'])
def calibrate_robot():
    """Set calibration factors"""
    try:
        data = request.json
        distance_factor = data.get('distance_factor', 1.0)
        rotation_factor = data.get('rotation_factor', 1.0)
        
        if robot_controller:
            robot_controller.set_calibration_factors(distance_factor, rotation_factor)
        
        return jsonify({
            'success': True,
            'message': 'Calibration factors updated'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera streaming"""
    global camera_streaming
    
    try:
        if not CAMERA_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Camera not available'
            })
        
        if not camera_streaming:
            if camera_stream.initialize():
                camera_stream.start_streaming()
                camera_streaming = True
                return jsonify({
                    'success': True,
                    'message': 'Camera streaming started'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to initialize camera'
                })
        else:
            return jsonify({
                'success': True,
                'message': 'Camera already streaming'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera streaming"""
    global camera_streaming
    
    try:
        if camera_streaming:
            camera_stream.stop_streaming()
            camera_streaming = False
        
        return jsonify({
            'success': True,
            'message': 'Camera streaming stopped'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/camera/feed')
def camera_feed():
    """MJPEG camera feed"""
    def generate():
        while camera_streaming:
            frame = camera_stream.get_frame_mjpeg()
            if frame:
                yield frame
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    
    # Send current robot status
    if robot_controller and robot_controller.is_initialized:
        pose = robot_controller.get_current_pose()
        if pose:
            emit('robot_status', {
                'initialized': True,
                'pose': pose
            })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_sensor_data')
def handle_sensor_data_request():
    """Handle request for sensor data"""
    global sensor_streaming, sensor_thread
    
    if not sensor_streaming:
        sensor_streaming = True
        sensor_thread = threading.Thread(target=sensor_data_broadcaster, daemon=True)
        sensor_thread.start()
        logger.info("Sensor data broadcasting started")

@socketio.on('stop_sensor_data')
def handle_stop_sensor_data():
    """Handle request to stop sensor data"""
    global sensor_streaming
    sensor_streaming = False
    logger.info("Sensor data broadcasting stopped")

@socketio.on('high_level_command')
def handle_high_level_command(data):
    """Handle high-level movement commands via WebSocket"""
    try:
        command = data.get('command', '')
        
        if not robot_controller or not robot_controller.is_initialized:
            emit('command_error', {'error': 'Robot not initialized'})
            return
        
        # Parse natural language commands
        if 'move' in command.lower() and 'forward' in command.lower():
            # Extract distance
            words = command.split()
            distance = 0.5  # default
            for i, word in enumerate(words):
                if word.lower() in ['meter', 'meters', 'm']:
                    try:
                        distance = float(words[i-1])
                        break
                    except:
                        pass
            
            result = robot_controller.move_distance(distance)
            emit('command_result', {
                'success': result.success,
                'message': f"Moved {result.actual_distance:.2f}m forward",
                'result': result.__dict__
            })
        
        elif 'turn' in command.lower():
            # Extract angle
            angle = 90  # default
            direction = 1  # right
            
            if 'left' in command.lower():
                direction = -1
            
            words = command.split()
            for i, word in enumerate(words):
                if word.lower() in ['degree', 'degrees', '°']:
                    try:
                        angle = float(words[i-1])
                        break
                    except:
                        pass
            
            result = robot_controller.turn_angle(angle * direction)
            emit('command_result', {
                'success': result.success,
                'message': f"Turned {result.actual_angle:.1f}°",
                'result': result.__dict__
            })
        
        else:
            emit('command_error', {'error': f'Unknown command: {command}'})
    
    except Exception as e:
        emit('command_error', {'error': str(e)})

def cleanup():
    """Cleanup all resources"""
    global robot_controller, camera_streaming, sensor_streaming
    
    logger.info("Cleaning up resources...")
    
    # Stop streaming
    sensor_streaming = False
    camera_streaming = False
    
    # Cleanup camera
    camera_stream.cleanup()
    
    # Cleanup robot controller
    if robot_controller:
        robot_controller.shutdown()
    
    logger.info("Cleanup completed")

if __name__ == '__main__':
    import atexit
    atexit.register(cleanup)
    
    print("🤖 Starting VocalGem Robot Control Service...")
    print("=" * 50)
    
    # Get network IP
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    print(f"🌐 Access the robot control interface at:")
    print(f"   - Local: http://localhost:5001")
    print(f"   - Network: http://{local_ip}:5001")
    print("")
    print("🧭 Features:")
    print("   - BNO085 absolute positioning")
    print("   - Precision movement control")
    print("   - Live camera streaming")
    print("   - High-level movement commands")
    print("   - Real-time sensor data")
    print("")
    print("Press Ctrl+C to stop the service")
    print("")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup() 