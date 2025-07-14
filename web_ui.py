#!/usr/bin/env python3
"""
VocalGem Web UI
A Flask-based web interface for managing VocalGem wake word detection service
"""

import os
import subprocess
import threading
import time
import json
import base64
import cv2
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit

# Import robot control modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'freenove_examples'))
try:
    # Import chassis functions from the main directory
    from chassis_functions import ChassisController
    # Import servo from freenove_examples
    from freenove_examples.servo import Servo
    ROBOT_HARDWARE_AVAILABLE = True
    print("✅ Robot hardware modules imported successfully")
except ImportError as e:
    print(f"⚠️  Robot hardware modules not available: {e}")
    ROBOT_HARDWARE_AVAILABLE = False

# Camera imports
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    print("⚠️  Camera not available")
    CAMERA_AVAILABLE = False

# Audio imports
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    print("⚠️  Audio not available")
    AUDIO_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vocalgem-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Helper function for immediate logging
def debug_log(msg):
    """Enhanced debug logging with timestamps and hardware state"""
    timestamp = time.strftime('%H:%M:%S.%f')[:-3]
    hardware_state = f"[Cam:{camera_stream_active}|Robot:{robot_mode_active}]"
    print(f"🐛 {timestamp} {hardware_state} {msg}")

def detailed_hardware_debug(operation, **kwargs):
    """Detailed hardware operation debugging"""
    timestamp = time.strftime('%H:%M:%S.%f')[:-3]
    hardware_info = {
        'camera_streaming': camera_stream_active,
        'robot_mode': robot_mode_active,
        'camera_obj': camera is not None,
        'servo_obj': servo_controller is not None,
        'chassis_obj': chassis_controller is not None
    }
    
    print(f"🔍 {timestamp} HARDWARE_DEBUG: {operation}")
    print(f"   State: {hardware_info}")
    for key, value in kwargs.items():
        print(f"   {key}: {value}")
    print("   " + "-" * 40)

# Global variables for log streaming
log_threads = {}
log_active = {}

# Global variables for robot control mode
robot_mode_active = False
chassis_controller = None
servo_controller = None
camera_stream_active = False
audio_stream_active = False
camera = None
audio_stream = None

def run_command(command, timeout=30):
    """Run a system command and return the result"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Command timed out after {timeout} seconds',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def stream_logs(session_id, command):
    """Stream logs in real-time via WebSocket"""
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        while log_active.get(session_id, False) and process.poll() is None:
            line = process.stdout.readline()
            if line:
                socketio.emit('log_line', {
                    'line': line.strip(),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, room=session_id)
            time.sleep(0.1)
        
        process.terminate()
        socketio.emit('log_end', {}, room=session_id)
        
    except Exception as e:
        socketio.emit('log_error', {'error': str(e)}, room=session_id)
    finally:
        if session_id in log_threads:
            del log_threads[session_id]
        if session_id in log_active:
            del log_active[session_id]

# ============================================================================
# ROBOT CONTROL MODE FUNCTIONS
# ============================================================================

def init_robot_hardware():
    """Initialize robot hardware controllers"""
    global chassis_controller, servo_controller
    
    if not ROBOT_HARDWARE_AVAILABLE:
        return False, "Robot hardware modules not available"
    
    try:
        print("🤖 Starting robot hardware initialization...")
        
        # Initialize chassis controller
        if chassis_controller is None:
            print("🚗 Initializing chassis controller...")
            chassis_controller = ChassisController()
            print("✅ Chassis controller initialized")
        
        # Initialize servo controller
        if servo_controller is None:
            print("📹 Initializing servo controller...")
            servo_controller = Servo()
            print("✅ Servo controller initialized")
            
            # Test servo initialization
            print("🔧 Setting initial camera position...")
            try:
                servo_controller.set_servo_pwm('0', 90)  # Pan center
                time.sleep(0.1)
                servo_controller.set_servo_pwm('1', 60)  # Tilt slightly up
                time.sleep(0.1)
                print("✅ Camera positioned to center")
            except Exception as servo_e:
                print(f"⚠️  Warning: Servo positioning failed: {servo_e}")
        
        print("✅ Robot hardware initialization completed successfully")
        return True, "Robot hardware initialized successfully"
        
    except Exception as e:
        print(f"❌ Robot hardware initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Failed to initialize robot hardware: {str(e)}"

def cleanup_robot_hardware():
    """Clean up robot hardware controllers"""
    global chassis_controller, servo_controller
    
    try:
        if chassis_controller:
            chassis_controller.close()
            chassis_controller = None
        if servo_controller:
            servo_controller.close()
            servo_controller = None
        return True, "Robot hardware cleaned up successfully"
    except Exception as e:
        return False, f"Failed to clean up robot hardware: {str(e)}"

def init_camera_stream():
    """Initialize camera for streaming with robust error handling"""
    global camera
    
    if not CAMERA_AVAILABLE:
        return False, "Camera module not available"
    
    try:
        debug_log("Starting camera initialization...")
        
        if camera is not None:
            debug_log("Camera already exists, checking if healthy...")
            try:
                # Test if camera is responsive
                test_frame = camera.capture_array()
                debug_log("Existing camera is healthy")
                return True, "Camera already initialized and healthy"
            except Exception as health_e:
                debug_log(f"Existing camera unhealthy: {health_e}, reinitializing...")
                try:
                    camera.stop()
                    camera.close()
                except Exception as close_e:
                    debug_log(f"Error closing unhealthy camera: {close_e}")
                camera = None
        
        # Force cleanup before new initialization
        try:
            import gc
            gc.collect()
            time.sleep(0.3)  # Give time for cleanup
        except Exception:
            pass
        
        debug_log("Creating new camera instance...")
        from picamera2 import Picamera2
        camera = Picamera2()
        
        # Use robust configuration
        debug_log("Configuring camera...")
        config = camera.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            buffer_count=3  # Moderate buffering for stability
        )
        camera.configure(config)
        
        debug_log("Starting camera...")
        camera.start()
        
        # Allow camera to fully initialize
        time.sleep(0.5)
        
        # Test capture to ensure everything works
        debug_log("Testing camera capture...")
        test_frame = camera.capture_array()
        debug_log(f"Camera test successful, frame shape: {test_frame.shape}")
        
        debug_log("Camera initialization completed successfully")
        return True, "Camera initialized successfully"
        
    except Exception as e:
        debug_log(f"Camera initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on failure
        if camera:
            try:
                camera.stop()
                camera.close()
            except Exception:
                pass
            camera = None
            
        return False, f"Failed to initialize camera: {str(e)}"

def cleanup_camera_stream():
    """Clean up camera streaming with thorough resource cleanup including threads"""
    global camera, camera_stream_active, video_thread, video_thread_running, video_frame_buffer
    
    try:
        print("🧹 Cleaning up camera and video threads...")
        
        # Stop streaming and threads first
        camera_stream_active = False
        video_thread_running = False
        
        # Wait for video thread to finish
        if video_thread and video_thread.is_alive():
            print("⏳ Waiting for video thread to stop...")
            video_thread.join(timeout=3.0)
            if video_thread.is_alive():
                print("⚠️ Video thread didn't stop gracefully")
            else:
                print("✅ Video thread stopped")
        
        # Clear frame buffer
        with video_frame_lock:
            video_frame_buffer = None
        
        if camera:
            try:
                print("📹 Stopping camera...")
                camera.stop()
                time.sleep(0.2)  # Allow stop to complete
                
                print("📹 Closing camera...")
                camera.close()
                time.sleep(0.2)  # Allow close to complete
                
                camera = None
                print("✅ Camera cleaned up successfully")
            except Exception as camera_error:
                print(f"⚠️ Camera cleanup error: {camera_error}")
                camera = None  # Force cleanup even on error
        
        # Force garbage collection to free resources
        import gc
        gc.collect()
        
        return True, "Camera and video threads cleaned up successfully"
    except Exception as e:
        print(f"❌ Camera cleanup failed: {str(e)}")
        # Force cleanup even on error
        camera = None
        camera_stream_active = False
        video_thread_running = False
        with video_frame_lock:
            video_frame_buffer = None
        return False, f"Failed to clean up camera: {str(e)}"

# Global variables for threaded video streaming
video_frame_buffer = None
video_frame_lock = threading.Lock()
video_thread = None
video_thread_running = False

def threaded_video_capture():
    """Threaded video capture function that runs independently"""
    global camera, video_frame_buffer, video_thread_running
    
    debug_log("Starting threaded video capture")
    frame_count = 0
    
    while video_thread_running and camera_stream_active:
        try:
            if not camera:
                debug_log("No camera available for threaded capture")
                time.sleep(0.5)
                continue
            
            # Capture frame with hardware-friendly timing
            try:
                capture_start = time.time()
                frame = camera.capture_array()
                capture_end = time.time()
                frame_count += 1
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Encode frame as JPEG with speed optimizations
                ret, buffer = cv2.imencode('.jpg', frame_bgr, [
                    cv2.IMWRITE_JPEG_QUALITY, 40,  # Lower quality for better performance
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Optimize for size
                ])
                
                if ret:
                    frame_bytes = buffer.tobytes()
                    
                    # Thread-safe frame buffer update
                    with video_frame_lock:
                        video_frame_buffer = frame_bytes
                
                # Log every 20th frame to reduce console spam
                if frame_count % 20 == 0:
                    detailed_hardware_debug("THREADED_CAPTURE", 
                                          frame_number=frame_count,
                                          capture_time_ms=(capture_end - capture_start) * 1000,
                                          buffer_size=len(frame_bytes) if ret else 0)
                
                # Longer sleep to yield CPU and hardware resources to robot controls
                time.sleep(0.15)  # ~6-7 FPS but much better resource sharing
                
            except Exception as capture_error:
                debug_log(f"Threaded capture error (frame {frame_count}): {capture_error}")
                time.sleep(0.5)  # Longer wait on error to reduce hardware stress
                continue
            
        except Exception as e:
            debug_log(f"Threaded video capture error: {e}")
            time.sleep(0.5)
    
    debug_log(f"Threaded video capture ended after {frame_count} frames")

def generate_video_stream():
    """Generate video frames for streaming using threaded capture"""
    global video_frame_buffer, video_thread, video_thread_running
    
    debug_log("Starting video stream generator (threaded mode)")
    
    # Start the threaded capture if not already running
    if not video_thread or not video_thread.is_alive():
        video_thread_running = True
        video_thread = threading.Thread(target=threaded_video_capture, daemon=True)
        video_thread.start()
        debug_log("Started threaded video capture")
        
        # Give the thread time to capture first frame
        time.sleep(0.5)
    
    last_served_frame = None
    frame_count = 0
    
    while camera_stream_active:
        try:
            # Get current frame from buffer (thread-safe)
            current_frame = None
            with video_frame_lock:
                if video_frame_buffer is not None:
                    current_frame = video_frame_buffer
            
            # Only serve new frames to reduce bandwidth
            if current_frame and current_frame != last_served_frame:
                frame_count += 1
                last_served_frame = current_frame
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
                
                # Log every 10th served frame
                if frame_count % 10 == 0:
                    debug_log(f"Served frame {frame_count} (size: {len(current_frame)} bytes)")
            
            # Fast refresh rate for web streaming, but actual capture is slower
            time.sleep(0.05)  # 20 FPS serving rate, but capture is ~6-7 FPS
            
        except Exception as e:
            debug_log(f"Video stream serving error: {e}")
            time.sleep(0.1)
    
    # Clean up thread
    video_thread_running = False
    if video_thread and video_thread.is_alive():
        debug_log("Waiting for video thread to stop...")
        video_thread.join(timeout=2.0)
        debug_log("Video thread stopped")
    
    debug_log(f"Video stream generator ended after serving {frame_count} frames")

def init_audio_stream():
    """Initialize audio streaming"""
    global audio_stream
    
    if not AUDIO_AVAILABLE:
        return False, "Audio not available"
    
    try:
        # Find ReSpeaker device
        pya = pyaudio.PyAudio()
        device_index = None
        
        for i in range(pya.get_device_count()):
            info = pya.get_device_info_by_index(i)
            if "respeaker" in info["name"].lower() or "seeed" in info["name"].lower():
                device_index = i
                break
        
        if device_index is None:
            pya.terminate()
            return False, "ReSpeaker device not found"
        
        # Initialize audio stream
        audio_stream = pya.open(
            format=pyaudio.paInt16,
            channels=1,  # Mono for web streaming
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )
        
        pya.terminate()
        return True, "Audio stream initialized successfully"
    except Exception as e:
        return False, f"Failed to initialize audio stream: {str(e)}"

def cleanup_audio_stream():
    """Clean up audio streaming"""
    global audio_stream
    
    try:
        if audio_stream:
            audio_stream.stop_stream()
            audio_stream.close()
            audio_stream = None
        return True, "Audio stream cleaned up successfully"
    except Exception as e:
        return False, f"Failed to clean up audio stream: {str(e)}"

# ============================================================================
# EXISTING ROUTES (unchanged)
# ============================================================================

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/service/<action>')
def service_action(action):
    """Handle service management actions"""
    commands = {
        'start': 'sudo systemctl start vocalgem',
        'stop': 'sudo systemctl stop vocalgem',
        'restart': 'sudo systemctl restart vocalgem',
        'status': 'sudo systemctl status vocalgem',
        'enable': 'sudo systemctl enable vocalgem',
        'disable': 'sudo systemctl disable vocalgem',
        'is-enabled': 'sudo systemctl is-enabled vocalgem',
        'is-active': 'sudo systemctl is-active vocalgem'
    }
    
    if action not in commands:
        return jsonify({'success': False, 'error': 'Invalid action'})
    
    result = run_command(commands[action])
    return jsonify(result)

@app.route('/logs/<log_type>')
def get_logs(log_type):
    """Get various types of logs"""
    commands = {
        'recent': 'sudo journalctl -u vocalgem -n 50 --no-pager',
        'today': 'sudo journalctl -u vocalgem --since today --no-pager',
        '10min': 'sudo journalctl -u vocalgem --since "10 minutes ago" --no-pager',
        'errors': 'sudo journalctl -u vocalgem --no-pager | grep -i error',
        'wake': 'sudo journalctl -u vocalgem --no-pager | grep -i "wake word"',
        'led': 'sudo journalctl -u vocalgem --no-pager | grep -i led'
    }
    
    if log_type not in commands:
        return jsonify({'success': False, 'error': 'Invalid log type'})
    
    result = run_command(commands[log_type])
    return jsonify(result)

@app.route('/test/<test_type>')
def run_test(test_type):
    """Run various tests"""
    commands = {
        'service': './check_service_setup.sh',
        'boot': 'sudo ./test_boot_simulation.sh',
        'led': 'python3 test_wake_led.py',
        'audio-list': 'aplay -l && arecord -l'
    }
    
    if test_type not in commands:
        return jsonify({'success': False, 'error': 'Invalid test type'})
    
    result = run_command(commands[test_type], timeout=60)
    return jsonify(result)

@app.route('/audio/<action>')
def audio_action(action):
    """Handle audio-related actions"""
    commands = {
        'devices': 'aplay -l && echo "=== RECORDING DEVICES ===" && arecord -l',
        'usb-check': 'lsusb | grep 2886',
        'reset-usb': 'sudo modprobe -r snd_usb_audio && sudo modprobe snd_usb_audio',
        'test-record': 'timeout 3 arecord -D plughw:1,0 -c 6 -r 16000 -f S16_LE test_ui.wav',
        'groups': 'groups $USER'
    }
    
    if action not in commands:
        return jsonify({'success': False, 'error': 'Invalid action'})
    
    result = run_command(commands[action])
    return jsonify(result)

@app.route('/troubleshoot/<action>')
def troubleshoot_action(action):
    """Handle troubleshooting actions"""
    commands = {
        'force-kill': 'sudo pkill -f wake_porcu.py',
        'force-stop': 'sudo systemctl kill vocalgem && sudo systemctl stop vocalgem',
        'gpio-cleanup': 'sudo python3 -c "import RPi.GPIO as GPIO; GPIO.cleanup()" 2>/dev/null || true',
        'complete-reset': '''
            sudo systemctl stop vocalgem;
            sudo pkill -f wake_porcu.py;
            sudo python3 -c "import RPi.GPIO as GPIO; GPIO.cleanup()" 2>/dev/null || true;
            sudo ./setup_startup.sh;
            sudo systemctl start vocalgem
        ''',
        'venv-check': 'ls -la venv/ && source venv/bin/activate && python3 -c "import pvporcupine; print(\'Porcupine OK\')" && python3 -c "import pyaudio; print(\'PyAudio OK\')"'
    }
    
    if action not in commands:
        return jsonify({'success': False, 'error': 'Invalid action'})
    
    result = run_command(commands[action], timeout=120)
    return jsonify(result)

@app.route('/diagnostics')
def diagnostics():
    """Run comprehensive diagnostics"""
    checks = [
        ('Service Status', 'sudo systemctl is-active vocalgem'),
        ('Service Enabled', 'sudo systemctl is-enabled vocalgem'),
        ('Audio Device', 'aplay -l | grep -q ReSpeaker && echo "✅ ReSpeaker detected" || echo "❌ ReSpeaker not found"'),
        ('Virtual Environment', '[ -d venv ] && echo "✅ Virtual env exists" || echo "❌ Virtual env missing"'),
        ('Wake Word Listening', 'sudo journalctl -u vocalgem --since "1 minute ago" | grep -q "Listening for wake words" && echo "✅ Listening" || echo "❌ Not listening"'),
        ('Recent Errors', 'sudo journalctl -u vocalgem --since "5 minutes ago" | grep -i error | wc -l'),
        ('USB Device', 'lsusb | grep 2886 && echo "✅ USB device detected" || echo "❌ USB device not found"')
    ]
    
    results = []
    for name, command in checks:
        result = run_command(command)
        results.append({
            'name': name,
            'success': result['success'],
            'output': result['stdout'].strip() if result['stdout'] else result['stderr'].strip(),
            'status': '✅' if result['success'] else '❌'
        })
    
    return jsonify({'diagnostics': results})

@app.route('/logs/cleanup')
def cleanup_logs():
    """Clean up old logs"""
    result = run_command('sudo journalctl --vacuum-time=1d')
    return jsonify(result)

# ============================================================================
# NEW ROBOT CONTROL MODE ROUTES
# ============================================================================

@app.route('/robot/mode/<action>')
def robot_mode_action(action):
    """Handle robot mode activation/deactivation"""
    global robot_mode_active, camera_stream_active, audio_stream_active
    
    if action == 'activate':
        if robot_mode_active:
            return jsonify({'success': False, 'error': 'Robot mode already active'})
        
        # Stop vocal services
        stop_result = run_command('sudo systemctl stop vocalgem')
        if not stop_result['success']:
            return jsonify({'success': False, 'error': 'Failed to stop vocal services'})
        
        # Initialize robot hardware
        hw_success, hw_message = init_robot_hardware()
        if not hw_success:
            return jsonify({'success': False, 'error': hw_message})
        
        # Initialize camera
        cam_success, cam_message = init_camera_stream()
        if not cam_success:
            cleanup_robot_hardware()
            return jsonify({'success': False, 'error': cam_message})
        
        robot_mode_active = True
        return jsonify({
            'success': True, 
            'message': 'Robot control mode activated',
            'hardware_available': ROBOT_HARDWARE_AVAILABLE,
            'camera_available': CAMERA_AVAILABLE,
            'audio_available': AUDIO_AVAILABLE
        })
    
    elif action == 'deactivate':
        if not robot_mode_active:
            return jsonify({'success': False, 'error': 'Robot mode not active'})
        
        try:
            print("🔄 Deactivating robot mode...")
            
            # Stop streams first
            camera_stream_active = False
            audio_stream_active = False
            
            # Wait for streams to stop gracefully
            time.sleep(0.5)
            
            # Clean up hardware with proper error handling
            print("🧹 Cleaning up camera...")
            cam_cleanup_success, cam_cleanup_msg = cleanup_camera_stream()
            if not cam_cleanup_success:
                print(f"⚠️ Camera cleanup warning: {cam_cleanup_msg}")
            
            print("🧹 Cleaning up audio...")
            audio_cleanup_success, audio_cleanup_msg = cleanup_audio_stream()
            if not audio_cleanup_success:
                print(f"⚠️ Audio cleanup warning: {audio_cleanup_msg}")
            
            print("🧹 Cleaning up robot hardware...")
            hw_cleanup_success, hw_cleanup_msg = cleanup_robot_hardware()
            if not hw_cleanup_success:
                print(f"⚠️ Hardware cleanup warning: {hw_cleanup_msg}")
            
            # Mark robot mode as inactive
            robot_mode_active = False
            
            # Wait a bit before restarting voice services
            time.sleep(1)
            
            # Restart vocal services
            print("🔄 Restarting voice services...")
            start_result = run_command('sudo systemctl start vocalgem')
            
            if start_result['success']:
                print("✅ Voice services restarted successfully")
                return jsonify({
                    'success': True, 
                    'message': 'Robot control mode deactivated, vocal services restarted successfully',
                    'vocal_service_started': True
                })
            else:
                print(f"⚠️ Voice service restart warning: {start_result.get('stderr', 'Unknown error')}")
                return jsonify({
                    'success': True, 
                    'message': 'Robot mode deactivated but voice service restart failed. You may need to start it manually.',
                    'vocal_service_started': False,
                    'warning': start_result.get('stderr', 'Voice service restart failed')
                })
            
        except Exception as e:
            print(f"❌ Error during robot mode deactivation: {e}")
            # Even if there's an error, mark as inactive to allow retry
            robot_mode_active = False
            camera_stream_active = False
            audio_stream_active = False
            
            return jsonify({
                'success': False, 
                'error': f'Error during deactivation: {str(e)}. Robot mode marked as inactive.',
                'robot_mode_forced_inactive': True
            })
    
    else:
        return jsonify({'success': False, 'error': 'Invalid action'})

@app.route('/robot/status')
def robot_status():
    """Get robot control mode status"""
    return jsonify({
        'robot_mode_active': robot_mode_active,
        'camera_stream_active': camera_stream_active,
        'audio_stream_active': audio_stream_active,
        'hardware_available': ROBOT_HARDWARE_AVAILABLE,
        'camera_available': CAMERA_AVAILABLE,
        'audio_available': AUDIO_AVAILABLE
    })

@app.route('/robot/move/<action>')
def robot_move(action):
    """Handle robot movement commands with improved hardware-sharing"""
    if not robot_mode_active:
        return jsonify({'success': False, 'error': 'Robot mode not active'})
    
    if not chassis_controller:
        return jsonify({'success': False, 'error': 'Chassis controller not initialized'})
    
    speed = int(request.args.get('speed', 1000))
    distance = float(request.args.get('distance', 20))  # cm
    
    detailed_hardware_debug("MOVEMENT_START", 
                          action=action, 
                          speed=speed, 
                          distance=distance,
                          request_time=time.time())
    
    def execute_movement_with_retry(movement_func, max_retries=2):
        """Execute movement command with retry logic for hardware contention"""
        for attempt in range(max_retries):
            try:
                movement_start = time.time()
                detailed_hardware_debug("MOVEMENT_ATTEMPT", 
                                      attempt=attempt + 1)
                
                result = movement_func()
                movement_end = time.time()
                
                detailed_hardware_debug("MOVEMENT_SUCCESS", 
                                      attempt=attempt + 1,
                                      result=result,
                                      execution_time_ms=(movement_end - movement_start) * 1000)
                return True, result
                
            except Exception as e:
                detailed_hardware_debug("MOVEMENT_FAILED", 
                                      attempt=attempt + 1,
                                      error=str(e))
                if attempt < max_retries - 1:
                    time.sleep(0.2)  # Brief pause before retry
                else:
                    return False, f"Movement failed after {max_retries} attempts: {e}"
        
        return False, "Movement failed unexpectedly"
    
    try:
        if action == 'forward':
            debug_log(f"About to move forward {distance}cm at speed {speed}")
            success, result = execute_movement_with_retry(
                lambda: chassis_controller.move_forward_distance(distance, speed)
            )
            if not success:
                return jsonify({'success': False, 'error': result})
                
        elif action == 'backward':
            debug_log(f"About to move backward {distance}cm at speed {speed}")
            success, result = execute_movement_with_retry(
                lambda: chassis_controller.move_forward_distance(-distance, -speed)
            )
            if not success:
                return jsonify({'success': False, 'error': result})
                
        elif action == 'stop':
            debug_log("Emergency stop requested")
            success, result = execute_movement_with_retry(
                lambda: chassis_controller.emergency_stop() or "Emergency stop executed"
            )
            if not success:
                return jsonify({'success': False, 'error': result})
                
        elif action == 'rotate_left':
            angle = float(request.args.get('angle', 90))
            debug_log(f"About to rotate left {angle}° at speed {speed}")
            success, result = execute_movement_with_retry(
                lambda: chassis_controller.rotate_angle(-angle, speed)
            )
            if not success:
                return jsonify({'success': False, 'error': result})
                
        elif action == 'rotate_right':
            angle = float(request.args.get('angle', 90))
            debug_log(f"About to rotate right {angle}° at speed {speed}")
            success, result = execute_movement_with_retry(
                lambda: chassis_controller.rotate_angle(angle, speed)
            )
            if not success:
                return jsonify({'success': False, 'error': result})
                
        else:
            return jsonify({'success': False, 'error': 'Invalid movement action'})
        
        debug_log(f"Movement completed successfully: {result}")
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        debug_log(f"Movement error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/robot/camera/<action>')
def robot_camera_control(action):
    """Handle camera servo control with improved hardware-sharing"""
    detailed_hardware_debug("SERVO_CONTROL_START", 
                          action=action,
                          request_time=time.time())
    
    if not robot_mode_active:
        debug_log("Camera control failed: Robot mode not active")
        return jsonify({'success': False, 'error': 'Robot mode not active'})
    
    if not servo_controller:
        debug_log("Camera control failed: Servo controller not initialized")
        return jsonify({'success': False, 'error': 'Servo controller not initialized'})
    
    def send_servo_command_with_retry(servo_id, angle, max_retries=3):
        """Send servo command with retry logic for hardware contention"""
        for attempt in range(max_retries):
            try:
                servo_start = time.time()
                detailed_hardware_debug("SERVO_COMMAND_ATTEMPT", 
                                      servo_id=servo_id, 
                                      angle=angle, 
                                      attempt=attempt + 1)
                
                servo_controller.set_servo_pwm(str(servo_id), angle)
                servo_end = time.time()
                
                detailed_hardware_debug("SERVO_COMMAND_SUCCESS", 
                                      servo_id=servo_id, 
                                      angle=angle, 
                                      attempt=attempt + 1,
                                      execution_time_ms=(servo_end - servo_start) * 1000)
                
                # Adaptive delay based on video streaming status
                if camera_stream_active:
                    time.sleep(0.4)  # Longer delay when video is active
                else:
                    time.sleep(0.2)  # Normal delay
                    
                return True, f"Servo {servo_id} moved to {angle}°"
                
            except Exception as e:
                debug_log(f"Servo {servo_id} attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.3)  # Wait before retry
                else:
                    return False, f"Servo {servo_id} failed after {max_retries} attempts: {e}"
        
        return False, f"Servo {servo_id} failed unexpectedly"
    
    try:
        if action == 'pan':
            angle = int(request.args.get('angle', 90))
            original_angle = angle
            angle = max(13, min(154, angle))  # Clamp to servo limits
            debug_log(f"About to pan camera to {angle}° (requested: {original_angle}°)")
            
            success, message = send_servo_command_with_retry(0, angle)
            if not success:
                return jsonify({'success': False, 'error': message})
            result = f"Camera panned to {angle}°"
            
        elif action == 'tilt':
            angle = int(request.args.get('angle', 60))
            original_angle = angle
            angle = max(36, min(85, angle))  # Clamp to servo limits
            debug_log(f"About to tilt camera to {angle}° (requested: {original_angle}°)")
            
            success, message = send_servo_command_with_retry(1, angle)
            if not success:
                return jsonify({'success': False, 'error': message})
            result = f"Camera tilted to {angle}°"
            
        elif action == 'center':
            debug_log("About to center camera (Pan: 90°, Tilt: 60°)")
            
            # Pan first
            success, message = send_servo_command_with_retry(0, 90)
            if not success:
                return jsonify({'success': False, 'error': f"Pan failed: {message}"})
            
            # Small delay between commands
            time.sleep(0.2)
            
            # Tilt second
            success, message = send_servo_command_with_retry(1, 60)
            if not success:
                return jsonify({'success': False, 'error': f"Tilt failed: {message}"})
            
            result = "Camera centered"
            
        else:
            debug_log(f"Invalid camera action: {action}")
            return jsonify({'success': False, 'error': 'Invalid camera action'})
        
        debug_log(f"Camera control completed successfully: {result}")
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        debug_log(f"Camera control error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/robot/video_feed')
def video_feed():
    """Video streaming route with improved error handling"""
    global camera_stream_active
    
    if not robot_mode_active:
        debug_log("Video feed rejected: Robot mode not active")
        return "Robot mode not active", 404
        
    if not camera:
        debug_log("Video feed rejected: No camera available")
        return "Camera not available", 404
    
    debug_log("Starting video feed endpoint")
    camera_stream_active = True
    
    # Simple generator without extra threading complexity
    def safe_video_stream():
        """Safe video stream generator with error recovery"""
        try:
            debug_log("Video stream generator started")
            for frame in generate_video_stream():
                if not camera_stream_active:
                    debug_log("Video stream stopped by flag")
                    break
                yield frame
        except Exception as e:
            debug_log(f"Video feed error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            debug_log("Video stream generator finished")
    
    return Response(safe_video_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/robot/stream/<stream_type>/<action>')
def stream_control(stream_type, action):
    """Control video/audio streams with improved threading support"""
    global camera_stream_active, audio_stream_active, video_thread_running
    
    if not robot_mode_active:
        return jsonify({'success': False, 'error': 'Robot mode not active'})
    
    if stream_type == 'video':
        if action == 'start':
            debug_log("Starting video stream (threaded mode)...")
            
            # Check if camera is available and healthy
            if not camera:
                debug_log("Attempting to reinitialize camera...")
                success, message = init_camera_stream()
                if not success:
                    return jsonify({'success': False, 'error': f'Camera initialization failed: {message}'})
            
            # Enable streaming flags
            camera_stream_active = True
            video_thread_running = True
            
            # Note: The actual thread will be started when video_feed route is called
            debug_log("Video stream enabled - thread will start on first request")
            return jsonify({'success': True, 'message': 'Video stream enabled (threaded mode)'})
            
        elif action == 'stop':
            debug_log("Stopping video stream (threaded mode)...")
            
            # Stop streaming flags
            camera_stream_active = False
            video_thread_running = False
            
            # Give time for stream and thread to stop gracefully
            time.sleep(0.5)
            
            debug_log("Video stream stopped")
            return jsonify({'success': True, 'message': 'Video stream stopped (threaded mode)'})
    
    elif stream_type == 'audio':
        if action == 'start':
            success, message = init_audio_stream()
            if success:
                audio_stream_active = True
            return jsonify({'success': success, 'message': message})
        elif action == 'stop':
            success, message = cleanup_audio_stream()
            if success:
                audio_stream_active = False
            return jsonify({'success': success, 'message': message})
    
    return jsonify({'success': False, 'error': 'Invalid stream control'})

@app.route('/robot/debug/camera_restart')
def debug_camera_restart():
    """Debug endpoint to restart camera if it's in a bad state"""
    global camera, camera_stream_active
    
    if not robot_mode_active:
        return jsonify({'success': False, 'error': 'Robot mode not active'})
    
    try:
        debug_log("Attempting camera restart...")
        
        # Stop any active streaming
        was_streaming = camera_stream_active
        camera_stream_active = False
        time.sleep(0.5)  # Wait for stream to stop
        
        # Force cleanup existing camera
        if camera:
            try:
                camera.stop()
                camera.close()
                debug_log("Old camera instance closed")
            except Exception as e:
                debug_log(f"Error closing old camera: {e}")
            camera = None
        
        # Force garbage collection
        import gc
        gc.collect()
        time.sleep(0.5)
        
        # Reinitialize camera
        success, message = init_camera_stream()
        
        if success:
            # Restore streaming if it was active
            if was_streaming:
                camera_stream_active = True
                debug_log("Camera restarted successfully, streaming resumed")
            else:
                debug_log("Camera restarted successfully")
                
            return jsonify({
                'success': True, 
                'message': 'Camera restarted successfully',
                'streaming_resumed': was_streaming
            })
        else:
            debug_log(f"Camera restart failed: {message}")
            return jsonify({'success': False, 'error': f'Camera restart failed: {message}'})
            
    except Exception as e:
        debug_log(f"Camera restart error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/robot/debug/ping')
def debug_ping():
    """Simple ping test to check if server is responsive"""
    debug_log("PING: Server is responsive")
    return jsonify({
        'success': True, 
        'message': 'Server is responsive',
        'robot_mode_active': robot_mode_active,
        'camera_stream_active': camera_stream_active,
        'timestamp': time.time()
    })

@app.route('/robot/debug/hardware_status')
def debug_hardware_status():
    """Detailed hardware status for debugging"""
    status = {
        'timestamp': time.time(),
        'robot_mode_active': robot_mode_active,
        'camera_stream_active': camera_stream_active,
        'audio_stream_active': audio_stream_active,
        'hardware_objects': {
            'camera': camera is not None,
            'servo_controller': servo_controller is not None,
            'chassis_controller': chassis_controller is not None
        },
        'hardware_availability': {
            'robot': ROBOT_HARDWARE_AVAILABLE,
            'camera': CAMERA_AVAILABLE,
            'audio': AUDIO_AVAILABLE
        }
    }
    
    detailed_hardware_debug("STATUS_CHECK", **status)
    return jsonify(status)

@app.route('/robot/debug/servo_test')
def debug_servo_test():
    """Debug endpoint to test servo functionality"""
    debug_log("Servo test endpoint called")
    
    if not robot_mode_active:
        debug_log("Robot mode not active")
        return jsonify({'success': False, 'error': 'Robot mode not active'})
    
    if not servo_controller:
        debug_log("Servo controller not initialized")
        return jsonify({'success': False, 'error': 'Servo controller not initialized'})
    
    try:
        debug_log("Starting servo functionality test...")
        
        # Test both servos
        results = []
        
        # Test Pan (servo 0)
        try:
            debug_log("Testing pan servo...")
            servo_controller.set_servo_pwm('0', 90)
            time.sleep(0.5)
            results.append("Pan servo (0): OK")
            debug_log("Pan servo test passed")
        except Exception as e:
            error_msg = f"Pan servo (0): ERROR - {str(e)}"
            results.append(error_msg)
            debug_log(error_msg)
        
        # Test Tilt (servo 1)  
        try:
            debug_log("Testing tilt servo...")
            servo_controller.set_servo_pwm('1', 60)
            time.sleep(0.5)
            results.append("Tilt servo (1): OK")
            debug_log("Tilt servo test passed")
        except Exception as e:
            error_msg = f"Tilt servo (1): ERROR - {str(e)}"
            results.append(error_msg)
            debug_log(error_msg)
        
        debug_log("Servo test completed successfully")
        
        response = {
            'success': True, 
            'results': results,
            'camera_stream_active': camera_stream_active,
            'servo_controller_available': servo_controller is not None
        }
        
        debug_log(f"Returning response: {response}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Servo test failed: {str(e)}"
        debug_log(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/robot/camera/force/<action>')
def robot_camera_force_control(action):
    """Force servo control without any camera coordination - for testing"""
    if not robot_mode_active:
        return jsonify({'success': False, 'error': 'Robot mode not active'})
    
    if not servo_controller:
        return jsonify({'success': False, 'error': 'Servo controller not initialized'})
    
    debug_log(f"FORCE Camera control command: {action} (bypassing camera coordination)")
    
    try:
        if action == 'pan':
            angle = int(request.args.get('angle', 90))
            original_angle = angle
            angle = max(13, min(154, angle))  # Clamp to servo limits
            debug_log(f"FORCE Panning camera to {angle}° (requested: {original_angle}°)")
            servo_controller.set_servo_pwm('0', angle)
            time.sleep(0.3)
            result = f"FORCE: Camera panned to {angle}°"
        elif action == 'tilt':
            angle = int(request.args.get('angle', 60))
            original_angle = angle
            angle = max(36, min(85, angle))  # Clamp to servo limits
            debug_log(f"FORCE Tilting camera to {angle}° (requested: {original_angle}°)")
            servo_controller.set_servo_pwm('1', angle)
            time.sleep(0.3)
            result = f"FORCE: Camera tilted to {angle}°"
        elif action == 'center':
            debug_log("FORCE Centering camera (Pan: 90°, Tilt: 60°)")
            servo_controller.set_servo_pwm('0', 90)  # Pan center
            time.sleep(0.2)
            servo_controller.set_servo_pwm('1', 60)  # Tilt slightly up
            time.sleep(0.3)
            result = "FORCE: Camera centered"
        else:
            return jsonify({'success': False, 'error': 'Invalid camera action'})
        
        debug_log(f"FORCE Camera control completed: {result}")
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        debug_log(f"FORCE Camera control error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

# ============================================================================
# WEBSOCKET EVENTS (existing + new)
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    session_id = request.sid
    if session_id in log_active:
        log_active[session_id] = False
    print(f"Client disconnected: {session_id}")

@socketio.on('start_log_stream')
def handle_start_log_stream(data):
    """Start streaming logs"""
    session_id = request.sid
    log_type = data.get('log_type', 'live')
    
    # Stop any existing stream for this session
    if session_id in log_active:
        log_active[session_id] = False
    
    # Start new stream
    log_active[session_id] = True
    
    commands = {
        'live': 'sudo journalctl -u vocalgem -f',
        'recent': 'sudo journalctl -u vocalgem -n 100 --no-pager',
        'today': 'sudo journalctl -u vocalgem --since today --no-pager'
    }
    
    command = commands.get(log_type, commands['live'])
    
    # Start streaming in a separate thread
    thread = threading.Thread(target=stream_logs, args=(session_id, command))
    log_threads[session_id] = thread
    thread.start()

@socketio.on('stop_log_stream')
def handle_stop_log_stream():
    """Stop streaming logs"""
    session_id = request.sid
    if session_id in log_active:
        log_active[session_id] = False

# New robot control WebSocket events
@socketio.on('robot_move_continuous')
def handle_robot_move_continuous(data):
    """Handle continuous movement commands with improved hardware-sharing"""
    if not robot_mode_active:
        emit('robot_error', {'error': 'Robot mode not active'})
        return
    
    if not chassis_controller:
        emit('robot_error', {'error': 'Chassis controller not initialized'})
        return
    
    def execute_continuous_movement(direction, speed, max_retries=2):
        """Execute continuous movement with retry for hardware contention"""
        for attempt in range(max_retries):
            try:
                debug_log(f"Continuous movement attempt {attempt + 1}: {direction} at speed {speed}")
                
                # Access the motor controller through the chassis controller
                if direction == 'start_forward':
                    chassis_controller.pwm.set_motor_model(speed, speed, speed, speed)
                elif direction == 'start_backward':
                    chassis_controller.pwm.set_motor_model(-speed, -speed, -speed, -speed)
                elif direction == 'start_left':
                    # For strafe left: left wheels backward, right wheels forward
                    chassis_controller.pwm.set_motor_model(-speed, -speed, speed, speed)
                elif direction == 'start_right':
                    # For strafe right: left wheels forward, right wheels backward
                    chassis_controller.pwm.set_motor_model(speed, speed, -speed, -speed)
                elif direction == 'stop':
                    chassis_controller.stop_motors()
                else:
                    return False, f"Invalid direction: {direction}"
                
                debug_log(f"Continuous movement successful on attempt {attempt + 1}: {direction}")
                return True, f"Movement {direction} executed"
                
            except Exception as e:
                debug_log(f"Continuous movement attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Brief pause before retry
                else:
                    return False, f"Movement failed after {max_retries} attempts: {e}"
        
        return False, "Movement failed unexpectedly"
    
    try:
        direction = data.get('direction')
        speed = data.get('speed', 1000)
        
        debug_log(f"Continuous movement: {direction} at speed {speed} - Video active: {camera_stream_active}")
        
        success, message = execute_continuous_movement(direction, speed)
        
        if success:
            debug_log(f"✅ Continuous movement command executed: {direction}")
            emit('robot_move_response', {'success': True, 'direction': direction})
        else:
            debug_log(f"❌ Continuous movement failed: {message}")
            emit('robot_error', {'error': message})
        
    except Exception as e:
        error_msg = f"Continuous movement error: {str(e)}"
        debug_log(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        emit('robot_error', {'error': error_msg})

@socketio.on('robot_move_discrete')
def handle_robot_move_discrete(data):
    """Handle discrete movement commands via WebSocket to bypass HTTP limits"""
    if not robot_mode_active:
        emit('robot_error', {'error': 'Robot mode not active'})
        return
    
    if not chassis_controller:
        emit('robot_error', {'error': 'Chassis controller not initialized'})
        return
    
    try:
        action = data.get('action')
        speed = data.get('speed', 1000)
        distance = data.get('distance', 20)
        angle = data.get('angle', 90)
        
        detailed_hardware_debug("WEBSOCKET_MOVEMENT_START", 
                              action=action, 
                              speed=speed, 
                              distance=distance,
                              angle=angle,
                              request_time=time.time())
        
        success = False
        result_message = ""
        
        if action == 'forward':
            success = chassis_controller.move_forward_distance(distance, speed)
            result_message = f"Moved forward {distance}cm"
        elif action == 'backward':
            success = chassis_controller.move_forward_distance(-distance, -speed)
            result_message = f"Moved backward {distance}cm"
        elif action == 'rotate_left':
            success = chassis_controller.rotate_angle(-angle, speed)
            result_message = f"Rotated left {angle}°"
        elif action == 'rotate_right':
            success = chassis_controller.rotate_angle(angle, speed)
            result_message = f"Rotated right {angle}°"
        elif action == 'stop':
            success = chassis_controller.emergency_stop()
            result_message = "Emergency stop executed"
        else:
            emit('robot_error', {'error': f'Invalid action: {action}'})
            return
        
        if success:
            detailed_hardware_debug("WEBSOCKET_MOVEMENT_SUCCESS", 
                                  action=action,
                                  result=result_message)
            emit('robot_move_response', {
                'success': True, 
                'action': action,
                'message': result_message
            })
        else:
            detailed_hardware_debug("WEBSOCKET_MOVEMENT_FAILED", 
                                  action=action,
                                  error="Movement execution failed")
            emit('robot_error', {'error': f'Movement failed: {action}'})
        
    except Exception as e:
        error_msg = f"WebSocket movement error: {str(e)}"
        detailed_hardware_debug("WEBSOCKET_MOVEMENT_ERROR", 
                              error=error_msg)
        import traceback
        traceback.print_exc()
        emit('robot_error', {'error': error_msg})

@socketio.on('robot_camera_control')
def handle_robot_camera_control(data):
    """Handle camera servo control via WebSocket to bypass HTTP limits"""
    if not robot_mode_active:
        emit('robot_error', {'error': 'Robot mode not active'})
        return
    
    if not servo_controller:
        emit('robot_error', {'error': 'Servo controller not initialized'})
        return
    
    try:
        action = data.get('action')
        angle = data.get('angle', 90)
        
        detailed_hardware_debug("WEBSOCKET_SERVO_START", 
                              action=action,
                              angle=angle,
                              request_time=time.time())
        
        success = False
        result_message = ""
        
        if action == 'pan':
            angle = max(13, min(154, angle))  # Clamp to servo limits
            servo_controller.set_servo_pwm('0', angle)
            success = True
            result_message = f"Camera panned to {angle}°"
        elif action == 'tilt':
            angle = max(36, min(85, angle))  # Clamp to servo limits
            servo_controller.set_servo_pwm('1', angle)
            success = True
            result_message = f"Camera tilted to {angle}°"
        elif action == 'center':
            servo_controller.set_servo_pwm('0', 90)  # Pan center
            time.sleep(0.1)
            servo_controller.set_servo_pwm('1', 60)  # Tilt slightly up
            success = True
            result_message = "Camera centered"
        else:
            emit('robot_error', {'error': f'Invalid camera action: {action}'})
            return
        
        if success:
            detailed_hardware_debug("WEBSOCKET_SERVO_SUCCESS", 
                                  action=action,
                                  angle=angle,
                                  result=result_message)
            emit('robot_camera_response', {
                'success': True,
                'action': action,
                'angle': angle,
                'message': result_message
            })
        else:
            detailed_hardware_debug("WEBSOCKET_SERVO_FAILED", 
                                  action=action,
                                  error="Servo execution failed")
            emit('robot_error', {'error': f'Camera control failed: {action}'})
        
    except Exception as e:
        error_msg = f"WebSocket camera control error: {str(e)}"
        detailed_hardware_debug("WEBSOCKET_SERVO_ERROR", 
                              error=error_msg)
        import traceback
        traceback.print_exc()
        emit('robot_error', {'error': error_msg})

if __name__ == '__main__':
    # Check if running with correct permissions
    if os.geteuid() != 0:
        print("⚠️  Warning: Running without sudo. Some commands may fail.")
        print("For full functionality, run: sudo python3 web_ui.py")
    
    print("🚀 Starting VocalGem Web UI...")
    print("🌐 Access the interface at:")
    print("   - Local: http://localhost:5000")
    print("   - Network: http://[your-pi-ip]:5000")
    print("   - Find Pi IP: hostname -I | cut -d' ' -f1")
    
    # Check hardware availability
    print(f"🤖 Robot Hardware: {'✅ Available' if ROBOT_HARDWARE_AVAILABLE else '❌ Not Available'}")
    print(f"📹 Camera: {'✅ Available' if CAMERA_AVAILABLE else '❌ Not Available'}")
    print(f"🎵 Audio: {'✅ Available' if AUDIO_AVAILABLE else '❌ Not Available'}")
    
    print("\n📋 Debug Information:")
    print("   - Web UI logs: sudo journalctl -u vocalgem-webui -f")
    print("   - Python output: Check terminal or systemd logs")
    print("   - Use debug buttons in Robot Status panel to test functionality")
    
    # Run on all interfaces to allow network access
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True) 