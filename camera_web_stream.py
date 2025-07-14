#!/usr/bin/env python3
"""
Simple web video stream for IMX500 camera on Raspberry Pi 5
Access via: http://raspberry_pi_ip:5000
"""

import time
import io
import threading
from flask import Flask, render_template, Response
from picamera2 import Picamera2
from picamera2.devices import IMX500
from libcamera import Transform
import cv2
import numpy as np

app = Flask(__name__)

# Global variables for camera and streaming
camera = None
output_frame = None
lock = threading.Lock()

def initialize_camera():
    """Initialize the IMX500 camera with the same configuration as the main script"""
    global camera
    
    try:
        # Initialize IMX500 
        imx500 = IMX500()
        camera = Picamera2(imx500.camera_num)
        print(f"Camera initialized with IMX500 on camera {imx500.camera_num}")
        
        # Configure camera - similar to main script
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
        
        # Set up network intrinsics for face detection (optional)
        if imx500.network_intrinsics:
            ni = imx500.network_intrinsics
            ni.task = "pose estimation"
            ni.inference_rate = 30.0
            print("IMX500 network intrinsics configured")
            
        camera.start()
        print("Camera started successfully")
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
            print("Fallback to regular camera successful")
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
            
            # Convert RGB to BGR for OpenCV (if needed)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add timestamp overlay
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Add camera info overlay
            cam_info = "IMX500 Camera Stream"
            cv2.putText(frame, cam_info, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if ret:
                # Update global frame with thread safety
                with lock:
                    output_frame = buffer.tobytes()
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            time.sleep(0.1)
        
        # Small delay to control frame rate (~30 FPS)
        time.sleep(1/30)

def generate_frames():
    """Generate frames for streaming"""
    global output_frame, lock
    
    while True:
        # Wait for frame to be available
        while output_frame is None:
            time.sleep(0.01)
        
        with lock:
            frame = output_frame
        
        # Yield frame in multipart format for MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Main page with video stream"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>IMX500 Camera Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            text-align: center;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .video-container {
            border: 2px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            margin: 20px auto;
            display: inline-block;
        }
        img {
            display: block;
            max-width: 100%;
            height: auto;
        }
        .info {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-top: 20px;
            text-align: left;
        }
        .status {
            color: #28a745;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Raspberry Pi 5 - IMX500 Camera Stream</h1>
        
        <div class="video-container">
            <img src="/video_feed" alt="Camera Stream">
        </div>
        
        <div class="info">
            <h3>📊 Stream Information</h3>
            <ul>
                <li><span class="status">Status:</span> Live Stream Active</li>
                <li><span class="status">Resolution:</span> 640x480</li>
                <li><span class="status">Camera:</span> IMX500 AI Camera</li>
                <li><span class="status">Format:</span> MJPEG Stream</li>
            </ul>
        </div>
        
        <div class="info">
            <h3>🔗 Access Information</h3>
            <p>This stream can be accessed from any device on your local network.</p>
            <p>Use your Raspberry Pi's IP address: <code>http://raspberry_pi_ip:5000</code></p>
        </div>
    </div>
</body>
</html>
'''

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Simple status endpoint"""
    return {
        'status': 'running',
        'camera': 'IMX500' if camera else 'not initialized',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

def main():
    """Main function to start the web server"""
    print("=" * 60)
    print("🤖 IMX500 Camera Web Stream Server")
    print("=" * 60)
    
    # Initialize camera
    print("📷 Initializing camera...")
    if not initialize_camera():
        print("❌ Failed to initialize camera. Exiting.")
        return
    
    # Start frame capture thread
    print("🎬 Starting frame capture thread...")
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    
    # Wait a moment for first frame
    print("⏳ Waiting for first frame...")
    while output_frame is None:
        time.sleep(0.1)
    
    print("✅ Camera stream ready!")
    print(f"🌐 Access the stream at: http://localhost:5000")
    print("📱 Or from other devices: http://YOUR_PI_IP:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start Flask server
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