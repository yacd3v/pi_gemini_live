#!/usr/bin/env python3
"""
Camera module for robot control dashboard
Handles IMX500 camera initialization, frame capture, and streaming
"""

import time
import threading
import cv2
import numpy as np
from collections import deque
from picamera2 import Picamera2
from picamera2.devices import IMX500
from libcamera import Transform

class CameraManager:
    """Manages camera operations including initialization, capture, and streaming"""
    
    def __init__(self, config):
        self.config = config
        self.camera = None
        self.frame_buffer = deque(maxlen=config['frame_buffer_size'])
        self.frame_lock = threading.Lock()
        self.frame_stats = {'dropped': 0, 'served': 0, 'avg_age': 0}
        self.is_running = False
        
    def initialize(self):
        """Initialize the IMX500 camera with ultra-low latency settings"""
        try:
            # Initialize IMX500 
            imx500 = IMX500()
            self.camera = Picamera2(imx500.camera_num)
            print(f"📷 Camera initialized with IMX500 on camera {imx500.camera_num}")
            
            # Configure camera for minimal latency - use MJPEG directly
            config = self.camera.create_video_configuration(
                main={"size": self.config['resolution'], "format": "XRGB8888"},
                buffer_count=1,  # Absolute minimum buffering
                queue=False      # No frame queuing
            )
            
            # Set transform if supported
            try:
                config["transform"] = Transform()
            except Exception as e:
                print(f"Could not set transform: {e}")
            
            self.camera.configure(config)
            self.camera.start()
            print(f"✓ Camera started at {self.config['resolution']} (ultra-low latency mode)")
            return True
            
        except Exception as e:
            print(f"Error initializing IMX500 camera: {e}")
            # Fallback to regular camera with minimal latency config
            try:
                self.camera = Picamera2()
                config = self.camera.create_video_configuration(
                    main={"size": self.config['resolution'], "format": "XRGB8888"},
                    buffer_count=1,
                    queue=False
                )
                self.camera.configure(config)
                self.camera.start()
                print("✓ Fallback to regular camera (ultra-low latency mode)")
                return True
            except Exception as fallback_e:
                print(f"Fallback camera initialization also failed: {fallback_e}")
                return False
    
    def start_capture_thread(self):
        """Start the frame capture thread"""
        if not self.camera:
            print("❌ Camera not initialized")
            return False
            
        self.is_running = True
        capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        capture_thread.start()
        print("✓ Camera capture thread started")
        return True
    
    def _capture_frames(self):
        """Ultra-low latency frame capture with minimal processing"""
        frame_interval = 1.0 / self.config['target_fps']
        last_capture = 0
        frame_count = 0
        
        while self.is_running:
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
                frame = self.camera.capture_array()
                capture_timestamp = time.time()
                
                # Skip all OpenCV processing if overlays disabled
                if self.config.get('skip_overlays', False):
                    # Direct encoding without any processing
                    # Convert XRGB to RGB for JPEG encoding
                    if frame.shape[2] == 4:  # XRGB format
                        frame_rgb = frame[:, :, :3]  # Drop alpha channel
                    else:
                        frame_rgb = frame
                    
                    # Ultra-fast JPEG encoding
                    encode_params = [
                        cv2.IMWRITE_JPEG_QUALITY, self.config['jpeg_quality'],
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
                    
                    # Fast encoding
                    encode_params = [
                        cv2.IMWRITE_JPEG_QUALITY, self.config['jpeg_quality'],
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
                    with self.frame_lock:
                        # Clear old frames and add new one
                        self.frame_buffer.clear()
                        self.frame_buffer.append(frame_data)
                        
            except Exception as e:
                print(f"Error capturing frame: {e}")
                # Don't sleep on error - continue immediately
    
    def get_latest_frame(self):
        """Get the most recent frame with minimal overhead"""
        with self.frame_lock:
            if not self.frame_buffer:
                return None
            
            # Since we only keep 1 frame now, just check age and return
            current_time = time.time()
            latest_frame = self.frame_buffer[0]
            age = current_time - latest_frame['timestamp']
            
            # Drop frame if too old
            if age > self.config['max_frame_age']:
                self.frame_buffer.clear()
                self.frame_stats['dropped'] += 1
                return None
            
            # Return frame immediately
            self.frame_stats['served'] += 1
            self.frame_stats['avg_age'] = age
            return latest_frame
    
    def generate_frames(self):
        """Ultra-low latency frame generator"""
        frame_count = 0
        
        while self.is_running:
            frame_data = self.get_latest_frame()
            
            if frame_data is None:
                continue  # Don't sleep - just continue immediately
            
            frame_count += 1
            
            # Less frequent stats logging to reduce overhead
            if frame_count % 200 == 0:
                print(f"Stream stats - Served: {self.frame_stats['served']}, "
                      f"Dropped: {self.frame_stats['dropped']}, "
                      f"Avg age: {self.frame_stats['avg_age']*1000:.1f}ms")
            
            # Simplified multipart frame with minimal headers
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_data['data'])).encode() + b'\r\n'
                   b'\r\n' + frame_data['data'] + b'\r\n')
    
    def wait_for_first_frame(self, timeout=10):
        """Wait for the first frame to be captured"""
        start_wait = time.time()
        while len(self.frame_buffer) == 0:
            time.sleep(0.01)
            if time.time() - start_wait > timeout:
                print("❌ Timeout waiting for first frame")
                return False
        return True
    
    def get_stats(self):
        """Get camera statistics"""
        return {
            'frame_stats': self.frame_stats.copy(),
            'buffer_size': len(self.frame_buffer),
            'is_running': self.is_running
        }
    
    def toggle_overlays(self):
        """Toggle overlay processing for debugging latency"""
        self.config['skip_overlays'] = not self.config.get('skip_overlays', True)
        mode = "disabled" if self.config['skip_overlays'] else "enabled"
        print(f"🎥 Overlays {mode} (lower latency when disabled)")
        return not self.config['skip_overlays']
    
    def cleanup(self):
        """Cleanup camera resources"""
        self.is_running = False
        if self.camera:
            self.camera.stop()
            self.camera.close()
        print("✓ Camera cleanup completed") 