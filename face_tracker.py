"""
Face Tracking Module for VocalGem Robot
Handles IMX500 face detection and servo tracking functionality.
"""

import os
import sys
import time
import numpy as np
from picamera2.devices import IMX500
from libcamera import Transform

# Add path for servo control
sys.path.append("freenove_examples")
try:
    from servo import Servo
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False
    print("Warning: Servo module not found. Face tracking will be disabled.")

class FaceTracker:
    """Handles face detection and servo tracking using IMX500 AI camera."""
    
    def __init__(self, enable_tracking=True, confidence_threshold=0.5):
        """
        Initialize face tracker.
        
        Args:
            enable_tracking (bool): Enable automatic face tracking
            confidence_threshold (float): Minimum confidence for face detection
        """
        self.enable_face_tracking = enable_tracking
        self.face_confidence_threshold = confidence_threshold
        
        # Tracking timing
        self.last_manual_movement_time = 0
        self.manual_movement_cooldown = 5.0  # Seconds to wait after manual movement
        self.face_tracking_interval = 0.5  # Detection interval
        self.last_face_tracking_time = 0
        
        # Tracking parameters - match test_autoaim.py
        self.face_history = []
        self.face_history_size = 3
        self.tracking_deadzone = 0.02  # Match test_autoaim.py deadzone
        self.tracking_speed = 0.6  # Match test_autoaim.py tracking speed
        self.max_move_per_step = 20  # Match test_autoaim.py max movement per step
        self.no_target_timeout = 5.0  # Return to center timeout
        self.last_target_time = time.time()
        
        # Servo limits - match test_autoaim.py exactly
        self.pan_limits = (20, 160)  # Match test_autoaim.py pan limits
        self.tilt_limits = (30, 100)  # Match test_autoaim.py tilt limits
        
        # Servo state
        self.current_pan_angle = 90  # Start at center pan
        self.current_tilt_angle = 50  # Start at 50° to look up from floor level
        
        # Initialize servo
        self.servo = None
        if SERVO_AVAILABLE and enable_tracking:
            self._init_servo()
        
        # IMX500 face detection setup
        self.imx500 = None
        self.face_detection_enabled = False
        self._init_face_detection()
    
    def _init_servo(self):
        """Initialize servo control for tracking"""
        try:
            self.servo = Servo()
            # Move to starting position (center pan, 50° tilt to look up)
            self.servo.set_servo_pwm('0', self.current_pan_angle)
            self.servo.set_servo_pwm('1', self.current_tilt_angle)
            time.sleep(0.5)  # Give servos time to move
            print(f"[FaceTracker] Servo tracking initialized - Pan: {self.current_pan_angle}°, Tilt: {self.current_tilt_angle}°")
            print(f"[FaceTracker] Pan limits: {self.pan_limits}, Tilt limits: {self.tilt_limits}")
        except Exception as e:
            print(f"[FaceTracker] Failed to initialize servo: {e}")
            self.servo = None
            self.enable_face_tracking = False
    
    def _init_face_detection(self):
        """Initialize IMX500 face detection"""
        try:
            model_file = "/usr/share/imx500-models/imx500_network_posenet.rpk"
            if os.path.exists(model_file):
                self.imx500 = IMX500(model_file)
                self.face_detection_enabled = True
                print("[FaceTracker] IMX500 face detection initialized.")
                print(f"[FaceTracker] Face tracking enabled: {self.enable_face_tracking}")
            else:
                print("[FaceTracker] IMX500 model not found. Face tracking disabled.")
        except Exception as e:
            print(f"[FaceTracker] Error initializing IMX500: {e}. Face tracking disabled.")
    
    def parse_face_detection(self, metadata):
        """Parse face detection results from IMX500 metadata"""
        detections = []
        
        if not self.face_detection_enabled or not self.imx500:
            return detections
            
        try:
            outputs = self.imx500.get_outputs(metadata)
            if outputs is None or len(outputs) == 0:
                return detections
            
            heatmaps = outputs[0]
            
            # Handle different heatmap formats
            if len(heatmaps.shape) == 3:
                if heatmaps.shape[2] >= 5:  # (H, W, keypoints)
                    h, w, num_keypoints = heatmaps.shape
                elif heatmaps.shape[0] >= 5:  # (keypoints, H, W)
                    num_keypoints, h, w = heatmaps.shape
                    heatmaps = np.transpose(heatmaps, (1, 2, 0))
                else:
                    return detections
            else:
                return detections
            
            # Define facial keypoint indices (COCO format)
            facial_keypoints = {
                'nose': 0,
                'left_eye': 1,
                'right_eye': 2,
                'left_ear': 3,
                'right_ear': 4
            }
            
            # Find facial keypoints
            face_keypoints = {}
            face_confidences = {}
            
            for name, idx in facial_keypoints.items():
                if idx < num_keypoints:
                    keypoint_heatmap = heatmaps[:, :, idx]
                    max_confidence = np.max(keypoint_heatmap)
                    
                    if max_confidence > self.face_confidence_threshold:
                        y_idx, x_idx = np.unravel_index(np.argmax(keypoint_heatmap), keypoint_heatmap.shape)
                        x_norm = x_idx / w
                        y_norm = y_idx / h
                        face_keypoints[name] = (x_norm, y_norm)
                        face_confidences[name] = max_confidence
            
            # If we found enough facial keypoints, create a detection
            if len(face_keypoints) >= 2:
                x_coords = [pos[0] for pos in face_keypoints.values()]
                y_coords = [pos[1] for pos in face_keypoints.values()]
                
                # Calculate face center
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                avg_confidence = np.mean(list(face_confidences.values()))
                
                detections.append({
                    "center_x": center_x,
                    "center_y": center_y,
                    "confidence": float(avg_confidence),
                    "keypoints": face_keypoints
                })
                        
        except Exception as e:
            # Silently handle detection errors to avoid spamming logs
            pass
        
        return detections
    
    def should_auto_track(self):
        """Check if automatic face tracking should be active"""
        if not self.enable_face_tracking:
            return False
            
        if not self.face_detection_enabled:
            return False
            
        # Don't track if we recently had manual movement
        time_since_manual = time.time() - self.last_manual_movement_time
        if time_since_manual < self.manual_movement_cooldown:
            return False
            
        # Check if enough time has passed since last tracking
        time_since_tracking = time.time() - self.last_face_tracking_time
        if time_since_tracking < self.face_tracking_interval:
            return False
            
        return True
    
    def track_face(self, face_center_x, face_center_y):
        """Perform smooth face tracking to center the face - using test_autoaim.py algorithm"""
        if not self.servo:
            return False
            
        current_time = time.time()
        
        # Convert normalized coordinates to pixel coordinates for error calculation
        frame_width = 640  # Camera frame width
        frame_height = 480  # Camera frame height
        
        target_center_x = face_center_x * frame_width
        target_center_y = face_center_y * frame_height
        
        # Calculate error from image center (normalized -1 to 1) - like test_autoaim.py
        image_center_x = frame_width / 2
        image_center_y = frame_height / 2
        
        error_x = (target_center_x - image_center_x) / (frame_width / 2)
        error_y = (target_center_y - image_center_y) / (frame_height / 2)
        
        # Add to target history for smoothing - like test_autoaim.py
        self.face_history.append((error_x, error_y))
        if len(self.face_history) > self.face_history_size:
            self.face_history.pop(0)
        
        # Calculate smoothed error (weighted average of recent positions) - like test_autoaim.py
        if len(self.face_history) >= 2:
            # Weight recent positions more heavily
            weights = [1.0, 1.5, 2.0][:len(self.face_history)]
            total_weight = sum(weights)
            
            smoothed_error_x = sum(w * pos[0] for w, pos in zip(weights, self.face_history)) / total_weight
            smoothed_error_y = sum(w * pos[1] for w, pos in zip(weights, self.face_history)) / total_weight
        else:
            smoothed_error_x = error_x
            smoothed_error_y = error_y
        
        # Apply deadzone to smoothed error - like test_autoaim.py
        if abs(smoothed_error_x) < self.tracking_deadzone:
            smoothed_error_x = 0
        if abs(smoothed_error_y) < self.tracking_deadzone:
            smoothed_error_y = 0
        
        # Update servo positions if there's significant error - like test_autoaim.py
        if smoothed_error_x != 0 or smoothed_error_y != 0:
            self._move_servos_smooth(smoothed_error_x, smoothed_error_y)
            self.last_target_time = current_time
            return True
        else:
            # No target detected - return to center after timeout
            if current_time - self.last_target_time > self.no_target_timeout:
                self._return_to_center()
            return False

    def _move_servos_smooth(self, error_x, error_y):
        """Move servos with smoothing - using test_autoaim.py algorithm"""
        # Calculate direct servo positions based on error - like test_autoaim.py
        pan_range = self.pan_limits[1] - self.pan_limits[0]
        pan_center = (self.pan_limits[0] + self.pan_limits[1]) / 2
        
        tilt_range = self.tilt_limits[1] - self.tilt_limits[0]
        tilt_center = (self.tilt_limits[0] + self.tilt_limits[1]) / 2
        
        # Calculate target servo positions - like test_autoaim.py
        pan_offset = error_x * (pan_range / 2) * self.tracking_speed
        tilt_offset = error_y * (tilt_range / 2) * self.tracking_speed
        
        target_pan = pan_center + pan_offset  # Match test_autoaim.py direction
        target_tilt = tilt_center + tilt_offset  # Match test_autoaim.py direction
        
        # Apply safety limits
        target_pan = max(self.pan_limits[0], min(self.pan_limits[1], target_pan))
        target_tilt = max(self.tilt_limits[0], min(self.tilt_limits[1], target_tilt))
        
        # Limit movement speed to prevent big jumps - like test_autoaim.py
        pan_diff = target_pan - self.current_pan_angle
        tilt_diff = target_tilt - self.current_tilt_angle
        
        # Clamp movement to max_move_per_step
        if abs(pan_diff) > self.max_move_per_step:
            pan_diff = self.max_move_per_step * (1 if pan_diff > 0 else -1)
        if abs(tilt_diff) > self.max_move_per_step:
            tilt_diff = self.max_move_per_step * (1 if tilt_diff > 0 else -1)
        
        new_pan = self.current_pan_angle + pan_diff
        new_tilt = self.current_tilt_angle + tilt_diff
        
        # Move if there's significant change - like test_autoaim.py
        if abs(new_pan - self.current_pan_angle) > 0.5 or abs(new_tilt - self.current_tilt_angle) > 0.5:
            self.current_pan_angle = new_pan
            self.current_tilt_angle = new_tilt
            
            # Move servos to new position
            self.servo.set_servo_pwm('0', int(self.current_pan_angle))
            self.servo.set_servo_pwm('1', int(self.current_tilt_angle))
            
            print(f"[FaceTracker] Servo moved: Pan={self.current_pan_angle:.1f}° (Δ{pan_diff:+.1f}°), Tilt={self.current_tilt_angle:.1f}° (Δ{tilt_diff:+.1f}°)")
            return True
            
        return False

    def _return_to_center(self):
        """Return servos to center position when no target is detected - like test_autoaim.py"""
        target_pan = 90  # Center pan position
        target_tilt = 50  # Center tilt position (looking up for floor-mounted camera)
        
        # Move gradually towards center
        pan_diff = target_pan - self.current_pan_angle
        tilt_diff = target_tilt - self.current_tilt_angle
        
        # Small movements towards center
        if abs(pan_diff) > 1.0:
            self.current_pan_angle += np.sign(pan_diff) * min(1.0, abs(pan_diff) * 0.1)
        else:
            self.current_pan_angle = target_pan
            
        if abs(tilt_diff) > 1.0:
            self.current_tilt_angle += np.sign(tilt_diff) * min(1.0, abs(tilt_diff) * 0.1)
        else:
            self.current_tilt_angle = target_tilt
        
        # Move servos
        self.servo.set_servo_pwm('0', int(self.current_pan_angle))
        self.servo.set_servo_pwm('1', int(self.current_tilt_angle))
        
        print(f"[FaceTracker] Returning to center: Pan={self.current_pan_angle:.1f}°, Tilt={self.current_tilt_angle:.1f}°")
    
    def manual_movement_occurred(self):
        """Call this when manual camera movement occurs to pause auto-tracking"""
        self.last_manual_movement_time = time.time()
        self.face_history.clear()  # Clear face tracking history after manual movement
    
    def toggle_tracking(self, enabled):
        """Enable or disable face tracking"""
        if not self.face_detection_enabled:
            return "Face tracking is not available. IMX500 model not found or initialization failed."
        
        self.enable_face_tracking = enabled
        if enabled:
            self.face_history.clear()  # Clear history when enabling
            return "Face tracking enabled. Karl will now automatically follow detected faces."
        else:
            return "Face tracking disabled. Karl will only move the camera when asked." 