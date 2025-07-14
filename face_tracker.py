"""
Face Tracking Module for VocalGem Robot
Handles IMX500 face detection and servo tracking functionality.
"""

import os
import sys
import time
import math
import numpy as np
from collections import deque
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
        self.manual_movement_cooldown = 3.0  # Reduced from 5.0 for faster recovery
        self.face_tracking_interval = 0.2  # Reduced from 0.5 for more responsive tracking
        self.last_face_tracking_time = 0
        
        # Enhanced tracking parameters
        self.face_history = deque(maxlen=5)  # Use deque for better performance
        self.face_history_size = 5  # Increased from 3 for better smoothing
        
        # Multi-level tracking parameters - BALANCED RESPONSIVENESS
        self.tracking_deadzone = 0.05  # Smaller deadzone for better centering (was 0.15)
        self.tracking_speed_base = 0.6  # Faster base speed for better response
        self.tracking_speed_close = 0.3  # Moderate speed when close to target
        self.tracking_speed_far = 0.8   # Faster speed when far from target
        
        # Movement limiting - BETTER RESPONSIVENESS
        self.max_move_per_step = 8  # Larger movements for better response (was 3)
        self.min_move_threshold = 0.5  # Lower threshold for more responsive tracking
        
        # Target management - DISABLE RETURN TO CENTER
        self.no_target_timeout = 300.0  # Very long timeout (5 minutes) - effectively disabled
        self.last_target_time = time.time()
        
        # AGGRESSIVE servo limits for maximum tracking range
        self.pan_limits = (0, 180)   # Full 180° pan range (0° = far left, 180° = far right)
        self.tilt_limits = (5, 115)  # Aggressive up/down range (5° = looking up, 115° = looking down)
        
        # Servo state with velocity tracking
        self.current_pan_angle = 90  # Start at center pan
        self.current_tilt_angle = 50  # Start at 50° to look up from floor level
        self.last_pan_angle = 90
        self.last_tilt_angle = 50
        self.pan_velocity = 0.0  # Track servo velocity for smoothing
        self.tilt_velocity = 0.0
        
        # Enhanced smoothing parameters
        self.velocity_decay = 0.8  # Velocity decay factor
        self.position_smoothing = 0.3  # Position smoothing factor
        
        # Debug and monitoring
        self.debug_enabled = True
        self.movement_history = deque(maxlen=20)  # Track recent movements
        self.tracking_stats = {
            'total_detections': 0,
            'successful_tracks': 0,
            'movements_executed': 0,
            'avg_confidence': 0.0
        }
        
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
            print(f"[FaceTracker] Enhanced smoothing enabled - Deadzone: {self.tracking_deadzone}, History: {self.face_history_size}")
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
        """Parse face detection results from IMX500 metadata with enhanced processing"""
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
            
            # Enhanced keypoint detection with multiple confidence levels
            face_keypoints = {}
            face_confidences = {}
            
            for name, idx in facial_keypoints.items():
                if idx < num_keypoints:
                    keypoint_heatmap = heatmaps[:, :, idx]
                    max_confidence = np.max(keypoint_heatmap)
                    
                    # Use dynamic threshold based on keypoint type
                    threshold = self.face_confidence_threshold
                    if name in ['nose', 'left_eye', 'right_eye']:
                        threshold *= 0.8  # More lenient for core facial features
                    
                    if max_confidence > threshold:
                        y_idx, x_idx = np.unravel_index(np.argmax(keypoint_heatmap), keypoint_heatmap.shape)
                        
                        # Apply gaussian smoothing around peak for sub-pixel accuracy
                        if y_idx > 0 and y_idx < h-1 and x_idx > 0 and x_idx < w-1:
                            # Get surrounding values for sub-pixel refinement
                            surrounding = keypoint_heatmap[y_idx-1:y_idx+2, x_idx-1:x_idx+2]
                            if surrounding.size > 0:
                                # Calculate weighted centroid with divide by zero protection
                                y_indices, x_indices = np.mgrid[0:3, 0:3]
                                surrounding_sum = np.sum(surrounding)
                                if surrounding_sum > 0:
                                    y_offset = np.sum(surrounding * (y_indices - 1)) / surrounding_sum
                                    x_offset = np.sum(surrounding * (x_indices - 1)) / surrounding_sum
                                else:
                                    y_offset = 0
                                    x_offset = 0
                                
                                # Apply sub-pixel correction
                                y_idx += y_offset
                                x_idx += x_offset
                        
                        x_norm = x_idx / w
                        y_norm = y_idx / h
                        face_keypoints[name] = (x_norm, y_norm)
                        face_confidences[name] = max_confidence
            
            # Enhanced face detection with geometric validation
            if len(face_keypoints) >= 2:
                # Calculate face center with weighted average
                weights = {
                    'nose': 3.0,      # Nose is most reliable
                    'left_eye': 2.0,  # Eyes are quite reliable
                    'right_eye': 2.0,
                    'left_ear': 1.0,  # Ears are less reliable
                    'right_ear': 1.0
                }
                
                total_weight = 0
                weighted_x = 0
                weighted_y = 0
                
                for name, (x, y) in face_keypoints.items():
                    weight = weights.get(name, 1.0) * face_confidences[name]
                    weighted_x += x * weight
                    weighted_y += y * weight
                    total_weight += weight
                
                if total_weight > 0:
                    center_x = weighted_x / total_weight
                    center_y = weighted_y / total_weight
                    avg_confidence = np.mean(list(face_confidences.values()))
                    
                    # Geometric validation - check if keypoints form reasonable face
                    is_valid_face = True
                    
                    # Check if eyes are roughly horizontal
                    if 'left_eye' in face_keypoints and 'right_eye' in face_keypoints:
                        left_eye = face_keypoints['left_eye']
                        right_eye = face_keypoints['right_eye']
                        eye_height_diff = abs(left_eye[1] - right_eye[1])
                        if eye_height_diff > 0.1:  # Eyes too far apart vertically
                            is_valid_face = False
                    
                    # Check if nose is roughly between eyes
                    if 'nose' in face_keypoints and 'left_eye' in face_keypoints and 'right_eye' in face_keypoints:
                        nose = face_keypoints['nose']
                        left_eye = face_keypoints['left_eye']
                        right_eye = face_keypoints['right_eye']
                        nose_x = nose[0]
                        eye_center_x = (left_eye[0] + right_eye[0]) / 2
                        if abs(nose_x - eye_center_x) > 0.15:  # Nose too far from eye center
                            is_valid_face = False
                    
                    if is_valid_face:
                        detection = {
                            "center_x": center_x,
                            "center_y": center_y,
                            "confidence": float(avg_confidence),
                            "keypoints": face_keypoints,
                            "num_keypoints": len(face_keypoints),
                            "weighted_center": True
                        }
                        detections.append(detection)
                        
                        # Update statistics
                        self.tracking_stats['total_detections'] += 1
                        self.tracking_stats['avg_confidence'] = (
                            self.tracking_stats['avg_confidence'] * 0.9 + avg_confidence * 0.1
                        )
                        
        except Exception as e:
            if self.debug_enabled:
                print(f"[FaceTracker] Detection error: {e}")
        
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
    
    def _calculate_adaptive_speed(self, error_magnitude):
        """Calculate adaptive tracking speed based on error magnitude"""
        if error_magnitude < 0.1:
            return self.tracking_speed_close
        elif error_magnitude > 0.5:
            return self.tracking_speed_far
        else:
            # Interpolate between close and far speeds
            ratio = (error_magnitude - 0.1) / 0.4
            return self.tracking_speed_close + ratio * (self.tracking_speed_far - self.tracking_speed_close)
    
    def track_face(self, face_center_x, face_center_y):
        """Enhanced face tracking with improved smoothing and adaptive speed"""
        if not self.servo:
            return False
            
        current_time = time.time()
        
        # Convert normalized coordinates to pixel coordinates for error calculation
        frame_width = 640
        frame_height = 480
        
        target_center_x = face_center_x * frame_width
        target_center_y = face_center_y * frame_height
        
        # Calculate error from image center (normalized -1 to 1)
        image_center_x = frame_width / 2
        image_center_y = frame_height / 2
        
        error_x = (target_center_x - image_center_x) / (frame_width / 2)
        error_y = (target_center_y - image_center_y) / (frame_height / 2)
        
        # Add to target history for smoothing
        self.face_history.append((error_x, error_y))
        
        # SIMPLIFIED smoothing with heavy damping to prevent oscillation
        if len(self.face_history) >= 2:
            # Simple moving average with more weight on recent values
            weights = [0.1, 0.2, 0.3, 0.4, 1.0]  # Linearly increasing weights
            weights = weights[-len(self.face_history):]  # Use only what we have
            total_weight = sum(weights)
            
            smoothed_error_x = sum(w * pos[0] for w, pos in zip(weights, self.face_history)) / total_weight
            smoothed_error_y = sum(w * pos[1] for w, pos in zip(weights, self.face_history)) / total_weight
            
            # MODERATE damping to prevent overshoot while maintaining responsiveness
            damping_factor = 0.7  # Moderate damping (was 0.5)
            smoothed_error_x *= damping_factor
            smoothed_error_y *= damping_factor
            
        else:
            smoothed_error_x = error_x * 0.7  # Less damping for single readings
            smoothed_error_y = error_y * 0.7
        
        # Apply adaptive deadzone based on recent movement - MODERATE
        adaptive_deadzone = self.tracking_deadzone
        recent_movement = len([m for m in self.movement_history if current_time - m['time'] < 1.0])
        if recent_movement > 3:  # Only increase deadzone after several movements
            adaptive_deadzone *= 1.5  # Moderate increase to prevent oscillation
        
        # Apply deadzone to smoothed error
        if abs(smoothed_error_x) < adaptive_deadzone:
            smoothed_error_x = 0
        if abs(smoothed_error_y) < adaptive_deadzone:
            smoothed_error_y = 0
        
        # ALWAYS update target time when face is being tracked (even if no movement needed)
        self.last_target_time = current_time  # Update whenever we're tracking a face
        
        # Update servo positions if there's significant error
        if smoothed_error_x != 0 or smoothed_error_y != 0:
            success = self._move_servos_enhanced(smoothed_error_x, smoothed_error_y)
            if success:
                self.tracking_stats['successful_tracks'] += 1
                return True
            else:
                # Face detected but no movement needed (within deadzone) - this is success!
                return True
        else:
            # Face detected and centered (within deadzone) - this is successful tracking!
            # Update velocity decay even when not moving
            self.pan_velocity *= self.velocity_decay
            self.tilt_velocity *= self.velocity_decay
            return True  # Return True because we ARE tracking a face (just perfectly centered)

    def _move_servos_enhanced(self, error_x, error_y):
        """Enhanced servo movement with velocity-based smoothing"""
        # Calculate adaptive speed based on error magnitude
        error_magnitude = math.sqrt(error_x**2 + error_y**2)
        tracking_speed = self._calculate_adaptive_speed(error_magnitude)
        
        # Calculate servo ranges and centers
        pan_range = self.pan_limits[1] - self.pan_limits[0]
        pan_center = (self.pan_limits[0] + self.pan_limits[1]) / 2
        
        tilt_range = self.tilt_limits[1] - self.tilt_limits[0]
        tilt_center = (self.tilt_limits[0] + self.tilt_limits[1]) / 2
        
        # Calculate target servo positions
        pan_offset = error_x * (pan_range / 2) * tracking_speed
        tilt_offset = error_y * (tilt_range / 2) * tracking_speed
        
        # CORRECTED DIRECTIONS: Both pan and tilt normal (after user feedback)
        target_pan = pan_center + pan_offset   # Normal: pan should follow face direction
        target_tilt = tilt_center + tilt_offset # Normal: tilt works correctly this way
        
        # Apply safety limits
        target_pan = max(self.pan_limits[0], min(self.pan_limits[1], target_pan))
        target_tilt = max(self.tilt_limits[0], min(self.tilt_limits[1], target_tilt))
        
        # Calculate desired movement
        pan_diff = target_pan - self.current_pan_angle
        tilt_diff = target_tilt - self.current_tilt_angle
        
        # SIMPLIFIED velocity tracking - just store for decay, don't use for movement limiting
        self.pan_velocity = pan_diff * 0.3  # Simple proportional velocity
        self.tilt_velocity = tilt_diff * 0.3
        
        # FIXED movement limiting - ignore velocity, just use max_move_per_step
        max_pan_move = self.max_move_per_step
        max_tilt_move = self.max_move_per_step
        
        # Clamp movement to maximum speeds
        if abs(pan_diff) > max_pan_move:
            pan_diff = max_pan_move * (1 if pan_diff > 0 else -1)
        if abs(tilt_diff) > max_tilt_move:
            tilt_diff = max_tilt_move * (1 if tilt_diff > 0 else -1)
        
        # Apply minimum movement threshold
        if abs(pan_diff) < self.min_move_threshold:
            pan_diff = 0
        if abs(tilt_diff) < self.min_move_threshold:
            tilt_diff = 0
        
        # Calculate new positions
        new_pan = self.current_pan_angle + pan_diff
        new_tilt = self.current_tilt_angle + tilt_diff
        
        # Execute movement if significant
        if abs(pan_diff) > 0 or abs(tilt_diff) > 0:
            # Store previous positions
            self.last_pan_angle = self.current_pan_angle
            self.last_tilt_angle = self.current_tilt_angle
            
            # Update current positions
            self.current_pan_angle = new_pan
            self.current_tilt_angle = new_tilt
            
            # Move servos
            self.servo.set_servo_pwm('0', int(self.current_pan_angle))
            self.servo.set_servo_pwm('1', int(self.current_tilt_angle))
            
            # Log movement
            movement_info = {
                'time': time.time(),
                'pan_diff': pan_diff,
                'tilt_diff': tilt_diff,
                'error_x': error_x,
                'error_y': error_y,
                'tracking_speed': tracking_speed
            }
            self.movement_history.append(movement_info)
            
            if self.debug_enabled:
                print(f"[FaceTracker] Enhanced move: Pan={self.current_pan_angle:.1f}° (Δ{pan_diff:+.1f}°), "
                      f"Tilt={self.current_tilt_angle:.1f}° (Δ{tilt_diff:+.1f}°), Speed={tracking_speed:.2f}")
            
            self.tracking_stats['movements_executed'] += 1
            return True
            
        return False

    def _return_to_center(self):
        """Return servos to center position with smooth movement"""
        target_pan = 90  # Center pan position
        target_tilt = 50  # Center tilt position
        
        # Calculate smooth movement towards center
        pan_diff = target_pan - self.current_pan_angle
        tilt_diff = target_tilt - self.current_tilt_angle
        
        # Use smaller steps for smooth return
        center_speed = 0.15  # Slow speed for center return
        
        if abs(pan_diff) > 0.5:
            pan_movement = pan_diff * center_speed
            self.current_pan_angle += pan_movement
        else:
            self.current_pan_angle = target_pan
            
        if abs(tilt_diff) > 0.5:
            tilt_movement = tilt_diff * center_speed
            self.current_tilt_angle += tilt_movement
        else:
            self.current_tilt_angle = target_tilt
        
        # Move servos
        self.servo.set_servo_pwm('0', int(self.current_pan_angle))
        self.servo.set_servo_pwm('1', int(self.current_tilt_angle))
        
        if self.debug_enabled:
            print(f"[FaceTracker] Smooth center return: Pan={self.current_pan_angle:.1f}°, Tilt={self.current_tilt_angle:.1f}°")
    
    def manual_movement_occurred(self):
        """Call this when manual camera movement occurs to pause auto-tracking"""
        self.last_manual_movement_time = time.time()
        self.face_history.clear()  # Clear face tracking history after manual movement
        self.pan_velocity = 0.0    # Reset velocity tracking
        self.tilt_velocity = 0.0
        
        if self.debug_enabled:
            print(f"[FaceTracker] Manual movement detected - auto-tracking paused for {self.manual_movement_cooldown}s")
    
    def toggle_tracking(self, enabled):
        """Enable or disable face tracking"""
        if not self.face_detection_enabled:
            return "Face tracking is not available. IMX500 model not found or initialization failed."
        
        self.enable_face_tracking = enabled
        if enabled:
            self.face_history.clear()  # Clear history when enabling
            self.pan_velocity = 0.0    # Reset velocity
            self.tilt_velocity = 0.0
            return "Enhanced face tracking enabled. Karl will now smoothly follow detected faces."
        else:
            return "Face tracking disabled. Karl will only move the camera when asked."
    
    def get_tracking_stats(self):
        """Get current tracking statistics"""
        return {
            **self.tracking_stats,
            'face_history_size': len(self.face_history),
            'movement_history_size': len(self.movement_history),
            'current_pan_velocity': self.pan_velocity,
            'current_tilt_velocity': self.tilt_velocity,
            'time_since_manual': time.time() - self.last_manual_movement_time
        }
    
    def set_debug_enabled(self, enabled):
        """Enable or disable debug output"""
        self.debug_enabled = enabled
        if enabled:
            print("[FaceTracker] Debug output enabled")
        else:
            print("[FaceTracker] Debug output disabled") 