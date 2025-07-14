#!/usr/bin/env python3
"""
Navigation module for robot control dashboard
Handles autonomous navigation with IMU-based heading control and collision avoidance
"""

import time
import threading
import math
from collections import deque

class NavigationManager:
    """Manages autonomous navigation with IMU feedback and collision avoidance"""
    
    def __init__(self, imu_manager, motor_manager, ultrasonic_manager, config=None):
        self.imu_manager = imu_manager
        self.motor_manager = motor_manager
        self.ultrasonic_manager = ultrasonic_manager
        
        # Default configuration
        self.config = config or {
            'collision_threshold': 0.20,  # 20cm collision threshold
            'base_speed': 1500,           # Base movement speed
            'turn_speed': 1000,           # Turning speed
            'angle_tolerance': 5.0,       # Degrees tolerance for heading
            'distance_tolerance': 0.05,   # 5cm distance tolerance
            'max_navigation_time': 60.0,  # Max time for navigation (seconds)
            'control_loop_freq': 10.0,    # Control loop frequency (Hz)
            'heading_correction_factor': 0.5  # PID-like factor for heading correction
        }
        
        # Navigation state
        self.is_navigating = False
        self.navigation_thread = None
        self.lock = threading.Lock()
        
        # Navigation tracking
        self.start_position = None
        self.target_angle = 0.0
        self.target_distance = 0.0
        self.start_time = 0.0
        self.estimated_distance_traveled = 0.0
        
        # IMU-based distance estimation
        self.velocity = 0.0  # Current velocity in m/s
        self.last_acceleration_time = None
        self.acceleration_history = deque(maxlen=10)  # For noise filtering
        self.velocity_history = deque(maxlen=100)  # For debugging
        self.distance_samples = deque(maxlen=20)  # For smoothing
        
        # Navigation status
        self.navigation_status = {
            'active': False,
            'progress': 0.0,
            'current_angle': 0.0,
            'target_angle': 0.0,
            'distance_traveled': 0.0,
            'target_distance': 0.0,
            'collision_detected': False,
            'status_message': 'Ready'
        }
        
    def initialize(self):
        """Initialize navigation system"""
        print("🧭 Initializing navigation system...")
        
        # Check if required managers are available
        if not self.imu_manager or not self.motor_manager or not self.ultrasonic_manager:
            print("⚠ Required managers not available - navigation disabled")
            return False
        
        # Check if managers are initialized
        try:
            imu_status = self.imu_manager.get_status()
            motor_status = self.motor_manager.get_status()
            ultrasonic_status = self.ultrasonic_manager.get_status()
            
            imu_init = bool(imu_status.get('initialized', False))
            motor_init = bool(motor_status.get('initialized', False))
            ultrasonic_init = bool(ultrasonic_status.get('initialized', False))
            
            if not (imu_init and motor_init and ultrasonic_init):
                print("⚠ Required sensors not initialized - navigation disabled")
                return False
        except Exception as e:
            print(f"Error checking manager status: {e}")
            return False
        
        print("✓ Navigation system initialized")
        return True
    
    def navigate_to_angle_distance(self, angle_degrees, distance_meters):
        """
        Navigate to a specific angle and distance
        
        Args:
            angle_degrees: Target angle in degrees (0 = forward, 90 = right, etc.)
            distance_meters: Distance to travel in meters
        """
        if self.is_navigating:
            return False, "Navigation already in progress"
        
        # Validate inputs
        if distance_meters <= 0:
            return False, "Distance must be positive"
        
        if abs(angle_degrees) > 360:
            return False, "Angle must be between -360 and 360 degrees"
        
        # Check if motors are enabled
        motor_status = self.motor_manager.get_status()
        if not motor_status['enabled']:
            return False, "Motors must be enabled before navigation"
        
        # Get current IMU data to determine starting heading
        imu_data = self.imu_manager.get_data()
        current_yaw = imu_data['euler'][2]  # Current heading
        
        # Calculate target heading based on current heading + desired angle
        # For forward (0°), we want to maintain current heading
        # For other angles, we want to turn relative to current heading
        if angle_degrees == 0:
            # Forward: maintain current heading
            target_heading = current_yaw
        else:
            # Turn relative to current heading
            target_heading = (current_yaw + angle_degrees) % 360
            # Normalize to -180 to 180 range
            if target_heading > 180:
                target_heading -= 360
        
        # Start navigation
        with self.lock:
            self.target_angle = target_heading
            self.target_distance = distance_meters
            self.start_time = time.time()
            self.estimated_distance_traveled = 0.0
            
            # Reset IMU-based tracking
            self.velocity = 0.0
            self.last_acceleration_time = None
            self.acceleration_history.clear()
            self.velocity_history.clear()
            
            self.is_navigating = True
            
            # Set start position for reference
            self.start_position = imu_data
            
            # Reset navigation status
            self.navigation_status.update({
                'active': True,
                'progress': 0.0,
                'current_angle': current_yaw,
                'target_angle': target_heading,
                'distance_traveled': 0.0,
                'target_distance': distance_meters,
                'collision_detected': False,
                'status_message': 'Navigating...'
            })
        
        # Start navigation thread
        self.navigation_thread = threading.Thread(target=self._navigation_loop, daemon=True)
        self.navigation_thread.start()
        
        print(f"🧭 Navigation started: {angle_degrees}° (target heading: {target_heading:.1f}°) for {distance_meters}m")
        return True, f"Navigation started: {angle_degrees}° turn (target heading: {target_heading:.1f}°) for {distance_meters}m"
    
    def _navigation_loop(self):
        """Main navigation control loop"""
        loop_period = 1.0 / self.config['control_loop_freq']
        
        while self.is_navigating:
            try:
                # Check for collision
                if self._check_collision():
                    self._stop_navigation("Collision detected")
                    break
                
                # Check timeout
                if time.time() - self.start_time > self.config['max_navigation_time']:
                    self._stop_navigation("Navigation timeout")
                    break
                
                # Get current sensor data
                imu_data = self.imu_manager.get_data()
                
                # Safety check: ensure IMU data is valid
                if not imu_data or 'euler' not in imu_data or len(imu_data['euler']) < 3:
                    print(f"Warning: Invalid IMU data: {imu_data}")
                    time.sleep(loop_period)
                    continue
                
                current_yaw = imu_data['euler'][2]  # Yaw angle
                
                # Safety check: ensure yaw is a valid number
                if not isinstance(current_yaw, (int, float)):
                    print(f"Warning: Invalid yaw angle: {current_yaw}")
                    time.sleep(loop_period)
                    continue
                
                # Calculate heading error
                heading_error = self._calculate_heading_error(current_yaw, self.target_angle)
                
                # Update distance estimation only when moving forward
                # (Don't count turning as distance traveled)
                if abs(heading_error) <= self.config['angle_tolerance']:
                    self._update_distance_estimation(imu_data)
                
                # Check if we've reached the target distance
                if self.estimated_distance_traveled >= self.target_distance:
                    self._stop_navigation("Target reached")
                    break
                
                # Determine movement strategy
                if abs(heading_error) > self.config['angle_tolerance']:
                    # Need to adjust heading
                    self._adjust_heading(heading_error)
                else:
                    # Move forward
                    self._move_forward()
                
                # Update navigation status
                with self.lock:
                    self.navigation_status.update({
                        'current_angle': current_yaw,
                        'distance_traveled': self.estimated_distance_traveled,
                        'progress': min(self.estimated_distance_traveled / self.target_distance, 1.0)
                    })
                
                time.sleep(loop_period)
                
            except Exception as e:
                print(f"Navigation error: {e}")
                self._stop_navigation(f"Navigation error: {e}")
                break
    
    def _check_collision(self):
        """Check if obstacle is too close"""
        try:
            distance = self.ultrasonic_manager.get_distance()
            
            # Safety check: ensure distance is a valid number
            if distance is None or not isinstance(distance, (int, float)):
                print(f"Warning: Invalid distance reading: {distance}")
                return False
            
            collision_detected = distance < self.config['collision_threshold']
            
            with self.lock:
                self.navigation_status['collision_detected'] = collision_detected
            
            return collision_detected
        except Exception as e:
            print(f"Collision check error: {e}")
            return False
    
    def _calculate_heading_error(self, current_yaw, target_angle):
        """Calculate the shortest angular difference between current and target heading"""
        try:
            # Ensure both values are floats
            current_yaw = float(current_yaw)
            target_angle = float(target_angle)
            
            error = target_angle - current_yaw
            
            # Normalize to [-180, 180] range
            while error > 180:
                error -= 360
            while error < -180:
                error += 360
            
            return float(error)
        except Exception as e:
            print(f"Error calculating heading error: {e}")
            return 0.0  # Return safe default
    
    def _adjust_heading(self, heading_error):
        """Adjust robot heading based on error"""
        try:
            # Determine turn direction and speed
            turn_speed = min(abs(heading_error) * self.config['heading_correction_factor'], 1.0) * self.config['turn_speed']
            
            # Ensure turn_speed is an integer for motor control
            turn_speed = int(turn_speed)
            
            if heading_error > 0:
                # Turn right
                self.motor_manager.motor_car.drive(-turn_speed, -turn_speed, turn_speed, turn_speed)
            else:
                # Turn left
                self.motor_manager.motor_car.drive(turn_speed, turn_speed, -turn_speed, -turn_speed)
        except Exception as e:
            print(f"Error in _adjust_heading: {e}")
            self.motor_manager.motor_car.stop()
    
    def _move_forward(self):
        """Move forward at base speed"""
        try:
            speed = int(self.config['base_speed'])  # Ensure speed is an integer
            # Use negative values to move forward (reversed for this robot configuration)
            self.motor_manager.motor_car.drive(-speed, -speed, -speed, -speed)
        except Exception as e:
            print(f"Error in _move_forward: {e}")
            self.motor_manager.motor_car.stop()
    
    def _update_distance_estimation(self, imu_data):
        """Update estimated distance traveled using IMU acceleration data"""
        try:
            current_time = float(time.time())
            
            # Initialize timing if first call
            if self.last_acceleration_time is None:
                self.last_acceleration_time = current_time
                self.velocity = 0.0
                return
            
            # Calculate time delta
            time_delta = current_time - self.last_acceleration_time
            if time_delta <= 0:
                return
            
            # Get acceleration data from IMU
            if 'acceleration' in imu_data and len(imu_data['acceleration']) >= 3:
                # Raw acceleration data [x, y, z] in m/s²
                accel_raw = imu_data['acceleration']
                
                # Convert to robot coordinate system
                # Assuming robot's forward direction is along one of the IMU axes
                # You may need to adjust this based on your IMU mounting orientation
                accel_forward = accel_raw[1]  # Try Y-axis acceleration (forward/backward)
                
                # Apply simple noise filtering (moving average)
                self.acceleration_history.append(accel_forward)
                if len(self.acceleration_history) > 3:
                    filtered_accel = sum(self.acceleration_history) / len(self.acceleration_history)
                else:
                    filtered_accel = accel_forward
                
                # Remove gravity component and small noise
                # Only consider significant accelerations for movement
                if abs(filtered_accel) < 0.2:  # Lowered threshold to detect smaller accelerations
                    filtered_accel = 0.0
                
                # Debug acceleration data
                if self.config.get('debug_distance', False):
                    print(f"IMU raw: x={accel_raw[0]:.2f}, y={accel_raw[1]:.2f}, z={accel_raw[2]:.2f}, filtered={filtered_accel:.2f}")
                
                # Integrate acceleration to get velocity
                velocity_change = filtered_accel * time_delta
                self.velocity += velocity_change
                
                # Apply velocity decay to account for friction and stopping
                # This helps prevent velocity drift when robot stops
                decay_factor = 0.98  # Reduced decay for better velocity tracking
                self.velocity *= decay_factor
                
                # Clamp velocity to reasonable bounds
                max_velocity = 1.0  # Maximum expected velocity in m/s
                self.velocity = max(-max_velocity, min(max_velocity, self.velocity))
                
                # Integrate velocity to get distance (only if moving forward)
                if abs(self.velocity) > 0.01:  # Count any significant movement
                    distance_increment = abs(self.velocity) * time_delta  # Use absolute value for distance
                    self.estimated_distance_traveled += distance_increment
                    
                    # Optional debug output
                    if self.config.get('debug_distance', False):
                        print(f"IMU: accel={filtered_accel:.2f}, vel={self.velocity:.3f}, dist=+{distance_increment:.3f}m, total={self.estimated_distance_traveled:.3f}m")
            
            self.last_acceleration_time = current_time
            
        except Exception as e:
            print(f"Error in IMU-based distance estimation: {e}")
            # Fall back to simple time-based estimation if IMU fails
            if hasattr(self, 'last_move_time'):
                time_delta = current_time - self.last_move_time
                if time_delta > 0:
                    # Simple fallback
                    distance_increment = 0.1 * time_delta  # Rough estimate
                    self.estimated_distance_traveled += distance_increment
            self.last_move_time = current_time
    
    def _stop_navigation(self, reason):
        """Stop navigation and cleanup"""
        with self.lock:
            self.is_navigating = False
            self.navigation_status.update({
                'active': False,
                'status_message': reason
            })
        
        # Stop motors
        self.motor_manager.motor_car.stop()
        print(f"🛑 Navigation stopped: {reason}")
    
    def stop_navigation(self):
        """Stop current navigation"""
        if self.is_navigating:
            self._stop_navigation("Stopped by user")
            return True, "Navigation stopped"
        return False, "No navigation in progress"
    
    def get_navigation_status(self):
        """Get current navigation status"""
        with self.lock:
            return self.navigation_status.copy()
    
    def get_status(self):
        """Get navigation system status"""
        return {
            'available': True,
            'initialized': True,
            'navigating': self.is_navigating,
            'config': self.config
        }
    
    def cleanup(self):
        """Cleanup navigation resources"""
        if self.is_navigating:
            self.stop_navigation()
        print("✓ Navigation cleanup completed") 