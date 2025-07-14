#!/usr/bin/env python3
"""
IMU module for robot control dashboard
Handles BNO085 IMU sensor initialization, data reading, and processing
"""

import time
import threading
import math
import sys

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

class IMUManager:
    """Manages IMU sensor operations including initialization, data reading, and processing"""
    
    def __init__(self):
        self.bno = None
        self.i2c = None
        self.is_running = False
        self.data = {
            'acceleration': [0, 0, 0],
            'quaternion': [1, 0, 0, 0],
            'euler': [0, 0, 0],
            'timestamp': time.time()
        }
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize the BNO085 IMU sensor"""
        if not IMU_AVAILABLE:
            print("⚠ IMU libraries not available - IMU features disabled")
            return False
        
        try:
            print("📡 Initializing BNO085 IMU sensor...")
            
            # Create I2C bus
            self.i2c = busio.I2C(board.SCL, board.SDA)
            
            # Scan for devices
            try:
                while not self.i2c.try_lock():
                    time.sleep(0.01)
                
                devices = self.i2c.scan()
                print(f"Found I2C devices: {[hex(addr) for addr in devices]}")
                
                # Check for BNO085
                bno_addresses = [0x4A, 0x4B]
                found_bno = [addr for addr in devices if addr in bno_addresses]
                
                if found_bno:
                    print(f"✓ BNO085 detected at {hex(found_bno[0])}")
                else:
                    print("⚠ BNO085 not found at expected addresses")
                    
            finally:
                self.i2c.unlock()
            
            # Create sensor instance
            self.bno = BNO08X_I2C(self.i2c)
            
            # Enable features
            self.bno.enable_feature(BNO_REPORT_ACCELEROMETER)
            self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            
            print("✓ BNO085 IMU initialized successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize IMU: {e}")
            return False
    
    def start_reading_thread(self):
        """Start the IMU data reading thread"""
        if not self.bno:
            print("❌ IMU not initialized")
            return False
            
        self.is_running = True
        imu_thread = threading.Thread(target=self._read_data_loop, daemon=True)
        imu_thread.start()
        print("✓ IMU thread started")
        return True
    
    def _read_data_loop(self):
        """Background thread for continuous IMU reading"""
        while self.is_running:
            self._read_data()
            time.sleep(0.05)  # 20 Hz update rate
    
    def _read_data(self):
        """Read IMU data and update internal state"""
        if not self.bno:
            return
        
        try:
            # Read sensor data
            if self.bno.acceleration is not None:
                acc = self.bno.acceleration
            else:
                acc = [0, 0, 0]
                
            if self.bno.quaternion is not None:
                quat = self.bno.quaternion
                euler = self._quaternion_to_euler(quat)
            else:
                quat = [1, 0, 0, 0]
                euler = [0, 0, 0]
            
            # Update data with thread safety
            with self.lock:
                self.data = {
                    'acceleration': acc,
                    'quaternion': quat,
                    'euler': euler,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            print(f"Error reading IMU data: {e}")
    
    def _quaternion_to_euler(self, q):
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
    
    def get_data(self):
        """Get current IMU data with thread safety"""
        with self.lock:
            return self.data.copy()
    
    def get_status(self):
        """Get IMU status information"""
        return {
            'available': IMU_AVAILABLE,
            'initialized': self.bno is not None,
            'running': self.is_running,
            'timestamp': time.time()
        }
    
    def cleanup(self):
        """Cleanup IMU resources"""
        self.is_running = False
        if self.i2c:
            try:
                self.i2c.unlock()
            except:
                pass
        print("✓ IMU cleanup completed") 