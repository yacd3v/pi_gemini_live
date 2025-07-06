#!/usr/bin/env python3
"""
BNO085 IMU Test Script for Raspberry Pi 5
=========================================

This script tests the BNO085 9-DOF IMU sensor connected to a Raspberry Pi 5.
It reads orientation data, acceleration, and other sensor values.

Hardware Connection (I2C):
- VIN -> 3.3V or 5V
- GND -> GND  
- SCL -> GPIO 3 (Pin 5)
- SDA -> GPIO 2 (Pin 3)
- RST -> GPIO 4 (Pin 7) [optional]

Installation:
sudo pip3 install adafruit-circuitpython-bno08x

Author: Auto-generated test script
Date: 2025
"""

import time
import board
import busio
import logging
import math
from datetime import datetime
import sys

try:
    from adafruit_bno08x.i2c import BNO08X_I2C
    from adafruit_bno08x import (
        BNO_REPORT_ACCELEROMETER,
        BNO_REPORT_GYROSCOPE,
        BNO_REPORT_MAGNETOMETER,
        BNO_REPORT_ROTATION_VECTOR,
        BNO_REPORT_LINEAR_ACCELERATION,
        BNO_REPORT_GRAVITY
    )
except ImportError as e:
    print("Error: BNO08x library not found!")
    print("Please install it with: sudo pip3 install adafruit-circuitpython-bno08x")
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bno085_test.log'),
        logging.StreamHandler()
    ]
)

class BNO085Tester:
    def __init__(self):
        self.bno = None
        self.i2c = None
        self.other_devices = []
        
    def initialize_sensor(self):
        """Initialize the BNO085 sensor"""
        print("Initializing BNO085 sensor...")
        
        try:
            # Create I2C bus with conservative settings for multi-device environment
            # Lower frequency to reduce conflicts with other devices
            self.i2c = busio.I2C(board.SCL, board.SDA, frequency=50000)
            print("I2C bus created successfully (50kHz for multi-device stability)")
            
            # Scan for devices first
            print("Scanning I2C bus for devices...")
            while not self.i2c.try_lock():
                pass
            try:
                devices = self.i2c.scan()
                print(f"Found I2C devices at addresses: {[hex(addr) for addr in devices]}")
                
                # Check if BNO085 is present (typically at 0x4A or 0x4B)
                bno_addresses = [0x4A, 0x4B]
                found_bno = [addr for addr in devices if addr in bno_addresses]
                
                # Store other devices for conflict analysis
                self.other_devices = [addr for addr in devices if addr not in bno_addresses]
                
                if found_bno:
                    print(f"✓ BNO085 detected at address: {hex(found_bno[0])}")
                else:
                    print("⚠ Warning: BNO085 not found at expected addresses (0x4A, 0x4B)")
                    print("This may cause initialization to fail")
                
                # Check for potential conflicts
                self.check_device_conflicts()
                    
            finally:
                self.i2c.unlock()
            
            # Create BNO08X sensor instance
            self.bno = BNO08X_I2C(self.i2c)
            print("BNO085 sensor instance created")
            
            # Enable the features we want to read
            print("Enabling sensor features...")
            self.bno.enable_feature(BNO_REPORT_ACCELEROMETER)
            self.bno.enable_feature(BNO_REPORT_GYROSCOPE)
            self.bno.enable_feature(BNO_REPORT_MAGNETOMETER)
            self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            self.bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
            self.bno.enable_feature(BNO_REPORT_GRAVITY)
            
            print("✓ BNO085 initialized successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize BNO085: {e}")
            logging.error(f"Sensor initialization failed: {e}")
            return False
    
    def check_device_conflicts(self):
        """Check for potential conflicts with other I2C devices"""
        if not self.other_devices:
            print("✓ No other I2C devices detected - minimal conflict risk")
            return
        
        print(f"⚠ Other I2C devices detected: {[hex(addr) for addr in self.other_devices]}")
        
        # Known device types and potential conflicts
        device_warnings = {
            0x2D: "OLED Display - May conflict with BNO085 clock stretching",
            0x3C: "OLED Display - May conflict with BNO085 clock stretching", 
            0x3D: "OLED Display - May conflict with BNO085 clock stretching",
            0x40: "Sensor/ADC - Generally compatible",
            0x48: "Sensor/ADC - Generally compatible",
            0x70: "I2C Multiplexer - Could help isolate devices",
            0x77: "Pressure Sensor - Generally compatible"
        }
        
        high_risk_devices = [0x2D, 0x3C, 0x3D]  # Display controllers
        
        for addr in self.other_devices:
            if addr in device_warnings:
                print(f"  {hex(addr)}: {device_warnings[addr]}")
            else:
                print(f"  {hex(addr)}: Unknown device")
        
        # Check for high-risk combinations
        display_devices = [addr for addr in self.other_devices if addr in high_risk_devices]
        if display_devices:
            print("\n🚨 WARNING: Display devices detected!")
            print("   Displays may conflict with BNO085 clock stretching.")
            print("   Recommendations:")
            print("   - Use lower I2C frequency (50kHz - already configured)")
            print("   - Consider using separate I2C buses")
            print("   - Test thoroughly for data corruption")
        
        if len(self.other_devices) >= 3:
            print("\n⚠ CAUTION: Multiple I2C devices detected!")
            print("   This may cause:")
            print("   - Power supply issues")
            print("   - Bus timing conflicts")
            print("   - Signal integrity problems")
            print("   Recommendations:")
            print("   - Use I2C multiplexer (TCA9548A)")
            print("   - Ensure adequate power supply")
            print("   - Keep wire lengths short")
    
    def read_sensor_data(self):
        """Read all available sensor data with multi-device error handling"""
        data = {}
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Add small delay between reads for multi-device stability
                if attempt > 0:
                    time.sleep(0.01)  # 10ms delay on retry
                
                # Read acceleration (m/s^2)
                if self.bno.acceleration is not None:
                    data['acceleration'] = self.bno.acceleration
                
                # Read angular velocity (rad/s)
                if self.bno.gyro is not None:
                    data['gyroscope'] = self.bno.gyro
                
                # Read magnetic field (uT)
                if self.bno.magnetic is not None:
                    data['magnetometer'] = self.bno.magnetic
                
                # Read rotation vector (quaternion)
                if self.bno.quaternion is not None:
                    data['quaternion'] = self.bno.quaternion
                
                # Read linear acceleration (m/s^2)
                if self.bno.linear_acceleration is not None:
                    data['linear_acceleration'] = self.bno.linear_acceleration
                
                # Read gravity vector (m/s^2)
                if self.bno.gravity is not None:
                    data['gravity'] = self.bno.gravity
                
                # If we got here, reading was successful
                break
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error reading sensor data after {max_retries} attempts: {e}")
                    logging.error(f"Data reading error: {e}")
                    if self.other_devices:
                        print("⚠ This may be due to I2C bus conflicts with other devices")
                else:
                    print(f"Read attempt {attempt + 1} failed, retrying...")
                    
        return data
    
    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
        if q is None or len(q) < 4:
            return None, None, None
            
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
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        
        return roll_deg, pitch_deg, yaw_deg
    
    def print_sensor_data(self, data):
        """Print formatted sensor data"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- BNO085 Sensor Data [{timestamp}] ---")
        
        # Acceleration
        if 'acceleration' in data:
            acc = data['acceleration']
            print(f"Acceleration (m/s²): X={acc[0]:.3f}, Y={acc[1]:.3f}, Z={acc[2]:.3f}")
        
        # Gyroscope
        if 'gyroscope' in data:
            gyro = data['gyroscope']
            print(f"Gyroscope (rad/s):   X={gyro[0]:.3f}, Y={gyro[1]:.3f}, Z={gyro[2]:.3f}")
        
        # Magnetometer
        if 'magnetometer' in data:
            mag = data['magnetometer']
            print(f"Magnetometer (uT):   X={mag[0]:.3f}, Y={mag[1]:.3f}, Z={mag[2]:.3f}")
        
        # Quaternion and Euler angles
        if 'quaternion' in data:
            quat = data['quaternion']
            print(f"Quaternion:          W={quat[0]:.3f}, X={quat[1]:.3f}, Y={quat[2]:.3f}, Z={quat[3]:.3f}")
            
            # Convert to Euler angles
            roll, pitch, yaw = self.quaternion_to_euler(quat)
            if roll is not None:
                print(f"Euler Angles (°):    Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}")
        
        # Linear acceleration
        if 'linear_acceleration' in data:
            lin_acc = data['linear_acceleration']
            print(f"Linear Accel (m/s²): X={lin_acc[0]:.3f}, Y={lin_acc[1]:.3f}, Z={lin_acc[2]:.3f}")
        
        # Gravity
        if 'gravity' in data:
            grav = data['gravity']
            print(f"Gravity (m/s²):      X={grav[0]:.3f}, Y={grav[1]:.3f}, Z={grav[2]:.3f}")
        
        print("-" * 50)
    
    def run_test(self, duration=60, interval=1.0):
        """Run the sensor test for specified duration"""
        if not self.initialize_sensor():
            return False
        
        print(f"\nStarting BNO085 test for {duration} seconds...")
        print(f"Reading interval: {interval} seconds")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        reading_count = 0
        
        try:
            while time.time() - start_time < duration:
                # Read sensor data
                data = self.read_sensor_data()
                
                if data:
                    self.print_sensor_data(data)
                    
                    # Log to file
                    logging.info(f"Reading #{reading_count + 1}: {data}")
                    reading_count += 1
                else:
                    print("No data received from sensor")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        except Exception as e:
            print(f"Error during test: {e}")
            logging.error(f"Test error: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"\nTest completed!")
        print(f"Total readings: {reading_count}")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Log file: bno085_test.log")
        
        return True

def main():
    """Main function"""
    print("BNO085 IMU Test Script")
    print("=" * 30)
    
    # Check if we're running as root (may be needed for I2C access)
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Running in virtual environment")
    
    # Create tester instance
    tester = BNO085Tester()
    
    # Run the test
    try:
        success = tester.run_test(duration=60, interval=2.0)
        if success:
            print("\n✓ Test completed successfully!")
        else:
            print("\n✗ Test failed!")
            return 1
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        logging.error(f"Main test error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
