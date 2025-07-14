#!/usr/bin/env python3
"""
Sensors module for robot control dashboard
Handles ultrasonic sensor, battery monitoring, ADC, and servo controls
"""

import time
import threading
import sys

# Ultrasonic sensor imports
try:
    sys.path.append("freenove_examples")
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

class UltrasonicManager:
    """Manages ultrasonic sensor operations"""
    
    def __init__(self):
        self.ultrasonic = None
        self.is_running = False
        self.distance = 0.0
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize the ultrasonic sensor"""
        if not ULTRASONIC_AVAILABLE:
            print("⚠ Ultrasonic libraries not available - ultrasonic features disabled")
            return False
        
        try:
            print("📐 Initializing ultrasonic sensor...")
            self.ultrasonic = Ultrasonic()
            print("✓ Ultrasonic sensor initialized")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize ultrasonic sensor: {e}")
            return False
    
    def start_reading_thread(self):
        """Start the ultrasonic reading thread"""
        if not self.ultrasonic:
            print("❌ Ultrasonic sensor not initialized")
            return False
            
        self.is_running = True
        ultrasonic_thread = threading.Thread(target=self._read_data_loop, daemon=True)
        ultrasonic_thread.start()
        print("✓ Ultrasonic thread started")
        return True
    
    def _read_data_loop(self):
        """Background thread for continuous ultrasonic reading"""
        while self.is_running:
            self._read_data()
            time.sleep(0.1)  # Read every 100ms
    
    def _read_data(self):
        """Read ultrasonic data and update internal state"""
        if not self.ultrasonic:
            return
        
        try:
            distance_cm = self.ultrasonic.get_distance()
            if distance_cm is not None:
                distance_m = distance_cm / 100.0  # Convert to meters
                with self.lock:
                    self.distance = distance_m
        except Exception as e:
            print(f"Error reading ultrasonic: {e}")
    
    def get_distance(self):
        """Get current distance with thread safety"""
        with self.lock:
            return self.distance
    
    def get_status(self):
        """Get ultrasonic status information"""
        return {
            'available': ULTRASONIC_AVAILABLE,
            'initialized': self.ultrasonic is not None,
            'running': self.is_running
        }
    
    def cleanup(self):
        """Cleanup ultrasonic resources"""
        self.is_running = False
        if self.ultrasonic:
            try:
                self.ultrasonic.close()
            except:
                pass
        print("✓ Ultrasonic cleanup completed")

class BatteryManager:
    """Manages battery monitoring operations"""
    
    def __init__(self):
        self.battery_monitor = None
        self.is_running = False
        self.data = {
            'battery_percent': 0,
            'battery_voltage': 0,
            'battery_current': 0,
            'charging_state': 'Unknown',
            'timestamp': time.time()
        }
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize the battery monitor"""
        if not BATTERY_AVAILABLE:
            print("⚠ Battery monitoring libraries not available - battery features disabled")
            return False
        
        try:
            print("🔋 Initializing battery monitor...")
            self.battery_monitor = UPSMonitor()
            print("✓ Battery monitor initialized")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize battery monitor: {e}")
            return False
    
    def start_reading_thread(self):
        """Start the battery reading thread"""
        if not self.battery_monitor:
            print("❌ Battery monitor not initialized")
            return False
            
        self.is_running = True
        battery_thread = threading.Thread(target=self._read_data_loop, daemon=True)
        battery_thread.start()
        print("✓ Battery thread started")
        return True
    
    def _read_data_loop(self):
        """Background thread for battery monitoring (low frequency)"""
        while self.is_running:
            self._read_data()
            time.sleep(5.0)  # Read every 5 seconds - low frequency to avoid impact
    
    def _read_data(self):
        """Read battery data and update internal state"""
        if not self.battery_monitor:
            return
        
        try:
            status = self.battery_monitor.get_battery_status()
            if status:
                with self.lock:
                    self.data = {
                        'battery_percent': status['battery_percent'],
                        'battery_voltage': status['battery_voltage'],
                        'battery_current': status['battery_current'],
                        'charging_state': status['charging_state'],
                        'timestamp': time.time()
                    }
        except Exception as e:
            print(f"Error reading battery data: {e}")
    
    def get_data(self):
        """Get current battery data with thread safety"""
        with self.lock:
            return self.data.copy()
    
    def get_status(self):
        """Get battery status information"""
        return {
            'available': BATTERY_AVAILABLE,
            'initialized': self.battery_monitor is not None,
            'running': self.is_running
        }
    
    def cleanup(self):
        """Cleanup battery resources"""
        self.is_running = False
        if self.battery_monitor:
            try:
                self.battery_monitor.bus.close()
            except:
                pass
        print("✓ Battery cleanup completed")

class ADCManager:
    """Manages ADC operations for chassis motor battery monitoring"""
    
    def __init__(self):
        self.adc_monitor = None
        self.is_running = False
        self.data = {
            'chassis_battery_voltage': 0.0,
            'chassis_battery_current': 0.0,
            'timestamp': time.time()
        }
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize the ADC for chassis motor battery monitoring"""
        if not ADC_AVAILABLE:
            print("⚠ ADC libraries not available - chassis battery features disabled")
            return False
        
        try:
            print("🔋 Initializing ADC for chassis battery monitoring...")
            self.adc_monitor = ADC()
            print("✓ ADC initialized")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize ADC: {e}")
            return False
    
    def start_reading_thread(self):
        """Start the ADC reading thread"""
        if not self.adc_monitor:
            print("❌ ADC monitor not initialized")
            return False
            
        self.is_running = True
        adc_thread = threading.Thread(target=self._read_data_loop, daemon=True)
        adc_thread.start()
        print("✓ ADC thread started")
        return True
    
    def _read_data_loop(self):
        """Background thread for ADC monitoring (moderate frequency)"""
        while self.is_running:
            self._read_data()
            time.sleep(1.0)  # Read every 1 second - moderate frequency
    
    def _read_data(self):
        """Read ADC data and update internal state"""
        if not self.adc_monitor:
            return
        
        try:
            # Read power voltage from channel 2 (as shown in the ADC example)
            # The power value is calculated based on PCB version
            power_voltage = self.adc_monitor.read_adc(2) * (3 if self.adc_monitor.pcb_version == 1 else 2)
            
            # For now, we'll estimate current based on voltage drop or set to 0
            # You might need to add a current sensor or calculate based on motor usage
            estimated_current = 0.0  # Placeholder - could be calculated from motor usage
            
            with self.lock:
                self.data = {
                    'chassis_battery_voltage': power_voltage,
                    'chassis_battery_current': estimated_current,
                    'timestamp': time.time()
                }
        except Exception as e:
            print(f"Error reading ADC data: {e}")
    
    def get_data(self):
        """Get current ADC data with thread safety"""
        with self.lock:
            return self.data.copy()
    
    def get_status(self):
        """Get ADC status information"""
        return {
            'available': ADC_AVAILABLE,
            'initialized': self.adc_monitor is not None,
            'running': self.is_running
        }
    
    def cleanup(self):
        """Cleanup ADC resources"""
        self.is_running = False
        if self.adc_monitor:
            try:
                self.adc_monitor.close_i2c()
            except:
                pass
        print("✓ ADC cleanup completed")

class ServoManager:
    """Manages servo operations for camera pan/tilt"""
    
    def __init__(self):
        self.servo_controller = None
        self.pan_angle = 90
        self.tilt_angle = 90
        
    def initialize(self):
        """Initialize the servo controller for camera pan/tilt"""
        if not SERVO_AVAILABLE:
            print("⚠ Servo libraries not available - camera pan/tilt features disabled")
            return False
        
        try:
            print("🎥 Initializing servo controller for camera pan/tilt...")
            self.servo_controller = Servo()
            
            # Initialize servos to center position
            self.servo_controller.set_servo_pwm('0', 90)  # Pan servo
            self.servo_controller.set_servo_pwm('1', 90)  # Tilt servo
            
            print("✓ Servo controller initialized - camera centered")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize servo controller: {e}")
            return False
    
    def set_pan(self, angle):
        """Set camera pan angle"""
        if not self.servo_controller:
            return False, "Servo controller not initialized"
        
        try:
            angle = max(0, min(180, int(angle)))  # Clamp to 0-180 degrees
            self.servo_controller.set_servo_pwm('0', angle)
            self.pan_angle = angle
            print(f"🎥 Camera pan set to {angle}°")
            return True, f'Pan set to {angle}°'
        except Exception as e:
            return False, str(e)
    
    def set_tilt(self, angle):
        """Set camera tilt angle"""
        if not self.servo_controller:
            return False, "Servo controller not initialized"
        
        try:
            angle = max(0, min(180, int(angle)))  # Clamp to 0-180 degrees
            self.servo_controller.set_servo_pwm('1', angle)
            self.tilt_angle = angle
            print(f"🎥 Camera tilt set to {angle}°")
            return True, f'Tilt set to {angle}°'
        except Exception as e:
            return False, str(e)
    
    def center_camera(self):
        """Center both servos"""
        if not self.servo_controller:
            return False, "Servo controller not initialized"
        
        try:
            # Center both servos
            self.servo_controller.set_servo_pwm('0', 90)
            self.servo_controller.set_servo_pwm('1', 90)
            self.pan_angle = 90
            self.tilt_angle = 90
            print("🎥 Camera centered")
            return True, "Camera centered"
        except Exception as e:
            return False, str(e)
    
    def get_angles(self):
        """Get current pan and tilt angles"""
        return {
            'pan': self.pan_angle,
            'tilt': self.tilt_angle
        }
    
    def get_status(self):
        """Get servo status information"""
        return {
            'available': SERVO_AVAILABLE,
            'initialized': self.servo_controller is not None,
            'pan_angle': self.pan_angle,
            'tilt_angle': self.tilt_angle
        }
    
    def cleanup(self):
        """Cleanup servo resources - center servos before exiting"""
        if self.servo_controller:
            try:
                self.servo_controller.set_servo_pwm('0', 90)
                self.servo_controller.set_servo_pwm('1', 90)
            except:
                pass
        print("✓ Servo cleanup completed") 