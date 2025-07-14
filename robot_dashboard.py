#!/usr/bin/env python3
"""
Enhanced low-latency web stream with IMX500 camera + BNO085 IMU + Motor Controls + Audio
Access via: http://raspberry_pi_ip:5001

Usage:
  python3 robot_dashboard.py           # Full mode (all features)
  python3 robot_dashboard.py --minimal # Minimal mode (camera + audio only)

⚠️  SAFETY: Motor controls included but not auto-tested. Test manually first!
"""

import time
import sys
from flask import Flask

# Import modules
from modules.config import STREAM_CONFIG, MOTOR_CONFIG, FLASK_CONFIG, AUDIO_CONFIG, NAVIGATION_CONFIG, print_config_summary
from modules.camera import CameraManager
from modules.imu import IMUManager
from modules.motors import MotorManager
from modules.sensors import UltrasonicManager, BatteryManager, ADCManager, ServoManager
from modules.audio import ReSpeakerManager
from modules.navigation import NavigationManager
from modules.web_routes import WebRoutes

class RobotDashboard:
    """Main robot dashboard application with full and minimal modes"""
    
    def __init__(self, minimal_mode=False):
        self.app = Flask(__name__)
        self.managers = {}
        self.is_running = False
        self.minimal_mode = minimal_mode
        
    def initialize_managers(self):
        """Initialize hardware managers based on mode"""
        if self.minimal_mode:
            print("🎬 Initializing minimal systems (Camera + Audio only)...")
        else:
            print("🎬 Initializing all hardware managers...")
        
        # Initialize camera manager (always needed)
        self.managers['camera'] = CameraManager(STREAM_CONFIG)
        if not self.managers['camera'].initialize():
            print("❌ Failed to initialize camera. Exiting.")
            return False
        
        # Initialize audio manager (always needed)
        self.managers['audio'] = ReSpeakerManager(AUDIO_CONFIG)
        self.audio_success = self.managers['audio'].initialize()
        if not self.audio_success:
            print("⚠ Audio initialization failed - continuing without audio features")
        
        if self.minimal_mode:
            # In minimal mode, skip other hardware
            print("🚀 Minimal mode: Skipping IMU, motors, and sensors for lowest latency")
            return True
        
        # Full mode: Initialize all hardware
        # Initialize IMU manager
        self.managers['imu'] = IMUManager()
        self.imu_success = self.managers['imu'].initialize()
        if not self.imu_success:
            print("⚠ IMU initialization failed - continuing without IMU features")
        
        # Initialize motor manager
        self.managers['motor'] = MotorManager(MOTOR_CONFIG)
        self.motor_success = self.managers['motor'].initialize()
        if not self.motor_success:
            print("⚠ Motor initialization failed - continuing without motor features")
        
        # Initialize sensor managers
        self.managers['ultrasonic'] = UltrasonicManager()
        self.ultrasonic_success = self.managers['ultrasonic'].initialize()
        if not self.ultrasonic_success:
            print("⚠ Ultrasonic initialization failed - continuing without ultrasonic features")
        
        self.managers['battery'] = BatteryManager()
        self.battery_success = self.managers['battery'].initialize()
        if not self.battery_success:
            print("⚠ Battery monitor initialization failed - continuing without battery features")
        
        self.managers['adc'] = ADCManager()
        self.adc_success = self.managers['adc'].initialize()
        if not self.adc_success:
            print("⚠ ADC initialization failed - continuing without chassis battery features")
        
        self.managers['servo'] = ServoManager()
        self.servo_success = self.managers['servo'].initialize()
        if not self.servo_success:
            print("⚠ Servo initialization failed - continuing without camera pan/tilt features")
        
        # Initialize navigation manager
        self.managers['navigation'] = NavigationManager(
            self.managers.get('imu'),
            self.managers.get('motor'),
            self.managers.get('ultrasonic'),
            NAVIGATION_CONFIG
        )
        self.navigation_success = self.managers['navigation'].initialize()
        if not self.navigation_success:
            print("⚠ Navigation initialization failed - continuing without navigation features")
        
        return True
    
    def start_background_threads(self):
        """Start background threads based on mode"""
        if self.minimal_mode:
            print("🎬 Starting minimal threads (Camera + Audio only)...")
        else:
            print("🎬 Starting all background threads...")
        
        # Start camera capture thread (always needed)
        if not self.managers['camera'].start_capture_thread():
            print("❌ Failed to start camera capture thread")
            return False
        
        # Start audio streaming (always needed)
        if self.audio_success:
            if not self.managers['audio'].start_streaming():
                print("⚠ Failed to start audio streaming")
        
        if self.minimal_mode:
            # In minimal mode, skip other threads
            return True
        
        # Full mode: Start all threads
        # Start IMU reading thread
        if hasattr(self, 'imu_success') and self.imu_success:
            if not self.managers['imu'].start_reading_thread():
                print("⚠ Failed to start IMU thread")
        
        # Start ultrasonic reading thread
        if hasattr(self, 'ultrasonic_success') and self.ultrasonic_success:
            if not self.managers['ultrasonic'].start_reading_thread():
                print("⚠ Failed to start ultrasonic thread")
        
        # Start battery reading thread
        if hasattr(self, 'battery_success') and self.battery_success:
            if not self.managers['battery'].start_reading_thread():
                print("⚠ Failed to start battery thread")
        
        # Start ADC reading thread
        if hasattr(self, 'adc_success') and self.adc_success:
            if not self.managers['adc'].start_reading_thread():
                print("⚠ Failed to start ADC thread")
        
        return True
    
    def setup_web_routes(self):
        """Setup web routes and API endpoints"""
        print("🌐 Setting up web routes...")
        self.web_routes = WebRoutes(self.app, self.managers)
        print("✓ Web routes configured")
    
    def wait_for_initialization(self):
        """Wait for first frame and system initialization"""
        print("⏳ Waiting for first frame...")
        if not self.managers['camera'].wait_for_first_frame():
            print("❌ Timeout waiting for first frame")
            return False
        
        print("✅ Robot Control Dashboard ready!")
        return True
    
    def print_startup_info(self):
        """Print startup information"""
        mode_name = "MINIMAL LATENCY" if self.minimal_mode else "FULL FEATURED"
        features = "Camera + Audio/DOA" if self.minimal_mode else "Camera + IMU + Ultrasonic + Battery + Chassis Battery + Camera Pan/Tilt + Motor Controls + Audio/DOA"
        
        print("=" * 80)
        print(f"🤖 Robot Control Dashboard - {mode_name} MODE")
        print("=" * 80)
        if not self.minimal_mode:
            print_config_summary()
        print(f"🌐 Access at: http://localhost:{FLASK_CONFIG['port']}")
        print("📱 Or from network: http://YOUR_PI_IP:5001")
        print(f"🎯 Features: {features}")
        if not self.minimal_mode:
            print("⚠️  SAFETY: Motors are DISABLED by default!")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 80)
    
    def run(self):
        """Run the robot dashboard application"""
        try:
            # Initialize managers
            if not self.initialize_managers():
                return
            
            # Start background threads
            if not self.start_background_threads():
                return
            
            # Setup web routes
            self.setup_web_routes()
            
            # Wait for initialization
            if not self.wait_for_initialization():
                return
            
            # Print startup info
            self.print_startup_info()
            
            # Start Flask application
            self.is_running = True
            self.app.run(
                host=FLASK_CONFIG['host'],
                port=FLASK_CONFIG['port'],
                debug=False if self.minimal_mode else FLASK_CONFIG['debug'],  # Disable debug in minimal mode
                threaded=FLASK_CONFIG['threaded'],
                use_reloader=False  # Always disable reloader for stability
            )
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup all resources"""
        print("🧹 Cleaning up resources...")
        
        # Stop all managers
        for name, manager in self.managers.items():
            try:
                manager.cleanup()
            except Exception as e:
                print(f"⚠ Error cleaning up {name}: {e}")
        
        print("✅ Cleanup completed")

def main():
    """Main function with command line argument parsing"""
    minimal_mode = "--minimal" in sys.argv or "-m" in sys.argv
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return
    
    dashboard = RobotDashboard(minimal_mode=minimal_mode)
    dashboard.run()

if __name__ == '__main__':
    main() 