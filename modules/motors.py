#!/usr/bin/env python3
"""
Motor control module for robot control dashboard
Handles Mecanum car motor initialization, control, and safety features
"""

import sys

# Motor control imports
try:
    sys.path.append("freenove_examples")
    from pca9685 import PCA9685
    MOTOR_AVAILABLE = True
except ImportError as e:
    print(f"Motor libraries not available: {e}")
    MOTOR_AVAILABLE = False

class MecanumCar:
    """Motor control class based on customMotor.py"""
    def __init__(self, addr=0x40, base_speed=2000, max_pwm=4095):
        if not MOTOR_AVAILABLE:
            raise Exception("Motor libraries not available")
        
        self.pwm = PCA9685(addr)
        self.pwm.set_pwm_freq(50)
        self.base = base_speed
        self.max = max_pwm
        self.is_enabled = False  # Safety feature

    def _limit(self, v):
        """Keep duty in range"""
        return max(min(v, self.max), -self.max)

    def _wheel(self, fwd_ch, rev_ch, duty):
        """Control one motor"""
        if not self.is_enabled:
            return  # Safety check
            
        if duty > 0:
            self.pwm.set_motor_pwm(rev_ch, 0)
            self.pwm.set_motor_pwm(fwd_ch, duty)
        elif duty < 0:
            self.pwm.set_motor_pwm(fwd_ch, 0)
            self.pwm.set_motor_pwm(rev_ch, -duty)
        else:  # brake/stop
            self.pwm.set_motor_pwm(fwd_ch, 4095)
            self.pwm.set_motor_pwm(rev_ch, 4095)

    def drive(self, fl, rl, fr, rr):
        """Control four motors: front-left, rear-left, front-right, rear-right"""
        if not self.is_enabled:
            return  # Safety check
            
        fl, rl, fr, rr = map(self._limit, (fl, rl, fr, rr))
        self._wheel(0, 1, fl)   # front-left
        self._wheel(3, 2, rl)   # rear-left
        self._wheel(6, 7, fr)   # front-right
        self._wheel(4, 5, rr)   # rear-right

    def stop(self):
        """Stop all motors"""
        self.drive(0, 0, 0, 0)

    def enable(self):
        """Enable motor control (safety feature)"""
        self.is_enabled = True

    def disable(self):
        """Disable motor control and stop"""
        self.stop()
        self.is_enabled = False

    def close(self):
        """Cleanup"""
        self.stop()
        if hasattr(self, 'pwm'):
            self.pwm.close()

class MotorManager:
    """Manages motor control operations including initialization, control, and safety"""
    
    def __init__(self, config):
        self.config = config
        self.motor_car = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize motor control system"""
        if not MOTOR_AVAILABLE:
            print("⚠ Motor libraries not available - motor features disabled")
            return False
        
        try:
            print("🚗 Initializing motor control system...")
            self.motor_car = MecanumCar(
                base_speed=self.config['base_speed'],
                max_pwm=self.config['max_pwm']
            )
            self.is_initialized = True
            print("✓ Motor control system initialized (DISABLED for safety)")
            print("⚠️  Use web interface to enable motors before testing")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize motors: {e}")
            return False
    
    def enable_motors(self):
        """Enable motor control"""
        if not self.is_initialized:
            return False, "Motors not initialized"
        
        self.motor_car.enable()
        print("🟢 Motors ENABLED via web interface")
        return True, "Motors enabled"
    
    def disable_motors(self):
        """Disable motor control"""
        if not self.is_initialized:
            return False, "Motors not initialized"
        
        self.motor_car.disable()
        print("⚫ Motors DISABLED via web interface")
        return True, "Motors disabled"
    
    def emergency_stop(self):
        """Emergency stop all motors"""
        if not self.is_initialized:
            return False, "Motors not initialized"
        
        self.motor_car.stop()
        print("🛑 Emergency stop triggered")
        return True, "Motors stopped"
    
    def move(self, direction, speed):
        """Move robot in specified direction"""
        if not self.is_initialized:
            return False, "Motors not initialized"
        
        if not self.motor_car.is_enabled:
            return False, "Motors disabled"
        
        # Movement mappings - CORRECTED for proper directions
        # Format: (front_left, rear_left, front_right, rear_right)
        movements = {
            'forward': (speed, speed, speed, speed),
            'backward': (-speed, -speed, -speed, -speed),
            'left': (speed, speed, -speed, -speed),      # spin left - CORRECTED
            'right': (-speed, -speed, speed, speed),     # spin right - CORRECTED
            'strafe_left': (-speed, speed, speed, -speed),
            'strafe_right': (speed, -speed, -speed, speed),
            'spin_left': (speed, speed, -speed, -speed),
            'spin_right': (-speed, -speed, speed, speed)
        }
        
        if direction in movements:
            fl, rl, fr, rr = movements[direction]
            self.motor_car.drive(fl, rl, fr, rr)
            return True, f'Moving {direction}'
        else:
            return False, 'Invalid direction'
    
    def get_status(self):
        """Get motor status information"""
        return {
            'available': MOTOR_AVAILABLE,
            'initialized': self.is_initialized,
            'enabled': self.motor_car.is_enabled if self.motor_car else False,
            'config': self.config
        }
    
    def cleanup(self):
        """Cleanup motor resources"""
        if self.motor_car:
            self.motor_car.close()
        print("✓ Motor cleanup completed") 