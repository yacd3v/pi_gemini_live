import time
import threading
import sys
import os

# Add the freenove_examples directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'freenove_examples'))

from freenove_examples.motor import Ordinary_Car
from freenove_examples.ultrasonic import Ultrasonic

class ChassisController:
    def __init__(self):
        self.pwm = Ordinary_Car()
        self.ultrasonic = Ultrasonic()
        self.stop_requested = False
        self.is_moving = False
        self.emergency_stop_distance = 15  # Hard stop if obstacle within 15cm
        self.warning_distance = 30  # Slow down if obstacle within 30cm
        self.sensor_poll_delay = 0.05  # Faster sensor polling (50ms instead of 100ms)
        
        # Rotation calibration - you'll need to measure and adjust this
        self.degrees_per_second = 140  # Fine-tuned: averaging multiple tests for better accuracy
        
    def move_forward_distance(self, target_cm, speed=1000, obstacle_detection=True):
        """
        Move the car forward for approximately the specified distance in centimeters.
        
        Args:
            target_cm (float): Distance to travel in centimeters
            speed (int): Motor speed (default 1000, range typically -2000 to 2000)
            obstacle_detection (bool): Enable ultrasonic obstacle detection
        
        Returns:
            str: Status message indicating how the movement ended
        """
        print(f"Starting movement: {target_cm}cm forward at speed {speed}")
        
        # Reset stop flag
        self.stop_requested = False
        self.is_moving = True
        
        # Rough calibration: adjust this value based on your measurements
        # Fine-tuned based on 70cm test (actual 80cm): robot moves ~40cm per second at speed 1000
        cm_per_second = 40.0  # Fine-tuned from 35.0 based on latest floor measurement
        estimated_time = target_cm / cm_per_second
        
        print(f"Estimated travel time: {estimated_time:.2f} seconds")
        
        try:
            start_time = time.time()
            current_speed = speed
            
            # Start moving forward
            self.pwm.set_motor_model(current_speed, current_speed, current_speed, current_speed)
            print("Car is moving forward...")
            
            while self.is_moving and not self.stop_requested:
                # Check elapsed time
                elapsed_time = time.time() - start_time
                if elapsed_time >= estimated_time:
                    self.stop_motors()
                    return f"Target distance reached! Traveled approximately {target_cm}cm in {elapsed_time:.2f}s"
                
                # Check for obstacles if enabled
                if obstacle_detection:
                    distance = self.ultrasonic.get_distance()
                    
                    if distance is not None:
                        # Emergency stop zone
                        if distance <= self.emergency_stop_distance:
                            print(f"EMERGENCY STOP! Obstacle at {distance}cm (emergency threshold: {self.emergency_stop_distance}cm)")
                            self.stop_motors()
                            return f"Emergency stop! Obstacle detected at {distance}cm after {elapsed_time:.2f}s"
                        
                        # Warning zone - reduce speed
                        elif distance <= self.warning_distance:
                            reduced_speed = int(speed * 0.4)  # Reduce to 40% of original speed
                            if current_speed != reduced_speed:
                                print(f"WARNING: Obstacle at {distance}cm - reducing speed to {reduced_speed}")
                                current_speed = reduced_speed
                                self.pwm.set_motor_model(current_speed, current_speed, current_speed, current_speed)
                            else:
                                print(f"Ultrasonic reading: {distance}cm (warning zone: <{self.warning_distance}cm, emergency: <{self.emergency_stop_distance}cm)")
                        
                        # Safe zone
                        else:
                            if current_speed != speed:
                                print(f"Clear path at {distance}cm - resuming normal speed")
                                current_speed = speed
                                self.pwm.set_motor_model(current_speed, current_speed, current_speed, current_speed)
                            else:
                                print(f"Ultrasonic reading: {distance}cm (clear - emergency: <{self.emergency_stop_distance}cm)")
                    else:
                        print("Ultrasonic reading: No reading")
                
                # Faster polling for better responsiveness
                time.sleep(self.sensor_poll_delay)
            
            # If we're here, stop was requested
            self.stop_motors()
            elapsed_time = time.time() - start_time
            return f"Movement stopped by user after {elapsed_time:.2f}s"
            
        except KeyboardInterrupt:
            self.stop_motors()
            return "Movement stopped by keyboard interrupt"
        except Exception as e:
            self.stop_motors()
            return f"Movement stopped due to error: {str(e)}"
    
    def stop_motors(self):
        """Safely stop all motors"""
        self.is_moving = False
        self.pwm.set_motor_model(0, 0, 0, 0)
        print("Motors stopped")
    
    def emergency_stop(self):
        """Emergency stop function that can be called from anywhere"""
        print("EMERGENCY STOP REQUESTED!")
        self.stop_requested = True
        self.stop_motors()
    
    def calibrate_distance(self, test_distance=50, speed=1000):
        """
        Helper function to calibrate the distance-to-time conversion.
        Run this with a known distance and measure the actual distance traveled.
        
        Args:
            test_distance (float): Distance to attempt in cm
            speed (int): Motor speed to use for calibration
        """
        print(f"Starting calibration run: {test_distance}cm at speed {speed}")
        print("Measure the actual distance traveled and update cm_per_second in move_forward_distance()")
        
        start_time = time.time()
        
        try:
            self.pwm.set_motor_model(speed, speed, speed, speed)
            
            # Simple time-based movement for calibration
            cm_per_second = 40.0  # Fine-tuned estimate from latest tests
            run_time = test_distance / cm_per_second
            
            print(f"Running for {run_time:.2f} seconds...")
            time.sleep(run_time)
            
            self.stop_motors()
            elapsed_time = time.time() - start_time
            print(f"Calibration complete. Ran for {elapsed_time:.2f} seconds")
            print(f"If actual distance != {test_distance}cm, update cm_per_second value")
            
        except KeyboardInterrupt:
            self.stop_motors()
            print("Calibration stopped by keyboard interrupt")
    
    def calibrate_rotation(self, test_degrees=90, speed=1000):
        """
        Helper function to calibrate the rotation angle-to-time conversion.
        Run this with a known angle and measure the actual rotation.
        
        Args:
            test_degrees (float): Angle to attempt in degrees
            speed (int): Motor speed to use for calibration
        """
        print(f"Starting rotation calibration: {test_degrees}° at speed {speed}")
        print("Measure the actual rotation angle and update degrees_per_second in rotate_angle()")
        
        start_time = time.time()
        direction = "clockwise" if test_degrees > 0 else "counter-clockwise"
        abs_degrees = abs(test_degrees)
        
        try:
            # Set motor pattern for rotation
            if test_degrees > 0:  # Clockwise
                self.pwm.set_motor_model(speed, speed, -speed, -speed)
            else:  # Counter-clockwise
                self.pwm.set_motor_model(-speed, -speed, speed, speed)
            
            # Simple time-based rotation for calibration
            run_time = abs_degrees / self.degrees_per_second
            
            print(f"Running {direction} for {run_time:.2f} seconds...")
            time.sleep(run_time)
            
            self.stop_motors()
            elapsed_time = time.time() - start_time
            print(f"Calibration complete. Rotated for {elapsed_time:.2f} seconds")
            print(f"If actual angle != {abs_degrees}°, update degrees_per_second value")
            
        except KeyboardInterrupt:
            self.stop_motors()
            print("Rotation calibration stopped by keyboard interrupt")
    
    def close(self):
        """Clean up resources"""
        self.stop_motors()
        self.pwm.close()
        self.ultrasonic.close()
        print("Chassis controller closed")

    def test_ultrasonic(self, duration=10):
        """
        Test the ultrasonic sensor for a specified duration without moving the car.
        
        Args:
            duration (int): How long to test in seconds (default 10)
        """
        print(f"Testing ultrasonic sensor for {duration} seconds...")
        print(f"Emergency stop threshold: {self.emergency_stop_distance}cm")
        print(f"Warning/slow speed threshold: {self.warning_distance}cm")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                distance = self.ultrasonic.get_distance()
                elapsed = time.time() - start_time
                
                if distance is not None:
                    if distance <= self.emergency_stop_distance:
                        status = "EMERGENCY STOP!"
                    elif distance <= self.warning_distance:
                        status = "WARNING - Slow down"
                    else:
                        status = "Clear"
                    print(f"[{elapsed:5.1f}s] Distance: {distance:6.1f}cm - {status}")
                else:
                    print(f"[{elapsed:5.1f}s] Distance: No reading")
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nUltrasonic test stopped by user")
        
        print("Ultrasonic test completed")

    def rotate_angle(self, target_degrees, speed=1000, obstacle_detection=True):
        """
        Rotate the robot by a specified angle in degrees.
        
        Args:
            target_degrees (float): Angle to rotate (-360 to +360 degrees)
                                   Positive = clockwise, Negative = counter-clockwise
            speed (int): Motor speed (default 1000, range typically -2000 to 2000)
            obstacle_detection (bool): Enable ultrasonic obstacle detection during rotation
        
        Returns:
            str: Status message indicating how the rotation ended
        """
        # Validate angle range
        if abs(target_degrees) > 360:
            return "Error: Angle must be between -360 and +360 degrees"
        
        if target_degrees == 0:
            return "No rotation needed (0 degrees)"
        
        direction = "clockwise" if target_degrees > 0 else "counter-clockwise"
        abs_degrees = abs(target_degrees)
        
        print(f"Starting rotation: {abs_degrees}° {direction} at speed {speed}")
        
        # Reset stop flag
        self.stop_requested = False
        self.is_moving = True
        
        # Calculate rotation time based on calibration
        estimated_time = abs_degrees / self.degrees_per_second
        print(f"Estimated rotation time: {estimated_time:.2f} seconds")
        
        try:
            start_time = time.time()
            
            # Set motor pattern for rotation
            if target_degrees > 0:  # Clockwise (right turn)
                self.pwm.set_motor_model(speed, speed, -speed, -speed)
                print("Rotating clockwise...")
            else:  # Counter-clockwise (left turn)
                self.pwm.set_motor_model(-speed, -speed, speed, speed)
                print("Rotating counter-clockwise...")
            
            while self.is_moving and not self.stop_requested:
                # Check elapsed time
                elapsed_time = time.time() - start_time
                if elapsed_time >= estimated_time:
                    self.stop_motors()
                    return f"Target rotation reached! Rotated {abs_degrees}° {direction} in {elapsed_time:.2f}s"
                
                # Optional obstacle detection during rotation
                if obstacle_detection:
                    distance = self.ultrasonic.get_distance()
                    if distance is not None:
                        if distance <= self.emergency_stop_distance:
                            print(f"EMERGENCY STOP! Obstacle at {distance}cm during rotation")
                            self.stop_motors()
                            return f"Emergency stop during rotation! Obstacle at {distance}cm after {elapsed_time:.2f}s"
                        elif distance <= self.warning_distance:
                            print(f"Warning: Obstacle at {distance}cm during rotation")
                
                # Faster polling for better control
                time.sleep(self.sensor_poll_delay)
                
            # If we're here, stop was requested
            self.stop_motors()
            elapsed_time = time.time() - start_time
            return f"Rotation stopped by user after {elapsed_time:.2f}s"
            
        except KeyboardInterrupt:
            self.stop_motors()
            return "Rotation stopped by keyboard interrupt"
        except Exception as e:
            self.stop_motors()
            return f"Rotation stopped due to error: {str(e)}"

# Convenience function for direct use
def move_car_forward(distance_cm, speed=1000, obstacle_detection=True):
    """
    Convenience function to move the car forward for a specified distance.
    
    Args:
        distance_cm (float): Distance to travel in centimeters
        speed (int): Motor speed (default 1000)
        obstacle_detection (bool): Enable obstacle detection
    
    Returns:
        str: Status message
    """
    controller = ChassisController()
    try:
        result = controller.move_forward_distance(distance_cm, speed, obstacle_detection)
        print(result)
        return result
    finally:
        controller.close()

def rotate_car(degrees, speed=1000, obstacle_detection=True):
    """
    Convenience function to rotate the car by a specified angle.
    
    Args:
        degrees (float): Angle to rotate (-360 to +360 degrees)
                        Positive = clockwise, Negative = counter-clockwise
        speed (int): Motor speed (default 1000)
        obstacle_detection (bool): Enable obstacle detection during rotation
    
    Returns:
        str: Status message
    """
    controller = ChassisController()
    try:
        result = controller.rotate_angle(degrees, speed, obstacle_detection)
        print(result)
        return result
    finally:
        controller.close()

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python chassis_functions.py <distance_cm> [speed] [obstacle_detection]")
        print("  python chassis_functions.py rotate <degrees> [speed] [obstacle_detection]")
        print("  python chassis_functions.py calibrate [test_distance] [speed]")
        print("  python chassis_functions.py calibrate_rotation [test_degrees] [speed]")
        print("  python chassis_functions.py test_ultrasonic [duration]")
        print("Examples:")
        print("  python chassis_functions.py 30        # Move 30cm forward with obstacle detection")
        print("  python chassis_functions.py 50 1500   # Move 50cm at speed 1500")
        print("  python chassis_functions.py rotate 90 # Rotate 90° clockwise")
        print("  python chassis_functions.py rotate -180 1200 # Rotate 180° counter-clockwise at speed 1200")
        print("  python chassis_functions.py calibrate 50  # Calibration run for 50cm")
        print("  python chassis_functions.py calibrate_rotation 90  # Rotation calibration for 90°")
        print("  python chassis_functions.py test_ultrasonic 15  # Test ultrasonic for 15 seconds")
        sys.exit(1)
    
    if sys.argv[1] == "calibrate":
        test_dist = float(sys.argv[2]) if len(sys.argv) > 2 else 50
        test_speed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        
        controller = ChassisController()
        try:
            controller.calibrate_distance(test_dist, test_speed)
        finally:
            controller.close()
    elif sys.argv[1] == "calibrate_rotation":
        test_angle = float(sys.argv[2]) if len(sys.argv) > 2 else 90
        test_speed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        
        controller = ChassisController()
        try:
            controller.calibrate_rotation(test_angle, test_speed)
        finally:
            controller.close()
    elif sys.argv[1] == "test_ultrasonic":
        test_duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        
        controller = ChassisController()
        try:
            controller.test_ultrasonic(test_duration)
        finally:
            controller.close()
    elif sys.argv[1] == "rotate":
        if len(sys.argv) < 3:
            print("Error: rotate command requires angle parameter")
            print("Usage: python chassis_functions.py rotate <degrees> [speed] [obstacle_detection]")
            sys.exit(1)
        
        angle = float(sys.argv[2])
        speed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        obstacle_det = sys.argv[4].lower() != 'false' if len(sys.argv) > 4 else True
        
        rotate_car(angle, speed, obstacle_det)
    else:
        distance = float(sys.argv[1])
        speed = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        obstacle_det = sys.argv[3].lower() != 'false' if len(sys.argv) > 3 else True
        
        move_car_forward(distance, speed, obstacle_det)
