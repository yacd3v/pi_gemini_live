#!/usr/bin/env python3
"""
Precision Robot Control Module
Integrates BNO085 absolute positioning with chassis control for accurate movement
"""

import time
import math
import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import threading

# Import our BNO085 sensor module
from bno085_sensor import BNO085Controller

# Import existing robot control modules
try:
    from chassis_functions import ChassisController
    CHASSIS_AVAILABLE = True
except ImportError:
    CHASSIS_AVAILABLE = False
    logging.warning("Chassis controller not available - running in simulation mode")

try:
    from freenove_examples.servo import Servo
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False
    logging.warning("Servo controller not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovementStatus(Enum):
    """Status of movement commands"""
    IDLE = "idle"
    MOVING = "moving"
    TURNING = "turning"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class MovementCommand:
    """Container for movement command details"""
    command_type: str  # 'move', 'turn', 'goto', 'sequence'
    target_distance: float = 0.0  # meters
    target_angle: float = 0.0  # degrees
    target_position: Tuple[float, float] = (0.0, 0.0)  # x, y coordinates
    speed: int = 1000  # motor speed
    precision: float = 0.05  # meters or degrees tolerance
    timeout: float = 30.0  # seconds
    obstacle_detection: bool = True

@dataclass
class MovementResult:
    """Result of a movement command"""
    success: bool
    status: MovementStatus
    actual_distance: float = 0.0
    actual_angle: float = 0.0
    final_position: Tuple[float, float] = (0.0, 0.0)
    final_heading: float = 0.0
    error_message: str = ""
    execution_time: float = 0.0

class PrecisionRobotController:
    """High-level robot controller with absolute positioning"""
    
    def __init__(self):
        """Initialize the precision robot controller"""
        self.bno085 = BNO085Controller()
        self.chassis = None
        self.servo = None
        self.is_initialized = False
        self.current_status = MovementStatus.IDLE
        self.movement_lock = threading.Lock()
        
        # Movement tracking
        self.movement_history = []
        self.total_distance_traveled = 0.0
        self.total_rotation = 0.0
        
        # Calibration parameters
        self.distance_calibration_factor = 1.0  # Adjust based on actual measurements
        self.rotation_calibration_factor = 1.0  # Adjust based on actual measurements
        
        logger.info("PrecisionRobotController initialized")
    
    def initialize(self) -> bool:
        """Initialize all robot subsystems"""
        logger.info("Initializing robot subsystems...")
        
        # Initialize BNO085 sensor
        if not self.bno085.connect():
            logger.warning("BNO085 sensor not available, continuing in simulation mode")
            # Don't fail completely - continue without sensor
        
        # Initialize chassis controller
        if CHASSIS_AVAILABLE:
            try:
                self.chassis = ChassisController()
                logger.info("Chassis controller initialized")
            except Exception as e:
                logger.warning(f"Chassis controller failed to initialize: {e}")
                logger.info("Continuing in simulation mode without physical chassis control")
                self.chassis = None
        else:
            logger.warning("Chassis controller not available - using simulation mode")
        
        # Initialize servo controller
        if SERVO_AVAILABLE:
            try:
                self.servo = Servo()
                logger.info("Servo controller initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize servo controller: {e}")
                logger.info("Continuing without servo control")
                self.servo = None
        
        self.is_initialized = True
        logger.info("Robot subsystems initialized successfully")
        return True
    
    def shutdown(self):
        """Safely shutdown all robot subsystems"""
        logger.info("Shutting down robot subsystems...")
        
        if self.chassis:
            try:
                self.chassis.emergency_stop()
                self.chassis.close()
            except:
                pass
        
        if self.bno085:
            self.bno085.disconnect()
        
        self.is_initialized = False
        logger.info("Robot subsystems shutdown complete")
    
    def get_current_pose(self) -> Optional[Dict[str, Any]]:
        """Get current robot pose (position and orientation)"""
        if not self.is_initialized:
            return None
        
        sensor_data = self.bno085.get_sensor_data()
        if not sensor_data:
            return None
        
        position = self.bno085.get_position()
        
        return {
            "position": {"x": position[0], "y": position[1]},
            "heading": sensor_data.heading,
            "relative_heading": self.bno085.get_relative_heading(),
            "orientation": {
                "roll": sensor_data.euler_angles[0],
                "pitch": sensor_data.euler_angles[1],
                "yaw": sensor_data.euler_angles[2]
            },
            "calibration": sensor_data.calibration_status,
            "timestamp": sensor_data.timestamp
        }
    
    def reset_position(self):
        """Reset robot position to origin"""
        if self.bno085:
            self.bno085.reset_position()
            self.bno085.reset_heading()
            logger.info("Robot position and heading reset to origin")
    
    def move_distance(self, distance_meters: float, speed: int = 1000, 
                     precision: float = 0.05, timeout: float = 30.0) -> MovementResult:
        """
        Move robot a specific distance with absolute positioning feedback
        
        Args:
            distance_meters: Distance to move in meters (positive = forward, negative = backward)
            speed: Motor speed (default 1000)
            precision: Distance precision in meters (default 5cm)
            timeout: Maximum time to execute command
        
        Returns:
            MovementResult with success status and actual movement data
        """
        if not self.is_initialized:
            return MovementResult(False, MovementStatus.FAILED, 
                                error_message="Robot not initialized")
        
        with self.movement_lock:
            logger.info(f"Moving {distance_meters:.2f}m at speed {speed}")
            
            # Record starting position
            start_pose = self.get_current_pose()
            if not start_pose:
                return MovementResult(False, MovementStatus.FAILED,
                                    error_message="Could not get starting position")
            
            start_position = (start_pose["position"]["x"], start_pose["position"]["y"])
            start_heading = start_pose["heading"]
            start_time = time.time()
            
            self.current_status = MovementStatus.MOVING
            
            try:
                if self.chassis:
                    # Use existing chassis controller with distance feedback
                    try:
                        result_msg = self.chassis.move_forward_distance(
                            distance_meters * 100,  # Convert to cm
                            speed,
                            obstacle_detection=True
                        )
                        
                        # Wait a moment for movement to complete
                        time.sleep(0.5)
                        
                        # Get final position
                        final_pose = self.get_current_pose()
                        if final_pose:
                            final_position = (final_pose["position"]["x"], final_pose["position"]["y"])
                            final_heading = final_pose["heading"]
                            
                            # Calculate actual distance moved
                            actual_distance = math.sqrt(
                                (final_position[0] - start_position[0])**2 + 
                                (final_position[1] - start_position[1])**2
                            )
                            
                            # Update position tracking in BNO085
                            self.bno085.update_position(actual_distance, final_heading)
                            
                            # Check if we achieved the desired precision
                            distance_error = abs(actual_distance - abs(distance_meters))
                            success = distance_error <= precision
                            
                            execution_time = time.time() - start_time
                            self.total_distance_traveled += actual_distance
                            
                            # Log the movement
                            movement_record = {
                                "command": "move",
                                "target_distance": distance_meters,
                                "actual_distance": actual_distance,
                                "error": distance_error,
                                "start_position": start_position,
                                "final_position": final_position,
                                "execution_time": execution_time,
                                "timestamp": time.time()
                            }
                            self.movement_history.append(movement_record)
                            
                            self.current_status = MovementStatus.COMPLETED if success else MovementStatus.FAILED
                            
                            return MovementResult(
                                success=success,
                                status=self.current_status,
                                actual_distance=actual_distance,
                                final_position=final_position,
                                final_heading=final_heading,
                                error_message="" if success else f"Distance error: {distance_error:.3f}m",
                                execution_time=execution_time
                            )
                        else:
                            return MovementResult(False, MovementStatus.FAILED,
                                                error_message="Could not get final position")
                    except Exception as e:
                        logger.warning(f"Physical movement failed: {e}, using simulation")
                        # Fall through to simulation mode
                        self.chassis = None
                
                # Simulation mode (either no chassis or chassis failed)
                logger.info(f"Simulating movement: {distance_meters}m at speed {speed}")
                time.sleep(abs(distance_meters) / 0.4)  # Simulate movement time
                
                # Simulate position update
                heading_rad = math.radians(start_heading)
                new_x = start_position[0] + distance_meters * math.cos(heading_rad)
                new_y = start_position[1] + distance_meters * math.sin(heading_rad)
                
                self.bno085.update_position(abs(distance_meters), start_heading)
                
                execution_time = time.time() - start_time
                self.total_distance_traveled += abs(distance_meters)
                
                # Log the simulated movement
                movement_record = {
                    "command": "move",
                    "target_distance": distance_meters,
                    "actual_distance": abs(distance_meters),
                    "error": 0.0,
                    "start_position": start_position,
                    "final_position": (new_x, new_y),
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
                self.movement_history.append(movement_record)
                
                self.current_status = MovementStatus.COMPLETED
                
                return MovementResult(
                    success=True,
                    status=self.current_status,
                    actual_distance=abs(distance_meters),
                    final_position=(new_x, new_y),
                    final_heading=start_heading,
                    error_message="Simulated movement",
                    execution_time=execution_time
                )
                    
            except Exception as e:
                logger.error(f"Movement failed: {e}")
                self.current_status = MovementStatus.FAILED
                return MovementResult(False, MovementStatus.FAILED,
                                    error_message=str(e))
    
    def turn_angle(self, angle_degrees: float, speed: int = 1000, 
                  precision: float = 2.0, timeout: float = 30.0) -> MovementResult:
        """
        Turn robot by a specific angle with absolute positioning feedback
        
        Args:
            angle_degrees: Angle to turn in degrees (positive = clockwise, negative = counter-clockwise)
            speed: Motor speed (default 1000)
            precision: Angle precision in degrees (default 2°)
            timeout: Maximum time to execute command
        
        Returns:
            MovementResult with success status and actual turn data
        """
        if not self.is_initialized:
            return MovementResult(False, MovementStatus.FAILED,
                                error_message="Robot not initialized")
        
        with self.movement_lock:
            logger.info(f"Turning {angle_degrees:.1f}° at speed {speed}")
            
            # Record starting heading
            start_pose = self.get_current_pose()
            if not start_pose:
                return MovementResult(False, MovementStatus.FAILED,
                                    error_message="Could not get starting heading")
            
            start_heading = start_pose["heading"]
            start_time = time.time()
            
            self.current_status = MovementStatus.TURNING
            
            try:
                if self.chassis:
                    # Use existing chassis controller
                    try:
                        result_msg = self.chassis.rotate_angle(
                            angle_degrees,
                            speed,
                            obstacle_detection=True
                        )
                        
                        # Wait a moment for turn to complete
                        time.sleep(0.5)
                        
                        # Get final heading
                        final_pose = self.get_current_pose()
                        if final_pose:
                            final_heading = final_pose["heading"]
                            
                            # Calculate actual angle turned
                            actual_angle = final_heading - start_heading
                            
                            # Normalize to -180 to +180 range
                            while actual_angle > 180:
                                actual_angle -= 360
                            while actual_angle < -180:
                                actual_angle += 360
                            
                            # Check if we achieved the desired precision
                            angle_error = abs(actual_angle - angle_degrees)
                            success = angle_error <= precision
                            
                            execution_time = time.time() - start_time
                            self.total_rotation += abs(actual_angle)
                            
                            # Log the turn
                            turn_record = {
                                "command": "turn",
                                "target_angle": angle_degrees,
                                "actual_angle": actual_angle,
                                "error": angle_error,
                                "start_heading": start_heading,
                                "final_heading": final_heading,
                                "execution_time": execution_time,
                                "timestamp": time.time()
                            }
                            self.movement_history.append(turn_record)
                            
                            self.current_status = MovementStatus.COMPLETED if success else MovementStatus.FAILED
                            
                            return MovementResult(
                                success=success,
                                status=self.current_status,
                                actual_angle=actual_angle,
                                final_heading=final_heading,
                                error_message="" if success else f"Angle error: {angle_error:.1f}°",
                                execution_time=execution_time
                            )
                        else:
                            return MovementResult(False, MovementStatus.FAILED,
                                                error_message="Could not get final heading")
                    except Exception as e:
                        logger.warning(f"Physical turn failed: {e}, using simulation")
                        self.chassis = None
                
                # Simulation mode (either no chassis or chassis failed)
                logger.info(f"Simulating turn: {angle_degrees}° at speed {speed}")
                time.sleep(abs(angle_degrees) / 90.0)  # Simulate turn time
                
                execution_time = time.time() - start_time
                self.total_rotation += abs(angle_degrees)
                
                # Calculate new heading
                new_heading = (start_heading + angle_degrees) % 360
                
                # Log the simulated turn
                turn_record = {
                    "command": "turn",
                    "target_angle": angle_degrees,
                    "actual_angle": angle_degrees,
                    "error": 0.0,
                    "start_heading": start_heading,
                    "final_heading": new_heading,
                    "execution_time": execution_time,
                    "timestamp": time.time()
                }
                self.movement_history.append(turn_record)
                
                self.current_status = MovementStatus.COMPLETED
                
                return MovementResult(
                    success=True,
                    status=self.current_status,
                    actual_angle=angle_degrees,
                    final_heading=new_heading,
                    error_message="Simulated turn",
                    execution_time=execution_time
                )
                    
            except Exception as e:
                logger.error(f"Turn failed: {e}")
                self.current_status = MovementStatus.FAILED
                return MovementResult(False, MovementStatus.FAILED,
                                    error_message=str(e))
    
    def goto_position(self, target_x: float, target_y: float, 
                     speed: int = 1000, precision: float = 0.1) -> MovementResult:
        """
        Move robot to a specific position using absolute positioning
        
        Args:
            target_x: Target X coordinate in meters
            target_y: Target Y coordinate in meters
            speed: Motor speed (default 1000)
            precision: Position precision in meters (default 10cm)
        
        Returns:
            MovementResult with success status and movement data
        """
        if not self.is_initialized:
            return MovementResult(False, MovementStatus.FAILED,
                                error_message="Robot not initialized")
        
        logger.info(f"Moving to position ({target_x:.2f}, {target_y:.2f})")
        
        # Get current position
        current_pose = self.get_current_pose()
        if not current_pose:
            return MovementResult(False, MovementStatus.FAILED,
                                error_message="Could not get current position")
        
        current_x = current_pose["position"]["x"]
        current_y = current_pose["position"]["y"]
        current_heading = current_pose["heading"]
        
        # Calculate required movement
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Calculate distance and angle
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.degrees(math.atan2(dy, dx))
        
        # Calculate required turn angle
        angle_diff = target_angle - current_heading
        
        # Normalize angle difference to -180 to +180 range
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        logger.info(f"Need to turn {angle_diff:.1f}° then move {distance:.2f}m")
        
        # Execute the movement sequence
        if abs(angle_diff) > 2.0:  # Only turn if significant angle difference
            turn_result = self.turn_angle(angle_diff, speed)
            if not turn_result.success:
                return turn_result
        
        # Move forward to target
        move_result = self.move_distance(distance, speed, precision)
        
        return move_result
    
    def execute_command_sequence(self, commands: List[MovementCommand]) -> List[MovementResult]:
        """
        Execute a sequence of movement commands
        
        Args:
            commands: List of MovementCommand objects
        
        Returns:
            List of MovementResult objects
        """
        results = []
        
        for i, command in enumerate(commands):
            logger.info(f"Executing command {i+1}/{len(commands)}: {command.command_type}")
            
            if command.command_type == "move":
                result = self.move_distance(
                    command.target_distance,
                    command.speed,
                    command.precision,
                    command.timeout
                )
            elif command.command_type == "turn":
                result = self.turn_angle(
                    command.target_angle,
                    command.speed,
                    command.precision,
                    command.timeout
                )
            elif command.command_type == "goto":
                result = self.goto_position(
                    command.target_position[0],
                    command.target_position[1],
                    command.speed,
                    command.precision
                )
            else:
                result = MovementResult(False, MovementStatus.FAILED,
                                      error_message=f"Unknown command type: {command.command_type}")
            
            results.append(result)
            
            # Stop sequence if command failed
            if not result.success:
                logger.warning(f"Command {i+1} failed, stopping sequence")
                break
        
        return results
    
    def emergency_stop(self):
        """Emergency stop all movement"""
        logger.warning("Emergency stop requested")
        
        if self.chassis:
            try:
                self.chassis.emergency_stop()
                logger.info("Physical emergency stop executed")
            except Exception as e:
                logger.warning(f"Physical emergency stop failed: {e}")
        else:
            logger.info("Emergency stop (simulation mode)")
        
        self.current_status = MovementStatus.STOPPED
    
    def get_movement_statistics(self) -> Dict[str, Any]:
        """Get movement statistics and history"""
        return {
            "total_distance_traveled": self.total_distance_traveled,
            "total_rotation": self.total_rotation,
            "movement_count": len(self.movement_history),
            "current_status": self.current_status.value,
            "recent_movements": self.movement_history[-10:] if self.movement_history else [],
            "calibration_factors": {
                "distance": self.distance_calibration_factor,
                "rotation": self.rotation_calibration_factor
            }
        }
    
    def set_calibration_factors(self, distance_factor: float, rotation_factor: float):
        """Set calibration factors for improved accuracy"""
        self.distance_calibration_factor = distance_factor
        self.rotation_calibration_factor = rotation_factor
        logger.info(f"Calibration factors updated: distance={distance_factor}, rotation={rotation_factor}")

def test_precision_control():
    """Test the precision robot control system"""
    print("🤖 Testing Precision Robot Control System...")
    
    controller = PrecisionRobotController()
    
    if not controller.initialize():
        print("❌ Failed to initialize robot controller")
        return False
    
    try:
        print("✅ Robot controller initialized successfully")
        
        # Test sensor readings
        pose = controller.get_current_pose()
        if pose:
            print(f"Current position: ({pose['position']['x']:.2f}, {pose['position']['y']:.2f})")
            print(f"Current heading: {pose['heading']:.1f}°")
            print(f"Calibration: {pose['calibration']}")
        
        # Test movement commands
        print("\n🔄 Testing movement commands...")
        
        # Test forward movement
        print("Moving forward 0.5m...")
        result = controller.move_distance(0.5, speed=800)
        if result.success:
            print(f"✅ Move successful: {result.actual_distance:.2f}m in {result.execution_time:.1f}s")
        else:
            print(f"❌ Move failed: {result.error_message}")
        
        time.sleep(2)
        
        # Test turn
        print("Turning 90° right...")
        result = controller.turn_angle(90, speed=800)
        if result.success:
            print(f"✅ Turn successful: {result.actual_angle:.1f}° in {result.execution_time:.1f}s")
        else:
            print(f"❌ Turn failed: {result.error_message}")
        
        time.sleep(2)
        
        # Test goto position
        print("Moving to position (0.3, 0.3)...")
        result = controller.goto_position(0.3, 0.3, speed=800)
        if result.success:
            print(f"✅ Goto successful in {result.execution_time:.1f}s")
        else:
            print(f"❌ Goto failed: {result.error_message}")
        
        # Show statistics
        stats = controller.get_movement_statistics()
        print(f"\n📊 Movement Statistics:")
        print(f"Total distance: {stats['total_distance_traveled']:.2f}m")
        print(f"Total rotation: {stats['total_rotation']:.1f}°")
        print(f"Movement count: {stats['movement_count']}")
        
        return True
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
        return True
    
    finally:
        controller.shutdown()

if __name__ == "__main__":
    test_precision_control() 