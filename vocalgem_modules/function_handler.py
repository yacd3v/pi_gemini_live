"""
Function Handler Module for VocalGem robot
Handles all tool functions that can be called by Gemini
"""

import time
import smbus2 as smbus
import asyncio
import websockets
from .config import I2C_ADDR, PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX

class FunctionHandler:
    """Handles all tool functions that can be called by Gemini"""
    
    def __init__(self, face_tracker, display_manager, sleep_requested_event, audio_manager):
        self.face_tracker = face_tracker
        self.display_manager = display_manager
        self.sleep_requested_event = sleep_requested_event
        self.audio_manager = audio_manager
        
        # Servo state
        if self.face_tracker.servo:
            self.servo = self.face_tracker.servo
            self.current_pan_angle = self.face_tracker.current_pan_angle
            self.current_tilt_angle = self.face_tracker.current_tilt_angle
            print("[FunctionHandler] Using FaceTracker servo for camera movement.")
        else:
            # Fallback servo initialization for manual movement only
            try:
                from freenove_examples.servo import Servo
                self.servo = Servo()
                self.current_pan_angle = 90  # Start at center pan
                self.current_tilt_angle = 50  # Start at 50° to look up from floor level
                self.servo.set_servo_pwm('0', self.current_pan_angle) # Pan servo
                self.servo.set_servo_pwm('1', self.current_tilt_angle) # Tilt servo
                print("[FunctionHandler] Servos initialized for manual movement only.")
            except Exception as e:
                self.servo = None
                print(f"[FunctionHandler] Error initializing servos: {e}. Camera movement will be disabled.")
                # Fallback: set angles so subsequent logic doesn't error if servo is None
                self.current_pan_angle = 90
                self.current_tilt_angle = 50

        self.current_speaking_emotion = "normal" # Default speaking emotion

        # Initialize available functions and map them
        self.available_functions = [
            self.get_time, self.get_date, self.set_display_brightness,
            self.get_battery_level, self.go_to_sleep, self.move_camera,
            self.set_emotion, self.toggle_face_tracking
        ]
        self.functions_map = {func.__name__: func for func in self.available_functions}

    def get_time(self):
        """Get the current time."""
        current_time = time.strftime("%H:%M:%S")
        return f"The current time is {current_time}"

    def get_date(self):
        """Get today's date."""
        current_date = time.strftime("%Y-%m-%d")
        return f"Today's date is {current_date}"

    def set_display_brightness(self, brightness: float):
        """Set the display brightness.
        
        Args:
            brightness: A value between 0.0 and 1.0 for display brightness
        """
        if brightness < 0 or brightness > 1:
            return "Brightness must be between 0 and 1"
        # Access disp directly as it's an instance variable
        if self.display_manager.disp is not None:
            self.display_manager.disp.bl_DutyCycle(int(brightness * 100)) 
            return f"Display brightness set to {brightness*100}%"
        else:
            return "Display is not available, brightness setting skipped"

    def get_battery_level(self):
        """Get the current battery level percentage."""
        bus = None
        try:
            bus = smbus.SMBus(1)  # 0 for RPi 1, 1 for RPi 2,3,4
            data = bus.read_i2c_block_data(I2C_ADDR, 0x20, 6)
            battery_percent = int(data[4] | data[5] << 8)
            return f"The current battery level is {battery_percent}%"
        except FileNotFoundError:
            return "Error: I2C bus not found. Ensure I2C is enabled and the device is connected."
        except OSError as e:
            if e.errno == 121: # Remote I/O error (device not found at address)
                return f"Error: UPS device not found at address {hex(I2C_ADDR)}. Please check the connection."
            return f"Error reading battery level: {e}"
        except Exception as e:
            return f"An unexpected error occurred while reading battery level: {e}"
        finally:
            if bus:
                bus.close()

    async def go_to_sleep(self):
        """Instructs the assistant to go to sleep and await wake word."""
        print("[GoToSleep] Initiated.")
        
        # Stop audio streams immediately to prevent QueueFull errors
        try:
            if self.audio_manager.input_stream and self.audio_manager.input_stream.is_active():
                self.audio_manager.input_stream.stop_stream()
                print("[GoToSleep] Input stream stopped")
            if self.audio_manager.output_stream and self.audio_manager.output_stream.is_active():
                self.audio_manager.output_stream.stop_stream()
                print("[GoToSleep] Output stream stopped")
        except Exception as e:
            print(f"[GoToSleep] Error stopping streams: {e}")
        
        # Now set the sleep event
        self.sleep_requested_event.set()
        print("[GoToSleep] Sleep event SET.")
        
        # Signal other tasks to wind down if they are waiting on queues
        # Use try_put_nowait to avoid blocking if queues are full (shouldn't happen often on shutdown)
        try:
            self.audio_manager.audio_out_q.put_nowait(None) 
            print("[GoToSleep] Sentinel PUSHED to audio_out_q.")
        except asyncio.QueueFull:
            print("[GoToSleep] audio_out_q was full, sentinel not pushed.")
        try:
            self.audio_manager.audio_in_q.put_nowait(None)
            print("[GoToSleep] Sentinel PUSHED to audio_in_q.")
        except asyncio.QueueFull:
            print("[GoToSleep] audio_in_q was full, sentinel not pushed.")
            
        print("[GoToSleep] Method finished.")
        return "Going to sleep. Say 'Salut Karl' to wake me up."

    def set_emotion(self, emotion: str):
        """Sets the speaking animation based on the provided emotion."""
        supported_emotions = ["normal", "furious", "crying"]
        if emotion in supported_emotions:
            self.current_speaking_emotion = emotion
            return f"Emotion set to {emotion}. Karl will now use the {emotion} speaking animation."
        else:
            return f"Error: Emotion '{emotion}' is not supported. Supported emotions are: {', '.join(supported_emotions)}."

    def toggle_face_tracking(self, enabled: bool):
        """Enable or disable automatic face tracking."""
        return self.face_tracker.toggle_tracking(enabled)

    def move_camera(self, pan_relative_angle: float = 0.0, tilt_relative_angle: float = 0.0):
        """Pans or tilts the camera by a specified number of degrees relative to the current position.

        Args:
            pan_relative_angle (float): Degrees to pan. Positive pans left, negative pans right.
            tilt_relative_angle (float): Degrees to tilt. Positive tilts up, negative tilts down.
        """
        if not self.servo:
            return "Camera control is disabled due to an initialization error."

        pan_changed = False
        tilt_changed = False

        # Calculate and clamp pan angle
        if pan_relative_angle != 0.0:
            # Invert pan direction for more intuitive control
            # When you see someone to the camera's right, pan left (negative) to center them
            # When asking to "look at me" when you're on the right, pan left (negative)
            # Therefore we need to INVERT pan_relative_angle - positive becomes negative
            new_pan_angle = self.current_pan_angle - pan_relative_angle  # Invert direction
            clamped_pan_angle = max(PAN_MIN, min(PAN_MAX, new_pan_angle))
            if clamped_pan_angle != self.current_pan_angle:
                self.current_pan_angle = clamped_pan_angle
                self.servo.set_servo_pwm('0', int(self.current_pan_angle))
                pan_changed = True
            print(f"[MoveCamera] Pan: current={self.current_pan_angle}, requested_rel={pan_relative_angle}, new_abs_target={new_pan_angle}, clamped={clamped_pan_angle}")

        # Calculate and clamp tilt angle
        # Positive tilt_relative_angle means "up", which corresponds to a *decrease* in servo angle value
        if tilt_relative_angle != 0.0:
            new_tilt_angle = self.current_tilt_angle - tilt_relative_angle # Subtract because lower angle = up
            clamped_tilt_angle = max(TILT_MIN, min(TILT_MAX, new_tilt_angle))
            if clamped_tilt_angle != self.current_tilt_angle:
                self.current_tilt_angle = clamped_tilt_angle
                self.servo.set_servo_pwm('1', int(self.current_tilt_angle))
                tilt_changed = True
            print(f"[MoveCamera] Tilt: current={self.current_tilt_angle}, requested_rel={tilt_relative_angle}, new_abs_target={new_tilt_angle}, clamped={clamped_tilt_angle}")

        if not pan_changed and not tilt_changed:
            return f"Camera already at target position or no change requested. Current Pan: {self.current_pan_angle:.0f}°, Tilt: {self.current_tilt_angle:.0f}°"

        # Record manual movement to pause auto-tracking
        self.face_tracker.manual_movement_occurred()

        return f"Camera moved. Pan: {self.current_pan_angle:.0f}°, Tilt: {self.current_tilt_angle:.0f}°" 