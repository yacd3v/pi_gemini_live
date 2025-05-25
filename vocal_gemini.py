import asyncio
import os
import sys
import time
import traceback
import queue
import wave
import audioop
import warnings
import logging
from dotenv import load_dotenv
import smbus2 as smbus # Added for I2C communication
import websockets # For exception handling during session close

# Simple warning suppression for inline_data
warnings.filterwarnings("ignore", message=".*non-text parts in the response.*inline_data.*")

import base64, cv2
from picamera2 import Picamera2
from picamera2.devices import IMX500  # Add IMX500 for face detection
from libcamera import Transform  # Add Transform import for proper camera configuration
import numpy as np  # Add numpy for face detection processing

# Load environment variables from .env file
load_dotenv()

# Import our new modules
from display_animator import DisplayAnimator
from face_tracker import FaceTracker

# Temporarily suppress ALSA/JACK warnings during PyAudio import only
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import pyaudio

# Restore stderr immediately after PyAudio import
sys.stderr.close()
sys.stderr = original_stderr

from google import genai
from google.genai import types

# Configure logging to suppress inline_data warnings
logging.basicConfig(level=logging.INFO)
class InlineDataFilter(logging.Filter):
    def filter(self, record):
        return "inline_data" not in record.getMessage()

# Apply filter to common loggers that might emit inline_data warnings
for logger_name in ["google.genai", "google.ai", "google", "", "__main__"]:
    logger = logging.getLogger(logger_name)
    logger.addFilter(InlineDataFilter())

# Add display imports
import spidev as SPI
sys.path.append("display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/..")
from lib import LCD_1inch28
from PIL import Image, ImageDraw, ImageFont, ImageSequence

# Add path for servo control
sys.path.append("freenove_examples")
from servo import Servo
from led import Led  # Add LED import

# ─── Audio constants ──────────────────────────────────────────────────────────
FORMAT = pyaudio.paInt16
IN_CH = 1   #  use processed mono channel
OUT_CH = 1   # we still play/record mono
CHUNK_MS = 20             # one packet = 20 ms of audio
SEND_SAMPLE_RATE = 16_000     # Gemini Live input format
RECEIVE_SAMPLE_RATE = 24_000  # Gemini Live output format
AEC_SAMPLE_RATE = 16_000          
RAW_CH = 6

# ─── Gemini constants ────────────────────────────────────────────────────────
# MODEL = "gemini-2.0-flash-live-001"  # Full function calling support
MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"  # Better audio quality with function calling support
# Define tools for the LiveConnectConfig - these are static declarations
tools_for_config = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_time",
            description="Get the current time.",
            parameters=types.Schema(type='OBJECT', properties={})
        ),
        types.FunctionDeclaration(
            name="get_date",
            description="Get today's date.",
            parameters=types.Schema(type='OBJECT', properties={})
        ),
        types.FunctionDeclaration(
            name="set_display_brightness",
            description="Set the display brightness.",
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'brightness': types.Schema(type='NUMBER', description="A value between 0.0 and 1.0 for display brightness")
                },
                required=['brightness']
            )
        ),
        types.FunctionDeclaration(
            name="get_battery_level",
            description="Get the current battery level percentage.",
            parameters=types.Schema(type='OBJECT', properties={}) # No parameters
        ),
        types.FunctionDeclaration(
            name="go_to_sleep",
            description="Instructs the assistant to go back to sleep which is stop listening and return to wake word detection mode.",
            parameters=types.Schema(type='OBJECT', properties={}) # No parameters
        ),
        types.FunctionDeclaration(
            name="move_camera",
            description="Moves the camera by panning (left/right) or tilting (up/down) relative to its current position.\nWhen the user asks to look at a specific object, person, or in a general direction (e.g., 'look at the red box', 'look at me', 'look a bit to the left'), you should first analyze the most recent image from the camera.\nBased on your visual analysis, estimate the `pan_relative_angle` and `tilt_relative_angle` needed to center the subject or achieve the desired view.\nThe camera starts at pan 90° (center) and tilt 90° (center).\nPan (servo 0) physical limits: 13° (far left) to 154° (far right).\nTilt (servo 1) physical limits: 36° (max up) to 85° (max down from base, noting 90° is center).\nPositive `pan_relative_angle` moves right, negative moves left.\nPositive `tilt_relative_angle` moves up, negative moves down.",
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'pan_relative_angle': types.Schema(
                        type='NUMBER',
                        description="Degrees to pan. Positive values pan left, negative values pan right. E.g., 10 pans left by 10 degrees, -5 pans right by 5 degrees."
                    ),
                    'tilt_relative_angle': types.Schema(
                        type='NUMBER',
                        description="Degrees to tilt. Positive values tilt up, negative values tilt down. E.g., 10 tilts up by 10 degrees, -5 tilts down by 5 degrees."
                    )
                },
                # No required parameters, as user might want to pan OR tilt, or neither (just get status if we add that later)
            )
        ),
        types.FunctionDeclaration(
            name="set_emotion",
            description="Sets the robot's emotional expression for speaking animations. Call this when the context suggests a strong emotion for the robot to express while speaking. The emotion persists until changed again.",
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'emotion': types.Schema(
                        type='STRING',
                        description="The emotion to express. Supported values: 'normal', 'furious', 'crying'.",
                        enum=['normal', 'furious', 'crying']
                    )
                },
                required=['emotion']
            )
        ),
        types.FunctionDeclaration(
            name="toggle_face_tracking",
            description="Enable or disable automatic face tracking. When enabled, the camera will automatically follow detected faces. When disabled, only manual camera movements work.",
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'enabled': types.Schema(
                        type='BOOLEAN',
                        description="True to enable face tracking, False to disable it."
                    )
                },
                required=['enabled']
            )
        )
    ])
]

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    media_resolution="MEDIA_RESOLUTION_LOW",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    system_instruction=types.Content(
        parts=[
            types.Part(
                text="""You are Karl, a smart and very funny robot. 
                        You are very smart and helpful for the people interacting with you. 
                        Your body is a robot chassis with a screen and a speaker, a camera as a head, 4 wheels and a microphone.
                       
                        You have access to several important functions and tools - please use them when appropriate:
                        - get_time and get_date: Use when asked about current time or date
                        - get_battery_level: Use when asked about battery status
                        - move_camera: Use when asked to look in a direction, move your head/camera
                        - set_emotion: Use when you want to express strong emotions while speaking (normal, furious, crying)
                        - go_to_sleep: Use ONLY when explicitly asked to go to sleep, rest, or stop listening
                        
                        Note: You have automatic face tracking that follows people's faces when enabled. 
                        Manual camera movements temporarily pause auto-tracking for a few seconds.
                        
                        Always actively use these functions when the context calls for them. Don't just describe what you could do - actually do it!"""
            )
        ]
    ),
    tools=tools_for_config  # Use the new tools configuration
)


""" CONFIG = {
    "response_modalities": ["AUDIO"],      # keep audio replies
    "speech_config": {                     # same voice as before
        "language_code": "en-US",
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": "Zephyr"}
        },
    },
    # NEW – ask the server to transcribe both directions
    "input_audio_transcription": {},       # user → text
    "output_audio_transcription": {},      # Gemini audio → text
} """

class AudioHandler:
    """Handles local audio + Gemini Live session."""

    def __init__(self):
        # Event‑loop reference for thread‑safe calls from PyAudio callback
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # called in main thread before loop exists; fixed later in run()
            self.loop = None

        self.sleep_requested_event = asyncio.Event() # Event to signal sleep

        self.pya = pyaudio.PyAudio()

        # queues (200 packets ≈ 4 s @ 20 ms)
        self.audio_out_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self.audio_in_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

        # logging WAVs
        self.sent_wf = wave.open("sent_audio.wav", "wb")
        self.sent_wf.setnchannels(OUT_CH)
        self.sent_wf.setsampwidth(self.pya.get_sample_size(FORMAT))
        self.sent_wf.setframerate(SEND_SAMPLE_RATE)

        self.recv_wf = wave.open("recv_audio.wav", "wb")
        self.recv_wf.setnchannels(OUT_CH)
        self.recv_wf.setsampwidth(self.pya.get_sample_size(FORMAT))
        self.recv_wf.setframerate(RECEIVE_SAMPLE_RATE)

        # streams + rates
        self.input_stream = None
        self.output_stream = None
        self.input_rate: int | None = None
        self.output_rate: int | None = None

        # resampler states
        self.rs_in_state = None
        self.rs_out_state = None

        # Gemini session + client
        self.client = None
        self.session = None

        # Initialize display for status feedback
        spi1 = SPI.SpiDev()
        spi1.open(1, 0)               # bus 1, CE0  → /dev/spidev1.0
        spi1.max_speed_hz = 40_000_000
        self.disp = LCD_1inch28.LCD_1inch28(
                spi = spi1,    # use SPI-1
                rst = 12,      # GPIO12  (pin 32)
                dc  = 26,      # GPIO26  (pin 37)
                bl  = 13)      # GPIO13  (pin 33, PWM back-light)
        self.disp.Init()
        self.disp.clear()
        self.disp.bl_DutyCycle(50)
        self.font = ImageFont.truetype("display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/../Font/Font01.ttf", 24)

        # Initialize display animator   
        self.anim = DisplayAnimator(self.disp, stop_event=self.sleep_requested_event)

        # Initialize Face Tracker
        self.face_tracker = FaceTracker(enable_tracking=True, confidence_threshold=0.5)

        # Initialize Servo for camera pan/tilt (use face tracker's servo if available)
        if self.face_tracker.servo:
            self.servo = self.face_tracker.servo
            self.current_pan_angle = self.face_tracker.current_pan_angle
            self.current_tilt_angle = self.face_tracker.current_tilt_angle
            print("[AudioHandler] Using FaceTracker servo for camera movement.")
        else:
            # Fallback servo initialization for manual movement only
            try:
                self.servo = Servo()
                self.current_pan_angle = 90  # Start at center pan
                self.current_tilt_angle = 50  # Start at 50° to look up from floor level
                self.servo.set_servo_pwm('0', self.current_pan_angle) # Pan servo
                self.servo.set_servo_pwm('1', self.current_tilt_angle) # Tilt servo
                print("[AudioHandler] Servos initialized for manual movement only.")
            except Exception as e:
                self.servo = None
                print(f"[AudioHandler] Error initializing servos: {e}. Camera movement will be disabled.")
                # Fallback: set angles so subsequent logic doesn't error if servo is None
                self.current_pan_angle = 90
                self.current_tilt_angle = 50

        self.current_speaking_emotion = "normal" # Default speaking emotion

        # Initialize available functions and map them
        self.available_functions = [
            self.get_time, self.get_date, self.set_display_brightness,
            self.get_battery_level, self.go_to_sleep, self.move_camera,
            self.set_emotion, self.toggle_face_tracking # Add new function
        ]
        self.functions_map = {func.__name__: func for func in self.available_functions}

    # --- Tool Functions (now methods) ---
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
        self.disp.bl_DutyCycle(int(brightness * 100)) 
        return f"Display brightness set to {brightness*100}%"

    def get_battery_level(self):
        """Get the current battery level percentage."""
        ADDR = 0x2d  # I2C address of the UPS
        bus = None
        try:
            bus = smbus.SMBus(1)  # 0 for RPi 1, 1 for RPi 2,3,4
            data = bus.read_i2c_block_data(ADDR, 0x20, 6)
            battery_percent = int(data[4] | data[5] << 8)
            return f"The current battery level is {battery_percent}%"
        except FileNotFoundError:
            return "Error: I2C bus not found. Ensure I2C is enabled and the device is connected."
        except OSError as e:
            if e.errno == 121: # Remote I/O error (device not found at address)
                return f"Error: UPS device not found at address {hex(ADDR)}. Please check the connection."
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
            if self.input_stream and self.input_stream.is_active():
                self.input_stream.stop_stream()
                print("[GoToSleep] Input stream stopped")
            if self.output_stream and self.output_stream.is_active():
                self.output_stream.stop_stream()
                print("[GoToSleep] Output stream stopped")
        except Exception as e:
            print(f"[GoToSleep] Error stopping streams: {e}")
        
        # Now set the sleep event
        self.sleep_requested_event.set()
        print("[GoToSleep] Sleep event SET.")
        
        if self.session:
            print("[GoToSleep] Session exists, attempting to close.")
            try:
                # Attempt to close the session gracefully.
                # This might raise an error if already closing or closed.
                await self.session.close()
                print("[GoToSleep] Gemini session close COMMANDED.")
            except websockets.exceptions.ConnectionClosedOK:
                print("[GoToSleep] Gemini session already closed (OK).")
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"[GoToSleep] Gemini session closed with error during explicit close: {e}")
            except Exception as e:
                print(f"[GoToSleep] Error during explicit session close: {e}")
        else:
            print("[GoToSleep] No active session to close.")
        
        # Signal other tasks to wind down if they are waiting on queues
        # Use try_put_nowait to avoid blocking if queues are full (shouldn't happen often on shutdown)
        try:
            self.audio_out_q.put_nowait(None) 
            print("[GoToSleep] Sentinel PUSHED to audio_out_q.")
        except asyncio.QueueFull:
            print("[GoToSleep] audio_out_q was full, sentinel not pushed.")
        try:
            self.audio_in_q.put_nowait(None)
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

    def _save_debug_image(self, image, target_angles=None):
        """Save debug image with center lines and target information."""
        # Convert to BGR for OpenCV operations if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            debug_img = image.copy()
        else:
            debug_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        h, w = debug_img.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw crosshair at center
        cv2.line(debug_img, (center_x, 0), (center_x, h), (0, 255, 0), 1)
        cv2.line(debug_img, (0, center_y), (w, center_y), (0, 255, 0), 1)
        cv2.circle(debug_img, (center_x, center_y), 10, (0, 255, 0), 1)
        
        # Add current camera position
        text = f"Pan: {self.current_pan_angle:.0f}, Tilt: {self.current_tilt_angle:.0f}"
        cv2.putText(debug_img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add target angles if provided
        if target_angles:
            pan_rel, tilt_rel = target_angles
            text = f"Target move: Pan {pan_rel:+.2f}, Tilt {tilt_rel:+.2f}"
            cv2.putText(debug_img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw arrow indicating requested movement
            arrow_length = 50
            # Calculate arrow endpoint (scaled and in correct direction)
            # Note: Pan direction matches new inverted servo control (positive pan = left movement)
            arrow_x = center_x - int(pan_rel * 5)  # Keep this direction for visualization
            arrow_y = center_y - int(tilt_rel * 5)  # Negative because lower tilt number = up
            cv2.arrowedLine(debug_img, (center_x, center_y), (arrow_x, arrow_y), (0, 0, 255), 2)
        
        # Save image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"debug_frame_{timestamp}.jpg"
        cv2.imwrite(filename, debug_img)
        print(f"[Debug] Saved debug image to {filename}")
        return filename

    def move_camera(self, pan_relative_angle: float = 0.0, tilt_relative_angle: float = 0.0):
        """Pans or tilts the camera by a specified number of degrees relative to the current position.

        Args:
            pan_relative_angle (float): Degrees to pan. Positive pans left, negative pans right.
            tilt_relative_angle (float): Degrees to tilt. Positive tilts up, negative tilts down.
        """
        # First, save the latest frame with debug info if we have one
        try:
            if hasattr(self, 'last_captured_frame') and self.last_captured_frame is not None:
                self._save_debug_image(
                    self.last_captured_frame, 
                    (pan_relative_angle, tilt_relative_angle)
                )
        except Exception as e:
            print(f"[MoveCamera] Error saving debug image: {e}")

        if not self.servo:
            return "Camera control is disabled due to an initialization error."

        # Define limits
        PAN_MIN = 13  # Left
        PAN_MAX = 154 # Right
        TILT_MIN = 36 # Up
        TILT_MAX = 85 # Down (Note: 90 is base, so lower numbers are more "up")

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

    def _strip_to_processed(self, data: bytes) -> bytes:
        frame = RAW_CH * 2              # 12 bytes per frame
        out = bytearray(len(data) // RAW_CH)  # space for 2-byte mono
        o = 0
        for i in range(0, len(data), frame):
            out[o:o+2] = data[i:i+2]    # copy both bytes of ch-0
            o += 2
        return bytes(out)

    # ────────────────────────── device helpers ──────────────────────────────
    def _find_input_device(self) -> int:
        for idx in range(self.pya.get_device_count()):
            #info = self.pya.get_device_info_by_index(idx)
            #if info.get("maxInputChannels", 0) and (
            #    "respeaker" in info["name"].lower() or "seeed" in info["name"].lower()):
            #    return idx
            name = self.pya.get_device_info_by_index(idx)["name"].lower()
            if "respeaker" in name or "seeed" in name:
                return idx
        raise RuntimeError("ReSpeaker output not found")
    
        #info = self.pya.get_default_input_device_info()
        #return info["index"]

    def _find_output_device(self) -> int:          # ③ lock to ReSpeaker
        for idx in range(self.pya.get_device_count()):
            name = self.pya.get_device_info_by_index(idx)["name"].lower()
            if "respeaker" in name or "seeed" in name:
                return idx
        raise RuntimeError("ReSpeaker output not found")

    # ────────────────────────── PyAudio callback ────────────────────────────
    def _mic_callback(self, in_data, frame_count, time_info, status):
        """Runs in **PyAudio thread** – push data into asyncio queue."""
        if self.loop is None:
            return (None, pyaudio.paContinue)
        
        # Check if we should stop processing
        if self.sleep_requested_event.is_set():
            return (None, pyaudio.paComplete)
            
        try:
            self.loop.call_soon_threadsafe(self.audio_out_q.put_nowait, in_data)
        except asyncio.QueueFull:
            # drop one packet – better than blocking and causing an overrun
            pass
        except Exception as e:
            # Handle other exceptions gracefully
            pass
        return (None, pyaudio.paContinue)

    # ────────────────────────── stream setup ────────────────────────────────
    async def setup_streams(self):
        # print("\nAvailable audio devices:")
        for i in range(self.pya.get_device_count()):
            info = self.pya.get_device_info_by_index(i)
            # print(f"  {i}: {info['name']}  (in {info['maxInputChannels']}, out {info['maxOutputChannels']}, {int(info['defaultSampleRate'])} Hz)")

        in_idx = self._find_input_device()
        out_idx = self._find_output_device()

        # input – prefer 16 kHz
        in_rates = [16_000]   # ReSpeaker native
        in_rates += [int(self.pya.get_device_info_by_index(in_idx)["defaultSampleRate"]), 48_000, 44_100]
        for r in in_rates:
            try:
                fpb = int(r * CHUNK_MS / 1000)
                self.input_stream = self.pya.open(format=FORMAT,
                                                  channels=RAW_CH,
                                                  rate=r,
                                                  input=True,
                                                  input_device_index=in_idx,   # still the same device
                                                  frames_per_buffer=fpb,
                                                  stream_callback=self._mic_callback)
                self.input_rate = r
                # print(f"Input stream opened at {r} Hz, fpb {fpb}")
                break
            except Exception as e:
                # print(f"  could not open at {r} Hz: {e}")
                pass
        if not self.input_stream:
            raise RuntimeError("No usable input stream")

        # output – use device default (first that works)
        #out_rates = [int(self.pya.get_device_info_by_index(out_idx)["defaultSampleRate"]), 48_000, 44_100]
        #out_rates = [AEC_SAMPLE_RATE, 48_000, 44_100]
        out_rates = [AEC_SAMPLE_RATE]
        
        for r in out_rates:
            try:
                fpb = int(r * CHUNK_MS / 1000)
                self.output_stream = self.pya.open(format=FORMAT,
                                                   channels=OUT_CH,
                                                   rate=r,
                                                   output=True,
                                                   output_device_index=out_idx,
                                                   frames_per_buffer=fpb)
                self.output_rate = r
                # print(f"Output stream opened at {r} Hz, fpb {fpb}")
                break
            except Exception as e:
                # print(f"  could not open output at {r} Hz: {e}")
                pass
        if not self.output_stream:
            raise RuntimeError("No usable output stream")

        self.input_stream.start_stream()

    def _capture_jpeg(self) -> bytes:
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        ))
        cam.start()
        time.sleep(0.25)           # quick AE settle
        rgb = cam.capture_array()
        cam.close()
        return cv2.imencode(".jpg", rgb)[1].tobytes()
    
    async def _vision_feed(self, interval=8):
        """Send a fresh camera frame every *interval* seconds."""
        # Initialize camera with IMX500 if face detection is enabled
        if self.face_tracker.face_detection_enabled:
            try:
                cam = Picamera2(self.face_tracker.imx500.camera_num)
                print("[Vision] Camera initialized with IMX500 for face detection")
                
                # Configure camera for both vision feed and AI detection - match test_autoaim.py
                config = cam.create_preview_configuration(
                    main={"size": (640, 480)},
                    raw={"size": (2028, 1520)},
                    encode="main",
                    buffer_count=6
                )
                
                # Set transform if supported - like test_autoaim.py
                try:
                    config["transform"] = Transform()
                except Exception as e:
                    print(f"[Vision] Could not set transform: {e}")
                
                cam.configure(config)
                
                # Set up network intrinsics for face detection - match test_autoaim.py
                if self.face_tracker.imx500.network_intrinsics:
                    ni = self.face_tracker.imx500.network_intrinsics
                    ni.task = "pose estimation"
                    ni.inference_rate = 30.0
                    # PoseNet doesn't use bbox_normalization or labels the same way
                    print("[Vision] IMX500 network intrinsics configured")
                    
            except Exception as e:
                print(f"[Vision] Error initializing IMX500 camera: {e}")
                # Fallback to regular camera
                cam = Picamera2()
                cam.configure(cam.create_still_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                ))
                self.face_tracker.face_detection_enabled = False
        else:
            # Regular camera initialization
            cam = Picamera2()
            cam.configure(cam.create_still_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            ))
            print("[Vision] Camera initialized (no face detection)")
        
        cam.start()
        print("[Vision] Camera started")
        
        # Delay first frame to allow session setup
        await asyncio.sleep(3)
        
        # Add attribute to store last frame
        self.last_captured_frame = None

        try:
            while not self.sleep_requested_event.is_set():
                try:
                    # Always capture with metadata if face detection is enabled
                    if self.face_tracker.face_detection_enabled:
                        request = cam.capture_request()
                        rgb = request.make_array("main")
                        metadata = request.get_metadata()
                        request.release()
                        
                        # Perform face detection and tracking regardless of session status
                        if self.face_tracker.should_auto_track():
                            face_detections = self.face_tracker.parse_face_detection(metadata)
                            if face_detections:
                                best_face = max(face_detections, key=lambda f: f["confidence"])
                                print(f"[Vision] Face detected (confidence: {best_face['confidence']:.2f})")
                                if self.face_tracker.track_face(best_face["center_x"], best_face["center_y"]):
                                    print(f"[Vision] Auto-tracked face (confidence: {best_face['confidence']:.2f})")
                                    # Update our current angles to match the face tracker
                                    self.current_pan_angle = self.face_tracker.current_pan_angle
                                    self.current_tilt_angle = self.face_tracker.current_tilt_angle
                            else:
                                print("[Vision] No face detected")
                    else:
                        # Regular capture without metadata
                        rgb = cam.capture_array()
                    
                    # Store the latest frame for debugging
                    self.last_captured_frame = rgb.copy()
                    
                    # Send frame to Gemini if session is active
                    if self.session:
                        # Check if we should skip sending to avoid overwhelming API
                        is_receiving = False
                        for _ in range(3):
                            if not self.audio_in_q.empty():
                                is_receiving = True
                                break
                            await asyncio.sleep(0.05)
                        
                        if not is_receiving:
                            # Send frame to Gemini
                            jpeg_bytes = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                            blob = types.Blob(
                                data=jpeg_bytes,
                                mime_type="image/jpeg"
                            )
                            
                            try:
                                await asyncio.wait_for(
                                    self.session.send_realtime_input(media=blob),
                                    timeout=2.0
                                )
                            except asyncio.TimeoutError:
                                print("[Vision] Frame send timed out, skipping")
                            except Exception as e:
                                print(f"[Vision] Frame send error: {e}")
                    
                    # Wait before next iteration - use face_tracking_interval for consistency
                    await asyncio.sleep(self.face_tracker.face_tracking_interval if self.face_tracker.face_detection_enabled else interval)
                    
                except Exception as e:
                    print(f"[Vision] Error in main loop: {e}")
                    await asyncio.sleep(interval)

        except Exception as e:
            print(f"[Vision] Feed loop error: {e}")
        finally:
            # Clean up camera
            cam.close()
            print("[Vision] Vision feed task exiting...")

    # ────────────────────────── coroutines ──────────────────────────────────
    async def _send_to_gemini(self):
        print("[Sender] Task started.")
        while True:
            if self.sleep_requested_event.is_set():
                print("[Sender] Sleep event detected, exiting send-loop.")
                break
            try:
                # Wait for data with a timeout to allow checking the sleep event
                data = await asyncio.wait_for(self.audio_out_q.get(), timeout=0.1)
                if data is None: # Sentinel value for shutdown
                    print("[Sender] Received sentinel, exiting send-loop.")
                    self.audio_out_q.task_done()
                    break
            except asyncio.TimeoutError:
                continue # No data, loop back to check sleep_requested_event
            except Exception as e:
                print(f"[Sender] Error getting data from queue: {e}")
                break

            try:
                data = self._strip_to_processed(data)       # ① use the AEC track only
                if self.input_rate != SEND_SAMPLE_RATE:
                    data, self.rs_in_state = audioop.ratecv(
                        data, 2, OUT_CH,
                        self.input_rate, SEND_SAMPLE_RATE,
                        self.rs_in_state)
                if self.session: # Rely on send_realtime_input to fail if session is closed
                    blob = types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
                    await self.session.send_realtime_input(media=blob)
                else:
                    print("[Sender] Session not active, not sending.")
                self.sent_wf.writeframes(data)
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"[Sender] Connection closed while sending: {e}. Exiting send-loop.")
                break
            except Exception as e:
                print(f"[Sender] Error processing or sending audio: {e}")
            finally:
                if 'data' in locals() and data is not None: # Ensure data was fetched and not sentinel
                    self.audio_out_q.task_done()
        
        print("[Sender] Send task exiting...")

    # Add helper to show status on display
    def _show_status(self, text):
        # Create an image with centered status text
        image = Image.new("RGB", (self.disp.width, self.disp.height), "BLACK")
        draw = ImageDraw.Draw(image)
        # Calculate text width and height using textbbox
        bbox = draw.textbbox((0, 0), text, font=self.font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (self.disp.width - w) // 2
        y = (self.disp.height - h) // 2
        draw.text((x, y), text, fill="WHITE", font=self.font)
        # No need to rotate since screen is physically upside down
        self.disp.ShowImage(image)

    async def _recv_from_gemini(self):
        print("[Receiver] Task started.")

        is_speaking = False                # ← persists
        
        while True:
            if self.sleep_requested_event.is_set():
                print("[Receiver] Sleep event detected, exiting receive-loop.")
                break

            if not self.session: # Rely on session.receive() to fail if session is closed
                if self.sleep_requested_event.is_set(): # Double check if sleep was requested during this gap
                    break
                await asyncio.sleep(0.1) # Brief pause before checking session again or sleep event
                continue
            
            try:
                # Use a timeout for receive to allow checking sleep_requested_event
                # However, session.receive() is a generator, making direct timeout tricky.
                # The primary exit from this loop when sleeping will be the ConnectionClosedError
                # when self.session.close() is called in go_to_sleep.

                turn_iterator = self.session.receive()
                async for resp in turn_iterator:

                    if self.sleep_requested_event.is_set():
                        print("[Receiver] Sleep event detected mid-turn, breaking from turn processing.")
                        break # Exit from processing messages in the current turn
                    
                    if resp.data and not is_speaking:
                        speak_animation_key = f"speak_{self.current_speaking_emotion}"
                        self.anim.set_mode(speak_animation_key)
                        is_speaking = True
                    
                    if resp.data:
                        await self.loop.run_in_executor(None, self.recv_wf.writeframes, resp.data) # Non-blocking
                        await self.audio_in_q.put(resp.data)
                    
                    #if resp.text:
                        #print(f"\nGemini text response: {resp.text}")
                    
                    # Handle function calls based on tool_call
                    if resp.tool_call and resp.tool_call.function_calls:
                        for fc in resp.tool_call.function_calls:
                            function_name = fc.name
                            function_args = fc.args or {}
                            function_id = fc.id  # Crucial for the response

                            print(f"\n{'='*50}")
                            print(f"Function Call: {function_name}")
                            print(f"Arguments: {function_args}")
                            
                            if function_name in self.functions_map:
                                func_to_call = self.functions_map[function_name]
                                try:
                                    # If it's an async function (like go_to_sleep), await it
                                    if asyncio.iscoroutinefunction(func_to_call):
                                        result = await func_to_call(**function_args)
                                    else:
                                        result = func_to_call(**function_args)
                                    
                                    print(f"Response: {result}")
                                    print(f"{'='*50}\n")
                                    
                                    # For go_to_sleep, the session will be closing, so don't attempt to send a response.
                                    if function_name == "go_to_sleep":
                                        print("Skipping tool response for go_to_sleep as session is closing.")
                                    else:
                                        # Send the result back to Gemini using send_tool_response
                                        tool_response_part = types.FunctionResponse(
                                            id=function_id,
                                            name=function_name,
                                            response={"result": result}
                                        )
                                        await self.session.send_tool_response(
                                            function_responses=[tool_response_part]
                                        )
                                except Exception as e:
                                    print(f"\n{'='*50}")
                                    print(f"Function Call Error: {function_name}")
                                    print(f"Arguments: {function_args}")
                                    print(f"Error: {str(e)}")
                                    print(f"{'='*50}\n")
                            else:
                                print(f"Function {function_name} not found in available functions map.")
                                # Send error response back to Gemini if function not found
                                if self.session:
                                    try:
                                        error_tool_response = types.FunctionResponse(
                                            id=function_id,
                                            name=function_name,
                                            response={"error": f"Function {function_name} not implemented or available."}
                                        )
                                        await self.session.send_tool_response(
                                            function_responses=[error_tool_response]
                                        )
                                    except Exception as e:
                                        print(f"Failed to send error response for {function_name}: {e}")
                
                # Show listening status when done speaking
                self.anim.set_mode("idle")
                is_speaking = False  # Reset for the next turn
                
            except websockets.exceptions.ConnectionClosedOK:
                print("[Receiver] Connection closed (OK). Exiting receive-loop.")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"[Receiver] Connection closed with error: {e}. Exiting receive-loop.")
                break
            except asyncio.TimeoutError:
                # This might occur if we implement a timeout around receive(), but it's complex with async iterators
                continue
            except Exception as e:
                print(f"Error in _recv_from_gemini: {e}")
                traceback.print_exc()
                if self.sleep_requested_event.is_set(): # If an error occurs, and sleep is requested, exit.
                    print("[Receiver] Exiting due to error and sleep request.")
                    break
                # Don't break the loop on other errors unless sleep is also set, just continue
                await asyncio.sleep(0.1) # Small delay before retrying or continuing
                continue

        print("[Receiver] Receive task exiting...")

    async def _playback(self):
        print("[Playback] Task started.")
        while True:
            if self.sleep_requested_event.is_set():
                print("[Playback] Sleep event detected, exiting playback-loop.")
                break
            try:
                # Wait for PCM data with a timeout
                pcm = await asyncio.wait_for(self.audio_in_q.get(), timeout=0.1)
                if pcm is None: # Sentinel value for shutdown
                    print("[Playback] Received sentinel, exiting playback-loop.")
                    self.audio_in_q.task_done()
                    break
            except asyncio.TimeoutError:
                continue # No data, loop back to check sleep_requested_event
            except Exception as e:
                print(f"[Playback] Error getting data from queue: {e}")
                break

            try:
                if self.output_rate != RECEIVE_SAMPLE_RATE:
                    pcm_converted, self.rs_out_state = audioop.ratecv(
                        pcm, 2, OUT_CH,
                        RECEIVE_SAMPLE_RATE, self.output_rate,
                        self.rs_out_state)
                else:
                    pcm_converted = pcm
                
                if self.output_stream and self.output_stream.is_active():
                    await self.loop.run_in_executor(None, self.output_stream.write, pcm_converted)
                else:
                    print("[Playback] Output stream not active, not playing audio.")
            except Exception as e:
                print(f"[Playback] Error processing or playing audio: {e}")
            finally:
                if 'pcm' in locals() and pcm is not None: # Ensure pcm was fetched and not sentinel
                    self.audio_in_q.task_done()
        
        print("[Playback] Playback task exiting...")

    # ────────────────────────── cleanup ─────────────────────────────────────
    def _cleanup(self):
        print("Cleaning up…")
        
        # Stop animation and turn off LEDs
        if hasattr(self, 'anim'):
            try:
                if hasattr(self.anim, 'led_turn_off'):
                    self.anim.led_turn_off()  # Turn off LEDs first
                self.sleep_requested_event.set()  # Signal animator to stop
                time.sleep(0.1)  # Brief pause for animator to stop
            except Exception as e:
                print(f"Error stopping animator and LEDs: {e}")
        
        # Cleanup audio streams with proper error handling
        for s in (self.input_stream, self.output_stream):
            try:
                if s and s.is_active():
                    s.stop_stream()
                    time.sleep(0.1)  # Small delay between stop and close
                if s:
                    s.close()
            except Exception as e:
                print(f"Error cleaning up stream: {e}")
        
        # Cleanup wave files
        for wf in (self.sent_wf, self.recv_wf):
            try:
                if wf:
                    wf.close()
            except Exception as e:
                print(f"Error closing wave file: {e}")
        
        # Terminate PyAudio with delay
        try:
            if self.pya:
                self.pya.terminate()
                time.sleep(0.5)  # Longer delay after PyAudio termination
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")
            
        # Cleanup display - this needs to be done carefully due to gpiozero
        try:
            if hasattr(self, 'disp'):
                # First try the display's own cleanup methods
                try:
                    self.disp.clear()
                    self.disp.bl_DutyCycle(0)
                    self.disp.module_exit()
                except Exception as disp_e:
                    print(f"Display cleanup methods failed: {disp_e}")
                
                # Now clean up GPIO resources
                try:
                    # First try to clean up gpiozero devices
                    import gpiozero
                    # Force close all gpiozero devices
                    try:
                        gpiozero.Device.pin_factory.close()
                        print("gpiozero pin factory closed successfully")
                    except AttributeError:
                        # Fallback for older gpiozero versions
                        try:
                            gpiozero.Device.pin_factory.reset()
                            print("gpiozero devices reset successfully")
                        except AttributeError:
                            print("gpiozero cleanup method not available")
                except ImportError:
                    print("gpiozero not available for cleanup")
                except Exception as gz_e:
                    print(f"gpiozero cleanup error: {gz_e}")
                
                # Give a moment for gpiozero to fully release pins
                time.sleep(0.5)
                
                # Then clean up RPi.GPIO as backup
                try:
                    import RPi.GPIO as GPIO
                    # Clean up the specific pins used by display
                    GPIO.setmode(GPIO.BCM)
                    for pin in [12, 13, 26]:  # rst, bl, dc pins
                        try:
                            GPIO.setup(pin, GPIO.IN)  # Set to input to release
                            GPIO.cleanup(pin)
                        except:
                            pass
                    GPIO.cleanup()  # Clean up all GPIO
                    print("RPi.GPIO cleanup completed")
                except Exception as gpio_e:
                    print(f"RPi.GPIO cleanup error: {gpio_e}")
                    
        except Exception as e:
            print(f"Error cleaning up display: {e}")
            
        print("Cleanup completed.")
        # Additional delay to ensure all resources are fully released
        time.sleep(1)

    # ────────────────────────── main entry ──────────────────────────────────
    async def run(self):
        print("[AudioHandler] Starting...")
        if not os.getenv("GOOGLE_API_KEY"):
            print("[AudioHandler] Error: GOOGLE_API_KEY environment variable not set")
            return

        self.sleep_requested_event.clear() # Clear event at the start of a new run

        # set loop reference (if not set in __init__)
        if self.loop is None:
            self.loop = asyncio.get_running_loop()

        try:
            await self.setup_streams()

            print("[AudioHandler] Connecting to Gemini Live...")
            self.client = genai.Client(
                http_options={"api_version": "v1beta"},
                api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            async with self.client.aio.live.connect(model=MODEL, config=CONFIG) as sess:
                print("[AudioHandler] Connected. Listening...")
                self.session = sess
                
                # Run LED initialization sequence
                await self.anim.run_initialization()
                
                self.anim.set_mode("idle") # Set initial animation to idle

                try:
                    async with asyncio.TaskGroup() as tg:
                        send_task = tg.create_task(self._send_to_gemini())
                        recv_task = tg.create_task(self._recv_from_gemini())
                        playback_task = tg.create_task(self._playback())
                        anim_task = tg.create_task(self.anim.run())
                        vision_task = tg.create_task(self._vision_feed(interval=self.face_tracker.face_tracking_interval if self.face_tracker.face_detection_enabled else 8))
                        
                        # Add a monitoring task to help debug when tasks complete
                        async def monitor_sleep():
                            while not self.sleep_requested_event.is_set():
                                await asyncio.sleep(0.1)
                            # Give tasks a moment to see the sleep event and start cleanup
                            await asyncio.sleep(1)
                        
                        monitor_task = tg.create_task(monitor_sleep())
                        
                except Exception as e:
                    print(f"[AudioHandler] Error in task group: {e}")
                    print("[AudioHandler] Task group traceback:")
                    traceback.print_exc()
                    raise

            self.session = None
        except Exception as e:
            print(f"[AudioHandler] Error in run method: {e}")
            print("[AudioHandler] Run method traceback:")
            traceback.print_exc()
            raise
        finally:
            print("[AudioHandler] Returning to wake_porcu.py")

    # ────────────────────────── static helper ───────────────────────────────


if __name__ == "__main__":
    print("[Main] Starting application...")
    handler = AudioHandler()
    try:
        print("[Main] Running main async loop...")
        asyncio.run(handler.run())
    except KeyboardInterrupt:
        print("[Main] Keyboard interrupt received")
    except Exception as e:
        print(f"[Main] Unexpected error occurred: {e}")
        print("[Main] Full traceback:")
        traceback.print_exc()
    finally:
        print("[Main] Starting cleanup...")
        handler._cleanup()
        print("[Main] Cleanup completed")