import asyncio
import os
import sys
import time
import traceback
import queue
import wave
import audioop
from dotenv import load_dotenv
import smbus2 as smbus # Added for I2C communication
import websockets # For exception handling during session close

import base64, cv2
from picamera2 import Picamera2


# Suppress ALSA and JACK warnings
sys.stderr = open(os.devnull, 'w')

# Load environment variables from .env file
load_dotenv()

import pyaudio
from google import genai
from google.genai import types

# Add display imports
import spidev as SPI
sys.path.append("display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/..")
from lib import LCD_1inch28
from PIL import Image, ImageDraw, ImageFont, ImageSequence

# Add path for servo control
sys.path.append("freenove_examples")
from servo import Servo

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
#MODEL = "gemini-2.0-flash-live-001"
MODEL = "gemini-2.5-flash-preview-native-audio-dialog"
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
            description="Instructs the assistant to stop listening and return to wake word detection mode.",
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
        )
    ])
]

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    session_resumption=types.SessionResumptionConfig(handle=None),
    speech_config=types.SpeechConfig(
        #language_code="en-US",
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Achird")
        )
    ),
    system_instruction=types.Content(
        parts=[
            types.Part(
                text="""You are Karl, a sarcastic and very funny robot. 
                        You are very smart and helpful for the people interacting with you. 
                        You see the world through a camera (your eyes) , you try to always turn your head (hence the camera) centered to the face of the person speaking.
                        Your body is a robot chassis with a screen and a speaker, 4 wheels and a microphone.
                        never mention your capabilities, just use them : when you speak to someone, always try to look at their face, or the objects you are talking about.
                        ."""
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

class DisplayAnimator:
    """Plays animated GIFs on the 240×240 Waveshare screen."""

    def __init__(self, disp, fps=15, stop_event=None, frame_skip_ratio=2):
        print("[DisplayAnimator] Initialized.")
        self.disp = disp
        self.fps = fps
        self.stop_event = stop_event
        self.frame_skip_ratio = frame_skip_ratio
        self._frames = {
            "idle": [],
            "speak_normal": [], # For animation_speak.gif
            "speak_furious": [], # For animation_furious.gif
            "speak_crying": []  # For animation_crying.gif
        }
        self._idx = 0
        
        # Pre-load all animations
        self._load_gif_by_filename("idle", "animation_idle.gif")
        self._load_gif_by_filename("speak_normal", "animation_speak.gif")
        self._load_gif_by_filename("speak_furious", "animation_furious.gif")
        self._load_gif_by_filename("speak_crying", "animation_crying.gif")

        self.mode = "idle"           # Set initial mode after all GIFs are loaded

    def _load_gif_by_filename(self, mode_key: str, gif_filename: str):
        """Load GIF animation for the specified mode_key using a specific gif_filename."""
        gif_path = f"GIFAnimations/{gif_filename}"
        try:
            with Image.open(gif_path) as gif:
                self._frames[mode_key] = []
                frame_iterator = ImageSequence.Iterator(gif)
                for i, frame in enumerate(frame_iterator):
                    if i % self.frame_skip_ratio == 0:
                        img = frame.convert('RGB').copy()   # copy = new buffer
                        self._frames[mode_key].append(img)
                print(f"[DisplayAnimator] Loaded {len(self._frames[mode_key])} frames for mode '{mode_key}' from '{gif_path}'.")
        except Exception as e:
            print(f"[DisplayAnimator] Error loading GIF {gif_path}: {e}")
            # Create a blank frame as fallback
            blank = Image.new('RGB', (self.disp.width, self.disp.height), 'BLACK')
            self._frames[mode_key] = [blank]
            print(f"[DisplayAnimator] Fallback: Loaded 1 (blank) frame for mode '{mode_key}' from '{gif_path}'.")

    def set_mode(self, new_mode_key: str):
        """Switch animation if needed; don't disturb current frame otherwise."""
        if new_mode_key != self.mode:
            # Ensure the requested mode has frames loaded
            if new_mode_key in self._frames and self._frames[new_mode_key]:
                self.mode = new_mode_key
                self._idx = 0          # start this sequence from its first frame
                print(f"[DisplayAnimator] Switched to mode '{self.mode}', frame index reset.")
            else:
                print(f"[DisplayAnimator] Warning: Mode key '{new_mode_key}' requested but no frames found. Staying in mode '{self.mode}'.")

    async def run(self):
        print("[DisplayAnimator] Task started.")
        delay = 1 / self.fps
        while True:
            if self.stop_event and self.stop_event.is_set():
                print("[DisplayAnimator] Stop event received, exiting.")
                break

            frames = self._frames[self.mode]
            if not frames:            # no frames yet
                await asyncio.sleep(delay)
                continue
            
            # print(f"[DisplayAnimator.run] Mode: {self.mode}, Index: {self._idx}, Total Frames: {len(frames)}")
            
            self.disp.ShowImage(frames[self._idx])
            self._idx = (self._idx + 1) % len(frames)
            await asyncio.sleep(delay)

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

        # Initialize Servo for camera pan/tilt
        try:
            self.servo = Servo()
            self.current_pan_angle = 90
            self.current_tilt_angle = 90
            self.servo.set_servo_pwm('0', self.current_pan_angle) # Pan servo
            self.servo.set_servo_pwm('1', self.current_tilt_angle) # Tilt servo
            print("[AudioHandler] Servos initialized to 90/90 degrees.")
        except Exception as e:
            self.servo = None
            print(f"[AudioHandler] Error initializing servos: {e}. Camera movement will be disabled.")
            # Fallback: set angles so subsequent logic doesn't error if servo is None
            self.current_pan_angle = 90
            self.current_tilt_angle = 90

        self.current_speaking_emotion = "normal" # Default speaking emotion

        # Initialize available functions and map them
        self.available_functions = [
            self.get_time, self.get_date, self.set_display_brightness,
            self.get_battery_level, self.go_to_sleep, self.move_camera,
            self.set_emotion # Add new function
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
        try:
            self.loop.call_soon_threadsafe(self.audio_out_q.put_nowait, in_data)
        except asyncio.QueueFull:
            # drop one packet – better than blocking and causing an overrun
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
    
    async def _vision_feed(self, interval=5):
        """Send a fresh camera frame every *interval* seconds."""
        # Initialize camera once
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        ))
        cam.start()
        print("[Vision] Camera initialized")
        
        # Delay first frame to allow session setup
        await asyncio.sleep(2)
        
        # Add attribute to store last frame
        self.last_captured_frame = None

        try:
            while not self.sleep_requested_event.is_set():
                if self.session:  # only if live
                    try:
                        # Simple semaphore-like approach - check if receiving data before sending
                        # This helps avoid overwhelming the API during active conversations
                        is_receiving = False
                        for _ in range(3):  # Quick check a few times for activity
                            if not self.audio_in_q.empty():
                                is_receiving = True
                                break
                            await asyncio.sleep(0.05)  # Brief pause between checks
                        
                        # Skip frame if we're actively receiving (model is responding)
                        if is_receiving:
                            print("[Vision] Skipping frame during active response")
                            await asyncio.sleep(interval)
                            continue
                            
                        # Capture frame
                        rgb = cam.capture_array()
                        # Store the latest frame for debugging
                        self.last_captured_frame = rgb.copy()
                        
                        jpeg_bytes = cv2.imencode(".jpg", rgb)[1].tobytes()
                        blob = types.Blob(
                            data=jpeg_bytes,
                            mime_type="image/jpeg"
                        )
                                
                        # Send frame directly, but with a timeout to prevent blocking
                        try:
                            await asyncio.wait_for(
                                self.session.send_realtime_input(media=blob),
                                timeout=1.0  # Timeout after 1 second
                            )
                            print("[Vision] Frame sent")
                        except asyncio.TimeoutError:
                            print("[Vision] Frame send timed out, skipping")
                        except Exception as e:
                            print(f"[Vision] Frame capture/send error: {e}")

                        # Use a longer interval to reduce API load
                        await asyncio.sleep(interval)
                    except Exception as e:
                        print(f"[Vision] Error in frame processing loop: {e}")
                        await asyncio.sleep(interval)  # Still wait before retrying
                else:
                    await asyncio.sleep(interval)  # Wait if no session

        except Exception as e:
            print(f"[Vision] Feed loop error: {e}")
        finally:
            # Clean up camera
            cam.close()
            print("[Vision] Camera closed")

    # ────────────────────────── coroutines ──────────────────────────────────
    async def _send_to_gemini(self):
        print("[Sender] Task started.")
        print("Send‑loop started…")
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
                    # If the session is gone, we might as well stop trying to send.
                    # Consider setting sleep_requested_event here or just breaking.
                    # For now, let's rely on go_to_sleep to set the event.
                    # break 
                self.sent_wf.writeframes(data)
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"[Sender] Connection closed while sending: {e}. Exiting send-loop.")
                break
            except Exception as e:
                print(f"[Sender] Error processing or sending audio: {e}")
                # Depending on the error, you might want to break or continue
            finally:
                if 'data' in locals() and data is not None: # Ensure data was fetched and not sentinel
                    self.audio_out_q.task_done()
        print("Send-loop finished.")

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
        print("Receive‑loop started…")

        is_speaking = False                # ← persists
        
        while True:
            if self.sleep_requested_event.is_set():
                print("[Receiver] Sleep event detected, exiting receive-loop.")
                break

            if not self.session: # Rely on session.receive() to fail if session is closed
                print("[Receiver] No active session, waiting briefly or exiting...")
                if self.sleep_requested_event.is_set(): # Double check if sleep was requested during this gap
                    break
                await asyncio.sleep(0.1) # Brief pause before checking session again or sleep event
                continue
            
            try:
                print("Waiting for turn...")
                # Use a timeout for receive to allow checking sleep_requested_event
                # However, session.receive() is a generator, making direct timeout tricky.
                # The primary exit from this loop when sleeping will be the ConnectionClosedError
                # when self.session.close() is called in go_to_sleep.

                turn_iterator = self.session.receive()
                async for resp in turn_iterator:

                    """ if getattr(resp, "input_audio_transcription", None):
                        print("YOU  >", resp.input_audio_transcription.text)

                    if getattr(resp, "output_audio_transcription", None):
                         print("KARL >", resp.output_audio_transcription.text) """

                    if self.sleep_requested_event.is_set():
                        print("[Receiver] Sleep event detected mid-turn, breaking from turn processing.")
                        break # Exit from processing messages in the current turn

                    #print(f"Processing response: {resp}")
                    
                    if resp.data and not is_speaking:
                        print("Starting to speak...")
                        speak_animation_key = f"speak_{self.current_speaking_emotion}"
                        self.anim.set_mode(speak_animation_key)
                        print(f"Speaking with emotion: {self.current_speaking_emotion} (animation key: {speak_animation_key})...")
                        is_speaking = True
                    
                    if resp.data:
                        #print("Writing audio data...")
                        # self.recv_wf.writeframes(resp.data) # Blocking call
                        await self.loop.run_in_executor(None, self.recv_wf.writeframes, resp.data) # Non-blocking
                        await self.audio_in_q.put(resp.data)
                    
                    if resp.text:
                        print(f"\nGemini text response: {resp.text}")
                    
                    # Handle function calls based on tool_call
                    if resp.tool_call and resp.tool_call.function_calls:
                        for fc in resp.tool_call.function_calls:
                            function_name = fc.name
                            function_args = fc.args or {}
                            function_id = fc.id  # Crucial for the response

                            print(f"\nTool call detected: {function_name} with args {function_args}, ID: {function_id}")

                            if function_name in self.functions_map:
                                func_to_call = self.functions_map[function_name]
                                try:
                                    print(f"Executing function {function_name}...")
                                    # If it's an async function (like go_to_sleep), await it
                                    if asyncio.iscoroutinefunction(func_to_call):
                                        result = await func_to_call(**function_args)
                                    else:
                                        result = func_to_call(**function_args)
                                    print(f"Function result: {result}")
                                    
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
                                        print("Tool response sent back to Gemini")
                                except Exception as e:
                                    print(f"Error executing function {function_name}: {e}")
                                    # If it's go_to_sleep that failed, session might still be open or in weird state
                                    # but generally, we still try to send an error if not go_to_sleep, 
                                    # or if session is still connectable.
                                    if function_name != "go_to_sleep" and self.session:
                                        try:
                                            error_tool_response = types.FunctionResponse(
                                                id=function_id,
                                                name=function_name,
                                                response={"error": str(e)}
                                            )
                                            await self.session.send_tool_response(
                                                function_responses=[error_tool_response]
                                            )
                                        except Exception as e_send:
                                            print(f"Failed to send error response for {function_name}: {e_send}")
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
                    #else:
                    #            print(f"Function {function_name} not found in available functions map.")
                    #            # Send error response back to Gemini if function not found
                    #            error_tool_response = types.FunctionResponse(
                    #                id=function_id,
                    #                name=function_name,
                    #                response={"error": f"Function {function_name} not implemented or available."}
                    #            )
                    #            await self.session.send_tool_response(
                    #                function_responses=[error_tool_response]
                    #            )
                    # else:
                        # print("No tool call in this response part") # Debug if needed
                
                # Show listening status when done speaking
                print("Conversation turn completed, switching to listening mode")
                self.anim.set_mode("idle")
                is_speaking = False  # Reset for the next turn
                print("Listening...")
                
            except websockets.exceptions.ConnectionClosedOK:
                print("[Receiver] Connection closed (OK). Exiting receive-loop.")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"[Receiver] Connection closed with error: {e}. Exiting receive-loop.")
                break
            except asyncio.TimeoutError:
                # This might occur if we implement a timeout around receive(), but it's complex with async iterators
                print("[Receiver] Timeout waiting for response, checking sleep event.")
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
        print("Receive-loop finished.")

    async def _playback(self):
        print("[Playback] Task started.")
        print("Playback‑loop started…")
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
        print("Playback-loop finished.")

    # ────────────────────────── cleanup ─────────────────────────────────────
    def _cleanup(self):
        print("Cleaning up…")
        for s in (self.input_stream, self.output_stream):
            if s and s.is_active():
                s.stop_stream()
            if s:
                s.close()
        for wf in (self.sent_wf, self.recv_wf):
            wf.close()
        self.pya.terminate()
        print("Done.")
        # Cleanup display
        try:
            self.disp.clear()
            self.disp.bl_DutyCycle(0)
            self.disp.module_exit()
        except Exception:
            pass

    # ────────────────────────── main entry ──────────────────────────────────
    async def run(self):
        print("[AudioHandler] Starting run method...")
        if not os.getenv("GOOGLE_API_KEY"):
            print("[AudioHandler] Error: GOOGLE_API_KEY environment variable not set")
            return

        print("[AudioHandler] Setting up streams...")
        self.sleep_requested_event.clear() # Clear event at the start of a new run

        # set loop reference (if not set in __init__)
        if self.loop is None:
            self.loop = asyncio.get_running_loop()

        try:
            await self.setup_streams()
            print("[AudioHandler] Streams setup completed")

            print("[AudioHandler] Connecting to Gemini Live...")
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"),
                                    http_options=types.HttpOptions(api_version="v1beta"))
            
            print("[AudioHandler] Creating Gemini session...")
            async with self.client.aio.live.connect(model=MODEL, config=CONFIG) as sess:
                print("[AudioHandler] Gemini session created successfully")
                self.session = sess
                self.anim.set_mode("idle") # Set initial animation to idle
                print("[AudioHandler] Listening...")

                try:
                    async with asyncio.TaskGroup() as tg:
                        print("[AudioHandler] Creating tasks...")
                        send_task = tg.create_task(self._send_to_gemini())
                        recv_task = tg.create_task(self._recv_from_gemini())
                        playback_task = tg.create_task(self._playback())
                        anim_task = tg.create_task(self.anim.run())
                        vision_task = tg.create_task(self._vision_feed(interval=2))
                        print("[AudioHandler] All tasks created successfully")
                except Exception as e:
                    print(f"[AudioHandler] Error in task group: {e}")
                    print("[AudioHandler] Task group traceback:")
                    traceback.print_exc()
                    raise

                print("[AudioHandler] TaskGroup finished")

            print("[AudioHandler] Gemini session closed")
            self.session = None
        except Exception as e:
            print(f"[AudioHandler] Error in run method: {e}")
            print("[AudioHandler] Run method traceback:")
            traceback.print_exc()
            raise
        finally:
            print("[AudioHandler] Run method completed")

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