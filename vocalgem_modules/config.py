"""
Configuration module for VocalGem robot
Contains all constants, settings, and configuration objects
"""

import os
import warnings
from dotenv import load_dotenv

# Conditional import for Google Genai types
try:
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    types = None

# Load environment variables from .env file
load_dotenv()

# Simple warning suppression for inline_data
warnings.filterwarnings("ignore", message=".*non-text parts in the response.*inline_data.*")

# ─── Audio constants ──────────────────────────────────────────────────────────
FORMAT = "paInt16"  # Will be converted to pyaudio.paInt16
IN_CH = 1   # use processed mono channel
OUT_CH = 1   # we still play/record mono
CHUNK_MS = 20             # one packet = 20 ms of audio
SEND_SAMPLE_RATE = 16_000     # Gemini Live input format
RECEIVE_SAMPLE_RATE = 24_000  # Gemini Live output format
AEC_SAMPLE_RATE = 16_000          
RAW_CH = 6

# ─── Gemini constants ────────────────────────────────────────────────────────
MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"  # Better audio quality with unstable function calling support

# Define tools for the LiveConnectConfig - these are static declarations
tools_for_config = []
if GOOGLE_GENAI_AVAILABLE:
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

CONFIG = None
if GOOGLE_GENAI_AVAILABLE:
    CONFIG = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        media_resolution="MEDIA_RESOLUTION_LOW",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Algenib")
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

# ─── Camera constants ────────────────────────────────────────────────────────
CAMERA_RESOLUTION = (640, 480)
CAMERA_JPEG_QUALITY = 80
CAMERA_FRAME_INTERVAL = 8  # seconds between frames when face detection is disabled

# ─── Servo constants ────────────────────────────────────────────────────────
PAN_MIN = 0   # Full left range (extended from 5)
PAN_MAX = 180 # Full right range (extended from 175)
TILT_MIN = 5  # Aggressive up range (extended from 20)
TILT_MAX = 115 # Extended down range (extended from 110)

# ─── Display constants ──────────────────────────────────────────────────────
DISPLAY_WIDTH = 240
DISPLAY_HEIGHT = 240
DISPLAY_FONT_PATH = "display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/../Font/Font01.ttf"
DISPLAY_FONT_SIZE = 24

# ─── I2C constants ──────────────────────────────────────────────────────────
I2C_ADDR = 0x2d  # I2C address of the UPS

# ─── Environment variables ──────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 