"""
Display Animation Module for VocalGem Robot
Handles animated GIFs on the 240×240 Waveshare screen and LED ring animations.
"""

import asyncio
import time
import sys
import os
import numpy as np
import struct
from collections import deque
from PIL import Image, ImageDraw, ImageFont, ImageSequence

# Add path for LED control
sys.path.append("freenove_examples")
try:
    from led import Led
    LED_AVAILABLE = True
except ImportError:
    LED_AVAILABLE = False
    print("Warning: LED module not found. LED animations will be disabled.")

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
        
        # Initialize LED controller
        try:
            if LED_AVAILABLE:
                self.led = Led()
                self.led_initialized = True
                print("[DisplayAnimator] LED controller initialized.")
            else:
                self.led = None
                self.led_initialized = False
        except Exception as e:
            self.led = None
            self.led_initialized = False
            print(f"[DisplayAnimator] LED initialization failed: {e}")
        
        # LED animation state variables
        self.led_init_index = 0
        self.led_init_complete = False
        self.led_breathing_brightness = 0
        self.led_breathing_direction = 1
        self.led_chase_index = 0
        
        # Audio spectrum analysis variables
        self.audio_buffer = deque(maxlen=1024)  # Buffer for audio samples
        self.spectrum_levels = [0] * 8  # 8 frequency bands for 8 LEDs
        self.spectrum_smoothing = 0.5  # Reduced smoothing for more responsive volume meter
        self.volume_threshold = 0.005  # Lower threshold to catch quieter speech
        self.max_volume_seen = 0.1  # Track maximum volume for auto-scaling
        
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
            if self.disp is not None:
                blank = Image.new('RGB', (self.disp.width, self.disp.height), 'BLACK')
            else:
                blank = Image.new('RGB', (240, 240), 'BLACK')  # Default size
            self._frames[mode_key] = [blank]
            print(f"[DisplayAnimator] Fallback: Loaded 1 (blank) frame for mode '{mode_key}' from '{gif_path}'.")

    def update_audio_data(self, pcm_data):
        """Update the audio buffer with new PCM data for spectrum analysis."""
        try:
            # Convert PCM bytes to numpy array (assuming 16-bit signed integers)
            if len(pcm_data) >= 2:
                # Unpack PCM data as 16-bit signed integers
                samples = struct.unpack(f'<{len(pcm_data)//2}h', pcm_data)
                
                # Add samples to buffer
                self.audio_buffer.extend(samples)
                
                # Calculate spectrum if we have enough data
                if len(self.audio_buffer) >= 512:
                    self._calculate_spectrum()
                    
        except Exception as e:
            print(f"[DisplayAnimator] Error processing audio data: {e}")

    def _calculate_spectrum(self):
        """Calculate volume level from audio buffer for VU meter visualization."""
        try:
            # Get the latest samples
            samples = list(self.audio_buffer)[-512:]  # Use last 512 samples
            
            # Convert to numpy array and normalize
            audio_data = np.array(samples, dtype=np.float32) / 32768.0
            
            # Calculate overall volume (RMS)
            volume = np.sqrt(np.mean(audio_data ** 2))
            
            # Update max volume for auto-scaling, but with a more reasonable range
            if volume > self.max_volume_seen:
                self.max_volume_seen = min(volume * 1.1, 0.5)  # Cap at 0.5 to prevent over-scaling
            
            # Only process if volume is above threshold
            if volume < self.volume_threshold:
                # Fade out all levels when quiet
                for i in range(8):
                    self.spectrum_levels[i] *= 0.85  # Faster fade for volume meter
                return
            
            # Normalize volume to 0-1 range with better scaling
            # Map volume levels more conservatively to spread across LED range
            if volume < 0.02:
                led_level = 0  # Very quiet - no LEDs
            elif volume < 0.05:
                led_level = 1  # Quiet - 1 LED
            elif volume < 0.08:
                led_level = 2  # Low - 2 LEDs
            elif volume < 0.12:
                led_level = 3  # Medium-low - 3 LEDs
            elif volume < 0.16:
                led_level = 4  # Medium - 4 LEDs
            elif volume < 0.20:
                led_level = 5  # Medium-high - 5 LEDs
            elif volume < 0.25:
                led_level = 6  # High - 6 LEDs
            elif volume < 0.30:
                led_level = 7  # Very high - 7 LEDs
            else:
                led_level = 8  # Maximum - 8 LEDs
            
            # Create target levels for each LED with gradual falloff
            new_levels = []
            for i in range(8):
                if led_level > i:
                    # Calculate brightness for this LED
                    if led_level >= i + 1:
                        # LED is fully on
                        brightness = 1.0
                    else:
                        # LED is partially on (smooth transition)
                        brightness = led_level - i
                    new_levels.append(brightness)
                else:
                    # LED should be off
                    new_levels.append(0.0)
            
            # Apply smoothing to prevent flickering
            for i in range(8):
                self.spectrum_levels[i] = (
                    self.spectrum_smoothing * self.spectrum_levels[i] + 
                    (1 - self.spectrum_smoothing) * new_levels[i]
                )
                
        except Exception as e:
            print(f"[DisplayAnimator] Error calculating volume: {e}")

    def led_initialization_sequence(self):
        """LED initialization: light up one by one until full ring blue."""
        if not self.led_initialized:
            return
            
        try:
            if self.led_init_index < 8:  # Assuming 8 LEDs in the ring
                # Light up LEDs one by one in blue
                for i in range(self.led_init_index + 1):
                    self.led.strip.set_led_rgb_data(i, [0, 0, 255])  # Blue
                for i in range(self.led_init_index + 1, 8):
                    self.led.strip.set_led_rgb_data(i, [0, 0, 0])    # Off
                self.led.strip.show()
                self.led_init_index += 1
            else:
                # All LEDs blue - initialization complete
                self.led.strip.set_all_led_color(0, 0, 255)  # All blue
                self.led.strip.show()
                self.led_init_complete = True
        except Exception as e:
            print(f"[DisplayAnimator] LED initialization error: {e}")

    def led_idle_breathing(self):
        """LED idle: purple breathing effect."""
        if not self.led_initialized:
            return
            
        try:
            # Purple breathing: fade in and out
            self.led_breathing_brightness += self.led_breathing_direction * 10
            if self.led_breathing_brightness >= 255:
                self.led_breathing_brightness = 255
                self.led_breathing_direction = -1
            elif self.led_breathing_brightness <= 30:  # Don't go completely dark
                self.led_breathing_brightness = 30
                self.led_breathing_direction = 1
            
            # Purple color (R=128, G=0, B=128) scaled by brightness
            r = int(128 * self.led_breathing_brightness / 255)
            g = 0
            b = int(128 * self.led_breathing_brightness / 255)
            
            self.led.strip.set_all_led_color(r, g, b)
            self.led.strip.show()
        except Exception as e:
            print(f"[DisplayAnimator] LED breathing error: {e}")

    def led_speaking_chase(self):
        """LED speaking: fast red chasing light like Kitt from K2000."""
        if not self.led_initialized:
            return
            
        try:
            # Clear all LEDs
            for i in range(8):
                self.led.strip.set_led_rgb_data(i, [0, 0, 0])
            
            # Set the current LED to red with some trailing effect
            self.led.strip.set_led_rgb_data(self.led_chase_index, [255, 0, 0])  # Bright red
            
            # Add trailing LEDs with dimmed red for smoother effect
            prev_index = (self.led_chase_index - 1) % 8
            self.led.strip.set_led_rgb_data(prev_index, [100, 0, 0])  # Dimmed red
            
            self.led.strip.show()
            
            # Move to next LED
            self.led_chase_index = (self.led_chase_index + 1) % 8
        except Exception as e:
            print(f"[DisplayAnimator] LED chase error: {e}")

    def led_spectrum_analyzer(self):
        """LED volume meter: visualize audio volume level."""
        if not self.led_initialized:
            return
            
        try:
            # Create a volume meter effect
            for i in range(8):
                level = self.spectrum_levels[i]
                
                # Color gradient from green (quiet) to red (loud)
                if level > 0.05:  # Only show if level is significant
                    # Create color gradient: green -> yellow -> orange -> red
                    if i < 3:  # First 3 LEDs: Green (quiet levels)
                        r = int(100 * level)
                        g = int(255 * level)
                        b = 0
                    elif i < 5:  # Middle LEDs: Yellow/Orange (medium levels)
                        r = int(255 * level)
                        g = int(200 * level)
                        b = 0
                    else:  # Last 3 LEDs: Red (loud levels)
                        r = int(255 * level)
                        g = int(50 * level)
                        b = 0
                    
                    self.led.strip.set_led_rgb_data(i, [r, g, b])
                else:
                    self.led.strip.set_led_rgb_data(i, [0, 0, 0])
            
            self.led.strip.show()
            
        except Exception as e:
            print(f"[DisplayAnimator] LED volume meter error: {e}")

    def led_turn_off(self):
        """Turn off all LEDs."""
        if not self.led_initialized:
            return
            
        try:
            self.led.strip.set_all_led_color(0, 0, 0)
            self.led.strip.show()
        except Exception as e:
            print(f"[DisplayAnimator] LED turn off error: {e}")

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

    async def run_initialization(self):
        """Run the LED initialization sequence."""
        if not self.led_initialized:
            print("[DisplayAnimator] LEDs not available for initialization.")
            return
            
        print("[DisplayAnimator] Starting LED initialization sequence...")
        while not self.led_init_complete and not (self.stop_event and self.stop_event.is_set()):
            self.led_initialization_sequence()
            await asyncio.sleep(0.2)  # Slower for initialization
        
        # Keep full blue ring for a moment
        await asyncio.sleep(1)
        print("[DisplayAnimator] LED initialization complete.")

    async def run(self):
        print("[DisplayAnimator] Task started.")
        delay = 1 / self.fps
        led_delay_counter = 0
        
        while True:
            if self.stop_event and self.stop_event.is_set():
                print("[DisplayAnimator] Stop event received, turning off LEDs and exiting.")
                self.led_turn_off()
                break

            frames = self._frames[self.mode]
            if not frames:            # no frames yet
                await asyncio.sleep(delay)
                continue
            
            # Update display (only if display is available)
            if self.disp is not None:
                self.disp.ShowImage(frames[self._idx])
            self._idx = (self._idx + 1) % len(frames)
            
            # Update LEDs based on mode (faster for spectrum analyzer)
            led_delay_counter += 1
            led_update_frequency = 1 if self.mode.startswith("speak") else 3
            
            if led_delay_counter >= led_update_frequency:
                led_delay_counter = 0
                
                if self.mode == "idle":
                    self.led_idle_breathing()
                elif self.mode.startswith("speak"):
                    # Use spectrum analyzer for all speaking modes
                    self.led_spectrum_analyzer()
            
            await asyncio.sleep(delay)
        print("[DisplayAnimator] Animation task exiting...") 