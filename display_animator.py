"""
Display Animation Module for VocalGem Robot
Handles animated GIFs on the 240×240 Waveshare screen and LED ring animations.
"""

import asyncio
import time
import sys
import os
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
            
            # Update display
            self.disp.ShowImage(frames[self._idx])
            self._idx = (self._idx + 1) % len(frames)
            
            # Update LEDs based on mode (slower than display to avoid overwhelming)
            led_delay_counter += 1
            if led_delay_counter >= 3:  # Update LEDs every 3 display frames
                led_delay_counter = 0
                
                if self.mode == "idle":
                    self.led_idle_breathing()
                elif self.mode.startswith("speak"):
                    self.led_speaking_chase()
            
            await asyncio.sleep(delay)
        print("[DisplayAnimator] Animation task exiting...") 