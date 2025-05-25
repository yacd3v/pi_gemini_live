#!/usr/bin/env python3
"""
Test script for LED integration with vocal_gemini.py
This script tests the LED animations without requiring Gemini connection.
"""

import asyncio
import time
import sys

# Add path for freenove examples
sys.path.append("freenove_examples")
from led import Led

class LEDTester:
    def __init__(self):
        try:
            self.led = Led()
            self.led_initialized = True
            print("[LEDTester] LED controller initialized successfully.")
        except Exception as e:
            self.led = None
            self.led_initialized = False
            print(f"[LEDTester] LED initialization failed: {e}")
            return

        # LED animation state variables
        self.led_init_index = 0
        self.led_init_complete = False
        self.led_breathing_brightness = 0
        self.led_breathing_direction = 1
        self.led_chase_index = 0

    def led_initialization_sequence(self):
        """LED initialization: light up one by one until full ring blue."""
        if not self.led_initialized:
            return False
            
        try:
            if self.led_init_index < 8:  # Assuming 8 LEDs in the ring
                # Light up LEDs one by one in blue
                for i in range(self.led_init_index + 1):
                    self.led.strip.set_led_rgb_data(i, [0, 0, 255])  # Blue
                for i in range(self.led_init_index + 1, 8):
                    self.led.strip.set_led_rgb_data(i, [0, 0, 0])    # Off
                self.led.strip.show()
                self.led_init_index += 1
                return False  # Not complete yet
            else:
                # All LEDs blue - initialization complete
                self.led.strip.set_all_led_color(0, 0, 255)  # All blue
                self.led.strip.show()
                self.led_init_complete = True
                return True  # Complete
        except Exception as e:
            print(f"[LEDTester] LED initialization error: {e}")
            return True  # Exit on error

    def led_idle_breathing(self):
        """LED idle: purple breathing effect."""
        if not self.led_initialized:
            return
            
        try:
            # Purple breathing: fade in and out
            self.led_breathing_brightness += self.led_breathing_direction * 5
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
            print(f"[LEDTester] LED breathing error: {e}")

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
            print(f"[LEDTester] LED chase error: {e}")

    def led_turn_off(self):
        """Turn off all LEDs."""
        if not self.led_initialized:
            return
            
        try:
            self.led.strip.set_all_led_color(0, 0, 0)
            self.led.strip.show()
        except Exception as e:
            print(f"[LEDTester] LED turn off error: {e}")

    async def test_sequence(self):
        """Test all LED animations in sequence."""
        if not self.led_initialized:
            print("LED controller not available. Exiting.")
            return

        print("\n" + "="*50)
        print("Testing LED Integration for Vocal Gemini")
        print("="*50)

        # Test 1: Initialization sequence
        print("\n1. Testing Initialization Sequence (Blue LEDs lighting up one by one)...")
        while not self.led_init_complete:
            complete = self.led_initialization_sequence()
            if complete:
                break
            await asyncio.sleep(0.2)
        
        # Keep full blue for a moment
        await asyncio.sleep(2)
        print("   ✓ Initialization sequence complete!")

        # Test 2: Idle breathing
        print("\n2. Testing Idle Breathing (Purple breathing for 10 seconds)...")
        start_time = time.time()
        while time.time() - start_time < 10:
            self.led_idle_breathing()
            await asyncio.sleep(0.1)
        print("   ✓ Idle breathing test complete!")

        # Test 3: Speaking chase
        print("\n3. Testing Speaking Chase (Red chasing light for 10 seconds)...")
        start_time = time.time()
        while time.time() - start_time < 10:
            self.led_speaking_chase()
            await asyncio.sleep(0.08)  # Faster for chase effect
        print("   ✓ Speaking chase test complete!")

        # Test 4: Turn off
        print("\n4. Turning off all LEDs...")
        self.led_turn_off()
        await asyncio.sleep(1)
        print("   ✓ LEDs turned off!")

        print("\n" + "="*50)
        print("All LED tests completed successfully!")
        print("You can now run vocal_gemini.py with LED support.")
        print("="*50)

async def main():
    tester = LEDTester()
    try:
        await tester.test_sequence()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        tester.led_turn_off()
    except Exception as e:
        print(f"Test error: {e}")
        tester.led_turn_off()

if __name__ == "__main__":
    print("Starting LED integration test...")
    asyncio.run(main()) 