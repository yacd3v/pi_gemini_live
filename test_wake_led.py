#!/usr/bin/env python3
"""
Test script for LED integration with wake_porcu.py
This script tests the LED controller functionality without running the full wake word detection.
"""

import sys
import time
import logging

# Add path for freenove examples
sys.path.append("freenove_examples")

# Import the LED controller from wake_porcu
from wake_porcu import LEDController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

def test_led_sequence():
    """Test the LED sequence used in wake word detection"""
    
    print("\n" + "="*50)
    print("Testing Wake Word LED Integration")
    print("="*50)
    
    # Initialize LED controller
    led_controller = LEDController()
    
    if not led_controller.led_initialized:
        print("❌ LED controller not available. Check your hardware setup.")
        return False
    
    try:
        print("\n1. Testing Listening Mode (Blue LEDs)...")
        led_controller.set_listening_mode()
        print("   💙 LEDs should be BLUE (listening for wake words)")
        time.sleep(3)
        
        print("\n2. Testing Wake Word Detected Mode (Green LEDs)...")
        led_controller.set_wake_detected_mode()
        print("   💚 LEDs should be GREEN (wake word detected)")
        time.sleep(2)
        
        print("\n3. Testing Turn Off (Gemini Session)...")
        led_controller.turn_off()
        print("   ⚫ LEDs should be OFF (during Gemini conversation)")
        time.sleep(2)
        
        print("\n4. Back to Listening Mode...")
        led_controller.set_listening_mode()
        print("   💙 LEDs should be BLUE again (ready for next wake word)")
        time.sleep(2)
        
        print("\n5. Final cleanup...")
        led_controller.cleanup()
        print("   ⚫ LEDs turned off")
        
        print("\n" + "="*50)
        print("✅ LED test completed successfully!")
        print("Your wake_porcu.py script is ready with LED indicators:")
        print("  💙 Blue = Listening for wake words")
        print("  💚 Green = Wake word detected")
        print("  ⚫ Off = During Gemini conversation")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ LED test failed: {e}")
        led_controller.cleanup()
        return False

def main():
    print("Starting LED integration test for wake word detection...")
    
    try:
        success = test_led_sequence()
        if success:
            print("\n🎉 Your LED setup is working correctly!")
            print("You can now run wake_porcu.py with LED indicators.")
        else:
            print("\n⚠️  LED setup needs attention. Check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main() 