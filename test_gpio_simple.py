#!/usr/bin/env python3
"""
Simple GPIO Test for Pi 5 - Correct gpiod 2.3.0 API
===================================================

Tests GPIO control using the correct gpiod 2.3.0 API.
This version uses the dictionary configuration method.
"""

import time
import sys
import traceback

try:
    import gpiod
    print("✓ gpiod library imported successfully")
    print(f"  gpiod version: {gpiod.__version__}")
except ImportError as e:
    print(f"❌ gpiod library import failed: {e}")
    sys.exit(1)

def test_gpio_control():
    """Test GPIO control functionality with correct API"""
    try:
        print("🔧 Testing GPIO Control for Pi 5...")
        print("-" * 50)
        
        # Define GPIO pins
        gpio_pins = [12, 26, 13]  # RST, DC, BL
        gpio_names = ['RST', 'DC', 'BL']
        
        # Create line settings for output
        line_settings = gpiod.LineSettings(
            direction=gpiod.line.Direction.OUTPUT,
            output_value=gpiod.line.Value.INACTIVE
        )
        
        # Create configuration dictionary
        config = {}
        for pin in gpio_pins:
            config[pin] = line_settings
        
        print("Requesting GPIO lines...")
        print(f"  Pins: {gpio_pins}")
        
        # Request lines using the correct API
        request = gpiod.request_lines(
            path='/dev/gpiochip0',
            config=config,
            consumer="gpio_test"
        )
        
        print("✓ GPIO lines requested successfully")
        
        # Test GPIO operations
        print("\n🧪 Testing GPIO Operations...")
        
        for pin, name in zip(gpio_pins, gpio_names):
            print(f"  Testing {name} pin (GPIO {pin})...")
            
            # Test pin control - toggle 3 times
            for i in range(3):
                request.set_value(pin, gpiod.line.Value.ACTIVE)
                time.sleep(0.1)
                request.set_value(pin, gpiod.line.Value.INACTIVE)
                time.sleep(0.1)
            
            print(f"    ✓ {name} pin control works")
        
        # Clean up
        request.release()
        print("✓ GPIO request released")
        
        print("\n✅ All GPIO controls working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ GPIO control test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🧪 Simple GPIO Test for Pi 5 (Correct API)")
    print("=" * 50)
    print("Testing GPIO control with gpiod 2.3.0...")
    print()
    
    success = test_gpio_control()
    
    if success:
        print("\n🎉 GPIO CONTROL IS WORKING!")
        print("=" * 50)
        print("✓ GPIO pins can be controlled")
        print("✓ gpiod 2.3.0 API working perfectly")
        print("✓ Ready for full display test")
        print("\n🚀 All three display control pins are working!")
        print("   - RST (GPIO 12): Reset control ✓")
        print("   - DC (GPIO 26): Command/Data control ✓") 
        print("   - BL (GPIO 13): Backlight control ✓")
        print("\nNext: Your display should work now!")
    else:
        print("\n❌ GPIO control failed")
        print("Check error messages above")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 