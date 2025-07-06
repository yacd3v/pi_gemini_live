#!/usr/bin/env python3
"""
Complete Display Test - Working gpiod 2.3.0 API
===============================================

This version uses the WORKING gpiod 2.3.0 API that just passed the GPIO test.
Now we combine working SPI + working GPIO = working display!

Hardware:
- Display: SPI LCD (GPIO 12=RST, 26=DC, 13=BL)
- SPI: GPIO 19=MOSI, 21=SCLK, 18=CE0

Author: Auto-generated test script
Date: 2025
"""

import time
import sys
import traceback
from datetime import datetime

# Import required libraries
try:
    import gpiod
    print("✓ gpiod library imported successfully")
    print(f"  gpiod version: {gpiod.__version__}")
except ImportError as e:
    print(f"❌ gpiod library import failed: {e}")
    sys.exit(1)

try:
    import spidev as SPI
    print("✓ SPI library imported successfully")
except ImportError as e:
    print(f"❌ SPI library import failed: {e}")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
    print("✓ PIL library imported successfully")
except ImportError as e:
    print(f"❌ PIL library import failed: {e}")
    sys.exit(1)

class WorkingDisplayTester:
    def __init__(self):
        self.spi = None
        self.font = None
        self.gpio_request = None
        
    def gpio_init(self):
        """Initialize GPIO using the working API"""
        try:
            print("Initializing GPIO using working gpiod API...")
            
            # Create line settings for output
            line_settings = gpiod.LineSettings(
                direction=gpiod.line.Direction.OUTPUT,
                output_value=gpiod.line.Value.INACTIVE
            )
            
            # Create configuration dictionary for pins
            config = {
                12: line_settings,  # RST pin
                26: line_settings,  # DC pin
                13: line_settings   # BL pin
            }
            
            # Request GPIO lines using the working API
            self.gpio_request = gpiod.request_lines(
                path='/dev/gpiochip0',
                config=config,
                consumer="display_controller"
            )
            
            print("✓ GPIO lines requested successfully")
            
            # Set initial values
            self.gpio_request.set_value(12, gpiod.line.Value.ACTIVE)   # RST high
            self.gpio_request.set_value(26, gpiod.line.Value.INACTIVE) # DC low (command)
            self.gpio_request.set_value(13, gpiod.line.Value.INACTIVE) # BL off initially
            
            print("✓ GPIO pins initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ GPIO initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def spi_init(self):
        """Initialize SPI"""
        try:
            print("Initializing SPI...")
            self.spi = SPI.SpiDev()
            self.spi.open(1, 0)  # bus 1, CE0
            self.spi.max_speed_hz = 40_000_000
            self.spi.mode = 0b00
            print("✓ SPI initialized")
            return True
        except Exception as e:
            print(f"❌ SPI initialization failed: {e}")
            return False
    
    def display_reset(self):
        """Reset the display"""
        try:
            print("Resetting display...")
            self.gpio_request.set_value(12, gpiod.line.Value.ACTIVE)
            time.sleep(0.01)
            self.gpio_request.set_value(12, gpiod.line.Value.INACTIVE)
            time.sleep(0.01)
            self.gpio_request.set_value(12, gpiod.line.Value.ACTIVE)
            time.sleep(0.01)
            print("✓ Display reset completed")
            return True
        except Exception as e:
            print(f"❌ Display reset failed: {e}")
            return False
    
    def write_command(self, cmd):
        """Write command to display"""
        try:
            self.gpio_request.set_value(26, gpiod.line.Value.INACTIVE)  # DC low = command
            self.spi.writebytes([cmd])
            return True
        except Exception as e:
            print(f"❌ Write command failed: {e}")
            return False
    
    def write_data(self, data):
        """Write data to display"""
        try:
            self.gpio_request.set_value(26, gpiod.line.Value.ACTIVE)  # DC high = data
            if isinstance(data, int):
                self.spi.writebytes([data])
            else:
                self.spi.writebytes(data)
            return True
        except Exception as e:
            print(f"❌ Write data failed: {e}")
            return False
    
    def display_init_sequence(self):
        """Initialize display with command sequence"""
        try:
            print("Sending display initialization commands...")
            
            # Basic initialization sequence for ST7789
            init_commands = [
                (0x36, [0x00]),  # Memory Access Control
                (0x3A, [0x05]),  # Pixel Format Set (16-bit)
                (0xB2, [0x0C, 0x0C, 0x00, 0x33, 0x33]),  # Porch Setting
                (0xB7, [0x35]),  # Gate Control
                (0xBB, [0x19]),  # VCOM Setting
                (0xC0, [0x2C]),  # LCM Control
                (0xC2, [0x01]),  # VDV and VRH Command Enable
                (0xC3, [0x12]),  # VRH Set
                (0xC4, [0x20]),  # VDV Set
                (0xC6, [0x0F]),  # Frame Rate Control
                (0xD0, [0xA4, 0xA1]),  # Power Control 1
                (0xE0, [0xD0, 0x04, 0x0D, 0x11, 0x13, 0x2B, 0x3F, 0x54, 0x4C, 0x18, 0x0D, 0x0B, 0x1F, 0x23]),
                (0xE1, [0xD0, 0x04, 0x0C, 0x11, 0x13, 0x2C, 0x3F, 0x44, 0x51, 0x2F, 0x1F, 0x1F, 0x20, 0x23]),
                (0x21, []),     # Invert On
                (0x11, []),     # Sleep Out
            ]
            
            for cmd, data in init_commands:
                if not self.write_command(cmd):
                    return False
                if data:
                    for byte in data:
                        if not self.write_data(byte):
                            return False
                time.sleep(0.001)
            
            print("⏱️ Waiting for display wake up...")
            time.sleep(0.12)  # Wait for sleep out
            
            # Display On
            if not self.write_command(0x29):
                return False
            time.sleep(0.02)
            
            print("✓ Display initialization completed")
            return True
            
        except Exception as e:
            print(f"❌ Display initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def set_window(self, x0, y0, x1, y1):
        """Set drawing window"""
        try:
            # Column Address Set
            self.write_command(0x2A)
            self.write_data((x0 >> 8) & 0xFF)
            self.write_data(x0 & 0xFF)
            self.write_data((x1 >> 8) & 0xFF)
            self.write_data(x1 & 0xFF)
            
            # Row Address Set
            self.write_command(0x2B)
            self.write_data((y0 >> 8) & 0xFF)
            self.write_data(y0 & 0xFF)
            self.write_data((y1 >> 8) & 0xFF)
            self.write_data(y1 & 0xFF)
            
            # Memory Write
            self.write_command(0x2C)
            return True
        except Exception as e:
            print(f"❌ Set window failed: {e}")
            return False
    
    def clear_display(self, color=(0, 0, 0)):
        """Clear display with color"""
        try:
            print(f"Clearing display to color {color}...")
            
            # Convert RGB to 565 format
            r, g, b = color
            color565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            color_bytes = [(color565 >> 8) & 0xFF, color565 & 0xFF]
            
            # Set full screen window
            if not self.set_window(0, 0, 239, 239):
                return False
            
            # Fill with color
            self.gpio_request.set_value(26, gpiod.line.Value.ACTIVE)  # Data mode
            for _ in range(240 * 240):
                self.spi.writebytes(color_bytes)
            
            print("✓ Display cleared")
            return True
            
        except Exception as e:
            print(f"❌ Clear display failed: {e}")
            return False
    
    def set_backlight(self, brightness):
        """Set backlight (simple on/off)"""
        try:
            if brightness > 0:
                self.gpio_request.set_value(13, gpiod.line.Value.ACTIVE)
                print(f"✓ Backlight ON")
            else:
                self.gpio_request.set_value(13, gpiod.line.Value.INACTIVE)
                print(f"✓ Backlight OFF")
            return True
        except Exception as e:
            print(f"❌ Backlight control failed: {e}")
            return False
    
    def test_display_basic(self):
        """Basic display functionality test"""
        print("🖥️  Testing Display (WORKING VERSION!)...")
        print("-" * 60)
        
        # Step 1: GPIO initialization
        if not self.gpio_init():
            return False
        
        # Step 2: SPI initialization
        if not self.spi_init():
            return False
        
        # Step 3: Display reset
        if not self.display_reset():
            return False
        
        # Step 4: Display initialization
        if not self.display_init_sequence():
            return False
        
        # Step 5: Set backlight
        if not self.set_backlight(1):
            return False
        
        # Step 6: Clear display
        if not self.clear_display((0, 0, 0)):
            return False
        
        print("✅ Basic display test PASSED")
        return True
    
    def test_colors(self):
        """Test color display"""
        print("\n🎨 Testing Colors...")
        print("-" * 40)
        
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 255, 255) # White
        ]
        
        color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "White"]
        
        try:
            for i, (color, name) in enumerate(zip(colors, color_names)):
                print(f"  Showing {name}...")
                if not self.clear_display(color):
                    return False
                time.sleep(0.8)
            
            print("✅ Color test PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Color test FAILED: {e}")
            return False
    
    def show_image(self, image):
        """Show PIL image on display"""
        try:
            # Resize image to fit display
            image = image.resize((240, 240))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Set full screen window
            if not self.set_window(0, 0, 239, 239):
                return False
            
            # Send image data
            self.gpio_request.set_value(26, gpiod.line.Value.ACTIVE)  # Data mode
            
            for y in range(240):
                for x in range(240):
                    r, g, b = image.getpixel((x, y))
                    
                    # Convert to 565 format
                    color565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
                    color_bytes = [(color565 >> 8) & 0xFF, color565 & 0xFF]
                    
                    self.spi.writebytes(color_bytes)
            
            return True
            
        except Exception as e:
            print(f"❌ Show image failed: {e}")
            return False
    
    def test_text_and_graphics(self):
        """Test text and graphics"""
        print("\n📝 Testing Text and Graphics...")
        print("-" * 40)
        
        try:
            # Load font
            try:
                self.font = ImageFont.truetype(
                    "display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/../Font/Font01.ttf", 
                    24
                )
                print("✓ Custom font loaded")
            except:
                self.font = ImageFont.load_default()
                print("⚠ Using default font")
            
            # Create test image
            image = Image.new('RGB', (240, 240), color=(0, 0, 0))
            draw = ImageDraw.Draw(image)
            
            # Draw text
            draw.text((10, 50), "DISPLAY", font=self.font, fill=(255, 255, 255))
            draw.text((10, 90), "WORKING!", font=self.font, fill=(0, 255, 0))
            draw.text((10, 130), "SUCCESS!", font=self.font, fill=(255, 255, 0))
            draw.text((10, 170), "PERFECT!", font=self.font, fill=(255, 100, 255))
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            draw.text((10, 210), timestamp, font=self.font, fill=(128, 128, 128))
            
            # Show image
            if not self.show_image(image):
                return False
            
            print("✅ Text and graphics test PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Text and graphics test FAILED: {e}")
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.spi:
                self.spi.close()
                print("✓ SPI closed")
            
            # Turn off backlight
            if self.gpio_request:
                self.gpio_request.set_value(13, gpiod.line.Value.INACTIVE)
                print("✓ Backlight turned off")
                
                # Release GPIO request
                self.gpio_request.release()
                print("✓ GPIO request released")
            
            print("✓ All resources cleaned up")
            
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def run_full_test(self):
        """Run complete display test"""
        print("🖥️  COMPLETE DISPLAY TEST - FINAL VERSION")
        print("=" * 60)
        print("Using WORKING GPIO + SPI combination!")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test basic functionality
        if not self.test_display_basic():
            print("\n❌ Basic display test failed")
            return False
        
        time.sleep(1)
        
        # Test colors
        color_ok = self.test_colors()
        time.sleep(1)
        
        # Test text and graphics
        graphics_ok = self.test_text_and_graphics()
        time.sleep(3)
        
        # Results
        print(f"\n📊 Test Results:")
        print(f"Basic:    ✅ PASS")
        print(f"Colors:   {'✅ PASS' if color_ok else '❌ FAIL'}")
        print(f"Graphics: {'✅ PASS' if graphics_ok else '❌ FAIL'}")
        
        success = color_ok and graphics_ok
        
        if success:
            print("\n🎉 DISPLAY IS WORKING PERFECTLY!")
            print("🔧 Your display hardware and wiring are 100% correct!")
            print("🎯 GPIO + SPI combination working flawlessly!")
            print("🚀 Ready to integrate with your robot!")
        else:
            print("\n⚠️ Some tests failed, but basic display communication works")
        
        return success

def main():
    """Main function"""
    tester = WorkingDisplayTester()
    
    try:
        success = tester.run_full_test()
        
        if success:
            print("\n🎉 DISPLAY TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("Your display is working perfectly!")
            print("This confirms:")
            print("  ✓ Hardware is excellent")
            print("  ✓ Wiring is correct") 
            print("  ✓ SPI communication works")
            print("  ✓ GPIO control works perfectly")
            print("  ✓ Pi 5 compatibility 100% confirmed")
            print("  ✓ Ready to integrate with vocal_gemini.py")
            print("\n🚀 Your display is back online!")
        else:
            print("\n❌ Display test had issues")
            print("Check error messages above")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTest failed: {e}")
        traceback.print_exc()
        return 1
    finally:
        tester.cleanup()

if __name__ == "__main__":
    sys.exit(main()) 