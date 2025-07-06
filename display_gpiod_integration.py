#!/usr/bin/env python3
"""
Display Integration - Working gpiod 2.3.0 API
=============================================

Drop-in replacement for vocal_gemini.py display initialization.
Uses the confirmed working gpiod 2.3.0 API instead of conflicting libraries.

This creates a display object that matches the interface expected by vocal_gemini.py
but uses the working GPIO control method we just confirmed.
"""

import time
import gpiod
import spidev as SPI
from PIL import Image

class WorkingLCD:
    """LCD display using working gpiod 2.3.0 API"""
    
    def __init__(self):
        self.width = 240
        self.height = 240
        self.spi = None
        self.gpio_request = None
        
    def Init(self):
        """Initialize the display using working gpiod API"""
        try:
            # Initialize SPI
            self.spi = SPI.SpiDev()
            self.spi.open(1, 0)  # bus 1, CE0
            self.spi.max_speed_hz = 40_000_000
            self.spi.mode = 0b00
            
            # Initialize GPIO using working API
            line_settings = gpiod.LineSettings(
                direction=gpiod.line.Direction.OUTPUT,
                output_value=gpiod.line.Value.INACTIVE
            )
            
            config = {
                12: line_settings,  # RST pin
                26: line_settings,  # DC pin
                13: line_settings   # BL pin
            }
            
            self.gpio_request = gpiod.request_lines(
                path='/dev/gpiochip0',
                config=config,
                consumer="vocal_gemini_display"
            )
            
            # Reset sequence
            self.gpio_request.set_value(12, gpiod.line.Value.ACTIVE)
            time.sleep(0.01)
            self.gpio_request.set_value(12, gpiod.line.Value.INACTIVE)
            time.sleep(0.01)
            self.gpio_request.set_value(12, gpiod.line.Value.ACTIVE)
            time.sleep(0.01)
            
            # Initialize display
            self._init_display_commands()
            
            print("✅ Display initialized with working gpiod API")
            return True
            
        except Exception as e:
            print(f"❌ Display initialization failed: {e}")
            self._cleanup()
            return False
    
    def _init_display_commands(self):
        """Send display initialization commands"""
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
            self._write_command(cmd)
            if data:
                for byte in data:
                    self._write_data(byte)
            time.sleep(0.001)
        
        time.sleep(0.12)  # Wait for sleep out
        
        # Display On
        self._write_command(0x29)
        time.sleep(0.02)
    
    def _write_command(self, cmd):
        """Write command to display"""
        self.gpio_request.set_value(26, gpiod.line.Value.INACTIVE)  # DC low = command
        self.spi.writebytes([cmd])
    
    def _write_data(self, data):
        """Write data to display"""
        self.gpio_request.set_value(26, gpiod.line.Value.ACTIVE)  # DC high = data
        if isinstance(data, int):
            self.spi.writebytes([data])
        else:
            self.spi.writebytes(data)
    
    def clear(self):
        """Clear display to black"""
        try:
            # Set full screen window
            self._write_command(0x2A)  # Column Address Set
            self._write_data([0x00, 0x00, 0x00, 0xEF])  # 0-239
            
            self._write_command(0x2B)  # Row Address Set
            self._write_data([0x00, 0x00, 0x00, 0xEF])  # 0-239
            
            self._write_command(0x2C)  # Memory Write
            
            # Send black pixels
            black_data = [0x00] * (240 * 240 * 2)
            self.gpio_request.set_value(26, gpiod.line.Value.ACTIVE)  # Data mode
            
            # Send in chunks to avoid memory issues
            chunk_size = 4096
            for i in range(0, len(black_data), chunk_size):
                chunk = black_data[i:i+chunk_size]
                self.spi.writebytes(chunk)
                
        except Exception as e:
            print(f"Error clearing display: {e}")
    
    def bl_DutyCycle(self, duty):
        """Set backlight brightness (0-100)"""
        try:
            if duty > 50:
                self.gpio_request.set_value(13, gpiod.line.Value.ACTIVE)
            else:
                self.gpio_request.set_value(13, gpiod.line.Value.INACTIVE)
        except Exception as e:
            print(f"Error setting backlight: {e}")
    
    def ShowImage(self, image):
        """Show PIL image on display"""
        try:
            # Resize image to fit display
            if image.size != (self.width, self.height):
                image = image.resize((self.width, self.height))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Set full screen window
            self._write_command(0x2A)  # Column Address Set
            self._write_data([0x00, 0x00, 0x00, 0xEF])  # 0-239
            
            self._write_command(0x2B)  # Row Address Set
            self._write_data([0x00, 0x00, 0x00, 0xEF])  # 0-239
            
            self._write_command(0x2C)  # Memory Write
            
            # Convert image to display format
            pixels = list(image.getdata())
            display_data = []
            
            for r, g, b in pixels:
                # Convert RGB888 to RGB565
                r565 = (r >> 3) << 11
                g565 = (g >> 2) << 5
                b565 = b >> 3
                rgb565 = r565 | g565 | b565
                
                # Split into bytes (big endian)
                display_data.append((rgb565 >> 8) & 0xFF)
                display_data.append(rgb565 & 0xFF)
            
            # Send image data
            self.gpio_request.set_value(26, gpiod.line.Value.ACTIVE)  # Data mode
            
            # Send in chunks
            chunk_size = 4096
            for i in range(0, len(display_data), chunk_size):
                chunk = display_data[i:i+chunk_size]
                self.spi.writebytes(chunk)
                
        except Exception as e:
            print(f"Error showing image: {e}")
    
    def module_exit(self):
        """Clean up display resources"""
        self._cleanup()
    
    def _cleanup(self):
        """Clean up resources"""
        try:
            if self.gpio_request:
                # Turn off backlight and reset display
                self.gpio_request.set_value(13, gpiod.line.Value.INACTIVE)  # BL off
                self.gpio_request.set_value(12, gpiod.line.Value.INACTIVE)  # RST low
                self.gpio_request.set_value(26, gpiod.line.Value.INACTIVE)  # DC low
                
                # Release GPIO
                self.gpio_request.release()
                self.gpio_request = None
            
            if self.spi:
                self.spi.close()
                self.spi = None
                
        except Exception as e:
            print(f"Error during cleanup: {e}")

def create_working_display():
    """Factory function to create a working display instance"""
    try:
        print("🔧 Creating display with working gpiod API...")
        disp = WorkingLCD()
        
        if disp.Init():
            disp.clear()
            disp.bl_DutyCycle(50)
            print("✅ Working display created successfully")
            return disp
        else:
            print("❌ Display initialization failed")
            return None
            
    except Exception as e:
        print(f"❌ Error creating working display: {e}")
        return None

# Test function
def test_working_display():
    """Test the working display"""
    disp = create_working_display()
    
    if disp:
        print("🧪 Testing display functionality...")
        
        # Test colors
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for color in colors:
            print(f"  Testing color {color}...")
            # Create solid color image
            img = Image.new('RGB', (240, 240), color)
            disp.ShowImage(img)
            time.sleep(1)
        
        # Clear to black
        disp.clear()
        
        print("✅ Display test completed successfully")
        disp.module_exit()
        return True
    else:
        print("❌ Display test failed")
        return False

if __name__ == "__main__":
    test_working_display() 