"""
Display Manager Module for VocalGem robot
Handles display initialization, GPIO management, and display operations
"""

import os
import sys
import time
import traceback
import warnings
import logging
import spidev as SPI
from PIL import Image, ImageDraw, ImageFont

# Add display imports
sys.path.append("display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/..")
from lib import LCD_1inch28

# Configure logging to suppress inline_data warnings
logging.basicConfig(level=logging.INFO)
class InlineDataFilter(logging.Filter):
    def filter(self, record):
        return "inline_data" not in record.getMessage()

# Apply filter to common loggers that might emit inline_data warnings
for logger_name in ["google.genai", "google.ai", "google", "", "__main__"]:
    logger = logging.getLogger(logger_name)
    logger.addFilter(InlineDataFilter())

class DisplayManager:
    """Manages display initialization and GPIO operations"""
    
    def __init__(self):
        self.disp = None
        self.font = None
        
    def initialize_display(self):
        """Initialize the display with proper error handling and fallbacks"""
        try:
            print("Starting display initialization...")
            
            # First, ensure any existing GPIO resources are properly cleaned up
            self._cleanup_gpio_resources()
            
            # Try to initialize display with gpiozero first
            try:
                # Initialize SPI first
                spi1 = SPI.SpiDev()
                spi1.open(1, 0)               # bus 1, CE0  → /dev/spidev1.0
                spi1.max_speed_hz = 40_000_000
                
                # Now initialize the display
                self.disp = LCD_1inch28.LCD_1inch28(
                        spi = spi1,    # use SPI-1
                        rst = 12,      # GPIO12  (pin 32)
                        dc  = 26,      # GPIO26  (pin 37)
                        bl  = 13)      # GPIO13  (pin 33, PWM back-light)
                self.disp.Init()
                self.disp.clear()
                self.disp.bl_DutyCycle(50)
                self.font = ImageFont.truetype("display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/../Font/Font01.ttf", 24)
                print("Display initialized successfully with gpiozero")
                
            except Exception as gpiozero_error:
                print(f"gpiozero display initialization failed: {gpiozero_error}")
                
                # If the error is about pins already in use, try RPi.GPIO fallback
                if "already in use" in str(gpiozero_error):
                    print("Attempting RPi.GPIO fallback for display initialization...")
                    try:
                        self._init_display_with_rpi_gpio()
                    except Exception as rpi_error:
                        print(f"RPi.GPIO fallback also failed: {rpi_error}")
                        print("Display initialization failed, continuing without display...")
                        self.disp = None
                        self.font = None
                else:
                    print("Display initialization failed, continuing without display...")
                    # Set display to None so other code can handle the missing display gracefully
                    self.disp = None
                    self.font = None
            
        except Exception as display_init_error:
            print(f"Error initializing display: {display_init_error}")
            import traceback
            traceback.print_exc()
            
            # Set display to None so other code can handle the missing display gracefully
            self.disp = None
            self.font = None
            print("Display initialization failed, continuing without display")

    def _cleanup_gpio_resources(self):
        """Properly cleanup GPIO resources before display initialization"""
        try:
            print("Cleaning up GPIO resources...")
            
            # IMPORTANT: Don't clean up LED pins (GPIO 18) as the LED system needs them
            # Only clean up display-specific pins that might be in conflict
            
            # Method 1: Clean up with RPi.GPIO first (more reliable)
            try:
                import RPi.GPIO as GPIO
                
                # Set mode first
                GPIO.setmode(GPIO.BCM)
                
                # Clean up ONLY display pins, avoid LED pin (18)
                display_pins = [12, 13, 26]  # rst, bl, dc pins (NOT pin 18 which is for LEDs)
                for pin in display_pins:
                    try:
                        # Try to read current state first to see if pin is in use
                        try:
                            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_OFF)
                        except Exception:
                            pass  # Pin might already be set up
                        
                        # Clean up the specific pin
                        GPIO.cleanup(pin)
                        print(f"Cleaned up GPIO pin {pin}")
                    except Exception as pin_e:
                        print(f"Error cleaning up GPIO pin {pin}: {pin_e}")
                
                # DON'T call GPIO.cleanup() as it would affect the LED system
                print("RPi.GPIO display pins cleanup completed")
                
            except Exception as gpio_e:
                print(f"RPi.GPIO cleanup error: {gpio_e}")
            
            # Method 2: DON'T recreate the gpiozero pin factory if LEDs are working
            # The LED system might be using it, so we'll work around the issue instead
            try:
                import gpiozero
                
                # Check if pin factory exists and is working
                if gpiozero.Device.pin_factory is not None:
                    try:
                        # Test if the factory is working by checking its handle
                        if hasattr(gpiozero.Device.pin_factory, '_handle'):
                            handle = gpiozero.Device.pin_factory._handle
                            if handle is not None:
                                print("gpiozero pin factory is working, keeping it")
                                return  # Don't mess with a working factory
                    except Exception:
                        pass  # Factory might be broken, continue with reset
                
                # Only reset if the factory is broken or missing
                print("gpiozero pin factory needs reset...")
                
                # If there's an existing pin factory, close it properly
                if gpiozero.Device.pin_factory is not None:
                    try:
                        # Close the existing factory
                        gpiozero.Device.pin_factory.close()
                        print("Closed existing gpiozero pin factory")
                    except Exception as e:
                        print(f"Error closing existing pin factory: {e}")
                    
                    # Set to None to force recreation
                    gpiozero.Device.pin_factory = None
                    print("Reset gpiozero pin factory to None")
                
                # Give time for cleanup
                time.sleep(0.5)
                
                # Create a fresh pin factory with explicit initialization
                from gpiozero.pins.lgpio import LGPIOFactory
                factory = LGPIOFactory()
                
                # Verify the factory is properly initialized
                if hasattr(factory, '_handle') and factory._handle is not None:
                    gpiozero.Device.pin_factory = factory
                    print("Created and verified fresh gpiozero pin factory")
                else:
                    print("Warning: Pin factory handle not properly initialized, retrying...")
                    # Try again with a delay
                    time.sleep(0.5)
                    factory = LGPIOFactory()
                    gpiozero.Device.pin_factory = factory
                    print("Created gpiozero pin factory on retry")
                
            except ImportError:
                print("gpiozero not available for cleanup")
            except Exception as e:
                print(f"gpiozero cleanup error: {e}")
                # If gpiozero fails, we can still proceed
            
            # Give extra time for all resources to be released
            time.sleep(0.5)
            print("GPIO cleanup completed")
            
        except Exception as e:
            print(f"Error during GPIO cleanup: {e}")

    def _cleanup_gpio_for_shutdown(self):
        """Cleanup GPIO resources during shutdown to prepare for next initialization"""
        try:
            print("Starting GPIO shutdown cleanup...")
            
            # DON'T clean up gpiozero pin factory during shutdown if LEDs are still active
            # The LED system in DisplayAnimator might still be running
            # We'll let the LED system handle its own cleanup
            
            # Clean up with RPi.GPIO for display pins only
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                # Clean up ONLY display pins, not LED pins
                for pin in [12, 13, 26]:  # rst, bl, dc pins (NOT pin 18 which is for LEDs)
                    try:
                        GPIO.setup(pin, GPIO.IN)  # Set to input to release
                        GPIO.cleanup(pin)
                    except:
                        pass
                # DON'T call GPIO.cleanup() as it would affect the LED system
                print("RPi.GPIO display pins cleanup completed")
            except Exception as gpio_e:
                print(f"RPi.GPIO cleanup error: {gpio_e}")
            
            # Give time for cleanup
            time.sleep(0.5)
            print("GPIO shutdown cleanup completed")
            
        except Exception as e:
            print(f"Error during GPIO shutdown cleanup: {e}")

    def _init_display_with_rpi_gpio(self):
        """Initialize display using RPi.GPIO directly as fallback"""
        try:
            import RPi.GPIO as GPIO
            
            # Set up GPIO mode
            GPIO.setmode(GPIO.BCM)
            
            # Define pins
            RST_PIN = 12
            DC_PIN = 26
            BL_PIN = 13
            
            # Force cleanup of these pins first (in case gpiozero is holding them)
            for pin in [RST_PIN, DC_PIN, BL_PIN]:
                try:
                    GPIO.cleanup(pin)
                except:
                    pass  # Ignore cleanup errors
            
            # Set up pins with force override
            try:
                GPIO.setup(RST_PIN, GPIO.OUT)
                GPIO.setup(DC_PIN, GPIO.OUT)
                GPIO.setup(BL_PIN, GPIO.OUT)
            except Exception as setup_error:
                print(f"GPIO setup error: {setup_error}")
                # Try to force setup by cleaning up all GPIO first
                GPIO.cleanup()
                time.sleep(0.5)
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(RST_PIN, GPIO.OUT)
                GPIO.setup(DC_PIN, GPIO.OUT)
                GPIO.setup(BL_PIN, GPIO.OUT)
            
            # Initialize SPI
            spi1 = SPI.SpiDev()
            spi1.open(1, 0)
            spi1.max_speed_hz = 40_000_000
            
            # Create a simple display wrapper that uses RPi.GPIO
            class RPiGPIODisplay:
                def __init__(self, spi, rst, dc, bl):
                    self.spi = spi
                    self.rst_pin = rst
                    self.dc_pin = dc
                    self.bl_pin = bl
                    self.width = 240
                    self.height = 240
                    
                def Init(self):
                    # Reset sequence
                    GPIO.output(self.rst_pin, GPIO.HIGH)
                    time.sleep(0.01)
                    GPIO.output(self.rst_pin, GPIO.LOW)
                    time.sleep(0.01)
                    GPIO.output(self.rst_pin, GPIO.HIGH)
                    time.sleep(0.01)
                    
                    # Basic LCD initialization commands
                    self._write_command(0x36)  # Memory Access Control
                    self._write_data(0x70)     # RGB order
                    
                    self._write_command(0x3A)  # Pixel Format
                    self._write_data(0x05)     # 16-bit color
                    
                    self._write_command(0x21)  # Display Inversion On
                    self._write_command(0x13)  # Normal Display Mode On
                    self._write_command(0x29)  # Display On
                    
                def _write_command(self, cmd):
                    GPIO.output(self.dc_pin, GPIO.LOW)
                    self.spi.writebytes([cmd])
                    
                def _write_data(self, data):
                    GPIO.output(self.dc_pin, GPIO.HIGH)
                    if isinstance(data, int):
                        self.spi.writebytes([data])
                    else:
                        self.spi.writebytes(data)
                
                def clear(self):
                    # Clear screen to black
                    self._write_command(0x2A)  # Column Address Set
                    self._write_data([0x00, 0x00, 0x00, 0xEF])  # 0-239
                    
                    self._write_command(0x2B)  # Row Address Set
                    self._write_data([0x00, 0x00, 0x00, 0xEF])  # 0-239
                    
                    self._write_command(0x2C)  # Memory Write
                    # Send black pixels (0x0000 for each pixel)
                    black_data = [0x00] * (240 * 240 * 2)
                    self._write_data(black_data)
                
                def bl_DutyCycle(self, duty):
                    # Simple PWM simulation - just turn on/off for now
                    if duty > 50:
                        GPIO.output(self.bl_pin, GPIO.HIGH)
                    else:
                        GPIO.output(self.bl_pin, GPIO.LOW)
                
                def ShowImage(self, image):
                    # Convert PIL image to display format and show
                    if image.size != (self.width, self.height):
                        image = image.resize((self.width, self.height))
                    
                    # Convert to RGB565 format
                    rgb_image = image.convert('RGB')
                    pixels = list(rgb_image.getdata())
                    
                    # Set display window
                    self._write_command(0x2A)  # Column Address Set
                    self._write_data([0x00, 0x00, 0x00, 0xEF])
                    
                    self._write_command(0x2B)  # Row Address Set
                    self._write_data([0x00, 0x00, 0x00, 0xEF])
                    
                    self._write_command(0x2C)  # Memory Write
                    
                    # Convert RGB888 to RGB565 and send
                    display_data = []
                    for r, g, b in pixels:
                        # Convert to RGB565
                        r565 = (r >> 3) << 11
                        g565 = (g >> 2) << 5
                        b565 = b >> 3
                        rgb565 = r565 | g565 | b565
                        
                        # Split into bytes (big endian)
                        display_data.append((rgb565 >> 8) & 0xFF)
                        display_data.append(rgb565 & 0xFF)
                    
                    self._write_data(display_data)
                
                def module_exit(self):
                    GPIO.output(self.rst_pin, GPIO.LOW)
                    GPIO.output(self.dc_pin, GPIO.LOW)
                    GPIO.output(self.bl_pin, GPIO.LOW)
            
            # Create the display instance
            self.disp = RPiGPIODisplay(spi1, RST_PIN, DC_PIN, BL_PIN)
            self.disp.Init()
            self.disp.clear()
            self.disp.bl_DutyCycle(50)
            
            # Load font
            self.font = ImageFont.truetype("display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/../Font/Font01.ttf", 24)
            
            print("Display initialized successfully with RPi.GPIO fallback")
            
        except Exception as e:
            print(f"RPi.GPIO display initialization also failed: {e}")
            self.disp = None
            self.font = None

    def show_status(self, text):
        """Show status text on display"""
        # Only show status if display is available
        if self.disp is None or self.font is None:
            return
            
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

    def cleanup(self):
        """Clean up display resources"""
        try:
            if self.disp is not None:
                print("Starting display cleanup...")
                
                # First try the display's own cleanup methods
                try:
                    self.disp.clear()
                    self.disp.bl_DutyCycle(0)
                    self.disp.module_exit()
                    print("Display module cleanup completed")
                except Exception as disp_e:
                    print(f"Display cleanup methods failed: {disp_e}")
                
                # Remove the display reference to prevent reuse
                self.disp = None
                self.font = None
                print("Display references cleared")
            else:
                print("No display to cleanup or display already None")
            
            # Always perform GPIO cleanup to ensure clean state for next initialization
            self._cleanup_gpio_for_shutdown()
                    
        except Exception as e:
            print(f"Error cleaning up display: {e}")
            
        print("Display cleanup completed.") 