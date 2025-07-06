# ============================================================================
# DISPLAY INITIALIZATION FIX FOR VOCAL_GEMINI.PY
# ============================================================================
# Replace the display initialization section (around lines 130-250) with this:

# Add this import at the top with other imports
from display_gpiod_integration import create_working_display

# Replace the entire display initialization section in __init__ with this:
try:
    print("Starting display initialization...")
    
    # Use the working gpiod display instead of the complex fallback system
    self.disp = create_working_display()
    
    if self.disp:
        # Load font
        try:
            self.font = ImageFont.truetype("display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/../Font/Font01.ttf", 24)
            print("✓ Font loaded successfully")
        except Exception as font_error:
            print(f"Font loading failed: {font_error}")
            self.font = ImageFont.load_default()
            print("Using default font")
        
        print("✅ Display initialized successfully with working gpiod API")
    else:
        print("❌ Display initialization failed")
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

# ============================================================================
# CLEANUP MODIFICATION
# ============================================================================
# Also update the _cleanup method to use the new display cleanup:

def _cleanup(self):
    print("Cleaning up…")
    
    # Stop animation and turn off LEDs
    if hasattr(self, 'anim'):
        try:
            if hasattr(self.anim, 'led_turn_off'):
                self.anim.led_turn_off()  # Turn off LEDs first
            self.sleep_requested_event.set()  # Signal animator to stop
            time.sleep(0.1)  # Brief pause for animator to stop
        except Exception as e:
            print(f"Error stopping animator and LEDs: {e}")
    
    # Cleanup audio streams with proper error handling
    for s in (self.input_stream, self.output_stream):
        try:
            if s and s.is_active():
                s.stop_stream()
                time.sleep(0.1)  # Small delay between stop and close
            if s:
                s.close()
        except Exception as e:
            print(f"Error cleaning up stream: {e}")
    
    # Cleanup wave files
    for wf in (self.sent_wf, self.recv_wf):
        try:
            if wf:
                wf.close()
        except Exception as e:
            print(f"Error closing wave file: {e}")
    
    # Terminate PyAudio with delay
    try:
        if self.pya:
            self.pya.terminate()
            time.sleep(0.5)  # Longer delay after PyAudio termination
    except Exception as e:
        print(f"Error terminating PyAudio: {e}")
        
    # NEW: Cleanup display using working method
    try:
        if hasattr(self, 'disp') and self.disp is not None:
            print("Cleaning up display...")
            self.disp.module_exit()  # Uses the working cleanup method
            self.disp = None
            self.font = None
            print("✓ Display cleanup completed")
    except Exception as e:
        print(f"Error cleaning up display: {e}")
        
    print("Cleanup completed.")
    time.sleep(1)

# ============================================================================
# REMOVE THESE METHODS (they're no longer needed):
# ============================================================================
# - _cleanup_gpio_resources()
# - _cleanup_gpio_for_shutdown() 
# - _init_display_with_existing_gpiozero()
# - _init_display_with_rpi_gpio()

# ============================================================================
# SUMMARY OF CHANGES:
# ============================================================================
# 1. Add import: from display_gpiod_integration import create_working_display
# 2. Replace complex display init with simple create_working_display() call
# 3. Update cleanup to use disp.module_exit()
# 4. Remove old GPIO cleanup methods
# ============================================================================ 