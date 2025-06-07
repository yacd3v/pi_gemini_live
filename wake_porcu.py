import json
import wave
import logging
import os
import asyncio
import sys
import time
from dotenv import load_dotenv

# Suppress ALSA warnings and errors
os.environ['ALSA_PCM_CARD'] = '0'
os.environ['ALSA_PCM_DEVICE'] = '0'

import numpy as np
import pyaudio
import pvporcupine
from vocal_gemini import AudioHandler

# LED control imports
sys.path.append("freenove_examples")
try:
    from led import Led
    LED_AVAILABLE = True
except ImportError:
    LED_AVAILABLE = False
    logging.warning("LED module not found. LED indicators will be disabled.")

# ---------- Todo ----------

# ✅ DONE : activate leds when wake word is detected
# TODO : debug "back to wake word listening" after gemini is done
# TODO : get direction of arrival when wake word is detected

# ---------- constants ----------
ACCESS_KEY = "RJD+PezWj2uCqiurXzcs8wxVBxT6LYcs6ekrSOQTbnFNQzQ5EMtv2A=="
KEYWORD_PATHS = ["salut-karl_fr.ppn"]
MODEL_PATH = "porcupine_params_fr.pv"  # French model file

CHUNK = 512          # 32 ms @16 kHz
FORMAT = pyaudio.paInt16
RATE = 16000
CHANNELS = 6         # full stream from ReSpeaker
BEAM_CH = 0          # beam‑formed channel

REC_SECONDS = 20     # write this many seconds to wav
OUT_WAV = "outputtest.wav"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- LED control ----------
class LEDController:
    """Simple LED controller for wake word detection status"""
    
    def __init__(self):
        self.led = None
        self.led_initialized = False
        
        if LED_AVAILABLE:
            try:
                self.led = Led()
                self.led_initialized = True
                logging.info("LED controller initialized successfully")
                # Turn off all LEDs initially
                self.turn_off()
            except Exception as e:
                logging.warning(f"LED initialization failed: {e}")
                self.led_initialized = False
        else:
            logging.info("LED module not available")
    
    def set_listening_mode(self):
        """Set LEDs to blue to indicate listening for wake words"""
        if not self.led_initialized:
            return
        
        try:
            # Set all LEDs to blue
            self.led.strip.set_all_led_color(0, 0, 255)  # RGB: Blue
            self.led.strip.show()
            logging.debug("LEDs set to blue (listening mode)")
        except Exception as e:
            logging.warning(f"Error setting LEDs to listening mode: {e}")
    
    def set_wake_detected_mode(self):
        """Set LEDs to green to indicate wake word detected"""
        if not self.led_initialized:
            return
        
        try:
            # Set all LEDs to green
            self.led.strip.set_all_led_color(0, 255, 0)  # RGB: Green
            self.led.strip.show()
            logging.debug("LEDs set to green (wake word detected)")
        except Exception as e:
            logging.warning(f"Error setting LEDs to wake detected mode: {e}")
    
    def turn_off(self):
        """Turn off all LEDs"""
        if not self.led_initialized:
            return
        
        try:
            self.led.strip.set_all_led_color(0, 0, 0)  # RGB: Off
            self.led.strip.show()
            logging.debug("LEDs turned off")
        except Exception as e:
            logging.warning(f"Error turning off LEDs: {e}")
    
    def cleanup(self):
        """Clean up LED resources"""
        self.turn_off()
        if self.led:
            try:
                # Add any specific cleanup if needed
                pass
            except Exception as e:
                logging.warning(f"Error during LED cleanup: {e}")

# ---------- helpers ----------
def reset_respeaker_usb():
    """Reset the ReSpeaker USB device to restore input capabilities"""
    try:
        import subprocess
        
        # First, try to unbind the device from the driver
        try:
            # Find the USB device path
            result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=10)
            if "2886:0018" in result.stdout:
                logging.info("Found ReSpeaker device, proceeding with reset...")
            else:
                logging.warning("ReSpeaker device not found in lsusb output")
        except Exception as e:
            logging.warning(f"Error checking USB devices: {e}")
        
        # Method 1: usb_modeswitch reset
        result = subprocess.run(['sudo', 'usb_modeswitch', '-v', '2886', '-p', '0018', '-R'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logging.info("USB mode switch reset completed")
        else:
            logging.warning(f"USB mode switch failed with return code {result.returncode}")
        
        time.sleep(2)
        
        # Method 2: Try to reset via USB bus
        try:
            # Find the USB device and reset it
            result = subprocess.run(['lsusb', '-d', '2886:0018'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                bus_info = result.stdout.strip()
                logging.info(f"Device found: {bus_info}")
                
                # Extract bus and device numbers
                parts = bus_info.split()
                if len(parts) >= 4:
                    bus_num = parts[1]
                    dev_num = parts[3].rstrip(':')
                    
                    # Reset the device via the USB bus
                    usb_path = f"/dev/bus/usb/{bus_num}/{dev_num}"
                    try:
                        result = subprocess.run(['sudo', 'usbreset', usb_path], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            logging.info("USB bus reset completed")
                        else:
                            logging.info(f"USB bus reset not available or failed: {result.stderr}")
                    except FileNotFoundError:
                        logging.info("usbreset utility not available, skipping")
                    except Exception as e:
                        logging.warning(f"USB bus reset failed: {e}")
        except Exception as e:
            logging.warning(f"USB bus reset attempt failed: {e}")
            
        # Give more time for device to re-enumerate
        logging.info("Waiting for device to re-enumerate...")
        time.sleep(3)
        
        # Verify the device is back and has input channels
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                try:
                    info = p.get_device_info_by_index(i)
                    if "ReSpeaker" in info["name"]:
                        input_channels = info.get("maxInputChannels", 0)
                        logging.info(f"ReSpeaker device found: {info['name']} with {input_channels} input channels")
                        p.terminate()
                        return input_channels > 0
                except Exception as e:
                    continue
            p.terminate()
            logging.warning("ReSpeaker device not found after reset")
            return False
        except Exception as e:
            logging.warning(f"Error verifying device after reset: {e}")
            return False
            
    except Exception as e:
        logging.warning(f"Failed to reset ReSpeaker USB device: {e}")
        return False

def cleanup_audio_resources(p, stream, wf, porcu, led_controller=None):
    """Safely cleanup all audio resources"""
    try:
        if stream and stream.is_active():
            stream.stop_stream()
        if stream:
            stream.close()
    except Exception as e:
        logging.warning(f"Error closing stream: {e}")
    
    try:
        if p:
            p.terminate()
    except Exception as e:
        logging.warning(f"Error terminating PyAudio: {e}")
    
    try:
        if wf:
            wf.close()
    except Exception as e:
        logging.warning(f"Error closing wave file: {e}")
    
    try:
        if porcu:
            porcu.delete()
    except Exception as e:
        logging.warning(f"Error deleting Porcupine: {e}")
    
    try:
        if led_controller:
            led_controller.cleanup()
    except Exception as e:
        logging.warning(f"Error cleaning up LEDs: {e}")

def open_pyaudio():
    """Open PyAudio with ReSpeaker device, with debugging info"""
    p = pyaudio.PyAudio()
    
    # Debug: List all available audio devices
    logging.info("Available audio devices:")
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            logging.info(f"  Device {i}: {info['name']} (in: {info['maxInputChannels']}, out: {info['maxOutputChannels']})")
        except Exception as e:
            logging.warning(f"  Device {i}: Error getting info - {e}")
    
    # Find ReSpeaker device
    dev_index = None
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            if "ReSpeaker" in info["name"]:
                dev_index = i
                logging.info(f"Found ReSpeaker at index {i}: {info['name']}")
                break
        except Exception as e:
            logging.warning(f"Error checking device {i}: {e}")
            continue
            
    if dev_index is None:
        p.terminate()
        raise RuntimeError("ReSpeaker device not found")

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=dev_index,
                        frames_per_buffer=CHUNK)
        logging.info(f"Successfully opened ReSpeaker stream at device index {dev_index}")
        return p, stream, dev_index
    except Exception as e:
        logging.error(f"Failed to open ReSpeaker stream: {e}")
        p.terminate()
        raise

def extract_beam(data):
    """take channel 0 from interleaved 16‑bit little‑endian frame buffer"""
    if data is None:
        logging.error("extract_beam received None data")
        return None
    if len(data) == 0:
        logging.error("extract_beam received empty data")
        return None
    
    try:
        # Add more detailed debugging
        expected_length = CHUNK * CHANNELS * 2  # 2 bytes per sample
        if len(data) != expected_length:
            logging.warning(f"Unexpected data length: got {len(data)}, expected {expected_length}")
        
        result = (np
                .frombuffer(data, dtype="<i2")
                .reshape(-1, CHANNELS)[:, BEAM_CH]
                .astype(np.int16))
        
        # Check the result for validity
        if result is None:
            logging.error("extract_beam result is None")
            return None
        if len(result) == 0:
            logging.error("extract_beam result is empty")
            return None
        if not isinstance(result, np.ndarray):
            logging.error(f"extract_beam result is not ndarray: {type(result)}")
            return None
        if result.dtype != np.int16:
            logging.error(f"extract_beam result has wrong dtype: {result.dtype}")
            return None
        
        # Check for expected array size
        expected_size = CHUNK
        if len(result) != expected_size:
            logging.error(f"extract_beam result has wrong size: got {len(result)}, expected {expected_size}")
            return None
            
        return result
        
    except Exception as e:
        logging.error(f"Error in extract_beam: {e}")
        logging.error(f"Data type: {type(data)}, Data length: {len(data) if data else 'None'}")
        return None

async def run_gemini():
    """Run the Gemini voice assistant"""
    handler = AudioHandler()
    try:
        await handler.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Error in Gemini: {e}")
    finally:
        handler._cleanup()
        # Add extra delay to ensure complete cleanup
        time.sleep(2)

# ---------- main ----------
def main():
    # sanity‑check keyword files
    for path in KEYWORD_PATHS:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    # Initialize LED controller once
    led_controller = LEDController()
    
    # Outer loop for restarting after Gemini sessions
    while True:
        porcu = None
        p = None
        stream = None
        wf = None
        resources_cleaned = False  # Flag to prevent double cleanup

        try:
            logging.info("=== INITIALIZING: Starting new wake word detection cycle...")
            
            # Set LEDs to listening mode (blue)
            led_controller.set_listening_mode()
            
            porcu = pvporcupine.create(
                access_key=ACCESS_KEY,
                keyword_paths=KEYWORD_PATHS,
                model_path=MODEL_PATH)

            p, stream, dev_index = open_pyaudio()

            # wav debug file
            wf = wave.open(OUT_WAV, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            frames_left = int(RATE / CHUNK * REC_SECONDS)

            logging.info("Listening for wake words…")

            consecutive_errors = 0
            max_consecutive_errors = 5  # Only reinitialize after 5 consecutive errors
            
            # Main wake word detection loop
            while True:
                try:
                    raw = stream.read(CHUNK, exception_on_overflow=False)
                    if raw is None:
                        logging.warning("Received None from stream.read(), retrying...")
                        continue
                    
                    # Check if raw data is valid before processing
                    if len(raw) != CHUNK * CHANNELS * 2:  # 2 bytes per sample
                        logging.debug(f"Invalid raw data length: {len(raw)}, expected {CHUNK * CHANNELS * 2}, skipping frame")
                        continue
                    
                    beam = extract_beam(raw)
                    if beam is None:
                        logging.warning("extract_beam returned None, retrying...")
                        continue

                    if frames_left == REC_SECONDS * RATE // CHUNK:     # first iteration only
                        pass
                    
                    # write first 20 s
                    if frames_left > 0:
                        wf.writeframes(beam.tobytes())
                        frames_left -= 1

                    # Porcupine expects a Python list / bytes array of ints
                    try:
                        # Add detailed debugging before Porcupine processing
                        if beam is None:
                            logging.error("Beam is None before Porcupine processing")
                            continue
                        if not isinstance(beam, np.ndarray):
                            logging.error(f"Beam is not ndarray: {type(beam)}")
                            continue
                        if beam.dtype != np.int16:
                            logging.error(f"Beam has wrong dtype: {beam.dtype}")
                            continue
                        if len(beam) != CHUNK:
                            logging.error(f"Beam has wrong length: {len(beam)}, expected {CHUNK}")
                            continue
                        
                        # Check for any problematic values
                        if np.any(beam == 0) and np.all(beam == 0):
                            logging.warning("Beam contains all zeros")
                        
                        idx = porcu.process(beam)
                    except Exception as porcu_error:
                        logging.error(f"Porcupine processing error: {porcu_error}")
                        logging.error(f"Beam type: {type(beam)}, Beam shape: {beam.shape if hasattr(beam, 'shape') else 'no shape'}")
                        if hasattr(beam, 'dtype'):
                            logging.error(f"Beam dtype: {beam.dtype}")
                        if hasattr(beam, '__len__'):
                            logging.error(f"Beam length: {len(beam)}")
                        continue
                        
                    # Reset consecutive error counter on successful processing
                    consecutive_errors = 0
                        
                    if idx >= 0:
                        logging.info("=== WAKE WORD: Wake word detected! Starting Gemini...")
                        
                        # Set LEDs to wake detected mode (green)
                        led_controller.set_wake_detected_mode()
                        time.sleep(0.5)  # Brief visual feedback
                        
                        # Turn off LEDs during Gemini session
                        led_controller.turn_off()
                        
                        # Cleanup current resources properly
                        cleanup_audio_resources(p, stream, wf, porcu, led_controller)
                        p = None
                        stream = None
                        wf = None
                        porcu = None
                        resources_cleaned = True  # Mark as cleaned
                        
                        # Add delay to ensure cleanup
                        time.sleep(1)
                        
                        # Run Gemini with error handling
                        try:
                            logging.info("=== GEMINI: About to start Gemini session...")
                            asyncio.run(run_gemini())
                            logging.info("=== GEMINI: Gemini session completed successfully")
                        except Exception as gemini_error:
                            logging.error(f"=== GEMINI ERROR: Error in Gemini session: {gemini_error}")
                            import traceback
                            traceback.print_exc()
                        
                        # Break from the inner loop to restart properly
                        logging.info("=== BREAK: Breaking from wake word loop to restart...")
                        break

                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    consecutive_errors += 1
                    
                    # Only reinitialize if we've had multiple consecutive errors
                    if consecutive_errors < max_consecutive_errors:
                        logging.info(f"Error {consecutive_errors}/{max_consecutive_errors}, continuing...")
                        time.sleep(0.1)  # Brief pause before retrying
                        continue
                    
                    logging.error(f"Too many consecutive errors ({consecutive_errors}), reinitializing...")
                    consecutive_errors = 0  # Reset counter
                    
                    # Try to recover by reinitializing everything
                    cleanup_audio_resources(p, stream, wf, porcu, led_controller)
                    time.sleep(2)
                    
                    # Reset USB device if it's a device-related error
                    if "Invalid number of channels" in str(e) or "device" in str(e).lower():
                        logging.info("Device error detected, resetting ReSpeaker USB...")
                        reset_respeaker_usb()
                    
                    # Handle GPIO errors (display pins already in use)
                    if "GPIO" in str(e) and "already in use" in str(e):
                        logging.info("GPIO error detected, cleaning up GPIO resources...")
                        
                        # Try gpiozero cleanup first
                        try:
                            import gpiozero
                            # Force close all gpiozero devices
                            try:
                                gpiozero.Device.pin_factory.close()
                                logging.info("gpiozero pin factory closed successfully")
                            except AttributeError:
                                # Fallback for older gpiozero versions
                                try:
                                    gpiozero.Device.pin_factory.reset()
                                    logging.info("gpiozero devices reset successfully")
                                except AttributeError:
                                    logging.info("gpiozero cleanup method not available")
                            time.sleep(1)
                        except ImportError:
                            logging.info("gpiozero not available for cleanup")
                        except Exception as gz_e:
                            logging.warning(f"gpiozero cleanup failed: {gz_e}")
                        
                        # Then try RPi.GPIO cleanup
                        try:
                            import RPi.GPIO as GPIO
                            GPIO.cleanup()
                            time.sleep(1)
                        except Exception as gpio_e:
                            logging.warning(f"GPIO cleanup failed: {gpio_e}")
                    
                    # Reinitialize
                    porcu = pvporcupine.create(
                        access_key=ACCESS_KEY,
                        keyword_paths=KEYWORD_PATHS,
                        model_path=MODEL_PATH)
                    p, stream, dev_index = open_pyaudio()
                    wf = wave.open(OUT_WAV, "wb")
                    wf.setnchannels(1)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    frames_left = int(RATE / CHUNK * REC_SECONDS)
                    
                    # Add warm-up period after error recovery
                    logging.info("Warming up audio stream after error recovery...")
                    for _ in range(10):
                        try:
                            _ = stream.read(CHUNK, exception_on_overflow=False)
                            time.sleep(0.01)
                        except Exception:
                            pass  # Ignore warmup errors

        except KeyboardInterrupt:
            logging.info("=== INTERRUPT: Stopping…")
            break  # Exit the outer loop
        except Exception as outer_error:
            logging.error(f"=== OUTER ERROR: Unexpected error in outer loop: {outer_error}")
            import traceback
            traceback.print_exc()
        finally:
            # Only cleanup if resources haven't been cleaned already
            if not resources_cleaned:
                logging.info("=== CLEANUP: Cleaning up resources in finally block...")
                cleanup_audio_resources(p, stream, wf, porcu, led_controller)
            else:
                logging.info("=== CLEANUP: Resources already cleaned, skipping finally cleanup...")

        # If we reach here, Gemini session has ended, restart wake word detection
        logging.info("=== RESTART: Gemini session ended. Restarting wake word detection in 3 seconds...")
        time.sleep(3)
        
        # Reset ReSpeaker USB device to restore input capabilities
        logging.info("=== RESTART: Resetting ReSpeaker USB device...")
        reset_success = reset_respeaker_usb()
        
        if not reset_success:
            logging.warning("=== RESTART: First reset attempt failed, trying again...")
            time.sleep(2)
            reset_success = reset_respeaker_usb()
            
            if not reset_success:
                logging.error("=== RESTART: USB device reset failed after 2 attempts")
        
        # Add longer delay to ensure device is fully ready
        logging.info("=== RESTART: Waiting for device to fully initialize...")
        time.sleep(5)
        
        logging.info("=== RESTART: Continuing to next iteration of outer loop...")
        # Continue to next iteration of outer loop to restart

if __name__ == "__main__":
    main()
