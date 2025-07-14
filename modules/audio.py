#!/usr/bin/env python3
"""
Audio module for robot control dashboard
Handles ReSpeaker v2 audio streaming and DOA (Direction of Arrival) detection
"""

import time
import threading
import struct
import numpy as np
import sys
import os
import subprocess

# Audio processing imports
try:
    # Temporarily suppress ALSA/JACK warnings during PyAudio import
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    import pyaudio
    sys.stderr.close()
    sys.stderr = original_stderr
    PYAUDIO_AVAILABLE = True
except ImportError as e:
    print(f"PyAudio not available: {e}")
    PYAUDIO_AVAILABLE = False

# ReSpeaker DOA imports
try:
    import usb.core
    import usb.util
    USB_AVAILABLE = True
except ImportError as e:
    print(f"USB libraries not available: {e}")
    USB_AVAILABLE = False

class ReSpeakerManager:
    """Manages ReSpeaker v2 audio operations including streaming and DOA detection"""
    
    def __init__(self, config):
        self.config = config
        self.is_running = False
        self.pya = None
        self.input_stream = None
        self.output_stream = None
        
        # Audio data storage
        self.audio_data = {
            'raw_audio': None,
            'processed_audio': None,
            'doa_angle': 0,
            'voice_activity': False,
            'volume_level': 0.0,
            'timestamp': time.time()
        }
        self.lock = threading.Lock()
        
        # ReSpeaker USB device
        self.usb_device = None
        self.simulation_mode = False  # For demo when ReSpeaker not available
        
        # Audio format constants - Updated for ReSpeaker 4 Mic Array
        self.FORMAT = pyaudio.paInt16
        self.RAW_CH = 6  # ReSpeaker 4 Mic Array has 6 channels (FL, FR, FC, LFE, RL, RR)
        self.IN_CH = 1   # Processed mono channel
        self.OUT_CH = 1  # Mono output
        self.CHUNK_MS = 20
        self.SAMPLE_RATE = 16000  # ReSpeaker 4 Mic Array supports 16kHz
        
        # Known ReSpeaker device configuration
        self.RESPEAKER_DEVICE_NAME = "hw:2,0"  # ALSA device name for ReSpeaker
        self.RESPEAKER_VENDOR_ID = 0x2886
        self.RESPEAKER_PRODUCT_ID = 0x0018
        
        # PipeWire management state
        self.pipewire_was_running = False
        self.audio_servers_masked = False
        
    def _manage_audio_servers(self, action='stop'):
        """Manage PipeWire/WirePlumber for exclusive audio access"""
        if not self.config.get('auto_manage_pipewire', True):
            return True
        
        try:
            if action == 'stop':
                print("🔇 Stopping audio servers for exclusive ReSpeaker access...")
                
                # Check if PipeWire is running
                result = subprocess.run(['pgrep', 'pipewire'], capture_output=True)
                self.pipewire_was_running = result.returncode == 0
                
                # Mask services to prevent auto-restart
                subprocess.run(['systemctl', '--user', 'mask', 'pipewire'], 
                              capture_output=True, check=False)
                subprocess.run(['systemctl', '--user', 'mask', 'wireplumber'], 
                              capture_output=True, check=False)
                self.audio_servers_masked = True
                
                # Kill running processes
                subprocess.run(['pkill', '-9', 'pipewire'], capture_output=True, check=False)
                subprocess.run(['pkill', '-9', 'wireplumber'], capture_output=True, check=False)
                
                # Wait a moment for processes to stop
                time.sleep(1)
                
                print("✓ Audio servers stopped for exclusive access")
                return True
                
            elif action == 'restore':
                if not self.config.get('restore_audio_on_exit', True):
                    return True
                    
                print("🔊 Restoring desktop audio servers...")
                
                # Unmask services
                if self.audio_servers_masked:
                    subprocess.run(['systemctl', '--user', 'unmask', 'pipewire'], 
                                  capture_output=True, check=False)
                    subprocess.run(['systemctl', '--user', 'unmask', 'wireplumber'], 
                                  capture_output=True, check=False)
                    self.audio_servers_masked = False
                
                # Restart services if they were running before
                if self.pipewire_was_running:
                    subprocess.run(['systemctl', '--user', 'start', 'pipewire'], 
                                  capture_output=True, check=False)
                    subprocess.run(['systemctl', '--user', 'start', 'wireplumber'], 
                                  capture_output=True, check=False)
                
                print("✓ Desktop audio servers restored")
                return True
                
        except Exception as e:
            print(f"⚠ Warning: Failed to manage audio servers: {e}")
            return False
        
        return True
        
    def initialize(self):
        """Initialize the ReSpeaker v2 device and audio system"""
        if not PYAUDIO_AVAILABLE:
            print("⚠ PyAudio not available - audio features disabled")
            return False
        
        try:
            print("🎤 Initializing ReSpeaker 4 Mic Array audio system...")
            
            # Manage audio servers for exclusive access
            self._manage_audio_servers('stop')
            
            # Initialize PyAudio
            self.pya = pyaudio.PyAudio()
            
            # Find ReSpeaker device using our known configuration
            device_index = self._find_respeaker_device()
            if device_index is None:
                print("⚠ ReSpeaker device not found - enabling simulation mode for demo")
                self.simulation_mode = True
                # Don't return False, continue with simulation
            
            # Initialize USB device for DOA (but don't fail if it doesn't work)
            if USB_AVAILABLE:
                self._initialize_usb_device()
            
            # Setup audio streams (only if not in simulation mode)
            if not self.simulation_mode:
                if not self._setup_audio_streams(device_index):
                    print("⚠ Failed to setup audio streams - switching to simulation mode")
                    self.simulation_mode = True
            
            print("✓ ReSpeaker 4 Mic Array audio system initialized")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize ReSpeaker: {e}")
            return False
    
    def _find_respeaker_device(self):
        """Find ReSpeaker device index using known configuration"""
        print("🔍 Searching for ReSpeaker 4 Mic Array...")
        
        # First, try to find device by our known ALSA device name
        try:
            # Try to open the device directly to test if it exists
            test_stream = self.pya.open(
                format=self.FORMAT,
                channels=self.RAW_CH,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=None,  # Will use default
                frames_per_buffer=1024
            )
            test_stream.close()
        except Exception as e:
            print(f"⚠ Direct device test failed: {e}")
        
        # List all available devices
        print("📋 Available audio devices:")
        respeaker_candidates = []
        
        for idx in range(self.pya.get_device_count()):
            try:
                info = self.pya.get_device_info_by_index(idx)
                name = info["name"].lower()
                max_input_channels = info.get('maxInputChannels', 0)
                
                print(f"  {idx}: {info['name']} (in: {max_input_channels}, out: {info.get('maxOutputChannels', 0)})")
                
                # Look for ReSpeaker by multiple criteria
                is_respeaker = (
                    "respeaker" in name or 
                    "seeed" in name or 
                    "array" in name or
                    "uac" in name or
                    max_input_channels >= 6  # ReSpeaker has 6+ input channels
                )
                
                if is_respeaker:
                    respeaker_candidates.append((idx, info))
                    print(f"    ✓ Potential ReSpeaker candidate found!")
                
            except Exception as e:
                print(f"  {idx}: Error getting device info: {e}")
        
        if respeaker_candidates:
            # Prefer the actual ReSpeaker device over PulseAudio
            for idx, info in respeaker_candidates:
                name = info['name'].lower()
                # Prioritize devices that are actually the ReSpeaker hardware
                if ('respeaker' in name or 'seeed' in name or 'uac' in name) and 'pulse' not in name:
                    print(f"✓ Found ReSpeaker device: {info['name']} (index {idx})")
                    print(f"  Channels: {info.get('maxInputChannels', 0)} in, {info.get('maxOutputChannels', 0)} out")
                    print(f"  Sample rate: {info.get('defaultSampleRate', 'unknown')} Hz")
                    return idx
            
            # Fallback to device with most input channels
            best_device = max(respeaker_candidates, key=lambda x: x[1].get('maxInputChannels', 0))
            idx, info = best_device
            print(f"⚠ Using fallback device: {info['name']} (index {idx})")
            print(f"  Channels: {info.get('maxInputChannels', 0)} in, {info.get('maxOutputChannels', 0)} out")
            print(f"  Sample rate: {info.get('defaultSampleRate', 'unknown')} Hz")
            return idx
        
        # If no candidates found, try to use the device by ALSA name
        print("⚠ No ReSpeaker found by name, trying to use known ALSA device...")
        try:
            # Try to find a device that supports our requirements
            for idx in range(self.pya.get_device_count()):
                info = self.pya.get_device_info_by_index(idx)
                if info.get('maxInputChannels', 0) >= 6:
                    print(f"✓ Using device with 6+ channels: {info['name']} (index {idx})")
                    return idx
        except Exception as e:
            print(f"Error searching for compatible device: {e}")
        
        # Last resort: try default input device
        try:
            default_input = self.pya.get_default_input_device_info()
            print(f"⚠ No ReSpeaker found, trying default input: {default_input['name']}")
            return default_input['index']
        except Exception as e:
            print(f"Error getting default input device: {e}")
            
        return None
    
    def _initialize_usb_device(self):
        """Initialize USB device for DOA control - but don't fail if it doesn't work"""
        try:
            print("🔌 Attempting to initialize ReSpeaker USB device for DOA...")
            
            self.usb_device = usb.core.find(idVendor=self.RESPEAKER_VENDOR_ID, idProduct=self.RESPEAKER_PRODUCT_ID)
            if self.usb_device is None:
                print("⚠ ReSpeaker USB device not found for DOA (this is normal)")
                return False
            
            print("✓ ReSpeaker USB device found")
            
            # Don't try to detach kernel driver - this can cause conflicts
            # The ALSA driver should handle the audio interface
            
            # Try to set configuration (but don't fail if it doesn't work)
            try:
                self.usb_device.set_configuration()
                print("✓ ReSpeaker USB device initialized for DOA")
                return True
            except usb.core.USBError as e:
                if "Resource busy" in str(e):
                    print("⚠ USB device busy - DOA will use audio-based estimation")
                    # Keep the device reference for potential future use
                    return False
                else:
                    print(f"⚠ USB configuration failed: {e}")
                    return False
            
        except Exception as e:
            print(f"⚠ Failed to initialize USB device: {e}")
            self.usb_device = None
            return False
    
    def _setup_audio_streams(self, device_index):
        """Setup input and output audio streams with correct ReSpeaker configuration"""
        try:
            # Calculate frames per buffer
            frames_per_buffer = int(self.SAMPLE_RATE * self.CHUNK_MS / 1000)
            
            # Get device info for debugging
            device_info = self.pya.get_device_info_by_index(device_index)
            print(f"📊 Using audio device: {device_info['name']}")
            print(f"📊 Device channels: {device_info.get('maxInputChannels', 0)} in, {device_info.get('maxOutputChannels', 0)} out")
            print(f"📊 Device sample rate: {device_info.get('defaultSampleRate', 'unknown')} Hz")
            print(f"📊 Our configuration: {self.RAW_CH} channels, {self.SAMPLE_RATE} Hz")
            
            # Setup input stream (6-channel from ReSpeaker)
            self.input_stream = self.pya.open(
                format=self.FORMAT,
                channels=self.RAW_CH,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=frames_per_buffer,
                stream_callback=self._audio_callback
            )
            
            # Setup output stream (for monitoring if needed)
            try:
                self.output_stream = self.pya.open(
                    format=self.FORMAT,
                    channels=self.OUT_CH,
                    rate=self.SAMPLE_RATE,
                    output=True,
                    output_device_index=device_index,
                    frames_per_buffer=frames_per_buffer
                )
            except Exception as e:
                print(f"⚠ Output stream setup failed (this is normal): {e}")
                self.output_stream = None
            
            print(f"✓ Audio streams setup (sample rate: {self.SAMPLE_RATE}, buffer: {frames_per_buffer})")
            return True
            
        except Exception as e:
            print(f"✗ Failed to setup audio streams: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback to process incoming audio data"""
        if not self.is_running:
            return (None, pyaudio.paComplete)
        
        try:
            # Convert bytes to numpy array
            raw_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Debug: Print audio data info occasionally
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            
            if self._debug_counter % 100 == 0:  # Every 100th callback
                print(f"[Audio Debug] Raw data length: {len(raw_data)}, Expected: {frame_count * self.RAW_CH}")
                if len(raw_data) > 0:
                    print(f"[Audio Debug] Sample values: {raw_data[:10]}")
            
            # Reshape to 6 channels (FL, FR, FC, LFE, RL, RR)
            if len(raw_data) >= self.RAW_CH:
                audio_channels = raw_data.reshape(-1, self.RAW_CH)
                
                # For ReSpeaker 4 Mic Array, we typically use the first 4 channels for DOA
                # Channels 0-3 are the microphone array, channels 4-5 are processed audio
                mic_channels = audio_channels[:, :4]  # First 4 channels are the mics
                processed_mono = audio_channels[:, 4] if audio_channels.shape[1] > 4 else audio_channels[:, 0]
                
                # Calculate volume level from processed channel
                volume = np.sqrt(np.mean(processed_mono.astype(np.float32) ** 2))
                
                # Detect voice activity (simple threshold)
                voice_activity = volume > self.config.get('voice_threshold', 500)
                
                # Get DOA (will try USB first, then fallback to audio estimation)
                doa_angle = self._get_doa_angle()
                
                # Update audio data with thread safety
                with self.lock:
                    self.audio_data = {
                        'raw_audio': raw_data.copy(),
                        'processed_audio': processed_mono.copy(),
                        'mic_channels': mic_channels.copy(),  # Store mic channels for DOA
                        'doa_angle': doa_angle,
                        'voice_activity': voice_activity,
                        'volume_level': float(volume),
                        'timestamp': time.time()
                    }
                
                # Debug: Print volume info occasionally
                if self._debug_counter % 200 == 0:  # Every 200th callback
                    print(f"[Audio Debug] Volume: {volume:.2f}, Voice: {voice_activity}, DOA: {doa_angle}°")
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
        
        return (None, pyaudio.paContinue)
    
    def _get_doa_angle(self):
        """Get Direction of Arrival angle from ReSpeaker"""
        # Try USB control interface first
        if self.usb_device:
            try:
                # Read DOA data from ReSpeaker USB interface
                # This is based on the ReSpeaker v2 USB protocol
                data = self.usb_device.ctrl_transfer(
                    usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                    0,  # request
                    0x0200,  # DOA register
                    0,  # index
                    4  # data length
                )
                
                if len(data) >= 4:
                    # Convert bytes to angle (little endian)
                    angle = struct.unpack('<I', data)[0]
                    # Convert to degrees (0-359)
                    return angle % 360
                
            except Exception as e:
                # USB control failed, try alternative method
                pass
        
        # Alternative: Try to estimate DOA from audio channels
        # This is a simplified approach when USB control is not available
        return self._estimate_doa_from_audio()
    
    def _estimate_doa_from_audio(self):
        """Estimate DOA from audio channel differences when USB control is not available"""
        if not hasattr(self, 'audio_data') or self.audio_data.get('mic_channels') is None:
            return 0
        
        try:
            # Get the 4-microphone channels
            mic_channels = self.audio_data['mic_channels']
            
            if mic_channels is None or mic_channels.size == 0:
                return 0
            
            # Calculate RMS for each microphone channel
            rms_values = np.sqrt(np.mean(mic_channels.astype(np.float32) ** 2, axis=0))
            
            # Simple DOA estimation based on channel differences
            # ReSpeaker 4 Mic Array has 4 mics arranged in a square pattern
            # Channel mapping: 0: front, 1: right, 2: back, 3: left
            if len(rms_values) >= 4:
                # Find the channel with highest energy
                max_channel = np.argmax(rms_values[:4])
                
                # Map channel to approximate angle
                angle_map = {0: 0, 1: 90, 2: 180, 3: 270}
                base_angle = angle_map.get(max_channel, 0)
                
                # Add some variation based on secondary channels
                # Find second highest channel
                sorted_channels = np.argsort(rms_values[:4])[::-1]
                if len(sorted_channels) >= 2:
                    second_channel = sorted_channels[1]
                    # Interpolate between the two strongest channels
                    ratio = rms_values[second_channel] / (rms_values[max_channel] + 1e-10)
                    offset = ratio * 45  # Max 45 degree offset
                    
                    # Determine direction of interpolation based on mic arrangement
                    # Clockwise: front(0) -> right(1) -> back(2) -> left(3) -> front(0)
                    if (max_channel == 0 and second_channel == 1) or \
                       (max_channel == 1 and second_channel == 2) or \
                       (max_channel == 2 and second_channel == 3) or \
                       (max_channel == 3 and second_channel == 0):
                        angle = base_angle + offset
                    else:
                        angle = base_angle - offset
                    
                    return angle % 360
                
                # If no secondary channel, just return base angle
                return base_angle
            
        except Exception as e:
            # Don't spam console with estimation errors
            pass
        
        return 0
    
    def _simulation_thread(self):
        """Simulation thread for demo when ReSpeaker is not available"""
        import math
        
        angle = 0
        while self.is_running:
            try:
                # Simulate a slowly rotating DOA
                angle = (angle + 2) % 360
                
                # Simulate voice activity (on for 3 seconds, off for 2 seconds)
                cycle_time = time.time() % 5
                voice_active = cycle_time < 3
                
                # Simulate volume level
                if voice_active:
                    volume = 1000 + 500 * math.sin(time.time() * 2)  # Varying volume
                else:
                    volume = 100  # Background noise
                
                # Generate fake audio data
                fake_audio = np.random.randint(-1000, 1000, 320, dtype=np.int16)
                
                # Update audio data with thread safety
                with self.lock:
                    self.audio_data = {
                        'raw_audio': fake_audio.copy(),
                        'processed_audio': fake_audio.copy(),
                        'doa_angle': angle,
                        'voice_activity': voice_active,
                        'volume_level': float(volume),
                        'timestamp': time.time()
                    }
                
                time.sleep(0.1)  # 10 Hz update rate
                
            except Exception as e:
                print(f"Error in simulation thread: {e}")
                break
    
    def start_streaming(self):
        """Start audio streaming"""
        if self.simulation_mode:
            # Start simulation thread
            self.is_running = True
            simulation_thread = threading.Thread(target=self._simulation_thread, daemon=True)
            simulation_thread.start()
            print("✓ Audio simulation started (demo mode)")
            return True
        
        if not self.input_stream:
            print("❌ Audio streams not initialized")
            return False
        
        try:
            self.is_running = True
            self.input_stream.start_stream()
            print("✓ Audio streaming started")
            return True
            
        except Exception as e:
            print(f"✗ Failed to start audio streaming: {e}")
            return False
    
    def stop_streaming(self):
        """Stop audio streaming"""
        try:
            self.is_running = False
            if self.input_stream and self.input_stream.is_active():
                self.input_stream.stop_stream()
            print("✓ Audio streaming stopped")
            
        except Exception as e:
            print(f"Error stopping audio streaming: {e}")
    
    def get_audio_data(self):
        """Get current audio data with thread safety"""
        with self.lock:
            return self.audio_data.copy()
    
    def get_doa_visualization_data(self):
        """Get DOA data formatted for web visualization"""
        audio_data = self.get_audio_data()
        
        # Convert angle to radians for visualization
        angle_rad = np.radians(audio_data['doa_angle'])
        
        return {
            'angle': audio_data['doa_angle'],
            'angle_rad': angle_rad,
            'x': np.cos(angle_rad),
            'y': np.sin(angle_rad),
            'voice_activity': audio_data['voice_activity'],
            'volume_level': audio_data['volume_level'],
            'volume_db': 20 * np.log10(max(audio_data['volume_level'], 1e-10)),
            'timestamp': audio_data['timestamp']
        }
    
    def get_audio_spectrum(self, fft_size=512):
        """Get audio spectrum for visualization"""
        audio_data = self.get_audio_data()
        
        if audio_data['processed_audio'] is None:
            return None
        
        try:
            # Apply window function
            windowed = audio_data['processed_audio'] * np.hanning(len(audio_data['processed_audio']))
            
            # Compute FFT
            fft = np.fft.rfft(windowed, n=fft_size)
            magnitude = np.abs(fft)
            
            # Convert to dB
            magnitude_db = 20 * np.log10(magnitude + 1e-10)
            
            # Create frequency bins
            freqs = np.fft.rfftfreq(fft_size, 1.0 / self.SAMPLE_RATE)
            
            return {
                'frequencies': freqs.tolist(),
                'magnitudes': magnitude_db.tolist(),
                'sample_rate': self.SAMPLE_RATE
            }
            
        except Exception as e:
            print(f"Error computing audio spectrum: {e}")
            return None
    
    def get_status(self):
        """Get audio system status"""
        return {
            'available': PYAUDIO_AVAILABLE and USB_AVAILABLE,
            'initialized': self.pya is not None or self.simulation_mode,
            'streaming': self.is_running,
            'usb_device': self.usb_device is not None,
            'simulation_mode': self.simulation_mode,
            'sample_rate': self.SAMPLE_RATE,
            'channels': self.RAW_CH,
            'device_name': self.RESPEAKER_DEVICE_NAME,
            'device_type': 'ReSpeaker 4 Mic Array (UAC1.0)'
        }
    
    def cleanup(self):
        """Cleanup audio resources"""
        print("🧹 Cleaning up audio resources...")
        
        # Stop streaming
        self.stop_streaming()
        
        # Close streams
        for stream in [self.input_stream, self.output_stream]:
            if stream:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                except Exception as e:
                    print(f"Error closing stream: {e}")
        
        # Terminate PyAudio
        if self.pya:
            try:
                self.pya.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
        
        # Release USB device
        if self.usb_device:
            try:
                usb.util.dispose_resources(self.usb_device)
            except Exception as e:
                print(f"Error releasing USB device: {e}")
        
        # Restore audio servers
        self._manage_audio_servers('restore')
        
        print("✓ Audio cleanup completed") 