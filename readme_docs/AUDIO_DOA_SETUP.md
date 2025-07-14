# Audio & DOA Setup Guide

## Overview

The robot dashboard now includes **ReSpeaker v2 audio streaming** and **DOA (Direction of Arrival)** detection capabilities. This provides real-time audio visualization and sound source localization in the web interface.

## Features Added

### 🎤 Audio System
- **ReSpeaker v2 Integration**: Full support for the 6-channel ReSpeaker v2 microphone array
- **Real-time Audio Streaming**: Continuous audio data processing at 16kHz
- **Voice Activity Detection**: Automatic detection of speech vs. background noise
- **Volume Level Monitoring**: Real-time audio level measurement in dB
- **Spectrum Analysis**: Audio frequency spectrum visualization (when available)

### 🎯 DOA (Direction of Arrival)
- **360° Sound Localization**: Determines the angle of incoming sound (0-359°)
- **Visual DOA Indicator**: Real-time compass-style visualization in the web UI
- **Voice Activity Integration**: DOA arrow only appears when voice is detected
- **Volume-based Visualization**: Arrow thickness and opacity based on volume level

### 🌐 Web Interface
- **DOA Compass**: Interactive circular visualization showing sound direction
- **Audio Status Panel**: Real-time display of DOA angle, volume, and voice activity
- **Status Indicators**: Visual feedback for audio system health
- **Live Updates**: 10Hz refresh rate for smooth real-time visualization

## Hardware Requirements

### ReSpeaker v2 4-Mic Array
- **Product**: Seeed Studio ReSpeaker 4-Mic Array for Raspberry Pi
- **Channels**: 6 input channels (4 microphones + 2 processed)
- **Connection**: USB interface
- **DOA Range**: 360° coverage
- **Frequency Response**: 100Hz - 10kHz

### Verification Commands
```bash
# Check USB connection
lsusb | grep -i seeed
# Should show: Bus 001 Device 002: ID 2886:0018 Seeed Technology Co., Ltd. ReSpeaker 4 Mic Array (UAC1.0)

# Check audio devices
arecord -l
# Should list the ReSpeaker as a capture device

# Check ALSA cards
cat /proc/asound/cards
# Should show ReSpeaker if properly configured
```

## Software Architecture

### Module Structure
```
modules/
├── audio.py          # ReSpeaker audio manager
├── config.py         # Audio configuration
└── web_routes.py     # Audio API endpoints
```

### Key Components

#### ReSpeakerManager Class
- **Audio Callback**: Processes 6-channel audio data in real-time
- **DOA Processing**: Extracts direction information from USB control interface
- **Thread Safety**: Mutex-protected data access
- **Simulation Mode**: Fallback demo mode when hardware unavailable

#### Audio Configuration
```python
AUDIO_CONFIG = {
    'sample_rate': 16000,          # 16 kHz sample rate
    'chunk_ms': 20,                # 20ms chunks
    'channels': 6,                 # ReSpeaker v2 has 6 channels
    'format': 'paInt16',           # 16-bit audio
    'voice_threshold': 500,        # Voice activity threshold
    'doa_update_rate': 0.1,        # 10 Hz DOA updates
    'spectrum_size': 512           # FFT size for spectrum analysis
}
```

## API Endpoints

### `/audio_data` (GET)
Returns real-time audio and DOA data:
```json
{
    "doa_angle": 45,           # Direction in degrees (0-359)
    "doa_x": 0.707,           # X component for visualization
    "doa_y": 0.707,           # Y component for visualization
    "voice_activity": true,    # Voice detected boolean
    "volume_level": 1250.5,   # Raw volume level
    "volume_db": -12.3,       # Volume in decibels
    "timestamp": 1752391093.7  # Unix timestamp
}
```

### `/audio_spectrum` (GET)
Returns audio frequency spectrum:
```json
{
    "frequencies": [0, 31.25, 62.5, ...],  # Frequency bins (Hz)
    "magnitudes": [-40.2, -35.1, ...],     # Magnitude in dB
    "sample_rate": 16000                    # Sample rate
}
```

### `/status` (GET)
Includes audio system status:
```json
{
    "audio": {
        "available": true,        # Audio libraries available
        "initialized": true,      # Audio system initialized
        "streaming": true,        # Audio streaming active
        "simulation_mode": false, # Using real hardware
        "sample_rate": 16000,     # Sample rate
        "channels": 6             # Channel count
    }
}
```

## Web Interface Integration

### DOA Visualization
The web interface includes a real-time DOA compass:

- **Circular Display**: 200x200 pixel canvas with compass directions
- **Direction Arrow**: Red arrow pointing to sound source
- **Volume Indicator**: Circle size varies with volume level
- **Voice Activity**: Arrow only appears when voice is detected
- **Smooth Updates**: 10Hz refresh rate for fluid visualization

### JavaScript Integration
```javascript
// DOA visualization updates at 10Hz
setInterval(updateAudioData, 100);

// Canvas-based compass with N/E/S/W indicators
function drawDOAVisualization(ctx, doaData) {
    // Draw compass, arrow, and volume indicators
}
```

## Configuration and Troubleshooting

### Audio Device Detection
The system automatically detects ReSpeaker devices by:
1. **Device Name**: Looking for "respeaker" or "seeed" in device names
2. **Channel Count**: Devices with 6+ input channels
3. **USB Audio**: USB-connected audio devices
4. **Fallback**: Default system audio input if no ReSpeaker found

### Common Issues and Solutions

#### 1. "ReSpeaker device not found"
**Problem**: Audio device not detected by PyAudio
**Solutions**:
```bash
# Check USB connection
lsusb | grep -i seeed

# Restart audio services
sudo systemctl restart alsa-state
sudo systemctl restart pulseaudio

# Check permissions
groups $USER  # Should include 'audio' group
```

#### 2. "Resource busy" USB Error
**Problem**: ReSpeaker USB interface in use by another process
**Solutions**:
```bash
# Check for other audio processes
ps aux | grep -i audio
ps aux | grep -i respeaker

# Kill conflicting processes
sudo pkill -f vocal_gemini.py  # If running simultaneously
```

#### 3. Audio Streaming but No DOA
**Problem**: Audio works but DOA shows 0°
**Solutions**:
- **USB Permission**: Ensure user has USB device access
- **Driver Issues**: ReSpeaker DOA requires specific USB control access
- **Simulation Mode**: System automatically falls back to demo mode

#### 4. High Audio Latency
**Problem**: Delayed audio visualization
**Solutions**:
- **Buffer Size**: Reduce chunk_ms in configuration
- **Sample Rate**: Ensure 16kHz sample rate
- **System Load**: Check CPU usage during operation

### Performance Optimization

#### Audio Processing
```python
# Optimized audio callback
def _audio_callback(self, in_data, frame_count, time_info, status):
    # Minimal processing in callback
    # Thread-safe data updates
    # Efficient numpy operations
```

#### Web Interface
```javascript
// Efficient canvas updates
function drawDOAVisualization(ctx, doaData) {
    // Clear only necessary areas
    // Use requestAnimationFrame for smooth animation
    // Batch DOM updates
}
```

## Testing and Validation

### Audio Test Script
```bash
python3 test_audio_doa.py
```

This script:
1. **Tests API Endpoints**: Verifies all audio endpoints respond correctly
2. **Live Monitoring**: Shows real-time DOA data for 10 seconds
3. **Status Validation**: Confirms audio system initialization

### Expected Output
```
🎤 Testing Audio & DOA Endpoints...
✓ DOA Angle: 45°
✓ Voice Activity: True
✓ Volume Level: -12.3 dB
✓ Audio Streaming: True
✓ Simulation Mode: False
```

### Manual Testing
1. **Open Web Interface**: Navigate to http://raspberry_pi_ip:5001
2. **Audio Section**: Look for "🎤 Audio & DOA" section
3. **DOA Compass**: Should show circular visualization
4. **Voice Test**: Speak near ReSpeaker, arrow should appear
5. **Direction Test**: Move around ReSpeaker, arrow should follow

## Integration with Existing Features

### Robot Dashboard
The audio system integrates seamlessly with:
- **Camera System**: Audio and video synchronized
- **Motor Controls**: Could be extended for audio-guided movement
- **IMU Data**: Combined spatial awareness
- **Web Interface**: Unified control panel

### Future Enhancements
- **Audio-guided Camera**: Automatically pan camera toward sound source
- **Voice Commands**: Integration with speech recognition
- **Audio Recording**: Save audio clips with DOA metadata
- **Multi-source Tracking**: Track multiple simultaneous sound sources
- **Noise Cancellation**: Advanced audio processing for better DOA accuracy

## Conclusion

The audio and DOA system provides a comprehensive solution for real-time sound localization and visualization. The modular design allows for easy extension and integration with other robot systems, while the web interface provides an intuitive way to monitor and interact with the audio capabilities.

The system gracefully handles hardware availability issues through simulation mode, ensuring the dashboard remains functional even when the ReSpeaker hardware is not available or properly configured. 