#!/usr/bin/env python3
"""
Configuration module for robot control dashboard
Centralizes all configuration settings
"""

# Performance configuration
STREAM_CONFIG = {
    'resolution': (320, 240),  # Further reduced resolution for lowest latency
    'jpeg_quality': 60,        # Lower quality for faster encoding
    'target_fps': 30,          # Higher FPS but with aggressive dropping
    'max_frame_age': 0.05,     # Much more aggressive - drop frames older than 50ms
    'frame_buffer_size': 1,    # Keep only 1 latest frame
    'skip_overlays': True      # Skip text overlays for speed
}

# Motor configuration
MOTOR_CONFIG = {
    'base_speed': 2000,        # Default motor speed
    'max_pwm': 4095,           # Maximum PWM value
    'enabled': False           # Safety: disabled by default
}

# Flask configuration
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5001,
    'debug': False,
    'threaded': True,
    'use_reloader': False
}

# Audio configuration
AUDIO_CONFIG = {
    'sample_rate': 16000,          # 16 kHz sample rate
    'chunk_ms': 20,                # 20ms chunks
    'channels': 6,                 # ReSpeaker v2 has 6 channels
    'format': 'paInt16',           # 16-bit audio
    'voice_threshold': 500,        # Voice activity threshold
    'doa_update_rate': 0.1,        # 10 Hz DOA updates
    'spectrum_size': 512,          # FFT size for spectrum analysis
    'auto_manage_pipewire': True,  # Automatically manage PipeWire/WirePlumber for exclusive access
    'restore_audio_on_exit': True  # Restore desktop audio when exiting robot mode
}

# Navigation configuration
NAVIGATION_CONFIG = {
    'collision_threshold': 0.20,       # 20cm collision threshold
    'base_speed': 1500,               # Base movement speed
    'turn_speed': 1000,               # Turning speed
    'angle_tolerance': 2.0,           # Degrees tolerance for heading
    'distance_tolerance': 0.05,       # 5cm distance tolerance
    'max_navigation_time': 60.0,      # Max time for navigation (seconds)
    'control_loop_freq': 10.0,        # Control loop frequency (Hz)
    'heading_correction_factor': 0.5, # PID-like factor for heading correction
    'imu_distance_estimation': True,  # Use IMU for distance estimation
    'debug_distance': False,          # Enable distance estimation debug output
    'enabled': True                   # Enable navigation features
}

# Thread timing configuration
THREAD_CONFIG = {
    'imu_update_rate': 0.05,      # 20 Hz IMU updates
    'ultrasonic_update_rate': 0.1, # 10 Hz ultrasonic updates
    'battery_update_rate': 5.0,    # 0.2 Hz battery updates (low frequency)
    'adc_update_rate': 1.0,        # 1 Hz ADC updates
    'audio_update_rate': 0.05      # 20 Hz audio data updates
}

# Safety configuration
SAFETY_CONFIG = {
    'motors_disabled_by_default': True,
    'emergency_stop_on_exit': True,
    'servo_center_on_exit': True,
    'max_servo_angle': 180,
    'min_servo_angle': 0
}

# Hardware configuration
HARDWARE_CONFIG = {
    'camera_type': 'IMX500',
    'imu_type': 'BNO085',
    'motor_type': 'Mecanum',
    'servo_channels': {
        'pan': '0',
        'tilt': '1'
    },
    'motor_channels': {
        'front_left': (0, 1),
        'rear_left': (3, 2),
        'front_right': (6, 7),
        'rear_right': (4, 5)
    }
}

# Web interface configuration
WEB_CONFIG = {
    'title': 'Robot Control Dashboard - Camera + IMU + Motors',
    'description': 'Camera + IMU + Motor Controls - Low Latency Stream',
    'safety_warning': 'Motor controls are DISABLED by default. Enable only when robot is in a safe area with adequate clearance.',
    'optimization_mode': 'ULTRA-LOW LATENCY MODE: Optimized for minimal delay. Lower resolution and quality for maximum responsiveness.',
    'keyboard_controls': 'KEYBOARD CONTROLS ENABLED: Use WASD for movement, arrow keys for camera, F1 to toggle motors, H for help.'
}

def get_all_config():
    """Get all configuration settings"""
    return {
        'stream': STREAM_CONFIG,
        'motor': MOTOR_CONFIG,
        'flask': FLASK_CONFIG,
        'thread': THREAD_CONFIG,
        'safety': SAFETY_CONFIG,
        'hardware': HARDWARE_CONFIG,
        'web': WEB_CONFIG,
        'audio': AUDIO_CONFIG,
        'navigation': NAVIGATION_CONFIG
    }

def print_config_summary():
    """Print a summary of the current configuration"""
    print("📊 Stream Configuration (Optimized for Speed):")
    print(f"   Resolution: {STREAM_CONFIG['resolution']} (reduced for speed)")
    print(f"   JPEG Quality: {STREAM_CONFIG['jpeg_quality']}% (optimized for latency)")
    print(f"   Target FPS: {STREAM_CONFIG['target_fps']} (with frame skipping)")
    print(f"   Max Frame Age: {STREAM_CONFIG['max_frame_age']*1000:.0f}ms (aggressive dropping)")
    print(f"   Overlays: {'Disabled' if STREAM_CONFIG.get('skip_overlays') else 'Minimal'}")
    print("-" * 80) 