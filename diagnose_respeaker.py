#!/usr/bin/env python3
"""
Diagnostic script for ReSpeaker v2 setup
"""

import pyaudio
import numpy as np
import time
import sys

def check_audio_devices():
    """Check all available audio devices"""
    print("🎤 Audio Device Diagnostics")
    print("=" * 50)
    
    pya = pyaudio.PyAudio()
    
    print("Available audio devices:")
    for i in range(pya.get_device_count()):
        info = pya.get_device_info_by_index(i)
        print(f"  {i}: {info['name']}")
        print(f"     Input channels: {info['maxInputChannels']}")
        print(f"     Output channels: {info['maxOutputChannels']}")
        print(f"     Sample rate: {info['defaultSampleRate']} Hz")
        print(f"     Host API: {info['hostApi']}")
        print()
    
    pya.terminate()

def test_respeaker_audio():
    """Test ReSpeaker audio capture"""
    print("🎵 Testing ReSpeaker Audio Capture")
    print("=" * 50)
    
    pya = pyaudio.PyAudio()
    
    # Find ReSpeaker device
    respeaker_index = None
    for i in range(pya.get_device_count()):
        info = pya.get_device_info_by_index(i)
        name = info['name'].lower()
        if "respeaker" in name or "seeed" in name or info['maxInputChannels'] >= 6:
            respeaker_index = i
            print(f"✓ Found potential ReSpeaker device: {info['name']} (index {i})")
            break
    
    if respeaker_index is None:
        print("⚠ No ReSpeaker device found, trying default input")
        try:
            default_info = pya.get_default_input_device_info()
            respeaker_index = default_info['index']
            print(f"Using default input: {default_info['name']}")
        except:
            print("✗ No default input device found")
            return
    
    # Test audio capture
    try:
        # Try 6-channel capture first
        print(f"\nTesting 6-channel capture from device {respeaker_index}...")
        
        stream = pya.open(
            format=pyaudio.paInt16,
            channels=6,
            rate=16000,
            input=True,
            input_device_index=respeaker_index,
            frames_per_buffer=320
        )
        
        print("✓ 6-channel stream opened successfully")
        
        # Capture some audio
        print("Recording 5 seconds of audio...")
        frames = []
        for i in range(50):  # 5 seconds at 10Hz
            try:
                data = stream.read(320, exception_on_overflow=False)
                frames.append(data)
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) >= 6:
                    # Calculate RMS for each channel
                    channels = audio_data.reshape(-1, 6)
                    rms_values = np.sqrt(np.mean(channels.astype(np.float32) ** 2, axis=0))
                    
                    print(f"Frame {i+1}: RMS values: {rms_values[:4]} (first 4 channels)")
                    
                    # Check if we're getting any signal
                    max_rms = np.max(rms_values)
                    if max_rms > 100:
                        print(f"  ✓ Audio signal detected! Max RMS: {max_rms:.2f}")
                    else:
                        print(f"  ⚠ Very low audio levels. Max RMS: {max_rms:.2f}")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error reading audio: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        
        # Try 1-channel capture as fallback
        print(f"\nTesting 1-channel capture from device {respeaker_index}...")
        
        stream = pya.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=respeaker_index,
            frames_per_buffer=320
        )
        
        print("✓ 1-channel stream opened successfully")
        
        # Capture some audio
        print("Recording 3 seconds of audio...")
        frames = []
        for i in range(30):  # 3 seconds at 10Hz
            try:
                data = stream.read(320, exception_on_overflow=False)
                frames.append(data)
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                
                print(f"Frame {i+1}: RMS: {rms:.2f}")
                
                if rms > 100:
                    print(f"  ✓ Audio signal detected! RMS: {rms:.2f}")
                else:
                    print(f"  ⚠ Very low audio levels. RMS: {rms:.2f}")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error reading audio: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"✗ Error testing audio capture: {e}")
    
    pya.terminate()

def check_usb_device():
    """Check USB device status"""
    print("\n🔌 USB Device Diagnostics")
    print("=" * 50)
    
    try:
        import usb.core
        
        # ReSpeaker v2 USB IDs
        VENDOR_ID = 0x2886
        PRODUCT_ID = 0x0018
        
        device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        if device:
            print(f"✓ ReSpeaker USB device found")
            print(f"  Vendor ID: {device.idVendor:04x}")
            print(f"  Product ID: {device.idProduct:04x}")
            print(f"  Manufacturer: {usb.util.get_string(device, device.iManufacturer)}")
            print(f"  Product: {usb.util.get_string(device, device.iProduct)}")
            
            # Check if kernel driver is active
            try:
                if device.is_kernel_driver_active(0):
                    print("  ⚠ Kernel driver is active")
                else:
                    print("  ✓ Kernel driver is not active")
            except:
                print("  ⚠ Could not check kernel driver status")
            
            # Try to read DOA data
            try:
                print("\nTesting DOA data read...")
                data = device.ctrl_transfer(
                    usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                    0,  # request
                    0x0200,  # DOA register
                    0,  # index
                    4  # data length
                )
                print(f"✓ DOA data read successful: {list(data)}")
            except Exception as e:
                print(f"✗ DOA data read failed: {e}")
                
        else:
            print("✗ ReSpeaker USB device not found")
            
    except ImportError:
        print("⚠ pyusb not available")
    except Exception as e:
        print(f"✗ Error checking USB device: {e}")

def main():
    """Main diagnostic function"""
    print("🔍 ReSpeaker v2 Diagnostic Tool")
    print("=" * 50)
    
    check_audio_devices()
    check_usb_device()
    test_respeaker_audio()
    
    print("\n✅ Diagnostic completed!")
    print("\nRecommendations:")
    print("1. If no audio signal is detected, check microphone permissions")
    print("2. If USB device is busy, try: sudo usbreset 2886 0018")
    print("3. If kernel driver is active, it may interfere with DOA access")
    print("4. Try running: amixer set 'ReSpeaker 4 Mic Array' 100%")

if __name__ == "__main__":
    main() 