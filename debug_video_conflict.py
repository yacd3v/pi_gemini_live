#!/usr/bin/env python3
"""
Debug script to isolate camera vs servo/motor conflicts
"""
import time
import threading
import sys
import os

# Add the current directory to Python path
sys.path.append('/home/yannis.achour/dev2/vocalgem')

def test_hardware_isolation():
    """Test each hardware component independently"""
    print("🔧 Testing hardware components independently...")
    
    # Test 1: Servo only
    print("\n1️⃣ Testing servo control alone...")
    try:
        from freenove_examples.servo import Servo
        servo = Servo()
        
        print("   Moving servo pan to 45°...")
        servo.set_servo_pwm('0', 45)
        time.sleep(1)
        
        print("   Moving servo pan to 135°...")
        servo.set_servo_pwm('0', 135)
        time.sleep(1)
        
        print("   Moving servo pan back to center...")
        servo.set_servo_pwm('0', 90)
        time.sleep(1)
        
        servo.close()
        print("   ✅ Servo test completed successfully")
        
    except Exception as e:
        print(f"   ❌ Servo test failed: {e}")
        return False
    
    # Test 2: Motor only
    print("\n2️⃣ Testing motor control alone...")
    try:
        from chassis_functions import ChassisController
        chassis = ChassisController()
        
        print("   Moving forward 10cm...")
        chassis.move_forward_distance(10, 800)
        time.sleep(1)
        
        print("   Moving backward 10cm...")
        chassis.move_forward_distance(-10, -800)
        time.sleep(1)
        
        print("   Emergency stop...")
        chassis.emergency_stop()
        
        chassis.close()
        print("   ✅ Motor test completed successfully")
        
    except Exception as e:
        print(f"   ❌ Motor test failed: {e}")
        return False
    
    # Test 3: Camera only
    print("\n3️⃣ Testing camera alone...")
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        
        config = camera.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            buffer_count=3
        )
        camera.configure(config)
        camera.start()
        time.sleep(0.5)
        
        print("   Capturing 5 test frames...")
        for i in range(5):
            frame = camera.capture_array()
            print(f"   Frame {i+1}: {frame.shape}")
            time.sleep(0.1)
        
        camera.stop()
        camera.close()
        print("   ✅ Camera test completed successfully")
        
    except Exception as e:
        print(f"   ❌ Camera test failed: {e}")
        return False
    
    return True

def test_concurrent_access():
    """Test concurrent camera and servo access"""
    print("\n🔄 Testing concurrent camera + servo access...")
    
    camera = None
    servo = None
    camera_running = False
    
    def camera_loop():
        nonlocal camera, camera_running
        try:
            from picamera2 import Picamera2
            camera = Picamera2()
            
            config = camera.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                buffer_count=3
            )
            camera.configure(config)
            camera.start()
            time.sleep(0.5)
            
            camera_running = True
            frame_count = 0
            
            print("   📹 Camera loop started")
            while camera_running and frame_count < 20:  # Run for ~2 seconds
                frame = camera.capture_array()
                frame_count += 1
                print(f"   📹 Frame {frame_count}: {frame.shape}")
                time.sleep(0.1)  # 10 FPS
                
        except Exception as e:
            print(f"   ❌ Camera loop error: {e}")
        finally:
            if camera:
                try:
                    camera.stop()
                    camera.close()
                except:
                    pass
            print("   📹 Camera loop ended")
    
    def servo_commands():
        nonlocal servo
        try:
            from freenove_examples.servo import Servo
            servo = Servo()
            
            print("   🤖 Starting servo commands while camera runs...")
            time.sleep(1)  # Let camera start first
            
            positions = [45, 90, 135, 90, 60, 90]
            for i, pos in enumerate(positions):
                if not camera_running:
                    break
                    
                print(f"   🤖 Servo command {i+1}: Move to {pos}°")
                start_time = time.time()
                
                try:
                    servo.set_servo_pwm('0', pos)
                    end_time = time.time()
                    print(f"   🤖 Servo command completed in {end_time - start_time:.3f}s")
                except Exception as servo_e:
                    print(f"   ❌ Servo command failed: {servo_e}")
                
                time.sleep(0.5)
                
        except Exception as e:
            print(f"   ❌ Servo commands error: {e}")
        finally:
            if servo:
                servo.close()
            print("   🤖 Servo commands ended")
    
    # Start both threads
    camera_thread = threading.Thread(target=camera_loop)
    servo_thread = threading.Thread(target=servo_commands)
    
    camera_thread.start()
    time.sleep(0.2)  # Small delay
    servo_thread.start()
    
    # Wait for both to complete
    camera_thread.join()
    servo_thread.join()
    
    # Stop camera if still running
    camera_running = False
    
    print("   ✅ Concurrent test completed")

def test_resource_timing():
    """Test timing between operations"""
    print("\n⏱️  Testing operation timing...")
    
    try:
        from freenove_examples.servo import Servo
        from picamera2 import Picamera2
        
        servo = Servo()
        camera = Picamera2()
        
        config = camera.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            buffer_count=3
        )
        camera.configure(config)
        camera.start()
        time.sleep(0.5)
        
        print("   Testing servo speed with camera active...")
        for i in range(5):
            start_time = time.time()
            
            # Capture frame
            frame = camera.capture_array()
            capture_time = time.time()
            
            # Move servo
            servo.set_servo_pwm('0', 45 + (i * 20))
            servo_time = time.time()
            
            print(f"   Test {i+1}: Capture={capture_time-start_time:.3f}s, Servo={servo_time-capture_time:.3f}s")
            time.sleep(0.2)
        
        camera.stop()
        camera.close()
        servo.close()
        
    except Exception as e:
        print(f"   ❌ Timing test failed: {e}")

if __name__ == "__main__":
    print("🐛 VocalGem Hardware Conflict Debug Tool")
    print("=" * 50)
    
    # Test 1: Individual components
    if not test_hardware_isolation():
        print("❌ Basic hardware tests failed - stopping here")
        sys.exit(1)
    
    # Test 2: Concurrent access
    test_concurrent_access()
    
    # Test 3: Timing analysis
    test_resource_timing()
    
    print("\n✅ Debug tests completed")
    print("Check the output above to identify timing issues or resource conflicts") 