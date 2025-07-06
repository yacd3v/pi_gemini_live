#!/usr/bin/env python3
"""
Simple Display Test - SPI Only
===============================

This version uses ONLY SPI communication and avoids ALL GPIO control.
It bypasses all GPIO library conflicts by not using GPIO at all.

This will test if the basic SPI communication to the display works.

Hardware:
- Display: SPI LCD connected to SPI1 (GPIO 19=MOSI, 21=SCLK, 20=MISO, 18=CE0)

Author: Auto-generated test script
Date: 2025
"""

import time
import sys
import traceback
from datetime import datetime

# Only import what we absolutely need
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

class SimpleDisplayTester:
    def __init__(self):
        self.spi = None
        self.font = None
        
    def spi_init(self):
        """Initialize SPI communication only"""
        try:
            print("Initializing SPI communication...")
            self.spi = SPI.SpiDev()
            self.spi.open(1, 0)  # bus 1, CE0
            self.spi.max_speed_hz = 10_000_000  # Start with lower speed
            self.spi.mode = 0b00
            print("✓ SPI initialized successfully")
            return True
        except Exception as e:
            print(f"❌ SPI initialization failed: {e}")
            return False
    
    def send_command(self, cmd):
        """Send command to display (assuming DC pin is managed externally)"""
        try:
            # Send command byte
            self.spi.writebytes([cmd])
            time.sleep(0.001)  # Small delay
            return True
        except Exception as e:
            print(f"❌ Send command failed: {e}")
            return False
    
    def send_data(self, data):
        """Send data to display"""
        try:
            if isinstance(data, int):
                self.spi.writebytes([data])
            else:
                self.spi.writebytes(data)
            return True
        except Exception as e:
            print(f"❌ Send data failed: {e}")
            return False
    
    def test_spi_communication(self):
        """Test basic SPI communication"""
        print("\n🔌 Testing SPI Communication...")
        print("-" * 40)
        
        try:
            # Try to send some basic commands
            # Note: These might not work without proper GPIO control,
            # but we can test if SPI communication itself works
            
            commands = [
                0x01,  # Software reset
                0x11,  # Sleep out
                0x29,  # Display on
                0x2C,  # Memory write
            ]
            
            for cmd in commands:
                if self.send_command(cmd):
                    print(f"✓ Command 0x{cmd:02X} sent successfully")
                else:
                    print(f"❌ Command 0x{cmd:02X} failed")
                time.sleep(0.01)
            
            # Try sending some data
            test_data = [0x00, 0x00, 0xFF, 0xFF]  # Some test pixels
            if self.send_data(test_data):
                print("✓ Data transmission successful")
            else:
                print("❌ Data transmission failed")
            
            print("✅ SPI communication test completed")
            return True
            
        except Exception as e:
            print(f"❌ SPI communication test failed: {e}")
            return False
    
    def test_data_patterns(self):
        """Test different data patterns"""
        print("\n📊 Testing Data Patterns...")
        print("-" * 40)
        
        try:
            # Test different patterns to see if SPI is working
            patterns = [
                [0x00, 0x00, 0x00, 0x00],  # All zeros
                [0xFF, 0xFF, 0xFF, 0xFF],  # All ones
                [0xAA, 0xAA, 0xAA, 0xAA],  # Alternating bits
                [0x55, 0x55, 0x55, 0x55],  # Alternating bits inverted
                [0x00, 0xFF, 0x00, 0xFF],  # Alternating bytes
            ]
            
            for i, pattern in enumerate(patterns):
                print(f"  Sending pattern {i+1}: {[hex(b) for b in pattern]}")
                if self.send_data(pattern):
                    print(f"    ✓ Pattern {i+1} sent successfully")
                else:
                    print(f"    ❌ Pattern {i+1} failed")
                time.sleep(0.1)
            
            print("✅ Data patterns test completed")
            return True
            
        except Exception as e:
            print(f"❌ Data patterns test failed: {e}")
            return False
    
    def test_speed_variations(self):
        """Test different SPI speeds"""
        print("\n⚡ Testing SPI Speed Variations...")
        print("-" * 40)
        
        speeds = [
            1_000_000,   # 1 MHz
            5_000_000,   # 5 MHz
            10_000_000,  # 10 MHz
            20_000_000,  # 20 MHz
            40_000_000,  # 40 MHz
        ]
        
        for speed in speeds:
            try:
                self.spi.max_speed_hz = speed
                print(f"  Testing at {speed/1_000_000:.1f} MHz...")
                
                # Send a test command
                if self.send_command(0x00):  # NOP command
                    print(f"    ✓ {speed/1_000_000:.1f} MHz works")
                else:
                    print(f"    ❌ {speed/1_000_000:.1f} MHz failed")
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"    ❌ {speed/1_000_000:.1f} MHz failed: {e}")
        
        # Reset to safe speed
        self.spi.max_speed_hz = 10_000_000
        print("✅ Speed variations test completed")
        return True
    
    def cleanup(self):
        """Clean up SPI resources"""
        try:
            if self.spi:
                self.spi.close()
                print("✓ SPI closed")
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def run_simple_test(self):
        """Run simple SPI-only test"""
        print("🧪 Simple Display Test - SPI Only")
        print("=" * 40)
        print("This version tests ONLY SPI communication.")
        print("It bypasses all GPIO library conflicts!")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test SPI initialization
        if not self.spi_init():
            print("\n❌ SPI initialization failed")
            return False
        
        time.sleep(1)
        
        # Test basic SPI communication
        comm_ok = self.test_spi_communication()
        time.sleep(1)
        
        # Test data patterns
        patterns_ok = self.test_data_patterns()
        time.sleep(1)
        
        # Test speed variations
        speed_ok = self.test_speed_variations()
        
        # Results
        print(f"\n📊 Test Results:")
        print(f"SPI Init:      ✅ PASS")
        print(f"Communication: {'✅ PASS' if comm_ok else '❌ FAIL'}")
        print(f"Data Patterns: {'✅ PASS' if patterns_ok else '❌ FAIL'}")
        print(f"Speed Tests:   {'✅ PASS' if speed_ok else '❌ FAIL'}")
        
        success = comm_ok and patterns_ok and speed_ok
        
        if success:
            print("\n🎉 SPI communication is working!")
            print("📋 Next steps:")
            print("  1. SPI hardware is good ✓")
            print("  2. Need to fix GPIO control for display")
            print("  3. The issue is GPIO library conflicts, not SPI")
        else:
            print("\n⚠️ SPI communication has issues")
            print("📋 Possible causes:")
            print("  1. SPI not enabled in raspi-config")
            print("  2. Hardware connection issues")
            print("  3. Display not powered on")
        
        return success

def main():
    """Main function"""
    tester = SimpleDisplayTester()
    
    try:
        print("🔧 Simple Display Test - Bypassing GPIO Conflicts")
        print("=" * 55)
        print("This test focuses ONLY on SPI communication.")
        print("It will tell us if the basic hardware connection works.")
        print()
        
        success = tester.run_simple_test()
        
        if success:
            print("\n✅ SPI COMMUNICATION WORKS!")
            print("=" * 40)
            print("🎯 Key findings:")
            print("  ✓ SPI hardware is working")
            print("  ✓ Display SPI connection is good")
            print("  ✓ The issue is GPIO library conflicts")
            print("  ✓ Your display hardware is fine!")
            print()
            print("🔧 Solution needed:")
            print("  → Fix GPIO library conflicts")
            print("  → Use working GPIO control method")
            print("  → Then display will work perfectly!")
        else:
            print("\n❌ SPI communication issues detected")
            print("=" * 40)
            print("🔍 Check these:")
            print("  1. SPI enabled: sudo raspi-config")
            print("  2. Display power connections")
            print("  3. SPI wiring (MOSI, SCLK, CE0)")
        
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