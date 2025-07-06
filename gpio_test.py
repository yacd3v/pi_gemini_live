#!/usr/bin/env python3
"""Quick GPIO test script"""

import os
import time

def test_gpio_export():
    """Test GPIO export functionality"""
    test_pin = 18  # Safe test pin
    
    try:
        # Export pin
        with open('/sys/class/gpio/export', 'w') as f:
            f.write(str(test_pin))
        time.sleep(0.1)
        
        # Set direction
        with open(f'/sys/class/gpio/gpio{test_pin}/direction', 'w') as f:
            f.write('out')
        
        # Test write
        with open(f'/sys/class/gpio/gpio{test_pin}/value', 'w') as f:
            f.write('1')
        
        # Cleanup
        with open('/sys/class/gpio/unexport', 'w') as f:
            f.write(str(test_pin))
        
        print("✅ GPIO sysfs access works!")
        return True
        
    except Exception as e:
        print(f"❌ GPIO sysfs access failed: {e}")
        return False

def test_spi_access():
    """Test SPI access"""
    try:
        import spidev
        spi = spidev.SpiDev()
        spi.open(1, 0)
        spi.close()
        print("✅ SPI access works!")
        return True
    except Exception as e:
        print(f"❌ SPI access failed: {e}")
        return False

def test_pil():
    """Test PIL/Pillow"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        print("✅ PIL/Pillow works!")
        return True
    except Exception as e:
        print(f"❌ PIL/Pillow failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Quick GPIO Library Test")
    print("=" * 30)
    
    gpio_ok = test_gpio_export()
    spi_ok = test_spi_access()
    pil_ok = test_pil()
    
    print("\n📊 Results:")
    print(f"GPIO:  {'✅ PASS' if gpio_ok else '❌ FAIL'}")
    print(f"SPI:   {'✅ PASS' if spi_ok else '❌ FAIL'}")
    print(f"PIL:   {'✅ PASS' if pil_ok else '❌ FAIL'}")
    
    if gpio_ok and spi_ok and pil_ok:
        print("\n🎉 All tests PASSED! GPIO libraries are working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check error messages above.")
