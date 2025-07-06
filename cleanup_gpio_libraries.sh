#!/bin/bash

# GPIO Library Cleanup Script
# ============================
# This script removes conflicting GPIO libraries and installs the correct ones

set -e

echo "🔧 GPIO Library Cleanup Script"
echo "==============================="
echo ""

# Function to check if running in virtual environment
check_venv() {
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "✓ Running in virtual environment: $VIRTUAL_ENV"
        return 0
    else
        echo "⚠ Not in virtual environment"
        return 1
    fi
}

# Function to remove conflicting libraries
remove_conflicting_libraries() {
    echo "🗑️  Removing conflicting GPIO libraries..."
    
    # List of conflicting libraries to remove
    CONFLICTING_LIBS=(
        "Jetson.GPIO"
        "jetson-gpio"
        "gpiozero"
        "lgpio"
        "pigpio"
        "wiringpi"
        "wiringpi2"
    )
    
    for lib in "${CONFLICTING_LIBS[@]}"; do
        echo "  Removing $lib..."
        pip uninstall -y "$lib" 2>/dev/null || echo "    $lib not installed"
    done
    
    echo "✓ Conflicting libraries removed"
}

# Function to install correct libraries
install_correct_libraries() {
    echo "📦 Installing correct GPIO libraries..."
    
    # Install essential libraries
    pip install --upgrade pip
    pip install RPi.GPIO
    pip install spidev
    pip install pillow
    
    echo "✓ Correct libraries installed"
}

# Function to test GPIO access
test_gpio_access() {
    echo "🧪 Testing GPIO access..."
    
    # Test sysfs GPIO access
    if [ -d "/sys/class/gpio" ]; then
        echo "✓ GPIO sysfs interface available"
    else
        echo "❌ GPIO sysfs interface not available"
        return 1
    fi
    
    # Test SPI access
    if [ -c "/dev/spidev1.0" ]; then
        echo "✓ SPI device available"
    else
        echo "❌ SPI device not available"
        echo "  Run: sudo raspi-config -> Interface Options -> SPI -> Enable"
        return 1
    fi
    
    # Test user permissions
    if groups | grep -q "gpio"; then
        echo "✓ User in gpio group"
    else
        echo "⚠ User not in gpio group"
        echo "  Run: sudo usermod -a -G gpio $USER"
        echo "  Then logout and login again"
    fi
    
    if groups | grep -q "spi"; then
        echo "✓ User in spi group"
    else
        echo "⚠ User not in spi group"
        echo "  Run: sudo usermod -a -G spi $USER"
        echo "  Then logout and login again"
    fi
    
    echo "✓ GPIO access test completed"
}

# Function to create test script
create_test_script() {
    echo "📝 Creating quick test script..."
    
    cat > gpio_test.py << 'EOF'
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
EOF

    chmod +x gpio_test.py
    echo "✓ Test script created: gpio_test.py"
}

# Main execution
main() {
    echo "Starting GPIO library cleanup..."
    echo ""
    
    # Check if in virtual environment
    if check_venv; then
        echo ""
    else
        echo "⚠ Recommended to run in virtual environment"
        echo "  source venv/bin/activate"
        echo ""
    fi
    
    # Remove conflicting libraries
    remove_conflicting_libraries
    echo ""
    
    # Install correct libraries
    install_correct_libraries
    echo ""
    
    # Test GPIO access
    test_gpio_access
    echo ""
    
    # Create test script
    create_test_script
    echo ""
    
    echo "🎉 GPIO library cleanup completed!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "1. Run the test script:  python3 gpio_test.py"
    echo "2. Test your display:    python3 test_display_sysfs.py"
    echo "3. If issues persist, reboot and try again"
    echo ""
    echo "The sysfs version should work regardless of library conflicts!"
}

# Check if script is being run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 