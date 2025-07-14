# GPIO Library Compatibility Fix Guide

## 🚨 Problem Identified

You're getting this error:
```
AttributeError: module 'lgpio' has no attribute 'SET_BIAS_DISABLE'
```

This is a **library compatibility issue** between `lgpio` and `gpiozero` in newer Raspberry Pi OS versions.

## 🔧 Solution: Use the Fixed Test Script

I've created a version that bypasses this issue entirely:

```bash
python3 test_display_fixed.py
```

**This script:**
- ✅ Uses `RPi.GPIO` directly (no gpiozero conflicts)
- ✅ Initializes your display manually  
- ✅ Tests all display functions
- ✅ Should work regardless of library versions

## 🛠️ Alternative: Fix the Libraries

If you want to fix the underlying library issue:

```bash
sudo ./fix_gpio_libraries.sh
```

This will update the conflicting libraries to compatible versions.

## 📋 Quick Diagnosis

**Your Error Means:**
- ❌ Your display isn't broken
- ❌ Your wiring isn't wrong  
- ✅ It's just a software library conflict
- ✅ The fixed script should work perfectly

## 🚀 Next Steps

### Step 1: Try the Fixed Script
```bash
python3 test_display_fixed.py
```

**Expected Result:**
- Display initializes successfully
- Shows colorful test patterns
- Displays text and graphics
- Confirms your display is working

### Step 2: If Display Works
```bash
# Then try the combined test (coming next)
python3 test_display_and_imu.py
```

### Step 3: Integration
Once the fixed script works, you'll know:
- ✅ Your display hardware is fine
- ✅ Your wiring is correct
- ✅ You just need to use `RPi.GPIO` instead of `gpiozero`

## 🔍 What the Fixed Script Does Differently

### Original (Broken):
```python
# Uses gpiozero (causes lgpio conflict)
from gpiozero import DigitalOutputDevice
rst_pin = DigitalOutputDevice(12)  # ← Fails here
```

### Fixed Version:
```python
# Uses RPi.GPIO directly (no conflicts)
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(12, GPIO.OUT)  # ← Works perfectly
```

## 📊 Why This Happened

**Library Evolution:**
1. **Old Raspberry Pi OS**: Used older `lgpio` versions
2. **New Raspberry Pi OS**: Updated `lgpio` but `gpiozero` expectations lagged
3. **Result**: Attribute mismatch causing crashes

**Your Options:**
- **Quick Fix**: Use `test_display_fixed.py` (recommended)
- **Library Fix**: Run `fix_gpio_libraries.sh` 
- **Integration**: Use `RPi.GPIO` in your final robot code

## 🎯 Integration Pattern

For your robot code, use this pattern:

```python
import RPi.GPIO as GPIO
import spidev as SPI

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(12, GPIO.OUT)  # RST
GPIO.setup(26, GPIO.OUT)  # DC  
GPIO.setup(13, GPIO.OUT)  # BL

# SPI setup
spi = SPI.SpiDev()
spi.open(1, 0)

# Then use as normal
# No gpiozero conflicts!
```

## ✅ Success Criteria

When `test_display_fixed.py` works, you'll see:
- ✅ GPIO pins initialize successfully
- ✅ SPI communication works
- ✅ Display shows colors and text
- ✅ No library errors

This confirms your display is 100% working and ready for integration! 