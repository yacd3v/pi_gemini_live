# Display Conflict Solution Guide

## 🚨 Problem Identified

Your round screen stopped working after running the BNO085 test script. This is a **GPIO/I2C conflict** issue.

## 🔍 Root Cause Analysis

### Your Setup:
- **Display**: SPI-based LCD (GPIO 12, 26, 13) - NOT I2C
- **BNO085**: I2C-based sensor (GPIO 2, 3)
- **Other devices**: Multiple I2C devices at addresses 0x2d, 0x40, 0x48, 0x70

### The Issue:
Even though your display uses SPI, the BNO085 test script is interfering with the **GPIO system** that your display needs. This can happen because:

1. **I2C bus conflicts** - Multiple devices competing for I2C bus time
2. **GPIO library conflicts** - Different GPIO libraries interfering with each other
3. **Timing issues** - I2C clock stretching from BNO085 affecting other systems

## 🛠️ Solutions (Try in Order)

### Solution 1: Use the Safe Test Script ⭐ RECOMMENDED

I've created a safer version that minimizes conflicts:

```bash
# Use this instead of the original test script
python3 test_IMU_safe.py
```

**Features:**
- Minimal I2C settings (no custom frequency)
- Reduced retry attempts
- Only essential sensor features enabled
- Shorter test duration (30 seconds)

### Solution 2: Test BNO085 Separately

```bash
# 1. Stop your vocal_gemini.py script
# 2. Test BNO085 alone
python3 test_IMU_safe.py

# 3. If BNO085 works, restart your display script
python3 vocal_gemini.py
```

### Solution 3: Power Cycle Between Tests

```bash
# 1. Stop all scripts
# 2. Unplug USB power
# 3. Wait 5 seconds
# 4. Plug power back in
# 5. Test BNO085 first
python3 test_IMU_safe.py

# 6. Then start your display script
python3 vocal_gemini.py
```

### Solution 4: I2C Multiplexer (Best Long-term)

Get a **TCA9548A I2C multiplexer** (~$5):

```python
# Example setup with multiplexer
import board
import busio
from adafruit_tca9548a import TCA9548A
from adafruit_bno08x.i2c import BNO08X_I2C

# Main I2C bus
i2c = busio.I2C(board.SCL, board.SDA)
tca = TCA9548A(i2c)

# BNO085 on channel 0
bno_i2c = tca[0]
bno = BNO08X_I2C(bno_i2c)

# Display on channel 1 (if it were I2C)
# display_i2c = tca[1]
```

## 🧪 Testing Strategy

### Step 1: Verify BNO085 Works
```bash
python3 test_IMU_safe.py
```
**Expected**: Should work without affecting display

### Step 2: Test Display Alone
```bash
python3 vocal_gemini.py
```
**Expected**: Display should work normally

### Step 3: Test Together
```bash
# Terminal 1: Start display
python3 vocal_gemini.py

# Terminal 2: Test BNO085 (in another terminal)
python3 test_IMU_safe.py
```

## 🔧 If Display Still Doesn't Work

### Quick Fix:
```bash
# 1. Stop all scripts
sudo pkill -f python3

# 2. Clean up GPIO
sudo python3 -c "
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.cleanup()
"

# 3. Reboot
sudo reboot

# 4. After reboot, test display first
python3 vocal_gemini.py
```

### GPIO Reset Script:
```bash
#!/bin/bash
echo "Resetting GPIO system..."
sudo python3 -c "
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.cleanup()
import time
time.sleep(1)
print('GPIO reset complete')
"
```

## 📊 Expected Behavior

### With Safe Script:
- ✅ BNO085 works perfectly
- ✅ Display continues working
- ⚠️ Minor I2C bus sharing (acceptable)

### With Original Script:
- ✅ BNO085 works perfectly  
- ❌ Display may stop working
- ⚠️ I2C conflicts with other devices

## 🎯 Integration Strategy

### For Your Robot:

1. **Use the safe script** for BNO085 testing
2. **Test separately** - BNO085 first, then display
3. **Consider multiplexer** for production use
4. **Monitor for conflicts** - if display acts up, power cycle

### Code Integration:
```python
# In your robot code, use minimal BNO085 initialization
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR
import board
import busio

# Minimal setup
i2c = busio.I2C(board.SCL, board.SDA)
bno = BNO08X_I2C(i2c)
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)

# Get orientation when needed
def get_orientation():
    quat = bno.quaternion
    return quat if quat else None
```

## 🆘 Emergency Recovery

If everything stops working:

```bash
# 1. Hard reset
sudo reboot

# 2. Test display first
python3 vocal_gemini.py

# 3. If display works, test BNO085
python3 test_IMU_safe.py

# 4. If both work, you're good!
```

## 📝 Summary

- **Use `test_IMU_safe.py`** instead of the original
- **Test devices separately** to isolate issues
- **Power cycle** if conflicts occur
- **Consider I2C multiplexer** for permanent solution

The safe script should solve your display conflict while still giving you full BNO085 functionality! 