# 🚗 Motor Control Safety Guide

## ⚠️ **CRITICAL SAFETY INFORMATION**

### **Before Using Motor Controls:**

1. **🔒 Motors are DISABLED by default** for safety
2. **🛡️ Always test in a safe, open area** with adequate clearance
3. **👥 Keep people and pets away** during testing
4. **🔌 Ensure stable power supply** to prevent unexpected behavior
5. **🚫 Never leave motors enabled unattended**

## 🎮 **Motor Control Features**

### **Web Interface Controls:**

#### **Safety Controls:**
- **🟢 ENABLE**: Activates motor control (requires confirmation)
- **🛑 E-STOP**: Immediate emergency stop (always accessible)
- **⚫ DISABLE**: Safely disables motors and stops movement

#### **Movement Controls:**
- **⬆️ FWD / ⬇️ BACK**: Forward/backward movement
- **⬅️ LEFT / ➡️ RIGHT**: Spin left/right (rotation in place)
- **↖️ STR-L / ↗️ STR-R**: Strafe left/right (sideways movement)
- **🔄 SPIN-L**: Additional spin left control

#### **Speed Control:**
- **Slider**: Adjusts speed from 10% to 100%
- **Default**: 50% speed for safe testing

### **Keyboard Controls (when motors enabled):**
- `W` - Forward
- `S` - Backward  
- `A` - Spin left
- `D` - Spin right
- `Q` - Strafe left
- `E` - Strafe right
- `SPACE` - Emergency stop

## 🔧 **Motor Mapping (Based on customMotor.py)**

The robot uses **Mecanum wheel** configuration:

```
Front-Left (FL)  [PWM 0,1]    Front-Right (FR) [PWM 6,7]
     ⚙️                           ⚙️
     
     
Rear-Left (RL)   [PWM 3,2]    Rear-Right (RR) [PWM 4,5]  
     ⚙️                           ⚙️
```

### **Movement Patterns:**
- **Forward**: All wheels forward
- **Backward**: All wheels backward
- **Spin Left**: Left wheels backward, right wheels forward
- **Spin Right**: Left wheels forward, right wheels backward
- **Strafe Left**: FL & RR backward, FR & RL forward
- **Strafe Right**: FL & RR forward, FR & RL backward

## 📋 **Pre-Testing Checklist**

### **Hardware Checks:**
- [ ] Robot is on a stable, flat surface
- [ ] Adequate clearance (2+ meters) in all directions
- [ ] Power supply is stable and adequate
- [ ] All wheels can move freely
- [ ] Emergency stop method ready (web interface + keyboard)

### **Software Checks:**
- [ ] IMU is reading correctly
- [ ] Camera stream is working
- [ ] Web interface loads properly
- [ ] All safety controls respond

### **Safety Setup:**
- [ ] Clear the area of people, pets, and obstacles
- [ ] Have emergency stop readily accessible
- [ ] Start with lowest speed setting (10-20%)
- [ ] Test each movement direction separately

## 🧪 **Testing Procedure**

### **Step 1: Initial System Check**
```bash
# Start the enhanced dashboard
python3 camera_imu_motor_stream.py

# Verify in browser at http://your_pi_ip:5000
# - Camera feed active
# - IMU data updating
# - Motors show "DISABLED" status
```

### **Step 2: Safe Area Setup**
1. Place robot in open area with 2+ meter clearance
2. Ensure stable power connection
3. Have emergency stop methods ready

### **Step 3: Enable Motors (First Time)**
1. Click **🟢 ENABLE** button
2. Confirm safety warning
3. Status should change to "Motors ENABLED"
4. Set speed to **20%** for initial testing

### **Step 4: Movement Testing**
1. **Test each direction individually:**
   - Press and hold **⬆️ FWD** briefly
   - Observe robot movement
   - Test **🛑 E-STOP** immediately
   
2. **If movement is correct:**
   - Test each direction: FWD, BACK, LEFT, RIGHT
   - Test strafe movements: STR-L, STR-R
   - Gradually increase speed as confidence builds

3. **If movement is incorrect:**
   - **🛑 E-STOP** immediately
   - **⚫ DISABLE** motors
   - Check hardware connections

## 🚨 **Emergency Procedures**

### **If Robot Moves Unexpectedly:**
1. **Press 🛑 E-STOP** (web interface)
2. **Press SPACEBAR** (keyboard shortcut)
3. **Disconnect power** if necessary
4. **⚫ DISABLE** motors when safe

### **If Web Interface Becomes Unresponsive:**
1. **Press Ctrl+C** in terminal running the script
2. **Disconnect robot power** if necessary
3. **Check network connection**

### **If Robot Doesn't Stop:**
1. **Disconnect power immediately**
2. **Check PCA9685 connections**
3. **Restart system before retesting**

## 🔍 **Troubleshooting**

### **Motors Don't Respond:**
- Check "Motor Status" shows "ENABLED"
- Verify PCA9685 connections and power
- Check console for error messages
- Ensure adequate power supply

### **Wrong Movement Direction:**
- Check wheel orientation
- Verify motor wire connections
- Consult customMotor.py for correct mapping

### **Jerky or Inconsistent Movement:**
- Check power supply stability
- Reduce speed setting
- Check for loose connections

## 📊 **Performance Monitoring**

### **Built-in Safety Features:**
- **Automatic disable** on page close/refresh
- **Movement timeout** if connection lost
- **Speed limiting** (10-100% range)
- **Emergency stop** always accessible

### **Status Monitoring:**
- Motor status indicator (ENABLED/DISABLED)
- Real-time speed display
- Movement direction feedback
- System status endpoint: `/status`

## 🎯 **Best Practices**

### **For Testing:**
1. **Always start with low speed** (10-20%)
2. **Test one movement at a time**
3. **Keep sessions short** initially
4. **Monitor for overheating** during extended use

### **For Operation:**
1. **Never leave enabled unattended**
2. **Disable when not actively controlling**
3. **Regular hardware checks**
4. **Monitor battery levels** (via IMU display)

### **For Development:**
1. **Test hardware separately** before integration
2. **Use safety delays** in automated sequences
3. **Implement additional safety checks** as needed
4. **Log all motor commands** for debugging

---

## ⚡ **Quick Start Command**

```bash
# SAFE way to start (motors disabled by default):
python3 camera_imu_motor_stream.py

# Access dashboard: http://your_pi_ip:5000
# Enable motors only when ready to test!
```

**Remember: Safety first! Motors can cause damage or injury if not used properly.** 🤖⚠️ 