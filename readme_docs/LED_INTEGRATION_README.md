# LED Integration for Vocal Gemini Robot

This document describes the LED integration added to the vocal_gemini.py robot assistant using the Freenove kit's LED capabilities.

## Features

The robot now includes three distinct LED animation modes:

### 1. **Initialization Sequence** 🔵
- **When**: During robot startup, after connecting to Gemini
- **Effect**: LEDs light up one by one in blue until the full ring is illuminated
- **Purpose**: Visual indication that the robot is ready to listen

### 2. **Idle Mode** 💜  
- **When**: When the robot is listening and waiting for input
- **Effect**: Gentle purple breathing effect (fading in and out)
- **Purpose**: Ambient indication that the robot is active and listening

### 3. **Speaking Mode** 🔴
- **When**: When the robot is responding/speaking
- **Effect**: Fast red chasing light around the ring (like Kitt from Knight Rider)
- **Purpose**: Visual indication that the robot is actively responding

## Testing the LED Integration

Before running the full vocal_gemini.py application, you can test the LED integration:

```bash
python3 test_led_integration.py
```

This test script will:
1. Initialize the LED controller
2. Run through all three animation modes
3. Verify each LED effect works correctly
4. Turn off the LEDs when complete

## How It Works

### Code Structure
- LED control is integrated into the `DisplayAnimator` class
- Each animation mode has its own method:
  - `led_initialization_sequence()`: Blue startup sequence
  - `led_idle_breathing()`: Purple breathing effect  
  - `led_speaking_chase()`: Red chasing effect
- LEDs are updated at a slower rate than the display to avoid overwhelming the system

### Integration Points
- **Initialization**: `run_initialization()` called after Gemini connection
- **Mode Changes**: LED animations automatically follow display animation modes
- **Cleanup**: LEDs are turned off when the program exits

## Configuration

### LED Strip Settings
- Assumes 8 LEDs in the ring (standard Freenove configuration)
- Compatible with both RPI and SPI LED control methods
- Automatic detection based on PCB and Raspberry Pi versions

### Animation Timing
- **Initialization**: 200ms delay between LED activations
- **Idle breathing**: Updates every 3 display frames (~200ms at 15fps)
- **Speaking chase**: Updates every 3 display frames with faster movement

### Colors Used
- **Blue (Initialization)**: RGB(0, 0, 255)
- **Purple (Idle)**: RGB(128, 0, 128) with breathing intensity
- **Red (Speaking)**: RGB(255, 0, 0) with trailing dimmed effect

## Troubleshooting

### Common Issues

1. **"LED initialization failed"**
   - Check that the Freenove LED strip is properly connected
   - Verify I2C/SPI permissions are set correctly
   - Ensure the freenove_examples directory is present

2. **LEDs not responding**
   - Test with the original Freenove LED examples first
   - Check power connections to the LED strip
   - Verify PCB version detection in parameter.py

3. **Inconsistent animations**
   - LED animations are designed to be graceful with failures
   - Check console output for specific error messages
   - Ensure sufficient power supply for all LEDs

### Fallback Behavior
- If LED initialization fails, the robot continues to work normally
- All LED methods include error handling and graceful degradation
- Display animations continue independently of LED status

## Customization

### Changing Colors
Modify the RGB values in the respective methods:
```python
# For purple breathing (idle)
r = int(128 * self.led_breathing_brightness / 255)  # Red component
g = 0                                               # Green component  
b = int(128 * self.led_breathing_brightness / 255)  # Blue component
```

### Adjusting Animation Speed
- **Initialization**: Change `await asyncio.sleep(0.2)` in `run_initialization()`
- **Breathing**: Modify the `+= 5` increment in `led_idle_breathing()`
- **Chase**: Adjust the sleep time in the speaking mode calls

### Adding New Modes
1. Create a new method in `DisplayAnimator` class
2. Add the mode condition in the `run()` method's LED update section
3. Call the new mode when needed in your application logic

## Dependencies

- `freenove_examples/led.py`: LED control library
- `freenove_examples/parameter.py`: Hardware detection
- Compatible hardware configuration (PCB v1/v2, RPi versions)

## Integration with Voice Commands

The LED system automatically responds to the robot's state:
- Connects during Gemini initialization
- Switches to idle when listening
- Activates speaking mode during responses
- Supports all emotion modes (normal, furious, crying)

No additional voice commands are needed - the LEDs follow the robot's natural conversation flow! 