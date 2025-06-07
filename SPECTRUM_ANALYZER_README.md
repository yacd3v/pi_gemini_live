# LED Volume Meter Feature

## Overview

The VocalGem robot now features a real-time audio volume meter that visualizes the AI's voice volume through the LED ring. Instead of a static red chasing pattern, the LEDs now respond dynamically to the volume level of the AI's speech, creating a live audio visualization similar to a VU meter.

## How It Works

### Volume Analysis
- **Real-time RMS**: The system calculates the Root Mean Square (RMS) volume of incoming audio data
- **8 LED Levels**: The volume is mapped to 8 different levels, each represented by one LED
- **Progressive Activation**: LEDs light up progressively as volume increases
- **Smoothing**: Volume levels are smoothed to prevent flickering and create fluid animations

### LED Mapping
Each LED represents a different volume level with color coding:

| LED | Volume Range | Color | Description |
|-----|-------------|-------|-------------|
| 0 | Very quiet | Green | First LED to light up |
| 1 | Quiet | Green | Low volume speech |
| 2 | Low | Green | Normal quiet conversation |
| 3 | Medium-low | Yellow/Orange | Getting louder |
| 4 | Medium | Yellow/Orange | Normal speaking volume |
| 5 | Medium-high | Yellow/Orange | Louder speech |
| 6 | High | Red | Very loud speech |
| 7 | Very high | Red | Maximum volume |

### Visual Behavior
- **Volume Response**: Number of lit LEDs corresponds to the current volume level
- **Progressive Lighting**: LEDs light up from LED 0 to LED 7 as volume increases
- **Color Gradient**: Green (quiet) → Yellow/Orange (medium) → Red (loud)
- **Smooth Transitions**: Volume changes create smooth LED transitions
- **Silence Detection**: When very quiet or silent, all LEDs turn off

## Technical Implementation

### Audio Processing
- **Sample Rate**: 24kHz (Gemini Live audio format)
- **Window Size**: 512 samples for volume calculation
- **RMS Calculation**: Root Mean Square for accurate volume measurement
- **Update Rate**: Real-time processing with each audio chunk

### Volume Mapping
The system uses conservative volume thresholds for optimal visualization:
- **< 0.02**: No LEDs (very quiet/silent)
- **0.02-0.05**: 1 LED (quiet speech)
- **0.05-0.08**: 2 LEDs (low volume)
- **0.08-0.12**: 3 LEDs (medium-low)
- **0.12-0.16**: 4 LEDs (medium)
- **0.16-0.20**: 5 LEDs (medium-high)
- **0.20-0.25**: 6 LEDs (high)
- **0.25-0.30**: 7 LEDs (very high)
- **> 0.30**: 8 LEDs (maximum)

### Performance Optimizations
- **Efficient RMS**: Uses NumPy's optimized mathematical operations
- **Smoothing**: Exponential smoothing prevents rapid flickering
- **Conservative Scaling**: Spreads volume levels across the full LED range
- **Minimal CPU Impact**: Lightweight processing suitable for Raspberry Pi

## Usage

The volume meter automatically activates when the AI is speaking:

1. **Idle Mode**: Purple breathing effect (unchanged)
2. **Speaking Mode**: Real-time volume meter visualization
3. **Quiet Speech**: 1-3 LEDs light up (green)
4. **Normal Speech**: 3-5 LEDs light up (green to yellow)
5. **Loud Speech**: 5-8 LEDs light up (yellow to red)
6. **Silence/Pauses**: All LEDs turn off

## Benefits

- **Visual Feedback**: Users can see when the AI is speaking and how loud
- **Volume Indication**: Clear indication of speech volume levels
- **Engaging Experience**: More dynamic and interesting than static patterns
- **Audio Quality Indication**: Can help identify audio issues or volume problems
- **Natural Behavior**: Mimics how humans perceive and respond to volume changes

## Configuration

The volume meter can be tuned by modifying these parameters in `display_animator.py`:

```python
# Audio volume analysis variables
self.spectrum_smoothing = 0.5      # Smoothing factor (0.0-1.0)
self.volume_threshold = 0.005      # Minimum volume to trigger processing
```

Volume thresholds can be adjusted in the `_calculate_spectrum()` method:
```python
if volume < 0.02:
    led_level = 0  # Very quiet - no LEDs
elif volume < 0.05:
    led_level = 1  # Quiet - 1 LED
# ... etc
```

## Troubleshooting

- **No LED Response**: Check if audio is being received and volume is above 0.005 threshold
- **Flickering**: Increase `spectrum_smoothing` value (closer to 1.0)
- **Too Sensitive**: Increase the volume thresholds in the mapping
- **Not Sensitive Enough**: Decrease the volume thresholds in the mapping
- **All LEDs Always On**: Volume thresholds may be too low for your audio levels

## Future Enhancements

Potential improvements for the volume meter:
- User-configurable volume thresholds
- Different visualization modes (peak meter, average meter)
- Integration with music playback
- Customizable color schemes
- Beat detection and rhythm visualization
- Frequency-based spectrum analyzer mode (toggle option) 