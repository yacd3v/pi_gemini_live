# Enhanced Face Tracking Debug System

## Overview

This enhanced face tracking system provides comprehensive debugging capabilities and improved tracking algorithms for the VocalGem robot. The system generates detailed debug images showing exactly what the tracking system is seeing and doing.

## Key Improvements

### 1. Enhanced Tracking Algorithm
- **Adaptive Speed Control**: Tracking speed adjusts based on distance from target
- **Velocity-Based Smoothing**: Reduces choppy movements with velocity tracking
- **Exponential Weighting**: Recent positions are weighted more heavily for smoother tracking
- **Adaptive Deadzone**: Deadzone adjusts based on recent movement to prevent oscillation

### 2. Comprehensive Debug Output
- **Visual Debug Images**: Show face detections, keypoints, tracking history, and servo status
- **Tracking Statistics**: Monitor detection rates, tracking efficiency, and movement patterns
- **Movement History**: Track recent servo movements for analysis
- **Real-time Status**: Display current tracking state and parameters

### 3. Improved Face Detection
- **Weighted Center Calculation**: Use confidence-weighted keypoint averaging
- **Geometric Validation**: Validate face detections for realistic facial geometry
- **Sub-pixel Accuracy**: Gaussian smoothing for more precise keypoint detection
- **Dynamic Thresholds**: Adjust thresholds based on keypoint importance

## File Structure

```
debug/
├── tracking/     # Debug images with comprehensive tracking info
├── frames/       # Raw camera frames for comparison
├── movements/    # Movement-specific debug images
└── stats/        # Tracking statistics and performance reports
```

## How to Use

### 1. Testing the Enhanced System

Run the debug test script to generate comprehensive debug images:

```bash
python test_debug_tracking.py
```

This will:
- Create debug folders
- Initialize the enhanced face tracker
- Capture frames with face detection
- Generate debug images every 10 frames
- Save tracking statistics
- Run for 30 seconds (or until Ctrl+C)

### 2. Running the Main System

The enhanced tracking is automatically integrated into the main system. Simply run:

```bash
python vocal_gemini.py
```

The system will now:
- Generate debug images when faces are detected
- Save images to `debug/tracking/` folder
- Show comprehensive tracking information
- Use improved smoothing algorithms

### 3. Understanding Debug Images

Debug images show:

#### Visual Elements
- **Green crosshair**: Image center target
- **Gray rectangle**: Tracking deadzone
- **Red circles**: Detected faces with confidence
- **Colored keypoints**: Facial features (nose=blue, eyes=cyan, ears=yellow)
- **Purple line**: Error vector from face to center
- **Orange dots**: Tracking history (numbered 0-4)

#### Text Information
- **Servo Status**: Current pan/tilt angles and limits
- **Velocity Info**: Current servo velocities
- **Tracking Status**: Whether auto-tracking is active
- **Statistics**: Detection counts, tracking efficiency
- **Error Values**: Normalized error coordinates

### 4. Key Parameters

You can adjust these parameters in `face_tracker.py`:

```python
# Tracking responsiveness
self.face_tracking_interval = 0.2  # Detection frequency (seconds)
self.manual_movement_cooldown = 3.0  # Auto-tracking pause after manual movement

# Smoothing parameters
self.tracking_deadzone = 0.015  # Smaller = more sensitive
self.max_move_per_step = 15  # Maximum degrees per movement
self.min_move_threshold = 0.3  # Minimum movement to execute

# Speed control
self.tracking_speed_close = 0.4  # Speed when close to target
self.tracking_speed_far = 1.0   # Speed when far from target
```

## Debugging Common Issues

### 1. Choppy Movement
- **Cause**: deadzone too small or tracking speed too high
- **Solution**: Increase `tracking_deadzone` or reduce tracking speeds
- **Debug**: Look for rapid oscillations in debug images

### 2. Slow Response
- **Cause**: deadzone too large or tracking speed too low
- **Solution**: Decrease `tracking_deadzone` or increase tracking speeds
- **Debug**: Check if faces are detected but not tracked

### 3. Servo Offset
- **Cause**: Incorrect servo limits or center positions
- **Solution**: Adjust `pan_limits`, `tilt_limits`, or center positions
- **Debug**: Check servo angle readouts in debug images

### 4. Poor Detection
- **Cause**: Low confidence threshold or poor lighting
- **Solution**: Adjust `confidence_threshold` or improve lighting
- **Debug**: Look for missing or weak keypoints

## Performance Analysis

### Tracking Statistics
The system tracks:
- **Total Detections**: How many faces were detected
- **Successful Tracks**: How many resulted in servo movement
- **Movements Executed**: Total servo movements
- **Average Confidence**: Mean detection confidence
- **Tracking Efficiency**: Success rate percentage

### Movement History
Recent movements are tracked with:
- **Timestamp**: When movement occurred
- **Pan/Tilt Diff**: Degrees moved
- **Error Values**: Target error that triggered movement
- **Tracking Speed**: Adaptive speed used

## Advanced Features

### 1. Geometric Validation
Faces are validated for realistic geometry:
- Eyes must be roughly horizontal
- Nose should be between eyes
- Keypoints must form coherent face structure

### 2. Adaptive Deadzone
The deadzone increases if the system is moving frequently, preventing oscillation in noisy conditions.

### 3. Velocity Tracking
Servo velocities are tracked and used for smoothing, reducing abrupt direction changes.

### 4. Sub-pixel Accuracy
Keypoint positions are refined using Gaussian weighting for more precise tracking.

## Manual Control Integration

The enhanced system properly handles manual camera movements:
- Auto-tracking pauses for 3 seconds after manual movement
- Tracking history is cleared after manual movement
- Velocity tracking is reset to prevent conflicts

## Function Integration

The system integrates with the voice assistant's camera control functions:
- `move_camera()`: Triggers debug image generation
- `toggle_face_tracking()`: Enables/disables enhanced tracking
- Debug images show manual movement commands

## Troubleshooting

### Common Error Messages
- **"IMX500 model not found"**: Check that the AI model is installed
- **"Servo initialization failed"**: Check servo connections and power
- **"Camera initialization error"**: Verify camera connection and permissions

### Performance Issues
- **High CPU usage**: Reduce detection frequency or image resolution
- **Memory usage**: Clear debug folders periodically
- **Servo jerky movement**: Increase smoothing parameters

## Next Steps

To further improve the system:

1. **Tune Parameters**: Adjust tracking parameters based on debug output
2. **Add Filters**: Implement Kalman filtering for even smoother tracking
3. **Multiple Faces**: Add support for tracking multiple faces
4. **Predictive Tracking**: Add motion prediction for better tracking
5. **Auto-calibration**: Automatically adjust parameters based on performance

The debug system provides all the information needed to identify and fix tracking issues. Use the debug images to understand what the system is seeing and adjust parameters accordingly. 