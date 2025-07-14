# VocalGem Refactoring Documentation

## Overview

The `vocal_gemini.py` file has been successfully refactored into a modular structure to improve maintainability and code organization. The original 1892-line monolithic file has been broken down into focused, single-responsibility modules within the `vocalgem_modules/` package.

## New Module Structure

### Core Modules (in `vocalgem_modules/` package)

1. **`config.py`** - Configuration constants and settings
   - Audio constants (sample rates, formats, etc.)
   - Gemini API configuration
   - Camera and servo settings
   - Display configuration
   - Environment variables

2. **`display_manager.py`** - Display initialization and GPIO management
   - Display initialization with fallback strategies
   - GPIO cleanup and resource management
   - Display status operations

3. **`audio_manager.py`** - Audio stream management and PyAudio operations
   - Audio stream setup and management
   - Audio data processing and resampling
   - Gemini audio communication

4. **`function_handler.py`** - Tool function implementations
   - All Gemini tool functions (get_time, get_date, etc.)
   - Camera movement control
   - Battery level monitoring
   - Display brightness control

5. **`vision_manager.py`** - Camera and vision feed management
   - Camera initialization and configuration
   - Vision feed to Gemini
   - Face tracking integration

6. **`gemini_client.py`** - Gemini API client and session management
   - Gemini client initialization
   - Session creation and management
   - API connection handling

### Main Script

7. **`main.py`** - Simplified main script that orchestrates everything
   - Coordinates all modules
   - Manages the main event loop
   - Handles cleanup and error recovery

### Package Structure

```
vocalgem_modules/
├── __init__.py          # Package initialization
├── config.py            # Configuration constants
├── display_manager.py   # Display and GPIO management
├── audio_manager.py     # Audio stream management
├── function_handler.py  # Tool functions
├── vision_manager.py    # Camera and vision
└── gemini_client.py     # Gemini API client
```

### Existing Modules (Unchanged)

- **`display_animator.py`** - Display animations and LED control
- **`face_tracker.py`** - Face detection and tracking

## Benefits of Refactoring

### 1. **Improved Maintainability**
- Each module has a single responsibility
- Easier to locate and fix specific issues
- Clear separation of concerns

### 2. **Better Code Organization**
- Related functionality is grouped together
- Reduced cognitive load when working on specific features
- Easier to understand the overall architecture

### 3. **Enhanced Testability**
- Individual modules can be tested in isolation
- Easier to mock dependencies
- Better unit test coverage

### 4. **Simplified Debugging**
- Issues can be isolated to specific modules
- Clearer error messages and stack traces
- Easier to add logging and debugging

### 5. **Reduced File Size**
- Main script is now only ~200 lines instead of 1892 lines
- Each module is focused and manageable
- Easier to navigate in IDEs

## Setup and Installation

### Virtual Environment Setup

The application requires a properly configured virtual environment:

```bash
# Create virtual environment (if not already created)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show venv paths)
which python  # Should show: /path/to/venv/bin/python
which pip     # Should show: /path/to/venv/bin/pip

# Install dependencies
pip install -r requirements.txt
```

**Important**: Always ensure the virtual environment is activated before running the application. You should see `(venv)` in your terminal prompt and `which python` should point to the venv directory.

### Dependencies

Key dependencies include:
- `google-generativeai` - Gemini API client
- `websockets` - WebSocket communication
- `pyaudio` - Audio processing
- `picamera2` - Camera interface
- `gpiozero` - GPIO control
- `opencv-python` - Computer vision
- `numpy` - Numerical processing

## Usage

### Running the Application

The application can be run with the virtual environment activated:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the wake word detection system
python3 wake_porcu.py

# Or run the main application directly
python3 main.py
```

### Module Dependencies

The modules have the following dependency structure:

```
main.py
├── vocalgem_modules/
│   ├── config.py
│   ├── display_manager.py
│   ├── audio_manager.py
│   ├── function_handler.py
│   ├── vision_manager.py
│   └── gemini_client.py
├── display_animator.py
└── face_tracker.py
```

### Configuration

All configuration is centralized in `vocalgem_modules/config.py`. To modify settings:

1. **Audio settings**: Modify constants in `config.py`
2. **Gemini settings**: Update `MODEL` and `CONFIG` in `config.py`
3. **Camera settings**: Adjust `CAMERA_RESOLUTION` and related constants
4. **Servo settings**: Update pan/tilt limits in `config.py`

## Migration Notes

### What Changed

1. **Import structure**: The main application now imports from `vocalgem_modules` package
2. **Class structure**: The `AudioHandler` class has been replaced with `VocalGemRobot`
3. **Module organization**: Functionality is now distributed across focused modules in a package
4. **Virtual environment**: Proper venv activation is now required

### What Stayed the Same

1. **All functionality**: Every feature from the original code is preserved
2. **API compatibility**: The `run_gemini()` function maintains the same interface
3. **Configuration**: All settings and constants are preserved
4. **Dependencies**: All external dependencies remain the same

### Backward Compatibility

The refactoring maintains full backward compatibility:
- `wake_porcu.py` continues to work without changes
- All existing functionality is preserved
- No breaking changes to the public API

## Development Guidelines

### Adding New Features

1. **New tool functions**: Add to `vocalgem_modules/function_handler.py`
2. **New audio features**: Extend `vocalgem_modules/audio_manager.py`
3. **New display features**: Extend `vocalgem_modules/display_manager.py`
4. **New vision features**: Extend `vocalgem_modules/vision_manager.py`

### Modifying Configuration

1. **Audio settings**: Update constants in `vocalgem_modules/config.py`
2. **Gemini settings**: Modify `CONFIG` object in `vocalgem_modules/config.py`
3. **Hardware settings**: Update relevant constants in `vocalgem_modules/config.py`

### Debugging

1. **Audio issues**: Check `audio_manager.py` logs
2. **Display issues**: Check `display_manager.py` logs
3. **Function issues**: Check `function_handler.py` logs
4. **Vision issues**: Check `vision_manager.py` logs

## Testing Results

The refactored application has been successfully tested and verified to work correctly:

✅ **Virtual environment activation** - Properly configured and working  
✅ **Module imports** - All modules import correctly from the package  
✅ **Display initialization** - GPIO and display systems working  
✅ **Audio system** - PyAudio and audio processing functional  
✅ **Camera system** - IMX500 camera initializing and working  
✅ **Gemini client** - Live connection to Gemini API established  
✅ **Face tracking** - Face detection system operational  
✅ **Display animations** - All GIF animations loading correctly  

## File Size Comparison

| File | Lines | Purpose |
|------|-------|---------|
| `vocal_gemini.py` (original) | 1892 | Monolithic main file |
| `main.py` (new) | ~200 | Simplified orchestrator |
| `vocalgem_modules/config.py` | ~150 | Configuration |
| `vocalgem_modules/display_manager.py` | ~430 | Display management |
| `vocalgem_modules/audio_manager.py` | ~420 | Audio management |
| `vocalgem_modules/function_handler.py` | ~180 | Tool functions |
| `vocalgem_modules/vision_manager.py` | ~180 | Vision management |
| `vocalgem_modules/gemini_client.py` | ~50 | Gemini client |

**Total new structure**: ~1610 lines across 8 focused files in organized package structure

## Troubleshooting

### Virtual Environment Issues

If you see system Python paths instead of venv paths:
```bash
# Check if venv is activated
which python  # Should show venv path, not /usr/bin/python

# If not activated properly, reactivate:
deactivate
source venv/bin/activate
```

### Import Errors

If you encounter import errors:
1. Ensure virtual environment is activated
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Check that `vocalgem_modules/__init__.py` exists

### Audio Warnings

ALSA warnings are common on Raspberry Pi and don't affect functionality. These can be safely ignored.

## Conclusion

The refactoring has been **completely successful**! The large, monolithic file has been transformed into a well-organized, modular structure while preserving all functionality. The codebase is now much more maintainable and easier to work with for future development.

**Status**: ✅ **COMPLETE AND TESTED** 