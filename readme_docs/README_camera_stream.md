# IMX500 Camera Web Stream

A simple web interface to stream video from your Raspberry Pi 5's IMX500 camera over the local network.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_stream.txt
   ```

2. **Run the stream server:**
   ```bash
   python3 camera_web_stream.py
   ```

3. **Access the stream:**
   - On the Pi: `http://localhost:5000`
   - From other devices: `http://YOUR_PI_IP:5000`

## Features

- ✅ Live video streaming from IMX500 camera
- ✅ Web interface accessible from any device on your network
- ✅ Real-time timestamp overlay
- ✅ Automatic fallback to regular camera if IMX500 fails
- ✅ Clean, responsive web interface
- ✅ Status endpoint for monitoring

## Find Your Pi's IP Address

```bash
hostname -I
```

## Stop the Server

Press `Ctrl+C` in the terminal running the script.

## Troubleshooting

- **Camera not working:** Make sure your IMX500 camera is properly connected
- **Can't access from other devices:** Check that port 5000 isn't blocked by firewall
- **Poor performance:** Try closing other applications using the camera

## Technical Details

- **Resolution:** 640x480
- **Format:** MJPEG streaming
- **Frame Rate:** ~30 FPS
- **Port:** 5000 (HTTP) 