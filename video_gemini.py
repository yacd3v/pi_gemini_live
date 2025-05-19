import asyncio
import io
import os
import sys
import time
from dotenv import load_dotenv

# Try to import Picamera2; if unavailable, we'll fall back to OpenCV
try:
    from picamera2 import Picamera2, Preview
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False

import cv2  # Safe to import even if we later use Picamera2
from PIL import Image

from google import genai
from google.genai import types

# ─── Constants ──────────────────────────────────────────────────────────────
MODEL = "models/gemini-2.0-flash-live-001"

# Load secrets
load_dotenv()


def capture_single_frame() -> Image.Image:
    """Capture a single frame from the first available camera.

    Preference order:
    1. Picamera2 (recommended for Raspberry Pi 5)
    2. OpenCV VideoCapture(0)

    Returns:
        A PIL.Image in RGB format.

    Raises:
        RuntimeError: if no camera frame could be captured.
    """
    if _PICAMERA2_AVAILABLE:
        picam = Picamera2()
        # Explicitly set preview size
        config = picam.create_still_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            buffer_count=3)

        picam.configure(config)

        picam.start_preview(Preview.NULL)


        picam.start()
        print("[Camera] Picamera2 started, warming up...")
        # Keep increased warmup time just in case
        time.sleep(1) 
        print("[Camera] Warmup complete, capturing frame...")
        # Simplify capture call for preview config
        frame = picam.capture_array() 
        # Metadata might not be available or relevant for preview, comment out for now
        # metadata = picam.capture_metadata()
        print("[Camera] Frame captured, stopping camera...")
        picam.stop()

        print("[Camera] Captured frame via Picamera2")
        print(f"[Camera] Resolution: {frame.shape[1]}×{frame.shape[0]}")
        # print(f"[Camera] Metadata keys: {list(metadata.keys())[:10]} …")

        # Picamera2 returns frames in RGB order already
        img_rgb = frame
    else:
        print("[Camera] Picamera2 not available – falling back to OpenCV.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Unable to open camera with OpenCV.")
        # Optional: set a smaller frame size for quick tests
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("[Camera] OpenCV camera started, capturing frame...")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture image with OpenCV.")

        print("[Camera] Captured frame via OpenCV")
        print(f"[Camera] Resolution: {frame.shape[1]}×{frame.shape[0]}")

        # OpenCV returns BGR; convert to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print("[Camera] Converting frame to PIL.Image...")
    pil_img = Image.fromarray(img_rgb)
    print("[Camera] PIL.Image conversion complete.")
    return pil_img


async def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("[Error] GOOGLE_API_KEY environment variable not set. Exiting.")
        return

    # 1. Capture frame
    try:
        print("[Main] Capturing frame...")
        pil_img = capture_single_frame()
        print("[Main] Frame captured.")
    except Exception as e:
        print(f"[Error] {e}")
        return

    # 2. Prepare Gemini Live config (only need TEXT response for this test)
    # Changed CONFIG to a dictionary to match example and potentially resolve header issue
    CONFIG = {
        "response_modalities": ["TEXT"],
        # "session_resumption": {"handle": None} # session_resumption might not be needed or compatible this way
    }

    print("[Main] Creating Gemini client...")
    client = genai.Client(http_options={"api_version": "v1beta"})

    # 3. Connect and send the image
    print("[Main] Connecting to Gemini Live API...")
    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
        print("[Gemini] Connected. Preparing to send image…")
        await asyncio.sleep(0.1)  # Just to ensure log order
        
        # 1. Convert PIL Image to JPEG bytes
        print("[Gemini] Converting PIL Image to JPEG bytes...")
        image_io = io.BytesIO()
        pil_img.save(image_io, format="JPEG")
        image_bytes = image_io.getvalue() # Use getvalue() for bytes
        image_io.close()

        # 2. Create types.Part for the image (This part will be unused if using send_realtime_input directly with PIL image)
        # print("[Gemini] Creating types.Part for the image...")
        # image_part = types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes))

        # 3. Send using send_realtime_input with the PIL image directly
        print("[Gemini] Sending PIL image using send_realtime_input...")
        await session.send_realtime_input(media=pil_img) # Reverted to send_realtime_input
        print("[Gemini] Image data sent. Waiting for response...")

        # 4. Wait for a single response chunk (text only expected)
        try:
            async for resp in session.receive():
                print(f"[Gemini] Received raw response object: {resp}")

                if resp.error:
                    print(f"[Gemini] Error in response: {resp.error}")
                    break  # Stop looping on error in the response

                if resp.text:
                    print("[Gemini] Response text:", resp.text)
                
                # Log other potential fields for more insight
                if resp.speech:
                    print(f"[Gemini] Speech response detected (first 50 chars): {str(resp.speech)[:50]}...")
                if resp.media_action:
                    print(f"[Gemini] Media action detected: {resp.media_action}")

                if resp.server_content:
                    print(f"[Gemini] Server content: {resp.server_content}")
                    if resp.server_content.turn_complete:
                        print("[Gemini] Turn complete indicated by server_content.")
                        break  # Exit loop as turn is complete
                
                # Fallback checks (though resp.error or turn_complete should handle most cases)
                if not session.is_active:
                    print("[Gemini] Session is_active is False. Exiting loop.")
                    break
                if session.error: # This might be redundant if resp.error or other mechanisms cover it
                    print(f"[Gemini] Session error property indicates an issue: {session.error}. Exiting loop.")
                    break

        except Exception as e:
            print(f"[Gemini] Unexpected error during receive loop: {e}")
            print(f"[Gemini] Type of error: {type(e)}")
            # Log session error state if an exception occurs in the loop
            if hasattr(session, 'error') and session.error:
                 print(f"[Gemini] Session error state after exception: {session.error}")
            # break # This break is not needed as exception exits the try block

    print("[Gemini] Loop finished or exited due to error/completion.")
    print("[Gemini] Session closing...") # Moved this log to be before actual close
    # The 'async with' block will handle closing the session automatically.
    # print("[Gemini] Session closed.") # This will be printed after the 'async with' block exits

    if hasattr(session, 'error') and session.error: # Check if session object still exists and has error
        print(f"[Gemini] Final session error state after loop: {session.error}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
