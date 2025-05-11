import asyncio
import os
import sys
import time
import traceback
import queue
import wave
import audioop
from dotenv import load_dotenv


# Suppress ALSA and JACK warnings
sys.stderr = open(os.devnull, 'w')

# Load environment variables from .env file
load_dotenv()

import pyaudio
from google import genai
from google.genai import types

# Add display imports
import spidev as SPI
sys.path.append("display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/..")
from lib import LCD_1inch28
from PIL import Image, ImageDraw, ImageFont, ImageSequence

# ─── Audio constants ──────────────────────────────────────────────────────────
FORMAT = pyaudio.paInt16
IN_CH = 1   #  use processed mono channel
OUT_CH = 1   # we still play/record mono
CHUNK_MS = 20             # one packet = 20 ms of audio
SEND_SAMPLE_RATE = 16_000     # Gemini Live input format
RECEIVE_SAMPLE_RATE = 24_000  # Gemini Live output format
AEC_SAMPLE_RATE = 16_000          
RAW_CH = 6

# ─── Gemini constants ────────────────────────────────────────────────────────
MODEL = "gemini-2.0-flash-live-001"

# Example functions that can be called during conversation
def get_time(**_):
    """Get the current time."""
    current_time = time.strftime("%H:%M:%S")
    return f"The current time is {current_time}"

def get_date(**_):
    """Get today's date."""
    current_date = time.strftime("%Y-%m-%d")
    return f"Today's date is {current_date}"

def set_display_brightness(brightness: float, **_):
    """Set the display brightness.
    
    Args:
        brightness: A value between 0.0 and 1.0 for display brightness
    """
    if brightness < 0 or brightness > 1:
        return "Brightness must be between 0 and 1"
    return f"Display brightness set to {brightness*100}%"

# List of available functions
available_functions = [get_time, get_date, set_display_brightness]

# Create a dictionary to map function names to function objects
functions_map = {func.__name__: func for func in available_functions}

# Define tools for the LiveConnectConfig
tools_for_config = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_time",
            description="Get the current time.",
            parameters=types.Schema(type='OBJECT', properties={})
        ),
        types.FunctionDeclaration(
            name="get_date",
            description="Get today's date.",
            parameters=types.Schema(type='OBJECT', properties={})
        ),
        types.FunctionDeclaration(
            name="set_display_brightness",
            description="Set the display brightness.",
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'brightness': types.Schema(type='NUMBER', description="A value between 0.0 and 1.0 for display brightness")
                },
                required=['brightness']
            )
        )
    ])
]

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    session_resumption=types.SessionResumptionConfig(handle=None),
    speech_config=types.SpeechConfig(
        language_code="en-US",
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    system_instruction=types.Content(
        parts=[
            types.Part(
                text="You are Karl, a sarcastic and very funny robot. You are very smart and helpful for the people interacting with you."
            )
        ]
    ),
    tools=tools_for_config  # Use the new tools configuration
)

class DisplayAnimator:
    """Plays animated GIFs on the 240×240 Waveshare screen."""

    def __init__(self, disp, fps=15):
        self.disp = disp
        self.fps = fps
        self._frames = {"idle": [], "speak": []}  # Initialize before loading
        self._idx = 0
        
        # Pre-load all animations
        self._load_gif("idle")
        self._load_gif("speak")

        self.mode = "idle"           # Set initial mode after all GIFs are loaded

    def _load_gif(self, mode):
        """Load GIF animation for the specified mode."""
        gif_path = f"GIFAnimations/animation_{mode}.gif"
        try:
            with Image.open(gif_path) as gif:
                self._frames[mode] = []
                for frame in ImageSequence.Iterator(gif):
                    # Convert frame to RGB if needed
                    current_frame = frame
                    if current_frame.mode != 'RGB':
                        current_frame = current_frame.convert('RGB')
                    # Rotate frame 180 degrees
                    current_frame = current_frame.rotate(180)
                    self._frames[mode].append(current_frame)
                # print(f"[DisplayAnimator] Loaded {len(self._frames[mode])} frames for mode '{mode}'. Path: {gif_path}")
        except Exception as e:
            print(f"[DisplayAnimator] Error loading GIF {gif_path}: {e}")
            # Create a blank frame as fallback
            blank = Image.new('RGB', (self.disp.width, self.disp.height), 'BLACK')
            self._frames[mode] = [blank]
            # print(f"[DisplayAnimator] Fallback: Loaded {len(self._frames[mode])} (blank) frame for mode '{mode}'.")

    def set_mode(self, new_mode: str):
        """Switch animation if needed; don't disturb current frame otherwise."""
        if new_mode != self.mode:
            # Ensure the requested mode has frames loaded
            if new_mode in self._frames and self._frames[new_mode]:
                self.mode = new_mode
                self._idx = 0          # start this sequence from its first frame
                # print(f"[DisplayAnimator] Switched to mode '{new_mode}', frame index reset.")
            else:
                print(f"[DisplayAnimator] Warning: Mode '{new_mode}' requested but no frames found. Staying in mode '{self.mode}'.")

    async def run(self):
        delay = 1 / self.fps
        while True:
            frames = self._frames[self.mode]
            if not frames:            # no frames yet
                await asyncio.sleep(delay)
                continue
            
            # print(f"[DisplayAnimator.run] Mode: {self.mode}, Index: {self._idx}, Total Frames: {len(frames)}")
            
            self.disp.ShowImage(frames[self._idx])
            self._idx = (self._idx + 1) % len(frames)
            await asyncio.sleep(delay)

class AudioHandler:
    """Handles local audio + Gemini Live session."""

    def __init__(self):
        # Event‑loop reference for thread‑safe calls from PyAudio callback
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # called in main thread before loop exists; fixed later in run()
            self.loop = None

        self.pya = pyaudio.PyAudio()

        # queues (200 packets ≈ 4 s @ 20 ms)
        self.audio_out_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self.audio_in_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

        # logging WAVs
        self.sent_wf = wave.open("sent_audio.wav", "wb")
        self.sent_wf.setnchannels(OUT_CH)
        self.sent_wf.setsampwidth(self.pya.get_sample_size(FORMAT))
        self.sent_wf.setframerate(SEND_SAMPLE_RATE)

        self.recv_wf = wave.open("recv_audio.wav", "wb")
        self.recv_wf.setnchannels(OUT_CH)
        self.recv_wf.setsampwidth(self.pya.get_sample_size(FORMAT))
        self.recv_wf.setframerate(RECEIVE_SAMPLE_RATE)

        # streams + rates
        self.input_stream = None
        self.output_stream = None
        self.input_rate: int | None = None
        self.output_rate: int | None = None

        # resampler states
        self.rs_in_state = None
        self.rs_out_state = None

        # Gemini session + client
        self.client = None
        self.session = None

        # Initialize display for status feedback
        self.disp = LCD_1inch28.LCD_1inch28()
        self.disp.Init()
        self.disp.clear()
        self.disp.bl_DutyCycle(50)
        self.font = ImageFont.truetype("display_examples/LCD_Module_RPI_code/RaspberryPi/python/example/../Font/Font01.ttf", 24)

        # Initialize display animator   
        self.anim = DisplayAnimator(self.disp)

    def _strip_to_processed(self, data: bytes) -> bytes:
        frame = RAW_CH * 2              # 12 bytes per frame
        out = bytearray(len(data) // RAW_CH)  # space for 2-byte mono
        o = 0
        for i in range(0, len(data), frame):
            out[o:o+2] = data[i:i+2]    # copy both bytes of ch-0
            o += 2
        return bytes(out)

    # ────────────────────────── device helpers ──────────────────────────────
    def _find_input_device(self) -> int:
        for idx in range(self.pya.get_device_count()):
            #info = self.pya.get_device_info_by_index(idx)
            #if info.get("maxInputChannels", 0) and (
            #    "respeaker" in info["name"].lower() or "seeed" in info["name"].lower()):
            #    return idx
            name = self.pya.get_device_info_by_index(idx)["name"].lower()
            if "respeaker" in name or "seeed" in name:
                return idx
        raise RuntimeError("ReSpeaker output not found")
    
        #info = self.pya.get_default_input_device_info()
        #return info["index"]

    def _find_output_device(self) -> int:          # ③ lock to ReSpeaker
        for idx in range(self.pya.get_device_count()):
            name = self.pya.get_device_info_by_index(idx)["name"].lower()
            if "respeaker" in name or "seeed" in name:
                return idx
        raise RuntimeError("ReSpeaker output not found")

    # ────────────────────────── PyAudio callback ────────────────────────────
    def _mic_callback(self, in_data, frame_count, time_info, status):
        """Runs in **PyAudio thread** – push data into asyncio queue."""
        if self.loop is None:
            return (None, pyaudio.paContinue)
        try:
            self.loop.call_soon_threadsafe(self.audio_out_q.put_nowait, in_data)
        except asyncio.QueueFull:
            # drop one packet – better than blocking and causing an overrun
            pass
        return (None, pyaudio.paContinue)

    # ────────────────────────── stream setup ────────────────────────────────
    async def setup_streams(self):
        # print("\nAvailable audio devices:")
        for i in range(self.pya.get_device_count()):
            info = self.pya.get_device_info_by_index(i)
            # print(f"  {i}: {info['name']}  (in {info['maxInputChannels']}, out {info['maxOutputChannels']}, {int(info['defaultSampleRate'])} Hz)")

        in_idx = self._find_input_device()
        out_idx = self._find_output_device()

        # input – prefer 16 kHz
        in_rates = [16_000]   # ReSpeaker native
        in_rates += [int(self.pya.get_device_info_by_index(in_idx)["defaultSampleRate"]), 48_000, 44_100]
        for r in in_rates:
            try:
                fpb = int(r * CHUNK_MS / 1000)
                self.input_stream = self.pya.open(format=FORMAT,
                                                  channels=RAW_CH,
                                                  rate=r,
                                                  input=True,
                                                  input_device_index=in_idx,   # still the same device
                                                  frames_per_buffer=fpb,
                                                  stream_callback=self._mic_callback)
                self.input_rate = r
                # print(f"Input stream opened at {r} Hz, fpb {fpb}")
                break
            except Exception as e:
                # print(f"  could not open at {r} Hz: {e}")
                pass
        if not self.input_stream:
            raise RuntimeError("No usable input stream")

        # output – use device default (first that works)
        #out_rates = [int(self.pya.get_device_info_by_index(out_idx)["defaultSampleRate"]), 48_000, 44_100]
        #out_rates = [AEC_SAMPLE_RATE, 48_000, 44_100]
        out_rates = [AEC_SAMPLE_RATE]
        
        for r in out_rates:
            try:
                fpb = int(r * CHUNK_MS / 1000)
                self.output_stream = self.pya.open(format=FORMAT,
                                                   channels=OUT_CH,
                                                   rate=r,
                                                   output=True,
                                                   output_device_index=out_idx,
                                                   frames_per_buffer=fpb)
                self.output_rate = r
                # print(f"Output stream opened at {r} Hz, fpb {fpb}")
                break
            except Exception as e:
                # print(f"  could not open output at {r} Hz: {e}")
                pass
        if not self.output_stream:
            raise RuntimeError("No usable output stream")

        self.input_stream.start_stream()

    # ────────────────────────── coroutines ──────────────────────────────────
    async def _send_to_gemini(self):
        print("Send‑loop started…")
        while True:
            data = await self.audio_out_q.get()

            data = self._strip_to_processed(data)       # ① use the AEC track only
            # down-mix 4-ch → mono before resample
            #if IN_CH != 1:
            #    data = audioop.tomono(data, 2, 0.25, 0.25)          # average L+R
            #    data = audioop.tomono(data, 2, 1, 0)                # collapse again
            if self.input_rate != SEND_SAMPLE_RATE:
                data, self.rs_in_state = audioop.ratecv(
                    data, 2, OUT_CH,
                    self.input_rate, SEND_SAMPLE_RATE,
                    self.rs_in_state)
            if self.session:
                blob = types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
                await self.session.send_realtime_input(media=blob)
            self.sent_wf.writeframes(data)
            self.audio_out_q.task_done()

    # Add helper to show status on display
    def _show_status(self, text):
        # Create an image with centered status text
        image = Image.new("RGB", (self.disp.width, self.disp.height), "BLACK")
        draw = ImageDraw.Draw(image)
        # Calculate text width and height using textbbox
        bbox = draw.textbbox((0, 0), text, font=self.font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (self.disp.width - w) // 2
        y = (self.disp.height - h) // 2
        draw.text((x, y), text, fill="WHITE", font=self.font)
        im_r = image.rotate(180)
        self.disp.ShowImage(im_r)

    async def _recv_from_gemini(self):
        print("Receive‑loop started…")

        is_speaking = False                # ← persists
        
        while True:
            if not self.session:
                print("No active session, waiting...")
                await asyncio.sleep(0.05)
                continue
            
            try:
                print("Waiting for turn...")
                turn = self.session.receive()
                print("Got turn")

                speaking_shown = False
                async for resp in turn:
                    #print(f"Processing response: {resp}")
                    
                    if resp.data and not speaking_shown:
                        print("Starting to speak...")
                        self.anim.set_mode("speak")
                        print("Speaking...")
                        speaking_shown = True
                    
                    if resp.data:
                        #print("Writing audio data...")
                        # self.recv_wf.writeframes(resp.data) # Blocking call
                        await self.loop.run_in_executor(None, self.recv_wf.writeframes, resp.data) # Non-blocking
                        await self.audio_in_q.put(resp.data)
                    
                    if resp.text:
                        print(f"\nGemini text response: {resp.text}")
                    
                    # Handle function calls based on tool_call
                    if resp.tool_call and resp.tool_call.function_calls:
                        for fc in resp.tool_call.function_calls:
                            function_name = fc.name
                            function_args = fc.args or {}
                            function_id = fc.id  # Crucial for the response

                            print(f"\nTool call detected: {function_name} with args {function_args}, ID: {function_id}")

                            if function_name in functions_map:
                                func_to_call = functions_map[function_name]
                                try:
                                    print(f"Executing function {function_name}...")
                                    result = func_to_call(**function_args)
                                    print(f"Function result: {result}")
                                    
                                    # Send the result back to Gemini using send_tool_response
                                    tool_response_part = types.FunctionResponse(
                                        id=function_id,
                                        name=function_name,
                                        response={"result": result}
                                    )
                                    await self.session.send_tool_response(
                                        function_responses=[tool_response_part]
                                    )
                                    print("Tool response sent back to Gemini")
                                except Exception as e:
                                    print(f"Error executing function {function_name}: {e}")
                                    error_tool_response = types.FunctionResponse(
                                        id=function_id,
                                        name=function_name,
                                        response={"error": str(e)}
                                    )
                                    await self.session.send_tool_response(
                                        function_responses=[error_tool_response]
                                    )
                            else:
                                print(f"Function {function_name} not found in available functions map.")
                                # Optionally send an error response back to Gemini if function not found
                                error_tool_response = types.FunctionResponse(
                                    id=function_id,
                                    name=function_name,
                                    response={"error": f"Function {function_name} not implemented or available."}
                                )
                                await self.session.send_tool_response(
                                    function_responses=[error_tool_response]
                                )
                    # else:
                        # print("No tool call in this response part") # Debug if needed

                # Show listening status when done speaking
                print("Conversation turn completed, switching to listening mode")
                self.anim.set_mode("idle")
                print("Listening...")
                
            except Exception as e:
                print(f"Error in _recv_from_gemini: {e}")
                traceback.print_exc()
                # Don't break the loop on error, just continue
                continue

    async def _playback(self):
        print("Playback‑loop started…")
        while True:
            pcm = await self.audio_in_q.get()
            if self.output_rate != RECEIVE_SAMPLE_RATE:
                pcm, self.rs_out_state = audioop.ratecv(
                    pcm, 2, OUT_CH,
                    RECEIVE_SAMPLE_RATE, self.output_rate,
                    self.rs_out_state)
            # self.output_stream.write(pcm) # Blocking call
            await self.loop.run_in_executor(None, self.output_stream.write, pcm) # Non-blocking
            self.audio_in_q.task_done()

    # ────────────────────────── cleanup ─────────────────────────────────────
    def _cleanup(self):
        print("Cleaning up…")
        for s in (self.input_stream, self.output_stream):
            if s and s.is_active():
                s.stop_stream()
            if s:
                s.close()
        for wf in (self.sent_wf, self.recv_wf):
            wf.close()
        self.pya.terminate()
        print("Done.")
        # Cleanup display
        try:
            self.disp.clear()
            self.disp.bl_DutyCycle(0)
            self.disp.module_exit()
        except Exception:
            pass

    # ────────────────────────── main entry ──────────────────────────────────
    async def run(self):
        if not os.getenv("GOOGLE_API_KEY"):
            # print("Set GOOGLE_API_KEY first.")
            return

        # set loop reference (if not set in __init__)
        if self.loop is None:
            self.loop = asyncio.get_running_loop()

        await self.setup_streams()

        # print("Connecting to Gemini Live…")
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"),
                                   http_options=types.HttpOptions(api_version="v1beta"))
        async with self.client.aio.live.connect(model=MODEL, config=CONFIG) as sess:
            self.session = sess
            # print("Connected – start talking!")
            # Show listening status upon connection
            self._show_status("Listening...")
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.anim.run())         
                tg.create_task(self._send_to_gemini())
                tg.create_task(self._recv_from_gemini())
                tg.create_task(self._playback())

    # ────────────────────────── static helper ───────────────────────────────


if __name__ == "__main__":
    # print("Press Ctrl+C to quit.")
    handler = AudioHandler()
    try:
        asyncio.run(handler.run())
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        handler._cleanup()