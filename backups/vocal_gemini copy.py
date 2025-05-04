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
from PIL import Image, ImageDraw, ImageFont

# ─── Audio constants ──────────────────────────────────────────────────────────
FORMAT = pyaudio.paInt16
IN_CH = 1   #  use processed mono channel
OUT_CH = 1   # we still play/record mono
CHUNK_MS = 20             # one packet = 20 ms of audio
SEND_SAMPLE_RATE = 16_000     # Gemini Live input format
RECEIVE_SAMPLE_RATE = 24_000  # Gemini Live output format
AEC_SAMPLE_RATE = 16_000          

# ─── Gemini constants ────────────────────────────────────────────────────────
MODEL = "gemini-2.0-flash-live-001"
CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    session_resumption=types.SessionResumptionConfig(handle=None),
    speech_config=types.SpeechConfig(
        language_code="fr-FR",
         voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
        #    #Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr.
         )
    ),
    system_instruction=types.Content(
        parts=[
            types.Part(
                text="Tu es la voix d'un petit robot nommé Karl, amicale mais directe.\n\n"
                "• Garde un débit naturel mais assez rapide.\n"
                "• Utilise les contractions courantes : « c'est », « j'ai », « on va ».\n"
                "• Préfère des phrases courtes.\n"
                "• Insère des pauses avec virgules, tirets ou « ... » là où on reprendrait son souffle.\n"
                "• Si l'utilisateur t'interrompt, termine ton mot puis tais-toi.\n"
                "• Ton conversationnel : pas de buzzwords ni de formules pompées.\n"
                "• Lis les nombres comme on les dit tous les jours : « vingt-trois », « deux mille ».\n"
                "• Mets un léger accent sur les mots importants.\n"
                "• Ne dis jamais « en tant qu'IA ».\n\n"
                "### Mini-exemples\n\n"
                "UTILISATEUR : Salut, quoi de neuf dans l'appli ?\n"
                "ASSISTANT : Alors... on a ajouté le mode hors-ligne et des recherches plus rapides. Tu veux plus de détails ?\n\n"
                "UTILISATEUR : Réserve-moi un resto pour deux demain à 19 h.\n"
                "ASSISTANT : C'est noté : table pour deux, demain, 19 heures. Ça te va ?"
            )
        ]
    )
)


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

    def _find_output_device(self) -> int:
        for idx in range(self.pya.get_device_count()):
            info = self.pya.get_device_info_by_index(idx)
            if info.get("maxOutputChannels", 0) and (
                "respeaker" in info["name"].lower() or "usb audio" in info["name"].lower()):
                return idx
        info = self.pya.get_default_output_device_info()
        return info["index"]

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
                                                  channels=1,
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
        out_rates = [AEC_SAMPLE_RATE, 48_000, 44_100]
        
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
        
        while True:
            if not self.session:
                await asyncio.sleep(0.05)
                continue
            turn = self.session.receive()
            speaking_shown = False
            async for resp in turn:
                if resp.data and not speaking_shown:
                    self._show_status("Speaking...")
                    print("Speaking...")
                    speaking_shown = True
                if resp.data:
                    self.recv_wf.writeframes(resp.data)
                    await self.audio_in_q.put(resp.data)
                if resp.text:
                    print(f"\nGemini: {resp.text}")
                    pass
            # Show listening status when done speaking
            self._show_status("Listening...")
            print("Listening...")
    async def _playback(self):
        print("Playback‑loop started…")
        while True:
            pcm = await self.audio_in_q.get()
            if self.output_rate != RECEIVE_SAMPLE_RATE:
                pcm, self.rs_out_state = audioop.ratecv(
                    pcm, 2, OUT_CH,
                    RECEIVE_SAMPLE_RATE, self.output_rate,
                    self.rs_out_state)
            self.output_stream.write(pcm)
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