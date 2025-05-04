import json
import wave
import logging
import os
import asyncio
import sys
from dotenv import load_dotenv

import numpy as np
import pyaudio
import pvporcupine
from vocal_gemini import AudioHandler

# ---------- constants ----------
ACCESS_KEY = "RJD+PezWj2uCqiurXzcs8wxVBxT6LYcs6ekrSOQTbnFNQzQ5EMtv2A=="
KEYWORD_PATHS = ["salut-karl_fr.ppn"]
MODEL_PATH = "porcupine_params_fr.pv"  # French model file

CHUNK = 512          # 32 ms @16 kHz
FORMAT = pyaudio.paInt16
RATE = 16000
CHANNELS = 6         # full stream from ReSpeaker
BEAM_CH = 0          # beam‑formed channel

REC_SECONDS = 20     # write this many seconds to wav
OUT_WAV = "outputtest.wav"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- helpers ----------
def open_pyaudio():
    p = pyaudio.PyAudio()
    dev_index = None
    for i in range(p.get_device_count()):
        if "ReSpeaker" in p.get_device_info_by_index(i)["name"]:
            dev_index = i
            break
    if dev_index is None:
        raise RuntimeError("ReSpeaker device not found")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=dev_index,
                    frames_per_buffer=CHUNK)
    return p, stream, dev_index

def extract_beam(data):
    """take channel 0 from interleaved 16‑bit little‑endian frame buffer"""
    return (np
            .frombuffer(data, dtype="<i2")
            .reshape(-1, CHANNELS)[:, BEAM_CH]
            .astype(np.int16))

async def run_gemini():
    """Run the Gemini voice assistant"""
    handler = AudioHandler()
    try:
        await handler.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Error in Gemini: {e}")
    finally:
        handler._cleanup()

# ---------- main ----------
def main():
    # sanity‑check keyword files
    for path in KEYWORD_PATHS:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    porcu = pvporcupine.create(
        access_key=ACCESS_KEY,
        keyword_paths=KEYWORD_PATHS,
        model_path=MODEL_PATH)

    p, stream, dev_index = open_pyaudio()

    # wav debug file
    wf = wave.open(OUT_WAV, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    frames_left = int(RATE / CHUNK * REC_SECONDS)

    logging.info("Listening for wake words…")

    try:
        while True:
            raw = stream.read(CHUNK, exception_on_overflow=False)
            beam = extract_beam(raw)

            if frames_left == REC_SECONDS * RATE // CHUNK:     # first iteration only
                pass
            
            # write first 20 s
            if frames_left > 0:
                wf.writeframes(beam.tobytes())
                frames_left -= 1

            # Porcupine expects a Python list / bytes array of ints
            idx = porcu.process(beam)
            if idx >= 0:
                logging.info("Wake word detected! Starting Gemini...")
                # Stop the wake word detection
                stream.stop_stream()
                stream.close()
                p.terminate()
                wf.close()
                porcu.delete()
                
                # Run Gemini
                asyncio.run(run_gemini())
                
                # After Gemini exits, restart wake word detection
                porcu = pvporcupine.create(
                    access_key=ACCESS_KEY,
                    keyword_paths=KEYWORD_PATHS,
                    model_path=MODEL_PATH)
                p, stream, dev_index = open_pyaudio()
                wf = wave.open(OUT_WAV, "wb")
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                frames_left = int(RATE / CHUNK * REC_SECONDS)
                logging.info("Listening for wake words…")

    except KeyboardInterrupt:
        logging.info("Stopping…")
    finally:
        porcu.delete()
        wf.close()
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
