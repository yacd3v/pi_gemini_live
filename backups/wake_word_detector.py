import pyaudio
import wave
import json
import os
import numpy as np
from vosk import Model, KaldiRecognizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CHUNK = 256
FORMAT = pyaudio.paInt16
CHANNELS = 6  # Updated for ReSpeaker 6-channel
RATE = 16000
WAKE_WORDS = {
    "en": "hey vocal",  # English wake word
    "fr": "salut Karl"  # French wake word
}

def setup_audio():
    """Initialize audio input stream"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    return p, stream

def setup_vosk():
    """Initialize Vosk models and recognizers for both languages"""
    models = {}
    recognizers = {}
    
    # Check and setup English model
    if not os.path.exists("model-en"):
        logging.info("Downloading English Vosk model...")
        logging.error("Please download the English Vosk model first")
        exit(1)
    models["en"] = Model("model-en")
    recognizers["en"] = KaldiRecognizer(models["en"], RATE)
    
    # Check and setup French model
    if not os.path.exists("model-fr"):
        logging.info("Downloading French Vosk model...")
        logging.error("Please download the French Vosk model first")
        exit(1)
    models["fr"] = Model("model-fr")
    recognizers["fr"] = KaldiRecognizer(models["fr"], RATE)
    
    return recognizers

def check_wake_word(text, language):
    """Check if the wake word for the given language is present in the text"""
    return WAKE_WORDS[language].lower() in text.lower()

def main():
    logging.info("Initializing wake word detection...")
    p, stream = setup_audio()
    recognizers = setup_vosk()
    
    logging.info("Listening for wake words in English and French...")
    logging.info(f"English wake word: '{WAKE_WORDS['en']}'")
    logging.info(f"French wake word: '{WAKE_WORDS['fr']}'")

     # --- recording setup ---------------------------------
    out_file = "outputtest.wav"
    record_seconds = 20
    max_frames = int(RATE / CHUNK * record_seconds)  # frames we will keep
    frames_written = 0

    wf = wave.open(out_file, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    # ------------------------------------------------------

    logging.info("Listening… (will also dump %s)", out_file)
    
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            mono = (np
                    .frombuffer(data, dtype="<i2")
                    .reshape(-1, CHANNELS)[:, 0]  # channel 0 = beam‑formed
                    .tobytes())

            # write to output file during the first 20 s
            if frames_written < max_frames:
                wf.writeframes(mono)
                frames_written += 1

            # speech recognition
            for lang, rec in recognizers.items():
                if rec.AcceptWaveform(mono):
                    txt = json.loads(rec.Result()).get("text", "")
                    if txt:
                        logging.info("Heard (%s): %s", lang, txt)
                        if check_wake_word(txt, lang):
                            logging.info("Wake word detected in %s!", lang.upper())
    except KeyboardInterrupt:
        logging.info("Stopping…")
    finally:
        wf.close()          # make sure the .wav header is written
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main() 