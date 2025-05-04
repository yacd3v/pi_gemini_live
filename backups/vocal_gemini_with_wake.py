#!/usr/bin/env python3
# Script that listens for a wake word and then starts a Gemini Live session.

import json
import asyncio
import traceback
from wake_word_detector import setup_audio, setup_vosk, check_wake_word, CHUNK
from vocal_gemini import AudioHandler


def listen_for_wake_word():
    p, stream = setup_audio()
    recognizers = setup_vosk()
    print('Listening for wake word...')
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            for lang, recognizer in recognizers.items():
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get('text', '')
                    if text and check_wake_word(text, lang):
                        print('Wake word detected:', text)
                        return lang
    except KeyboardInterrupt:
        print('Wake word listening interrupted by user.')
        return None
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    lang = listen_for_wake_word()
    if not lang:
        print('No wake word detected, exiting.')
        return
    print('Starting Gemini session for language', lang)
    handler = AudioHandler()
    # Optionally adjust Gemini language based on detected language
    try:
        cfg = handler.CONFIG
        if hasattr(cfg.speech_config, 'language_code'):
            cfg.speech_config.language_code = 'en-US' if lang == 'en' else 'fr-FR'
    except Exception:
        pass

    try:
        asyncio.run(handler.run())
    except KeyboardInterrupt:
        print('Session interrupted by user.')
    except Exception:
        traceback.print_exc()
    finally:
        handler._cleanup()


if __name__ == '__main__':
    main() 