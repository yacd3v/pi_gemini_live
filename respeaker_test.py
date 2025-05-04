import sounddevice as sd
import soundfile as sf

DURATION = 5        # seconds
RATE     = 16000    # sample rate
CHANS    = 4        # 4 microphones
FILE     = '/tmp/test.wav'

# Use the first input device that contains “ReSpeaker”
mic_dev = next(i for i, d in enumerate(sd.query_devices()) if 'ReSpeaker' in d['name'])
# Keep default output device for playback
sd.default.device = (mic_dev, None)

print(f"Recording {DURATION}s from device #{mic_dev}")
audio = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=CHANS, dtype='int16')
sd.wait()

sf.write(FILE, audio, RATE)
print("Playing back…")
data, fs = sf.read(FILE, dtype='int16')
sd.play(data, fs)
sd.wait()
