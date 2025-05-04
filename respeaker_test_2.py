#!/usr/bin/env python3
"""
Quick confidence test for a ReSpeaker USB 4-Mic Array on a Raspberry Pi 5:
 • Records <DURATION> s of audio while logging DoA every 0.1 s
 • LED ring red during capture
 • Plays the clip back through the 3.5 mm jack, LED ring green
 • CSV log: timestamp, doa° saved next to the WAV
"""

import time
import csv
import datetime as dt
from pathlib import Path
import sounddevice as sd
import soundfile as sf

from pixel_ring import pixel_ring            # LED control
from usb_4_mic_array.tuning import find     # DoA helper (RATE fixed at 16 kHz)

# ---------- tweakables ----------
RATE      = 16_000          # Hz (built-in rate)
CHANNELS  = 4               # four digital mics
DURATION  = 5               # seconds to record
WAV_FILE  = Path("/tmp/respeaker_test.wav")
LOG_FILE  = WAV_FILE.with_suffix(".csv")
# --------------------------------

def log_doa(run_event):
    """Poll the array every 100 ms and append DoA to CSV until run_event() is False."""
    dev = find()
    if not dev:
        raise RuntimeError("No ReSpeaker device found")
    
    with LOG_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utc_iso", "degrees"])
        while run_event():
            deg = dev.direction               # 0-359, -1 means 'no sound'
            writer.writerow([dt.datetime.utcnow().isoformat(), deg])
            time.sleep(0.1)
    dev.close()

def main():
    # -------- record --------
    pixel_ring.set_color(0xFF0000)          # full-ring red
    start = time.time()
    recorder_running = lambda: time.time() - start < DURATION

    # fire the logger in a very small thread
    import threading
    t = threading.Thread(target=log_doa, args=(recorder_running,), daemon=True)
    t.start()

    audio = sd.rec(int(DURATION * RATE),
                   samplerate=RATE,
                   channels=CHANNELS,
                   dtype='int16')
    sd.wait()                                            # blocks DURATION s
    pixel_ring.off()                                     # end red

    sf.write(WAV_FILE, audio, RATE)
    t.join()                                             # ensure log is flushed

    # -------- playback --------
    pixel_ring.set_color(0x00FF00)          # full-ring green
    data, fs = sf.read(WAV_FILE, dtype='int16')
    sd.play(data, fs)
    sd.wait()
    pixel_ring.off()

    print(f"✔ Audio: {WAV_FILE}")
    print(f"✔ DoA log: {LOG_FILE}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pixel_ring.off()
        print("\nCancelled.")
