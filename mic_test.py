import pyaudio
import wave
import audioop
import asyncio
import queue
import time

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono audio
CHUNK_MS = 20  # 20ms chunks for low latency
TARGET_SAMPLE_RATE = 16000  # Target sample rate for Gemini
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "mic_test.wav"
MAX_QUEUE_SIZE = 50  # Limit queue size to prevent memory issues

class AudioTest:
    def __init__(self):
        self.pya = pyaudio.PyAudio()
        self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.frames = []
        self.input_stream = None
        self.output_stream = None
        self.input_rate = None
        self.chunk_size = None
        self.rs_state = None  # Resampler state

    def _mic_callback(self, in_data, frame_count, time_info, status):
        """Callback function for microphone input"""
        try:
            self.audio_queue.put_nowait(in_data)
        except queue.Full:
            # Drop the chunk if queue is full
            pass
        return (None, pyaudio.paContinue)

    def find_input_device(self):
        """Find and return the input device index"""
        for i in range(self.pya.get_device_count()):
            info = self.pya.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                name = info.get('name', '')
                if 'MPOW' in name or 'HC6' in name:
                    print(f"Selected input device {i}: {name}")
                    return i
        # Fallback to default input device
        default_info = self.pya.get_default_input_device_info()
        print(f"Using default input device {default_info['index']}: {default_info['name']}")
        return default_info['index']

    def setup_streams(self):
        """Setup input and output streams"""
        try:
            # Find input device
            input_index = self.find_input_device()
            
            # Try supported sample rates, preferring 16kHz
            mic_info = self.pya.get_device_info_by_index(input_index)
            default_in_rate = int(mic_info['defaultSampleRate'])
            print(f"Default input sample rate: {default_in_rate} Hz")
            
            for rate in [16000, default_in_rate, 48000, 44100]:
                try:
                    # Calculate chunk size based on actual sample rate
                    self.chunk_size = int(rate * CHUNK_MS / 1000)
                    print(f"Attempting to open input stream at {rate} Hz with chunk size {self.chunk_size}")
                    
                    self.input_stream = self.pya.open(
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        input_device_index=input_index,
                        frames_per_buffer=self.chunk_size,
                        stream_callback=self._mic_callback
                    )
                    self.input_rate = rate
                    print(f"Input stream opened at {rate} Hz")
                    break
                except Exception as e:
                    print(f"Failed opening input at {rate} Hz: {e}")
            
            if not self.input_stream:
                raise Exception("Unable to open input stream on any supported sample rate")

            # Start the input stream
            self.input_stream.start_stream()
            print("Input stream started.")

        except Exception as e:
            print(f"Error setting up streams: {e}")
            self.cleanup()
            return False
        return True

    def record_audio(self):
        """Record audio for specified duration"""
        print(f"Recording for {RECORD_SECONDS} seconds...")
        end_time = time.time() + RECORD_SECONDS
        
        while time.time() < end_time:
            try:
                # Get audio data from queue with timeout
                data = self.audio_queue.get(timeout=0.1)
                
                # Compute RMS for VU meter
                level = audioop.rms(data, 2)
                bars = "#" * (level // 500)  # adjust divisor for sensitivity
                print(f"\rMic level: [{bars:<20}] {level:4d}", end="")
                
                # Resample if needed, maintaining state
                if self.input_rate and self.input_rate != TARGET_SAMPLE_RATE:
                    data, self.rs_state = audioop.ratecv(
                        data,
                        2,  # Sample width in bytes
                        CHANNELS,
                        self.input_rate,
                        TARGET_SAMPLE_RATE,
                        self.rs_state
                    )
                
                self.frames.append(data)
                
            except queue.Empty:
                # No data available, continue recording
                continue
            except Exception as e:
                print(f"\nError during recording: {e}")
                break
        
        print("\nRecording finished.")

    def save_audio(self):
        """Save recorded audio to WAV file"""
        print(f"Saving recording to {WAVE_OUTPUT_FILENAME}")
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.pya.get_sample_size(FORMAT))
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print("Audio saved successfully.")

    def play_audio(self):
        """Play back the recorded audio"""
        print("Playing back recording...")
        try:
            # Open the WAV file
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')
            
            # Calculate chunk size for playback
            playback_chunk_size = int(wf.getframerate() * CHUNK_MS / 1000)
            
            # Open output stream
            output_stream = self.pya.open(
                format=self.pya.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                frames_per_buffer=playback_chunk_size
            )
            
            # Play the audio
            data = wf.readframes(playback_chunk_size)
            while data:
                output_stream.write(data)
                data = wf.readframes(playback_chunk_size)
            
            # Cleanup
            output_stream.stop_stream()
            output_stream.close()
            wf.close()
            print("Playback finished.")
            
        except Exception as e:
            print(f"Error during playback: {e}")

    def cleanup(self):
        """Clean up audio resources"""
        print("Cleaning up audio resources...")
        try:
            if self.input_stream and self.input_stream.is_active():
                self.input_stream.stop_stream()
            if self.input_stream:
                self.input_stream.close()
            if self.pya:
                self.pya.terminate()
            print("Audio resources cleaned up.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    audio_test = AudioTest()
    try:
        if audio_test.setup_streams():
            audio_test.record_audio()
            audio_test.save_audio()
            audio_test.play_audio()
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    finally:
        audio_test.cleanup()

if __name__ == '__main__':
    main() 