"""
Audio Manager Module for VocalGem robot
Handles audio stream management, PyAudio operations, and audio processing
"""

import os
import sys
import time
import wave
import audioop
import asyncio
import queue
import pyaudio
import websockets

# Temporarily suppress ALSA/JACK warnings during PyAudio import only
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import pyaudio

# Restore stderr immediately after PyAudio import
sys.stderr.close()
sys.stderr = original_stderr

from .config import (
    FORMAT, IN_CH, OUT_CH, CHUNK_MS, SEND_SAMPLE_RATE, 
    RECEIVE_SAMPLE_RATE, AEC_SAMPLE_RATE, RAW_CH
)

class AudioManager:
    """Handles audio stream management and PyAudio operations"""

    def __init__(self, sleep_requested_event):
        # Event‑loop reference for thread‑safe calls from PyAudio callback
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # called in main thread before loop exists; fixed later in run()
            self.loop = None

        self.sleep_requested_event = sleep_requested_event
        self.pya = pyaudio.PyAudio()

        # queues (200 packets ≈ 4 s @ 20 ms)
        self.audio_out_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self.audio_in_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

        # logging WAVs
        self.sent_wf = wave.open("sent_audio.wav", "wb")
        self.sent_wf.setnchannels(OUT_CH)
        self.sent_wf.setsampwidth(self.pya.get_sample_size(getattr(pyaudio, FORMAT)))
        self.sent_wf.setframerate(SEND_SAMPLE_RATE)

        self.recv_wf = wave.open("recv_audio.wav", "wb")
        self.recv_wf.setnchannels(OUT_CH)
        self.recv_wf.setsampwidth(self.pya.get_sample_size(getattr(pyaudio, FORMAT)))
        self.recv_wf.setframerate(RECEIVE_SAMPLE_RATE)

        # streams + rates
        self.input_stream = None
        self.output_stream = None
        self.input_rate: int | None = None
        self.output_rate: int | None = None

        # resampler states
        self.rs_in_state = None
        self.rs_out_state = None

    def _strip_to_processed(self, data: bytes) -> bytes:
        """Process raw audio data to extract mono channel"""
        frame = RAW_CH * 2              # 12 bytes per frame
        out = bytearray(len(data) // RAW_CH)  # space for 2-byte mono
        o = 0
        for i in range(0, len(data), frame):
            out[o:o+2] = data[i:i+2]    # copy both bytes of ch-0
            o += 2
        return bytes(out)

    def _find_input_device(self) -> int:
        """Find ReSpeaker input device"""
        for idx in range(self.pya.get_device_count()):
            name = self.pya.get_device_info_by_index(idx)["name"].lower()
            if "respeaker" in name or "seeed" in name:
                return idx
        raise RuntimeError("ReSpeaker input not found")

    def _find_output_device(self) -> int:
        """Find ReSpeaker output device"""
        for idx in range(self.pya.get_device_count()):
            name = self.pya.get_device_info_by_index(idx)["name"].lower()
            if "respeaker" in name or "seeed" in name:
                return idx
        raise RuntimeError("ReSpeaker output not found")

    def _mic_callback(self, in_data, frame_count, time_info, status):
        """Runs in **PyAudio thread** – push data into asyncio queue."""
        if self.loop is None:
            return (None, pyaudio.paContinue)
        
        # Check if we should stop processing
        if self.sleep_requested_event.is_set():
            return (None, pyaudio.paComplete)
            
        try:
            self.loop.call_soon_threadsafe(self.audio_out_q.put_nowait, in_data)
        except asyncio.QueueFull:
            # drop one packet – better than blocking and causing an overrun
            pass
        except Exception as e:
            # Handle other exceptions gracefully
            pass
        return (None, pyaudio.paContinue)

    async def setup_streams(self):
        """Setup audio input and output streams"""
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
                self.input_stream = self.pya.open(format=getattr(pyaudio, FORMAT),
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
        out_rates = [AEC_SAMPLE_RATE]
        
        for r in out_rates:
            try:
                fpb = int(r * CHUNK_MS / 1000)
                self.output_stream = self.pya.open(format=getattr(pyaudio, FORMAT),
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

    async def send_to_gemini(self, session):
        """Send audio data to Gemini"""
        print("[Sender] Task started.")
        while True:
            if self.sleep_requested_event.is_set():
                print("[Sender] Sleep event detected, exiting send-loop.")
                break
            try:
                # Wait for data with a timeout to allow checking the sleep event
                data = await asyncio.wait_for(self.audio_out_q.get(), timeout=0.1)
                if data is None: # Sentinel value for shutdown
                    print("[Sender] Received sentinel, exiting send-loop.")
                    self.audio_out_q.task_done()
                    break
            except asyncio.TimeoutError:
                continue # No data, loop back to check sleep_requested_event
            except Exception as e:
                print(f"[Sender] Error getting data from queue: {e}")
                break

            try:
                data = self._strip_to_processed(data)       # ① use the AEC track only
                if self.input_rate != SEND_SAMPLE_RATE:
                    data, self.rs_in_state = audioop.ratecv(
                        data, 2, OUT_CH,
                        self.input_rate, SEND_SAMPLE_RATE,
                        self.rs_in_state)
                if session: # Rely on send_realtime_input to fail if session is closed
                    from google.genai import types
                    blob = types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}")
                    await session.send_realtime_input(media=blob)
                else:
                    print("[Sender] Session not active, not sending.")
                self.sent_wf.writeframes(data)
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"[Sender] Connection closed while sending: {e}. Exiting send-loop.")
                break
            except Exception as e:
                print(f"[Sender] Error processing or sending audio: {e}")
            finally:
                if 'data' in locals() and data is not None: # Ensure data was fetched and not sentinel
                    self.audio_out_q.task_done()
        
        print("[Sender] Send task exiting...")

    async def receive_from_gemini(self, session, function_handler, display_animator):
        """Receive audio data from Gemini"""
        print("[Receiver] Task started.")

        is_speaking = False                # ← persists
        
        while True:
            if self.sleep_requested_event.is_set():
                print("[Receiver] Sleep event detected, exiting receive-loop.")
                break

            if not session: # Rely on session.receive() to fail if session is closed
                if self.sleep_requested_event.is_set(): # Double check if sleep was requested during this gap
                    break
                await asyncio.sleep(0.1) # Brief pause before checking session again or sleep event
                continue
            
            try:
                # Use a timeout for receive to allow checking sleep_requested_event
                # However, session.receive() is a generator, making direct timeout tricky.
                # The primary exit from this loop when sleeping will be the ConnectionClosedError
                # when self.session.close() is called in go_to_sleep.

                turn_iterator = session.receive()
                async for resp in turn_iterator:

                    if self.sleep_requested_event.is_set():
                        print("[Receiver] Sleep event detected mid-turn, breaking from turn processing.")
                        break # Exit from processing messages in the current turn
                    
                    if resp.data and not is_speaking:
                        speak_animation_key = f"speak_{function_handler.current_speaking_emotion}"
                        display_animator.set_mode(speak_animation_key)
                        is_speaking = True
                    
                    if resp.data:
                        await self.loop.run_in_executor(None, self.recv_wf.writeframes, resp.data) # Non-blocking
                        await self.audio_in_q.put(resp.data)
                    
                    #if resp.text:
                        #print(f"\nGemini text response: {resp.text}")
                    
                    # Handle function calls based on tool_call
                    if resp.tool_call and resp.tool_call.function_calls:
                        for fc in resp.tool_call.function_calls:
                            function_name = fc.name
                            function_args = fc.args or {}
                            function_id = fc.id  # Crucial for the response

                            print(f"\n{'='*50}")
                            print(f"Function Call: {function_name}")
                            print(f"Arguments: {function_args}")
                            
                            if function_name in function_handler.functions_map:
                                func_to_call = function_handler.functions_map[function_name]
                                try:
                                    # If it's an async function (like go_to_sleep), await it
                                    if asyncio.iscoroutinefunction(func_to_call):
                                        result = await func_to_call(**function_args)
                                    else:
                                        result = func_to_call(**function_args)
                                    
                                    print(f"Response: {result}")
                                    print(f"{'='*50}\n")
                                    
                                    # For go_to_sleep, the session will be closing, so don't attempt to send a response.
                                    if function_name == "go_to_sleep":
                                        print("Skipping tool response for go_to_sleep as session is closing.")
                                    else:
                                        # Send the result back to Gemini using send_tool_response
                                        from google.genai import types
                                        tool_response_part = types.FunctionResponse(
                                            id=function_id,
                                            name=function_name,
                                            response={"result": result}
                                        )
                                        await session.send_tool_response(
                                            function_responses=[tool_response_part]
                                        )
                                except Exception as e:
                                    print(f"\n{'='*50}")
                                    print(f"Function Call Error: {function_name}")
                                    print(f"Arguments: {function_args}")
                                    print(f"Error: {str(e)}")
                                    print(f"{'='*50}\n")
                            else:
                                print(f"Function {function_name} not found in available functions map.")
                                # Send error response back to Gemini if function not found
                                if session:
                                    try:
                                        from google.genai import types
                                        error_tool_response = types.FunctionResponse(
                                            id=function_id,
                                            name=function_name,
                                            response={"error": f"Function {function_name} not implemented or available."}
                                        )
                                        await session.send_tool_response(
                                            function_responses=[error_tool_response]
                                        )
                                    except Exception as e:
                                        print(f"Failed to send error response for {function_name}: {e}")
                
                # Show listening status when done speaking
                display_animator.set_mode("idle")
                is_speaking = False  # Reset for the next turn
                
            except websockets.exceptions.ConnectionClosedOK:
                print("[Receiver] Connection closed (OK). Exiting receive-loop.")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"[Receiver] Connection closed with error: {e}. Exiting receive-loop.")
                break
            except asyncio.TimeoutError:
                # This might occur if we implement a timeout around receive(), but it's complex with async iterators
                continue
            except Exception as e:
                print(f"Error in _recv_from_gemini: {e}")
                import traceback
                traceback.print_exc()
                if self.sleep_requested_event.is_set(): # If an error occurs, and sleep is requested, exit.
                    print("[Receiver] Exiting due to error and sleep request.")
                    break
                # Don't break the loop on other errors unless sleep is also set, just continue
                await asyncio.sleep(0.1) # Small delay before retrying or continuing
                continue

        print("[Receiver] Receive task exiting...")

    async def playback(self, display_animator):
        """Play audio from Gemini"""
        print("[Playback] Task started.")
        while True:
            if self.sleep_requested_event.is_set():
                print("[Playback] Sleep event detected, exiting playback-loop.")
                break
            try:
                # Wait for PCM data with a timeout
                pcm = await asyncio.wait_for(self.audio_in_q.get(), timeout=0.1)
                if pcm is None: # Sentinel value for shutdown
                    print("[Playback] Received sentinel, exiting playback-loop.")
                    self.audio_in_q.task_done()
                    break
            except asyncio.TimeoutError:
                continue # No data, loop back to check sleep_requested_event
            except Exception as e:
                print(f"[Playback] Error getting data from queue: {e}")
                break

            try:
                if self.output_rate != RECEIVE_SAMPLE_RATE:
                    pcm_converted, self.rs_out_state = audioop.ratecv(
                        pcm, 2, OUT_CH,
                        RECEIVE_SAMPLE_RATE, self.output_rate,
                        self.rs_out_state)
                else:
                    pcm_converted = pcm
                
                # Send audio data to display animator for spectrum analysis
                # Use the original PCM data (before rate conversion) for better frequency analysis
                if display_animator:
                    display_animator.update_audio_data(pcm)
                
                if self.output_stream and self.output_stream.is_active():
                    await self.loop.run_in_executor(None, self.output_stream.write, pcm_converted)
                else:
                    print("[Playback] Output stream not active, not playing audio.")
            except Exception as e:
                print(f"[Playback] Error processing or playing audio: {e}")
            finally:
                if 'pcm' in locals() and pcm is not None: # Ensure pcm was fetched and not sentinel
                    self.audio_in_q.task_done()
        
        print("[Playback] Playback task exiting...")

    def cleanup(self):
        """Clean up audio resources"""
        print("Cleaning up audio resources...")
        
        # Cleanup audio streams with proper error handling
        for s in (self.input_stream, self.output_stream):
            try:
                if s and s.is_active():
                    s.stop_stream()
                    time.sleep(0.1)  # Small delay between stop and close
                if s:
                    s.close()
            except Exception as e:
                print(f"Error cleaning up stream: {e}")
        
        # Cleanup wave files
        for wf in (self.sent_wf, self.recv_wf):
            try:
                if wf:
                    wf.close()
            except Exception as e:
                print(f"Error closing wave file: {e}")
        
        # Terminate PyAudio with delay
        try:
            if self.pya:
                self.pya.terminate()
                time.sleep(0.5)  # Longer delay after PyAudio termination
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")
            
        print("Audio cleanup completed.") 