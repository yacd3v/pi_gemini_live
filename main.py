"""
Main script for VocalGem robot
Simplified orchestrator that coordinates all modules
"""

import asyncio
import time
import traceback
import sys

# Import our modules
from vocalgem_modules import (
    GOOGLE_API_KEY, DisplayManager, AudioManager, 
    FunctionHandler, VisionManager, GeminiClient
)
from display_animator import DisplayAnimator
from face_tracker import FaceTracker

class VocalGemRobot:
    """Main robot class that orchestrates all components"""
    
    def __init__(self):
        # Create sleep event for coordination
        self.sleep_requested_event = asyncio.Event()
        
        # Initialize display manager
        self.display_manager = DisplayManager()
        self.display_manager.initialize_display()
        
        # Initialize audio manager
        self.audio_manager = AudioManager(self.sleep_requested_event)
        
        # Initialize face tracker
        self.face_tracker = FaceTracker(enable_tracking=True, confidence_threshold=0.4)
        
        # Initialize display animator
        self.anim = DisplayAnimator(self.display_manager.disp, stop_event=self.sleep_requested_event)
        
        # Initialize function handler
        self.function_handler = FunctionHandler(
            self.face_tracker, 
            self.display_manager, 
            self.sleep_requested_event, 
            self.audio_manager
        )
        
        # Initialize vision manager
        self.vision_manager = VisionManager(self.face_tracker)
        
        # Initialize Gemini client
        self.gemini_client = GeminiClient()
        
        # Set loop reference for audio manager
        if self.audio_manager.loop is None:
            try:
                self.audio_manager.loop = asyncio.get_running_loop()
            except RuntimeError:
                pass  # Will be set in run method

    async def run(self):
        """Main run method that orchestrates all components"""
        print("[VocalGemRobot] Starting...")
        
        if not GOOGLE_API_KEY:
            print("[VocalGemRobot] Error: GOOGLE_API_KEY environment variable not set")
            return

        self.sleep_requested_event.clear() # Clear event at the start of a new run

        # Set loop reference (if not set in __init__)
        if self.audio_manager.loop is None:
            self.audio_manager.loop = asyncio.get_running_loop()

        try:
            # Setup audio streams
            await self.audio_manager.setup_streams()

            # Create Gemini session context manager
            session_context = self.gemini_client.create_session()
            
            # Use the session with async with
            async with session_context as session:
                print("[GeminiClient] Connected. Listening...")
                self.vision_manager.set_session(session)
                
                # Run LED initialization sequence
                await self.anim.run_initialization()
                
                self.anim.set_mode("idle") # Set initial animation to idle

                try:
                    async with asyncio.TaskGroup() as tg:
                        send_task = tg.create_task(self.audio_manager.send_to_gemini(session))
                        recv_task = tg.create_task(self.audio_manager.receive_from_gemini(session, self.function_handler, self.anim))
                        playback_task = tg.create_task(self.audio_manager.playback(self.anim))
                        anim_task = tg.create_task(self.anim.run())
                        vision_task = tg.create_task(self.vision_manager.vision_feed(
                            interval=self.face_tracker.face_tracking_interval if self.face_tracker.face_detection_enabled else 8
                        ))
                        
                        # Add a monitoring task to help debug when tasks complete
                        async def monitor_sleep():
                            while not self.sleep_requested_event.is_set():
                                await asyncio.sleep(0.1)
                            # Give tasks a moment to see the sleep event and start cleanup
                            await asyncio.sleep(1)
                        
                        monitor_task = tg.create_task(monitor_sleep())
                        
                except Exception as e:
                    print(f"[VocalGemRobot] Error in task group: {e}")
                    print("[VocalGemRobot] Task group traceback:")
                    traceback.print_exc()
                    raise
            
        except Exception as e:
            print(f"[VocalGemRobot] Error in run method: {e}")
            print("[VocalGemRobot] Run method traceback:")
            traceback.print_exc()
            raise
        finally:
            print("[VocalGemRobot] Returning to wake_porcu.py")

    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up VocalGem robot...")
        
        # Stop animation and turn off LEDs
        if hasattr(self, 'anim'):
            try:
                if hasattr(self.anim, 'led_turn_off'):
                    self.anim.led_turn_off()  # Turn off LEDs first
                self.sleep_requested_event.set()  # Signal animator to stop
                time.sleep(0.1)  # Brief pause for animator to stop
            except Exception as e:
                print(f"Error stopping animator and LEDs: {e}")
        
        # Cleanup audio manager
        if hasattr(self, 'audio_manager'):
            self.audio_manager.cleanup()
            
        # Cleanup display manager
        if hasattr(self, 'display_manager'):
            self.display_manager.cleanup()
            
        # Additional delay to ensure all resources are fully released
        time.sleep(1)
        print("VocalGem robot cleanup completed.")

async def run_gemini():
    """Run the Gemini voice assistant"""
    robot = VocalGemRobot()
    try:
        await robot.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in Gemini: {e}")
    finally:
        robot.cleanup()
        # Add extra delay to ensure complete cleanup
        time.sleep(2)

if __name__ == "__main__":
    print("[Main] Starting application...")
    try:
        print("[Main] Running main async loop...")
        asyncio.run(run_gemini())
    except KeyboardInterrupt:
        print("[Main] Keyboard interrupt received")
    except Exception as e:
        print(f"[Main] Unexpected error occurred: {e}")
        print("[Main] Full traceback:")
        traceback.print_exc()
    finally:
        print("[Main] Application completed") 