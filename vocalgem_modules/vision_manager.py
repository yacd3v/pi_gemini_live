"""
Vision Manager Module for VocalGem robot
Handles camera operations, vision feed, and face tracking integration
"""

import asyncio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500
from libcamera import Transform
from google.genai import types
from .config import CAMERA_RESOLUTION, CAMERA_JPEG_QUALITY, CAMERA_FRAME_INTERVAL

class VisionManager:
    """Handles camera operations and vision feed"""
    
    def __init__(self, face_tracker, session=None):
        self.face_tracker = face_tracker
        self.session = session
        self.last_captured_frame = None
        
    def _capture_jpeg(self) -> bytes:
        """Capture a JPEG image from the camera"""
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(
            main={"size": CAMERA_RESOLUTION, "format": "RGB888"}
        ))
        cam.start()
        time.sleep(0.25)           # quick AE settle
        rgb = cam.capture_array()
        cam.close()
        return cv2.imencode(".jpg", rgb)[1].tobytes()
    
    async def vision_feed(self, interval=8):
        """Send a fresh camera frame every *interval* seconds."""
        # Initialize camera with IMX500 if face detection is enabled
        if self.face_tracker.face_detection_enabled:
            try:
                cam = Picamera2(self.face_tracker.imx500.camera_num)
                print("[Vision] Camera initialized with IMX500 for face detection")
                
                # Configure camera for both vision feed and AI detection - match test_autoaim.py
                config = cam.create_preview_configuration(
                    main={"size": CAMERA_RESOLUTION},
                    raw={"size": (2028, 1520)},
                    encode="main",
                    buffer_count=6
                )
                
                # Set transform if supported - like test_autoaim.py
                try:
                    config["transform"] = Transform()
                except Exception as e:
                    print(f"[Vision] Could not set transform: {e}")
                
                cam.configure(config)
                
                # Set up network intrinsics for face detection - match test_autoaim.py
                if self.face_tracker.imx500.network_intrinsics:
                    ni = self.face_tracker.imx500.network_intrinsics
                    ni.task = "pose estimation"
                    ni.inference_rate = 30.0
                    # PoseNet doesn't use bbox_normalization or labels the same way
                    print("[Vision] IMX500 network intrinsics configured")
                    
            except Exception as e:
                print(f"[Vision] Error initializing IMX500 camera: {e}")
                # Fallback to regular camera
                cam = Picamera2()
                cam.configure(cam.create_still_configuration(
                    main={"size": CAMERA_RESOLUTION, "format": "RGB888"}
                ))
                self.face_tracker.face_detection_enabled = False
        else:
            # Regular camera initialization
            cam = Picamera2()
            cam.configure(cam.create_still_configuration(
                main={"size": CAMERA_RESOLUTION, "format": "RGB888"}
            ))
            print("[Vision] Camera initialized (no face detection)")
        
        cam.start()
        print("[Vision] Camera started")
        
        # Delay first frame to allow session setup
        await asyncio.sleep(3)

        try:
            while True:  # This will be controlled by the sleep_requested_event in the main loop
                try:
                    # Always capture with metadata if face detection is enabled
                    face_detections = []
                    tracking_info = {}
                    
                    if self.face_tracker.face_detection_enabled:
                        request = cam.capture_request()
                        rgb = request.make_array("main")
                        metadata = request.get_metadata()
                        request.release()
                        
                        # Parse face detections regardless of tracking status for debug
                        face_detections = self.face_tracker.parse_face_detection(metadata)
                        
                        # Perform face detection and tracking regardless of session status
                        if self.face_tracker.should_auto_track():
                            if face_detections:
                                best_face = max(face_detections, key=lambda f: f["confidence"])
                                print(f"[Vision] Face detected (confidence: {best_face['confidence']:.2f})")
                                
                                # Track the face and get tracking info
                                tracking_result = self.face_tracker.track_face(best_face["center_x"], best_face["center_y"])
                                # tracking_result is now always True when a face is being tracked
                                print(f"[Vision] Tracking face (confidence: {best_face['confidence']:.2f})")
                                
                                # Collect tracking info for debug
                                tracking_info = {
                                    'status': 'tracking',
                                    'confidence': best_face['confidence'],
                                    'history_size': len(self.face_tracker.face_history),
                                    'target_x': best_face["center_x"],
                                    'target_y': best_face["center_y"]
                                }
                                
                                # Add smoothed error info if available
                                if self.face_tracker.face_history:
                                    # Calculate smoothed error like the tracker does
                                    frame_width = CAMERA_RESOLUTION[0]
                                    frame_height = CAMERA_RESOLUTION[1]
                                    target_center_x = best_face["center_x"] * frame_width
                                    target_center_y = best_face["center_y"] * frame_height
                                    image_center_x = frame_width / 2
                                    image_center_y = frame_height / 2
                                    error_x = (target_center_x - image_center_x) / (frame_width / 2)
                                    error_y = (target_center_y - image_center_y) / (frame_height / 2)
                                    
                                    # Get smoothed values
                                    weights = [1.0, 1.5, 2.0][:len(self.face_tracker.face_history)]
                                    total_weight = sum(weights)
                                    smoothed_error_x = sum(w * pos[0] for w, pos in zip(weights, self.face_tracker.face_history)) / total_weight
                                    smoothed_error_y = sum(w * pos[1] for w, pos in zip(weights, self.face_tracker.face_history)) / total_weight
                                    
                                    tracking_info['smoothed_error'] = (smoothed_error_x, smoothed_error_y)
                                    tracking_info['raw_error'] = (error_x, error_y)
                            else:
                                tracking_info = {
                                    'status': 'no_face',
                                    'history_size': len(self.face_tracker.face_history)
                                }
                                print("[Vision] No face detected")
                                
                                # Handle return to center when no faces detected for a while
                                current_time = time.time()
                                if current_time - self.face_tracker.last_target_time > self.face_tracker.no_target_timeout:
                                    print("[Vision] No face detected for a while, returning to center")
                                    self.face_tracker._return_to_center()
                        else:
                            tracking_info = {
                                'status': 'tracking_disabled',
                                'reason': 'manual_cooldown' if (time.time() - self.face_tracker.last_manual_movement_time) < self.face_tracker.manual_movement_cooldown else 'disabled'
                            }
                    else:
                        # Regular capture without metadata
                        rgb = cam.capture_array()
                        tracking_info = {'status': 'face_detection_disabled'}
                    
                    # Store the latest frame for debugging
                    self.last_captured_frame = rgb.copy()
                    
                    # Send frame to Gemini if session is active
                    if self.session:
                        # Check if we should skip sending to avoid overwhelming API
                        is_receiving = False
                        for _ in range(3):
                            if not hasattr(self, 'audio_in_q') or not self.audio_in_q.empty():
                                is_receiving = True
                                break
                            await asyncio.sleep(0.05)
                        
                        if not is_receiving:
                            # Send frame to Gemini
                            jpeg_bytes = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, CAMERA_JPEG_QUALITY])[1].tobytes()
                            blob = types.Blob(
                                data=jpeg_bytes,
                                mime_type="image/jpeg"
                            )
                            
                            try:
                                await asyncio.wait_for(
                                    self.session.send_realtime_input(media=blob),
                                    timeout=2.0
                                )
                            except asyncio.TimeoutError:
                                print("[Vision] Frame send timed out, skipping")
                            except Exception as e:
                                print(f"[Vision] Frame send error: {e}")
                    
                    # Wait before next iteration - use face_tracking_interval for consistency
                    await asyncio.sleep(self.face_tracker.face_tracking_interval if self.face_tracker.face_detection_enabled else interval)
                    
                except Exception as e:
                    print(f"[Vision] Error in main loop: {e}")
                    await asyncio.sleep(interval)

        except Exception as e:
            print(f"[Vision] Feed loop error: {e}")
        finally:
            # Clean up camera
            cam.close()
            print("[Vision] Vision feed task exiting...")

    def set_session(self, session):
        """Set the Gemini session for vision feed"""
        self.session = session 