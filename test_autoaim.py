#!/usr/bin/env python3
"""
AI-Enhanced Detection Script for Raspberry Pi 5 with IMX500 Camera
Uses embedded AI for fast person and face detection with servo auto-tracking
"""

import cv2
import logging
import time
import numpy as np
import sys
import os
from datetime import datetime
from picamera2 import Picamera2
from picamera2.devices import IMX500
from libcamera import Transform
import argparse
import pathlib

# Import servo control
try:
    # Add freenove_examples directory to Python path
    servo_path = os.path.join(os.path.dirname(__file__), 'freenove_examples')
    sys.path.insert(0, servo_path)
    from servo import Servo
    SERVO_AVAILABLE = True
except ImportError as e:
    SERVO_AVAILABLE = False
    print(f"Warning: Servo module not found: {e}. Servo tracking will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only log to console, not to file
    ]
)
logger = logging.getLogger(__name__)

class AIDetector:
    def __init__(self, detection_interval=0.5, detection_mode="person", save_detections=False, 
                 confidence_threshold=0.7, enable_tracking=False, pan_limits=(20, 160), 
                 tilt_limits=(30, 100)):
        """
        Initialize the AI detector using IMX500's embedded AI
        
        Args:
            detection_interval (float): Time between detection attempts in seconds
            detection_mode (str): "person" or "face" detection mode
            save_detections (bool): Whether to save images with detected objects (default: False)
            confidence_threshold (float): Minimum confidence for detections (0.0-1.0)
            enable_tracking (bool): Enable servo auto-tracking
            pan_limits (tuple): (min, max) servo angles for pan (servo 0) - default (20, 160)
            tilt_limits (tuple): (min, max) servo angles for tilt (servo 1) - default (30, 100)
        """
        self.detection_interval = detection_interval
        self.detection_mode = detection_mode
        self.camera = None
        self.imx500 = None
        self.running = False
        self.save_detections = save_detections
        self.confidence_threshold = confidence_threshold
        
        # Servo tracking parameters
        self.enable_tracking = enable_tracking and SERVO_AVAILABLE
        self.servo = None
        self.pan_limits = pan_limits  # (min, max) degrees
        self.tilt_limits = tilt_limits  # (min, max) degrees
        self.current_pan = 90  # Start at center
        self.current_tilt = 50  # Start at 50° to look up from floor level
        
        # Tracking parameters
        self.tracking_speed = 0.6  # Reduced from 0.8 to prevent overshooting
        self.deadzone = 0.02  # Don't move if target is within this fraction of center (reduced from 0.05)
        self.no_target_timeout = 5.0  # Seconds before returning to center when no target
        self.last_target_time = time.time()
        
        # Smoothing parameters to prevent oscillation
        self.target_history = []  # Store recent target positions for smoothing
        self.history_size = 3  # Number of recent positions to average
        self.max_move_per_step = 20  # Maximum degrees to move in one step (prevent big jumps)
        
        # COCO class labels for object detection
        self.coco_labels = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Create directory for saved detections
        if self.save_detections:
            self.detections_dir = pathlib.Path("ai_detections")
            self.detections_dir.mkdir(exist_ok=True)
            logger.info(f"Created directory for detections: {self.detections_dir}")
        
        # Initialize servo if tracking is enabled
        if self.enable_tracking:
            self._init_servo()
        
        # Initialize camera with AI model
        self._init_camera()
    
    def _init_servo(self):
        """Initialize servo control for tracking"""
        try:
            if SERVO_AVAILABLE:
                self.servo = Servo()
                # Move to starting position (center pan, 50° tilt to look up)
                self.servo.set_servo_pwm('0', self.current_pan)
                self.servo.set_servo_pwm('1', self.current_tilt)
                time.sleep(0.5)  # Give servos time to move
                logger.info(f"Servo tracking initialized - Pan: {self.current_pan}°, Tilt: {self.current_tilt}°")
                logger.info(f"Pan limits: {self.pan_limits}, Tilt limits: {self.tilt_limits}")
            else:
                logger.warning("Servo module not available, tracking disabled")
                self.enable_tracking = False
        except Exception as e:
            logger.error(f"Failed to initialize servo: {e}")
            self.enable_tracking = False
    
    def _init_camera(self):
        """Initialize the Raspberry Pi camera with IMX500 AI model"""
        try:
            # Configure model file based on detection mode
            if self.detection_mode == "person":
                # Use object detection model that can detect people
                model_file = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
                task = "object detection"
            elif self.detection_mode == "face":
                # Use pose estimation model which can detect facial keypoints
                model_file = "/usr/share/imx500-models/imx500_network_posenet.rpk"
                task = "pose estimation"
            else:
                raise ValueError(f"Unsupported detection mode: {self.detection_mode}")
            
            # Check if AI model exists
            if not os.path.exists(model_file):
                logger.error(f"AI model not found: {model_file}")
                logger.error("Please install IMX500 models with: sudo apt install imx500-models")
                raise FileNotFoundError("AI model not found")
            
            # Initialize IMX500 device with the model
            logger.info(f"Loading AI model: {model_file}")
            self.imx500 = IMX500(model_file)
            
            # Initialize Picamera2
            self.camera = Picamera2(self.imx500.camera_num)
            
            # Configure camera for AI detection
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480)},
                raw={"size": (2028, 1520)},
                encode="main",
                buffer_count=6
            )
            
            # Set transform if supported
            try:
                config["transform"] = Transform()
            except Exception as e:
                logger.warning(f"Could not set transform: {e}")
            
            self.camera.configure(config)
            
            # Set up network intrinsics for better detection
            if self.imx500.network_intrinsics:
                ni = self.imx500.network_intrinsics
                ni.task = task
                if self.detection_mode == "person":
                    ni.inference_rate = 30.0
                    ni.bbox_normalization = True
                    ni.bbox_order = "xy"
                    ni.labels = self.coco_labels
                    ni.ignore_dash_labels = True
                elif self.detection_mode == "face":
                    ni.inference_rate = 30.0
                    # PoseNet doesn't use bbox_normalization or labels the same way
                logger.info("Network intrinsics configured")
            
            logger.info(f"Camera initialized with IMX500 AI model")
            logger.info(f"Detection mode: {self.detection_mode}")
            logger.info(f"Model file: {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise
    
    def parse_detections(self, metadata):
        """
        Parse detection results from IMX500 metadata
        
        Args:
            metadata: Camera metadata containing AI results
            
        Returns:
            list: List of detected objects with bounding boxes and confidence
        """
        detections = []
        
        try:
            # Get AI outputs from IMX500
            outputs = self.imx500.get_outputs(metadata)
            if outputs is None:
                return detections
            
            # Debug: Print output shapes and first few values
            logger.debug(f"Number of outputs: {len(outputs)}")
            for i, output in enumerate(outputs):
                logger.debug(f"Output {i} shape: {output.shape}, dtype: {output.dtype}")
                if output.size > 0:
                    logger.debug(f"Output {i} first few values: {output.flat[:5]}")
            
            if self.detection_mode == "person":
                detections = self._parse_object_detection(outputs)
            elif self.detection_mode == "face":
                detections = self._parse_pose_estimation(outputs)
                        
        except Exception as e:
            logger.error(f"Error parsing AI metadata: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return detections
    
    def _parse_object_detection(self, outputs):
        """Parse SSD object detection outputs"""
        detections = []
        
        # For SSD MobileNet object detection, outputs typically contain:
        # outputs[0]: detection boxes [num_detections, 4]
        # outputs[1]: detection classes [num_detections]  
        # outputs[2]: detection scores [num_detections]
        # outputs[3]: number of detections [1]
        
        if len(outputs) >= 4:
            boxes = outputs[0]
            classes = outputs[1] 
            scores = outputs[2]
            num_detections_raw = outputs[3]
            
            # Handle different possible formats for num_detections
            if num_detections_raw.size > 0:
                num_detections = int(num_detections_raw.flat[0])
            else:
                num_detections = min(len(scores), len(classes), len(boxes))
            
            logger.debug(f"Raw num_detections: {num_detections}")
            logger.debug(f"Scores shape: {scores.shape}, Classes shape: {classes.shape}, Boxes shape: {boxes.shape}")
            
            # Limit to reasonable number of detections
            max_detections = min(num_detections, len(scores), 50)  # Cap at 50 detections max
            
            valid_detections = []
            
            for i in range(max_detections):
                if i >= len(scores) or i >= len(classes) or i >= len(boxes):
                    break
                    
                # Get confidence score - handle different possible ranges
                confidence_raw = float(scores[i])
                
                # Convert confidence to 0-1 range if it's in 0-100 range
                if confidence_raw > 1.0:
                    confidence = confidence_raw / 100.0
                else:
                    confidence = confidence_raw
                
                class_id = int(classes[i])
                
                # Filter by confidence threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Only include people (class_id 0 in COCO)
                if class_id != 0:
                    continue
                
                # Get bounding box
                if i >= len(boxes):
                    continue
                    
                bbox = boxes[i]
                if len(bbox) < 4:
                    continue
                
                # Handle different bbox formats: [y0, x0, y1, x1] or [x0, y0, x1, y1]
                # Try to detect the format by checking if coordinates are normalized
                if all(0 <= coord <= 1 for coord in bbox):
                    # Normalized coordinates - assume [y0, x0, y1, x1] format (common for SSD)
                    y0, x0, y1, x1 = bbox[:4]
                    x = x0
                    y = y0
                    w = x1 - x0
                    h = y1 - y0
                else:
                    # Pixel coordinates - assume [x0, y0, x1, y1] format
                    x0, y0, x1, y1 = bbox[:4]
                    x = x0
                    y = y0
                    w = x1 - x0
                    h = y1 - y0
                
                # Validate bounding box
                if w <= 0 or h <= 0:
                    continue
                
                # Get class name
                class_name = self.coco_labels[class_id] if 0 <= class_id < len(self.coco_labels) else f"class_{class_id}"
                
                valid_detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x, y, w, h]
                })
            
            # Apply Non-Maximum Suppression to remove overlapping detections
            detections = self._apply_nms(valid_detections, iou_threshold=0.5)
            
            # Limit final detections to reasonable number
            detections = detections[:10]  # Max 10 detections
            
            logger.debug(f"After filtering and NMS: {len(detections)} detections")
        
        return detections
    
    def _parse_pose_estimation(self, outputs):
        """Parse PoseNet pose estimation outputs for face detection"""
        detections = []
        
        # PoseNet outputs keypoints for human poses
        # We'll look for facial keypoints (nose, eyes, ears) to detect faces
        
        if len(outputs) == 0:
            return detections
        
        logger.debug("PoseNet output analysis:")
        for i, output in enumerate(outputs):
            logger.debug(f"  Output {i}: shape={output.shape}, min={output.min():.3f}, max={output.max():.3f}")
        
        # PoseNet typically outputs heatmaps for keypoints
        # For COCO pose estimation, facial keypoints are:
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        
        if len(outputs) > 0:
            heatmaps = outputs[0]  # Main output should be keypoint heatmaps
            
            # Expected shape is typically (height, width, num_keypoints) or (num_keypoints, height, width)
            if len(heatmaps.shape) == 3:
                if heatmaps.shape[2] >= 5:  # (H, W, keypoints)
                    h, w, num_keypoints = heatmaps.shape
                elif heatmaps.shape[0] >= 5:  # (keypoints, H, W)
                    num_keypoints, h, w = heatmaps.shape
                    heatmaps = np.transpose(heatmaps, (1, 2, 0))  # Convert to (H, W, keypoints)
                else:
                    logger.debug(f"Unexpected heatmap shape: {heatmaps.shape}")
                    return detections
            else:
                logger.debug(f"Unexpected heatmap dimensions: {heatmaps.shape}")
                return detections
            
            # Define facial keypoint indices (COCO format)
            facial_keypoints = {
                'nose': 0,
                'left_eye': 1,
                'right_eye': 2,
                'left_ear': 3,
                'right_ear': 4
            }
            
            # Find facial keypoints
            face_keypoints = {}
            face_confidences = {}
            
            for name, idx in facial_keypoints.items():
                if idx < num_keypoints:
                    keypoint_heatmap = heatmaps[:, :, idx]
                    max_confidence = np.max(keypoint_heatmap)
                    
                    if max_confidence > self.confidence_threshold:
                        # Find the location of maximum confidence
                        y_idx, x_idx = np.unravel_index(np.argmax(keypoint_heatmap), keypoint_heatmap.shape)
                        
                        # Convert to normalized coordinates (0-1)
                        x_norm = x_idx / w
                        y_norm = y_idx / h
                        
                        face_keypoints[name] = (x_norm, y_norm)
                        face_confidences[name] = max_confidence
                        
                        logger.debug(f"{name}: confidence={max_confidence:.3f}, pos=({x_norm:.3f}, {y_norm:.3f})")
            
            # If we found facial keypoints, create a bounding box
            if len(face_keypoints) >= 2:  # Need at least 2 facial keypoints
                # Extract x and y coordinates
                x_coords = [pos[0] for pos in face_keypoints.values()]
                y_coords = [pos[1] for pos in face_keypoints.values()]
                
                # Calculate bounding box with some padding
                min_x = min(x_coords)
                max_x = max(x_coords)
                min_y = min(y_coords)
                max_y = max(y_coords)
                
                # Add padding around the keypoints to create a face bounding box
                padding_x = (max_x - min_x) * 0.5 if max_x > min_x else 0.1
                padding_y = (max_y - min_y) * 0.5 if max_y > min_y else 0.1
                
                bbox_x = max(0, min_x - padding_x)
                bbox_y = max(0, min_y - padding_y)
                bbox_w = min(1 - bbox_x, (max_x - min_x) + 2 * padding_x)
                bbox_h = min(1 - bbox_y, (max_y - min_y) + 2 * padding_y)
                
                # Calculate average confidence
                avg_confidence = np.mean(list(face_confidences.values()))
                
                # Only add detection if confidence is high enough
                if avg_confidence > self.confidence_threshold:
                    detections.append({
                        "class_id": 0,
                        "class_name": "face",
                        "confidence": float(avg_confidence),
                        "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                        "keypoints": face_keypoints  # Store keypoints for debugging
                    })
                    
                    logger.debug(f"Face detected: bbox=({bbox_x:.3f}, {bbox_y:.3f}, {bbox_w:.3f}, {bbox_h:.3f})")
                    logger.debug(f"Keypoints found: {list(face_keypoints.keys())}")
        
        return detections
    
    def _track_target(self, detections, frame_shape):
        """
        Track the detected target with servo movement
        
        Args:
            detections: List of detected objects
            frame_shape: Shape of the frame (height, width, channels)
        """
        if not self.enable_tracking or not self.servo:
            return
        
        current_time = time.time()
        
        if len(detections) > 0:
            # Get the first (highest confidence) detection
            target = detections[0]
            bbox = target["bbox"]
            
            # Calculate target center in image coordinates
            target_center_x, target_center_y = self._get_target_center(bbox, frame_shape)
            
            # Calculate error from image center (normalized -1 to 1)
            image_center_x = frame_shape[1] / 2
            image_center_y = frame_shape[0] / 2
            
            error_x = (target_center_x - image_center_x) / (frame_shape[1] / 2)
            error_y = (target_center_y - image_center_y) / (frame_shape[0] / 2)
            
            # Add to target history for smoothing
            self.target_history.append((error_x, error_y))
            if len(self.target_history) > self.history_size:
                self.target_history.pop(0)
            
            # Calculate smoothed error (weighted average of recent positions)
            if len(self.target_history) >= 2:
                # Weight recent positions more heavily
                weights = [1.0, 1.5, 2.0][:len(self.target_history)]
                total_weight = sum(weights)
                
                smoothed_error_x = sum(w * pos[0] for w, pos in zip(weights, self.target_history)) / total_weight
                smoothed_error_y = sum(w * pos[1] for w, pos in zip(weights, self.target_history)) / total_weight
            else:
                smoothed_error_x = error_x
                smoothed_error_y = error_y
            
            # Apply deadzone to smoothed error
            if abs(smoothed_error_x) < self.deadzone:
                smoothed_error_x = 0
            if abs(smoothed_error_y) < self.deadzone:
                smoothed_error_y = 0
            
            # Update servo positions if there's significant error
            if smoothed_error_x != 0 or smoothed_error_y != 0:
                self._move_servos_smooth(smoothed_error_x, smoothed_error_y)
                self.last_target_time = current_time
                
                logger.debug(f"Tracking: target=({target_center_x:.0f},{target_center_y:.0f}), "
                           f"raw_error=({error_x:.3f},{error_y:.3f}), "
                           f"smooth_error=({smoothed_error_x:.3f},{smoothed_error_y:.3f}), "
                           f"servos=({self.current_pan:.1f}°,{self.current_tilt:.1f}°)")
        else:
            # No target detected - clear history and return to center after timeout
            self.target_history.clear()
            if current_time - self.last_target_time > self.no_target_timeout:
                self._return_to_center()
    
    def _get_target_center(self, bbox, frame_shape):
        """
        Calculate the center of the target in pixel coordinates
        
        Args:
            bbox: Bounding box [x, y, w, h] (may be normalized or pixel coordinates)
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            tuple: (center_x, center_y) in pixel coordinates
        """
        x, y, w, h = bbox
        
        # Convert to pixel coordinates if normalized
        if x <= 1.0 and y <= 1.0:  # Normalized coordinates
            frame_height, frame_width = frame_shape[:2]
            x = int(x * frame_width)
            y = int(y * frame_height)
            w = int(w * frame_width)
            h = int(h * frame_height)
        
        center_x = x + w // 2
        center_y = y + h // 2
        
        return center_x, center_y
    
    def _move_servos_smooth(self, error_x, error_y):
        """
        Move servos with smoothing and step limiting to prevent oscillation
        
        Args:
            error_x: Horizontal error (-1 to 1, positive = target is right of center)
            error_y: Vertical error (-1 to 1, positive = target is below center)
        """
        # Calculate direct servo positions based on error
        pan_range = self.pan_limits[1] - self.pan_limits[0]
        pan_center = (self.pan_limits[0] + self.pan_limits[1]) / 2
        
        tilt_range = self.tilt_limits[1] - self.tilt_limits[0]
        tilt_center = (self.tilt_limits[0] + self.tilt_limits[1]) / 2
        
        # Calculate target servo positions
        pan_offset = error_x * (pan_range / 2) * self.tracking_speed
        tilt_offset = error_y * (tilt_range / 2) * self.tracking_speed
        
        target_pan = pan_center + pan_offset
        target_tilt = tilt_center + tilt_offset
        
        # Apply safety limits
        target_pan = max(self.pan_limits[0], min(self.pan_limits[1], target_pan))
        target_tilt = max(self.tilt_limits[0], min(self.tilt_limits[1], target_tilt))
        
        # Limit movement speed to prevent big jumps
        pan_diff = target_pan - self.current_pan
        tilt_diff = target_tilt - self.current_tilt
        
        # Clamp movement to max_move_per_step
        if abs(pan_diff) > self.max_move_per_step:
            pan_diff = self.max_move_per_step * (1 if pan_diff > 0 else -1)
        if abs(tilt_diff) > self.max_move_per_step:
            tilt_diff = self.max_move_per_step * (1 if tilt_diff > 0 else -1)
        
        new_pan = self.current_pan + pan_diff
        new_tilt = self.current_tilt + tilt_diff
        
        # Move if there's significant change
        if abs(new_pan - self.current_pan) > 0.5 or abs(new_tilt - self.current_tilt) > 0.5:
            self.current_pan = new_pan
            self.current_tilt = new_tilt
            
            # Move servos to new position
            self.servo.set_servo_pwm('0', int(self.current_pan))
            self.servo.set_servo_pwm('1', int(self.current_tilt))
            
            logger.debug(f"Servo moved smoothly: Pan={self.current_pan:.1f}° (Δ{pan_diff:+.1f}°), "
                        f"Tilt={self.current_tilt:.1f}° (Δ{tilt_diff:+.1f}°)")
    
    def _move_servos(self, error_x, error_y):
        """
        Legacy method - redirects to smooth movement
        """
        self._move_servos_smooth(error_x, error_y)
    
    def _return_to_center(self):
        """Return servos to center position when no target is detected"""
        target_pan = 90  # Center pan position
        target_tilt = 50  # Center tilt position (looking up for floor-mounted camera)
        
        # Move gradually towards center
        pan_diff = target_pan - self.current_pan
        tilt_diff = target_tilt - self.current_tilt
        
        # Small movements towards center
        if abs(pan_diff) > 1.0:
            self.current_pan += np.sign(pan_diff) * min(1.0, abs(pan_diff) * 0.1)
        else:
            self.current_pan = target_pan
            
        if abs(tilt_diff) > 1.0:
            self.current_tilt += np.sign(tilt_diff) * min(1.0, abs(tilt_diff) * 0.1)
        else:
            self.current_tilt = target_tilt
        
        # Move servos
        self.servo.set_servo_pwm('0', int(self.current_pan))
        self.servo.set_servo_pwm('1', int(self.current_tilt))
        
        logger.debug(f"Returning to center: Pan={self.current_pan:.1f}°, Tilt={self.current_tilt:.1f}°")
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression to filter overlapping detections
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression
            
        Returns:
            List of filtered detections
        """
        if len(detections) == 0:
            return detections
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        
        for i, detection in enumerate(detections):
            # Check if this detection overlaps too much with any already selected detection
            keep = True
            bbox1 = detection['bbox']
            
            for selected in filtered_detections:
                bbox2 = selected['bbox']
                iou = self._calculate_iou(bbox1, bbox2)
                
                if iou > iou_threshold:
                    keep = False
                    break
            
            if keep:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) of two bounding boxes
        
        Args:
            bbox1, bbox2: [x, y, w, h] format bounding boxes
            
        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to [x1, y1, x2, y2] format
        x1_max = x1 + w1
        y1_max = y1 + h1
        x2_max = x2 + w2
        y2_max = y2 + h2
        
        # Calculate intersection
        intersect_x1 = max(x1, x2)
        intersect_y1 = max(y1, y2)
        intersect_x2 = min(x1_max, x2_max)
        intersect_y2 = min(y1_max, y2_max)
        
        if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
            return 0.0
        
        intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersect_area
        
        if union_area <= 0:
            return 0.0
        
        return intersect_area / union_area
    
    def save_detection_image(self, frame, detections):
        """
        Save an image with detection boxes
        
        Args:
            frame: The original frame
            detections: List of detected objects
        """
        if not self.save_detections or not detections:
            return
            
        # Create a copy of the frame to draw on
        frame_with_boxes = frame.copy()
        
        # Draw rectangles around detections
        for detection in detections:
            bbox = detection["bbox"]
            x, y, w, h = bbox
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            
            # Convert relative coordinates to absolute if needed
            if x <= 1.0 and y <= 1.0:  # Relative coordinates
                frame_height, frame_width = frame.shape[:2]
                x = int(x * frame_width)
                y = int(y * frame_height)
                w = int(w * frame_width)
                h = int(h * frame_height)
            else:
                # Already absolute coordinates
                x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Draw rectangle
            cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame_with_boxes, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Add text with class and confidence
            rel_x = (center_x / frame.shape[1]) * 100
            rel_y = (center_y / frame.shape[0]) * 100
            text = f"{class_name}: {confidence:.2f} ({center_x}, {center_y}) - ({rel_x:.1f}%, {rel_y:.1f}%)"
            cv2.putText(frame_with_boxes, text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.detections_dir / f"ai_detection_{timestamp}.jpg"
        
        # Save the image
        cv2.imwrite(str(filename), frame_with_boxes)
        logger.info(f"Saved AI detection image: {filename}")
    
    def log_detections(self, detections, frame_shape):
        """
        Log detection results - simplified to only show face detection status
        """
        if len(detections) == 0:
            logger.info("No face detected")
        else:
            confidence = detections[0]["confidence"]
            logger.info(f"Face detected (confidence: {confidence:.2f})")
    
    def start_detection(self):
        """Start the AI detection loop"""
        try:
            self.camera.start()
            logger.info("Starting face detection...")
            logger.info("Press Ctrl+C to stop")
            
            self.running = True
            
            while self.running:
                try:
                    # Capture frame with metadata
                    request = self.camera.capture_request()
                    frame = request.make_array("main")
                    metadata = request.get_metadata()
                    request.release()
                    
                    # Convert from RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Parse AI detections from metadata
                    detections = self.parse_detections(metadata)
                    
                    # Track target
                    self._track_target(detections, frame_bgr.shape)
                    
                    # Save detection image if detections found and saving is enabled
                    if len(detections) > 0 and self.save_detections:
                        self.save_detection_image(frame_bgr, detections)
                    
                    # Log results
                    self.log_detections(detections, frame_bgr.shape)
                    
                    # Wait for next detection
                    time.sleep(self.detection_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Detection stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error during detection: {e}")
                    time.sleep(1)  # Brief pause before retrying
            
        except Exception as e:
            logger.error(f"Failed to start detection: {e}")
        finally:
            self.stop_detection()
    
    def stop_detection(self):
        """Stop the detection and cleanup"""
        self.running = False
        
        # Reset servos to center before stopping
        if self.enable_tracking and self.servo:
            logger.info("Resetting servos to center position...")
            self.servo.set_servo_pwm('0', 90)  # Center pan
            self.servo.set_servo_pwm('1', 50)  # Center tilt (looking up for floor-mounted camera)
            time.sleep(0.5)
        
        if self.camera:
            self.camera.stop()
            logger.info("Camera stopped")
    
    def capture_debug_frame(self, filename="debug_ai_frame.jpg"):
        """
        Capture and save a single frame for debugging
        
        Args:
            filename (str): Output filename for the captured frame
        """
        try:
            if not self.camera.started:
                self.camera.start()
                time.sleep(2)  # Let camera warm up
            
            # Capture frame with metadata
            request = self.camera.capture_request()
            frame = request.make_array("main")
            metadata = request.get_metadata()
            request.release()
            
            # Convert from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Parse AI detections
            detections = self.parse_detections(metadata)
            
            # Draw detections
            for detection in detections:
                bbox = detection["bbox"]
                x, y, w, h = bbox
                confidence = detection["confidence"]
                class_name = detection["class_name"]
                
                # Convert relative coordinates to absolute if needed
                if x <= 1.0 and y <= 1.0:
                    frame_height, frame_width = frame_bgr.shape[:2]
                    x = int(x * frame_width)
                    y = int(y * frame_height)
                    w = int(w * frame_width)
                    h = int(h * frame_height)
                else:
                    x, y, w, h = int(x), int(y), int(w), int(h)
                
                cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(frame_bgr, (center_x, center_y), 5, (0, 0, 255), -1)
                
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame_bgr, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(filename, frame_bgr)
            logger.info(f"Debug frame saved as {filename} with {len(detections)} detection(s) marked")
            
            if not self.running:
                self.camera.stop()
            
        except Exception as e:
            logger.error(f"Failed to capture debug frame: {e}")

def main():
    parser = argparse.ArgumentParser(description='AI Detection for Raspberry Pi IMX500 Camera with Servo Tracking')
    parser.add_argument('--interval', type=float, default=0.5,
                       help='Detection interval in seconds (default: 0.5)')
    parser.add_argument('--mode', choices=['person', 'face'], default='person',
                       help='Detection mode: person or face (default: person)')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold (0.0-1.0, default: 0.7)')
    parser.add_argument('--debug', action='store_true',
                       help='Capture and save a debug frame with detected objects marked')
    parser.add_argument('--save', action='store_true',
                       help='Enable saving detection images (disabled by default)')
    
    # Servo tracking arguments
    parser.add_argument('--track', action='store_true',
                       help='Enable servo auto-tracking')
    parser.add_argument('--pan-limits', nargs=2, type=int, default=[20, 160],
                       help='Pan servo limits in degrees (default: 20 160)')
    parser.add_argument('--tilt-limits', nargs=2, type=int, default=[30, 100],
                       help='Tilt servo limits in degrees (default: 30 100)')
    parser.add_argument('--tracking-speed', type=float, default=0.6,
                       help='Tracking speed (0.1=slow, 1.0=fast, default: 0.6)')
    
    args = parser.parse_args()
    
    # Validate servo limits
    if args.pan_limits[0] >= args.pan_limits[1]:
        parser.error("Pan min limit must be less than max limit")
    if args.tilt_limits[0] >= args.tilt_limits[1]:
        parser.error("Tilt min limit must be less than max limit")
    if not (0 <= args.pan_limits[0] <= 180 and 0 <= args.pan_limits[1] <= 180):
        parser.error("Pan limits must be between 0 and 180 degrees")
    if not (0 <= args.tilt_limits[0] <= 180 and 0 <= args.tilt_limits[1] <= 180):
        parser.error("Tilt limits must be between 0 and 180 degrees")
    
    try:
        detector = AIDetector(
            detection_interval=args.interval,
            detection_mode=args.mode,
            save_detections=args.save,
            confidence_threshold=args.confidence,
            enable_tracking=args.track,
            pan_limits=tuple(args.pan_limits),
            tilt_limits=tuple(args.tilt_limits)
        )
        
        # Set tracking speed if specified
        if hasattr(detector, 'tracking_speed'):
            detector.tracking_speed = args.tracking_speed
        
        if args.debug:
            detector.capture_debug_frame()
        else:
            detector.start_detection()
            
    except Exception as e:
        logger.error(f"Failed to start AI detector: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
