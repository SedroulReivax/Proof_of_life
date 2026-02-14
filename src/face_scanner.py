"""
Face Mesh Scanner with Liveness Detection
A lightweight, modular component for real-time face mesh detection and sci-fi scanning animation.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceScanner:
    """Real-time face mesh scanner with passive liveness detection."""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh and animation state."""
        # Initialize MediaPipe Face Landmarker
        model_path = 'face_landmarker.task'
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(options)
        
        # Animation state
        self.scan_position = 0
        self.scan_direction = 1
        self.scan_speed = 5
        
        # Liveness tracking
        self.ear_history = []
        self.depth_history = []
        self.max_history = 30
        self.blink_detected = False
        self.blink_counter = 0
        
        # Eye landmarks indices for EAR calculation
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
    def update_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Process a video frame and return annotated frame with liveness info.
        
        Args:
            frame: Input BGR frame from webcam
            
        Returns:
            Tuple of (annotated_frame, liveness_score, is_real)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face mesh
        results = self.face_mesh.process(rgb_frame)
        
        liveness_score = 0.0
        is_real = False
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate liveness score
            liveness_score = self._get_liveness_score(face_landmarks, frame.shape)
            is_real = liveness_score > 0.6
            
            # Draw HUD and mesh
            frame = self._draw_hud(frame, face_landmarks, liveness_score, is_real)
            
            # Update scan animation
            self._update_scan_animation()
        else:
            # No face detected
            self._draw_no_face_message(frame)
        
        return frame, liveness_score, is_real
    
    def _get_liveness_score(self, face_landmarks, frame_shape: Tuple[int, int, int]) -> float:
        """
        Calculate liveness score based on depth variance and eye aspect ratio.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_shape: Shape of the video frame (height, width, channels)
            
        Returns:
            Liveness score between 0.0 (fake) and 1.0 (real)
        """
        landmarks = face_landmarks.landmark
        
        # 1. Calculate depth variance (Z-axis)
        depth_score = self._calculate_depth_score(landmarks)
        
        # 2. Calculate Eye Aspect Ratio (blink detection)
        ear_score = self._calculate_ear_score(landmarks)
        
        # 3. Weighted combination
        liveness_score = (depth_score * 0.6) + (ear_score * 0.4)
        
        return np.clip(liveness_score, 0.0, 1.0)
    
    def _calculate_depth_score(self, landmarks) -> float:
        """Calculate score based on 3D depth variance."""
        # Get key landmarks (nose tip, left ear, right ear, forehead, chin)
        nose_tip = landmarks[1]  # Nose tip
        left_ear = landmarks[234]
        right_ear = landmarks[454]
        forehead = landmarks[10]
        chin = landmarks[152]
        
        # Extract Z coordinates (depth)
        depths = np.array([
            nose_tip.z,
            left_ear.z,
            right_ear.z,
            forehead.z,
            chin.z
        ])
        
        # Calculate variance
        depth_variance = np.var(depths)
        
        # Store in history
        self.depth_history.append(depth_variance)
        if len(self.depth_history) > self.max_history:
            self.depth_history.pop(0)
        
        # Real faces have higher depth variance (> 0.001)
        # 2D photos have low variance (< 0.0005)
        avg_variance = np.mean(self.depth_history) if self.depth_history else 0
        
        if avg_variance > 0.001:
            return 1.0
        elif avg_variance > 0.0005:
            return 0.5
        else:
            return 0.0
    
    def _calculate_ear_score(self, landmarks) -> float:
        """Calculate Eye Aspect Ratio for blink detection."""
        def eye_aspect_ratio(eye_points):
            # Compute vertical distances
            v1 = np.linalg.norm(
                np.array([eye_points[1].x, eye_points[1].y]) - 
                np.array([eye_points[5].x, eye_points[5].y])
            )
            v2 = np.linalg.norm(
                np.array([eye_points[2].x, eye_points[2].y]) - 
                np.array([eye_points[4].x, eye_points[4].y])
            )
            
            # Compute horizontal distance
            h = np.linalg.norm(
                np.array([eye_points[0].x, eye_points[0].y]) - 
                np.array([eye_points[3].x, eye_points[3].y])
            )
            
            # EAR formula
            ear = (v1 + v2) / (2.0 * h)
            return ear
        
        # Get eye landmarks
        left_eye_points = [landmarks[i] for i in self.LEFT_EYE]
        right_eye_points = [landmarks[i] for i in self.RIGHT_EYE]
        
        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        ear = (left_ear + right_ear) / 2.0
        
        # Store in history
        self.ear_history.append(ear)
        if len(self.ear_history) > self.max_history:
            self.ear_history.pop(0)
        
        # Detect blink (EAR drops below threshold)
        EAR_THRESHOLD = 0.2
        if ear < EAR_THRESHOLD:
            self.blink_detected = True
            self.blink_counter += 1
        
        # Calculate variance in EAR (real faces blink, photos don't)
        if len(self.ear_history) > 10:
            ear_variance = np.var(self.ear_history)
            
            # Real faces have EAR variance > 0.001
            if ear_variance > 0.001 or self.blink_counter > 0:
                return 1.0
            else:
                return 0.0
        
        return 0.5  # Not enough data yet
    
    def _draw_hud(self, frame: np.ndarray, face_landmarks, liveness_score: float, is_real: bool) -> np.ndarray:
        """Draw sci-fi HUD with face mesh and scanning animation."""
        h, w, _ = frame.shape
        landmarks = face_landmarks.landmark
        
        # Choose color based on liveness
        if is_real:
            mesh_color = (0, 255, 0)  # Green
            status_text = "ACCESS GRANTED"
            status_color = (0, 255, 0)
        else:
            mesh_color = (0, 0, 255)  # Red
            status_text = "WARNING: 2D DETECTED"
            status_color = (0, 0, 255)
        
        # Draw face mesh (wireframe)
        self._draw_face_mesh(frame, landmarks, mesh_color, w, h)
        
        # Get face bounding box
        x_coords = [landmark.x * w for landmark in landmarks]
        y_coords = [landmark.y * h for landmark in landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Draw scanning beam
        self._draw_scan_beam(frame, x_min, x_max, y_min, y_max, mesh_color)
        
        # Draw status text
        cv2.putText(frame, status_text, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        # Draw liveness score
        score_text = f"Liveness: {liveness_score:.2f}"
        cv2.putText(frame, score_text, (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw blink counter
        if self.blink_counter > 0:
            blink_text = f"Blinks: {self.blink_counter}"
            cv2.putText(frame, blink_text, (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _draw_face_mesh(self, frame: np.ndarray, landmarks, color: Tuple[int, int, int], w: int, h: int):
        """Draw wireframe face mesh."""
        # MediaPipe Face Mesh tesselation connections
        FACE_CONNECTIONS = self.mp_face_mesh.FACEMESH_TESSELATION
        
        for connection in FACE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            
            start = (int(start_point.x * w), int(start_point.y * h))
            end = (int(end_point.x * w), int(end_point.y * h))
            
            cv2.line(frame, start, end, color, 1, cv2.LINE_AA)
    
    def _draw_scan_beam(self, frame: np.ndarray, x_min: int, x_max: int, 
                       y_min: int, y_max: int, color: Tuple[int, int, int]):
        """Draw animated scanning beam."""
        # Calculate beam position
        beam_y = y_min + int((y_max - y_min) * (self.scan_position / 100))
        
        # Draw beam with gradient effect
        beam_thickness = 3
        
        # Main beam line
        cv2.line(frame, (x_min, beam_y), (x_max, beam_y), color, beam_thickness)
        
        # Gradient trail
        for i in range(1, 15):  # Reduced range for a tighter glow
            alpha = 1.0 - (i / 15)  # Fading effect
            trail_color = tuple(int(c * alpha) for c in color)
            
            # Above the beam
            trail_y_above = beam_y - i * 2
            if trail_y_above >= y_min:
                cv2.line(frame, (x_min, trail_y_above), (x_max, trail_y_above), trail_color, 1)
            
            # Below the beam
            trail_y_below = beam_y + i * 2
            if trail_y_below <= y_max:
                cv2.line(frame, (x_min, trail_y_below), (x_max, trail_y_below), trail_color, 1)
    
    def _update_scan_animation(self):
        """Update scanning beam position."""
        self.scan_position += self.scan_speed * self.scan_direction
        
        # Bounce at edges
        if self.scan_position >= 100 or self.scan_position <= 0:
            self.scan_direction *= -1
    
    def _draw_no_face_message(self, frame: np.ndarray):
        """Draw message when no face is detected."""
        h, w, _ = frame.shape
        text = "NO FACE DETECTED"
        cv2.putText(frame, text, (w // 2 - 200, h // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
    
    def reset(self):
        """Reset liveness tracking state."""
        self.ear_history.clear()
        self.depth_history.clear()
        self.blink_detected = False
        self.blink_counter = 0
