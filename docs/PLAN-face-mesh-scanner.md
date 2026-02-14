# Project Plan: Real-Time Face Mesh Scanner & Liveness Detection

> **Goal**: Create a lightweight, modular Python component to detect faces via webcam, overlay a "sci-fi" scanning animation, and perform passive liveness detection to reject 2D photos.

## 1. Project Overview

- **Input**: Real-time webcam feed (frame-by-frame).
- **Output**: Processed frames with UI overlays (mesh + scan line) and a status flag (Real/Fake).
- **Constraints**: 
  - Lightweight (standard laptop).
  - "UI Gimmick" focus (visually appealing).
  - Passive liveness detection (no user interaction required).
  - Open Source libraries only.

## 2. Technology Stack

- **Language**: Python 3.9+
- **Core Vision Library**: `opencv-python` (cv2) for image handling and drawing.
- **Face/Mesh Detection**: `mediapipe` (Google's lightweight, high-performance ML solution).
  - *Why*: Much faster and more robust than dlib/Haar cascades for mesh generation on CPU.
- **Math**: `numpy` for vector calculations (liveness logic).

## 3. Architecture Design

The system will be encapsulated in a single Python class `FaceScanner` to allow easy integration into other projects.

### Class Structure: `FaceScanner`
- `__init__(self, camera_id=0)`: Initialize MediaPipe models.
- `update_frame(self, frame)`: Core method to process a video frame.
  - Detects Face Mesh.
  - Calculates Liveness Score.
  - Updates Animation State (scan line position).
  - Draws Overlays.
- `_check_liveness(self, landmarks)`: Internal method for anti-spoofing logic.
- `_draw_hud(self, frame)`: Internal method for "sci-fi" visuals.

## 4. Liveness Detection Strategy (Lightweight & Passive)

Since we cannot use heavy Deep Learning models (requires PyTorch/TF generic dependencies) and need to stay "lightweight", we will use **Geometric Heuristics** based on MediaPipe's 3D landmarks:

1.  **3D Depth Consistency**:
    - MediaPipe returns `x, y, z` coordinates.
    - A 2D photo held up to a camera lacks relative depth depth variance between the nose (tip) and ears/ears-edge when the head rotates.
    - We will measure the depth delta between central and peripheral landmarks.
2.  **Micro-Motion / Blink**:
    - Calculate Eye Aspect Ratio (EAR).
    - If eyes never change aspect ratio over $N$ frames, likely a photo.
    - *Note*: This is passive; we just wait to see a blink.

## 5. Visual Effects (The "Gimmick")

- **Wireframe Mesh**: Draw connections between key face landmarks (tesselation) with a holographic color (Cyan/Green).
- **Scanning Beam**:
  - A horizontal bar moving up and down across the face bounding box.
  - Gradient trail behind the bar.
  - "Scanning..." text indicator.
- **Status feedback**:
  - **Valid Face**: Green Scan + "ACCESS GRANTED" style lock-on.
  - **2D/Spoof**: Red Scan + "WARNING: 2D DETECTED".

## 6. Implementation Phases

### Phase 1: Setup & Core Detection
- [ ] Initialize Python virtual environment.
- [ ] Install `opencv-python`, `mediapipe`, `numpy`.
- [ ] Create `FaceScanner` class foundation.
- [ ] Implement basic MediaPipe Face Mesh extraction.

### Phase 2: Liveness Logic (The "Brain")
- [ ] Implement `_get_depth_score()`: Analyze Z-axis depth of nose vs. face edges.
- [ ] Implement `_detect_blink()`: Calculate Eye Aspect Ratio (EAR).
- [ ] created weighted "Real Probability" score.

### Phase 3: UI & Animation (The "Beauty")
- [ ] Implement the "Scanner Bar" animation logic (sine wave or linear bounce).
- [ ] Draw the face mesh with custom lightweight drawing utils (not default MediaPipe styles, to look cooler).
- [ ] Add conditional coloring (Red = Fake, Green = Real).

### Phase 4: Integration & Optimization
- [ ] Optimize loop for >30 FPS on laptop CPU.
- [ ] Create a `main.py` demo script to run the webcam loop.
- [ ] Code cleanup and packaging.

## 7. Verification Plan

- **Test A (Photo)**: Hold a phone screen or printed photo of a face. System should display "FAKE" or "2D".
- **Test B (Real User)**: Sit in front of camera. System should display "REAL" and show mesh.
- **Test C (Performance)**: Ensure low CPU usage (<30% on standard core) and smooth animation.
