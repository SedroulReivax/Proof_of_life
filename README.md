# Face Mesh Scanner

Real-time face mesh scanner with passive liveness detection and sci-fi scanning animation.

## Features

- **Real-time Face Mesh Detection**: 468 3D facial landmarks using MediaPipe
- **Passive Liveness Detection**: Rejects 2D photos using depth variance and blink detection
- **Sci-Fi Scanning Animation**: Animated scanning beam with gradient trail
- **Visual Feedback**: Color-coded status (Green = Real, Red = 2D)
- **Lightweight**: CPU-only, runs on standard laptops

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

**Controls:**
- Press `q` to quit
- Press `r` to reset liveness tracking

## How It Works

### Liveness Detection

1. **Depth Variance Analysis** (60% weight)
   - Measures Z-axis variance across key facial landmarks
   - Real faces have 3D depth; photos are flat

2. **Blink Detection** (40% weight)
   - Calculates Eye Aspect Ratio (EAR)
   - Real faces blink naturally; photos don't

**Liveness Score:** Combines both methods, threshold at 0.6

### Visual Effects

- **Green Mesh**: Real face detected ("ACCESS GRANTED")
- **Red Mesh**: 2D photo detected ("WARNING: 2D DETECTED")
- **Scanning Beam**: Animated horizontal bar with gradient trail
- **HUD**: Real-time liveness score and blink counter

## Testing

**Test with Real Face:**
1. Run the demo
2. Position your face in front of webcam
3. Should show green mesh and "ACCESS GRANTED"
4. Blink counter should increment

**Test with Photo:**
1. Hold up a photo (printed or on phone screen)
2. Should show red mesh and "WARNING: 2D DETECTED"
3. Liveness score should be low

## Requirements

- Python 3.9+
- Webcam
- Dependencies: opencv-python, mediapipe, numpy

## Project Structure

```
Liveness/
├── src/
│   └── face_scanner.py    # FaceScanner class
├── main.py                # Demo application
├── requirements.txt       # Dependencies
└── docs/
    └── PLAN-face-mesh-scanner.md  # Implementation plan
```

## Technical Details

- **Framework**: MediaPipe Face Mesh
- **Performance**: ~30 FPS on standard laptop CPU
- **Liveness Methods**: Geometric heuristics (depth variance + EAR)
- **Face Landmarks**: 468 points with 3D coordinates

## License

Open source libraries used: OpenCV, MediaPipe, NumPy
