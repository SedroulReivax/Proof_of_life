"""
Face Mesh Scanner Demo
Real-time webcam demo of the FaceScanner class.
"""

import cv2
from src.face_scanner import FaceScanner


def main():
    """Run the face scanner demo."""
    # Initialize scanner
    scanner = FaceScanner()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Face Mesh Scanner Started")
    print("Instructions:")
    print("  - Point webcam at your face for real detection")
    print("  - Hold up a photo to test 2D detection")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset liveness tracking")
    print()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Process frame
        annotated_frame, liveness_score, is_real = scanner.update_frame(frame)
        
        # Display result
        cv2.imshow('Face Mesh Scanner', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            scanner.reset()
            print("Liveness tracking reset.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Face Mesh Scanner stopped.")


if __name__ == "__main__":
    main()
