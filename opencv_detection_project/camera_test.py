#!/usr/bin/env python3
"""
Camera Test Script using OpenCV VideoCapture
For Raspberry Pi with CSI Camera (V4L2 backend)
"""

import cv2
import numpy as np
import time
import config

def test_camera():
    """Test camera capture using OpenCV"""
    print("Initializing camera with OpenCV VideoCapture...")
    print(f"Camera index: {config.CAMERA_INDEX}")
    print(f"Resolution: {config.CAMERA_RESOLUTION}")

    # Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("Failed to open camera!")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    # Verify settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Actual resolution: {actual_width}x{actual_height}")
    print(f"Actual FPS: {actual_fps}")

    print("Camera started!")

    # Warm up camera
    print("Warming up camera for 2 seconds...")
    time.sleep(2)

    print("\nCamera ready!")
    print("Press 'q' to quit")
    print("Press 's' to save a snapshot")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame")
                break

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Display FPS on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display frame
            cv2.imshow("Camera Test - OpenCV", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Snapshot saved: {filename}")

            # Print FPS every 30 frames
            if frame_count % 30 == 0:
                print(f"FPS: {fps:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed successfully")

if __name__ == "__main__":
    print("="*50)
    print("Raspberry Pi Camera Test (OpenCV VideoCapture)")
    print("="*50)
    test_camera()
