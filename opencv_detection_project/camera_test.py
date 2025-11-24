#!/usr/bin/env python3
"""
Camera Test Script for Raspberry Pi Camera Rev 1.3
Tests camera functionality with picamera library
"""

import cv2
import numpy as np
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import config

def test_camera():
    """Test camera capture and display"""
    print("Initializing Raspberry Pi Camera Rev 1.3...")
    print(f"Resolution: {config.CAMERA_RESOLUTION}")
    print(f"Framerate: {config.CAMERA_FRAMERATE}")

    # Initialize camera
    camera = PiCamera()
    camera.resolution = config.CAMERA_RESOLUTION
    camera.framerate = config.CAMERA_FRAMERATE
    camera.rotation = config.CAMERA_ROTATION

    # Initialize capture array
    raw_capture = PiRGBArray(camera, size=config.CAMERA_RESOLUTION)

    # Allow camera to warm up
    print(f"Warming up camera for {config.CAMERA_WARMUP_TIME} seconds...")
    time.sleep(config.CAMERA_WARMUP_TIME)

    print("\nCamera ready!")
    print("Press 'q' to quit")
    print("Press 's' to save a snapshot")

    frame_count = 0
    start_time = time.time()

    try:
        # Capture frames continuously
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            # Get the frame
            image = frame.array

            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")

            # Display FPS on frame
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add instructions
            cv2.putText(image, "Press 'q' to quit, 's' to save", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display frame
            cv2.imshow("Camera Test", image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, image)
                print(f"Snapshot saved: {filename}")

            # Clear the stream for next frame
            raw_capture.truncate(0)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        camera.close()
        cv2.destroyAllWindows()
        print("Camera closed successfully")

if __name__ == "__main__":
    print("="*50)
    print("Raspberry Pi Camera Test")
    print("="*50)
    test_camera()
