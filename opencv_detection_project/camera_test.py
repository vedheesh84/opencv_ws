#!/usr/bin/env python3
"""
Camera Test Script for Raspberry Pi Camera Rev 1.3 on Ubuntu 22.04
Uses Picamera2 (libcamera-based) instead of legacy picamera
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2
import config

def test_camera():
    """Test camera capture and display"""
    print("Initializing Raspberry Pi Camera Rev 1.3 with Picamera2...")
    print(f"Resolution: {config.CAMERA_RESOLUTION}")
    print(f"Format: {config.CAMERA_CONFIG['format']}")

    # Initialize Picamera2
    picam2 = Picamera2()

    # Create camera configuration
    camera_config = picam2.create_preview_configuration(
        main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
    )

    # Configure camera
    picam2.configure(camera_config)

    # Apply rotation if needed
    if config.CAMERA_ROTATION != 0:
        picam2.set_controls({"Transform": config.CAMERA_ROTATION})

    # Start camera
    print("Starting camera...")
    picam2.start()

    # Allow camera to warm up
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
            frame = picam2.capture_array()

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Display FPS on frame
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add instructions
            cv2.putText(frame_bgr, "Press 'q' to quit, 's' to save", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display frame
            cv2.imshow("Camera Test - Picamera2", frame_bgr)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame_bgr)
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
        picam2.stop()
        cv2.destroyAllWindows()
        print("Camera closed successfully")

if __name__ == "__main__":
    print("="*50)
    print("Raspberry Pi Camera Test (Picamera2/Ubuntu)")
    print("="*50)
    test_camera()
