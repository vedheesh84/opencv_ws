#!/usr/bin/env python3
"""
HSV Color Tuner for Raspberry Pi Camera
Adjust sliders to find the correct HSV values for color detection.

Controls:
  - Adjust trackbars to isolate your target color
  - Press 'p' to print current values
  - Press 'q' to quit and print final values
"""

import cv2
import numpy as np

def nothing(x):
    pass

def main():
    print("="*60)
    print("HSV Color Tuner")
    print("="*60)
    print("\nInstructions:")
    print("1. Point camera at a colored object")
    print("2. Adjust sliders until ONLY that color is white in mask")
    print("3. Press 'p' to print values, 'q' to quit")
    print("="*60)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create window with trackbars
    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 300)

    # Create trackbars for HSV range
    # Start with full range so you see something
    cv2.createTrackbar('Low H', 'Trackbars', 0, 179, nothing)
    cv2.createTrackbar('High H', 'Trackbars', 179, 179, nothing)
    cv2.createTrackbar('Low S', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('High S', 'Trackbars', 255, 255, nothing)
    cv2.createTrackbar('Low V', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('High V', 'Trackbars', 255, 255, nothing)

    print("\nCamera ready. Adjust the trackbars...")
    print("Mask window: WHITE = detected, BLACK = not detected\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Apply blur to reduce noise
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convert to HSV
        hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)

        # Get trackbar positions
        l_h = cv2.getTrackbarPos('Low H', 'Trackbars')
        h_h = cv2.getTrackbarPos('High H', 'Trackbars')
        l_s = cv2.getTrackbarPos('Low S', 'Trackbars')
        h_s = cv2.getTrackbarPos('High S', 'Trackbars')
        l_v = cv2.getTrackbarPos('Low V', 'Trackbars')
        h_v = cv2.getTrackbarPos('High V', 'Trackbars')

        # Create mask
        lower = np.array([l_h, l_s, l_v])
        upper = np.array([h_h, h_s, h_v])
        mask = cv2.inRange(hsv, lower, upper)

        # Apply mask to original frame
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Add text overlay to original frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Low HSV: ({l_h}, {l_s}, {l_v})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"High HSV: ({h_h}, {h_s}, {h_v})", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'p' to print, 'q' to quit", (10, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show windows
        cv2.imshow('Original', display_frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n" + "="*60)
            print("FINAL HSV VALUES:")
            print("="*60)
            print(f"Lower: ({l_h}, {l_s}, {l_v})")
            print(f"Upper: ({h_h}, {h_s}, {h_v})")
            print("\nFor config.py COLOR_RANGES, use:")
            print(f"    'color_name': [(({l_h}, {l_s}, {l_v}), ({h_h}, {h_s}, {h_v}))],")
            print("="*60)
            break

        elif key == ord('p'):
            print(f"\nCurrent: Low=({l_h}, {l_s}, {l_v}), High=({h_h}, {h_s}, {h_v})")
            print(f"For config.py: [(({l_h}, {l_s}, {l_v}), ({h_h}, {h_s}, {h_v}))]")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
