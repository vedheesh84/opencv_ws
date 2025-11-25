#!/usr/bin/env python3
"""Simple camera view using OpenCV VideoCapture"""

import cv2
import config

# Initialize camera
cap = cv2.VideoCapture(config.CAMERA_INDEX)

if not cap.isOpened():
    print("Failed to open camera!")
    exit(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])

print("Press 'q' to quit...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        cv2.imshow('Live Camera Feed', frame)

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
