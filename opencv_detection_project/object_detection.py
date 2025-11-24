#!/usr/bin/env python3
"""
Haar Cascade Object Detection Script for Ubuntu 22.04
Uses Picamera2 for camera access
"""

import cv2
import numpy as np
import time
import os
from picamera2 import Picamera2
import config

class ObjectDetector:
    def __init__(self, cascade_type='face'):
        """
        Initialize object detector with specified cascade type
        cascade_type: 'face', 'eye', 'fullbody', or 'upperbody'
        """
        self.cascade_type = cascade_type
        self.cascade_path = config.HAAR_CASCADE_PATHS.get(cascade_type)

        if not self.cascade_path or not os.path.exists(self.cascade_path):
            print(f"Warning: Cascade file not found at {self.cascade_path}")
            print("Trying alternative path...")
            # Try alternative OpenCV installation path
            alt_path = f"/usr/local/share/opencv4/haarcascades/haarcascade_{cascade_type}.xml"
            if cascade_type == 'face':
                alt_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

            if os.path.exists(alt_path):
                self.cascade_path = alt_path
                print(f"Using alternative path: {alt_path}")
            else:
                print("ERROR: Could not find Haar cascade file")
                print("Please install opencv-data: sudo apt-get install opencv-data")
                raise FileNotFoundError(f"Cascade file not found: {self.cascade_path}")

        # Load cascade classifier
        self.cascade = cv2.CascadeClassifier(self.cascade_path)

        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade from {self.cascade_path}")

        print(f"Loaded {cascade_type} cascade successfully")

    def detect_objects(self, frame):
        """
        Detect objects in the frame
        Returns: list of detected objects with their bounding boxes
        """
        # Convert to grayscale (Haar cascades work on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)

        # Detect objects
        objects = self.cascade.detectMultiScale(
            gray,
            scaleFactor=config.SCALE_FACTOR,
            minNeighbors=config.MIN_NEIGHBORS,
            minSize=config.MIN_SIZE
        )

        detected_objects = []

        for (x, y, w, h) in objects:
            detected_objects.append({
                'type': self.cascade_type,
                'bbox': (x, y, w, h),
                'center': (x + w // 2, y + h // 2),
                'size': (w, h)
            })

        return detected_objects

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels for detected objects"""
        for detection in detections:
            obj_type = detection['type']
            x, y, w, h = detection['bbox']
            center_x, center_y = detection['center']

            # Draw bounding rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                         config.BOX_COLOR_OBJECT, 2)

            # Draw center point
            cv2.circle(frame, (center_x, center_y), 5,
                      config.BOX_COLOR_OBJECT, -1)

            # Prepare label
            label = f"{obj_type.upper()} [{w}x{h}]"

            # Draw label background
            label_size = cv2.getTextSize(label, config.FONT,
                                        config.FONT_SCALE,
                                        config.FONT_THICKNESS)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10),
                         (x + label_size[0], y),
                         config.BOX_COLOR_OBJECT, -1)

            # Draw label text
            cv2.putText(frame, label, (x, y - 5),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

        return frame

class MultiObjectDetector:
    """Detector that can use multiple cascades simultaneously"""
    def __init__(self, cascade_types=['face', 'eye']):
        """Initialize multiple detectors"""
        self.detectors = {}
        for cascade_type in cascade_types:
            try:
                self.detectors[cascade_type] = ObjectDetector(cascade_type)
            except (FileNotFoundError, ValueError) as e:
                print(f"Could not load {cascade_type} detector: {e}")

        if not self.detectors:
            raise ValueError("No detectors could be loaded")

    def detect_all(self, frame):
        """Run all detectors on the frame"""
        all_detections = []
        for detector in self.detectors.values():
            detections = detector.detect_objects(frame)
            all_detections.extend(detections)
        return all_detections

    def draw_detections(self, frame, detections):
        """Draw all detections"""
        if self.detectors:
            # Use first detector's draw method
            detector = list(self.detectors.values())[0]
            return detector.draw_detections(frame, detections)
        return frame

def main():
    """Main function to run object detection"""
    print("Initializing Object Detection System...")
    print(f"Resolution: {config.CAMERA_RESOLUTION}")

    # Initialize Picamera2
    picam2 = Picamera2()

    # Create camera configuration
    camera_config = picam2.create_preview_configuration(
        main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
    )

    # Configure camera
    picam2.configure(camera_config)

    if config.CAMERA_ROTATION != 0:
        picam2.set_controls({"Transform": config.CAMERA_ROTATION})

    # Initialize multi-object detector (face and eye detection)
    try:
        detector = MultiObjectDetector(cascade_types=['face', 'eye'])
    except ValueError as e:
        print(f"Error initializing detectors: {e}")
        return

    # Start camera
    print("Starting camera...")
    picam2.start()

    # Allow camera to warm up
    print("Warming up camera for 2 seconds...")
    time.sleep(2)

    print("\nObject detection started!")
    print("Press 'q' to quit")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()

            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect objects
            detections = detector.detect_all(image)

            # Draw detections
            image = detector.draw_detections(image, detections)

            # Calculate and display FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

            # Display detection count
            cv2.putText(image, f"Detected: {len(detections)}", (10, 60),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

            # Display frame
            cv2.imshow("Object Detection - Picamera2", image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        # Cleanup
        picam2.stop()
        cv2.destroyAllWindows()
        print("Object detection stopped")

if __name__ == "__main__":
    print("="*50)
    print("Haar Cascade Object Detection (Picamera2/Ubuntu)")
    print("="*50)
    main()
