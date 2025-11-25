#!/usr/bin/env python3
"""
Haar Cascade Object Detection using OpenCV VideoCapture
For Raspberry Pi with CSI Camera (V4L2 backend)
"""

import cv2
import numpy as np
import time
import os
import config

class ObjectDetector:
    def __init__(self, cascade_type='face'):
        """Initialize object detector"""
        self.cascade_type = cascade_type
        self.cascade_path = config.HAAR_CASCADE_PATHS.get(cascade_type)

        if not self.cascade_path or not os.path.exists(self.cascade_path):
            print(f"Warning: Cascade file not found at {self.cascade_path}")
            alt_path = f"/usr/local/share/opencv4/haarcascades/haarcascade_{cascade_type}.xml"
            if cascade_type == 'face':
                alt_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

            if os.path.exists(alt_path):
                self.cascade_path = alt_path
            else:
                raise FileNotFoundError(f"Cascade file not found. Install: sudo apt install opencv-data")

        self.cascade = cv2.CascadeClassifier(self.cascade_path)

        if self.cascade.empty():
            raise ValueError(f"Failed to load cascade from {self.cascade_path}")

        print(f"Loaded {cascade_type} cascade successfully")

    def detect_objects(self, frame):
        """Detect objects in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

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
        """Draw bounding boxes"""
        for detection in detections:
            obj_type = detection['type']
            x, y, w, h = detection['bbox']
            center_x, center_y = detection['center']

            cv2.rectangle(frame, (x, y), (x + w, y + h),
                         config.BOX_COLOR_OBJECT, 2)
            cv2.circle(frame, (center_x, center_y), 5,
                      config.BOX_COLOR_OBJECT, -1)

            label = f"{obj_type.upper()} [{w}x{h}]"
            label_size = cv2.getTextSize(label, config.FONT,
                                        config.FONT_SCALE,
                                        config.FONT_THICKNESS)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10),
                         (x + label_size[0], y),
                         config.BOX_COLOR_OBJECT, -1)
            cv2.putText(frame, label, (x, y - 5),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

        return frame

class MultiObjectDetector:
    """Multiple cascade detectors"""
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
        """Run all detectors"""
        all_detections = []
        for detector in self.detectors.values():
            detections = detector.detect_objects(frame)
            all_detections.extend(detections)
        return all_detections

    def draw_detections(self, frame, detections):
        """Draw all detections"""
        if self.detectors:
            detector = list(self.detectors.values())[0]
            return detector.draw_detections(frame, detections)
        return frame

def main():
    """Main function"""
    print("Initializing Object Detection System...")
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

    print("Camera started")

    # Give camera time to adjust
    time.sleep(2)

    # Initialize detector
    try:
        detector = MultiObjectDetector(cascade_types=['face', 'eye'])
    except ValueError as e:
        print(f"Error: {e}")
        cap.release()
        return

    print("Camera ready!")
    print("Press 'q' to quit")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame")
                break

            # Detect objects
            detections = detector.detect_all(frame)

            # Draw detections
            frame = detector.draw_detections(frame, detections)

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

            cv2.putText(frame, f"Detected: {len(detections)}", (10, 60),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

            cv2.imshow("Object Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Object detection stopped")

if __name__ == "__main__":
    print("="*50)
    print("Haar Cascade Object Detection (OpenCV)")
    print("="*50)
    main()
