#!/usr/bin/env python3
"""
Haar Cascade Object Detection Script
Detects objects (faces, eyes, bodies) using classical ML cascades
"""

import cv2
import numpy as np
import time
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
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

    # Initialize camera
    camera = PiCamera()
    camera.resolution = config.CAMERA_RESOLUTION
    camera.framerate = config.CAMERA_FRAMERATE
    camera.rotation = config.CAMERA_ROTATION

    # Initialize capture array
    raw_capture = PiRGBArray(camera, size=config.CAMERA_RESOLUTION)

    # Initialize multi-object detector (face and eye detection)
    try:
        detector = MultiObjectDetector(cascade_types=['face', 'eye'])
    except ValueError as e:
        print(f"Error initializing detectors: {e}")
        camera.close()
        return

    # Allow camera to warm up
    print(f"Warming up camera for {config.CAMERA_WARMUP_TIME} seconds...")
    time.sleep(config.CAMERA_WARMUP_TIME)

    print("\nObject detection started!")
    print("Press 'q' to quit")

    frame_count = 0
    start_time = time.time()

    try:
        # Capture frames continuously
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            # Get the frame
            image = frame.array

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
            cv2.imshow("Object Detection", image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break

            # Clear the stream for next frame
            raw_capture.truncate(0)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        camera.close()
        cv2.destroyAllWindows()
        print("Object detection stopped")

if __name__ == "__main__":
    print("="*50)
    print("Haar Cascade Object Detection System")
    print("="*50)
    main()
