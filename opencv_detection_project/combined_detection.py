#!/usr/bin/env python3
"""
Combined Object and Color Detection using OpenCV VideoCapture
Works on Ubuntu 22.04 without picamera dependencies
"""

import cv2
import numpy as np
import time
import config
from object_detection_opencv import MultiObjectDetector
from color_detection_opencv import ColorDetector

class CombinedDetector:
    """Combined detector for objects and colors"""
    def __init__(self, cascade_types=['face', 'eye']):
        """Initialize both detectors"""
        print("Initializing combined detection system...")

        try:
            self.object_detector = MultiObjectDetector(cascade_types=cascade_types)
            self.object_detection_enabled = True
            print("Object detection initialized")
        except ValueError as e:
            print(f"Warning: Object detection disabled - {e}")
            self.object_detection_enabled = False

        self.color_detector = ColorDetector()
        print("Color detection initialized")

    def detect_all(self, frame):
        """Run both detectors"""
        object_detections = []
        color_detections = []

        if self.object_detection_enabled:
            object_detections = self.object_detector.detect_all(frame)

        color_detections = self.color_detector.detect_colors(frame)

        return object_detections, color_detections

    def draw_all(self, frame, object_detections, color_detections):
        """Draw both types of detections"""
        frame = self.color_detector.draw_detections(frame, color_detections)

        if self.object_detection_enabled:
            frame = self.object_detector.draw_detections(frame, object_detections)

        return frame

    def analyze_detections(self, object_detections, color_detections):
        """Analyze relationships between objects and colors"""
        results = []

        for obj in object_detections:
            obj_x, obj_y, obj_w, obj_h = obj['bbox']
            overlapping_colors = []

            for color in color_detections:
                color_x, color_y, color_w, color_h = color['bbox']

                if self.rectangles_overlap(
                    (obj_x, obj_y, obj_w, obj_h),
                    (color_x, color_y, color_w, color_h)
                ):
                    overlapping_colors.append(color['name'])

            if overlapping_colors:
                results.append({
                    'object_type': obj['type'],
                    'bbox': obj['bbox'],
                    'colors': overlapping_colors
                })

        return results

    def rectangles_overlap(self, rect1, rect2):
        """Check if two rectangles overlap"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        if x1 > x2 + w2 or x2 > x1 + w1:
            return False
        if y1 > y2 + h2 or y2 > y1 + h1:
            return False

        return True

def main():
    """Main function"""
    print("="*50)
    print("Combined Detection (OpenCV)")
    print("="*50)
    print(f"Resolution: {config.CAMERA_RESOLUTION}")

    # Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    # Initialize detector
    detector = CombinedDetector(cascade_types=['face', 'eye'])

    print("Camera ready!")
    print("Press 'q' to quit")
    print("Press 'a' to toggle analysis mode")

    frame_count = 0
    start_time = time.time()
    show_analysis = True

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Failed to capture frame")
                break

            # Run detections
            object_detections, color_detections = detector.detect_all(frame)

            # Draw detections
            frame = detector.draw_all(frame, object_detections, color_detections)

            # Analyze if enabled
            if show_analysis and object_detections and color_detections:
                analysis = detector.analyze_detections(object_detections, color_detections)

                y_offset = 120
                if analysis:
                    cv2.putText(frame, "Analysis:", (10, y_offset),
                               config.FONT, 0.5, (0, 255, 255), 1)
                    y_offset += 25

                    for result in analysis[:3]:
                        obj_type = result['object_type']
                        colors = ', '.join(result['colors'])
                        text = f"{obj_type}: {colors}"
                        cv2.putText(frame, text, (10, y_offset),
                                   config.FONT, 0.4, (0, 255, 255), 1)
                        y_offset += 20

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

            cv2.putText(frame, f"Objects: {len(object_detections)}", (10, 60),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

            cv2.putText(frame, f"Colors: {len(color_detections)}", (10, 90),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

            cv2.imshow("Combined Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('a'):
                show_analysis = not show_analysis
                print(f"Analysis mode: {'ON' if show_analysis else 'OFF'}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Combined detection stopped")

if __name__ == "__main__":
    main()
