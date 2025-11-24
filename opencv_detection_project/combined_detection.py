#!/usr/bin/env python3
"""
Combined Object and Color Detection Script for Ubuntu 22.04
Uses Picamera2 for camera access
Runs both Haar Cascade object detection and HSV color detection simultaneously
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2
import config
from object_detection_ubuntu import MultiObjectDetector
from color_detection_ubuntu import ColorDetector

class CombinedDetector:
    """Combined detector for objects and colors"""
    def __init__(self, cascade_types=['face', 'eye']):
        """Initialize both object and color detectors"""
        print("Initializing combined detection system...")

        # Initialize object detector
        try:
            self.object_detector = MultiObjectDetector(cascade_types=cascade_types)
            self.object_detection_enabled = True
            print("Object detection initialized")
        except ValueError as e:
            print(f"Warning: Object detection disabled - {e}")
            self.object_detection_enabled = False

        # Initialize color detector
        self.color_detector = ColorDetector()
        print("Color detection initialized")

    def detect_all(self, frame):
        """Run both object and color detection"""
        object_detections = []
        color_detections = []

        # Detect objects
        if self.object_detection_enabled:
            object_detections = self.object_detector.detect_all(frame)

        # Detect colors
        color_detections = self.color_detector.detect_colors(frame)

        return object_detections, color_detections

    def draw_all(self, frame, object_detections, color_detections):
        """Draw both object and color detections"""
        # Draw color detections first (so they appear behind objects)
        frame = self.color_detector.draw_detections(frame, color_detections)

        # Draw object detections
        if self.object_detection_enabled:
            frame = self.object_detector.draw_detections(frame, object_detections)

        return frame

    def analyze_detections(self, object_detections, color_detections):
        """
        Analyze relationships between detected objects and colors
        Returns list of objects with their dominant colors
        """
        results = []

        for obj in object_detections:
            obj_x, obj_y, obj_w, obj_h = obj['bbox']

            # Find colors that overlap with this object
            overlapping_colors = []

            for color in color_detections:
                color_x, color_y, color_w, color_h = color['bbox']

                # Check if color region overlaps with object region
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

        # Check if one rectangle is to the left of the other
        if x1 > x2 + w2 or x2 > x1 + w1:
            return False

        # Check if one rectangle is above the other
        if y1 > y2 + h2 or y2 > y1 + h1:
            return False

        return True

def main():
    """Main function to run combined detection"""
    print("="*50)
    print("Combined Detection (Picamera2/Ubuntu)")
    print("="*50)
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

    # Initialize combined detector
    detector = CombinedDetector(cascade_types=['face', 'eye'])

    # Start camera
    print("Starting camera...")
    picam2.start()

    # Allow camera to warm up
    print("Warming up camera for 2 seconds...")
    time.sleep(2)

    print("\nCombined detection started!")
    print("Press 'q' to quit")
    print("Press 'a' to toggle analysis mode")

    frame_count = 0
    start_time = time.time()
    show_analysis = True

    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()

            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Run detections
            object_detections, color_detections = detector.detect_all(image)

            # Draw detections
            image = detector.draw_all(image, object_detections, color_detections)

            # Analyze relationships if enabled
            if show_analysis and object_detections and color_detections:
                analysis = detector.analyze_detections(object_detections, color_detections)

                # Display analysis results
                y_offset = 120
                if analysis:
                    cv2.putText(image, "Analysis:", (10, y_offset),
                               config.FONT, 0.5, (0, 255, 255), 1)
                    y_offset += 25

                    for result in analysis[:3]:  # Show max 3 results
                        obj_type = result['object_type']
                        colors = ', '.join(result['colors'])
                        text = f"{obj_type}: {colors}"
                        cv2.putText(image, text, (10, y_offset),
                                   config.FONT, 0.4, (0, 255, 255), 1)
                        y_offset += 20

            # Calculate and display FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

            # Display detection counts
            cv2.putText(image, f"Objects: {len(object_detections)}", (10, 60),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

            cv2.putText(image, f"Colors: {len(color_detections)}", (10, 90),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

            # Display frame
            cv2.imshow("Combined Detection - Picamera2", image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('a'):
                show_analysis = not show_analysis
                print(f"Analysis mode: {'ON' if show_analysis else 'OFF'}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        # Cleanup
        picam2.stop()
        cv2.destroyAllWindows()
        print("Combined detection stopped")

if __name__ == "__main__":
    main()
