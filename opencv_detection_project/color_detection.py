#!/usr/bin/env python3
"""
HSV-based Color Detection Script for Ubuntu 22.04
Uses Picamera2 for camera access
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2
import config

class ColorDetector:
    def __init__(self):
        """Initialize color detector"""
        self.color_ranges = config.COLOR_RANGES
        self.min_contour_area = config.MIN_CONTOUR_AREA

    def detect_colors(self, frame):
        """
        Detect colors in the frame
        Returns: list of detected colors with their bounding boxes
        """
        # Convert RGB/BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detected_colors = []

        # Check each color
        for color_name, ranges in self.color_ranges.items():
            # Create mask for this color
            mask = None

            for (lower, upper) in ranges:
                lower_bound = np.array(lower)
                upper_bound = np.array(upper)

                if mask is None:
                    mask = cv2.inRange(hsv, lower_bound, upper_bound)
                else:
                    # Combine masks for colors with multiple ranges (like red)
                    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_bound, upper_bound))

            # Apply morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)

                if area > self.min_contour_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center
                    center_x = x + w // 2
                    center_y = y + h // 2

                    detected_colors.append({
                        'name': color_name,
                        'bbox': (x, y, w, h),
                        'center': (center_x, center_y),
                        'area': area
                    })

        return detected_colors

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels for detected colors"""
        for detection in detections:
            color_name = detection['name']
            x, y, w, h = detection['bbox']
            center_x, center_y = detection['center']
            area = detection['area']

            # Get color for bounding box
            box_color = self.get_box_color(color_name)

            # Draw bounding rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Draw center point
            cv2.circle(frame, (center_x, center_y), 5, box_color, -1)

            # Prepare label with color name and area
            label = f"{color_name.upper()}: {area:.0f}px"

            # Draw label background
            label_size = cv2.getTextSize(label, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10),
                         (x + label_size[0], y), box_color, -1)

            # Draw label text
            cv2.putText(frame, label, (x, y - 5),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

        return frame

    def get_box_color(self, color_name):
        """Get BGR color for bounding box based on detected color"""
        color_map = {
            'red': config.BOX_COLOR_RED,
            'green': config.BOX_COLOR_GREEN,
            'blue': config.BOX_COLOR_BLUE,
            'yellow': config.BOX_COLOR_YELLOW,
            'orange': config.BOX_COLOR_ORANGE,
            'purple': config.BOX_COLOR_PURPLE,
            'white': config.BOX_COLOR_WHITE,
            'black': config.BOX_COLOR_BLACK,
        }
        return color_map.get(color_name, (255, 255, 255))

def main():
    """Main function to run color detection"""
    print("Initializing Color Detection System...")
    print(f"Resolution: {config.CAMERA_RESOLUTION}")
    print(f"Detecting colors: {', '.join(config.COLOR_RANGES.keys())}")

    # Initialize Picamera2
    picam2 = Picamera2()

    # Create camera configuration
    camera_config = picam2.create_preview_configuration(
        main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
    )

    # Configure and start camera
    picam2.configure(camera_config)

    if config.CAMERA_ROTATION != 0:
        picam2.set_controls({"Transform": config.CAMERA_ROTATION})

    print("Starting camera...")
    picam2.start()

    # Initialize color detector
    detector = ColorDetector()

    # Allow camera to warm up
    print("Warming up camera for 2 seconds...")
    time.sleep(2)

    print("\nColor detection started!")
    print("Press 'q' to quit")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()

            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect colors
            detections = detector.detect_colors(image)

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
            cv2.imshow("Color Detection - Picamera2", image)

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
        print("Color detection stopped")

if __name__ == "__main__":
    print("="*50)
    print("HSV Color Detection System (Picamera2/Ubuntu)")
    print("="*50)
    main()
