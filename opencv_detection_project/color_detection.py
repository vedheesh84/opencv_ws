#!/usr/bin/env python3
"""
HSV-based Color Detection using OpenCV VideoCapture
Works on Ubuntu 22.04 without picamera dependencies
"""

import cv2
import numpy as np
import time
import config

class ColorDetector:
    def __init__(self):
        """Initialize color detector"""
        self.color_ranges = config.COLOR_RANGES
        self.min_contour_area = config.MIN_CONTOUR_AREA

    def detect_colors(self, frame):
        """Detect colors in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_colors = []

        for color_name, ranges in self.color_ranges.items():
            mask = None

            for (lower, upper) in ranges:
                lower_bound = np.array(lower)
                upper_bound = np.array(upper)

                if mask is None:
                    mask = cv2.inRange(hsv, lower_bound, upper_bound)
                else:
                    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_bound, upper_bound))

            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > self.min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
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
        """Draw bounding boxes and labels"""
        for detection in detections:
            color_name = detection['name']
            x, y, w, h = detection['bbox']
            center_x, center_y = detection['center']
            area = detection['area']

            box_color = self.get_box_color(color_name)

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.circle(frame, (center_x, center_y), 5, box_color, -1)

            label = f"{color_name.upper()}: {area:.0f}px"
            label_size = cv2.getTextSize(label, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10),
                         (x + label_size[0], y), box_color, -1)
            cv2.putText(frame, label, (x, y - 5),
                       config.FONT, config.FONT_SCALE, (255, 255, 255),
                       config.FONT_THICKNESS)

        return frame

    def get_box_color(self, color_name):
        """Get BGR color for bounding box"""
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
    """Main function"""
    print("Initializing Color Detection System...")
    print(f"Resolution: {config.CAMERA_RESOLUTION}")
    print(f"Camera: /dev/video{config.CAMERA_INDEX}")

    # Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        print("Try: ls /dev/video*")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    # Initialize detector
    detector = ColorDetector()

    print("Camera ready!")
    print("Press 'q' to quit")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Failed to capture frame")
                break

            # Detect colors
            detections = detector.detect_colors(frame)

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

            cv2.imshow("Color Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Color detection stopped")

if __name__ == "__main__":
    print("="*50)
    print("HSV Color Detection System (OpenCV)")
    print("="*50)
    main()
