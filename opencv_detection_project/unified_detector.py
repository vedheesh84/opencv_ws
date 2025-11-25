#!/usr/bin/env python3
"""
Unified Detection System
Combines motion detection, object detection, and color detection
with Arduino serial control for Raspberry Pi auto-start

Features:
- Motion detection triggers system activation
- Object and color detection when motion is detected
- Arduino serial communication for servo control
- Automatic startup on boot
- Headless operation support (no display required)
"""

import cv2
import numpy as np
import time
import signal
import sys
import os
import logging
from datetime import datetime

import config
from motion_detector import MotionDetector
from color_detection import ColorDetector
from object_detection import MultiObjectDetector
from arduino_controller import ArduinoController


class UnifiedDetector:
    """
    Main unified detection system that coordinates all detectors
    and Arduino communication
    """

    def __init__(self, headless=False):
        """
        Initialize the unified detection system

        Args:
            headless: If True, run without display output
        """
        self.headless = headless
        self.running = False
        self.camera = None

        # Detection state
        self.motion_active = False
        self.objects_detected = False
        self.last_motion_time = 0
        self.no_motion_timeout = config.NO_MOTION_TIMEOUT

        # Initialize logging
        self.setup_logging()
        self.logger.info("Initializing Unified Detection System...")

        # Initialize components
        self.init_camera()
        self.init_detectors()
        self.init_arduino()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.logger.info("Unified Detection System initialized")

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = os.path.dirname(config.LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def init_camera(self):
        """Initialize the camera"""
        self.logger.info(f"Opening camera {config.CAMERA_INDEX}...")

        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)

        if not self.camera.isOpened():
            self.logger.error("Failed to open camera!")
            raise RuntimeError("Camera initialization failed")

        # Configure camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
        self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        # Warm up camera
        time.sleep(config.CAMERA_WARMUP_TIME)

        # Verify camera is working
        ret, frame = self.camera.read()
        if not ret:
            self.logger.error("Camera opened but failed to capture frame")
            raise RuntimeError("Camera capture failed")

        self.logger.info(f"Camera initialized: {config.CAMERA_RESOLUTION[0]}x{config.CAMERA_RESOLUTION[1]} @ {config.CAMERA_FPS}fps")

    def init_detectors(self):
        """Initialize all detectors"""
        self.logger.info("Initializing detectors...")

        # Motion detector (always enabled)
        self.motion_detector = MotionDetector()
        self.logger.info("Motion detector initialized")

        # Color detector
        self.color_detector = ColorDetector()
        self.logger.info("Color detector initialized")

        # Object detector (optional, may fail if cascades not found)
        try:
            self.object_detector = MultiObjectDetector(cascade_types=config.CASCADE_TYPES)
            self.object_detection_enabled = True
            self.logger.info("Object detector initialized")
        except (ValueError, FileNotFoundError) as e:
            self.logger.warning(f"Object detection disabled: {e}")
            self.object_detector = None
            self.object_detection_enabled = False

    def init_arduino(self):
        """Initialize Arduino controller"""
        self.logger.info("Initializing Arduino controller...")

        try:
            self.arduino = ArduinoController()
            if self.arduino.is_ready():
                self.logger.info("Arduino controller ready")
            else:
                self.logger.warning("Arduino not responding")
        except Exception as e:
            self.logger.warning(f"Arduino initialization failed: {e}")
            self.arduino = None

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def process_frame(self, frame):
        """
        Process a single frame through all detectors

        Returns:
            dict: Detection results
        """
        results = {
            'motion_detected': False,
            'motion_level': 0,
            'motion_regions': [],
            'colors': [],
            'objects': [],
            'timestamp': datetime.now()
        }

        # Always run motion detection
        motion_detected, motion_level, motion_regions = self.motion_detector.detect_motion(frame)
        results['motion_detected'] = motion_detected
        results['motion_level'] = motion_level
        results['motion_regions'] = motion_regions

        # Update motion timing
        if motion_detected:
            self.last_motion_time = time.time()
            self.motion_active = True
        else:
            # Check if we should deactivate due to timeout
            if time.time() - self.last_motion_time > self.no_motion_timeout:
                self.motion_active = False

        # Only run color and object detection when motion is active
        if self.motion_active:
            # Color detection
            results['colors'] = self.color_detector.detect_colors(frame)

            # Object detection (if enabled)
            if self.object_detection_enabled and self.object_detector:
                results['objects'] = self.object_detector.detect_all(frame)

        return results

    def update_arduino(self, results):
        """Send detection results to Arduino"""
        if not self.arduino or not self.arduino.is_ready():
            return

        if self.motion_active:
            # Motion detected - start eye movement sequence
            # This sends "move_loop" command to Arduino
            self.arduino.start_sequence()
        else:
            # No motion - close eyes and stop sequence
            # This sends "close" command to Arduino
            self.arduino.stop_sequence()

    def draw_overlay(self, frame, results):
        """Draw detection overlay on frame"""
        # Draw motion regions
        frame = self.motion_detector.draw_motion(
            frame,
            results['motion_regions'],
            results['motion_level']
        )

        # Draw color detections (only when active)
        if self.motion_active and results['colors']:
            frame = self.color_detector.draw_detections(frame, results['colors'])

        # Draw object detections (only when active)
        if self.motion_active and self.object_detection_enabled and results['objects']:
            frame = self.object_detector.draw_detections(frame, results['objects'])

        # Draw status overlay
        status_color = (0, 255, 0) if self.motion_active else (0, 0, 255)
        status_text = "ACTIVE" if self.motion_active else "STANDBY"

        cv2.putText(frame, f"Status: {status_text}", (10, 30),
                   config.FONT, 0.7, status_color, 2)

        # Arduino status
        if self.arduino and self.arduino.is_ready():
            status = self.arduino.get_status()
            eye_state = "Looping" if status['looping'] else ("Open" if status['eyes_open'] else "Closed")
            arduino_text = f"Arduino: Connected | Eyes: {eye_state}"
        else:
            arduino_text = "Arduino: Disconnected"
        cv2.putText(frame, arduino_text, (10, 60),
                   config.FONT, 0.5, (255, 255, 255), 1)

        # Detection counts
        y_offset = 90
        cv2.putText(frame, f"Colors: {len(results['colors'])}", (10, y_offset),
                   config.FONT, 0.5, (255, 255, 255), 1)

        if self.object_detection_enabled:
            cv2.putText(frame, f"Objects: {len(results['objects'])}", (10, y_offset + 25),
                       config.FONT, 0.5, (255, 255, 255), 1)

        return frame

    def run(self):
        """Main detection loop"""
        self.running = True
        self.logger.info("Starting detection loop...")

        frame_count = 0
        fps_start_time = time.time()
        current_fps = 0

        # Initial stabilization period
        self.logger.info("Camera stabilization period...")
        for _ in range(30):
            ret, frame = self.camera.read()
            if ret:
                self.motion_detector.detect_motion(frame)
            time.sleep(0.033)
        self.logger.info("Stabilization complete, starting detection")

        try:
            while self.running:
                ret, frame = self.camera.read()

                if not ret:
                    self.logger.warning("Failed to capture frame, retrying...")
                    time.sleep(0.1)
                    continue

                # Process frame
                results = self.process_frame(frame)

                # Update Arduino
                self.update_arduino(results)

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    current_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start_time = time.time()

                # Display (if not headless)
                if not self.headless:
                    display_frame = self.draw_overlay(frame.copy(), results)
                    cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                               (frame.shape[1] - 100, 30),
                               config.FONT, 0.5, (255, 255, 255), 1)

                    cv2.imshow("Unified Detection", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("Quit requested by user")
                        break
                    elif key == ord('r'):
                        self.motion_detector.reset()
                        self.logger.info("Motion detector reset")

                # Periodic logging
                if frame_count == 0 and config.LOG_LEVEL == 'DEBUG':
                    self.logger.debug(
                        f"Motion: {results['motion_detected']}, "
                        f"Level: {results['motion_level']:.1f}%, "
                        f"Colors: {len(results['colors'])}, "
                        f"Objects: {len(results['objects'])}, "
                        f"Active: {self.motion_active}"
                    )

        except Exception as e:
            self.logger.error(f"Error in detection loop: {e}")
            raise
        finally:
            self.cleanup()

    def stop(self):
        """Stop the detection system"""
        self.running = False
        self.logger.info("Stop requested")

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up...")

        # Stop Arduino sequence
        if self.arduino:
            self.arduino.stop_sequence()
            self.arduino.disconnect()

        # Release camera
        if self.camera:
            self.camera.release()

        # Close windows
        if not self.headless:
            cv2.destroyAllWindows()

        self.logger.info("Cleanup complete")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Unified Detection System')
    parser.add_argument('--headless', action='store_true',
                       help='Run without display output')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (30 second timeout)')
    args = parser.parse_args()

    print("=" * 60)
    print("Unified Detection System")
    print("Motion + Object + Color Detection with Arduino Control")
    print("=" * 60)

    # Check if we can display (for non-headless mode)
    headless = args.headless
    if not headless:
        display = os.environ.get('DISPLAY')
        if not display:
            print("No display detected, running in headless mode")
            headless = True

    try:
        detector = UnifiedDetector(headless=headless)

        if args.test:
            print("Test mode: Running for 30 seconds...")
            import threading
            timer = threading.Timer(30.0, detector.stop)
            timer.start()

        detector.run()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Unified Detection System stopped")


if __name__ == "__main__":
    main()
