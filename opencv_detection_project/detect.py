#!/usr/bin/env python3
"""
Unified Detection System - Single Script
=========================================
Combines motion detection, color detection, object detection,
and Arduino servo control in one standalone script.

Usage:
    python3 detect.py              # Run with display
    python3 detect.py --headless   # Run without display (SSH/service mode)

Controls:
    q - Quit
    r - Reset motion detector
    m - Toggle motion detection
    c - Toggle color detection
    o - Toggle object detection
    a - Toggle Arduino control
"""

import cv2
import numpy as np
import serial
import time
import signal
import sys
import os
import argparse
import threading
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # Camera
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    CAMERA_WARMUP = 2.0

    # Arduino
    ARDUINO_PORT = '/dev/ttyACM0'
    ARDUINO_BAUDRATE = 9600
    ARDUINO_TIMEOUT = 1.0
    ARDUINO_INIT_DELAY = 2.0

    # Motion Detection
    MOTION_THRESHOLD = 0.5          # Minimum motion % to trigger
    MIN_MOTION_AREA = 1500          # Minimum contour area
    MOTION_HISTORY = 200            # Background model frames
    MOTION_VAR_THRESHOLD = 50       # Background subtractor threshold
    NO_MOTION_TIMEOUT = 3.0         # Seconds before stopping Arduino
    MOTION_CONSECUTIVE_FRAMES = 3   # Frames required to confirm motion
    MOTION_COOLDOWN_FRAMES = 30     # Keep active after motion stops

    # Color Detection
    MIN_CONTOUR_AREA = 500
    COLOR_RANGES = {
        'red': [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
        'green': [((35, 50, 50), (85, 255, 255))],
        'blue': [((100, 50, 50), (130, 255, 255))],
        'yellow': [((20, 100, 100), (35, 255, 255))],
    }

    # Object Detection (Haar Cascades)
    CASCADE_PATHS = {
        'face': '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        'eye': '/usr/share/opencv4/haarcascades/haarcascade_eye.xml',
    }
    SCALE_FACTOR = 1.1
    MIN_NEIGHBORS = 5
    MIN_SIZE = (30, 30)

    # Display
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1


# =============================================================================
# ARDUINO CONTROLLER
# =============================================================================
class ArduinoController:
    """Controls Arduino via serial - sends start/stop commands"""

    def __init__(self, port, baudrate, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.is_connected = False
        self.is_running = False
        self.sequence_triggered = False
        self.lock = threading.Lock()

    def connect(self):
        """Connect to Arduino"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(Config.ARDUINO_INIT_DELAY)
            self.serial_conn.reset_input_buffer()
            self.is_connected = True
            print(f"[Arduino] Connected on {self.port}")

            # Read startup message
            time.sleep(0.5)
            while self.serial_conn.in_waiting > 0:
                msg = self.serial_conn.readline().decode('utf-8').strip()
                if msg:
                    print(f"[Arduino] {msg}")
            return True
        except Exception as e:
            print(f"[Arduino] Connection failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from Arduino"""
        if self.serial_conn and self.serial_conn.is_open:
            self.stop_sequence()
            time.sleep(0.1)
            self.serial_conn.close()
            self.is_connected = False
            print("[Arduino] Disconnected")

    def send_command(self, cmd):
        """Send command to Arduino"""
        if not self.is_connected:
            return False
        with self.lock:
            try:
                self.serial_conn.write(f"{cmd}\n".encode('utf-8'))
                self.serial_conn.flush()
                return True
            except Exception as e:
                print(f"[Arduino] Send error: {e}")
                self.is_connected = False
                return False

    def read_response(self, timeout=0.5):
        """Read response from Arduino"""
        if not self.is_connected:
            return None
        try:
            start = time.time()
            while time.time() - start < timeout:
                if self.serial_conn.in_waiting > 0:
                    return self.serial_conn.readline().decode('utf-8').strip()
                time.sleep(0.01)
        except Exception:
            pass
        return None

    def check_responses(self):
        """Process any pending responses"""
        if not self.is_connected:
            return
        try:
            while self.serial_conn.in_waiting > 0:
                response = self.serial_conn.readline().decode('utf-8').strip()
                if response:
                    self._handle_response(response)
        except Exception:
            pass

    def _handle_response(self, response):
        """Handle Arduino response"""
        if response == "STARTED":
            self.is_running = True
            print("[Arduino] Sequence started")
        elif response == "DONE":
            self.is_running = False
            self.sequence_triggered = False
            print("[Arduino] Sequence complete")
        elif response == "STOPPED":
            self.is_running = False
            self.sequence_triggered = False
            print("[Arduino] Sequence stopped")
        elif response == "RUNNING":
            self.is_running = True
        elif response == "IDLE":
            self.is_running = False

    def start_sequence(self):
        """Trigger animation sequence"""
        self.check_responses()
        if self.is_running or self.sequence_triggered:
            return True
        if self.send_command("start"):
            self.sequence_triggered = True
            response = self.read_response()
            if response:
                self._handle_response(response)
            return True
        return False

    def stop_sequence(self):
        """Stop animation and return to rest"""
        self.check_responses()
        if not self.is_running and not self.sequence_triggered:
            return True
        if self.send_command("stop"):
            response = self.read_response()
            if response:
                self._handle_response(response)
            self.sequence_triggered = False
            return True
        return False

    def get_status(self):
        """Get status string"""
        if not self.is_connected:
            return "Disconnected"
        return "Running" if self.is_running else "Idle"


# =============================================================================
# MOTION DETECTOR
# =============================================================================
class MotionDetector:
    """Detects motion using background subtraction"""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=Config.MOTION_HISTORY,
            varThreshold=Config.MOTION_VAR_THRESHOLD,
            detectShadows=False
        )
        self.learning_rate = 0.002
        self.consecutive_count = 0
        self.cooldown_frames = 0
        self.frame_buffer = []
        self.stabilization_frames = 0
        self.stabilization_period = 100

    def detect(self, frame):
        """Detect motion in frame. Returns (detected, level, regions)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Frame averaging
        self.frame_buffer.append(gray.astype(np.float32))
        if len(self.frame_buffer) > 5:
            self.frame_buffer.pop(0)

        if len(self.frame_buffer) >= 3:
            avg_frame = np.mean(self.frame_buffer, axis=0).astype(np.uint8)
        else:
            avg_frame = gray

        # Background subtraction
        fg_mask = self.bg_subtractor.apply(avg_frame, learningRate=self.learning_rate)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=2)
        thresh = cv2.dilate(thresh, kernel, iterations=3)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_regions = []
        total_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > Config.MIN_MOTION_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / h if h > 0 else 0
                if 0.2 < aspect < 5.0:
                    motion_regions.append({'bbox': (x, y, w, h), 'area': area})
                    total_area += area

        # Calculate motion level
        frame_area = frame.shape[0] * frame.shape[1]
        motion_level = (total_area / frame_area) * 100

        # Stabilization period
        if self.stabilization_frames < self.stabilization_period:
            self.stabilization_frames += 1
            return False, motion_level, []

        # Check if motion detected
        has_motion = motion_level > Config.MOTION_THRESHOLD and len(motion_regions) >= 1

        if has_motion:
            self.consecutive_count += 1
        else:
            self.consecutive_count = max(0, self.consecutive_count - 1)

        detected = self.consecutive_count >= Config.MOTION_CONSECUTIVE_FRAMES

        # Cooldown
        if detected:
            self.cooldown_frames = Config.MOTION_COOLDOWN_FRAMES
        elif self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            detected = True

        return detected, motion_level, motion_regions

    def reset(self):
        """Reset detector"""
        self.consecutive_count = 0
        self.cooldown_frames = 0
        self.frame_buffer = []
        self.stabilization_frames = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=Config.MOTION_HISTORY,
            varThreshold=Config.MOTION_VAR_THRESHOLD,
            detectShadows=False
        )
        print("[Motion] Detector reset")


# =============================================================================
# COLOR DETECTOR
# =============================================================================
class ColorDetector:
    """Detects colors using HSV ranges"""

    def detect(self, frame):
        """Detect colors in frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []

        for color_name, ranges in Config.COLOR_RANGES.items():
            mask = None
            for (lower, upper) in ranges:
                m = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = m if mask is None else cv2.bitwise_or(mask, m)

            # Cleanup
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > Config.MIN_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                    detections.append({
                        'type': 'color',
                        'name': color_name,
                        'bbox': (x, y, w, h),
                        'area': area
                    })

        return detections


# =============================================================================
# OBJECT DETECTOR
# =============================================================================
class ObjectDetector:
    """Detects objects using Haar cascades"""

    def __init__(self):
        self.cascades = {}
        for name, path in Config.CASCADE_PATHS.items():
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self.cascades[name] = cascade
                    print(f"[Object] Loaded {name} cascade")
            else:
                print(f"[Object] Cascade not found: {path}")

    def detect(self, frame):
        """Detect objects in frame"""
        if not self.cascades:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections = []

        for name, cascade in self.cascades.items():
            objects = cascade.detectMultiScale(
                gray,
                scaleFactor=Config.SCALE_FACTOR,
                minNeighbors=Config.MIN_NEIGHBORS,
                minSize=Config.MIN_SIZE
            )
            for (x, y, w, h) in objects:
                detections.append({
                    'type': 'object',
                    'name': name,
                    'bbox': (x, y, w, h)
                })

        return detections


# =============================================================================
# MAIN DETECTOR CLASS
# =============================================================================
class UnifiedDetector:
    """Main detection system combining all components"""

    def __init__(self, headless=False):
        self.headless = headless
        self.running = False
        self.camera = None

        # Feature toggles
        self.motion_enabled = True
        self.color_enabled = True
        self.object_enabled = True
        self.arduino_enabled = True

        # State
        self.motion_active = False
        self.last_motion_time = 0

        # Initialize components
        print("=" * 60)
        print("UNIFIED DETECTION SYSTEM")
        print("Motion + Color + Object Detection + Arduino Control")
        print("=" * 60)

        self._init_camera()
        self._init_detectors()
        self._init_arduino()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _init_camera(self):
        """Initialize camera"""
        print("\n[Camera] Initializing...")
        self.camera = cv2.VideoCapture(Config.CAMERA_INDEX)

        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)

        time.sleep(Config.CAMERA_WARMUP)

        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Camera cannot capture frames")

        print(f"[Camera] Ready - {frame.shape[1]}x{frame.shape[0]}")

    def _init_detectors(self):
        """Initialize all detectors"""
        print("\n[Detectors] Initializing...")
        self.motion_detector = MotionDetector()
        self.color_detector = ColorDetector()
        self.object_detector = ObjectDetector()
        print("[Detectors] Ready")

    def _init_arduino(self):
        """Initialize Arduino connection"""
        print("\n[Arduino] Initializing...")
        self.arduino = ArduinoController(
            Config.ARDUINO_PORT,
            Config.ARDUINO_BAUDRATE,
            Config.ARDUINO_TIMEOUT
        )
        if not self.arduino.connect():
            print("[Arduino] WARNING: Not connected - running without Arduino")
            self.arduino_enabled = False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[System] Signal {signum} received, shutting down...")
        self.stop()

    def process_frame(self, frame):
        """Process single frame through all detectors"""
        results = {
            'motion_detected': False,
            'motion_level': 0,
            'motion_regions': [],
            'colors': [],
            'objects': []
        }

        # Motion detection (always runs to maintain background model)
        if self.motion_enabled:
            detected, level, regions = self.motion_detector.detect(frame)
            results['motion_detected'] = detected
            results['motion_level'] = level
            results['motion_regions'] = regions

            if detected:
                self.last_motion_time = time.time()
                self.motion_active = True
            elif time.time() - self.last_motion_time > Config.NO_MOTION_TIMEOUT:
                self.motion_active = False

        # Color and object detection only when motion is active
        if self.motion_active:
            if self.color_enabled:
                results['colors'] = self.color_detector.detect(frame)

            if self.object_enabled:
                results['objects'] = self.object_detector.detect(frame)

        return results

    def update_arduino(self, results):
        """Control Arduino based on detection results"""
        if not self.arduino_enabled or not self.arduino.is_connected:
            return

        self.arduino.check_responses()

        if self.motion_active:
            self.arduino.start_sequence()
        else:
            if self.arduino.is_running or self.arduino.sequence_triggered:
                self.arduino.stop_sequence()

    def draw_overlay(self, frame, results, fps):
        """Draw detection overlays on frame"""
        h, w = frame.shape[:2]

        # Status bar background
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)

        # Status text
        status = "ACTIVE" if self.motion_active else "STANDBY"
        color = (0, 255, 0) if self.motion_active else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 25),
                    Config.FONT, 0.7, color, 2)

        # Arduino status
        arduino_status = self.arduino.get_status() if self.arduino_enabled else "Disabled"
        cv2.putText(frame, f"Arduino: {arduino_status}", (10, 50),
                    Config.FONT, 0.5, (255, 255, 255), 1)

        # Detection counts
        info = f"Motion: {results['motion_level']:.1f}% | Colors: {len(results['colors'])} | Objects: {len(results['objects'])}"
        cv2.putText(frame, info, (10, 75),
                    Config.FONT, 0.5, (255, 255, 255), 1)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 80, 25),
                    Config.FONT, 0.5, (255, 255, 255), 1)

        # Feature toggles
        toggles = f"[M]otion:{self.motion_enabled} [C]olor:{self.color_enabled} [O]bject:{self.object_enabled} [A]rduino:{self.arduino_enabled}"
        cv2.putText(frame, toggles, (10, 95),
                    Config.FONT, 0.4, (150, 150, 150), 1)

        # Draw motion regions
        for region in results['motion_regions']:
            x, y, rw, rh = region['bbox']
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

        # Draw color detections
        color_map = {'red': (0, 0, 255), 'green': (0, 255, 0),
                     'blue': (255, 0, 0), 'yellow': (0, 255, 255)}
        for det in results['colors']:
            x, y, rw, rh = det['bbox']
            c = color_map.get(det['name'], (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), c, 2)
            cv2.putText(frame, det['name'].upper(), (x, y - 5),
                        Config.FONT, 0.5, c, 1)

        # Draw object detections
        for det in results['objects']:
            x, y, rw, rh = det['bbox']
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
            cv2.putText(frame, det['name'].upper(), (x, y - 5),
                        Config.FONT, 0.5, (0, 255, 0), 2)

        # Motion level bar
        bar_y = h - 30
        cv2.rectangle(frame, (10, bar_y), (210, bar_y + 20), (50, 50, 50), -1)
        bar_width = min(int(results['motion_level'] * 10), 200)
        bar_color = (0, 255, 0) if results['motion_detected'] else (100, 100, 100)
        if bar_width > 0:
            cv2.rectangle(frame, (10, bar_y), (10 + bar_width, bar_y + 20), bar_color, -1)
        cv2.putText(frame, f"Motion Level", (10, bar_y - 5),
                    Config.FONT, 0.4, (255, 255, 255), 1)

        return frame

    def run(self):
        """Main detection loop"""
        self.running = True
        print("\n" + "=" * 60)
        print("DETECTION STARTED")
        if not self.headless:
            print("Press 'q' to quit, 'r' to reset motion detector")
            print("Toggle: [m]otion, [c]olor, [o]bject, [a]rduino")
        print("=" * 60 + "\n")

        # Stabilization
        print("[System] Camera stabilization...")
        for _ in range(50):
            ret, frame = self.camera.read()
            if ret:
                self.motion_detector.detect(frame)
            time.sleep(0.02)
        print("[System] Stabilization complete")
        print("[System] Detection active\n")

        frame_count = 0
        fps_time = time.time()
        fps = 0

        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("[Camera] Frame capture failed, retrying...")
                    time.sleep(0.1)
                    continue

                # Process frame
                results = self.process_frame(frame)

                # Update Arduino
                if self.arduino_enabled:
                    self.update_arduino(results)

                # Calculate FPS
                frame_count += 1
                if time.time() - fps_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    fps_time = time.time()

                # Display
                if not self.headless:
                    display = self.draw_overlay(frame.copy(), results, fps)
                    cv2.imshow("Unified Detection", display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[System] Quit requested")
                        break
                    elif key == ord('r'):
                        self.motion_detector.reset()
                    elif key == ord('m'):
                        self.motion_enabled = not self.motion_enabled
                        print(f"[Toggle] Motion: {self.motion_enabled}")
                    elif key == ord('c'):
                        self.color_enabled = not self.color_enabled
                        print(f"[Toggle] Color: {self.color_enabled}")
                    elif key == ord('o'):
                        self.object_enabled = not self.object_enabled
                        print(f"[Toggle] Object: {self.object_enabled}")
                    elif key == ord('a'):
                        self.arduino_enabled = not self.arduino_enabled
                        print(f"[Toggle] Arduino: {self.arduino_enabled}")

        except Exception as e:
            print(f"[Error] {e}")
        finally:
            self.cleanup()

    def stop(self):
        """Stop detection"""
        self.running = False

    def cleanup(self):
        """Cleanup resources"""
        print("\n[System] Cleaning up...")

        if self.arduino and self.arduino.is_connected:
            self.arduino.stop_sequence()
            self.arduino.disconnect()

        if self.camera:
            self.camera.release()

        if not self.headless:
            cv2.destroyAllWindows()

        print("[System] Cleanup complete")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Unified Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (when display is enabled):
  q - Quit
  r - Reset motion detector
  m - Toggle motion detection
  c - Toggle color detection
  o - Toggle object detection
  a - Toggle Arduino control
        """
    )
    parser.add_argument('--headless', action='store_true',
                        help='Run without display (for SSH/service mode)')
    args = parser.parse_args()

    # Check for display
    headless = args.headless
    if not headless and not os.environ.get('DISPLAY'):
        print("[Warning] No display detected, running in headless mode")
        headless = True

    try:
        detector = UnifiedDetector(headless=headless)
        detector.run()
    except KeyboardInterrupt:
        print("\n[System] Interrupted by user")
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

    print("\n[System] Detection system stopped")


if __name__ == "__main__":
    main()
