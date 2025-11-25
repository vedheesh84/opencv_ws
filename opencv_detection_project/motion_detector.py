#!/usr/bin/env python3
"""
Motion Detection Module
Detects movement in camera frames using background subtraction
Tuned to reduce false positives from camera noise
"""

import cv2
import numpy as np
import config


class MotionDetector:
    """Detects motion using background subtraction with noise filtering"""

    def __init__(self):
        """Initialize motion detector"""
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOTION_HISTORY,
            varThreshold=config.MOTION_VAR_THRESHOLD,
            detectShadows=config.MOTION_DETECT_SHADOWS
        )
        # Set learning rate for background model
        self.learning_rate = getattr(config, 'MOTION_LEARNING_RATE', 0.005)

        self.motion_threshold = config.MOTION_THRESHOLD
        self.min_motion_area = config.MIN_MOTION_AREA
        self.min_contours = getattr(config, 'MOTION_MIN_CONTOURS', 2)
        self.cooldown_frames = 0
        self.cooldown_max = config.MOTION_COOLDOWN_FRAMES

        # Consecutive frame detection for robustness
        self.consecutive_required = getattr(config, 'MOTION_CONSECUTIVE_FRAMES', 3)
        self.consecutive_count = 0

        # Frame averaging for noise reduction
        self.frame_buffer = []
        self.buffer_size = 5  # Increased buffer for better averaging

        # Stabilization counter
        self.stabilization_frames = 0
        self.stabilization_period = getattr(config, 'MOTION_STABILIZATION_FRAMES', 150)  # Frames to wait before detecting

    def detect_motion(self, frame):
        """
        Detect motion in the frame
        Returns: (motion_detected: bool, motion_level: float, motion_regions: list)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply stronger blur to reduce noise
        gray = cv2.GaussianBlur(gray, (31, 31), 0)

        # Frame averaging for additional noise reduction
        self.frame_buffer.append(gray.astype(np.float32))
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

        # Use averaged frame
        if len(self.frame_buffer) >= self.buffer_size:
            avg_frame = np.mean(self.frame_buffer, axis=0).astype(np.uint8)
        else:
            avg_frame = gray

        # Apply background subtraction with controlled learning rate
        fg_mask = self.background_subtractor.apply(avg_frame, learningRate=self.learning_rate)

        # Stronger threshold to eliminate noise
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)

        # Erode to remove small noise
        thresh = cv2.erode(thresh, kernel_small, iterations=2)
        # Dilate to restore object size
        thresh = cv2.dilate(thresh, kernel_large, iterations=2)
        # Close to fill holes
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large)
        # Open to remove remaining small blobs
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_regions = []
        total_motion_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_motion_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter out very thin or very wide contours (likely noise)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio
                    motion_regions.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w // 2, y + h // 2),
                        'area': area
                    })
                    total_motion_area += area

        # Calculate motion level as percentage of frame
        frame_area = frame.shape[0] * frame.shape[1]
        motion_level = (total_motion_area / frame_area) * 100

        # Stabilization period - don't detect motion during initial frames
        if self.stabilization_frames < self.stabilization_period:
            self.stabilization_frames += 1
            return False, motion_level, []

        # Determine if significant motion detected in this frame
        # Require both threshold AND minimum number of contours
        frame_has_motion = (
            motion_level > self.motion_threshold and
            len(motion_regions) >= self.min_contours
        )

        # Consecutive frame detection - require multiple frames with motion
        if frame_has_motion:
            self.consecutive_count += 1
        else:
            self.consecutive_count = max(0, self.consecutive_count - 1)  # Gradually decrease

        # Only trigger motion if we have enough consecutive frames
        motion_detected = self.consecutive_count >= self.consecutive_required

        # Handle cooldown
        if motion_detected:
            self.cooldown_frames = self.cooldown_max
        elif self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            motion_detected = True  # Keep motion active during cooldown

        return motion_detected, motion_level, motion_regions

    def draw_motion(self, frame, motion_regions, motion_level):
        """Draw motion regions on frame"""
        for region in motion_regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), config.BOX_COLOR_MOTION, 2)
            # Draw center point
            cx, cy = region['center']
            cv2.circle(frame, (cx, cy), 5, config.BOX_COLOR_MOTION, -1)

        # Draw motion level indicator bar
        bar_width = int((motion_level / 20) * 200)  # Scale: 20% = full bar
        bar_width = min(bar_width, 200)

        # Background bar
        cv2.rectangle(frame, (10, frame.shape[0] - 30),
                     (210, frame.shape[0] - 10),
                     (50, 50, 50), -1)

        # Threshold marker
        threshold_x = int((self.motion_threshold / 20) * 200) + 10
        cv2.line(frame, (threshold_x, frame.shape[0] - 30),
                (threshold_x, frame.shape[0] - 10), (0, 0, 255), 2)

        # Motion level bar
        bar_color = (0, 255, 0) if motion_level > self.motion_threshold else (0, 255, 255)
        if bar_width > 0:
            cv2.rectangle(frame, (10, frame.shape[0] - 30),
                         (10 + bar_width, frame.shape[0] - 10),
                         bar_color, -1)

        cv2.putText(frame, f"Motion: {motion_level:.1f}% (thresh: {self.motion_threshold}%)",
                   (10, frame.shape[0] - 35),
                   config.FONT, 0.5, config.BOX_COLOR_MOTION, 1)

        return frame

    def reset(self):
        """Reset the motion detector"""
        self.cooldown_frames = 0
        self.consecutive_count = 0
        self.frame_buffer = []
        self.stabilization_frames = 0
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOTION_HISTORY,
            varThreshold=config.MOTION_VAR_THRESHOLD,
            detectShadows=config.MOTION_DETECT_SHADOWS
        )
