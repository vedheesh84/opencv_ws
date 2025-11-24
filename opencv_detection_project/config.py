"""
Configuration file for OpenCV object detection and color identification
Optimized for Raspberry Pi 4 Model B with Camera Rev 1.3 on Ubuntu 22.04
Using Picamera2 (libcamera-based)
"""

# Camera Settings for Picamera2
CAMERA_RESOLUTION = (640, 480)  # (width, height)
CAMERA_FRAMERATE = 30
CAMERA_ROTATION = 0  # 0, 90, 180, or 270 degrees

# Picamera2 Configuration
CAMERA_CONFIG = {
    "size": CAMERA_RESOLUTION,
    "format": "RGB888"  # or "BGR888" for direct OpenCV compatibility
}

# Haar Cascade Settings
HAAR_CASCADE_PATHS = {
    'face': '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    'eye': '/usr/share/opencv4/haarcascades/haarcascade_eye.xml',
    'fullbody': '/usr/share/opencv4/haarcascades/haarcascade_fullbody.xml',
    'upperbody': '/usr/share/opencv4/haarcascades/haarcascade_upperbody.xml',
}

# Detection Parameters
SCALE_FACTOR = 1.1  # How much image size is reduced at each scale
MIN_NEIGHBORS = 5   # How many neighbors each candidate rectangle should have
MIN_SIZE = (30, 30) # Minimum object size

# HSV Color Ranges (Hue, Saturation, Value)
# Adjust these ranges based on lighting conditions
COLOR_RANGES = {
    'red': [
        # Red wraps around in HSV, so we need two ranges
        ((0, 100, 100), (10, 255, 255)),
        ((160, 100, 100), (180, 255, 255))
    ],
    'green': [((40, 50, 50), (80, 255, 255))],
    'blue': [((100, 100, 100), (130, 255, 255))],
    'yellow': [((20, 100, 100), (30, 255, 255))],
    'orange': [((10, 100, 100), (20, 255, 255))],
    'purple': [((130, 50, 50), (160, 255, 255))],
    'white': [((0, 0, 200), (180, 30, 255))],
    'black': [((0, 0, 0), (180, 255, 50))],
}

# Minimum contour area for color detection (filters out noise)
MIN_CONTOUR_AREA = 500

# Display Settings
DISPLAY_WINDOW_NAME = "OpenCV Detection"
DISPLAY_FPS = True
FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Detection box colors (BGR format)
BOX_COLOR_OBJECT = (0, 255, 0)  # Green for objects
BOX_COLOR_RED = (0, 0, 255)
BOX_COLOR_GREEN = (0, 255, 0)
BOX_COLOR_BLUE = (255, 0, 0)
BOX_COLOR_YELLOW = (0, 255, 255)
BOX_COLOR_ORANGE = (0, 165, 255)
BOX_COLOR_PURPLE = (255, 0, 255)
BOX_COLOR_WHITE = (255, 255, 255)
BOX_COLOR_BLACK = (0, 0, 0)
