"""
Configuration file for OpenCV object detection and color identification
Using standard OpenCV VideoCapture (works on any Linux system)
"""

# Camera Settings
CAMERA_INDEX = 0  # Usually 0 for /dev/video0, try 1, 2... if not working
CAMERA_RESOLUTION = (640, 480)  # (width, height)
CAMERA_FPS = 30
CAMERA_ROTATION = 0  # Not used with VideoCapture, rotate manually if needed

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
COLOR_RANGES = {
    'red': [
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

# Minimum contour area for color detection
MIN_CONTOUR_AREA = 500

# Display Settings
FONT = cv2.FONT_HERSHEY_SIMPLEX if 'cv2' in dir() else 1
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Detection box colors (BGR format)
BOX_COLOR_OBJECT = (0, 255, 0)
BOX_COLOR_RED = (0, 0, 255)
BOX_COLOR_GREEN = (0, 255, 0)
BOX_COLOR_BLUE = (255, 0, 0)
BOX_COLOR_YELLOW = (0, 255, 255)
BOX_COLOR_ORANGE = (0, 165, 255)
BOX_COLOR_PURPLE = (255, 0, 255)
BOX_COLOR_WHITE = (255, 255, 255)
BOX_COLOR_BLACK = (0, 0, 0)

import cv2
FONT = cv2.FONT_HERSHEY_SIMPLEX
