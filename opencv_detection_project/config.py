"""
Configuration file for OpenCV object detection and color identification
Using OpenCV VideoCapture for Raspberry Pi Camera (V4L2 backend)
"""
import cv2

# Camera Settings
CAMERA_INDEX = 0  # Camera device index (0 for /dev/video0)
CAMERA_RESOLUTION = (640, 480)  # (width, height)
CAMERA_FPS = 30
CAMERA_ROTATION = 0  # 0, 90, 180, or 270 degrees

# Haar Cascade Settings
HAAR_CASCADE_PATHS = {
    'face': '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    'eye': '/usr/share/opencv4/haarcascades/haarcascade_eye.xml',
    'fullbody': '/usr/share/opencv4/haarcascades/haarcascade_fullbody.xml',
    'upperbody': '/usr/share/opencv4/haarcascades/haarcascade_upperbody.xml',
}

# Detection Parameters
SCALE_FACTOR = 1.1  # How much image size is reduced at each scale
MIN_NEIGHBORS = 3   # How many neighbors each candidate rectangle should have
MIN_SIZE = (30, 30) # Minimum object size

# HSV Color Ranges (Hue, Saturation, Value)
COLOR_RANGES = {
    'red': [
        ((0, 70, 500), (10, 255, 255)),
        ((152, 107, 91), (179, 194, 255))
    ],
    'green': [((61, 90, 151), (75, 159, 249))],
    'blue': [((93, 115, 28), (116, 235, 207))],
    'yellow': [((14, 38, 124), (24, 184, 222))],
    'orange': [((7, 71, 108), (18, 190, 237))],
    'purple': [((118, 18, 86), (147, 96, 130))],
    'white': [((18, 4, 183), (75, 28, 231))],
    'black': [((126, 0, 10), (180, 65, 79))],
}

# Minimum contour area for color detection
MIN_CONTOUR_AREA = 500

# Display Settings
FONT = cv2.FONT_HERSHEY_SIMPLEX if 'cv2' in dir() else 1
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Detection box colors (BGR format)
BOX_COLOR_OBJECT = (0, 255, 0)
BOX_COLOR_RED = (0, 0, 225)
BOX_COLOR_GREEN = (0, 225, 0)
BOX_COLOR_BLUE = (225, 0, 0)
BOX_COLOR_YELLOW = (0, 225, 225)
BOX_COLOR_ORANGE = (0, 165, 225)
BOX_COLOR_PURPLE = (135, 57, 105)
BOX_COLOR_WHITE = (200, 200, 200)
BOX_COLOR_BLACK = (100, 100, 100)

FONT = cv2.FONT_HERSHEY_SIMPLEX
