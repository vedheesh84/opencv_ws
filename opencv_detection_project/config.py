"""
Configuration file for OpenCV object detection and color identification
Using OpenCV VideoCapture for Raspberry Pi Camera (V4L2 backend)
Includes settings for motion detection and Arduino serial communication
"""
import cv2

# =============================================================================
# Camera Settings
# =============================================================================
CAMERA_INDEX = 0  # Camera device index (0 for /dev/video0)
CAMERA_RESOLUTION = (640, 480)  # (width, height)
CAMERA_FPS = 30
CAMERA_ROTATION = 0  # 0, 90, 180, or 270 degrees
CAMERA_WARMUP_TIME = 2  # Seconds to wait for camera to stabilize

# =============================================================================
# Arduino Serial Settings
# =============================================================================
ARDUINO_PORT = '/dev/ttyACM0'  # Serial port for Arduino
ARDUINO_BAUDRATE = 9600        # Must match Arduino Serial.begin() rate
ARDUINO_TIMEOUT = 1.0          # Serial read timeout in seconds
ARDUINO_INIT_DELAY = 2.0       # Seconds to wait after opening serial connection

# =============================================================================
# Motion Detection Settings
# =============================================================================
MOTION_THRESHOLD = 0.5         # Minimum motion level (%) to trigger detection
MIN_MOTION_AREA = 1500         # Minimum contour area to consider as motion
MOTION_HISTORY = 200           # Number of frames for background model
MOTION_VAR_THRESHOLD = 50      # Threshold for background subtractor
MOTION_DETECT_SHADOWS = False  # Detect shadows (slower if True)
MOTION_COOLDOWN_FRAMES = 30    # Keep motion active for this many frames after last detection (~1 sec)
NO_MOTION_TIMEOUT = 3.0        # Seconds of no motion before stopping Arduino sequence
MOTION_MIN_CONTOURS = 1        # Minimum number of motion regions to trigger detection
MOTION_LEARNING_RATE = 0.002   # Background learning rate
MOTION_CONSECUTIVE_FRAMES = 3  # Number of consecutive frames with motion required to trigger
MOTION_STABILIZATION_FRAMES = 150  # Frames to wait before detecting (5 seconds at 30fps)

# =============================================================================
# Logging Settings
# =============================================================================
LOG_FILE = '/home/pi/opencv_ws/opencv_detection_project/logs/detection.log'
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR

# =============================================================================
# Object Detection Settings
# =============================================================================
CASCADE_TYPES = ['face', 'eye']  # Which cascades to load for object detection

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
BOX_COLOR_MOTION = (0, 255, 255)  # Cyan for motion regions

FONT = cv2.FONT_HERSHEY_SIMPLEX
