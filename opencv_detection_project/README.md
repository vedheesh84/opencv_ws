# OpenCV Object Detection and Color Identification for Ubuntu 22.04

Complete setup for object detection and color identification using OpenCV ML algorithms on **Raspberry Pi 4 Model B with Camera Rev 1.3 running Ubuntu Desktop 22.04**.

This version uses **Picamera2** (libcamera-based) which is compatible with Ubuntu, unlike the legacy picamera library.

## Hardware Requirements

- Raspberry Pi 4 Model B
- Raspberry Pi Camera Rev 1.3
- Ubuntu Desktop 22.04 (64-bit)
- Display (for viewing detection results)

## Features

- **Haar Cascade Object Detection**: Detect faces, eyes, full bodies, and upper bodies
- **HSV Color Detection**: Identify and track 8 colors (red, green, blue, yellow, orange, purple, white, black)
- **Combined Detection**: Run both object and color detection simultaneously with relationship analysis
- **Real-time Processing**: Optimized for Raspberry Pi 4 performance
- **Picamera2 Support**: Uses modern libcamera framework for Ubuntu compatibility

## Project Structure

```
opencv_detection_project/
├── camera_test_ubuntu.py          # Test camera functionality
├── object_detection_ubuntu.py     # Haar Cascade object detection
├── color_detection_ubuntu.py      # HSV-based color identification
├── combined_detection_ubuntu.py   # Both detections running together
├── config_ubuntu.py               # Configuration settings
├── requirements_ubuntu.txt        # Dependency information
└── README_UBUNTU.md               # This file
```

## Installation

### Step 1: Install System Dependencies

This is the most important step for Ubuntu. Picamera2 must be installed via APT, not pip.

```bash
# Update system packages
sudo apt update
sudo apt upgrade -y

# Install Picamera2 and libcamera
sudo apt install -y python3-picamera2
sudo apt install -y python3-libcamera python3-kms++

# Install OpenCV and dependencies
sudo apt install -y python3-opencv opencv-data
sudo apt install -y python3-numpy

# Install additional dependencies
sudo apt install -y libcap-dev
```

### Step 2: Verify Camera Connection

Check that your camera is detected:

```bash
# Check camera detection
libcamera-hello --list-cameras

# Test camera with a 5-second preview
libcamera-hello -t 5000
```

You should see your camera listed and a preview window should appear.

### Step 3: Set Up Project Files

Copy all the Ubuntu-specific files to your project directory:

```bash
cd /home/pi/opencv_ws/opencv_detection_project

# Copy the Ubuntu-specific files (provided separately)
# - camera_test_ubuntu.py
# - color_detection_ubuntu.py
# - object_detection_ubuntu.py
# - combined_detection_ubuntu.py
# - config_ubuntu.py

# Make scripts executable
chmod +x *.py
```

### Step 4: Verify Picamera2 Installation

Test that Picamera2 is working:

```bash
python3 -c "from picamera2 import Picamera2; print('Picamera2 OK')"
```

If this works without errors, you're ready to go!

### Step 5: Test Camera

Run the camera test script:

```bash
python3 camera_test_ubuntu.py
```

You should see a live camera feed with FPS counter.

## Usage

### 1. Camera Test

Test your camera setup:

```bash
python3 camera_test_ubuntu.py
```

**Controls:**
- `q` - Quit
- `s` - Save snapshot

### 2. Object Detection

Detect faces and eyes using Haar Cascade classifiers:

```bash
python3 object_detection_ubuntu.py
```

**Controls:**
- `q` - Quit

### 3. Color Detection

Detect and track colors in real-time:

```bash
python3 color_detection_ubuntu.py
```

**Controls:**
- `q` - Quit

### 4. Combined Detection

Run both object and color detection together:

```bash
python3 combined_detection_ubuntu.py
```

**Controls:**
- `q` - Quit
- `a` - Toggle analysis mode (shows which colors appear in detected objects)

## Configuration

Edit `config_ubuntu.py` to customize settings:

### Camera Settings

```python
CAMERA_RESOLUTION = (640, 480)  # Reduce for better performance
CAMERA_FRAMERATE = 30
CAMERA_ROTATION = 0  # 0, 90, 180, or 270 degrees
```

### Detection Parameters

```python
SCALE_FACTOR = 1.1  # Object detection sensitivity
MIN_NEIGHBORS = 5   # Higher = fewer false positives
MIN_SIZE = (30, 30) # Minimum object size in pixels
```

### Color Ranges

Adjust HSV ranges based on lighting:

```python
COLOR_RANGES = {
    'red': [((0, 100, 100), (10, 255, 255))],
    'green': [((40, 50, 50), (80, 255, 255))],
    # ... customize as needed
}
```

## Troubleshooting

### Camera Not Detected

1. **Check physical connection**: Ensure cable is properly connected to camera port (not HDMI)

2. **Test with libcamera**:
   ```bash
   libcamera-hello --list-cameras
   ```

3. **Check permissions**:
   ```bash
   sudo usermod -aG video $USER
   # Log out and log back in
   ```

### ImportError: No module named 'picamera2'

Picamera2 must be installed via APT on Ubuntu:

```bash
sudo apt install -y python3-picamera2 python3-libcamera
```

**DO NOT use pip to install picamera2 on Ubuntu** - it won't work properly.

### Haar Cascade Files Not Found

Install opencv-data package:

```bash
sudo apt install opencv-data
```

Verify cascade files exist:

```bash
ls /usr/share/opencv4/haarcascades/
```

### Low FPS / Performance Issues

1. **Reduce resolution** in `config_ubuntu.py`:
   ```python
   CAMERA_RESOLUTION = (320, 240)
   ```

2. **Increase MIN_NEIGHBORS**:
   ```python
   MIN_NEIGHBORS = 7  # More strict detection
   ```

3. **Close other applications**

4. **Enable GPU acceleration**: Ensure you're using system OpenCV:
   ```bash
   python3 -c "import cv2; print(cv2.getBuildInformation())"
   ```

### cv2.imshow() Not Working

If running over SSH or headless:

1. **Enable X11 forwarding** (if using SSH):
   ```bash
   ssh -X pi@your-pi-ip
   ```

2. **Or save frames instead of displaying**:
   Comment out `cv2.imshow()` lines and add:
   ```python
   cv2.imwrite(f'frame_{count}.jpg', image)
   ```

### picamera vs picamera2 Confusion

- **picamera** (v1): Legacy library, requires MMAL (Raspberry Pi OS only)
- **picamera2**: Modern library, uses libcamera (works on Ubuntu)

On Ubuntu 22.04, you MUST use picamera2.

## Performance Tips

### For Best Performance on Raspberry Pi 4 + Ubuntu:

1. **Use system packages**: Install via apt, not pip
2. **Resolution**: 640x480 or lower for smooth FPS
3. **Cooling**: Use heatsink or active cooling
4. **Power**: Official Raspberry Pi power supply (5V 3A)
5. **Swap**: Increase if needed:
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile
   # Set CONF_SWAPSIZE=2048
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

## Key Differences from Raspberry Pi OS Version

| Feature | Raspberry Pi OS | Ubuntu 22.04 |
|---------|----------------|--------------|
| Camera Library | picamera (v1) | Picamera2 |
| Backend | MMAL | libcamera |
| Installation | pip | apt |
| raspi-config | Available | Not available |
| Performance | Similar | Similar |

## Technical Details

### Picamera2 API

```python
from picamera2 import Picamera2

# Initialize
picam2 = Picamera2()

# Configure
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)

# Start
picam2.start()

# Capture frame
frame = picam2.capture_array()  # Returns numpy array in RGB format

# Convert to BGR for OpenCV
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Stop
picam2.stop()
```

### Color Space Conversion

Picamera2 returns frames in RGB format by default, but OpenCV expects BGR:

```python
# Always convert when using with OpenCV
frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
```

## Advanced Usage

### Run Headless (No Display)

To run without X11 display:

```python
# Comment out cv2.imshow() lines
# Save or process frames without displaying:
cv2.imwrite(f'output_{count}.jpg', image)
```

### Custom Haar Cascades

To detect custom objects:

1. Train your cascade (using opencv_traincascade)
2. Add to `config_ubuntu.py`:
   ```python
   HAAR_CASCADE_PATHS['custom'] = '/path/to/custom.xml'
   ```

### Adjust Color Detection for Lighting

Calibrate HSV ranges for your environment:

```python
# Run this to find color ranges:
import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('HSV Calibration')
cv2.createTrackbar('H_min', 'HSV Calibration', 0, 179, nothing)
cv2.createTrackbar('H_max', 'HSV Calibration', 179, 179, nothing)
cv2.createTrackbar('S_min', 'HSV Calibration', 0, 255, nothing)
cv2.createTrackbar('S_max', 'HSV Calibration', 255, 255, nothing)
cv2.createTrackbar('V_min', 'HSV Calibration', 0, 255, nothing)
cv2.createTrackbar('V_max', 'HSV Calibration', 255, 255, nothing)

# Capture and adjust ranges to isolate your target color
```

## Resources

- [Picamera2 Documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [libcamera Documentation](https://libcamera.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Raspberry Pi Camera on Ubuntu](https://ubuntu.com/tutorials/how-to-use-a-raspberry-pi-camera-with-ubuntu)

## Common Commands Reference

```bash
# Check camera
libcamera-hello --list-cameras

# Test camera capture
libcamera-still -o test.jpg

# Check libcamera version
dpkg -l | grep libcamera

# Check picamera2 version
python3 -c "import picamera2; print(picamera2.__version__)"

# Reinstall if needed
sudo apt install --reinstall python3-picamera2

# Check OpenCV build info
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i libcamera
```

## License

This project uses OpenCV (Apache 2.0 License) and Picamera2 (BSD License).

---

**Optimized for Raspberry Pi 4 Model B with Camera Rev 1.3 on Ubuntu Desktop 22.04 LTS**

For Raspberry Pi OS (Bullseye/Bookworm), use the original picamera-based version instead.
