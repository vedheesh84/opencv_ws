# OpenCV Object Detection and Color Identification

Complete setup for object detection and color identification using OpenCV ML algorithms on Raspberry Pi 4 Model B with Camera Rev 1.3.

## Hardware Requirements

- Raspberry Pi 4 Model B
- Raspberry Pi Camera Rev 1.3
- Ubuntu Desktop 22.04
- Display (for viewing detection results)

## Features

- **Haar Cascade Object Detection**: Detect faces, eyes, full bodies, and upper bodies
- **HSV Color Detection**: Identify and track 8 colors (red, green, blue, yellow, orange, purple, white, black)
- **Combined Detection**: Run both object and color detection simultaneously with relationship analysis
- **Real-time Processing**: Optimized for Raspberry Pi 4 performance
- **Camera Test Utility**: Verify camera setup before running detection

## Project Structure

```
opencv_detection_project/
├── camera_test.py          # Test camera functionality
├── object_detection.py     # Haar Cascade object detection
├── color_detection.py      # HSV-based color identification
├── combined_detection.py   # Both detections running together
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### Step 1: Enable Legacy Camera Support

The picamera library requires legacy camera support to be enabled:

```bash
sudo raspi-config
```

Navigate to: **Interface Options** → **Legacy Camera** → **Enable**

Then reboot:

```bash
sudo reboot
```

### Step 2: Install System Dependencies

```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install OpenCV dependencies
sudo apt-get install -y python3-opencv opencv-data

# Install pip if not already installed
sudo apt-get install -y python3-pip

# Install development libraries
sudo apt-get install -y libatlas-base-dev libjasper-dev libqtgui4 libqt4-test
```

### Step 3: Install Python Dependencies

```bash
cd opencv_detection_project

# Install Python packages
pip3 install -r requirements.txt

# Or install individually:
pip3 install opencv-python==4.8.1.78
pip3 install opencv-contrib-python==4.8.1.78
pip3 install numpy==1.24.3
pip3 install picamera==1.13
```

### Step 4: Verify Installation

Test your camera setup:

```bash
python3 camera_test.py
```

You should see a live camera feed. Press 'q' to quit, 's' to save a snapshot.

## Usage

### 1. Camera Test

Test your camera before running detection:

```bash
python3 camera_test.py
```

**Controls:**
- `q` - Quit
- `s` - Save snapshot

### 2. Object Detection

Detect objects using Haar Cascade classifiers:

```bash
python3 object_detection.py
```

**Detects:**
- Faces (frontal)
- Eyes
- Full bodies
- Upper bodies

**Controls:**
- `q` - Quit

### 3. Color Detection

Detect and track colors in real-time:

```bash
python3 color_detection.py
```

**Detects:**
- Red, Green, Blue
- Yellow, Orange, Purple
- White, Black

**Controls:**
- `q` - Quit

### 4. Combined Detection

Run both object and color detection together:

```bash
python3 combined_detection.py
```

**Features:**
- Simultaneous object and color detection
- Relationship analysis (which colors appear in detected objects)
- Real-time FPS counter

**Controls:**
- `q` - Quit
- `a` - Toggle analysis mode on/off

## Configuration

Edit `config.py` to customize detection parameters:

### Camera Settings

```python
CAMERA_RESOLUTION = (640, 480)  # Reduce for better performance
CAMERA_FRAMERATE = 30
CAMERA_ROTATION = 0  # Set to 180 if camera is upside down
```

### Detection Parameters

```python
SCALE_FACTOR = 1.1  # Object detection sensitivity
MIN_NEIGHBORS = 5   # Higher = fewer false positives
MIN_SIZE = (30, 30) # Minimum object size in pixels
```

### Color Ranges

Adjust HSV color ranges based on your lighting conditions:

```python
COLOR_RANGES = {
    'red': [((0, 100, 100), (10, 255, 255))],
    'green': [((40, 50, 50), (80, 255, 255))],
    # ... customize as needed
}
```

### Performance Optimization

```python
MIN_CONTOUR_AREA = 500  # Increase to reduce noise
CAMERA_RESOLUTION = (320, 240)  # Lower resolution = higher FPS
```

## Troubleshooting

### Camera Not Detected

1. Check camera cable connection
2. Enable legacy camera: `sudo raspi-config`
3. Verify camera is detected:
   ```bash
   vcgencmd get_camera
   ```
   Should show: `supported=1 detected=1`

### ImportError: No module named 'picamera'

```bash
pip3 install picamera==1.13
```

### Haar Cascade Files Not Found

Install OpenCV data files:

```bash
sudo apt-get install opencv-data
```

Or manually download cascades:

```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

Update paths in `config.py`:

```python
HAAR_CASCADE_PATHS = {
    'face': '/path/to/haarcascade_frontalface_default.xml',
}
```

### Low FPS / Performance Issues

1. Reduce camera resolution in `config.py`:
   ```python
   CAMERA_RESOLUTION = (320, 240)
   ```

2. Increase MIN_NEIGHBORS to reduce processing:
   ```python
   MIN_NEIGHBORS = 7
   ```

3. Disable X11 forwarding if running over SSH

4. Close other applications to free resources

### Color Detection Not Working

1. Adjust HSV ranges in `config.py` based on lighting
2. Increase MIN_CONTOUR_AREA to filter noise
3. Ensure adequate lighting in the room
4. Calibrate for your specific environment

## Performance Tips

### For Best Performance on Raspberry Pi 4:

1. **Resolution**: Use 640x480 or lower
2. **Framerate**: 30 FPS is optimal
3. **Cooling**: Ensure adequate cooling (heatsink/fan)
4. **Power**: Use official Raspberry Pi power supply
5. **OS**: Close unnecessary applications
6. **GPU Memory**: Increase GPU memory to 256MB:
   ```bash
   sudo raspi-config
   # Performance Options → GPU Memory → 256
   ```

## Technical Details

### Object Detection (Haar Cascades)

- **Algorithm**: Viola-Jones cascade classifier
- **Trained on**: Thousands of positive and negative images
- **Advantages**: Fast, low computational cost, works on Pi 4
- **Best for**: Frontal faces, eyes, basic shapes

### Color Detection (HSV)

- **Color Space**: HSV (Hue, Saturation, Value)
- **Method**: Threshold-based segmentation with morphological operations
- **Advantages**: Robust to lighting changes
- **Process**: BGR → HSV → Threshold → Contour detection

## Advanced Usage

### Custom Color Ranges

To detect custom colors, add to `config.py`:

```python
COLOR_RANGES['pink'] = [((140, 50, 50), (170, 255, 255))]
```

### Custom Haar Cascades

To detect custom objects:

1. Train your own Haar Cascade
2. Add path to `config.py`:
   ```python
   HAAR_CASCADE_PATHS['custom'] = '/path/to/custom.xml'
   ```
3. Use in detection code

### Headless Mode (No Display)

For running without display (SSH/headless):

1. Comment out `cv2.imshow()` calls
2. Save frames to disk instead:
   ```python
   cv2.imwrite(f'frame_{count}.jpg', image)
   ```

## Examples

### Detect Red Objects Only

Modify `color_detection.py`:

```python
# In config.py, keep only red:
COLOR_RANGES = {
    'red': [((0, 100, 100), (10, 255, 255)),
            ((160, 100, 100), (180, 255, 255))]
}
```

### Face Detection with Dominant Color

Use `combined_detection.py` with analysis mode enabled (press 'a').

## License

This project uses OpenCV which is licensed under Apache 2.0 License.

## Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Raspberry Pi Camera Documentation](https://picamera.readthedocs.io/)
- [Haar Cascade Training](https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html)
- [HSV Color Space](https://en.wikipedia.org/wiki/HSL_and_HSV)

## Support

For issues or questions:
1. Check troubleshooting section above
2. Verify hardware connections
3. Review configuration settings
4. Check system logs: `dmesg | grep -i camera`

---

**Created for Raspberry Pi 4 Model B with Camera Rev 1.3 running Ubuntu Desktop 22.04**
