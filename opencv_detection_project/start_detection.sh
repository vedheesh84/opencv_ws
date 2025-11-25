#!/bin/bash
#
# Startup script for Unified Detection System
# This script handles initialization and runs the detection program
#

# Configuration
PROJECT_DIR="/home/pi/opencv_ws/opencv_detection_project"
LOG_DIR="${PROJECT_DIR}/logs"
PYTHON_SCRIPT="${PROJECT_DIR}/unified_detector.py"
PYTHON_BIN="/usr/bin/python3"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Log startup
echo "$(date): Starting Unified Detection System" >> "${LOG_DIR}/startup.log"

# Wait for system to fully initialize
sleep 5

# Wait for camera device to be available
MAX_WAIT=30
WAIT_COUNT=0
while [ ! -e /dev/video0 ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    echo "$(date): Waiting for camera device..." >> "${LOG_DIR}/startup.log"
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ ! -e /dev/video0 ]; then
    echo "$(date): ERROR - Camera device not found after ${MAX_WAIT}s" >> "${LOG_DIR}/startup.log"
    exit 1
fi

echo "$(date): Camera device found" >> "${LOG_DIR}/startup.log"

# Wait for Arduino serial device (optional - continue if not found)
WAIT_COUNT=0
ARDUINO_FOUND=0
while [ $WAIT_COUNT -lt 10 ]; do
    if [ -e /dev/ttyACM0 ]; then
        ARDUINO_FOUND=1
        echo "$(date): Arduino device found at /dev/ttyACM0" >> "${LOG_DIR}/startup.log"
        break
    elif [ -e /dev/ttyUSB0 ]; then
        ARDUINO_FOUND=1
        echo "$(date): Arduino device found at /dev/ttyUSB0" >> "${LOG_DIR}/startup.log"
        break
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ $ARDUINO_FOUND -eq 0 ]; then
    echo "$(date): WARNING - Arduino device not found, continuing without it" >> "${LOG_DIR}/startup.log"
fi

# Change to project directory
cd "${PROJECT_DIR}"

# Run the detection system in headless mode
echo "$(date): Launching detection system" >> "${LOG_DIR}/startup.log"
exec ${PYTHON_BIN} ${PYTHON_SCRIPT} --headless 2>&1 | tee -a "${LOG_DIR}/detection.log"
