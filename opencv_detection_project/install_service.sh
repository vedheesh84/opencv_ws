#!/bin/bash
#
# Installation script for the Unified Detection System service
# Run this script with sudo to install and enable the service
#

set -e

PROJECT_DIR="/home/pi/opencv_ws/opencv_detection_project"
SERVICE_FILE="${PROJECT_DIR}/detection.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "=========================================="
echo "Unified Detection System - Service Setup"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: Please run this script with sudo"
    echo "Usage: sudo bash install_service.sh"
    exit 1
fi

# Create logs directory
echo "Creating logs directory..."
mkdir -p "${PROJECT_DIR}/logs"
chown -R pi:pi "${PROJECT_DIR}/logs"

# Make scripts executable
echo "Setting script permissions..."
chmod +x "${PROJECT_DIR}/start_detection.sh"
chmod +x "${PROJECT_DIR}/unified_detector.py"

# Add pi user to required groups for hardware access
echo "Adding user 'pi' to required groups..."
usermod -aG video pi 2>/dev/null || true
usermod -aG dialout pi 2>/dev/null || true  # For serial port access

# Copy service file to systemd directory
echo "Installing systemd service..."
cp "${SERVICE_FILE}" "${SYSTEMD_DIR}/detection.service"

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable the service to start on boot
echo "Enabling service to start on boot..."
systemctl enable detection.service

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Available commands:"
echo "  Start service:    sudo systemctl start detection"
echo "  Stop service:     sudo systemctl stop detection"
echo "  Restart service:  sudo systemctl restart detection"
echo "  Check status:     sudo systemctl status detection"
echo "  View logs:        sudo journalctl -u detection -f"
echo "  Disable autostart: sudo systemctl disable detection"
echo ""
echo "Log files location: ${PROJECT_DIR}/logs/"
echo ""
echo "The service will start automatically on next boot."
echo "To start it now, run: sudo systemctl start detection"
echo ""
