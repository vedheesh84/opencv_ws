#!/bin/bash
#
# Uninstallation script for the Unified Detection System service
# Run this script with sudo to disable and remove the service
#

set -e

SYSTEMD_DIR="/etc/systemd/system"

echo "=========================================="
echo "Unified Detection System - Service Removal"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: Please run this script with sudo"
    echo "Usage: sudo bash uninstall_service.sh"
    exit 1
fi

# Stop the service if running
echo "Stopping service..."
systemctl stop detection.service 2>/dev/null || true

# Disable the service
echo "Disabling service..."
systemctl disable detection.service 2>/dev/null || true

# Remove the service file
echo "Removing service file..."
rm -f "${SYSTEMD_DIR}/detection.service"

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

echo ""
echo "=========================================="
echo "Uninstallation complete!"
echo "=========================================="
echo ""
echo "The service has been removed and will no longer start on boot."
echo "Project files are still in /home/pi/opencv_ws/opencv_detection_project/"
echo ""
