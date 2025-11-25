#!/usr/bin/env python3
"""
Arduino Serial Controller Module
Handles serial communication with Arduino for servo control

Arduino Commands:
  - "start"  : Trigger one full animation sequence (runs autonomously, returns to rest)
  - "stop"   : Interrupt current sequence and return to rest position
  - "status" : Query current state (RUNNING/IDLE)

The Arduino runs a single-loop sequence:
  1. Move from rest to active position
  2. Perform eyeball movement animation (2 cycles)
  3. Automatically return to rest position
"""

import serial
import time
import threading
import config


class ArduinoController:
    """Controls Arduino via serial communication for eye servo system"""

    # Command constants matching Arduino code
    CMD_START = "start"
    CMD_STOP = "stop"
    CMD_STATUS = "status"

    def __init__(self):
        """Initialize Arduino controller"""
        self.serial_port = None
        self.is_connected = False
        self.is_running = False
        self.sequence_triggered = False
        self.lock = threading.Lock()
        self.connect()

    def connect(self):
        """Establish serial connection to Arduino"""
        try:
            self.serial_port = serial.Serial(
                port=config.ARDUINO_PORT,
                baudrate=config.ARDUINO_BAUDRATE,
                timeout=config.ARDUINO_TIMEOUT
            )
            # Wait for Arduino to reset after serial connection
            time.sleep(config.ARDUINO_INIT_DELAY)
            self.is_connected = True
            print(f"Arduino connected on {config.ARDUINO_PORT}")

            # Clear any startup messages
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()

            # Read and display Arduino startup message
            time.sleep(0.5)
            while self.serial_port.in_waiting > 0:
                msg = self.serial_port.readline().decode('utf-8').strip()
                if msg:
                    print(f"Arduino: {msg}")

            return True
        except serial.SerialException as e:
            print(f"Failed to connect to Arduino: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Close serial connection"""
        if self.serial_port and self.serial_port.is_open:
            self.stop_sequence()
            time.sleep(0.1)
            self.serial_port.close()
            self.is_connected = False
            print("Arduino disconnected")

    def send_command(self, command):
        """Send command to Arduino"""
        if not self.is_connected:
            if not self.connect():
                return False

        with self.lock:
            try:
                cmd = f"{command}\n"
                self.serial_port.write(cmd.encode('utf-8'))
                self.serial_port.flush()
                return True
            except serial.SerialException as e:
                print(f"Serial write error: {e}")
                self.is_connected = False
                return False

    def read_response(self, timeout=0.5):
        """Read response from Arduino"""
        if not self.is_connected:
            return None

        try:
            start_time = time.time()
            while (time.time() - start_time) < timeout:
                if self.serial_port.in_waiting > 0:
                    response = self.serial_port.readline().decode('utf-8').strip()
                    return response
                time.sleep(0.01)
        except serial.SerialException as e:
            print(f"Serial read error: {e}")
            self.is_connected = False
        return None

    def check_responses(self):
        """Check for any pending responses from Arduino (non-blocking)"""
        if not self.is_connected:
            return

        try:
            while self.serial_port.in_waiting > 0:
                response = self.serial_port.readline().decode('utf-8').strip()
                if response:
                    self._handle_response(response)
        except serial.SerialException:
            pass

    def _handle_response(self, response):
        """Handle Arduino response and update state"""
        if response == "STARTED":
            self.is_running = True
            print("Arduino: Sequence started")
        elif response == "DONE":
            self.is_running = False
            self.sequence_triggered = False
            print("Arduino: Sequence complete")
        elif response == "STOPPED":
            self.is_running = False
            self.sequence_triggered = False
            print("Arduino: Sequence stopped")
        elif response == "RUNNING":
            self.is_running = True
        elif response == "IDLE":
            self.is_running = False
        elif response == "STOPPING":
            print("Arduino: Stopping...")
        elif response == "READY":
            print("Arduino: Ready")
        else:
            print(f"Arduino: {response}")

    def start_sequence(self):
        """
        Trigger the animation sequence.
        The Arduino will run the full sequence and automatically return to rest.
        Only sends command if not already running.
        """
        # Check for any pending responses first
        self.check_responses()

        if self.is_running or self.sequence_triggered:
            return True  # Already running, don't spam commands

        success = self.send_command(self.CMD_START)
        if success:
            self.sequence_triggered = True
            response = self.read_response()
            if response:
                self._handle_response(response)
        return success

    def stop_sequence(self):
        """
        Stop the current sequence and return to rest position.
        """
        self.check_responses()

        if not self.is_running and not self.sequence_triggered:
            return True  # Already stopped

        success = self.send_command(self.CMD_STOP)
        if success:
            response = self.read_response()
            if response:
                self._handle_response(response)
            self.sequence_triggered = False
        return success

    def query_status(self):
        """Query Arduino for current status"""
        success = self.send_command(self.CMD_STATUS)
        if success:
            response = self.read_response()
            if response:
                self._handle_response(response)
        return self.is_running

    def is_ready(self):
        """Check if controller is ready"""
        return self.is_connected

    def get_status(self):
        """Get current status"""
        self.check_responses()  # Update state from any pending messages
        return {
            'connected': self.is_connected,
            'running': self.is_running,
            'triggered': self.sequence_triggered
        }

    def __del__(self):
        """Cleanup on destruction"""
        self.disconnect()
