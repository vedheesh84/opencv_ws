#!/usr/bin/env python3
"""
Arduino Serial Controller Module
Handles serial communication with Arduino for servo control

Arduino Commands:
  - "open"      : Open eyes (one-time move)
  - "move_loop" : Open eyes and start eyeball movement loop
  - "close"     : Close eyes (rest position)
"""

import serial
import time
import threading
import config


class ArduinoController:
    """Controls Arduino via serial communication for eye servo system"""

    # Command constants matching Arduino code
    CMD_OPEN = "open"
    CMD_MOVE_LOOP = "move_loop"
    CMD_CLOSE = "close"

    def __init__(self):
        """Initialize Arduino controller"""
        self.serial_port = None
        self.is_connected = False
        self.is_looping = False
        self.eyes_open = False
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

            # Clear any startup messages (Arduino sends "System Ready...")
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
            self.close_eyes()
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

    def open_eyes(self):
        """Send 'open' command - opens eyes without looping"""
        if not self.eyes_open:
            success = self.send_command(self.CMD_OPEN)
            if success:
                self.eyes_open = True
                self.is_looping = False
                response = self.read_response()
                if response:
                    print(f"Arduino: {response}")
            return success
        return True

    def start_move_loop(self):
        """Send 'move_loop' command - opens eyes and starts eyeball movement"""
        if not self.is_looping:
            success = self.send_command(self.CMD_MOVE_LOOP)
            if success:
                self.is_looping = True
                self.eyes_open = True
                response = self.read_response()
                if response:
                    print(f"Arduino: {response}")
            return success
        return True  # Already looping

    def close_eyes(self):
        """Send 'close' command - closes eyes and stops any movement"""
        if self.eyes_open or self.is_looping:
            success = self.send_command(self.CMD_CLOSE)
            if success:
                self.is_looping = False
                self.eyes_open = False
                response = self.read_response()
                if response:
                    print(f"Arduino: {response}")
            return success
        return True  # Already closed

    # Aliases to match the unified_detector interface
    def start_sequence(self):
        """Start the eye movement sequence (alias for start_move_loop)"""
        return self.start_move_loop()

    def stop_sequence(self):
        """Stop the sequence and close eyes (alias for close_eyes)"""
        return self.close_eyes()

    def is_ready(self):
        """Check if controller is ready"""
        return self.is_connected

    def get_status(self):
        """Get current status"""
        return {
            'connected': self.is_connected,
            'eyes_open': self.eyes_open,
            'looping': self.is_looping
        }

    def __del__(self):
        """Cleanup on destruction"""
        self.disconnect()
