# Arduino Communication Protocol

This document describes the serial communication protocol between the Raspberry Pi and Arduino for the eye servo control system.

## Serial Settings

- **Port:** `/dev/ttyACM0`
- **Baud Rate:** 9600
- **Data Bits:** 8
- **Stop Bits:** 1
- **Parity:** None
- **Line Ending:** Newline (`\n`)

## Commands (Pi → Arduino)

| Command | Description | Arduino Response |
|---------|-------------|------------------|
| `open` | Open eyes (one-time move, no loop) | `Opened.` |
| `move_loop` | Open eyes and start eyeball movement loop | `Move Loop Active.` |
| `close` | Close eyes and stop all movement (rest position) | `Closed.` |

## System Behavior

### On Startup
1. Arduino initializes PCA9685 servo driver
2. Eyes close to default rest position
3. Arduino sends: `System Ready. Default = CLOSE state`

### Motion Detection Flow
1. **Motion Detected** → Pi sends `move_loop`
   - Eyes open
   - Eyeballs start bouncing animation
2. **No Motion for 3 seconds** → Pi sends `close`
   - Eyeball loop stops
   - Eyes close to rest position

### Servo Channels (PCA9685)

| Channel | Servo | Description |
|---------|-------|-------------|
| 0 | Left Upper Eyelid | Range: -40° to 60° |
| 1 | Left Eyeball | Range: 0° to 25° |
| 2 | Left Lower Eyelid | Range: -45° to 0° |
| 3 | Right Upper Eyelid | Range: -180° to -130° |
| 4 | Right Eyeball | Range: -60° to 0° |
| 5 | Right Lower Eyelid | Range: -80° to 0° |

## Wiring

### Arduino to PCA9685
- VCC → 5V
- GND → GND
- SDA → A4 (or SDA pin)
- SCL → A5 (or SCL pin)

### Arduino to Raspberry Pi
- Arduino USB → Raspberry Pi USB port
- Creates `/dev/ttyACM0` device

### Power
- Servos need separate 5V power supply
- Connect servo power to PCA9685 V+ terminal
- Share common ground between all components

## Testing Commands

You can test Arduino communication manually:

```bash
# Open serial connection
screen /dev/ttyACM0 9600

# Or use echo
echo "open" > /dev/ttyACM0
echo "move_loop" > /dev/ttyACM0
echo "close" > /dev/ttyACM0
```

## Python Test Script

```python
import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # Wait for Arduino reset

# Read startup message
while ser.in_waiting:
    print(ser.readline().decode().strip())

# Test commands
ser.write(b'open\n')
time.sleep(2)
print(ser.readline().decode().strip())

ser.write(b'move_loop\n')
time.sleep(5)
print(ser.readline().decode().strip())

ser.write(b'close\n')
time.sleep(2)
print(ser.readline().decode().strip())

ser.close()
```

## Configuration

Edit `config.py` to adjust:
- `ARDUINO_PORT`: Serial port (default: `/dev/ttyACM0`)
- `ARDUINO_BAUDRATE`: Baud rate (default: `9600`)
- `NO_MOTION_TIMEOUT`: Seconds before closing eyes (default: `3.0`)
