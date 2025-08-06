# Robot Sensor Integration Guide

## Overview
This document explains how to integrate your robot with the VILA Robot Hub sensor monitoring system. The hub provides real-time sensor data display and monitoring capabilities through a centralized GUI interface.

## System Architecture

```
Robot (Jetson Orin Nano) → VILA Robot Hub Server → GUI Client
    ↓                           ↓                     ↓
Sensor Data              Store & Broadcast      Display & Monitor
```

## Server Code Location
- **Robot Server Code**: `/home/marc/yahboomcar_ros2/yahboomcar_ws`
- **Client Code**: `/home/marc/Robot_LLM` (this workspace)
- **Interface Package**: `slam_nav` package in the ROS2 system

## Required Integration Steps

### 1. Install HTTP Client Dependencies
```bash
pip3 install requests
```

### 2. Import the Robot Client
```python
import sys
sys.path.append('/home/marc/Robot_LLM')
from robot_client_examples import HTTPRobotClient
```

### 3. Initialize Robot Client
```python
# Replace with your robot's unique ID
robot_client = HTTPRobotClient(
    server_url="http://localhost:5000",  # Hub server URL
    robot_id="your_robot_id_here"
)
```

### 4. Register Your Robot
```python
robot_info = {
    'robot_id': 'your_robot_id_here',
    'name': 'Your Robot Name',
    'capabilities': ['navigation', 'camera', 'sensors'],
    'connection_type': 'http',
    'position': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'heading': 0.0},
    'battery_level': 95.0,  # Current battery percentage
    'sensor_data': {
        # Initial sensor readings (optional)
        'battery_voltage': 12.4,
        'temperature': 25.0
    }
}

success = robot_client.register(robot_info)
if not success:
    print("❌ Failed to register with hub")
    exit(1)
```

## Sensor Data Format

### Supported Sensors
The GUI supports these sensor types with specific key names:

| Sensor Type | Key Name | Expected Type | Unit | GUI Display |
|-------------|----------|---------------|------|-------------|
| **Lidar** | `lidar_distance` | float | meters | "2.45 meters" |
| **IMU** | `imu_heading` | float | degrees | "90.0 degrees" |
| **GPS** | `gps_lat`, `gps_lon` | float | coordinates | "37.7749, -122.4194" |
| **Camera** | `camera_status` | string | status | "active" |
| **Battery** | `battery_voltage` | float | volts | "12.1 volts" |
| **Temperature** | `temperature` | float | °C | "35.2 °C" |

### Data Validation Rules
- **Numbers**: Must be valid float/int values
- **Strings**: Non-empty strings for status fields
- **GPS**: Both `gps_lat` and `gps_lon` required for display
- **Invalid data**: Shows "Invalid data" in red
- **Missing data**: Shows "----" in red

## Sending Sensor Data

### Basic Usage
```python
def send_sensor_readings():
    # Collect real sensor data from your robot
    sensor_data = {
        'battery_voltage': get_battery_voltage(),    # e.g., 12.1
        'temperature': get_motor_temperature(),      # e.g., 35.2
        'lidar_distance': get_lidar_reading(),       # e.g., 2.45
        'imu_heading': get_imu_heading(),            # e.g., 90.0
        'camera_status': get_camera_status(),        # e.g., "active"
        'gps_lat': get_gps_latitude(),               # e.g., 37.7749
        'gps_lon': get_gps_longitude()               # e.g., -122.4194
    }
    
    # Send to hub
    success = robot_client.send_sensor_data(sensor_data)
    if success:
        print("📊 Sensor data sent successfully")
    else:
        print("❌ Failed to send sensor data")
```

### Periodic Updates
```python
import threading
import time

def sensor_update_loop():
    """Send sensor updates every 5 seconds"""
    while True:
        try:
            send_sensor_readings()
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            print(f"Sensor update error: {e}")
            time.sleep(10)  # Wait longer on error

# Start sensor updates in background thread
sensor_thread = threading.Thread(target=sensor_update_loop, daemon=True)
sensor_thread.start()
```

## ROS2 Integration Example

For ROS2 robots, integrate with your existing node:

```python
import rospy
from sensor_msgs.msg import BatteryState, Temperature
from geometry_msgs.msg import PoseWithCovarianceStamped
from robot_client_examples import HTTPRobotClient

class RobotSensorReporter:
    def __init__(self):
        # Initialize robot client
        self.robot_client = HTTPRobotClient(
            server_url="http://localhost:5000",
            robot_id="yahboomcar_x3_01"  # Your robot ID
        )
        
        # Register robot
        self.register_robot()
        
        # ROS subscribers for sensor data
        self.battery_sub = rospy.Subscriber('/battery_state', BatteryState, self.battery_callback)
        self.temp_sub = rospy.Subscriber('/temperature', Temperature, self.temperature_callback)
        self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        
        # Store latest sensor readings
        self.latest_sensors = {}
        
        # Start periodic reporting
        self.timer = rospy.Timer(rospy.Duration(5.0), self.report_sensors)
    
    def register_robot(self):
        robot_info = {
            'robot_id': 'yahboomcar_x3_01',
            'name': 'YahBoom Car X3',
            'capabilities': ['navigation', 'camera', 'sensors', 'slam'],
            'connection_type': 'http',
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'heading': 0.0},
            'battery_level': 100.0
        }
        self.robot_client.register(robot_info)
    
    def battery_callback(self, msg):
        self.latest_sensors['battery_voltage'] = msg.voltage
        # Also update battery percentage
        battery_percent = (msg.voltage - 10.0) / (12.6 - 10.0) * 100.0
        self.robot_client.update_status({'battery_level': max(0, min(100, battery_percent))})
    
    def temperature_callback(self, msg):
        self.latest_sensors['temperature'] = msg.temperature
    
    def pose_callback(self, msg):
        # Extract heading from quaternion
        import tf.transformations
        orientation = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        self.latest_sensors['imu_heading'] = math.degrees(yaw)
    
    def report_sensors(self, event):
        """Periodic sensor reporting"""
        if self.latest_sensors:
            self.robot_client.send_sensor_data(self.latest_sensors)
```

## Error Handling

### Connection Errors
```python
def robust_sensor_update():
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            success = robot_client.send_sensor_data(sensor_data)
            if success:
                return True
            retry_count += 1
        except requests.exceptions.ConnectionError:
            print(f"Connection failed, retry {retry_count + 1}/{max_retries}")
            retry_count += 1
            time.sleep(2)  # Wait before retry
        except Exception as e:
            print(f"Sensor update error: {e}")
            break
    
    return False
```

### Data Validation
```python
def validate_sensor_data(data):
    """Validate sensor data before sending"""
    validated = {}
    
    # Validate numeric sensors
    numeric_sensors = ['battery_voltage', 'temperature', 'lidar_distance', 'imu_heading', 'gps_lat', 'gps_lon']
    for key in numeric_sensors:
        if key in data:
            try:
                value = float(data[key])
                if not math.isnan(value) and not math.isinf(value):
                    validated[key] = value
            except (ValueError, TypeError):
                print(f"Invalid numeric value for {key}: {data[key]}")
    
    # Validate string sensors
    string_sensors = ['camera_status']
    for key in string_sensors:
        if key in data and isinstance(data[key], str) and data[key].strip():
            validated[key] = data[key].strip()
    
    return validated
```

## GUI Display Behavior

### Visual States
- **Live Data**: Black text with actual values (e.g., "12.45 volts")
- **No Data**: Red "----" for unavailable sensors
- **Invalid Data**: Red "Invalid data" for corrupted values
- **Status**: Green "Live data (N sensors)" when receiving data

### Update Frequency
- **Recommended**: 5-10 second intervals for most sensors
- **Battery**: Every 30-60 seconds (slow changing)
- **Lidar/IMU**: Every 1-5 seconds (fast changing)
- **GPS**: Every 10-30 seconds (moderate changing)

## Testing Your Integration

### 1. Test Registration
```python
# Test basic registration
success = robot_client.register(robot_info)
assert success, "Registration failed"
print("✅ Registration successful")
```

### 2. Test Sensor Data
```python
# Test with known good data
test_data = {
    'battery_voltage': 12.1,
    'temperature': 25.5,
    'camera_status': 'active'
}

success = robot_client.send_sensor_data(test_data)
assert success, "Sensor data failed"
print("✅ Sensor data sent successfully")
```

### 3. Monitor GUI
1. Open the Robot GUI application
2. Select your robot in the "Robots" tab
3. Switch to "Monitoring" tab
4. Verify sensor readings appear in black text
5. Check that missing sensors show red "----"

## API Reference

### HTTPRobotClient Methods

```python
class HTTPRobotClient:
    def __init__(self, server_url="http://localhost:5000", robot_id="robot_01"):
        """Initialize robot client"""
    
    def register(self, robot_info: dict) -> bool:
        """Register robot with hub"""
    
    def send_sensor_data(self, sensor_data: dict) -> bool:
        """Send sensor readings to hub"""
    
    def update_status(self, status_updates: dict) -> bool:
        """Update robot status (battery, position, etc.)"""
    
    def get_pending_commands(self) -> list:
        """Get pending commands from hub"""
```

## Troubleshooting

### Common Issues

1. **"Registration failed"**
   - Check server is running on port 5000
   - Verify robot_id is unique
   - Check network connectivity

2. **"Sensor data failed"**
   - Validate data types (numbers as float/int)
   - Check for NaN or infinite values
   - Ensure robot is registered first

3. **GUI shows "----" for all sensors**
   - Verify sensor data keys match expected names
   - Check data validation
   - Confirm robot is selected in GUI

4. **GUI shows "Invalid data"**
   - Check data types (strings for status, numbers for measurements)
   - Validate numeric ranges
   - Remove NaN/infinite values

### Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
logger = logging.getLogger('RobotClient')
logger.setLevel(logging.DEBUG)
```

## Support

For integration issues:
1. Check the robot hub logs: `/home/marc/Robot_LLM/robot_hub.log`
2. Verify server status: `http://localhost:5000/health`
3. Test with the demo: `python3 robot_client_examples.py` (option 4)

---

**Remember**: Only send real sensor data. The GUI will show red "----" for any unavailable sensors, providing clear visual feedback about your robot's sensor status.