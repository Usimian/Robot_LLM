# Robot Client Integration Guide

## Overview

This guide provides complete integration information for controlling the Yahboom X3 robot and reading its sensors via ROS2. The robot uses a **simplified command interface** - you send movement commands via ROS2 services and receive sensor data via ROS2 topics.

**Note**: This guide is client-agnostic. Whether you're using VILA, other AI models, or manual control, the robot interface remains the same.

### Prerequisites
- **ROS2 Humble** installed on your client machine (AMD64 Ubuntu 22.04)
- **Network connection** to the robot (Jetson Orin Nano)
- **robot_msgs** package built on your client system

## 📡 Communication Architecture

```
Client (Your Code) ←→ Network ←→ Robot (Jetson Orin Nano)
     ↓                                    ↓
  ROS2 Service Calls              Robot Interface Node
  ROS2 Topic Subscriptions       Hardware Drivers
```

### Key Principles
- **Robot is a hardware interface** - no AI processing on robot side
- **Client does all processing** - AI models, navigation planning, decision making
- **Simple command interface** - basic move/turn/stop commands
- **Pull-based sensor data** - client subscribes to sensor topics

## 🎯 Robot Control Interface

### Primary Command Service
**Service**: `/robot/execute_command`  
**Type**: `robot_msgs/srv/ExecuteCommand`  
**Purpose**: Send movement commands to the robot

### Service Definition
```yaml
# Request
robot_msgs/RobotCommand command
---
# Response  
bool success                       # Command started successfully?
string result_message              # Human-readable status/error message
```

### RobotCommand Message
```yaml
# robot_msgs/msg/RobotCommand
string robot_id              # Always "yahboomcar_x3_01"
string command_type           # "move", "turn", or "stop"
float64 linear_x              # Forward/backward direction (-1.0 to 1.0)
float64 linear_y              # Left/right direction (-1.0 to 1.0)  
float64 angular_z             # Not used (legacy field)
float64 duration              # Optional timeout (seconds, 0 = no limit)
float64 distance              # Target distance for move (meters)
float64 angular               # Target angle for turn (degrees)
float64 linear_speed          # Movement speed (m/s)
float64 angular_speed         # Rotation speed (rad/s)
int64 timestamp_ns            # Command timestamp (nanoseconds)
string source_node            # Your node name (for logging)
```

## 🎮 Command Types

### 1. Move Command
**Purpose**: Linear movement with precision encoder feedback

```python
command.command_type = "move"
command.linear_x = 1.0        # 1.0=forward, -1.0=backward, 0.0=no movement
command.linear_y = 0.0        # 1.0=right, -1.0=left, 0.0=no strafe  
command.distance = 0.5        # Distance in meters (e.g., 0.1 = 10cm)
command.linear_speed = 0.2    # Speed in m/s (recommended: 0.1-0.5)
command.duration = 10.0       # Optional timeout in seconds
```

**Examples**:
- Forward 20cm: `linear_x=1.0, distance=0.2, linear_speed=0.1`
- Backward 10cm: `linear_x=-1.0, distance=0.1, linear_speed=0.1`
- Diagonal: `linear_x=0.707, linear_y=0.707, distance=0.141` (10cm diagonal)

### 2. Turn Command  
**Purpose**: Rotational movement with precision encoder feedback

```python
command.command_type = "turn"
command.angular = 90.0        # Degrees: +90=left, -90=right
command.angular_speed = 0.5   # Speed in rad/s (recommended: 0.3-0.8)
command.duration = 8.0        # Optional timeout in seconds
```

**Examples**:
- Turn left 90°: `angular=90.0, angular_speed=0.5`
- Turn right 45°: `angular=-45.0, angular_speed=0.3`
- Full rotation: `angular=360.0, angular_speed=0.5`

### 3. Stop Command
**Purpose**: Immediate emergency stop

```python
command.command_type = "stop"
# All other parameters ignored
```

## 📊 Sensor Data Interface

### Primary Sensor Topic
**Topic**: `/robot/sensors`  
**Type**: `robot_msgs/msg/SensorData`  
**Frequency**: ~2Hz  

### SensorData Message
```yaml
# robot_msgs/msg/SensorData
string robot_id              # "yahboomcar_x3_01"
float64 battery_voltage       # Battery voltage (V)
float64 cpu_temp             # CPU temperature (°C)
float64 distance_front       # Front lidar distance (m)
float64 distance_left        # Left lidar distance (m) 
float64 distance_right       # Right lidar distance (m)
float64 cpu_usage            # CPU usage (0-100%)
string camera_status         # "active" or "unknown"
int64 timestamp_ns           # Timestamp (nanoseconds)
```

### Additional Sensor Topics
```bash
# Camera (Intel RealSense D435i)
/realsense/camera/color/image_raw           # RGB images (sensor_msgs/Image)
/realsense/camera/depth/image_rect_raw      # Depth images (sensor_msgs/Image)  
/realsense/camera/aligned_depth_to_color/image_raw  # Aligned depth (sensor_msgs/Image)

# Lidar (S2 LiDAR)
/scan                                       # Laser scan (sensor_msgs/LaserScan)

# IMU and Motion
/imu/data_raw                              # Raw IMU data (sensor_msgs/Imu)
/imu/mag                                   # Magnetometer (sensor_msgs/MagneticField)
/odom                                      # Filtered odometry (nav_msgs/Odometry)
/odom_raw                                  # Raw wheel odometry (nav_msgs/Odometry)

# System Status  
/voltage                                   # Battery voltage (std_msgs/Float32)
# Manual robot command
ros2 service call /robot/execute_command robot_msgs/srv/ExecuteCommand \
  "{command: {robot_id: 'yahboomcar_x3_01', command_type: 'move', 
  linear_x: 1.0, distance: 0.1, linear_speed: 0.1, source_node: 'test'}}"

This guide provides everything needed to integrate with the robot system. The robot acts as a simple hardware interface - you send movement commands and receive sensor data. All AI processing (VILA, GPT-4V, custom models, etc.), path planning, and decision making happens on your client system.
```
