# Gazebo Harmonic Simulation Setup

**Date:** October 28, 2025

This guide covers the Gazebo Harmonic simulation setup for the mecanum robot, replacing Isaac Sim.

## Overview

- **Host System**: Ubuntu 24.04 with ROS2 Jazzy (native)
- **Simulation**: Gazebo Harmonic in Docker container
- **Communication**: ROS2 topics via host network
- **Sensors**: RGB camera, depth camera, IMU, lidar, mecanum wheel odometry

## Architecture

```
┌─────────────────────────────────────┐
│   Ubuntu 24.04 Host (Native)        │
│                                      │
│  ┌────────────────────────────┐    │
│  │  Client ROS2 Nodes         │    │
│  │  - VILA nodes              │    │
│  │  - GUI                     │    │
│  │  - Navigation              │    │
│  └────────────────────────────┘    │
│              │                       │
│              │ ROS2 Topics           │
│              │ (host network)        │
└──────────────┼───────────────────────┘
               │
┌──────────────┼───────────────────────┐
│   Docker Container                   │
│              │                       │
│  ┌───────────▼──────────────┐       │
│  │  Gazebo Harmonic         │       │
│  │  - Simulated Robot       │       │
│  │  - Sensors (RGB/Depth/   │       │
│  │    IMU/Lidar/Odom)       │       │
│  │  - ROS2 Bridge           │       │
│  └──────────────────────────┘       │
└──────────────────────────────────────┘
```

## Quick Start

### 1. Build the Container

```bash
./build_gazebo_container.sh
```

This builds a Docker image with:
- Ubuntu 24.04
- ROS2 Jazzy
- Gazebo Harmonic
- ros_gz_bridge

### 2. Run the Simulation

**With GUI:**
```bash
./run_gazebo_sim.sh
```

**Headless (no GUI):**
```bash
./run_gazebo_headless.sh
```

**Development Shell:**
```bash
./run_gazebo_shell.sh
```

### 3. Verify Topics

In a new terminal on the host:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list
```

You should see:
- `/scan` - Lidar data (sensor_msgs/LaserScan)
- `/camera/image_raw` - RGB camera (sensor_msgs/Image)
- `/depth/image_raw` - Depth camera (sensor_msgs/Image)
- `/imu` - IMU data (sensor_msgs/Imu)
- `/odom` - Odometry (nav_msgs/Odometry)
- `/cmd_vel` - Velocity commands (geometry_msgs/Twist)

### 4. Test Robot Control

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.0}}"
```

## File Structure

```
Robot_LLM/
├── Dockerfile.gazebo_jazzy          # Container definition
├── build_gazebo_container.sh        # Build script
├── run_gazebo_sim.sh                # Run with GUI
├── run_gazebo_headless.sh           # Run headless
├── run_gazebo_shell.sh              # Development shell
└── gazebo_sim/
    ├── mecanum_bot/
    │   └── urdf/
    │       └── mecanum_bot_gazebo.urdf.xacro  # Robot with Gazebo plugins
    └── robot_sim/
        ├── package.xml              # ROS2 package
        ├── CMakeLists.txt
        ├── launch/
        │   └── robot_sim.launch.py  # Main launch file
        └── worlds/
            └── empty.sdf            # Gazebo world with obstacles
```

## Sensor Configuration

### Lidar
- **Topic**: `/scan`
- **Type**: sensor_msgs/LaserScan
- **Rate**: 10 Hz
- **Range**: 0.12m - 10m
- **FOV**: 360°
- **Samples**: 360

### RGB Camera
- **Topic**: `/camera/image_raw`
- **Type**: sensor_msgs/Image
- **Rate**: 30 Hz
- **Resolution**: 640x480
- **FOV**: 60° (1.047 rad)

### Depth Camera
- **Topic**: `/depth/image_raw`
- **Type**: sensor_msgs/Image
- **Rate**: 30 Hz
- **Resolution**: 640x480
- **Range**: 0.1m - 10m

### IMU
- **Topic**: `/imu`
- **Type**: sensor_msgs/Imu
- **Rate**: 100 Hz
- **Data**: Angular velocity, linear acceleration

### Odometry
- **Topic**: `/odom`
- **Type**: nav_msgs/Odometry
- **Rate**: 50 Hz
- **Frame**: odom → base_link

## Integration with Client

The simulation acts as a drop-in replacement for the physical robot (Jetson Orin Nano). Your client nodes running natively on Ubuntu 24.04 will connect to the simulated robot automatically via ROS2 discovery.

### Using with Your Client System

1. Start the Gazebo simulation (in container)
2. In a separate terminal, run your client nodes natively:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/marc/Robot_LLM
colcon build
source install/setup.bash
ros2 launch robot_vila_system client_system.launch.py
```

## Troubleshooting

### No Topics Visible

Check ROS_DOMAIN_ID matches:
```bash
# On host
echo $ROS_DOMAIN_ID  # Should be 0

# In container (via run_gazebo_shell.sh)
echo $ROS_DOMAIN_ID  # Should be 0
```

### GUI Not Showing

Ensure X11 forwarding is enabled:
```bash
xhost +local:docker
```

### Robot Not Moving

Check that cmd_vel topic is being published:
```bash
ros2 topic echo /cmd_vel
```

### Slow Performance

Run in headless mode:
```bash
./run_gazebo_headless.sh
```

## Customization

### Modify Robot

Edit: `gazebo_sim/mecanum_bot/urdf/mecanum_bot_gazebo.urdf.xacro`

### Modify World

Edit: `gazebo_sim/robot_sim/worlds/empty.sdf`

Add obstacles, change lighting, etc.

### Change Sensor Parameters

In the URDF file, adjust plugin parameters:
- Update rate
- Resolution
- Range
- Noise levels

## Advantages Over Isaac Sim

1. **Lighter Weight**: Faster startup, lower resource usage
2. **Native ROS2**: No bridge complexity
3. **Better Documentation**: Extensive Gazebo community
4. **Standard Tools**: Works with RViz, rqt, etc.
5. **Easier Debugging**: Standard ROS2 tooling

## Next Steps

1. Test with your VILA vision nodes
2. Integrate with navigation stack
3. Test GUI interactions
4. Validate sensor data quality
5. Eventually switch to physical robot (same topics!)

## Notes

- Container uses `--network host` for ROS2 discovery
- URDF is automatically processed by xacro
- Bridge handles Gazebo ↔ ROS2 message conversion
- Simulation time is enabled by default

