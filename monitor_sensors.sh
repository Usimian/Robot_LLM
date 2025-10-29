#!/bin/bash

# Monitor all Gazebo sensor topics

source /opt/ros/jazzy/setup.bash

echo "========================================"
echo "   Gazebo Mecanum Robot Sensor Status"
echo "========================================"
echo ""

# Function to check topic rate
check_topic() {
    local topic=$1
    local name=$2
    echo -n "$name: "
    
    # Try to get rate for 2 seconds
    result=$(timeout 2 ros2 topic hz $topic 2>&1 | head -1)
    
    if [[ $result == *"average rate"* ]]; then
        rate=$(echo $result | grep -oP 'average rate: \K[0-9.]+')
        echo "✅ ${rate} Hz"
    else
        echo "❌ Not publishing"
    fi
}

check_topic "/cmd_vel" "Drive Control   "
check_topic "/scan" "Lidar          "
check_topic "/imu" "IMU            "
check_topic "/camera/image" "RGB Camera     "
check_topic "/depth_camera/image" "Depth Camera   "

echo ""
echo "========================================"
echo "To view sensor data:"
echo "  ros2 topic echo /scan --once"
echo "  ros2 topic echo /camera/image --once"
echo "  ros2 topic echo /depth_camera/image --once"
echo "  ros2 topic echo /imu --once"
echo "========================================"

