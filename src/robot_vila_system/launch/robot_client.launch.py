#!/usr/bin/env python3
"""
ROS2 Launch file for robot client
Launches only the robot client node (for robot-side deployment)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    """Generate launch description for robot client"""
    
    # Launch arguments
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='yahboomcar_x3_01',
        description='Robot ID for the system'
    )
    
    camera_device_arg = DeclareLaunchArgument(
        'camera_device',
        default_value='0',
        description='Camera device index'
    )
    
    sensor_simulation_arg = DeclareLaunchArgument(
        'sensor_simulation',
        default_value='true',
        description='Whether to use simulated sensors'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for robot client'
    )
    
    # Get launch configurations
    robot_id = LaunchConfiguration('robot_id')
    camera_device = LaunchConfiguration('camera_device')
    sensor_simulation = LaunchConfiguration('sensor_simulation')
    log_level = LaunchConfiguration('log_level')
    
    # Package path
    package_share = FindPackageShare('robot_vila_system').find('robot_vila_system')
    config_dir = os.path.join(package_share, 'config')
    
    # Robot client node
    robot_client_node = Node(
        package='robot_vila_system',
        executable='robot_client',
        name='robot_client',
        namespace='robot_system',
        parameters=[
            {'robot_id': robot_id},
            {'camera_device': camera_device},
            {'sensor_simulation': sensor_simulation},
            os.path.join(config_dir, 'robot_client_config.yaml')
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        # Launch arguments
        robot_id_arg,
        camera_device_arg,
        sensor_simulation_arg,
        log_level_arg,
        
        # System info
        LogInfo(msg='🤖 Launching ROS2 Robot Client'),
        LogInfo(msg=['Robot ID: ', robot_id]),
        LogInfo(msg='📡 Publishing sensor data and camera images'),
        LogInfo(msg='🎯 Receiving commands via ROS2 topics'),
        
        # Robot client node
        robot_client_node,
        
        LogInfo(msg='✅ Robot Client launch complete'),
    ])
