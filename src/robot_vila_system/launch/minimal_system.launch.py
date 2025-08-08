#!/usr/bin/env python3
"""
ROS2 Launch file for minimal VILA Robot System
Launches only core server and vision nodes
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    """Generate launch description for minimal VILA robot system"""
    
    # Launch arguments
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='yahboomcar_x3_01',
        description='Robot ID for the system'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for all nodes'
    )
    
    # Get launch configurations
    robot_id = LaunchConfiguration('robot_id')
    log_level = LaunchConfiguration('log_level')
    
    # Package path
    package_share = FindPackageShare('robot_vila_system').find('robot_vila_system')
    config_dir = os.path.join(package_share, 'config')
    
    # Core system nodes only
    vila_server_node = Node(
        package='robot_vila_system',
        executable='vila_server',
        name='vila_server',
        namespace='robot_system',
        parameters=[
            {'robot_id': robot_id},
            os.path.join(config_dir, 'vila_server_config.yaml')
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen',
        emulate_tty=True,
    )
    
    vila_vision_node = Node(
        package='robot_vila_system',
        executable='vila_vision',
        name='vila_vision',
        namespace='robot_system',
        parameters=[
            {'robot_id': robot_id},
            os.path.join(config_dir, 'vila_vision_config.yaml')
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        # Launch arguments
        robot_id_arg,
        log_level_arg,
        
        # System info
        LogInfo(msg='🚀 Launching Minimal ROS2 VILA Robot System'),
        LogInfo(msg=['Robot ID: ', robot_id]),
        LogInfo(msg='📡 Core server and vision nodes only'),
        
        # Core nodes
        vila_server_node,
        vila_vision_node,
        
        LogInfo(msg='✅ Minimal VILA Robot System launch complete'),
    ])
