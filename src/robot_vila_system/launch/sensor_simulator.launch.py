#!/usr/bin/env python3
"""
ROS2 Launch file for the sensor simulator
Launches a standalone sensor simulator for testing when robot is not available
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """Generate launch description for sensor simulator"""
    
    # Launch arguments
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='simulator_robot_001',
        description='Robot ID for the sensor simulator'
    )
    
    battery_level_arg = DeclareLaunchArgument(
        'initial_battery',
        default_value='85.0',
        description='Initial battery level percentage'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'sensor_rate',
        default_value='5.0',
        description='Sensor data publish rate in Hz'
    )
    
    camera_enabled_arg = DeclareLaunchArgument(
        'camera',
        default_value='true',
        description='Enable camera image simulation'
    )
    
    # Get launch configurations
    robot_id = LaunchConfiguration('robot_id')
    initial_battery = LaunchConfiguration('initial_battery')
    sensor_rate = LaunchConfiguration('sensor_rate')
    camera_enabled = LaunchConfiguration('camera')
    
    # Sensor simulator node
    sensor_simulator_node = Node(
        package='robot_vila_system',
        executable='sensor_simulator_node.py',
        name='sensor_simulator',
        namespace='simulator',
        parameters=[{
            'robot_id': robot_id,
            'initial_battery_level': initial_battery,
            'sensor_publish_rate': sensor_rate,
            'camera_enabled': camera_enabled,
            'use_sim_time': False,
        }],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        robot_id_arg,
        battery_level_arg,
        publish_rate_arg,
        camera_enabled_arg,
        
        LogInfo(msg='🤖 Starting Sensor Simulator for Testing...'),
        LogInfo(msg=['Simulated Robot ID: ', robot_id]),
        LogInfo(msg=['Initial Battery: ', initial_battery, '%']),
        LogInfo(msg=['Sensor Rate: ', sensor_rate, ' Hz']),
        LogInfo(msg=['Camera Enabled: ', camera_enabled]),
        LogInfo(msg='📝 Use this when the physical robot is not available'),
        
        sensor_simulator_node,
    ])