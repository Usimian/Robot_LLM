#!/usr/bin/env python3
"""
ROS2 Launch file for the client system
Launches the complete robot client system including GUI and VILA processing
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    """Generate launch description for client system"""
    
    # Launch arguments
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='robot_001',
        description='Robot ID to connect to'
    )
    
    gui_enabled_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Enable GUI interface'
    )
    
    vila_enabled_arg = DeclareLaunchArgument(
        'vila',
        default_value='true',
        description='Enable VILA vision processing'
    )
    
    # Get launch configurations
    robot_id = LaunchConfiguration('robot_id')
    gui_enabled = LaunchConfiguration('gui')
    vila_enabled = LaunchConfiguration('vila')
    
    # Robot GUI Node
    robot_gui_node = Node(
        package='robot_vila_system',
        executable='robot_gui_node.py',
        name='robot_gui',
        namespace='client',
        parameters=[{
            'robot_id': robot_id,
        }],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(gui_enabled)
    )
    
    # VILA Server Node
    vila_server_node = Node(
        package='robot_vila_system',
        executable='vila_server_node.py',
        name='vila_server',
        namespace='client',
        parameters=[{
            'robot_id': robot_id,
        }],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(vila_enabled)
    )
    
    # VILA Vision Node
    vila_vision_node = Node(
        package='robot_vila_system',
        executable='vila_vision_node.py',
        name='vila_vision',
        namespace='client',
        parameters=[{
            'robot_id': robot_id,
        }],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(vila_enabled)
    )
    
    # Gateway Validator Node
    gateway_validator_node = Node(
        package='robot_vila_system',
        executable='gateway_validator_node.py',
        name='gateway_validator',
        namespace='client',
        parameters=[{
            'robot_id': robot_id,
        }],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        robot_id_arg,
        gui_enabled_arg,
        vila_enabled_arg,
        
        LogInfo(msg='🚀 Starting Robot Client System...'),
        LogInfo(msg=['Target Robot ID: ', robot_id]),
        LogInfo(msg=['GUI Enabled: ', gui_enabled]),
        LogInfo(msg=['VILA Enabled: ', vila_enabled]),
        
        gateway_validator_node,
        robot_gui_node,
        vila_server_node,
        vila_vision_node,
    ])
