#!/usr/bin/env python3
"""
ROS2 Launch file for the robot system
Launches nodes that should run on the robot (Jetson Orin Nano)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    """Generate launch description for robot system"""
    
    # Launch arguments
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='yahboomcar_x3_01',
        description='Robot ID'
    )
    
    gui_enabled_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Enable GUI interface'
    )
    
    validator_enabled_arg = DeclareLaunchArgument(
        'validator',
        default_value='true',
        description='Enable gateway validator'
    )
    
    # Get launch configurations
    robot_id = LaunchConfiguration('robot_id')
    gui_enabled = LaunchConfiguration('gui')
    validator_enabled = LaunchConfiguration('validator')
    
    # Robot GUI Node (runs on robot but displays on client via X11/VNC)
    robot_gui_node = Node(
        package='robot_vila_system',
        executable='robot_gui_node.py',
        name='robot_gui',
        namespace='robot',
        parameters=[{
            'robot_id': robot_id,
        }],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(gui_enabled)
    )
    
    # Robot Client Node (hardware interface - must run on robot)
    robot_client_node = Node(
        package='robot_vila_system',
        executable='robot_client_node.py',
        name='robot_client',
        namespace='robot',
        parameters=[{
            'robot_id': robot_id,
        }],
        output='screen',
        emulate_tty=True
    )
    
    # Gateway Validator Node (monitors command compliance)
    gateway_validator_node = Node(
        package='robot_vila_system',
        executable='gateway_validator_node.py',
        name='gateway_validator',
        namespace='robot',
        parameters=[{
            'robot_id': robot_id,
        }],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(validator_enabled)
    )
    
    return LaunchDescription([
        robot_id_arg,
        gui_enabled_arg,
        validator_enabled_arg,
        
        LogInfo(msg='🤖 Starting Robot System...'),
        LogInfo(msg=['Robot ID: ', robot_id]),
        LogInfo(msg=['GUI Enabled: ', gui_enabled]),
        LogInfo(msg=['Validator Enabled: ', validator_enabled]),
        LogInfo(msg='📝 Note: VILA nodes should run on client PC'),
        
        robot_client_node,
        robot_gui_node,
        gateway_validator_node,
    ])
