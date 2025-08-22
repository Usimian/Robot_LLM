#!/usr/bin/env python3
"""
ROS2 Launch file for the client system
Launches VILA-related nodes that must run on the client PC where VILA model is available
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration, FindExecutable, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    """Generate launch description for client system - VILA nodes only"""
    
    # Launch arguments
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='yahboomcar_x3_01',
        description='Robot ID to connect to'
    )
    
    vila_enabled_arg = DeclareLaunchArgument(
        'vila',
        default_value='true',
        description='Enable VILA vision processing'
    )
    
    gui_enabled_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Enable GUI interface'
    )
    
    vila_server_enabled_arg = DeclareLaunchArgument(
        'vila_server',
        default_value='false',  # Disable old HTTP server - using integrated Cosmos Nemotron VLA
        description='Enable VILA server auto-start (deprecated - using integrated Cosmos Nemotron VLA)'
    )
    
    # Get launch configurations
    robot_id = LaunchConfiguration('robot_id')
    vila_enabled = LaunchConfiguration('vila')
    gui_enabled = LaunchConfiguration('gui')
    vila_server_enabled = LaunchConfiguration('vila_server')
    
    # Get workspace root path for VILA server script
    # Use absolute path to the workspace root where simple_vila_server.py is located
    vila_server_script = "/home/marc/Robot_LLM/simple_vila_server.py"
    
    # VILA Server Process (HTTP server for VILA model)
    vila_server_process = ExecuteProcess(
        cmd=[
            FindExecutable(name='python3'),
            vila_server_script,
            '--port', '8000'
        ],
        name='vila_server_process',
        output='screen',
        condition=IfCondition(vila_server_enabled)
    )
    
    # VILA Server Node (runs on client - has direct access to VILA model)
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
    
    # Robot GUI Node (runs on client - displays the control interface)
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
    
    return LaunchDescription([
        robot_id_arg,
        vila_enabled_arg,
        gui_enabled_arg,
        vila_server_enabled_arg,
        
        LogInfo(msg='🚀 Starting Client System...'),
        LogInfo(msg=['Target Robot ID: ', robot_id]),
        LogInfo(msg=['VILA Enabled: ', vila_enabled]),
        LogInfo(msg=['GUI Enabled: ', gui_enabled]),
        LogInfo(msg=['VILA Server Enabled: ', vila_server_enabled]),
        LogInfo(msg=['VILA Server Script: ', vila_server_script]),
        LogInfo(msg='📝 Complete client system with GUI and integrated Cosmos Nemotron VLA'),
        
        vila_server_process,
        vila_server_node,
        robot_gui_node,
    ])
