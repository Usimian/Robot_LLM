#!/usr/bin/env python3
"""
ROS2 Launch file for complete VILA Robot System
Proper ROS2 launch file following best practices
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    """Generate launch description for complete VILA robot system"""
    
    # Launch arguments
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='yahboomcar_x3_01',
        description='Robot ID for the system'
    )
    
    launch_gui_arg = DeclareLaunchArgument(
        'launch_gui',
        default_value='true',
        description='Whether to launch the GUI'
    )
    
    launch_robot_client_arg = DeclareLaunchArgument(
        'launch_robot_client',
        default_value='false',
        description='Whether to launch robot client (for testing on same machine)'
    )
    
    enable_validation_arg = DeclareLaunchArgument(
        'enable_validation',
        default_value='true',
        description='Whether to enable gateway validation monitoring'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for all nodes'
    )
    
    # Get launch configurations
    robot_id = LaunchConfiguration('robot_id')
    launch_gui = LaunchConfiguration('launch_gui')
    launch_robot_client = LaunchConfiguration('launch_robot_client')
    enable_validation = LaunchConfiguration('enable_validation')
    log_level = LaunchConfiguration('log_level')
    
    # Package path
    package_share = FindPackageShare('robot_vila_system').find('robot_vila_system')
    config_dir = os.path.join(package_share, 'config')
    
    # Core system nodes
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
    
    # Optional GUI node
    gui_group = GroupAction(
        condition=IfCondition(launch_gui),
        actions=[
            LogInfo(msg='Launching Robot GUI...'),
            Node(
                package='robot_vila_system',
                executable='robot_gui',
                name='robot_gui',
                namespace='robot_system',
                parameters=[
                    {'robot_id': robot_id},
                    os.path.join(config_dir, 'robot_gui_config.yaml')
                ],
                arguments=['--ros-args', '--log-level', log_level],
                output='screen',
                emulate_tty=True,
            )
        ]
    )
    
    # Optional robot client (for testing)
    robot_client_group = GroupAction(
        condition=IfCondition(launch_robot_client),
        actions=[
            LogInfo(msg='Launching Robot Client (testing mode)...'),
            Node(
                package='robot_vila_system',
                executable='robot_client',
                name='robot_client',
                namespace='robot_system',
                parameters=[
                    {'robot_id': robot_id},
                    os.path.join(config_dir, 'robot_client_config.yaml')
                ],
                arguments=['--ros-args', '--log-level', log_level],
                output='screen',
                emulate_tty=True,
            )
        ]
    )
    
    # Optional gateway validator
    validator_group = GroupAction(
        condition=IfCondition(enable_validation),
        actions=[
            LogInfo(msg='Launching Gateway Validator...'),
            Node(
                package='robot_vila_system',
                executable='gateway_validator',
                name='gateway_validator',
                namespace='robot_system',
                parameters=[
                    {'robot_id': robot_id},
                    {'monitoring_enabled': True}
                ],
                arguments=['--ros-args', '--log-level', log_level],
                output='screen',
                emulate_tty=True,
            )
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        robot_id_arg,
        launch_gui_arg,
        launch_robot_client_arg,
        enable_validation_arg,
        log_level_arg,
        
        # System info
        LogInfo(msg='🚀 Launching ROS2 VILA Robot System'),
        LogInfo(msg=['Robot ID: ', robot_id]),
        LogInfo(msg='📡 All communication via ROS2 topics and services'),
        LogInfo(msg='🛡️ Single command gateway maintained'),
        
        # Core nodes (always launched)
        vila_server_node,
        vila_vision_node,
        
        # Optional nodes
        gui_group,
        robot_client_group,
        validator_group,
        
        LogInfo(msg='✅ VILA Robot System launch complete'),
    ])
