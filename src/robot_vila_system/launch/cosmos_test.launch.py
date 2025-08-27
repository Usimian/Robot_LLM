#!/usr/bin/env python3

"""
ROS2 Launch file for testing Cosmos-Transfer1 model only
No GUI, no cv_bridge dependencies - just the Cosmos model server
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Cosmos model testing"""
    
    # Launch arguments
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='yahboomcar_x3_01',
        description='Target robot ID'
    )
    
    model_dir_arg = DeclareLaunchArgument(
        'cosmos_model_dir',
        default_value='/home/marc/Robot_LLM/models/Cosmos-Transfer1-7B',
        description='Path to Cosmos-Transfer1 model directory'
    )
    
    # Launch info
    launch_info = LogInfo(
        msg=[
            '🚀 Testing Cosmos-Transfer1 Model Only...\n',
            'Target Robot ID: ', LaunchConfiguration('robot_id'), '\n',
            'Cosmos Model Dir: ', LaunchConfiguration('cosmos_model_dir'), '\n',
            '📝 Pure Cosmos model testing - no GUI dependencies'
        ]
    )
    
    # VILA Server Node (contains Cosmos model)
    vila_server_node = Node(
        package='robot_vila_system',
        executable='vila_server_node.py',
        name='vila_server',
        namespace='client',
        parameters=[{
            'robot_id': LaunchConfiguration('robot_id'),
            'cosmos_model_dir': LaunchConfiguration('cosmos_model_dir'),
        }],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        robot_id_arg,
        model_dir_arg,
        launch_info,
        vila_server_node,
    ])
