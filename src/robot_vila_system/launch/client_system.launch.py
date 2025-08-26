#!/usr/bin/env python3
"""
ROS2 Launch file for the complete client system
Launches all client-side nodes including VILA, GUI, and Cosmos-Transfer1 video generation
This provides a complete integrated system for robot control and video generation
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, FindExecutable, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """Generate launch description for complete client system with VILA, GUI, and Cosmos-Transfer1"""
    
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

    # Cosmos-Transfer1 arguments
    cosmos_model_dir_arg = DeclareLaunchArgument(
        'cosmos_model_dir',
        default_value='/home/marc/Robot_LLM/models/Cosmos-Transfer1-7B',
        description='Directory containing Cosmos-Transfer1 model files'
    )

    cosmos_node_name_arg = DeclareLaunchArgument(
        'cosmos_node_name',
        default_value='cosmos_transfer1_node',
        description='Name of the Cosmos-Transfer1 ROS2 node'
    )

    cosmos_service_name_arg = DeclareLaunchArgument(
        'cosmos_service_name',
        default_value='/cosmos_transfer1/execute',
        description='Name of the Cosmos-Transfer1 ROS2 service'
    )

    cosmos_enabled_arg = DeclareLaunchArgument(
        'cosmos_enabled',
        default_value='false',
        description='Enable Cosmos-Transfer1 video generation service (node implementation pending)'
    )
    
    # Get launch configurations
    robot_id = LaunchConfiguration('robot_id')
    vila_enabled = LaunchConfiguration('vila')
    gui_enabled = LaunchConfiguration('gui')
    vila_server_enabled = LaunchConfiguration('vila_server')

    # Cosmos-Transfer1 configurations
    cosmos_model_dir = LaunchConfiguration('cosmos_model_dir')
    cosmos_node_name = LaunchConfiguration('cosmos_node_name')
    cosmos_service_name = LaunchConfiguration('cosmos_service_name')
    cosmos_enabled = LaunchConfiguration('cosmos_enabled')
    
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

    # Cosmos-Transfer1 Node (provides video generation services)
    cosmos_transfer1_node = Node(
        package='robot_vila_system',
        executable='cosmos_transfer1_node.py',
        name=cosmos_node_name,
        output='screen',
        emulate_tty=True,
        parameters=[{
            'model_dir': cosmos_model_dir,
        }],
        condition=IfCondition(cosmos_enabled)
    )
    
    return LaunchDescription([
        # Launch arguments
        robot_id_arg,
        vila_enabled_arg,
        gui_enabled_arg,
        vila_server_enabled_arg,
        cosmos_model_dir_arg,
        cosmos_node_name_arg,
        cosmos_service_name_arg,
        cosmos_enabled_arg,

        # Status messages
        LogInfo(msg='🚀 Starting Complete Client System...'),
        LogInfo(msg=['Target Robot ID: ', robot_id]),
        LogInfo(msg=['VILA Enabled: ', vila_enabled]),
        LogInfo(msg=['GUI Enabled: ', gui_enabled]),
        LogInfo(msg=['VILA Server Enabled: ', vila_server_enabled]),
        LogInfo(msg=['VILA Server Script: ', vila_server_script]),
        LogInfo(msg=['Cosmos-Transfer1 Enabled: ', cosmos_enabled]),
        LogInfo(msg=['Cosmos-Transfer1 Model Dir: ', cosmos_model_dir]),
        LogInfo(msg='📝 Complete client system with GUI, VILA, and Cosmos-Transfer1 video generation'),

        # Nodes and processes
        ExecuteProcess(
            cmd=['echo', '🔄 Starting Cosmos-Transfer1 service...'],
            output='screen',
            condition=IfCondition(cosmos_enabled)
        ),
        vila_server_process,
        vila_server_node,
        robot_gui_node,
        cosmos_transfer1_node,
    ])
