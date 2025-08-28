#!/usr/bin/env python3
"""
ROS2 Launch file for the Cosmos-Transfer1 client system
Launches GUI and Cosmos-Transfer1 nodes for robot control and analysis
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    """Generate launch description for Cosmos-Transfer1 client system with GUI"""
    
    # Launch arguments
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='yahboomcar_x3_01',
        description='Robot ID to connect to'
    )
    
    gui_enabled_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Enable GUI interface'
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
    gui_enabled = LaunchConfiguration('gui')

    # Cosmos-Transfer1 configurations
    cosmos_model_dir = LaunchConfiguration('cosmos_model_dir')
    cosmos_node_name = LaunchConfiguration('cosmos_node_name')
    cosmos_service_name = LaunchConfiguration('cosmos_service_name')
    cosmos_enabled = LaunchConfiguration('cosmos_enabled')
    
    # Cosmos-Transfer1 only - no VILA server needed
    
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
        gui_enabled_arg,
        cosmos_model_dir_arg,
        cosmos_node_name_arg,
        cosmos_service_name_arg,
        cosmos_enabled_arg,

        # Status messages
        LogInfo(msg='🚀 Starting Cosmos-Transfer1 Client System...'),
        LogInfo(msg=['Target Robot ID: ', robot_id]),
        LogInfo(msg=['GUI Enabled: ', gui_enabled]),
        LogInfo(msg=['Cosmos-Transfer1 Enabled: ', cosmos_enabled]),
        LogInfo(msg=['Cosmos-Transfer1 Model Dir: ', cosmos_model_dir]),
        LogInfo(msg='📝 Cosmos-Transfer1 client system with GUI'),

        # Nodes
        ExecuteProcess(
            cmd=['echo', '🔄 Starting Cosmos-Transfer1 system...'],
            output='screen',
            condition=IfCondition(cosmos_enabled)
        ),
        robot_gui_node,
        cosmos_transfer1_node,
    ])
