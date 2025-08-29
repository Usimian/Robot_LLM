#!/usr/bin/env python3
"""
ROS2 Launch file for the Local VLM Navigation client system
Launches GUI and Local VLM Navigation nodes for robot control and analysis
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    """Generate launch description for Local VLM Navigation client system with GUI"""
    
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

    # Local VLM Navigation arguments
    vlm_model_name_arg = DeclareLaunchArgument(
        'vlm_model_name',
        default_value='Qwen/Qwen2-VL-7B-Instruct',
        description='HuggingFace model name for local VLM'
    )

    vlm_node_name_arg = DeclareLaunchArgument(
        'vlm_node_name',
        default_value='local_vlm_navigation_node',
        description='Name of the Local VLM Navigation ROS2 node'
    )

    vlm_service_name_arg = DeclareLaunchArgument(
        'vlm_service_name',
        default_value='/vlm/analyze_scene',
        description='Name of the Local VLM Navigation ROS2 service'
    )

    vlm_enabled_arg = DeclareLaunchArgument(
        'vlm_enabled',
        default_value='true',
        description='Enable Local VLM Navigation service'
    )
    
    # Get launch configurations
    robot_id = LaunchConfiguration('robot_id')
    gui_enabled = LaunchConfiguration('gui')

    # Local VLM Navigation configurations
    vlm_model_name = LaunchConfiguration('vlm_model_name')
    vlm_node_name = LaunchConfiguration('vlm_node_name')
    vlm_service_name = LaunchConfiguration('vlm_service_name')
    vlm_enabled = LaunchConfiguration('vlm_enabled')
    
    # Local VLM Navigation only - no external services needed
    
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

    # Local VLM Navigation Node (provides local AI navigation services)
    local_vlm_navigation_node = Node(
        package='robot_vila_system',
        executable='local_vlm_navigation_node.py',
        name=vlm_node_name,
        output='screen',
        emulate_tty=True,
        parameters=[{
            'model_name': vlm_model_name,
        }],
        condition=IfCondition(vlm_enabled)
    )
    
    return LaunchDescription([
        # Launch arguments
        robot_id_arg,
        gui_enabled_arg,
        vlm_model_name_arg,
        vlm_node_name_arg,
        vlm_service_name_arg,
        vlm_enabled_arg,

        # Status messages
        LogInfo(msg='🚀 Starting Local VLM Navigation Client System...'),
        LogInfo(msg=['Target Robot ID: ', robot_id]),
        LogInfo(msg=['GUI Enabled: ', gui_enabled]),
        LogInfo(msg=['Local VLM Enabled: ', vlm_enabled]),
        LogInfo(msg=['VLM Model: ', vlm_model_name]),
        LogInfo(msg='📝 Local VLM Navigation client system with GUI'),

        # Nodes
        ExecuteProcess(
            cmd=['echo', '🔄 Starting Local VLM Navigation system...'],
            output='screen',
            condition=IfCondition(vlm_enabled)
        ),
        robot_gui_node,
        local_vlm_navigation_node,
    ])
