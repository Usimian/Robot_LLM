#!/usr/bin/env python3
"""
ROS2 Launch file for the Local VLM Navigation client system
Launches GUI and Local VLM Navigation nodes for robot control and analysis
Optional: Can launch Gazebo simulation
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition

# Import configuration
from robot_vila_system.gui_config import GUIConfig

def generate_launch_description():
    """Generate launch description for Local VLM Navigation client system with GUI"""
    
    # Launch arguments - simplified for single robot setup
    
    gui_enabled_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Enable GUI interface'
    )

    # Local VLM Navigation arguments
    vlm_model_name_arg = DeclareLaunchArgument(
        'vlm_model_name',
        default_value=GUIConfig.DEFAULT_VLM_MODEL,
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

    # Gazebo simulation arguments
    gazebo_arg = DeclareLaunchArgument(
        'gazebo',
        default_value='false',
        description='Launch Gazebo simulation'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='mecanum_working.sdf',
        description='World file to load (mecanum_working.sdf or small_house.world)'
    )

    # Get launch configurations
    gui_enabled = LaunchConfiguration('gui')
    gazebo_enabled = LaunchConfiguration('gazebo')
    world_name = LaunchConfiguration('world')

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
            # Single robot setup - no robot_id needed
        }],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(gui_enabled)
    )


    # RoboMP2 Navigation Node (provides RoboMP2-enhanced AI navigation services)
    robomp2_navigation_node = Node(
        package='robot_vila_system',
        executable='local_vlm_navigation_node.py',
        name='robomp2_navigation_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'model_name': vlm_model_name,
        }],
        condition=IfCondition(vlm_enabled)
    )

    # Gazebo components (only if gazebo=true)
    # Get robot_sim package directory
    try:
        robot_sim_dir = get_package_share_directory('robot_sim')

        # World file path
        world_file = PathJoinSubstitution([
            robot_sim_dir,
            'worlds',
            world_name
        ])

        # Bridge config path
        bridge_config = os.path.join(robot_sim_dir, 'config', 'bridge.yaml')

        # Gazebo models path
        models_path = os.path.join(robot_sim_dir, 'models')

        # Environment variables for NVIDIA GPU and Gazebo models
        nvidia_env_vars = [
            SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
            SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
            SetEnvironmentVariable('__VK_LAYER_NV_optimus', 'NVIDIA_only'),
            SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', models_path),
        ]

        # Gazebo process
        gazebo_process = ExecuteProcess(
            cmd=['gz', 'sim', world_file, '-r'],
            output='screen',
            shell=False,
            condition=IfCondition(gazebo_enabled)
        )

        # ROS-Gazebo bridge
        bridge_node = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            parameters=[{'config_file': bridge_config}],
            output='screen',
            remappings=[
                ('/camera/image', '/realsense/camera/color/image_raw'),
                ('/imu', '/imu/data_raw'),
            ],
            condition=IfCondition(gazebo_enabled)
        )

        gazebo_available = True
    except Exception as e:
        # robot_sim package not available
        nvidia_env_vars = []
        gazebo_process = None
        bridge_node = None
        gazebo_available = False

    # Build launch description
    launch_list = [
        # Launch arguments
        gui_enabled_arg,
        vlm_model_name_arg,
        vlm_node_name_arg,
        vlm_service_name_arg,
        vlm_enabled_arg,
        gazebo_arg,
        world_arg,
    ]

    # Add environment variables if Gazebo is available
    if gazebo_available:
        launch_list.extend(nvidia_env_vars)

    # Status messages
    launch_list.extend([
        LogInfo(msg='🚀 Starting Local VLM Navigation Client System...'),
        LogInfo(msg='Target: Single Robot Configuration'),
        LogInfo(msg=['GUI Enabled: ', gui_enabled]),
        LogInfo(msg=['Gazebo Enabled: ', gazebo_enabled]),
        LogInfo(msg=['Local VLM Enabled: ', vlm_enabled]),
        LogInfo(msg=['VLM Model: ', vlm_model_name]),
        LogInfo(msg='📝 Local VLM Navigation client system with optional Gazebo'),
        ExecuteProcess(
            cmd=['echo', '🔄 Starting RoboMP2 Navigation system with NLP parser...'],
            output='screen',
            condition=IfCondition(vlm_enabled)
        ),
    ])

    # Add Gazebo components if available
    if gazebo_available:
        launch_list.extend([
            ExecuteProcess(
                cmd=['echo', '🎮 Launching Gazebo simulation...'],
                output='screen',
                condition=IfCondition(gazebo_enabled)
            ),
            gazebo_process,
            bridge_node,
        ])

    # Add core nodes
    launch_list.extend([
        robot_gui_node,
        robomp2_navigation_node,
    ])

    return LaunchDescription(launch_list)
