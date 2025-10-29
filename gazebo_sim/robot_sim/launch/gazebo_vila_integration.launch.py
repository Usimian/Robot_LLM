#!/usr/bin/env python3
"""
Integrated Gazebo + robot_vila_system Launch File

This launch file starts:
1. Gazebo simulation with mecanum robot
2. ROS-Gazebo bridge for sensor/command topics (including cmd_vel)
3. robot_vila_system GUI (publishes standard cmd_vel messages)

Topic Mapping:
Gazebo -> robot_vila_system:
  /camera/image -> /realsense/camera/color/image_raw
  /scan -> /scan (direct)
  /imu -> /imu/data_raw
  
robot_vila_system -> Gazebo:
  /cmd_vel -> /model/vehicle_blue/cmd_vel (standard ROS2 Twist messages)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """Generate integrated launch description"""
    
    # Get package directories
    robot_sim_dir = get_package_share_directory('robot_sim')
    
    # Paths
    world_file = os.path.join(robot_sim_dir, 'worlds', 'mecanum_working.sdf')
    bridge_config = os.path.join(robot_sim_dir, 'config', 'bridge.yaml')
    
    # Environment variables for NVIDIA GPU
    nvidia_env_vars = [
        SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
        SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
        SetEnvironmentVariable('__VK_LAYER_NV_optimus', 'NVIDIA_only'),
    ]
    
    # 1. Launch Gazebo with world
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', world_file, '-r'],
        output='screen',
        shell=False
    )
    
    # 2. ROS-Gazebo bridge (using YAML config with remappings)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_config}],
        output='screen',
        remappings=[
            ('/camera/image', '/realsense/camera/color/image_raw'),
            ('/imu', '/imu/data_raw'),
        ]
    )
    
    # 3. GUI Node - publishes standard cmd_vel messages
    robot_gui = Node(
        package='robot_vila_system',
        executable='robot_gui_node.py',
        name='robot_gui',
        namespace='client',
        output='screen',
        emulate_tty=True
    )
    
    # 4. VLM Navigation Node (RoboMP2 + Qwen2.5-VL)
    vlm_node = Node(
        package='robot_vila_system',
        executable='local_vlm_navigation_node.py',
        name='robomp2_navigation_node',
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        *nvidia_env_vars,
        
        # Status messages
        ExecuteProcess(cmd=['echo', '🚀 Launching Gazebo + robot_vila_system integration'], output='screen'),
        ExecuteProcess(cmd=['echo', '   └── Gazebo simulation: mecanum_working.sdf'], output='screen'),
        ExecuteProcess(cmd=['echo', '   └── ROS-Gazebo bridge with topic remapping'], output='screen'),
        ExecuteProcess(cmd=['echo', '   └── GUI publishes standard cmd_vel messages'], output='screen'),
        ExecuteProcess(cmd=['echo', ''], output='screen'),
        
        # Launch nodes
        gazebo,
        bridge,
        robot_gui,
        vlm_node,  # RoboMP2 VLM Navigation
    ])

