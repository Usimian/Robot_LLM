#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    pkg_robot_sim = get_package_share_directory('robot_sim')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Paths
    world_file = os.path.join(pkg_robot_sim, 'worlds', 'mecanum_working.sdf')
    bridge_config = os.path.join(pkg_robot_sim, 'config', 'bridge.yaml')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Force NVIDIA GPU for rendering
    nvidia_env = SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1')
    nvidia_driver = SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')
    vk_layer = SetEnvironmentVariable('__VK_LAYER_NV_optimus', 'NVIDIA_only')
    
    # Gazebo Sim with the working world
    # -r flag starts the simulation running (not paused)
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r {world_file}'
        }.items()
    )

    # ROS-Gazebo Bridge using config file
    # Maps ROS /cmd_vel -> Gazebo /model/vehicle_blue/cmd_vel
    bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'config_file': bridge_config,
        }]
    )

    return LaunchDescription([
        nvidia_env,
        nvidia_driver,
        vk_layer,
        gz_sim,
        bridge_cmd,
    ])

