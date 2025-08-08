# Deprecated HTTP Files

This directory contains the original HTTP-based implementation files
that have been replaced by ROS2 versions.

## Migration Mapping
- robot_vila_server.py → robot_vila_server_ros2.py
- vila_ros_node.py (ROS1) → vila_ros2_node.py (ROS2)
- robot_gui.py → robot_gui_ros2.py
- unified_robot_controller.py → (functionality merged into ROS2 server)
- unified_robot_client.py → robot_client_ros2.py
- robot_client_examples.py → (examples updated for ROS2)
- start_unified_system.sh → launch_ros2_system.py

These files are kept for reference but should not be used in the ROS2 system.
