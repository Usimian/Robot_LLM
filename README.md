# Robot LLM System

A ROS2-based robot control system featuring vision-language model (VLM) integration with RoboMP2 navigation capabilities. This system enables natural language robot control through a Tkinter GUI interface, leveraging Qwen2.5-VL for advanced scene understanding and navigation.

## 🏗️ System Architecture

### Client/Server Design
- **Client (This Workspace)**: AMD64 Ubuntu 22.04 PC
  - Python/Tkinter GUI interface
  - ROS2 node infrastructure
  - Local VLM (Qwen2.5-VL) for scene analysis
  - RoboMP2 navigation policies
  
- **Robot Server**: Jetson Orin Nano (192.168.1.166)
  - Hardware interface nodes
  - Gateway validator
  - Sensor data collection and publishing

### Hardware Components
- RGB camera
- Depth camera
- IMU (Inertial Measurement Unit)
- Lidar
- Mecanum wheel odometry

## 📋 Prerequisites

### System Requirements
- Ubuntu 22.04 (AMD64)
- ROS2 (Humble or later)
- Python 3.8+
- CUDA-capable GPU (recommended for VLM inference)
- Network connection to robot (192.168.1.166)

### ROS2 Installation
Ensure ROS2 is installed and sourced:
```bash
# Source ROS2 environment
source /opt/ros/humble/setup.bash
```

## 🚀 Installation

### 1. Clone Repository
```bash
cd ~
git clone <repository-url> Robot_LLM
cd Robot_LLM
```

### 2. Install Python Dependencies
```bash
pip3 install -r requirements_ros2.txt
```

**Key Dependencies:**
- `rclpy` - ROS2 Python client library
- `torch` & `torchvision` - PyTorch for deep learning
- `transformers` - Hugging Face transformers (Qwen2.5-VL)
- `qwen-vl-utils` - Qwen VLM utilities
- `opencv-python` - Computer vision
- `Pillow` - Image processing

### 3. Build ROS2 Packages
```bash
# Build without symlink-install (required by project standards)
colcon build

# Source the workspace
source install/setup.bash
```

## 🎯 Usage

### Launch Client System
Start the client system with GUI and RoboMP2 navigation:
```bash
source install/setup.bash
ros2 launch robot_vila_system client_system.launch.py
```

### Launch Options
```bash
# Launch with custom VLM model
ros2 launch robot_vila_system client_system.launch.py vlm_model_name:="Qwen/Qwen2.5-VL-7B-Instruct"

# Launch without GUI
ros2 launch robot_vila_system client_system.launch.py gui:=false

# Launch without VLM service
ros2 launch robot_vila_system client_system.launch.py vlm_enabled:=false
```

## 📁 Project Structure

```
Robot_LLM/
├── src/
│   ├── robot_msgs/              # Custom ROS2 message definitions
│   │   ├── msg/
│   │   │   ├── RobotCommand.msg # Movement commands
│   │   │   └── SensorData.msg   # Sensor telemetry
│   │   └── srv/
│   │       └── ExecuteCommand.srv
│   │
│   └── robot_vila_system/       # Main system package
│       ├── launch/
│       │   └── client_system.launch.py
│       └── robot_vila_system/
│           ├── gateway_validator_node.py    # Communication gateway
│           ├── robot_gui_node.py            # Tkinter GUI
│           ├── local_vlm_navigation_node.py # RoboMP2 VLM node
│           ├── gui_components.py            # GUI widgets
│           ├── gui_config.py                # GUI configuration
│           ├── gui_utils.py                 # GUI utilities
│           └── robomp2_components.py        # RoboMP2 integration
│
├── hf_cache/                    # Hugging Face model cache
├── robomp2_policies.pkl         # RoboMP2 navigation policies
└── requirements_ros2.txt        # Python dependencies
```

## 📡 Communication Architecture

### ROS2-Only Messaging
All communications use ROS2 topics and services. HTTP and WebSockets are **strictly prohibited**.

### Gateway Pattern
All robot communications flow through a **single gateway** (`gateway_validator_node.py`):
- Validates all commands before execution
- Provides centralized logging and monitoring
- Ensures command safety and consistency

### Key Topics & Services
- `/cmd_vel` - Velocity commands (never published to directly)
- `/scan` - Lidar data access
- `/vlm/analyze_scene` - VLM scene analysis service
- Custom message types defined in `robot_msgs` package

## 🛠️ Development Guidelines

### Code Standards
- **Logging**: Use ROS2 `get_logger()` exclusively
  - ❌ No `print()` statements
  - ❌ No Python `logging` module
  
- **Data**: No simulated, mock, or fallback data
  - System must fail with clear errors if data unavailable
  
- **Build**: Always build packages without `--symlink-install`
  ```bash
  colcon build  # Correct
  # colcon build --symlink-install  # ❌ Incorrect
  ```

- **Testing**: Run tests only when explicitly requested
  ```bash
  # Don't run automatically
  colcon test --packages-select robot_vila_system
  ```

### Communication Rules
1. **Never** publish directly to `/cmd_vel` topic
2. **Always** route commands through the gateway
3. **Never** modify `SensorData` message definitions
4. Access lidar values via `/scan` topic only

### Process Management
- Prefer manual process termination
- Avoid automatic `pkill` commands
- Ask before terminating any processes

## 🔧 Troubleshooting

### Build Issues
```bash
# Clean and rebuild
rm -rf build/ install/ log/
colcon build
source install/setup.bash
```

### Model Loading Issues
Check Hugging Face cache and model availability:
```bash
# Verify model cache
ls -lh hf_cache/

# Check GPU availability
nvidia-smi
```

### ROS2 Connection Issues
```bash
# Check ROS2 environment
echo $ROS_DOMAIN_ID

# Verify node discovery
ros2 node list

# Check topic communication
ros2 topic list
ros2 topic echo /sensor_data
```

### Robot Connection
Verify network connectivity to Jetson Orin Nano:
```bash
ping 192.168.1.166
ssh <user>@192.168.1.166
```

## 📝 Version Information
- **Package Version**: 1.0.0
- **ROS2 Distribution**: Humble (or compatible)
- **License**: MIT

## 🤝 Contributing
When contributing to this project:
1. Follow all development guidelines above
2. Use ROS2 packages under `src/` directory
3. Execute code via ROS2 launch files
4. Ask for clarification when requirements are ambiguous
5. Always check terminal output for errors

## 📧 Contact
Maintainer: Robot Developer (robot@example.com)

---

**Last Updated**: October 26, 2025

