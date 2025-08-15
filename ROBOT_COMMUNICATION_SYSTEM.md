# Robot Communication System Documentation

## Overview

This document describes the ROS2-based communication architecture for the Robot VILA System. The system implements a **single command gateway architecture** where all robot communications flow through centralized ROS2 topics and services, ensuring coordinated control and monitoring.

## System Architecture

### Core Principle: Single Command Gateway 🛡️
- **ALL robot commands** must go through the designated gateway
- **NO direct /cmd_vel publishing** is allowed
- **Centralized validation** ensures command integrity
- **ROS2 topics and services** replace HTTP/WebSocket communication

## ROS2 Communication Structure

### 1. Packages
```
robot_msgs/                    # Message and service definitions
├── msg/
│   ├── RobotCommand.msg      # Robot command structure
│   ├── SensorData.msg        # Sensor data and status structure
│   └── VILAAnalysis.msg      # VILA analysis results
└── srv/
    ├── ExecuteCommand.srv    # Command execution service
    └── RequestVILAAnalysis.srv # VILA analysis service

robot_vila_system/            # Main system nodes
├── vila_server_node.py       # VILA AI server
├── vila_vision_node.py       # Vision processing
├── robot_client_node.py      # Robot hardware interface
├── robot_gui_node.py         # User interface
└── gateway_validator_node.py # Command gateway validator
```

### 2. ROS2 Topics

#### Robot Data and Control
```bash
# Robot data and status (published by robot_client)
/robot/sensors                              # SensorData messages (includes status)

# Command flow (single gateway)
/robot/commands                             # RobotCommand messages
/robot/navigation_commands                  # Navigation command strings
```

#### Camera and Vision
```bash
# Camera feed
/robot/camera/image_raw                     # Raw camera images
/robot/camera/image_compressed              # Compressed images

# VILA analysis
/vila/analysis                              # VILAAnalysis results
/vila/status                               # VILA system status
```

#### System Monitoring
```bash
# Gateway validation (monitors compliance)
/system/gateway_violations                  # Command violations
/system/command_statistics                  # Command flow stats
```

### 3. ROS2 Services

#### Command Execution (Single Gateway)
```bash
# Primary command service (THE gateway)
/robot/execute_command                     # ExecuteCommand service
```

#### VILA Analysis
```bash
# VILA processing services
/vila/request_analysis                     # RequestVILAAnalysis service
/vila/get_model_status                    # Model status queries
```

## Message Definitions

### RobotCommand.msg
```yaml
# Robot command message (flows through gateway only)
string robot_id                    # Target robot identifier
string command_type                 # "move", "rotate", "stop", "custom"
float64 linear_x                   # Forward/backward velocity (m/s)
float64 linear_y                   # Left/right velocity (m/s)  
float64 angular_z                  # Rotational velocity (rad/s)
float64 duration                   # Command duration (seconds)
string parameters_json             # Additional parameters as JSON
int64 timestamp_ns                 # Command timestamp
string source_node                 # Originating node name
```

### SensorData.msg
```yaml
# Sensor data and robot status from robot
string robot_id                    # Robot identifier
float64 battery_voltage            # Battery voltage (V)
float64 cpu_temp                   # Jetson CPU Temperature (°C)
float64 distance_front             # Front distance sensor (m)
float64 distance_left              # Left distance sensor (m)
float64 distance_right             # Right distance sensor (m)
float64 cpu_usage                  # CPU usage (%)
string camera_status               # Camera status
int64 timestamp_ns                 # Sensor reading timestamp
```

### VILAAnalysis.msg
```yaml
# VILA analysis results
string robot_id                    # Target robot
string prompt                      # Analysis prompt used
string analysis_result             # Text analysis result
string navigation_commands_json    # Generated commands as JSON
float64 confidence                 # Analysis confidence (0-1)
bool success                       # Analysis success flag
string error_message               # Error details if failed
int64 timestamp_ns                 # Analysis timestamp
```



## Service Definitions

### ExecuteCommand.srv
```yaml
# Request
RobotCommand command               # Command to execute

---
# Response
bool success                       # Execution success
string result_message              # Result description
string execution_id                # Unique execution identifier
int64 estimated_duration_ns        # Estimated completion time
```

### RequestVILAAnalysis.srv
```yaml
# Request
string robot_id                    # Target robot
string prompt                      # Analysis prompt
bool include_navigation            # Generate navigation commands

---
# Response
VILAAnalysis analysis              # Complete analysis result
bool success                       # Request success
string error_message               # Error if failed
```

## Node Communication Patterns

### 1. Robot Client → System
```mermaid
graph LR
    A[Robot Hardware] --> B[robot_client_node]
    B --> C[/robot/sensors]
    B --> D[/robot/camera/image_raw]
```

### 2. Command Flow (Single Gateway)
```mermaid
graph TD
    A[GUI/External] --> B[/robot/execute_command]
    B --> C[vila_server_node]
    C --> D[Command Validation]
    D --> E[Robot Execution]
    F[gateway_validator] --> G[Monitor All Commands]
```

### 3. VILA Processing
```mermaid
graph LR
    A[Camera Feed] --> B[vila_vision_node]
    B --> C[VILA Model]
    C --> D[Analysis Results]
    D --> E[/vila/analysis]
    E --> F[Navigation Commands]
```

## System Nodes

### vila_server_node
- **Purpose**: Central VILA AI processing server
- **Subscribes**: Camera images, analysis requests
- **Publishes**: VILA analysis results, system status
- **Services**: Provides execute_command service (THE GATEWAY)
- **Key Function**: Processes all robot commands through single gateway

### vila_vision_node  
- **Purpose**: Computer vision processing with VILA
- **Subscribes**: Camera feed, sensor data
- **Publishes**: Vision analysis, navigation suggestions
- **Services**: Calls execute_command for movement
- **Key Function**: Converts vision analysis to robot commands

### robot_client_node
- **Purpose**: Hardware interface for robot
- **Subscribes**: Command execution results
- **Publishes**: Sensor data (includes robot status), camera feed
- **Services**: None (receives commands only)
- **Key Function**: Interfaces with physical robot hardware

### robot_gui_node
- **Purpose**: User interface and monitoring
- **Subscribes**: Robot sensor data (includes status), VILA results
- **Publishes**: User-initiated commands
- **Services**: Calls execute_command and request_analysis
- **Key Function**: Provides human operator interface

### gateway_validator_node
- **Purpose**: Ensures single gateway compliance
- **Subscribes**: All command topics, /cmd_vel (to detect violations)
- **Publishes**: Violation reports, statistics
- **Services**: None
- **Key Function**: Monitors and enforces gateway architecture

## Communication Flow Examples

### 1. Manual Robot Control
```
1. User clicks "Move Forward" in GUI
2. GUI calls /robot/execute_command service
3. vila_server validates and executes command
4. robot_client receives and executes movement
5. gateway_validator logs legitimate command
```

### 2. VILA-Guided Navigation
```
1. Camera publishes image to /robot/camera/image_raw
2. vila_vision processes image with VILA model
3. VILA generates navigation analysis
4. vila_vision calls /robot/execute_command with movement
5. Command flows through single gateway to robot_client
6. gateway_validator confirms proper command flow
```

### 3. System Monitoring
```
1. robot_client continuously publishes sensor data (includes robot status)
2. All nodes subscribe to relevant sensor topics
3. GUI displays real-time sensor data and robot status
4. gateway_validator monitors for command violations
5. Any direct /cmd_vel publishing triggers alerts
```

## Security and Validation

### Command Gateway Validation
- **Single Entry Point**: All commands via /robot/execute_command
- **Source Tracking**: Every command tagged with source_node
- **Violation Detection**: Direct /cmd_vel publishing blocked
- **Audit Trail**: All commands logged with timestamps
- **Rate Limiting**: Commands validated for safety

### Safety Features
- **Command Timeouts**: All commands have maximum duration
- **Emergency Stop**: Immediate stop commands prioritized
- **Sensor Integration**: Commands validated against sensor data
- **Error Recovery**: Automatic fallback for failed commands

## Configuration

### Robot Identification
- **Single Robot System**: `robot_id = "yahboomcar_x3_01"`
- **Fixed IP**: Robot at `192.168.1.166`
- **ROS2 Domain**: Default domain (configurable)

### Network Configuration
- **RMW**: `rmw_fastrtps_cpp`
- **Discovery**: FastDDS with **multicast discovery** (working)
- **Multicast Groups**: 
  - Default multicast group: `239.255.0.1:7400`
  - Automatic peer discovery across network segments
- **QoS**: Reliable for commands, best-effort for sensor data
- **Cross-Platform**: Jetson Orin Nano (ARM64) ↔ Ubuntu 22.04 PC (AMD64)

## Multicast Discovery Implementation ✅

### Status: **WORKING**
The ROS2 multicast discovery has been successfully implemented and tested between the Jetson Orin Nano robot and the Ubuntu 22.04 client PC.

### Key Benefits
- **Automatic Discovery**: Nodes automatically discover each other across the network
- **No Manual Configuration**: No need to specify peer IP addresses
- **Network Resilience**: Handles network changes and reconnections gracefully
- **Cross-Architecture**: Works seamlessly between ARM64 (Jetson) and AMD64 (PC)

### Implementation Details
```bash
# FastDDS uses default multicast configuration
# No custom XML configuration required
# Default multicast group: 239.255.0.1:7400
# Discovery works across network segments
```

### Verification Commands
```bash
# Check ROS2 nodes across network
ros2 node list

# Monitor cross-network topic communication  
ros2 topic list
ros2 topic echo /robot/sensors

# Verify service discovery
ros2 service list | grep robot
```

### Network Requirements Met
- ✅ Multicast traffic allowed on network
- ✅ Firewall configured for ROS2 ports
- ✅ Same ROS2 domain ID on both systems
- ✅ FastDDS multicast discovery enabled (default)

## Development Guidelines

### For Robot Coders
1. **NEVER publish directly to /cmd_vel**
2. **ALWAYS use the execute_command service**
3. **Include proper error handling**
4. **Log all command attempts**
5. **Respect command timeouts**
6. **Test with gateway_validator running**

### Adding New Commands
1. Extend RobotCommand.msg if needed
2. Update vila_server command processing
3. Test through execute_command service
4. Verify gateway_validator compliance
5. Update this documentation

### Debugging Communication
```bash
# Monitor command flow
ros2 topic echo /robot/commands

# Check gateway violations  
ros2 topic echo /system/gateway_violations

# View all active topics
ros2 topic list

# Monitor service calls
ros2 service list
```

## Troubleshooting

### Common Issues
1. **Commands not executing**: Check execute_command service availability
2. **Gateway violations**: Ensure no direct /cmd_vel publishing
3. **VILA not responding**: Verify VILA model loading
4. **Sensor data/status missing**: Check robot_client node and /robot/sensors topic
5. **GUI not updating**: Verify ROS2 topic subscriptions
6. **Cross-network discovery failing**: 
   - Verify multicast is enabled on network
   - Check ROS2 domain ID matches on both systems
   - Ensure firewall allows multicast traffic (239.255.0.1:7400)

### Diagnostic Commands
```bash
# Check node status
ros2 node list

# Test command service
ros2 service call /robot/execute_command robot_msgs/srv/ExecuteCommand "{command: {robot_id: 'robot_001', command_type: 'stop', timestamp_ns: 0, source_node: 'test'}}"

# Monitor system health
ros2 topic hz /robot/sensors

# Multicast discovery diagnostics
ros2 doctor --report  # Check ROS2 system health
ros2 topic list       # Verify cross-network topic discovery
ros2 node info /node_name  # Check node connectivity

# Network multicast testing
ping -c 3 239.255.0.1  # Test multicast connectivity
netstat -g             # Show multicast group memberships
```

---

## Summary

This ROS2-based communication system provides:
- ✅ **Centralized command control** through single gateway
- ✅ **Real-time sensor monitoring** via ROS2 topics  
- ✅ **VILA AI integration** with vision processing
- ✅ **Comprehensive validation** and safety features
- ✅ **Scalable architecture** using standard ROS2 patterns
- ✅ **Automatic multicast discovery** between Jetson and PC
- ✅ **Cross-platform communication** (ARM64 ↔ AMD64)

**Key Principle**: ALL robot communication flows through ROS2 topics and services, with command execution centralized through the single gateway service to ensure coordinated, safe, and auditable robot control.

**Network Success**: Multicast discovery is fully operational, enabling seamless automatic node discovery across the robot (Jetson Orin Nano) and client PC (Ubuntu 22.04) without manual configuration.
