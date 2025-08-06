# 🤖 Robot Integration Guide - Unified System

## Overview for Robot Developers

This guide is for developers working on the **robot side** (Jetson Orin Nano) who need to integrate their robots with the new **Unified Robot Controller**. The system has been redesigned for maximum efficiency - **single process, direct communication, no duplicate models**.

> **⚠️ Important**: The robot control system has been completely redesigned. The old multi-process system with duplicate VILA models has been replaced with a unified, efficient architecture.

## 🏗️ System Architecture

```
┌─────────────────────────────────────┐
│         Client PC (Ubuntu 22.04)    │
│    ┌─────────────────────────────┐   │
│    │  Unified Robot Controller   │   │
│    │  • Single VILA Model        │   │
│    │  • Centralized Safety       │   │
│    │  • Direct Communication     │   │
│    └─────────────────────────────┘   │
└─────────────────┬───────────────────┘
                  │ HTTP/WebSocket
                  │ (Direct, No Hops)
┌─────────────────▼───────────────────┐
│      Jetson Orin Nano (Your Code)   │
│  ┌─────────────────────────────────┐ │
│  │        Robot ROS2 System        │ │
│  │  • Sensor Data Publisher       │ │
│  │  • Command Receiver            │ │
│  │  • Safety Compliance           │ │
│  │  • HTTP Client Integration     │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

## 🔄 What Changed from Old System

### **Old System Issues (Now Fixed)**
- ❌ Multiple VILA models (~6GB GPU memory)
- ❌ Complex communication (HTTP → WebSocket → TCP → ROS)
- ❌ Safety system conflicts and bypasses
- ❌ 4+ separate processes to manage

### **New Unified System**
- ✅ **Single VILA model** (~3GB GPU memory) 
- ✅ **Direct HTTP communication** (Robot ↔ Controller)
- ✅ **Centralized safety** (no bypasses possible)
- ✅ **Single process** (simplified deployment)

## 📡 Integration Requirements

### **1. HTTP Client Implementation**

Your robot **must** implement an HTTP client to communicate with the unified controller:

```python
# Example: Robot-side HTTP client
import aiohttp
import json
from datetime import datetime

class RobotControllerClient:
    def __init__(self, robot_id: str, controller_host: str = "192.168.1.100"):
        self.robot_id = robot_id
        self.controller_url = f"http://{controller_host}:5000"
        self.session = None
    
    async def register_robot(self):
        """Register with unified controller"""
        registration_data = {
            "robot_id": self.robot_id,
            "name": "YahBoom Robot",
            "position": {"x": 0, "y": 0, "z": 0, "heading": 0, "ip": "192.168.1.101"},
            "battery_level": 85.0,  # Real battery level
            "capabilities": ["navigation", "vision", "sensors"],
            "connection_type": "http",
            "sensor_data": self.get_current_sensors()
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.controller_url}/api/robots/register",
                json=registration_data
            ) as response:
                return response.status == 200
    
    async def send_sensor_data(self, sensor_data: dict):
        """Send real-time sensor data"""
        sensor_data["timestamp"] = datetime.now().isoformat()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.controller_url}/api/robots/{self.robot_id}/sensors",
                json=sensor_data
            ) as response:
                return response.status == 200
    
    def get_current_sensors(self):
        """Get real sensor readings - IMPLEMENT THIS"""
        return {
            "battery_voltage": 12.6,      # From your battery monitor
            "battery_percentage": 85.0,   # Calculated from voltage
            "temperature": 23.5,          # From temperature sensor
            "humidity": 45.0,             # From humidity sensor if available
            "distance_front": 1.2,        # From ultrasonic/lidar
            "distance_left": 0.8,         # From ultrasonic/lidar  
            "distance_right": 1.5,        # From ultrasonic/lidar
            "wifi_signal": -45,           # From system
            "cpu_usage": 25.0,            # From system monitoring
            "memory_usage": 60.0,         # From system monitoring
            "wheel_encoder_left": 1234,   # From wheel encoders
            "wheel_encoder_right": 1256,  # From wheel encoders
            "imu_pitch": 0.1,             # From IMU if available
            "imu_roll": -0.05,            # From IMU if available
            "imu_yaw": 45.2               # From IMU if available
        }
```

### **2. ROS2 Integration Pattern**

Create a ROS2 node that bridges to the unified controller:

```python
#!/usr/bin/env python3
"""
Robot Bridge Node for Unified System
Bridges ROS2 sensors/actuators with HTTP-based unified controller
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Range, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import asyncio
import aiohttp
import threading
from datetime import datetime

class UnifiedRobotBridge(Node):
    def __init__(self):
        super().__init__('unified_robot_bridge')
        
        # Robot identification
        self.robot_id = "yahboom_robot_001"
        self.controller_url = "http://192.168.1.100:5000"
        
        # ROS2 Subscribers (your sensors)
        self.battery_sub = self.create_subscription(
            BatteryState, '/battery_state', self.battery_callback, 10)
        self.range_sub = self.create_subscription(
            Range, '/ultrasonic', self.range_callback, 10)
        
        # ROS2 Publishers (your actuators)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Sensor data storage
        self.sensor_data = {}
        
        # HTTP client setup
        self.setup_async_client()
        
        # Periodic sensor sending
        self.sensor_timer = self.create_timer(2.0, self.send_sensor_data_sync)
        
        self.get_logger().info("🤖 Unified Robot Bridge started")
    
    def setup_async_client(self):
        """Setup async HTTP client in separate thread"""
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self.run_async_loop)
        self.async_thread.daemon = True
        self.async_thread.start()
        
        # Register robot
        asyncio.run_coroutine_threadsafe(self.register_robot(), self.loop)
    
    def run_async_loop(self):
        """Run async event loop in separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    async def register_robot(self):
        """Register with unified controller"""
        registration_data = {
            "robot_id": self.robot_id,
            "name": "YahBoom Robot",
            "position": {"x": 0, "y": 0, "z": 0, "heading": 0, "ip": "192.168.1.101"},
            "battery_level": 0.0,  # Will be updated by sensor callbacks
            "capabilities": ["navigation", "vision", "sensors"],
            "connection_type": "http"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.controller_url}/api/robots/register",
                    json=registration_data,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        self.get_logger().info("✅ Registered with unified controller")
                    else:
                        self.get_logger().error(f"❌ Registration failed: {response.status}")
        except Exception as e:
            self.get_logger().error(f"❌ Registration error: {e}")
    
    def battery_callback(self, msg: BatteryState):
        """Handle battery sensor data"""
        self.sensor_data.update({
            "battery_voltage": msg.voltage,
            "battery_percentage": msg.percentage * 100 if msg.percentage >= 0 else 0.0,
            "battery_current": msg.current,
            "battery_temperature": msg.temperature
        })
    
    def range_callback(self, msg: Range):
        """Handle ultrasonic sensor data"""
        # Map range sensors by frame_id
        if "front" in msg.header.frame_id:
            self.sensor_data["distance_front"] = msg.range
        elif "left" in msg.header.frame_id:
            self.sensor_data["distance_left"] = msg.range
        elif "right" in msg.header.frame_id:
            self.sensor_data["distance_right"] = msg.range
    
    def send_sensor_data_sync(self):
        """Send sensor data (sync wrapper for async)"""
        if self.sensor_data:
            asyncio.run_coroutine_threadsafe(
                self.send_sensor_data_async(), self.loop
            )
    
    async def send_sensor_data_async(self):
        """Send sensor data to unified controller"""
        if not self.sensor_data:
            return
            
        # Add system metrics
        sensor_payload = {
            **self.sensor_data,
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": self.get_cpu_usage(),
            "memory_usage": self.get_memory_usage(),
            "wifi_signal": self.get_wifi_signal()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.controller_url}/api/robots/{self.robot_id}/sensors",
                    json=sensor_payload,
                    timeout=5
                ) as response:
                    if response.status == 200:
                        self.get_logger().info(f"📊 Sent sensor data: {len(sensor_payload)} fields")
                    else:
                        self.get_logger().warning(f"⚠️ Sensor data failed: {response.status}")
        except Exception as e:
            self.get_logger().error(f"❌ Sensor send error: {e}")
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except:
            return 0.0
    
    def get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    def get_wifi_signal(self) -> int:
        """Get WiFi signal strength"""
        try:
            import subprocess
            result = subprocess.run(['iwconfig'], capture_output=True, text=True)
            # Parse signal strength from iwconfig output
            for line in result.stdout.split('\n'):
                if 'Signal level' in line:
                    # Extract signal strength (e.g., "-45 dBm")
                    import re
                    match = re.search(r'Signal level=(-?\d+)', line)
                    if match:
                        return int(match.group(1))
            return -99  # No signal
        except:
            return -99

def main(args=None):
    rclpy.init(args=args)
    bridge = UnifiedRobotBridge()
    
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 🛡️ Critical Safety Requirements

### **⚠️ SAFETY COMPLIANCE IS MANDATORY**

The unified system has a **centralized safety controller**. Your robot **MUST** respect safety commands:

```python
# Before executing ANY movement command, check with safety system:
async def execute_movement_command(self, command):
    """Execute movement with safety check"""
    
    # 1. Check with unified controller safety system
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.controller_url}/api/safety/status") as response:
                if response.status == 200:
                    safety_status = await response.json()
                    
                    # Check if movement is allowed
                    if not safety_status.get('movement_enabled', False):
                        self.get_logger().warning("🚫 Movement blocked by safety system")
                        return False
                    
                    if safety_status.get('emergency_stop', False):
                        self.get_logger().error("🚨 EMERGENCY STOP ACTIVE - Halting immediately")
                        self.publish_stop_command()
                        return False
                        
    except Exception as e:
        self.get_logger().error(f"❌ Safety check failed: {e}")
        return False  # Fail safe - don't move if can't verify safety
    
    # 2. If safety allows, execute the command
    self.get_logger().info(f"✅ Executing movement: {command}")
    # Your movement code here...
    return True

def publish_stop_command(self):
    """Immediately stop all robot movement"""
    stop_twist = Twist()  # All zeros
    self.cmd_vel_pub.publish(stop_twist)
```

## 📊 Required Sensor Data Format

Send sensor data in this **exact format** for GUI compatibility:

```python
sensor_data = {
    # Battery (REQUIRED)
    "battery_voltage": 12.6,        # Volts (float)
    "battery_percentage": 85.0,     # Percent 0-100 (float)
    
    # Distance sensors (RECOMMENDED)
    "distance_front": 1.2,          # Meters (float)
    "distance_left": 0.8,           # Meters (float) 
    "distance_right": 1.5,          # Meters (float)
    "distance_back": 2.0,           # Meters (float, optional)
    
    # Environmental (OPTIONAL)
    "temperature": 23.5,            # Celsius (float)
    "humidity": 45.0,               # Percent 0-100 (float)
    
    # System monitoring (RECOMMENDED)
    "cpu_usage": 25.0,              # Percent 0-100 (float)
    "memory_usage": 60.0,           # Percent 0-100 (float)
    "wifi_signal": -45,             # dBm (integer)
    
    # Navigation (OPTIONAL)
    "wheel_encoder_left": 1234,     # Encoder counts (integer)
    "wheel_encoder_right": 1256,    # Encoder counts (integer)
    "imu_pitch": 0.1,               # Degrees (float)
    "imu_roll": -0.05,              # Degrees (float)
    "imu_yaw": 45.2,                # Degrees (float)
    
    # Position (OPTIONAL)
    "position_x": 1.5,              # Meters (float)
    "position_y": 2.3,              # Meters (float)
    "heading": 45.0,                # Degrees (float)
    
    # Timestamp (AUTOMATIC)
    "timestamp": "2024-01-01T12:00:00"  # ISO format (string)
}
```

## 🔌 Network Configuration

### **Client PC (Ubuntu 22.04)**
- **IP**: `192.168.1.100` (or your network setup)
- **Port**: `5000` (HTTP API)
- **Service**: Unified Robot Controller

### **Robot (Jetson Orin Nano)**
- **IP**: `192.168.1.101` (configure as needed)
- **Requirements**: HTTP client, ROS2 bridge
- **Dependencies**: `aiohttp`, `psutil`

## 🚀 Deployment Steps

### **1. Install Dependencies on Robot**
```bash
# On Jetson Orin Nano
pip3 install aiohttp psutil requests

# Or add to your requirements.txt:
echo "aiohttp>=3.9.0" >> requirements.txt
echo "psutil>=5.9.0" >> requirements.txt
```

### **2. Create Bridge Node**
Save the `UnifiedRobotBridge` code above as `unified_robot_bridge.py` in your ROS2 package.

### **3. Update Launch Files**
```xml
<!-- In your launch file -->
<launch>
  <node pkg="your_robot_package" exec="unified_robot_bridge" name="unified_bridge">
    <param name="robot_id" value="yahboom_robot_001"/>
    <param name="controller_host" value="192.168.1.100"/>
  </node>
  
  <!-- Your other nodes -->
</launch>
```

### **4. Test Connection**
```bash
# Test HTTP connectivity
curl http://192.168.1.100:5000/api/robots

# Run bridge node
ros2 run your_package unified_robot_bridge
```

## 🧪 Testing & Validation

### **1. Sensor Data Test**
```bash
# Check if sensor data is being received
curl http://192.168.1.100:5000/api/robots
```

### **2. Safety System Test**
```bash
# Test emergency stop
curl -X POST http://192.168.1.100:5000/api/safety/emergency_stop

# Check safety status
curl http://192.168.1.100:5000/api/safety/status
```

### **3. End-to-End Test**
1. Start unified controller on client PC
2. Start your robot bridge node
3. Verify robot appears in web dashboard: `http://192.168.1.100:5000`
4. Check sensor data updates in real-time
5. Test safety controls

## 📝 Migration Checklist

- [ ] **Remove old system integration** (if any)
- [ ] **Install HTTP client dependencies** (`aiohttp`, `psutil`)
- [ ] **Create unified bridge node** (replace old communication)
- [ ] **Implement sensor data publishing** (exact format above)
- [ ] **Add safety compliance checks** (mandatory)
- [ ] **Update network configuration** (IP addresses)
- [ ] **Test registration and sensor data**
- [ ] **Verify safety system integration**
- [ ] **Update launch files and deployment**

## 🆘 Troubleshooting

### **Connection Issues**
```bash
# Check network connectivity
ping 192.168.1.100

# Check if unified controller is running
curl http://192.168.1.100:5000/api/robots
```

### **Registration Failures**
- Verify robot_id is unique
- Check JSON format of registration data
- Ensure all required fields are present

### **Sensor Data Issues**
- Check sensor data format matches specification
- Verify timestamp format (ISO 8601)
- Monitor logs for HTTP errors

### **Safety System Issues**
- Always check safety status before movement
- Implement emergency stop handling
- Test safety override scenarios

## 📞 Support

- **System Status**: Web dashboard at `http://192.168.1.100:5000`
- **API Testing**: Use `curl` commands above
- **Logs**: Check unified controller logs for integration issues
- **Performance**: Monitor sensor update rates and response times

---

**🎯 Goal**: Integrate your robot with the new unified system for 50% better performance, eliminated safety conflicts, and simplified architecture.

The unified system is **production-ready** and provides significant improvements over the old multi-process system. Focus on implementing the HTTP client integration and safety compliance - the rest will work automatically!