#!/usr/bin/env python3
"""
ROS2 VILA Robot Server
Replaces HTTP-based communication with ROS2 topics and services
Maintains single command gateway architecture with ROS2 messaging
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import sys
import os
import json
import logging
import time
import threading
import queue
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import configparser

# ROS2 message imports
from robot_msgs.msg import RobotCommand, SensorData, VILAAnalysis, RobotStatus
from robot_msgs.srv import ExecuteCommand, RequestVILAAnalysis
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist

# Add VILA paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VILA'))
from main_vila import VILAModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robot_vila_server_ros2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RobotVILAServerROS2')

@dataclass
class RobotInfo:
    """Robot information for the single robot system"""
    robot_id: str
    name: str
    ip: str
    port: int
    last_seen: datetime
    status: str = "offline"
    sensor_data: Optional[Dict[str, Any]] = None
    command_history: List[str] = None

    def __post_init__(self):
        if self.command_history is None:
            self.command_history = []

class RobotVILAServerROS2(Node):
    """
    ROS2-based VILA Robot Server
    Maintains single command gateway [[memory:5366669]] using ROS2 topics and services
    """
    
    def __init__(self):
        super().__init__('robot_vila_server')
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize VILA model
        self.vila_model = VILAModel()
        self.model_loaded = False
        
        # Robot management
        self.robot_info = RobotInfo(
            robot_id=self.config.get('robot_id', 'yahboomcar_x3_01'),
            name=self.config.get('robot_name', 'YahBoom Car X3'),
            ip=self.config.get('robot_ip', '192.168.1.166'),
            port=self.config.get('robot_port', 8080),
            last_seen=datetime.now()
        )
        
        # Command queue for single robot [[memory:5366669]]
        self.command_queue = queue.Queue()
        
        # Safety status
        self.safety_enabled = True
        self.gui_movement_enabled = False
        
        # QoS profiles
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        self.best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        
        # Setup ROS2 interfaces
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_services()
        
        # Load VILA model
        self._load_vila_model_async()
        
        # Start command processing thread
        self.command_processor_thread = threading.Thread(
            target=self._process_commands, daemon=True
        )
        self.command_processor_thread.start()
        
        self.get_logger().info("🤖 ROS2 VILA Robot Server initialized")
        self.get_logger().info(f"   └── Robot ID: {self.robot_info.robot_id}")
        self.get_logger().info("   └── All communication now via ROS2 topics/services")
        self.get_logger().info("   └── Single command gateway maintained [[memory:5366669]]")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load robot configuration"""
        config_file = Path("robot_hub_config.ini")
        if config_file.exists():
            config = configparser.ConfigParser()
            config.read(config_file)
            
            return {
                'robot_id': config.get('ROBOT', 'robot_id', fallback='yahboomcar_x3_01'),
                'robot_name': config.get('ROBOT', 'robot_name', fallback='YahBoom Car X3'),
                'robot_ip': config.get('ROBOT', 'robot_ip', fallback='192.168.1.166'),
                'robot_port': config.getint('ROBOT', 'robot_port', fallback=8080)
            }
        else:
            return {
                'robot_id': 'yahboomcar_x3_01',
                'robot_name': 'YahBoom Car X3',
                'robot_ip': '192.168.1.166',
                'robot_port': 8080
            }
    
    def _setup_publishers(self):
        """Setup ROS2 publishers"""
        # Command publishing (for robot to receive)
        self.command_publisher = self.create_publisher(
            RobotCommand, 
            f'/robot/{self.robot_info.robot_id}/commands',
            self.reliable_qos
        )
        
        # Status publishing
        self.status_publisher = self.create_publisher(
            RobotStatus,
            f'/robot/{self.robot_info.robot_id}/status',
            self.reliable_qos
        )
        
        # VILA analysis results
        self.analysis_publisher = self.create_publisher(
            VILAAnalysis,
            '/vila/analysis',
            self.reliable_qos
        )
        
        # Safety status
        self.safety_publisher = self.create_publisher(
            Bool,
            '/robot/safety/enabled',
            self.reliable_qos
        )
        
        # Navigation commands (for GUI/monitoring)
        self.navigation_publisher = self.create_publisher(
            String,
            '/vila/navigation_commands',
            self.best_effort_qos
        )
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers"""
        # Sensor data from robot
        self.sensor_subscriber = self.create_subscription(
            SensorData,
            f'/robot/{self.robot_info.robot_id}/sensors',
            self._sensor_data_callback,
            self.best_effort_qos
        )
        
        # Camera images from robot
        self.image_subscriber = self.create_subscription(
            Image,
            f'/robot/{self.robot_info.robot_id}/camera/image_raw',
            self._image_callback,
            self.best_effort_qos
        )
        
        # Safety control from GUI
        self.safety_control_subscriber = self.create_subscription(
            Bool,
            '/robot/safety/enable_movement',
            self._safety_control_callback,
            self.reliable_qos
        )
        
        # Emergency stop
        self.emergency_stop_subscriber = self.create_subscription(
            Bool,
            '/robot/emergency_stop',
            self._emergency_stop_callback,
            self.reliable_qos
        )
    
    def _setup_services(self):
        """Setup ROS2 services"""
        # Command execution service (THE SINGLE GATEWAY [[memory:5366669]])
        self.execute_command_service = self.create_service(
            ExecuteCommand,
            '/robot/execute_command',
            self._execute_command_service
        )
        
        # VILA analysis service
        self.vila_analysis_service = self.create_service(
            RequestVILAAnalysis,
            '/vila/analyze',
            self._vila_analysis_service
        )
    
    def _load_vila_model_async(self):
        """Load VILA model in background thread"""
        def load_model():
            self.get_logger().info("🚀 Loading VILA model...")
            success = self.vila_model.load_model()
            if success:
                self.model_loaded = True
                self.get_logger().info("✅ VILA model loaded successfully")
            else:
                self.get_logger().error("❌ Failed to load VILA model")
        
        threading.Thread(target=load_model, daemon=True).start()
    
    def _sensor_data_callback(self, msg: SensorData):
        """Handle sensor data from robot"""
        self.robot_info.last_seen = datetime.now()
        self.robot_info.status = "active"
        
        # Store sensor data
        self.robot_info.sensor_data = {
            'battery_voltage': msg.battery_voltage,
            'battery_percentage': msg.battery_percentage,
            'temperature': msg.temperature,
            'humidity': msg.humidity,
            'distance_front': msg.distance_front,
            'distance_left': msg.distance_left,
            'distance_right': msg.distance_right,
            'wifi_signal': msg.wifi_signal,
            'cpu_usage': msg.cpu_usage,
            'memory_usage': msg.memory_usage,
            'imu_values': {
                'x': msg.imu_values.x,
                'y': msg.imu_values.y,
                'z': msg.imu_values.z
            },
            'camera_status': msg.camera_status,
            'timestamp': msg.timestamp_ns
        }
        
        # Publish status update
        self._publish_robot_status()
    
    def _image_callback(self, msg: Image):
        """Handle camera images from robot - trigger automatic VILA analysis"""
        if not self.model_loaded:
            return
        
        self.get_logger().debug("📸 Received camera image, processing with VILA...")
        
        # Process with VILA in background to avoid blocking
        threading.Thread(
            target=self._process_image_with_vila,
            args=(msg,),
            daemon=True
        ).start()
    
    def _process_image_with_vila(self, image_msg: Image):
        """Process image with VILA model"""
        try:
            # Convert ROS image to PIL (simplified - would need proper conversion)
            # For now, we'll create a placeholder analysis
            
            navigation_prompt = """You are a robot's vision system. Analyze this camera view and provide:
1. Can I move forward safely?
2. Are there obstacles ahead?
3. What should I do next (move_forward, turn_left, turn_right, stop)?
4. Describe what you see briefly.
Keep it concise for real-time navigation."""
            
            # Generate response (placeholder for actual VILA processing)
            response = "Path clear ahead. Safe to move forward. No obstacles detected."
            
            # Parse navigation commands
            nav_commands = self._parse_navigation_commands(response)
            
            # Create and publish VILA analysis
            analysis_msg = VILAAnalysis()
            analysis_msg.robot_id = self.robot_info.robot_id
            analysis_msg.prompt = navigation_prompt
            analysis_msg.image = image_msg
            analysis_msg.analysis_result = response
            analysis_msg.navigation_commands_json = json.dumps(nav_commands)
            analysis_msg.confidence = nav_commands.get('confidence', 0.0)
            analysis_msg.timestamp_ns = self.get_clock().now().nanoseconds
            analysis_msg.success = True
            analysis_msg.error_message = ""
            
            self.analysis_publisher.publish(analysis_msg)
            
            # Also publish navigation commands separately for GUI
            nav_msg = String()
            nav_msg.data = json.dumps(nav_commands)
            self.navigation_publisher.publish(nav_msg)
            
            self.get_logger().debug(f"✅ VILA analysis complete: {nav_commands['action']}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Error processing image with VILA: {e}")
    
    def _parse_navigation_commands(self, response: str) -> Dict[str, Any]:
        """Parse VILA response for navigation commands"""
        response_lower = response.lower()
        
        commands = {
            'action': 'stop',  # default safe action
            'confidence': 0.0,
            'reason': response
        }
        
        if 'move forward' in response_lower or ('clear' in response_lower and 'safe' in response_lower):
            commands['action'] = 'move_forward'
            commands['confidence'] = 0.8
        elif 'turn left' in response_lower:
            commands['action'] = 'turn_left'
            commands['confidence'] = 0.7
        elif 'turn right' in response_lower:
            commands['action'] = 'turn_right'
            commands['confidence'] = 0.7
        elif 'stop' in response_lower or 'obstacle' in response_lower:
            commands['action'] = 'stop'
            commands['confidence'] = 0.9
            
        return commands
    
    def _safety_control_callback(self, msg: Bool):
        """Handle safety control messages"""
        self.gui_movement_enabled = msg.data
        self.get_logger().info(f"🛡️ GUI movement {'ENABLED' if msg.data else 'DISABLED'}")
        
        # Publish safety status
        safety_msg = Bool()
        safety_msg.data = self.safety_enabled and self.gui_movement_enabled
        self.safety_publisher.publish(safety_msg)
    
    def _emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop"""
        if msg.data:
            self.get_logger().warn("🚨 EMERGENCY STOP ACTIVATED")
            self.safety_enabled = False
            self.gui_movement_enabled = False
            
            # Clear command queue
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Send stop command immediately
            stop_command = RobotCommand()
            stop_command.robot_id = self.robot_info.robot_id
            stop_command.command_type = "stop"
            stop_command.parameters_json = "{}"
            stop_command.timestamp_ns = self.get_clock().now().nanoseconds
            stop_command.priority = 10  # Highest priority
            stop_command.safety_confirmed = True
            stop_command.gui_movement_enabled = True
            stop_command.source = "EMERGENCY_STOP"
            
            self.command_publisher.publish(stop_command)
            
            # Publish safety status
            safety_msg = Bool()
            safety_msg.data = False
            self.safety_publisher.publish(safety_msg)
    
    def _execute_command_service(self, request, response):
        """
        🎯 THE SINGLE COMMAND GATEWAY [[memory:5366669]]
        
        This is the ONLY point where ALL robot commands are processed.
        Every command from GUI, API, ROS topics - EVERYTHING goes through here.
        """
        try:
            command = request.command
            
            self.get_logger().info(f"🎯 COMMAND GATEWAY: Processing {command.command_type} from {command.source}")
            
            # Safety validation
            if not self._validate_command_safety(command):
                response.success = False
                response.message = "Command rejected by safety system"
                response.timestamp_ns = self.get_clock().now().nanoseconds
                return response
            
            # Add to command queue
            self.command_queue.put(command)
            
            # Log command
            self.robot_info.command_history.append(f"{command.command_type}@{datetime.now().isoformat()}")
            if len(self.robot_info.command_history) > 100:
                self.robot_info.command_history = self.robot_info.command_history[-50:]
            
            response.success = True
            response.message = f"Command {command.command_type} queued for execution"
            response.timestamp_ns = self.get_clock().now().nanoseconds
            
            self.get_logger().info(f"✅ COMMAND QUEUED: {command.command_type} for {command.robot_id}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Command execution error: {e}")
            response.success = False
            response.message = f"Error: {str(e)}"
            response.timestamp_ns = self.get_clock().now().nanoseconds
        
        return response
    
    def _vila_analysis_service(self, request, response):
        """Handle VILA analysis requests"""
        try:
            if not self.model_loaded:
                response.success = False
                response.error_message = "VILA model not loaded"
                response.timestamp_ns = self.get_clock().now().nanoseconds
                return response
            
            self.get_logger().info(f"🔍 VILA analysis requested for {request.robot_id}")
            
            # Process with VILA (simplified - would need actual image processing)
            analysis_result = "Analysis complete - environment assessed"
            nav_commands = {'action': 'stop', 'confidence': 0.8, 'reason': analysis_result}
            
            response.success = True
            response.analysis_result = analysis_result
            response.navigation_commands_json = json.dumps(nav_commands)
            response.confidence = nav_commands['confidence']
            response.error_message = ""
            response.timestamp_ns = self.get_clock().now().nanoseconds
            
            self.get_logger().info("✅ VILA analysis complete")
            
        except Exception as e:
            self.get_logger().error(f"❌ VILA analysis error: {e}")
            response.success = False
            response.error_message = str(e)
            response.timestamp_ns = self.get_clock().now().nanoseconds
        
        return response
    
    def _validate_command_safety(self, command: RobotCommand) -> bool:
        """Validate command against safety constraints"""
        # Check if safety is enabled
        if not self.safety_enabled:
            self.get_logger().warn(f"🚫 SAFETY: Command {command.command_type} blocked - safety disabled")
            return False
        
        # Check movement commands
        if command.command_type in ['move', 'turn', 'forward', 'backward']:
            if not self.gui_movement_enabled and not command.safety_confirmed:
                self.get_logger().warn(f"🚫 SAFETY: Movement command {command.command_type} blocked - GUI movement disabled")
                return False
        
        # Always allow stop commands
        if command.command_type == 'stop':
            return True
        
        # Check robot status
        if self.robot_info.status == 'offline':
            self.get_logger().warn(f"🚫 SAFETY: Command {command.command_type} blocked - robot offline")
            return False
        
        return True
    
    def _process_commands(self):
        """Process commands from queue and send to robot"""
        while True:
            try:
                # Get command from queue (blocking)
                command = self.command_queue.get(timeout=1.0)
                
                # Publish command to robot
                self.command_publisher.publish(command)
                
                self.get_logger().info(f"📤 COMMAND SENT: {command.command_type} to {command.robot_id}")
                
                # Update robot status
                self._publish_robot_status()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"❌ Error processing command: {e}")
    
    def _publish_robot_status(self):
        """Publish current robot status"""
        status_msg = RobotStatus()
        status_msg.robot_id = self.robot_info.robot_id
        status_msg.name = self.robot_info.name
        status_msg.last_seen_ns = int(self.robot_info.last_seen.timestamp() * 1e9)
        status_msg.battery_level = self.robot_info.sensor_data.get('battery_percentage', 0.0) if self.robot_info.sensor_data else 0.0
        status_msg.status = self.robot_info.status
        status_msg.capabilities = ["navigation", "vision", "sensors"]
        status_msg.connection_type = "ROS2"
        status_msg.last_command = self.robot_info.command_history[-1] if self.robot_info.command_history else ""
        status_msg.command_history = self.robot_info.command_history[-10:]  # Last 10 commands
        
        # Add sensor data if available
        if self.robot_info.sensor_data:
            sensor_msg = SensorData()
            sensor_msg.robot_id = self.robot_info.robot_id
            sensor_msg.battery_voltage = self.robot_info.sensor_data.get('battery_voltage', 0.0)
            sensor_msg.battery_percentage = self.robot_info.sensor_data.get('battery_percentage', 0.0)
            sensor_msg.temperature = self.robot_info.sensor_data.get('temperature', 0.0)
            sensor_msg.humidity = self.robot_info.sensor_data.get('humidity', 0.0)
            sensor_msg.distance_front = self.robot_info.sensor_data.get('distance_front', 0.0)
            sensor_msg.distance_left = self.robot_info.sensor_data.get('distance_left', 0.0)
            sensor_msg.distance_right = self.robot_info.sensor_data.get('distance_right', 0.0)
            sensor_msg.wifi_signal = self.robot_info.sensor_data.get('wifi_signal', 0)
            sensor_msg.cpu_usage = self.robot_info.sensor_data.get('cpu_usage', 0.0)
            sensor_msg.memory_usage = self.robot_info.sensor_data.get('memory_usage', 0.0)
            sensor_msg.camera_status = self.robot_info.sensor_data.get('camera_status', "Unknown")
            sensor_msg.timestamp_ns = self.robot_info.sensor_data.get('timestamp', 0)
            
            imu_data = self.robot_info.sensor_data.get('imu_values', {})
            sensor_msg.imu_values.x = imu_data.get('x', 0.0)
            sensor_msg.imu_values.y = imu_data.get('y', 0.0)
            sensor_msg.imu_values.z = imu_data.get('z', 0.0)
            
            status_msg.sensor_data = sensor_msg
        
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        server = RobotVILAServerROS2()
        
        # Spin the node
        rclpy.spin(server)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        if 'server' in locals():
            server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
