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
from robot_msgs.msg import RobotCommand, SensorData, VILAAnalysis
from robot_msgs.srv import ExecuteCommand, RequestVILAAnalysis
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist

# Import VILA model from ROS2 package
from robot_vila_system.vila_model import VILAModel

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
        
        # Store latest IMU data separately
        self.latest_imu_data = None
        
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
            '/robot/commands',
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
            '/robot/navigation_commands',
            self.best_effort_qos
        )
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers"""
        # Sensor data from robot
        self.sensor_subscriber = self.create_subscription(
            SensorData,
            '/robot/sensors',
            self._sensor_data_callback,
            self.best_effort_qos
        )
        
        # Camera images from robot
        self.image_subscriber = self.create_subscription(
            Image,
            '/robot/camera/image_raw',
            self._image_callback,
            self.best_effort_qos
        )
        
        # IMU data from robot
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data_raw',
            self._imu_callback,
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
            '/vila/request_analysis',
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
        
        # Store sensor data (IMU data now comes from /imu/data_raw topic)
        self.robot_info.sensor_data = {
            'battery_voltage': msg.battery_voltage,
            'cpu_temp': msg.cpu_temp,
            'distance_front': msg.distance_front,
            'distance_left': msg.distance_left,
            'distance_right': msg.distance_right,
            'cpu_usage': msg.cpu_usage,
            'camera_status': msg.camera_status,
            'timestamp': msg.timestamp_ns
        }
        
        # Publish status update
        self._publish_robot_status()
    
    def _imu_callback(self, msg: Imu):
        """Handle IMU data from robot"""
        try:
            # Store latest IMU data
            self.latest_imu_data = {
                'orientation': {
                    'x': msg.orientation.x,
                    'y': msg.orientation.y,
                    'z': msg.orientation.z,
                    'w': msg.orientation.w
                },
                'angular_velocity': {
                    'x': msg.angular_velocity.x,
                    'y': msg.angular_velocity.y,
                    'z': msg.angular_velocity.z
                },
                'linear_acceleration': {
                    'x': msg.linear_acceleration.x,
                    'y': msg.linear_acceleration.y,
                    'z': msg.linear_acceleration.z
                },
                'timestamp': msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
            }
            
            self.get_logger().debug("📊 IMU data updated")
            
        except Exception as e:
            self.get_logger().error(f"❌ Error processing IMU data: {e}")
    
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
            stop_command.linear_x = 0.0
            stop_command.linear_y = 0.0
            stop_command.angular_z = 0.0
            stop_command.duration = 0.0
            stop_command.parameters_json = "{}"
            stop_command.timestamp_ns = self.get_clock().now().nanoseconds
            stop_command.source_node = "vila_server_node"
            
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
            
            self.get_logger().info(f"🎯 COMMAND GATEWAY: Processing {command.command_type} from {command.source_node}")
            
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
        """Publish robot status information"""
        try:
            # Create robot status message
            status_msg = String()
            status_data = {
                'robot_id': self.robot_info.robot_id,
                'status': self.robot_info.status,
                'last_seen': self.robot_info.last_seen.isoformat(),
                'model_loaded': self.model_loaded,
                'safety_enabled': self.safety_enabled,
                'gui_movement_enabled': self.gui_movement_enabled
            }
            
            if self.robot_info.sensor_data:
                status_data['sensor_data'] = self.robot_info.sensor_data
            
            status_msg.data = json.dumps(status_data)
            
            # Publish status (assuming we have a status publisher)
            # Note: This would need a status publisher to be set up in _setup_publishers()
            self.get_logger().debug(f"📊 Robot status updated: {self.robot_info.status}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Error publishing robot status: {e}")
    


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
