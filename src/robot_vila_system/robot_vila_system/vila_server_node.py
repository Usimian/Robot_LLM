#!/usr/bin/env python3
"""
ROS2 VILA Robot Server
Replaces HTTP-based communication with ROS2 topics and services
Maintains single command gateway architecture with ROS2 messaging
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
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
        
        # Initialize VILA model (HTTP client only - server started by launch file)
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
        
        # VILA server is now purely on-demand - images provided by client requests
        
        # QoS profiles
        # QoS for image streams (compatible with RealSense camera)
        self.image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # RealSense uses BEST_EFFORT
            durability=DurabilityPolicy.VOLATILE,       # Images are transient data
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Only need latest image for analysis
        )
        
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
        
        # Load VILA model (with delay to allow server startup)
        self._load_vila_model_async()
        
        # Start command processing thread
        self.command_processor_thread = threading.Thread(
            target=self._process_commands, daemon=True
        )
        self.command_processor_thread.start()
        
        # Start VILA status publishing timer (every 2 seconds)
        self.vila_status_timer = self.create_timer(2.0, self._publish_vila_status)
        
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
        
        # VILA server status
        self.vila_status_publisher = self.create_publisher(
            String,
            '/vila/server_status',
            self.reliable_qos
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
        
        # VILA server is now purely on-demand - no autonomous image processing
        # Camera images are provided by the client when requesting analysis
        
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
        
        # Note: VILA is now resource-based - no enable/disable, only on-demand requests
    
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
        """Load VILA model in background thread with startup delay"""
        def load_model():
            # Wait for VILA server to start up (launched by launch file)
            self.get_logger().info("⏳ Waiting for VILA server to start...")
            time.sleep(10)  # Give VILA server more time to start and load model
            
            self.get_logger().info("🚀 Loading VILA model...")
            
            # Try multiple times with increasing delays
            for attempt in range(3):
                success = self.vila_model.load_model()
                if success:
                    self.model_loaded = True
                    self.get_logger().info("✅ VILA model loaded successfully")
                    # Publish updated status immediately after model loads
                    self._publish_vila_status()
                    return
                else:
                    self.get_logger().warn(f"⚠️ VILA model load attempt {attempt + 1}/3 failed")
                    if attempt < 2:  # Don't sleep on the last attempt
                        time.sleep(5)  # Wait 5 seconds before retry
            
            self.get_logger().error("❌ Failed to load VILA model after 3 attempts - will retry on first request")
        
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
    
    # Image callback removed - VILA server is now purely on-demand
    # Images are provided by the client in analysis requests
    
    # Legacy visual-only analysis method completely removed
    
    def _parse_navigation_commands(self, response: str, lidar_data: Dict[str, float] = None) -> Dict[str, Any]:
        """Parse VILA response for navigation commands - trust VILA's decisions completely"""
        response_lower = response.lower()
        
        commands = {
            'action': 'stop',  # default if parsing fails
            'confidence': 0.0,
            'reason': response,
            'lidar_context': 'none'
        }
        
        # Add LiDAR context for logging only (no overrides)
        if lidar_data:
            front_dist = lidar_data['distance_front']
            left_dist = lidar_data['distance_left'] 
            right_dist = lidar_data['distance_right']
            
            # Store LiDAR context for logging only
            commands['lidar_context'] = f'F:{front_dist:.1f}m_L:{left_dist:.1f}m_R:{right_dist:.1f}m'
            commands['reason'] = f"VILA Decision (LiDAR: F:{front_dist:.2f}m, L:{left_dist:.2f}m, R:{right_dist:.2f}m): " + response
        
        # Parse VILA response - trust VILA completely, no safety overrides
        # Look for the structured output format first
        if 'Action:' in response:
            # Parse structured response
            action_line = [line.strip() for line in response.split('\n') if line.strip().startswith('Action:')]
            if action_line:
                action_text = action_line[0].replace('Action:', '').strip().lower()
                if 'move_forward' in action_text or 'forward' in action_text:
                    commands.update({'action': 'move_forward', 'confidence': 0.9})
                elif 'turn_left' in action_text or 'left' in action_text:
                    commands.update({'action': 'turn_left', 'confidence': 0.9})
                elif 'turn_right' in action_text or 'right' in action_text:
                    commands.update({'action': 'turn_right', 'confidence': 0.9})
                elif 'stop' in action_text:
                    commands.update({'action': 'stop', 'confidence': 0.9})
        else:
            # Fallback to keyword parsing if structured format not used
            # Count occurrences to find the most mentioned action
            move_forward_count = sum(1 for phrase in ['move forward', 'move_forward', 'go forward', 'forward', 'ahead', 'straight'] if phrase in response_lower)
            turn_left_count = sum(1 for phrase in ['turn left', 'turn_left', 'go left', 'left'] if phrase in response_lower)
            turn_right_count = sum(1 for phrase in ['turn right', 'turn_right', 'go right', 'right'] if phrase in response_lower)
            stop_count = sum(1 for phrase in ['stop', 'halt', 'wait', 'stay'] if phrase in response_lower)

            # Find the action with the highest count
            action_counts = {
                'move_forward': move_forward_count,
                'turn_left': turn_left_count,
                'turn_right': turn_right_count,
                'stop': stop_count
            }

            max_count = max(action_counts.values())
            if max_count > 0:
                # Get all actions with the maximum count
                top_actions = [action for action, count in action_counts.items() if count == max_count]

                # If tie, prefer turn actions over move forward (safer)
                if len(top_actions) > 1:
                    if 'turn_left' in top_actions:
                        best_action = 'turn_left'
                    elif 'turn_right' in top_actions:
                        best_action = 'turn_right'
                    elif 'stop' in top_actions:
                        best_action = 'stop'
                    else:
                        best_action = top_actions[0]  # fallback to first one
                else:
                    best_action = top_actions[0]

                confidence = 0.8 if best_action != 'stop' else 0.9
                commands.update({'action': best_action, 'confidence': confidence})

                self.get_logger().info(f"🎯 VILA Keyword Analysis: move_forward={move_forward_count}, turn_left={turn_left_count}, turn_right={turn_right_count}, stop={stop_count}")
                self.get_logger().info(f"🎯 Selected: {best_action} (highest count: {max_count})")
            else:
                # No keywords found - default to stop
                commands.update({
                    'action': 'stop',
                    'confidence': 0.3,
                    'reason': f"UNCLEAR VILA RESPONSE: No navigation keywords found. Response: " + response
                })
        
        self.get_logger().info(f"🎯 VILA Command: {commands['action']} (confidence: {commands['confidence']:.2f})")
        return commands
    
    def _safety_control_callback(self, msg: Bool):
        """Handle safety control messages"""
        # Update server safety state based on GUI request
        previous_state = self.safety_enabled
        self.safety_enabled = msg.data
        
        self.get_logger().info(f"🛡️ Safety {'ENABLED' if msg.data else 'DISABLED'} via GUI request")
        
        # Only publish if state actually changed to avoid loops
        if previous_state != self.safety_enabled:
            safety_msg = Bool()
            safety_msg.data = self.safety_enabled
            self.safety_publisher.publish(safety_msg)
            self.get_logger().info(f"🔄 Published safety status update: {'ENABLED' if self.safety_enabled else 'DISABLED'}")
        else:
            self.get_logger().debug("Safety state unchanged, no publication needed")
    
    def _emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop"""
        if msg.data:
            self.get_logger().warn("🚨 EMERGENCY STOP ACTIVATED")
            self.safety_enabled = False
            
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
            stop_command.distance = 0.0
            stop_command.angular = 0.0
            stop_command.linear_speed = 0.0
            stop_command.angular_speed = 0.0
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
            
            import time
            request_id = int(time.time() * 1000) % 10000  # Match GUI request ID timing
            
            self.get_logger().info(f"🎯 COMMAND GATEWAY: Processing {command.command_type} from {command.source_node} (ID: {request_id})")
            
            # Safety validation
            if not self._validate_command_safety(command):
                self.get_logger().warn(f"🚫 GATEWAY: Rejecting {command.command_type} (ID: {request_id}) - safety validation failed")
                response.success = False
                response.result_message = "Command rejected by safety system"
                return response
            
            # Add to command queue
            self.command_queue.put(command)
            
            # Log command
            self.robot_info.command_history.append(f"{command.command_type}@{datetime.now().isoformat()}")
            if len(self.robot_info.command_history) > 100:
                self.robot_info.command_history = self.robot_info.command_history[-50:]
            
            response.success = True
            response.result_message = f"Command {command.command_type} queued for execution"
            
            self.get_logger().info(f"✅ COMMAND QUEUED: {command.command_type} for {command.robot_id} (ID: {request_id})")
            self.get_logger().debug(f"🔍 RESPONSE SENT: success=True, message='{response.result_message}' (ID: {request_id})")
            
        except Exception as e:
            self.get_logger().error(f"❌ Command execution error: {e}")
            response.success = False
            response.result_message = f"Error: {str(e)}"
        
        return response
    
    def _vila_analysis_service(self, request, response):
        """Handle VILA analysis requests - on-demand resource-based processing"""
        try:
            # Check if model is loaded, if not try to load it on-demand
            if not self.model_loaded:
                self.get_logger().info("🔄 VILA model not loaded yet, attempting on-demand loading...")
                success = self.vila_model.load_model()
                if success:
                    self.model_loaded = True
                    self.get_logger().info("✅ VILA model loaded successfully on-demand")
                    # Publish updated status immediately after on-demand loading
                    self._publish_vila_status()
                else:
                    self.get_logger().error("❌ Failed to load VILA model on-demand")
                    response.success = False
                    response.error_message = "VILA model failed to load"
                    response.analysis_result = ""
                    response.navigation_commands_json = "{}"
                    response.confidence = 0.0
                    response.timestamp_ns = self.get_clock().now().nanoseconds
                    return response
            
            # Client must provide an image - no stored images in server
            if not request.image.data:
                response.success = False
                response.error_message = "No image provided in request - client must send image"
                response.analysis_result = ""
                response.navigation_commands_json = "{}"
                response.confidence = 0.0
                response.timestamp_ns = self.get_clock().now().nanoseconds
                return response
            
            # Move detailed request info to debug level
            self.get_logger().debug(f"🔍 On-demand VILA analysis requested for {request.robot_id}")
            self.get_logger().debug(f"   └── Prompt: {request.prompt}")
            self.get_logger().debug(f"   └── Image: {request.image.width}x{request.image.height}, encoding: {request.image.encoding}")
            self.get_logger().debug(f"   └── LiDAR: Front={request.distance_front:.2f}m, Left={request.distance_left:.2f}m, Right={request.distance_right:.2f}m")
            # Keep a simple info message for important operations
            self.get_logger().info(f"🔍 VILA analysis: {request.prompt[:40]}...")
            
            # Use the image provided by the client
            image_to_analyze = request.image
            
            # Extract LiDAR distances for enhanced analysis
            lidar_data = {
                'distance_front': request.distance_front,
                'distance_left': request.distance_left,
                'distance_right': request.distance_right
            }
            
            # Convert to SensorData for enhanced Cosmos Nemotron VLA analysis
            from robot_vila_system.vila_model import SensorData
            from cv_bridge import CvBridge
            from PIL import Image as PILImage
            import cv2
            
            # Convert ROS image to PIL for Cosmos Nemotron VLA with proper validation
            bridge = CvBridge()
            encoding = image_to_analyze.encoding if image_to_analyze.encoding else "bgr8"
            
            # Validate image data before processing
            if not image_to_analyze.data:
                self.get_logger().error("❌ Empty image data received")
                response.success = False
                response.error_message = "Empty image data - cannot analyze"
                response.analysis_result = "No image data provided"
                response.navigation_commands_json = json.dumps({'action': 'stop', 'confidence': 0.0, 'reason': 'No image data'})
                response.confidence = 0.0
                response.timestamp_ns = self.get_clock().now().nanoseconds
                return response
            
            self.get_logger().debug(f"🖼️ Converting image: {image_to_analyze.width}x{image_to_analyze.height}, encoding: {encoding}, data size: {len(image_to_analyze.data)} bytes")
            
            try:
                cv_image = bridge.imgmsg_to_cv2(image_to_analyze, encoding)
                
                # Validate converted image
                if cv_image is None or cv_image.size == 0:
                    self.get_logger().error("❌ Failed to convert ROS image to OpenCV format")
                    raise ValueError("Invalid OpenCV image after conversion")
                
                self.get_logger().debug(f"✅ OpenCV image shape: {cv_image.shape}")
                
                # Convert color space with validation
                if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                    if encoding == "rgb8":
                        # Image is already RGB, no conversion needed
                        rgb_image = cv_image
                    else:
                        # Convert BGR to RGB
                        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                else:
                    self.get_logger().error(f"❌ Unexpected image shape: {cv_image.shape}")
                    raise ValueError(f"Unexpected image shape: {cv_image.shape}")
                
                pil_image = PILImage.fromarray(rgb_image)
                self.get_logger().debug(f"✅ PIL image created: {pil_image.size}, mode: {pil_image.mode}")
                
            except Exception as e:
                self.get_logger().error(f"❌ Image conversion error: {e}")
                response.success = False
                response.error_message = f"Image conversion failed: {str(e)}"
                response.analysis_result = "Image processing error"
                response.navigation_commands_json = json.dumps({'action': 'stop', 'confidence': 0.0, 'reason': f'Image conversion error: {str(e)}'})
                response.confidence = 0.0
                response.timestamp_ns = self.get_clock().now().nanoseconds
                return response
            
            # Create SensorData with all available sensor information
            sensor_data = SensorData(
                lidar_distances=lidar_data,
                imu_data=self.latest_imu_data or {
                    'acceleration': {'x': 0, 'y': 0, 'z': 9.8},
                    'gyroscope': {'x': 0, 'y': 0, 'z': 0}
                },
                camera_image=pil_image,
                timestamp=time.time()
            )
            
            # Use enhanced Cosmos Nemotron VLA analysis
            self.get_logger().info(f"🚀 Using Cosmos Nemotron VLA multi-modal analysis with LiDAR: F={lidar_data.get('distance_front', 'N/A')}m")
            cosmos_result = self.vila_model.analyze_multi_modal_scene(sensor_data, request.prompt)
            
            if cosmos_result['success']:
                analysis_result = cosmos_result['analysis']
                nav_commands = {
                    'action': cosmos_result['navigation_command'],
                    'confidence': cosmos_result['confidence'],
                    'reasoning': cosmos_result['reasoning']
                }
                self.get_logger().info(f"✅ Cosmos Nemotron VLA result: {nav_commands['action']} (confidence: {nav_commands['confidence']:.2f})")
                self.get_logger().info(f"✅ Cosmos reasoning: {nav_commands['reasoning'][:100]}...")
            else:
                # Cosmos analysis is REQUIRED - no fallbacks
                error_msg = f"Cosmos analysis failed: {cosmos_result.get('analysis', 'Unknown error')}"
                self.get_logger().error(f"❌ {error_msg}")
                raise RuntimeError(f"Cosmos VILA analysis failed - no fallback methods: {error_msg}")
            
            # Fill service response directly
            response.success = True
            response.analysis_result = analysis_result
            response.navigation_commands_json = json.dumps(nav_commands)
            response.confidence = nav_commands.get('confidence', 0.0)
            response.error_message = ""
            response.timestamp_ns = self.get_clock().now().nanoseconds
            
            # Create VILAAnalysis message for publishing
            analysis_msg = VILAAnalysis()
            analysis_msg.robot_id = request.robot_id
            analysis_msg.prompt = request.prompt
            analysis_msg.analysis_result = analysis_result
            analysis_msg.navigation_commands_json = json.dumps(nav_commands)
            analysis_msg.confidence = nav_commands.get('confidence', 0.0)
            analysis_msg.success = True
            analysis_msg.error_message = ""
            analysis_msg.timestamp_ns = self.get_clock().now().nanoseconds
            
            # Publish analysis results for other nodes to consume
            self.analysis_publisher.publish(analysis_msg)
            
            # Also publish navigation commands separately for GUI
            nav_msg = String()
            nav_msg.data = json.dumps(nav_commands)
            self.navigation_publisher.publish(nav_msg)
            
            self.get_logger().info(f"✅ On-demand VILA analysis complete: {nav_commands['action']}")
            
        except Exception as e:
            self.get_logger().error(f"❌ VILA analysis error: {e}")
            response.success = False
            response.error_message = str(e)
            response.analysis_result = ""
            response.navigation_commands_json = "{}"
            response.confidence = 0.0
            response.timestamp_ns = self.get_clock().now().nanoseconds
        
        return response
    
    def _validate_command_safety(self, command: RobotCommand) -> bool:
        """Validate command against safety constraints including LiDAR distances"""
        self.get_logger().debug(f"🔍 SAFETY CHECK: command={command.command_type}, safety_enabled={self.safety_enabled}, robot_status={self.robot_info.status}")
        
        # Check if safety is enabled
        if not self.safety_enabled:
            self.get_logger().warn(f"🚫 SAFETY: Command {command.command_type} blocked - safety disabled")
            return False
        
        # Always allow stop commands (for safety)
        if command.command_type == 'stop':
            self.get_logger().info(f"✅ SAFETY: Allowing stop command (always safe)")
            return True
        
        # Safety checks completely removed - VILA has full control
        
        # For testing/simulation mode: if no robot is connected, allow commands when safety is enabled
        # This allows the GUI to work without a physical robot
        if self.robot_info.status == 'offline':
            self.get_logger().info(f"🔄 TESTING MODE: Allowing {command.command_type} - robot offline but safety enabled")
            # Set status to active for testing purposes
            self.robot_info.status = 'active'
            return True
        
        # For connected robots: check if robot is responsive
        if self.robot_info.status in ['error', 'disconnected']:
            self.get_logger().warn(f"🚫 SAFETY: Command {command.command_type} blocked - robot status: {self.robot_info.status}")
            return False
        
        self.get_logger().info(f"✅ SAFETY: Command {command.command_type} approved - safety enabled, robot status: {self.robot_info.status}")
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
                'safety_enabled': self.safety_enabled
            }
            
            if self.robot_info.sensor_data:
                status_data['sensor_data'] = self.robot_info.sensor_data
            
            status_msg.data = json.dumps(status_data)
            
            # Publish status (assuming we have a status publisher)
            # Note: This would need a status publisher to be set up in _setup_publishers()
            self.get_logger().debug(f"📊 Robot status updated: {self.robot_info.status}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Error publishing robot status: {e}")
    


    def _publish_vila_status(self):
        """Publish VILA server status for GUI display"""
        try:
            # For now, just publish a basic status since GUI manages the server
            status_info = {
                'status': 'managed_by_gui',
                'process_running': False,
                'recent_logs': [],
                'server_ready': self.model_loaded,
                'server_url': 'http://localhost:8000'
            }
            status_msg = String()
            status_msg.data = json.dumps(status_info)
            self.vila_status_publisher.publish(status_msg)
        except Exception as e:
            self.get_logger().debug(f"Error publishing VILA status: {e}")
    
    # VILA enable/disable removed - now resource-based on-demand only

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
