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
        self.vila_model = VILAModel(auto_start_server=False)
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
    
    def _process_image_with_vila_sync(self, image_msg: Image, prompt: str, lidar_data: Dict[str, float] = None) -> str:
        """Process image with VILA model synchronously for on-demand requests with LiDAR integration"""
        try:
            from cv_bridge import CvBridge
            from PIL import Image as PILImage
            import cv2
            
            # Create pure visual analysis prompt (LiDAR handled separately for late fusion)
            if not prompt:
                prompt = """You are a robot's vision-language navigation system. Focus ONLY on visual analysis of the camera image:

🔍 VISUAL ANALYSIS TASK:
1. What do you see in this camera view?
2. Are there visible obstacles, walls, or objects in the path ahead?
3. What does the visual scene suggest about safe navigation directions?
4. Based on VISUAL INFORMATION ONLY, what navigation action would you recommend?

🎯 OUTPUT FORMAT:
Action: [move_forward/turn_left/turn_right/stop]
Reason: [Brief explanation based purely on visual analysis]

Focus on visual semantics - spatial safety will be handled by separate LiDAR systems."""
            else:
                # User provided custom prompt - keep it pure visual, no LiDAR integration
                prompt = f"""🔍 VISUAL ANALYSIS: {prompt}

Focus on visual analysis only. Provide:
Action: [move_forward/turn_left/turn_right/stop]
Reason: [Brief visual-based explanation]

Spatial safety constraints will be handled separately."""
            
            # Check if image has valid data and encoding
            if not image_msg.data:
                self.get_logger().error("❌ Empty image data for VILA processing")
                return "Error: Empty image data"
            
            # Check encoding - default to bgr8 if empty
            encoding = image_msg.encoding if image_msg.encoding else "bgr8"
            self.get_logger().debug(f"Processing image with encoding: {encoding}, size: {len(image_msg.data)} bytes")
            
            # Convert ROS image to PIL
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(image_msg, encoding)
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # Generate VILA response
            vila_response = self.vila_model.generate_response(
                prompt=prompt,
                image=pil_image
            )
            
            # Extract the analysis text from the response dictionary
            if isinstance(vila_response, dict) and vila_response.get('success', False):
                response = vila_response.get('analysis', 'No analysis available')
            else:
                response = str(vila_response)
            
            self.get_logger().debug(f"✅ VILA analysis complete for prompt: {prompt[:50]}...")
            
            return response
            
        except Exception as e:
            self.get_logger().error(f"❌ Error processing image with VILA: {e}")
            return f"Error during analysis: {str(e)}"
    
    def _parse_navigation_commands(self, response: str, lidar_data: Dict[str, float] = None) -> Dict[str, Any]:
        """Parse VILA response for navigation commands with LiDAR-enhanced safety"""
        response_lower = response.lower()
        
        commands = {
            'action': 'stop',  # default safe action
            'confidence': 0.0,
            'reason': response,
            'lidar_safety': 'unknown'
        }
        
        # Emergency LiDAR safety override (only for critical situations where VILA might have missed something)
        if lidar_data:
            front_dist = lidar_data['distance_front']
            left_dist = lidar_data['distance_left'] 
            right_dist = lidar_data['distance_right']
            
            # SAFETY OVERRIDE: Override if front obstacle too close for forward movement
            if front_dist < 0.5:  # 50cm safety threshold for forward movement
                # Check if VILA suggested forward movement despite close obstacle
                if 'move_forward' in response_lower or ('Action:' in response and 'move_forward' in response.lower()):
                    self.get_logger().warn(f"🚨 SAFETY OVERRIDE TRIGGERED: Front obstacle at {front_dist:.2f}m, VILA suggested forward movement")
                    commands.update({
                        'action': 'stop',
                        'confidence': 0.95,
                        'reason': f"SAFETY OVERRIDE: Front obstacle at {front_dist:.2f}m too close for forward movement. VILA suggested forward but overridden for safety. Original: " + response[:100] + "...",
                        'lidar_safety': 'safety_override'
                    })
                    return commands
            
            # EMERGENCY ONLY: Override if front obstacle extremely close
            if front_dist < 0.3:  # 30cm emergency threshold 
                commands.update({
                    'action': 'stop',
                    'confidence': 0.98,
                    'reason': f"EMERGENCY OVERRIDE: Front obstacle at {front_dist:.2f}m < 0.3m. VILA response was: " + response[:100] + "...",
                    'lidar_safety': 'emergency_stop'
                })
                return commands
            
            # Store LiDAR context for logging
            commands['lidar_safety'] = f'F:{front_dist:.1f}m_L:{left_dist:.1f}m_R:{right_dist:.1f}m'
            
            # Parse VILA response - VILA now has LiDAR data and should make informed decisions
            # Look for the structured output format we requested
            if 'Action:' in response:
                # Parse structured response
                action_line = [line.strip() for line in response.split('\n') if line.strip().startswith('Action:')]
                if action_line:
                    action_text = action_line[0].replace('Action:', '').strip().lower()
                    if 'move_forward' in action_text:
                        commands.update({'action': 'move_forward', 'confidence': 0.9})
                    elif 'turn_left' in action_text:
                        commands.update({'action': 'turn_left', 'confidence': 0.9})
                    elif 'turn_right' in action_text:
                        commands.update({'action': 'turn_right', 'confidence': 0.9})
                    elif 'stop' in action_text:
                        commands.update({'action': 'stop', 'confidence': 0.9})
            else:
                # Fallback to keyword parsing if structured format not used
                if 'move forward' in response_lower or 'move_forward' in response_lower:
                    commands.update({'action': 'move_forward', 'confidence': 0.8})
                elif 'turn left' in response_lower or 'turn_left' in response_lower:
                    commands.update({'action': 'turn_left', 'confidence': 0.8})
                elif 'turn right' in response_lower or 'turn_right' in response_lower:
                    commands.update({'action': 'turn_right', 'confidence': 0.8})
                elif 'stop' in response_lower:
                    commands.update({'action': 'stop', 'confidence': 0.9})
                else:
                    # VILA response unclear - default to stop for safety
                    commands.update({
                        'action': 'stop',
                        'confidence': 0.6,
                        'reason': f"UNCLEAR VILA RESPONSE: Defaulting to stop for safety. " + response
                    })
            
            # Update reason with LiDAR context
            commands['reason'] = f"VILA+LiDAR Decision (F:{front_dist:.2f}m, L:{left_dist:.2f}m, R:{right_dist:.2f}m): " + response
            
            # FINAL SAFETY CHECK: Override dangerous decisions after parsing
            if front_dist < 0.5 and commands['action'] == 'move_forward':
                self.get_logger().warn(f"🚨 FINAL SAFETY OVERRIDE: Blocking move_forward with front obstacle at {front_dist:.2f}m")
                commands.update({
                    'action': 'stop',
                    'confidence': 0.98,
                    'reason': f"FINAL SAFETY OVERRIDE: Front obstacle at {front_dist:.2f}m too close for forward movement. VILA+parsing suggested forward but overridden. Original decision: {commands['reason']}",
                    'lidar_safety': 'final_safety_override'
                })
                    
        else:
            # Fallback to original parsing when no LiDAR data
            if 'move forward' in response_lower or ('clear' in response_lower and 'safe' in response_lower):
                commands.update({'action': 'move_forward', 'confidence': 0.8})
            elif 'turn left' in response_lower:
                commands.update({'action': 'turn_left', 'confidence': 0.7})
            elif 'turn right' in response_lower:
                commands.update({'action': 'turn_right', 'confidence': 0.7})
            elif 'stop' in response_lower or 'obstacle' in response_lower:
                commands.update({'action': 'stop', 'confidence': 0.9})
            
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
            self.get_logger().info(f"🔍 RESPONSE SENT: success=True, message='{response.result_message}' (ID: {request_id})")
            
        except Exception as e:
            self.get_logger().error(f"❌ Command execution error: {e}")
            response.success = False
            response.result_message = f"Error: {str(e)}"
        
        return response
    
    def _vila_analysis_service(self, request, response):
        """Handle VILA analysis requests - on-demand resource-based processing"""
        try:
            if not self.model_loaded:
                response.success = False
                response.error_message = "VILA model not loaded"
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
            
            self.get_logger().info(f"🔍 On-demand VILA analysis requested for {request.robot_id}")
            self.get_logger().info(f"   └── Prompt: {request.prompt}")
            self.get_logger().info(f"   └── Image: {request.image.width}x{request.image.height}, encoding: {request.image.encoding}")
            self.get_logger().info(f"   └── LiDAR: Front={request.distance_front:.2f}m, Left={request.distance_left:.2f}m, Right={request.distance_right:.2f}m")
            
            # Use the image provided by the client
            image_to_analyze = request.image
            
            # Extract LiDAR distances for enhanced analysis
            lidar_data = {
                'distance_front': request.distance_front,
                'distance_left': request.distance_left,
                'distance_right': request.distance_right
            }
            
            # Process with VILA using the client-provided image and LiDAR data
            analysis_result = self._process_image_with_vila_sync(image_to_analyze, request.prompt, lidar_data)
            nav_commands = self._parse_navigation_commands(analysis_result, lidar_data)
            
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
        self.get_logger().info(f"🔍 SAFETY CHECK: command={command.command_type}, safety_enabled={self.safety_enabled}, robot_status={self.robot_info.status}")
        
        # Check if safety is enabled
        if not self.safety_enabled:
            self.get_logger().warn(f"🚫 SAFETY: Command {command.command_type} blocked - safety disabled")
            return False
        
        # Always allow stop commands (for safety)
        if command.command_type == 'stop':
            self.get_logger().info(f"✅ SAFETY: Allowing stop command (always safe)")
            return True
        
        # LiDAR-based safety validation for movement commands
        if self.robot_info.sensor_data:
            front_dist = self.robot_info.sensor_data.get('distance_front', float('inf'))
            left_dist = self.robot_info.sensor_data.get('distance_left', float('inf'))
            right_dist = self.robot_info.sensor_data.get('distance_right', float('inf'))
            
            # Critical safety check: Block forward movement if front obstacle too close
            if command.command_type == 'move' and command.linear_x > 0:  # Forward movement
                if front_dist < 0.25:  # 25cm critical safety threshold
                    self.get_logger().warn(f"🚫 LIDAR SAFETY: Forward movement blocked - front obstacle at {front_dist:.2f}m < 0.25m")
                    return False
                elif front_dist < 0.4:  # 40cm warning threshold
                    self.get_logger().warn(f"⚠️ LIDAR WARNING: Forward movement with caution - front obstacle at {front_dist:.2f}m")
            
            # Check turn safety
            elif command.command_type == 'turn':
                if command.angular > 0 and left_dist < 0.3:  # Left turn
                    self.get_logger().warn(f"🚫 LIDAR SAFETY: Left turn blocked - obstacle at {left_dist:.2f}m < 0.3m")
                    return False
                elif command.angular < 0 and right_dist < 0.3:  # Right turn
                    self.get_logger().warn(f"🚫 LIDAR SAFETY: Right turn blocked - obstacle at {right_dist:.2f}m < 0.3m")
                    return False
            
            self.get_logger().info(f"✅ LIDAR SAFETY: Command {command.command_type} approved (F:{front_dist:.2f}m, L:{left_dist:.2f}m, R:{right_dist:.2f}m)")
        
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
