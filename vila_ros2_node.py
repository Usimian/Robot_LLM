#!/usr/bin/env python3
"""
VILA ROS2 Node
Integrates VILA vision-language model with ROS2 for robotic applications
Replaces ROS1 implementation with proper ROS2 topics and services
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import sys
import os
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import numpy as np
import threading
import json
import time

# Custom ROS2 messages
from robot_msgs.msg import RobotCommand, SensorData, VILAAnalysis, RobotStatus
from robot_msgs.srv import ExecuteCommand, RequestVILAAnalysis

# Add VILA paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VILA'))
from main_vila import VILAModel

class VILARos2Node(Node):
    """
    ROS2 VILA Node
    Processes camera images with VILA and publishes analysis results
    Maintains single command gateway architecture [[memory:5366669]]
    """
    
    def __init__(self):
        super().__init__('vila_vision_node')
        
        # Initialize VILA model
        self.vila_model = VILAModel()
        self.model_loaded = False
        self.bridge = CvBridge()
        
        # Robot configuration
        self.robot_id = "yahboomcar_x3_01"  # Hardcoded for single robot system
        
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
        self._setup_clients()
        
        # Load model
        self._load_model_async()
        
        # Status publishing timer
        self.status_timer = self.create_timer(1.0, self._publish_status)
        
        self.get_logger().info("🤖 VILA ROS2 Node initialized")
        self.get_logger().info("   └── Using ROS2 topics and services")
        self.get_logger().info("   └── Commands routed through single gateway [[memory:5366669]]")
    
    def _setup_publishers(self):
        """Setup ROS2 publishers"""
        # VILA analysis results
        self.analysis_pub = self.create_publisher(
            VILAAnalysis, 
            '/vila/analysis', 
            self.reliable_qos
        )
        
        # Navigation commands (parsed from VILA)
        self.navigation_pub = self.create_publisher(
            String, 
            '/vila/navigation_command', 
            self.reliable_qos
        )
        
        # Status publishing
        self.status_pub = self.create_publisher(
            Bool, 
            '/vila/status', 
            self.reliable_qos
        )
        
        # 🛡️ SAFETY: Direct cmd_vel publishing is DISABLED
        # All commands must go through the single gateway
        # self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().warn("🚫 SAFETY: Direct /cmd_vel publishing DISABLED")
        self.get_logger().warn("   └── All commands routed through single gateway")
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers"""
        # Camera image subscription
        self.image_sub = self.create_subscription(
            Image,
            f'/robot/{self.robot_id}/camera/image_raw',
            self._image_callback,
            self.best_effort_qos
        )
        
        # Custom query subscription
        self.query_sub = self.create_subscription(
            String,
            '/vila/query',
            self._query_callback,
            self.reliable_qos
        )
        
        # Command acknowledgments from robot
        self.command_ack_sub = self.create_subscription(
            RobotCommand,
            f'/robot/{self.robot_id}/command_ack',
            self._command_ack_callback,
            self.reliable_qos
        )
    
    def _setup_services(self):
        """Setup ROS2 services"""
        # VILA analysis service
        self.analysis_service = self.create_service(
            RequestVILAAnalysis,
            '/vila/analyze_image',
            self._analyze_image_service
        )
    
    def _setup_clients(self):
        """Setup ROS2 service clients"""
        # Command execution client (connects to single gateway)
        self.execute_command_client = self.create_client(
            ExecuteCommand,
            '/robot/execute_command'
        )
        
        # Wait for service
        while not self.execute_command_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for execute_command service...')
    
    def _load_model_async(self):
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
    
    def _image_callback(self, msg: Image):
        """Process incoming camera images with VILA"""
        if not self.model_loaded:
            return
        
        try:
            # Convert ROS2 image to PIL
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # Process in background thread to avoid blocking
            threading.Thread(
                target=self._process_image_with_vila,
                args=(msg, pil_image),
                daemon=True
            ).start()
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def _process_image_with_vila(self, image_msg: Image, pil_image: PILImage):
        """Process image with VILA model"""
        try:
            # Navigation-focused prompt
            navigation_prompt = """You are a robot's vision system. Analyze this camera view and provide:
1. Can I move forward safely?
2. Are there obstacles ahead?
3. What should I do next (move_forward, turn_left, turn_right, stop)?
4. Describe what you see briefly.
Keep it concise for real-time navigation."""
            
            # Generate VILA response
            response = self.vila_model.generate_response(
                prompt=navigation_prompt,
                image=pil_image
            )
            
            # Parse navigation commands
            nav_commands = self._parse_navigation_commands(response)
            
            # Create VILA analysis message
            analysis_msg = VILAAnalysis()
            analysis_msg.robot_id = self.robot_id
            analysis_msg.prompt = navigation_prompt
            analysis_msg.image = image_msg
            analysis_msg.analysis_result = response
            analysis_msg.navigation_commands_json = json.dumps(nav_commands)
            analysis_msg.confidence = nav_commands.get('confidence', 0.0)
            analysis_msg.timestamp_ns = self.get_clock().now().nanoseconds
            analysis_msg.success = True
            analysis_msg.error_message = ""
            
            # Publish analysis
            self.analysis_pub.publish(analysis_msg)
            
            # Publish navigation commands separately
            nav_msg = String()
            nav_msg.data = json.dumps(nav_commands)
            self.navigation_pub.publish(nav_msg)
            
            # 🛡️ SAFETY: Route commands through single gateway instead of direct control
            if nav_commands['action'] != 'stop' and nav_commands['confidence'] > 0.7:
                self._send_command_through_gateway(nav_commands)
            
            self.get_logger().debug(f"✅ VILA analysis: {nav_commands['action']} (confidence: {nav_commands['confidence']:.2f})")
            
        except Exception as e:
            self.get_logger().error(f"❌ Error in VILA processing: {e}")
    
    def _parse_navigation_commands(self, response: str) -> dict:
        """Parse VILA response for navigation commands"""
        response_lower = response.lower()
        
        commands = {
            'action': 'stop',  # default safe action
            'confidence': 0.0,
            'reason': response,
            'speed': 0.2,      # default speed
            'duration': 1.0    # default duration
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
    
    def _send_command_through_gateway(self, nav_commands: dict):
        """
        🎯 Send commands through single gateway [[memory:5366669]]
        
        This ensures all robot commands go through proper safety validation
        """
        try:
            # Create command message
            command_msg = RobotCommand()
            command_msg.robot_id = self.robot_id
            command_msg.command_type = nav_commands['action']
            command_msg.parameters_json = json.dumps({
                'speed': nav_commands.get('speed', 0.2),
                'duration': nav_commands.get('duration', 1.0)
            })
            command_msg.timestamp_ns = self.get_clock().now().nanoseconds
            command_msg.priority = 1
            command_msg.safety_confirmed = False  # Let gateway decide
            command_msg.gui_movement_enabled = False  # VILA-generated command
            command_msg.source = "VILA_VISION"
            
            # Create service request
            request = ExecuteCommand.Request()
            request.command = command_msg
            
            # Call service asynchronously
            future = self.execute_command_client.call_async(request)
            
            # Add callback for response
            future.add_done_callback(self._command_response_callback)
            
            self.get_logger().info(f"🎯 VILA command sent through gateway: {nav_commands['action']}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Error sending command through gateway: {e}")
    
    def _command_response_callback(self, future):
        """Handle response from command gateway"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().debug(f"✅ Gateway accepted VILA command: {response.message}")
            else:
                self.get_logger().warn(f"🚫 Gateway rejected VILA command: {response.message}")
        except Exception as e:
            self.get_logger().error(f"❌ Error in command response: {e}")
    
    def _query_callback(self, msg: String):
        """Handle custom queries from other ROS2 nodes"""
        if not self.model_loaded:
            self.get_logger().warn("VILA model not loaded, ignoring query")
            return
        
        self.get_logger().info(f"📝 Received VILA query: {msg.data}")
        
        # Process query in background
        threading.Thread(
            target=self._process_custom_query,
            args=(msg.data,),
            daemon=True
        ).start()
    
    def _process_custom_query(self, query: str):
        """Process custom query with VILA"""
        try:
            # For custom queries, we'd need the latest camera image
            # This is a simplified implementation
            self.get_logger().info(f"Processing custom query: {query}")
            
            # Would need to implement image buffering for this to work properly
            # For now, just acknowledge
            response = f"Query received: {query}. Image processing would happen here."
            
            # Publish response
            response_msg = String()
            response_msg.data = response
            self.navigation_pub.publish(response_msg)
            
        except Exception as e:
            self.get_logger().error(f"❌ Error processing custom query: {e}")
    
    def _command_ack_callback(self, msg: RobotCommand):
        """Handle command acknowledgments from robot"""
        self.get_logger().debug(f"📨 Command ACK: {msg.command_type} executed by robot")
    
    def _analyze_image_service(self, request, response):
        """Handle VILA analysis service requests"""
        try:
            if not self.model_loaded:
                response.success = False
                response.error_message = "VILA model not loaded"
                response.timestamp_ns = self.get_clock().now().nanoseconds
                return response
            
            self.get_logger().info(f"🔍 VILA analysis service called for {request.robot_id}")
            
            # Convert ROS2 image to PIL
            cv_image = self.bridge.imgmsg_to_cv2(request.image, "bgr8")
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # Generate VILA response
            analysis_result = self.vila_model.generate_response(
                prompt=request.prompt,
                image=pil_image
            )
            
            # Parse navigation commands
            nav_commands = self._parse_navigation_commands(analysis_result)
            
            # Fill response
            response.success = True
            response.analysis_result = analysis_result
            response.navigation_commands_json = json.dumps(nav_commands)
            response.confidence = nav_commands.get('confidence', 0.0)
            response.error_message = ""
            response.timestamp_ns = self.get_clock().now().nanoseconds
            
            self.get_logger().info("✅ VILA analysis service completed")
            
        except Exception as e:
            self.get_logger().error(f"❌ VILA analysis service error: {e}")
            response.success = False
            response.error_message = str(e)
            response.timestamp_ns = self.get_clock().now().nanoseconds
        
        return response
    
    def _publish_status(self):
        """Publish VILA node status"""
        status_msg = Bool()
        status_msg.data = self.model_loaded
        self.status_pub.publish(status_msg)
    
    def send_velocity_commands_disabled(self, commands):
        """
        🚫 DISABLED: Direct velocity commands bypass safety system
        
        This method is disabled to enforce single command gateway architecture.
        All robot commands must go through the gateway for proper safety validation.
        """
        self.get_logger().warn("🚫 DISABLED: send_velocity_commands() called but blocked for safety")
        self.get_logger().warn(f"   └── Commands that would have been sent: {commands}")
        self.get_logger().warn("   └── Use single gateway for ALL robot commands [[memory:5366669]]")
        self.get_logger().warn("   └── This ensures proper safety validation and single point of control")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VILARos2Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
