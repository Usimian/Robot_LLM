#!/usr/bin/env python3
"""
Robot Control GUI - ROS2 Version
Tkinter application for monitoring and controlling robots via ROS2 topics and services
Replaces HTTP/WebSocket communication with ROS2 messaging
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import json
import threading
import time
from datetime import datetime
import base64
import io
from typing import List
from typing import Dict, List, Optional
import logging
import queue
from dataclasses import dataclass
from PIL import Image, ImageTk
import cv2
import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# ROS2 message imports
from robot_msgs.msg import RobotCommand, SensorData, VILAAnalysis
from robot_msgs.srv import ExecuteCommand, RequestVILAAnalysis
from sensor_msgs.msg import Image as RosImage, Imu
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RobotGUIROS2')

class RobotGUIROS2Node(Node):
    """ROS2 node for the robot GUI"""
    
    def __init__(self, gui_callback):
        super().__init__('robot_gui')
        
        self.gui_callback = gui_callback
        self.robot_id = "yahboomcar_x3_01"  # Hardcoded for single robot system
        self.bridge = CvBridge()
        self.last_vila_update = None  # Track VILA model activity
        self.latest_imu_data = None  # Store latest IMU data
        self.current_camera_image = None  # Store current camera image for VILA requests
        self.current_sensor_data = None  # Store current sensor data including LiDAR distances
        
        # QoS profiles
        # QoS for image streams (FIXED: Match robot publisher QoS to prevent frame drops)
        self.image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,        # Match robot publisher
            durability=DurabilityPolicy.TRANSIENT_LOCAL,   # Match robot publisher
            history=HistoryPolicy.KEEP_LAST,
            depth=10  # Reasonable buffer for network transmission
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
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_service_clients()
        
        self.get_logger().info("🖥️ Robot GUI ROS2 node initialized")
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers"""

        
        # Sensor data
        self.sensor_subscriber = self.create_subscription(
            SensorData,
            '/robot/sensors',
            self._sensor_data_callback,
            self.best_effort_qos
        )
        
        # Camera images
        self.image_subscriber = self.create_subscription(
            RosImage,
            '/realsense/camera/color/image_raw',
            self._image_callback,
            self.image_qos
        )
        self.get_logger().info(f"📸 Created camera subscription with QoS: reliability={self.image_qos.reliability}, durability={self.image_qos.durability}")
        
        # VILA analysis results
        self.analysis_subscriber = self.create_subscription(
            VILAAnalysis,
            '/vila/analysis',
            self._vila_analysis_callback,
            self.reliable_qos
        )
        
        # Navigation commands
        self.navigation_subscriber = self.create_subscription(
            String,
            '/robot/navigation_commands',
            self._navigation_commands_callback,
            self.best_effort_qos
        )
        
        # IMU data for acceleration display
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data_raw',
            self._imu_callback,
            self.best_effort_qos
        )
        
        # Safety status
        self.safety_subscriber = self.create_subscription(
            Bool,
            '/robot/safety/enabled',
            self._safety_status_callback,
            self.reliable_qos
        )
        
        # Command acknowledgments
        self.command_ack_subscriber = self.create_subscription(
            RobotCommand,
            '/robot/command_ack',
            self._command_ack_callback,
            self.reliable_qos
        )
    
    def _setup_publishers(self):
        """Setup ROS2 publishers"""
        # Safety control
        self.safety_control_publisher = self.create_publisher(
            Bool,
            '/robot/safety/enable_movement',
            self.reliable_qos
        )
        
        # Emergency stop
        self.emergency_stop_publisher = self.create_publisher(
            Bool,
            '/robot/emergency_stop',
            self.reliable_qos
        )
        
        # VILA queries
        self.vila_query_publisher = self.create_publisher(
            String,
            '/vila/query',
            self.reliable_qos
        )
        
        # Note: VILA enable/disable removed - now resource-based on-demand only
    
    def _setup_service_clients(self):
        """Setup ROS2 service clients"""
        # Command execution service (single gateway [[memory:5366669]])
        self.execute_command_client = self.create_client(
            ExecuteCommand,
            '/robot/execute_command'
        )
        
        # VILA analysis service
        self.vila_analysis_client = self.create_client(
            RequestVILAAnalysis,
            '/vila/request_analysis'
        )
        
        # VILA server status subscriber
        self.vila_status_subscriber = self.create_subscription(
            String,
            '/vila/server_status',
            self._vila_status_callback,
            self.reliable_qos
        )
    

    
    def _sensor_data_callback(self, msg: SensorData):
        """Handle sensor data updates"""
        try:
            sensor_data = {
                'robot_id': msg.robot_id,
                'battery_voltage': msg.battery_voltage,
                'cpu_temp': msg.cpu_temp,
                'distance_front': msg.distance_front,
                'distance_left': msg.distance_left,
                'distance_right': msg.distance_right,
                'cpu_usage': msg.cpu_usage,
                'camera_status': msg.camera_status,
                'timestamp': msg.timestamp_ns
            }
            
            # Store current sensor data for VILA analysis requests
            self.current_sensor_data = sensor_data
            
            # Send to GUI
            self.gui_callback('sensor_data', sensor_data)
            
        except Exception as e:
            self.get_logger().error(f"Error processing sensor data: {e}")
    
    def _imu_callback(self, msg: Imu):
        """Handle IMU data updates"""
        try:
            # Store latest IMU data with acceleration values
            self.latest_imu_data = {
                'imu_accel_x': msg.linear_acceleration.x,
                'imu_accel_y': msg.linear_acceleration.y,
                'imu_accel_z': msg.linear_acceleration.z,
                'timestamp': msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
            }
            
            # Send to GUI for display
            self.gui_callback('imu_data', self.latest_imu_data)
            
        except Exception as e:
            self.get_logger().error(f"Error processing IMU data: {e}")
    
    def _image_callback(self, msg: RosImage):
        """Handle camera image updates"""
        try:
            current_time = time.time()
            if not hasattr(self, '_last_image_log_time'):
                self._last_image_log_time = 0
            
            # Store the ROS image for VILA analysis requests
            self.current_camera_image = msg
            
            # Log every second to track update frequency
            if current_time - self._last_image_log_time >= 1.0:
                self.get_logger().debug(f"📸 Received camera image: {msg.width}x{msg.height}, encoding: {msg.encoding}")
                self._last_image_log_time = current_time
            
            # Check encoding - default to bgr8 if empty, but handle rgb8 from robot
            encoding = msg.encoding if msg.encoding else "bgr8"
            
            # Convert ROS image to OpenCV
            if encoding == "rgb8":
                cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                # Convert RGB to BGR for OpenCV
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Convert to PIL Image for GUI
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Send to GUI
            self.gui_callback('camera_image', pil_image)
            self.get_logger().debug("📸 Camera image sent to GUI")
            
        except Exception as e:
            self.get_logger().error(f"Error processing camera image: {e}")
    
    def _vila_analysis_callback(self, msg: VILAAnalysis):
        """Handle VILA analysis results (from published messages)"""
        try:
            # Track VILA activity for model state display
            self.last_vila_update = time.time()
            
            # NOTE: We only process this for status tracking, not display
            # Display is handled by service response to avoid duplicates
            self.get_logger().debug(f"VILA analysis published: {msg.analysis_result[:50]}...")
            
        except Exception as e:
            self.get_logger().error(f"Error processing VILA analysis: {e}")
    
    def _navigation_commands_callback(self, msg: String):
        """Handle navigation commands"""
        try:
            nav_commands = json.loads(msg.data) if msg.data else {}
            self.gui_callback('navigation_commands', nav_commands)
        except Exception as e:
            self.get_logger().error(f"Error processing navigation commands: {e}")
    
    def _safety_status_callback(self, msg: Bool):
        """Handle safety status updates"""
        self.gui_callback('safety_status', msg.data)
    
    def _command_ack_callback(self, msg: RobotCommand):
        """Handle command acknowledgments"""
        try:
            ack_data = {
                'robot_id': msg.robot_id,
                'command_type': msg.command_type,
                'linear_x': msg.linear_x,
                'linear_y': msg.linear_y,
                'angular_z': msg.angular_z,
                'duration': msg.duration,
                'distance': msg.distance,
                'angular': msg.angular,
                'linear_speed': msg.linear_speed,
                'angular_speed': msg.angular_speed,
                'timestamp': msg.timestamp_ns,
                'source_node': msg.source_node
            }
            
            self.gui_callback('command_ack', ack_data)
            
        except Exception as e:
            self.get_logger().error(f"Error processing command ack: {e}")
    
    def send_robot_command(self, command_type: str, parameters: Dict = None, safety_confirmed: bool = False):
        """Send robot command through single gateway [[memory:5366669]]"""
        try:
            # CRITICAL FIX: Create fresh service client for each call to avoid cached responses
            # This is a workaround for ROS2 service client caching issues
            from robot_msgs.srv import ExecuteCommand
            fresh_client = self.create_client(ExecuteCommand, '/robot/execute_command')
            
            if not fresh_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error("Execute command service not available")
                fresh_client.destroy()
                return False
            
            # Create command message
            command_msg = RobotCommand()
            command_msg.robot_id = self.robot_id
            command_msg.command_type = command_type
            
            # Map GUI commands to robot's proper command structure per CLIENT_INTEGRATION_GUIDE.md
            if command_type == "move_forward":
                command_msg.command_type = "move"
                command_msg.linear_x = 1.0  # Forward direction (normalized)
                command_msg.linear_y = 0.0  # No strafe
                command_msg.angular_z = 0.0  # Not used (legacy field)
                command_msg.distance = parameters.get('distance', 0.2)  # Default 20cm as per guide
                command_msg.linear_speed = parameters.get('speed', 0.1)  # Default 0.1 m/s as per guide
                command_msg.duration = parameters.get('duration', 10.0)  # Optional timeout
            elif command_type == "move_backward":
                command_msg.command_type = "move"
                command_msg.linear_x = -1.0  # Backward direction (normalized)
                command_msg.linear_y = 0.0  # No strafe
                command_msg.angular_z = 0.0  # Not used (legacy field)
                command_msg.distance = parameters.get('distance', 0.1)  # Default 10cm as per guide
                command_msg.linear_speed = parameters.get('speed', 0.1)  # Default 0.1 m/s as per guide
                command_msg.duration = parameters.get('duration', 10.0)  # Optional timeout
            elif command_type == "turn_left":
                command_msg.command_type = "turn"
                command_msg.linear_x = 0.0  # No forward movement
                command_msg.linear_y = 0.0  # No strafe
                command_msg.angular_z = 0.0  # Not used (legacy field)
                command_msg.angular = parameters.get('angle', 90.0)  # Default 90 degrees as per guide
                command_msg.angular_speed = parameters.get('speed', 0.5)  # Default 0.5 rad/s as per guide
                command_msg.duration = parameters.get('duration', 8.0)  # Optional timeout
            elif command_type == "turn_right":
                command_msg.command_type = "turn"
                command_msg.linear_x = 0.0  # No forward movement
                command_msg.linear_y = 0.0  # No strafe
                command_msg.angular_z = 0.0  # Not used (legacy field)
                command_msg.angular = parameters.get('angle', -90.0)  # Default -90 degrees as per guide
                command_msg.angular_speed = parameters.get('speed', 0.5)  # Default 0.5 rad/s as per guide
                command_msg.duration = parameters.get('duration', 8.0)  # Optional timeout
            elif command_type == "stop":
                command_msg.command_type = "stop"
                command_msg.linear_x = 0.0
                command_msg.linear_y = 0.0
                command_msg.angular_z = 0.0
                command_msg.distance = 0.0
                command_msg.angular = 0.0
                command_msg.linear_speed = 0.0
                command_msg.angular_speed = 0.0
                command_msg.duration = 0.0
            
            command_msg.timestamp_ns = self.get_clock().now().nanoseconds
            command_msg.source_node = "robot_gui_node"
            
            # Create service request
            request = ExecuteCommand.Request()
            request.command = command_msg
            
            # Call service with timestamp for debugging
            import time
            request_id = int(time.time() * 1000) % 10000  # Short unique ID
            
            self.get_logger().info(f"🎯 GUI command sent through gateway: {command_type} (ID: {request_id})")
            
            # CRITICAL FIX: Use synchronous call to avoid cached response issues
            # The async callbacks were receiving stale cached responses from previous failed calls
            try:
                self.get_logger().info(f"🔄 Making synchronous service call for {command_type} (ID: {request_id})")
                
                # Use fresh client to get real response (not cached)
                response = fresh_client.call(request)
                
                # Clean up the fresh client
                fresh_client.destroy()
                
                if response is not None:
                    self.get_logger().info(f"🔍 Direct response for {command_type} (ID: {request_id}): success={response.success}, message='{response.result_message}'")
                    
                    if response.success:
                        self.get_logger().info(f"✅ Gateway accepted GUI command: {command_type} (ID: {request_id})")
                        self.gui_callback('command_success', {'command': command_type, 'message': response.result_message})
                    else:
                        self.get_logger().warn(f"🚫 Gateway rejected GUI command: {command_type} (ID: {request_id}) - {response.result_message}")
                        self.gui_callback('command_error', {'command': command_type, 'message': response.result_message})
                else:
                    self.get_logger().error(f"❌ Service call timeout for {command_type} (ID: {request_id})")
                    self.gui_callback('command_error', {'command': command_type, 'message': "Service call timeout"})
                    
            except Exception as call_e:
                self.get_logger().error(f"❌ Service call exception for {command_type} (ID: {request_id}): {call_e}")
                self.gui_callback('command_error', {'command': command_type, 'message': f"Service call error: {str(call_e)}"})
                # Clean up fresh client on exception
                try:
                    fresh_client.destroy()
                except:
                    pass
                
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error sending robot command: {e}")
            return False
    
    # REMOVED: _command_response_callback - now using synchronous service calls to avoid stale response cache issues
    
    def request_vila_analysis(self, prompt: str, image: RosImage = None):
        """Request VILA analysis - always sends current camera image to VILA server"""
        try:
            # Check service availability with longer timeout
            self.get_logger().debug("🔍 Checking VILA analysis service availability...")
            if not self.vila_analysis_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error("VILA analysis service not available after 5 seconds")
                self.get_logger().error("Make sure the VILA server node is running")
                return False
            
            # Use provided image or current camera image
            image_to_send = image if image else self.current_camera_image
            
            if not image_to_send:
                self.get_logger().error("No camera image available for VILA analysis")
                self.get_logger().error("Waiting for camera images from /realsense/camera/color/image_raw...")
                # Try to wait a bit for an image
                import time
                for i in range(10):  # Wait up to 5 seconds
                    if self.current_camera_image:
                        image_to_send = self.current_camera_image
                        self.get_logger().info("✅ Camera image received, proceeding with analysis")
                        break
                    time.sleep(0.5)
                
                if not image_to_send:
                    self.get_logger().error("❌ Still no camera image available after waiting")
                    return False
            
            request = RequestVILAAnalysis.Request()
            request.robot_id = self.robot_id
            request.prompt = prompt
            
            # Include current LiDAR distances - REAL SENSOR DATA ONLY
            if hasattr(self, 'current_sensor_data') and self.current_sensor_data:
                # Only proceed if we have actual sensor data - NO DEFAULT VALUES
                if ('distance_front' in self.current_sensor_data and 
                    'distance_left' in self.current_sensor_data and 
                    'distance_right' in self.current_sensor_data):
                    request.distance_front = self.current_sensor_data['distance_front']
                    request.distance_left = self.current_sensor_data['distance_left']
                    request.distance_right = self.current_sensor_data['distance_right']
                    self.get_logger().debug(f"🔍 Including REAL LiDAR data: F={request.distance_front:.2f}m, L={request.distance_left:.2f}m, R={request.distance_right:.2f}m")
                else:
                    self.get_logger().error("❌ Incomplete sensor data - missing LiDAR distances")
                    return False
            else:
                # NO MOCK DATA - Real sensor data required
                self.get_logger().error("❌ No sensor data available - VILA analysis requires real sensor data")
                return False
            request.image = image_to_send
            
            self.get_logger().debug(f"🔍 Sending VILA analysis request with image: {image_to_send.width}x{image_to_send.height}")
            
            future = self.vila_analysis_client.call_async(request)
            future.add_done_callback(self._vila_analysis_response_callback)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error requesting VILA analysis: {e}")
            return False
    
    def _vila_analysis_response_callback(self, future):
        """Handle VILA analysis response"""
        try:
            response = future.result()
            if response.success:
                analysis_data = {
                    'analysis_result': response.analysis_result,
                    'navigation_commands': json.loads(response.navigation_commands_json) if response.navigation_commands_json else {},
                    'confidence': response.confidence,
                    'timestamp': response.timestamp_ns
                }
                self.gui_callback('vila_analysis_response', analysis_data)
            else:
                self.gui_callback('vila_analysis_error', {'error': response.error_message})
        except Exception as e:
            self.get_logger().error(f"Error in VILA analysis response: {e}")
    
    def set_safety_enabled(self, enabled: bool):
        """Set safety status"""
        try:
            msg = Bool()
            msg.data = enabled
            self.safety_control_publisher.publish(msg)
            self.get_logger().info(f"🛡️ Safety {'ENABLED' if enabled else 'DISABLED'} via GUI")
        except Exception as e:
            self.get_logger().error(f"Error setting safety status: {e}")
    
    def emergency_stop(self):
        """Trigger emergency stop"""
        try:
            msg = Bool()
            msg.data = True
            self.emergency_stop_publisher.publish(msg)
            self.get_logger().warn("🚨 EMERGENCY STOP triggered via GUI")
        except Exception as e:
            self.get_logger().error(f"Error triggering emergency stop: {e}")
    
    def send_vila_query(self, query: str):
        """Send custom VILA query"""
        try:
            msg = String()
            msg.data = query
            self.vila_query_publisher.publish(msg)
            self.get_logger().info(f"📝 VILA query sent: {query}")
        except Exception as e:
            self.get_logger().error(f"Error sending VILA query: {e}")
    
    def _vila_status_callback(self, msg: String):
        """Handle VILA server status updates"""
        try:
            import json
            status_data = json.loads(msg.data)
            self.gui_callback('vila_status', status_data)
        except Exception as e:
            self.get_logger().debug(f"Error parsing VILA status: {e}")

class RobotGUIROS2:
    """Main GUI class using ROS2 communication"""
    
    def __init__(self, root):
        print("🔧 DEBUG: RobotGUIROS2.__init__ entered")
        print("🔧 DEBUG: About to set root")
        self.root = root
        print("🔧 DEBUG: root set")
        print("🔧 DEBUG: About to set base_title")
        self.base_title = "Robot Nemotron Client"
        print("🔧 DEBUG: base_title set")
        # Skip title and geometry for now to avoid segmentation fault
        print("🔧 DEBUG: Tkinter title/geometry SKIPPED to avoid segfault")
        # self.root.title(self.base_title)
        # self.root.geometry("1200x800")
        
        # Initialize ROS2
        print("🔧 DEBUG: Initializing ROS2...")
        rclpy.init()
        print("🔧 DEBUG: ROS2 initialized")
        print("🔧 DEBUG: Creating ROS2 node...")
        self.ros_node = RobotGUIROS2Node(self._ros_callback)
        print("🔧 DEBUG: ROS2 node created successfully")

        # GUI state
        print("🔧 DEBUG: About to set GUI state variables")
        self.robot_id = "yahboomcar_x3_01"
        print("🔧 DEBUG: robot_id set")
        self.robot_data = {}
        print("🔧 DEBUG: robot_data initialized")
        self.sensor_data = {}
        print("🔧 DEBUG: sensor_data initialized")
        self.current_image = None
        print("🔧 DEBUG: current_image set")
        self.loaded_image = None  # For manually loaded images
        print("🔧 DEBUG: loaded_image set")
        self.vila_analysis = {}
        print("🔧 DEBUG: vila_analysis initialized")
        self.vila_server_status = {}  # Track VILA server status
        print("🔧 DEBUG: vila_server_status initialized")
        self.safety_enabled = False  # Start with safety disabled to match checkbox
        print("🔧 DEBUG: safety_enabled initialized")
        self.movement_enabled = False
        print("🔧 DEBUG: movement_enabled initialized")
        # VILA automatic analysis state
        self.vila_auto_analysis = False
        self.exploration_enabled = True  # Enable autonomous exploration by default
        print("🔧 DEBUG: vila_auto_analysis initialized")
        self.vila_auto_interval = 5.0  # seconds
        print("🔧 DEBUG: vila_auto_interval initialized")
        self.vila_last_auto_analysis = 0
        print("🔧 DEBUG: vila_last_auto_analysis initialized")
        self.camera_source = "robot"  # "robot", "sim", or "loaded"
        print("🔧 DEBUG: camera_source initialized")
        self.update_queue = queue.Queue()
        print("🔧 DEBUG: update_queue initialized")

        # Initialize checkbox variables early (will be used later in GUI creation)
        self.vila_auto_var = None  # Will be created in _create_vila_tab()
        print("🔧 DEBUG: vila_auto_var initialized")
        self._updating_safety_checkbox = False  # Flag to prevent callback loops
        print("🔧 DEBUG: _updating_safety_checkbox initialized")

        # Create GUI
        print("🔧 DEBUG: About to call _create_gui()")
        self._create_gui()
        print("🔧 DEBUG: _create_gui() completed")

        # Start GUI update processing first
        print("🔧 DEBUG: About to call _process_updates()")
        self._process_updates()
        print("🔧 DEBUG: _process_updates() completed")

        # Start ROS2 spinning in background (delayed to ensure GUI is ready)
        print("🔧 DEBUG: About to call root.after()")
        self.root.after(500, self._delayed_ros_start)
        print("🔧 DEBUG: root.after() completed")

        # Start system status updates (every 1 second)
        self._update_system_status()

        # Move initialization messages to debug logger
        logger.debug("🤖 Robot GUI ROS2 initialized")
        logger.debug("   └── All communication via ROS2 topics and services")
        logger.debug("   └── Single command gateway maintained [[memory:5366669]]")

        # VILA server managed manually - start simple_vila_server.py separately
        self.vila_model = None

        # Start automatic VILA analysis timer (after GUI is fully initialized)
        self.root.after(2000, self._check_auto_analysis)  # Delay 2 seconds for full initialization
    
    def _start_vila_server(self):
        """Start VILA server in GUI thread"""
        try:
            from robot_vila_system.vila_model import VILAModel
            self.vila_model = VILAModel()
            self.log_message("🚀 VILA server startup initiated from GUI")
        except Exception as e:
            self.log_message(f"❌ Failed to start VILA server: {e}")
    
    def _ros_callback(self, message_type: str, data):
        """Callback from ROS2 node to update GUI"""
        self.update_queue.put((message_type, data))
    
    def _delayed_ros_start(self):
        """Start ROS2 spinning after GUI is fully initialized"""
        try:
            self.ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
            self.ros_thread.start()
            logger.debug("🔄 ROS2 background thread started")
        except Exception as e:
            logger.error(f"Failed to start ROS2 thread: {e}")
    
    def _spin_ros(self):
        """Spin ROS2 node in background thread with non-blocking approach"""
        try:
            while rclpy.ok():
                rclpy.spin_once(self.ros_node, timeout_sec=0.1)
        except Exception as e:
            logger.error(f"ROS2 spinning error: {e}")
    
    def _process_updates(self):
        """Process updates from ROS2 callbacks (fast updates for real-time data)"""
        try:
            while not self.update_queue.empty():
                message_type, data = self.update_queue.get_nowait()
                self._handle_ros_update(message_type, data)
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing updates: {e}")
        
        # Schedule next update (50ms for responsive ROS message processing)
        self.root.after(50, self._process_updates)
    

    

    
    def _update_system_status(self):
        """Update system status display (runs every 1 second)"""
        try:
            # Update VILA model state
            if hasattr(self, 'sensor_data') and self.sensor_data:
                vila_state = self._get_vila_model_state()
                self.sensor_data['vila_model_state'] = vila_state
                # Set VILA server status based on model state
                if vila_state in ["Active", "Ready"]:
                    self.sensor_data['vila_server_status'] = "Online"
                else:
                    self.sensor_data['vila_server_status'] = "Offline"
                
                # Update robot status based on current sensor data
                self.sensor_data['robot_status'] = self._determine_robot_status(self.sensor_data)
                
                # Update individual labels in the new single-page design
                if hasattr(self, 'battery_label') and 'battery_voltage' in self.sensor_data:
                    voltage = self.sensor_data['battery_voltage']
                    voltage_text = f"Battery: {voltage:.2f}V"
                    if voltage <= 10.4:
                        self.battery_label.config(text=voltage_text, foreground="red")
                    elif voltage <= 10.6:
                        self.battery_label.config(text=voltage_text, foreground="orange")
                    else:
                        self.battery_label.config(text=voltage_text, foreground="green")
                
                if hasattr(self, 'cpu_label') and 'cpu_usage' in self.sensor_data:
                    cpu_usage = self.sensor_data['cpu_usage']
                    self.cpu_label.config(text=f"CPU: {cpu_usage:.1f}%")
                
                # Update CPU temperature display
                if hasattr(self, 'cpu_temp_label') and 'cpu_temp' in self.sensor_data:
                    cpu_temp = self.sensor_data['cpu_temp']
                    self.cpu_temp_label.config(text=f"CPU: {cpu_temp:.1f}°C")
                
                # Update LiDAR distances display with individual color coding
                if hasattr(self, 'lidar_front_label'):
                    front_dist = self.sensor_data.get('distance_front', 0.0)
                    left_dist = self.sensor_data.get('distance_left', 0.0)
                    right_dist = self.sensor_data.get('distance_right', 0.0)
                    
                    def get_distance_color(distance):
                        """Get color based on distance value"""
                        if distance < 0.3:
                            return "red"      # Danger - very close obstacles
                        elif distance < 0.6:
                            return "orange"   # Caution - moderate distance
                        elif distance < 1.0:
                            return "blue"     # Normal - safe distance
                        else:
                            return "green"    # Clear - plenty of space
                    
                    # Update each distance with individual color coding
                    self.lidar_front_label.config(
                        text=f"{front_dist:.1f}m", 
                        foreground=get_distance_color(front_dist)
                    )
                    self.lidar_left_label.config(
                        text=f"{left_dist:.1f}m", 
                        foreground=get_distance_color(left_dist)
                    )
                    self.lidar_right_label.config(
                        text=f"{right_dist:.1f}m", 
                        foreground=get_distance_color(right_dist)
                    )
                
                # Update Robot status based on battery voltage
                if hasattr(self, 'ros_status_label') and 'battery_voltage' in self.sensor_data:
                    battery_voltage = self.sensor_data['battery_voltage']
                    # Create composite status display with label in black and status in color
                    if battery_voltage > 8.0:
                        self._update_composite_status(self.ros_status_label, "Robot:", "Online", "green")
                    else:
                        self._update_composite_status(self.ros_status_label, "Robot:", "Offline", "red")
                
                # Update VILA status
                # Update VILA status in System Status (moved from VILA Analysis frame)
                if hasattr(self, 'vila_status_label'):
                    vila_state = self._get_vila_model_state()
                    if vila_state in ["Active", "Ready"]:
                        self._update_composite_status(self.vila_status_label, "VILA:", "Online", "green")
                        self._set_vila_frame_enabled(True)
                    else:
                        self._update_composite_status(self.vila_status_label, "VILA:", "Offline", "red")
                        self._set_vila_frame_enabled(False)
                        
        except Exception as e:
            logger.error(f"Error updating system status: {e}")
        
        # Schedule next system status update (1000ms = 1 second)
        if hasattr(self, 'root') and self.root:
            self.root.after(1000, self._update_system_status)
    
    def _handle_ros_update(self, message_type: str, data):
        """Handle different types of ROS2 updates"""
        try:
            if message_type == 'robot_status':
                self._update_robot_status(data)
            elif message_type == 'sensor_data':
                self._update_sensor_data(data)
            elif message_type == 'imu_data':
                self._update_imu_data(data)
            elif message_type == 'camera_image':
                self._update_camera_image(data)
            elif message_type == 'vila_analysis':
                self._update_vila_analysis(data)
            elif message_type == 'navigation_commands':
                self._update_navigation_commands(data)
            elif message_type == 'safety_status':
                self._update_safety_status(data)
            elif message_type == 'command_ack':
                self._handle_command_ack(data)
            elif message_type == 'command_success':
                self._handle_command_success(data)
            elif message_type == 'command_error':
                self._handle_command_error(data)
            elif message_type == 'vila_analysis_response':
                self._handle_vila_analysis_response(data)
            elif message_type == 'vila_analysis_error':
                self._handle_vila_analysis_error(data)
            elif message_type == 'vila_status':
                self._update_vila_server_status(data)
        except Exception as e:
            logger.error(f"Error handling ROS update {message_type}: {e}")
    
    def _create_gui(self):
        """Create single-page GUI with sections"""
        print("🔧 DEBUG: Creating single-page GUI design")
        
        # Skip window title and geometry for now to avoid segfault
        print("🔧 DEBUG: Skipping title/geometry to avoid segfault")
        
        try:
            # Main container
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Top section: System Status (resizable)
            print("🔧 DEBUG: Creating system status section...")
            status_frame = ttk.LabelFrame(main_frame, text="🖥️ System Status", padding=10)
            status_frame.pack(fill=tk.X, pady=(0, 10))
            self._create_system_status_section(status_frame)
            
            # Middle section: Main controls (3 columns)
            middle_frame = ttk.Frame(main_frame)
            middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
            # Left column: Movement Controls (half width, fixed size)
            print("🔧 DEBUG: Creating movement controls section...")
            movement_frame = ttk.LabelFrame(middle_frame, text="🎮 Movement Controls", padding=10)
            movement_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
            movement_frame.pack_propagate(False)  # Prevent frame from shrinking
            movement_frame.configure(width=200)  # 10% wider than 180 (180 * 1.1 = 198, rounded to 200)
            self._create_movement_section(movement_frame)
            
            # Center column: Camera Feed (fixed width for native camera size)
            print("🔧 DEBUG: Creating camera section...")
            camera_frame = ttk.LabelFrame(middle_frame, text="📹 Camera Feed", padding=10)
            camera_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
            camera_frame.pack_propagate(False)  # Prevent frame from changing size
            # Size for 640x480 camera + controls (with padding)
            camera_frame.configure(width=440, height=400)  # Native camera size + padding
            self._create_camera_section(camera_frame)
            
            # Right column: VILA Analysis (resizable)
            print("🔧 DEBUG: Creating VILA section...")
            vila_frame = ttk.LabelFrame(middle_frame, text="🤖 VILA Analysis", padding=10)
            vila_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
            self._create_vila_section(vila_frame)
            
            # Bottom section: Activity Log (resizable)
            print("🔧 DEBUG: Creating activity log section...")
            log_frame = ttk.LabelFrame(main_frame, text="📋 Activity Log", padding=10)
            log_frame.pack(fill=tk.BOTH, expand=True)
            self._create_log_section(log_frame)
            
            # Now that GUI is created, safely set the window title
            try:
                self.root.title(self.base_title)
                print(f"🔧 DEBUG: Window title set to: {self.base_title}")
            except Exception as e:
                print(f"🔧 DEBUG: Failed to set window title: {e}")
            
            print("🔧 DEBUG: Single-page GUI created successfully")
            
        except Exception as e:
            print(f"🔧 DEBUG: Error creating GUI: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_system_status_panel(self):
        """Create the always-visible system status panel"""
        # Configure grid column weights to prevent resizing
        self.system_status_frame.grid_columnconfigure(0, weight=0, minsize=120)
        self.system_status_frame.grid_columnconfigure(1, weight=1, minsize=100)
        
        # System status displays
        self.sensor_labels = {}
        # Reorganized sensor names for cleaner display (removed redundant status items)
        sensor_names = [
            ("Battery", "battery_voltage", "V"),
            ("CPU Temp", "cpu_temp", "°C"),
            ("CPU Usage", "cpu_usage", "%"),
            ("LiDAR F", "distance_front", "m"),
            ("LiDAR L", "distance_left", "m"),
            ("LiDAR R", "distance_right", "m"),
            ("IMU X", "imu_accel_x", "m/s²"),
            ("IMU Y", "imu_accel_y", "m/s²"),
            ("IMU Z", "imu_accel_z", "m/s²")
        ]
        
        for i, (name, key, unit) in enumerate(sensor_names):
            # Fixed width label for names
            name_label = ttk.Label(self.system_status_frame, text=f"{name}:", width=15)
            name_label.grid(row=i, column=0, sticky=tk.W, pady=2)
            
            # Fixed width label for values to prevent resizing
            value_label = ttk.Label(self.system_status_frame, text="---", width=12, anchor=tk.W)
            value_label.grid(row=i, column=1, sticky=tk.W, padx=10, pady=2)
            self.sensor_labels[key] = (value_label, unit)
    
    def _create_control_tab(self):
        """Create robot control tab - UNUSED: Legacy tabbed interface code"""
        # Robot info frame
        info_frame = ttk.LabelFrame(self.control_frame, text="Robot Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Remove Robot ID display as requested
        
        # Connection status
        ttk.Label(info_frame, text="Connection:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        # Create composite connection status frame
        conn_frame = ttk.Frame(info_frame)
        conn_frame.grid(row=0, column=3, sticky=tk.W, padx=10)
        
        # Create composite status with black label and colored status
        conn_prefix = ttk.Label(conn_frame, text="Robot:", foreground="black")
        conn_prefix.pack(side=tk.LEFT)
        
        self.connection_label = ttk.Label(conn_frame, text="Offline", foreground="red", font=("Arial", 10, "bold"))
        self.connection_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Movement controls frame
        movement_frame = ttk.LabelFrame(self.control_frame, text="Movement Controls", padding=10)
        movement_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Safety toggle - Start with movement disabled
        self.safety_var = tk.BooleanVar(value=False)
        self.safety_checkbox = ttk.Checkbutton(
            movement_frame, 
            text="Movement Enable", 
            variable=self.safety_var,
            command=self._toggle_movement_safety
        )
        self.safety_checkbox.pack(anchor=tk.W)
        
        # Movement buttons
        button_frame = ttk.Frame(movement_frame)
        button_frame.pack(pady=10)
        
        # Forward button (ultra compact)
        ttk.Button(button_frame, text="↑", width=3,
                  command=lambda: self._send_movement_command("move_forward")).grid(row=0, column=1, padx=1, pady=1)
        
        # Left and right buttons (ultra compact)
        ttk.Button(button_frame, text="←", width=3,
                  command=lambda: self._send_movement_command("turn_left")).grid(row=1, column=0, padx=1, pady=1)
        ttk.Button(button_frame, text="→", width=3,
                  command=lambda: self._send_movement_command("turn_right")).grid(row=1, column=2, padx=1, pady=1)
        
        # Backward button (ultra compact)
        ttk.Button(button_frame, text="↓", width=3,
                  command=lambda: self._send_movement_command("move_backward")).grid(row=2, column=1, padx=1, pady=1)
        
        # Stop button (ultra compact)
        stop_btn = ttk.Button(button_frame, text="⏹", width=3,
                             command=lambda: self._send_movement_command("stop"))
        stop_btn.grid(row=1, column=1, padx=1, pady=1)
        stop_btn.configure(style="Accent.TButton")
        
        # Emergency stop
        emergency_btn = ttk.Button(movement_frame, text="🚨 EMERGENCY STOP", 
                                  command=self._emergency_stop)
        emergency_btn.pack(pady=10)
        
        # Log frame
        log_frame = ttk.LabelFrame(self.control_frame, text="Activity Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    

    def _create_monitoring_tab(self):
        """Create monitoring tab - UNUSED: Legacy tabbed interface code"""
        # Camera frame with controls
        camera_frame = ttk.LabelFrame(self.monitoring_frame, text="Camera Feed", padding=10)
        camera_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera controls frame
        controls_frame = ttk.Frame(camera_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Camera source selection
        source_frame = ttk.LabelFrame(controls_frame, text="Camera Source", padding=5)
        source_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.camera_source_var = tk.StringVar(value="robot")
        ttk.Radiobutton(source_frame, text="Robot Camera", variable=self.camera_source_var, 
                       value="robot", command=self._on_camera_source_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="Simulator", variable=self.camera_source_var, 
                       value="sim", command=self._on_camera_source_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="Loaded Image", variable=self.camera_source_var, 
                       value="loaded", command=self._on_camera_source_change).pack(side=tk.LEFT, padx=5)
        
        # Load image button
        ttk.Button(controls_frame, text="📁 Load Image", 
                  command=self._load_image_file).pack(side=tk.RIGHT, padx=5)
        
        # Camera display
        self.camera_label = ttk.Label(camera_frame, text="No camera feed")
        self.camera_label.pack(expand=True)
    
    def _create_vila_tab(self):
        """Create VILA analysis tab - UNUSED: Legacy tabbed interface code"""
        # VILA controls frame
        controls_frame = ttk.LabelFrame(self.vila_frame, text="VILA Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Top row: Status and Auto Analysis
        top_row = ttk.Frame(controls_frame)
        top_row.pack(fill=tk.X, pady=2)
        
        # VILA resource status - create frame for composite status
        vila_frame = ttk.Frame(top_row)
        vila_frame.pack(side=tk.LEFT, padx=10)
        
        # Create composite status with black label and colored status
        vila_label = ttk.Label(vila_frame, text="VILA:", foreground="black")
        vila_label.pack(side=tk.LEFT)
        
        self.vila_status_label = ttk.Label(vila_frame, text="Offline", foreground="red")
        self.vila_status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Automatic analysis toggle button (avoiding BooleanVar due to segfault issues)
        self.vila_auto_button = ttk.Button(
            top_row,
            text="🔄 Auto Analysis: OFF",
            command=self._toggle_vila_auto_analysis_button
        )
        self.vila_auto_button.pack(side=tk.RIGHT, padx=10)
        
        # Quick analysis buttons
        quick_frame = ttk.Frame(controls_frame)
        quick_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(quick_frame, text="Quick Analysis:").pack(side=tk.LEFT, padx=5)
        ttk.Button(quick_frame, text="🚦 Navigation", 
                  command=lambda: self._quick_vila_analysis("navigation")).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_frame, text="🔍 Objects", 
                  command=lambda: self._quick_vila_analysis("objects")).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_frame, text="🌍 Scene", 
                  command=lambda: self._quick_vila_analysis("scene")).pack(side=tk.LEFT, padx=2)
        
        # Analysis request frame
        request_frame = ttk.LabelFrame(self.vila_frame, text="Request Analysis", padding=10)
        request_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(request_frame, text="Prompt:").pack(anchor=tk.W)
        self.vila_prompt_text = tk.Text(request_frame, height=3, wrap=tk.WORD)
        self.vila_prompt_text.pack(fill=tk.X, pady=5)
        self.vila_prompt_text.insert(tk.END, "Analyze the current camera view for navigation.")
        
        ttk.Button(request_frame, text="Request VILA Analysis", 
                  command=self._request_vila_analysis).pack(pady=5)
        
        # Analysis results frame
        results_frame = ttk.LabelFrame(self.vila_frame, text="Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.vila_results_text = scrolledtext.ScrolledText(results_frame, height=15, wrap=tk.WORD)
        self.vila_results_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_text = ttk.Label(self.status_bar, text="Ready - ROS2 Communication Active")
        self.status_text.pack(side=tk.LEFT, padx=10, pady=5)
        
        # ROS2 status indicator - create frame for composite status
        ros_frame = ttk.Frame(self.status_bar)
        ros_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Create composite status with black label and colored status
        ros_label = ttk.Label(ros_frame, text="ROS2:", foreground="black")
        ros_label.pack(side=tk.LEFT)
        
        self.ros_status = ttk.Label(ros_frame, text="Online", foreground="green")
        self.ros_status.pack(side=tk.LEFT, padx=(5, 0))
    
    def _send_movement_command(self, command_type: str):
        """Send movement command through ROS2"""
        try:
            if not self.safety_var.get():
                messagebox.showwarning("Safety", "Movement is disabled. Enable movement first.")
                self.log_message("⚠️ Movement command blocked - safety disabled")
                return
            
            parameters = {
                'speed': 0.1,  # Default linear speed 0.1 m/s as per integration guide
                'distance': 0.2,  # Default distance 20cm for forward movement
                'duration': 10.0  # Longer timeout as per guide
            }
            
            if self.ros_node and hasattr(self.ros_node, 'send_robot_command'):
                success = self.ros_node.send_robot_command(
                    command_type, 
                    parameters, 
                    safety_confirmed=True
                )
                
                if success:
                    self.log_message(f"🎯 Command sent: {command_type}")
                else:
                    self.log_message(f"❌ Failed to send command: {command_type}")
            else:
                self.log_message(f"❌ Cannot send command: {command_type} (ROS node not available)")
                
        except Exception as e:
            self.log_message(f"❌ Error sending movement command: {e}")
            logger.error(f"Error in _send_movement_command: {e}")
    
    def _toggle_movement_safety(self):
        """Toggle movement safety"""
        try:
            # Prevent callback loops when we're programmatically updating the checkbox
            if self._updating_safety_checkbox:
                return
            
            enabled = self.safety_var.get()
            self.safety_enabled = enabled  # Keep internal state in sync
            self.movement_enabled = enabled
            
            # Set a flag to prevent ROS messages from overriding for a short time
            self._updating_safety_checkbox = True
            
            # Send to ROS node if available
            if self.ros_node and hasattr(self.ros_node, 'set_safety_enabled'):
                self.ros_node.set_safety_enabled(enabled)
                self.log_message(f"🛡️ Movement {'ENABLED' if enabled else 'DISABLED'} via GUI")
            else:
                self.log_message(f"🛡️ Movement {'ENABLED' if enabled else 'DISABLED'} (ROS node not available)")
            
            # Update button states immediately
            self._update_button_states(enabled)
            
            # Clear the flag after a short delay to allow the change to propagate
            def clear_flag():
                self._updating_safety_checkbox = False
            
            self.root.after(500, clear_flag)  # Wait 500ms before allowing ROS updates again
                
        except Exception as e:
            self.log_message(f"❌ Error toggling movement safety: {e}")
            logger.error(f"Error in _toggle_movement_safety: {e}")
            self._updating_safety_checkbox = False  # Clear flag on error
    
    def _emergency_stop(self):
        """Trigger emergency stop"""
        try:
            if self.ros_node and hasattr(self.ros_node, 'emergency_stop'):
                self.ros_node.emergency_stop()
                self.log_message("🚨 EMERGENCY STOP triggered via GUI")
            else:
                self.log_message("🚨 EMERGENCY STOP requested (ROS node not available)")
            
            # Use flag to prevent callback loop when programmatically updating checkbox
            self._updating_safety_checkbox = True
            self.safety_var.set(False)
            self._updating_safety_checkbox = False
            self.safety_enabled = False  # Keep internal state in sync
            self.movement_enabled = False
            
            # Disable movement buttons
            self._update_button_states(False)
            
        except Exception as e:
            self.log_message(f"❌ Error triggering emergency stop: {e}")
            logger.error(f"Error in _emergency_stop: {e}")
    
    def _update_button_states(self, enabled: bool):
        """Enable or disable movement buttons based on safety state"""
        try:
            if hasattr(self, 'movement_buttons'):
                state = 'normal' if enabled else 'disabled'
                for button in self.movement_buttons:
                    button.config(state=state)
                logger.debug(f"Movement buttons {'ENABLED' if enabled else 'DISABLED'}")
        except Exception as e:
            logger.error(f"Error updating button states: {e}")
    
    def _request_vila_analysis(self):
        """Request VILA analysis"""
        try:
            prompt = self.vila_prompt_text.get(1.0, tk.END).strip()
            if not prompt:
                messagebox.showwarning("Input Error", "Please enter a prompt for analysis.")
                return
            
            if self.ros_node and hasattr(self.ros_node, 'request_vila_analysis'):
                success = self.ros_node.request_vila_analysis(prompt)
                if success:
                    self.log_message(f"📝 VILA analysis requested: {prompt[:50]}...")
                else:
                    self.log_message("❌ Failed to request VILA analysis")
            else:
                self.log_message("❌ Cannot request VILA analysis (ROS node not available)")
                
        except Exception as e:
            self.log_message(f"❌ Error requesting VILA analysis: {e}")
            logger.error(f"Error in _request_vila_analysis: {e}")
    
    def _update_robot_status(self, data):
        """Update robot status display"""
        self.robot_data = data
        
        # Update connection status based on battery voltage
        battery_voltage = data.get('battery_voltage', 0)
        if battery_voltage > 8.0:
            self._update_composite_status(self.connection_label, "Robot:", "Online", "green")
        else:
            self._update_composite_status(self.connection_label, "Robot:", "Offline", "red")
        
        # Update robot status in system status panel with more detailed info
        if 'robot_status' in self.sensor_labels:
            status_label, unit = self.sensor_labels['robot_status']
            
            # Create detailed status string
            if 'status' in data:
                status_text = data['status'].title()
                # Add connection indicator
                status_text += " (Connected)"
                status_label.config(text=status_text, foreground="green")
            else:
                status_label.config(text="Offline", foreground="red")
    
    def _update_sensor_data(self, data):
        """Update sensor data storage (display updates handled by _update_system_status)"""
        self.sensor_data = data
    
    def _update_imu_data(self, data):
        """Update IMU data storage (display updates handled by _update_system_status)"""
        # Store IMU data in sensor_data for system status display
        if not hasattr(self, 'sensor_data'):
            self.sensor_data = {}
        
        # Merge IMU data into sensor_data
        for key in ['imu_accel_x', 'imu_accel_y', 'imu_accel_z']:
            if key in data:
                self.sensor_data[key] = data[key]
    
    def _get_vila_model_state(self):
        """Get current VILA model state"""
        try:
            # Check VILA server status from published status messages
            if hasattr(self, 'vila_server_status') and self.vila_server_status:
                # Check if model is loaded (server_ready indicates model loading completion)
                if self.vila_server_status.get('server_ready', False):
                    # Model is loaded, now check if we have recent VILA analysis (within last 30 seconds)
                    if hasattr(self.ros_node, 'last_vila_update') and self.ros_node.last_vila_update:
                        time_diff = time.time() - self.ros_node.last_vila_update
                        if time_diff < 30:
                            return "Active"
                        else:
                            return "Ready"
                    else:
                        return "Ready"
                else:
                    # VILA server is running but model not yet loaded
                    return "Starting"
            else:
                # No VILA status received yet or VILA node not running
                return "Offline"
        except Exception:
            return "Unknown"
    
    def _determine_robot_status(self, sensor_data):
        """Determine robot status based on sensor data, using battery voltage as primary indicator"""
        try:
            # Check if we have recent sensor data (within last 5 seconds)
            current_time = time.time() * 1000000000  # Convert to nanoseconds
            sensor_timestamp = sensor_data.get('timestamp', 0)
            
            if sensor_timestamp == 0:
                return "Offline"
            
            time_diff = (current_time - sensor_timestamp) / 1000000000  # Convert back to seconds
            
            if time_diff > 10:  # No data for more than 10 seconds
                return "Offline"
            elif time_diff > 5:  # Stale data (5-10 seconds)
                return "Stale"
            else:
                # Use battery voltage as primary indicator of robot status
                battery_voltage = sensor_data.get('battery_voltage', 0)
                
                if battery_voltage <= 0:
                    return "Offline"
                elif 7.0 <= battery_voltage <= 13.0:
                    # Robot is running - battery voltage is in operational range
                    return "Running"
                elif battery_voltage < 7.0:
                    return "Low Battery"
                elif battery_voltage > 13.0:
                    return "High Voltage"
                else:
                    return "Unknown"
        except Exception as e:
            logger.error(f"Error determining robot status: {e}")
            return "Unknown"
    
    def _update_camera_image(self, pil_image):
        """Update camera image display"""
        try:
            self.current_image = pil_image
            current_time = time.time()
            if not hasattr(self, '_last_gui_image_log_time'):
                self._last_gui_image_log_time = 0
            
            # Log every second to track GUI update frequency
            if current_time - self._last_gui_image_log_time >= 1.0:
                logger.debug(f"📸 GUI received camera image, source={self.camera_source}")
                self._last_gui_image_log_time = current_time
            
            # Only display if camera source is set to robot
            if self.camera_source == "robot":
                self._update_camera_display(pil_image)
                logger.debug("📸 Camera image displayed")
            else:
                logger.debug(f"📸 Camera image not displayed, source is {self.camera_source}")
            
        except Exception as e:
            logger.error(f"Error updating camera image: {e}")
    
    def _update_vila_analysis(self, data):
        """Update VILA analysis display"""
        self.vila_analysis = data
        
        # Format analysis result
        timestamp = datetime.fromtimestamp(data['timestamp'] / 1e9).strftime("%H:%M:%S")
        analysis_text = f"[{timestamp}] VILA Analysis:\n"
        analysis_text += f"Confidence: {data['confidence']:.2f}\n"
        analysis_text += f"Result: {data['analysis_result']}\n"
        
        if data['navigation_commands']:
            nav_cmd = data['navigation_commands']
            analysis_text += f"Navigation: {nav_cmd.get('action', 'unknown')} "
            analysis_text += f"(confidence: {nav_cmd.get('confidence', 0):.2f})\n"
        
        analysis_text += "-" * 50 + "\n"
        
        # Add to results
        self.vila_results_text.insert(tk.END, analysis_text)
        self.vila_results_text.see(tk.END)
    
    def _update_navigation_commands(self, data):
        """Update navigation commands display"""
        if data:
            self.log_message(f"🧭 Navigation: {data.get('action', 'unknown')} "
                           f"(confidence: {data.get('confidence', 0):.2f})")
    
    def _update_safety_status(self, enabled):
        """Update safety status - only if not currently being changed by user"""
        # Don't override user's checkbox if they're actively changing it
        if self._updating_safety_checkbox:
            return
            
        # Only update if the state has actually changed to prevent spam
        if hasattr(self, 'safety_enabled') and self.safety_enabled == enabled:
            return
            
        self.safety_enabled = enabled
        
        # Only update the checkbox if it's different from ROS state
        current_gui_state = self.safety_var.get() if hasattr(self, 'safety_var') else False
        if current_gui_state != enabled:
            # Use flag to prevent callback loop when programmatically updating checkbox
            self._updating_safety_checkbox = True
            self.safety_var.set(enabled)
            self._updating_safety_checkbox = False
            
            # Update button states to match ROS safety state
            self._update_button_states(enabled)
            
            self.log_message(f"🛡️ Safety status updated from ROS: {'ENABLED' if enabled else 'DISABLED'}")
        else:
            # States match, just log quietly
            logger.debug(f"Safety status confirmed: {'ENABLED' if enabled else 'DISABLED'}")
    
    def _handle_command_ack(self, data):
        """Handle command acknowledgment"""
        success_text = "✅" if data['success'] else "❌"
        self.log_message(f"{success_text} Command ACK: {data['command_type']} from robot")
    
    def _handle_command_success(self, data):
        """Handle command success"""
        self.log_message(f"✅ Gateway accepted: {data['command']} - {data['message']}")
    
    def _handle_command_error(self, data):
        """Handle command error"""
        self.log_message(f"🚫 Gateway rejected: {data['command']} - {data['message']}")
        messagebox.showerror("Command Error", f"Command rejected: {data['message']}")
    
    def _handle_vila_analysis_response(self, data):
        """Handle VILA analysis response"""
        # Status will be updated by the system status update cycle
        
        timestamp = datetime.fromtimestamp(data['timestamp'] / 1e9).strftime("%H:%M:%S")
        analysis_text = f"[{timestamp}] VILA Analysis Complete:\n"
        analysis_text += f"Confidence: {data['confidence']:.2f}\n"
        analysis_text += f"Result: {data['analysis_result']}\n"
        
        if data.get('navigation_commands'):
            nav_cmd = data['navigation_commands']
            action = nav_cmd.get('action', 'unknown')
            confidence = nav_cmd.get('confidence', 0)
            
            analysis_text += f"Navigation: {action} "
            analysis_text += f"(confidence: {confidence:.2f})\n"
            
            # AUTOMATIC COMMAND EXECUTION for auto-analysis
            # Only execute if this was an automatic analysis (not manual user request)
            if (hasattr(self, 'vila_auto_analysis') and self.vila_auto_analysis and 
                confidence > 0.5 and self.safety_enabled and self.movement_enabled):
                
                try:
                    # COSMOS NEMOTRON VLA: Enhanced multi-modal decision with built-in sensor fusion
                    # The Cosmos Nemotron VLA model already includes LiDAR+IMU+vision fusion
                    self.log_message(f"🚀 Cosmos Nemotron VLA Decision: {action} (confidence: {confidence:.2f})")
                    
                    # Trust the enhanced VLA model's integrated sensor fusion and safety
                    current_sensor_data = getattr(self.ros_node, 'current_sensor_data', None)
                    if current_sensor_data:
                        # Log sensor context for monitoring (but don't override the VLA decision)
                        if isinstance(current_sensor_data, dict):
                            front_dist = current_sensor_data.get('distance_front', 0.0)
                            left_dist = current_sensor_data.get('distance_left', 0.0)
                            right_dist = current_sensor_data.get('distance_right', 0.0)
                        else:
                            front_dist = getattr(current_sensor_data, 'distance_front', 0.0)
                            left_dist = getattr(current_sensor_data, 'distance_left', 0.0) 
                            right_dist = getattr(current_sensor_data, 'distance_right', 0.0)
                        
                        self.log_message(f"📊 Sensor Context - LiDAR: F:{front_dist:.2f}m L:{left_dist:.2f}m R:{right_dist:.2f}m")
                    
                    # Map VILA actions to robot commands (only if safety checks passed)
                    if action == 'move_forward' or action == 'move':
                        self.log_message(f"🤖 Auto-executing: move_forward (VILA confidence: {confidence:.2f})")
                        # Use shorter, safer movement parameters
                        self.ros_node.send_robot_command('move_forward', parameters={'distance': 0.1, 'speed': 0.05})
                    elif action == 'turn_left' or (action == 'turn' and 'left' in nav_cmd.get('reason', '').lower()):
                        self.log_message(f"🤖 Auto-executing: turn_left (VILA confidence: {confidence:.2f})")
                        self.ros_node.send_robot_command('turn_left', parameters={'angle': 45.0})  # Smaller turn
                    elif action == 'turn_right' or (action == 'turn' and 'right' in nav_cmd.get('reason', '').lower()):
                        self.log_message(f"🤖 Auto-executing: turn_right (VILA confidence: {confidence:.2f})")
                        self.ros_node.send_robot_command('turn_right', parameters={'angle': 45.0})  # Smaller turn
                    elif action == 'stop':
                        self.log_message(f"🤖 Auto-executing: stop (VILA confidence: {confidence:.2f})")
                        self.ros_node.send_robot_command('stop', parameters={})
                    else:
                        self.log_message(f"🤔 VILA suggested '{action}' but no clear mapping to robot command")
                        
                except Exception as e:
                    self.log_message(f"❌ Error auto-executing VILA command: {e}")
            elif hasattr(self, 'vila_auto_analysis') and self.vila_auto_analysis:
                # Log why we didn't execute
                reasons = []
                if confidence <= 0.5:
                    reasons.append(f"low confidence ({confidence:.2f})")
                if not self.safety_enabled:
                    reasons.append("safety disabled")
                if not self.movement_enabled:
                    reasons.append("movement disabled")
                    
                if reasons:
                    self.log_message(f"⏸️ VILA suggested '{action}' but not executing: {', '.join(reasons)}")
        
        analysis_text += "-" * 50 + "\n"
        
        self.vila_results_text.insert(tk.END, analysis_text)
        self.vila_results_text.see(tk.END)
    
    def _handle_vila_analysis_error(self, data):
        """Handle VILA analysis error"""
        # Status will be updated by the system status update cycle
        
        error_text = f"❌ VILA Analysis Error: {data['error']}\n"
        error_text += "-" * 50 + "\n"
        
        self.vila_results_text.insert(tk.END, error_text)
        self.vila_results_text.see(tk.END)
    
    def _update_vila_server_status(self, status_data: dict):
        """Update VILA server status display"""
        self.vila_server_status = status_data
        
        # Log for debugging
        # Move technical status updates to debug logger
        logger.debug(f"🔧 VILA server status updated: server_ready={status_data.get('server_ready', False)}")
        
        # Force immediate update of VILA status labels
        self._force_vila_status_update()

        # Update the system status display (only if sensor_labels exists)
        if hasattr(self, 'sensor_labels') and 'vila_server_status' in self.sensor_labels:
            status_label, unit = self.sensor_labels['vila_server_status']
            
            # Format status for display
            status = status_data.get('status', 'unknown')
            if status == 'running':
                self._update_composite_status(status_label, "VILA:", "Online", "green")
            elif status == 'starting':
                self._update_composite_status(status_label, "VILA:", "Offline", "red")
            elif status == 'error':
                self._update_composite_status(status_label, "VILA:", "Offline", "red")
            else:
                self._update_composite_status(status_label, "VILA:", "Offline", "red")
    
    def _update_composite_status(self, status_label, prefix, status_text, color):
        """Update a status label with colored text while keeping prefix in black"""
        try:
            # For single labels that were converted to composite frames
            if hasattr(status_label, 'master') and len(status_label.master.winfo_children()) > 1:
                # This is a composite status with separate prefix and status labels
                status_label.config(text=status_text, foreground=color, font=("Arial", 10, "bold"))
            else:
                # Fallback for labels that haven't been converted yet
                status_label.config(text=f"{prefix} {status_text}", foreground=color, font=("Arial", 10, "bold"))
        except Exception as e:
            logger.debug(f"Error updating composite status: {e}")
            # Fallback to simple text update
            status_label.config(text=f"{prefix} {status_text}", foreground=color, font=("Arial", 10, "bold"))

    def _force_vila_status_update(self):
        """Force immediate update of all VILA status labels"""
        try:
            vila_state = self._get_vila_model_state()
            
            # Update VILA status in System Status and manage VILA frame state
            if hasattr(self, 'vila_status_label'):
                if vila_state in ["Active", "Ready"]:
                    self._update_composite_status(self.vila_status_label, "VILA:", "Online", "green")
                    self._set_vila_frame_enabled(True)
                else:
                    self._update_composite_status(self.vila_status_label, "VILA:", "Offline", "red")
                    self._set_vila_frame_enabled(False)
                    
            logger.debug(f"🔧 VILA status labels updated to: {vila_state}")
        except Exception as e:
            self.log_message(f"❌ Error updating VILA status labels: {e}")
    
    def _set_vila_frame_enabled(self, enabled):
        """Enable or disable (grey out) the VILA Analysis frame"""
        try:
            if hasattr(self, 'vila_analysis_frame'):
                # Set the state of all child widgets in the VILA Analysis frame
                state = "normal" if enabled else "disabled"
                self._set_widget_state_recursive(self.vila_analysis_frame, state)
        except Exception as e:
            logger.debug(f"Error setting VILA frame state: {e}")
    
    def _set_widget_state_recursive(self, widget, state):
        """Recursively set the state of all child widgets"""
        try:
            # Try to set state if widget supports it
            if hasattr(widget, 'config'):
                try:
                    widget.config(state=state)
                except:
                    pass  # Some widgets don't support state
            
            # Recursively process child widgets
            for child in widget.winfo_children():
                self._set_widget_state_recursive(child, state)
        except Exception as e:
            logger.debug(f"Error setting widget state: {e}")
    
    def _get_vila_server_status_from_gui(self):
        """Get VILA server status from GUI's VILA model"""
        if self.vila_model:
            try:
                # Server methods removed - return direct model status
                return {
                    'status': 'model_loaded' if self.vila_model.model_loaded else 'model_not_loaded',
                    'process_running': False,  # No server process
                    'recent_logs': ['Direct model loading - no server process'],
                    'server_ready': self.vila_model.model_loaded,
                    'server_url': 'direct_model'
                }
            except:
                pass
        return {'status': 'stopped', 'process_running': False, 'recent_logs': [], 'server_ready': False, 'server_url': 'direct_model'}
    
    def _cleanup_vila_server(self):
        """Clean up VILA server on shutdown"""
        if self.vila_model:
            try:
                # Server methods removed - just log cleanup
                self.log_message("🧹 VILA model cleanup completed")
            except:
                pass
    
    def log_message(self, message):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Print to console always (for debugging)
        print(f"GUI: {log_entry.strip()}")
        
        # Only update GUI if log_text widget exists
        if hasattr(self, 'log_text') and self.log_text:
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
    
    def _toggle_vila_auto_analysis(self):
        """Toggle automatic VILA analysis"""
        # Follow the same pattern as the working safety checkbox
        if hasattr(self, 'vila_auto_var') and self.vila_auto_var:
            self.vila_auto_analysis = self.vila_auto_var.get()
        else:
            # Fallback if BooleanVar doesn't exist
            self.vila_auto_analysis = False
        
        if self.vila_auto_analysis:
            self.log_message("🔄 Automatic VILA analysis enabled (5s interval)")
            # Reset the timer to start fresh
            self.vila_last_auto_analysis = 0
        else:
            self.log_message("⏸️ Automatic VILA analysis disabled")
    
    def _toggle_vila_auto_analysis_button(self):
        """Toggle automatic VILA analysis using button fallback"""
        self.vila_auto_analysis = not self.vila_auto_analysis
        
        # Update button text
        if hasattr(self, 'vila_auto_button'):
            if self.vila_auto_analysis:
                self.vila_auto_button.config(text="🔄 Auto Analysis: ON")
                self.log_message("🔄 Automatic VILA analysis enabled (5s interval)")
                self.vila_last_auto_analysis = 0
            else:
                self.vila_auto_button.config(text="🔄 Auto Analysis: OFF")
                self.log_message("⏸️ Automatic VILA analysis disabled")
    
    def _check_auto_analysis(self):
        """Check if automatic analysis should be triggered"""
        try:
            # Safety checks: ensure GUI and components are fully initialized
            if (not hasattr(self, 'vila_prompt_text') or 
                not hasattr(self, 'vila_auto_analysis') or
                not hasattr(self, 'current_image')):
                return
                
            current_time = time.time()
            
            # Check if auto analysis is enabled and enough time has passed
            if (self.vila_auto_analysis and 
                self.current_image is not None and 
                current_time - self.vila_last_auto_analysis >= self.vila_auto_interval):
                
                # Trigger automatic analysis with navigation prompt
                try:
                    self._quick_vila_analysis("navigation", auto=True)
                    self.vila_last_auto_analysis = current_time
                except Exception as analysis_error:
                    logger.error(f"Error in automatic VILA analysis: {analysis_error}")
                
        except Exception as e:
            logger.error(f"Error in auto analysis check: {e}")
        finally:
            # Schedule next check in 1 second (only if root still exists and not shutting down)
            try:
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(1000, self._check_auto_analysis)
            except:
                pass  # GUI might be shutting down
    
    def _quick_vila_analysis(self, analysis_type: str, auto: bool = False):
        """Perform quick VILA analysis with predefined prompts"""
        prompts = {
            "navigation": """🚨 CRITICAL ROBOT NAVIGATION - LiDAR DATA IS TRUTH:

⚠️ LiDAR SAFETY DATA (meters):
• Front: OBSTACLE DETECTED! (real-time sensor data)
• Left: Clear path available
• Right: Clear path available

🚫 SAFETY RULES:
• Front obstacle = CANNOT MOVE FORWARD
• LiDAR data takes precedence over visual analysis
• Current situation: OBSTACLE AHEAD - must acknowledge and avoid

📷 VISUAL ANALYSIS (secondary):
Look at camera image, but LiDAR sensors show obstacle ahead.

REQUIRED RESPONSE FORMAT:
1. [YES/NO] Can I move forward? (based on LiDAR obstacle)
2. [YES/NO] Are there obstacles? (must acknowledge LiDAR data)
3. Action: [move_forward/turn_left/turn_right/stop] (respect LiDAR safety)
4. Visual description: [brief scene description]

SAFETY FIRST: LiDAR shows obstacle - do not ignore sensor data.""",
            
            "objects": """Identify and describe the objects you can see in this image. List:
1. What objects are present?
2. Where are they located (left, center, right)?
3. Are any objects blocking the path?
4. Any objects that require attention?""",
            
            "scene": """Describe this scene in detail:
1. What type of environment is this?
2. What is the lighting condition?
3. What surfaces and textures do you see?
4. Overall scene assessment for robot navigation."""
        }
        
        prompt = prompts.get(analysis_type, "Analyze this image.")
        self.vila_prompt_text.delete(1.0, tk.END)
        self.vila_prompt_text.insert(tk.END, prompt)
        
        # Automatically trigger analysis
        self._request_vila_analysis()
        
        if auto:
            self.log_message(f"🔄 Auto VILA analysis: {analysis_type}")
        else:
            logger.debug(f"🔍 Quick VILA analysis requested: {analysis_type}")
    
    def _request_vila_analysis(self):
        """Request VILA analysis with current prompt"""
        try:
            prompt = self.vila_prompt_text.get(1.0, tk.END).strip()
            if not prompt:
                messagebox.showwarning("No Prompt", "Please enter a prompt for VILA analysis.")
                return
            
            # Update status to show processing
            self._update_composite_status(self.vila_status_label, "VILA:", "Processing...", "orange")
            
            # Determine which image to use based on camera source
            image_to_use = None
            if self.camera_source == "loaded" and self.loaded_image:
                # Convert PIL image to ROS Image for analysis
                image_to_use = self._pil_to_ros_image(self.loaded_image)
                logger.debug(f"📷 Using loaded image for VILA analysis")
            elif self.camera_source == "robot" and self.current_image:
                # Use current camera image (already in ROS format in ros_node)
                logger.debug(f"📷 Using robot camera image for VILA analysis")
            
            # Send request to VILA server
            success = self.ros_node.request_vila_analysis(prompt, image_to_use)
            
            if success:
                # Keep important VILA analysis requests in user log
                self.log_message(f"🔍 VILA analysis: {prompt[:30]}...")
            else:
                self._update_composite_status(self.vila_status_label, "VILA:", "Offline", "red")
                messagebox.showerror("Service Error", "VILA analysis service is not available.")
            
        except Exception as e:
            self.log_message(f"❌ Error requesting VILA analysis: {e}")
            self._update_composite_status(self.vila_status_label, "VILA:", "Offline", "red")
    
    def _on_camera_source_change(self):
        """Handle camera source change"""
        self.camera_source = self.camera_source_var.get()
        logger.debug(f"📷 Camera source changed to: {self.camera_source}")
        
        # Update display based on source
        if self.camera_source == "loaded" and self.loaded_image:
            self._update_camera_display(self.loaded_image)
        elif self.camera_source == "robot" and self.current_image:
            self._update_camera_display(self.current_image)
        elif self.camera_source == "sim":
            # TODO: Load simulator image if available
            logger.debug("📷 Simulator camera not yet implemented")
    
    def _load_image_file(self):
        """Load an image file for analysis"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Load and display the image
                self.loaded_image = Image.open(file_path)
                # Keep file loading in user log but make it more concise
                filename = file_path.split('/')[-1] if '/' in file_path else file_path
                self.log_message(f"📁 Loaded: {filename}")
                
                # Switch to loaded image source and display
                self.camera_source_var.set("loaded")
                self.camera_source = "loaded"
                self._update_camera_display(self.loaded_image)
                
        except Exception as e:
            self.log_message(f"❌ Error loading image: {e}")
            messagebox.showerror("Load Error", f"Failed to load image: {e}")
    
    def _update_camera_display(self, pil_image):
        """Update the camera display with a PIL image"""
        try:
            # Keep native size for camera feed (640x480 or scale to fit frame)
            display_image = pil_image.copy()
            # Scale to fit the camera frame while maintaining aspect ratio
            # Handle PIL version compatibility
            try:
                # PIL 10.0.0+
                display_image.thumbnail((420, 350), Image.Resampling.LANCZOS)
            except AttributeError:
                # PIL < 10.0.0
                display_image.thumbnail((420, 350), Image.LANCZOS)
            
            # Convert to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(display_image)
            
            # Update label
            self.camera_label.config(image=photo, text="")
            self.camera_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.log_message(f"❌ Error updating camera display: {e}")
    
    def _pil_to_ros_image(self, pil_image):
        """Convert PIL Image to ROS Image message"""
        try:
            import cv2
            import numpy as np
            from sensor_msgs.msg import Image as RosImage
            from cv_bridge import CvBridge
            
            # Convert PIL to OpenCV format (RGB to BGR)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Fix mirroring: Flip image horizontally to correct orientation
            # This ensures loaded images appear the same way as they do in the camera feed
            cv_image = cv2.flip(cv_image, 1)  # 1 = horizontal flip
            
            # Convert to ROS Image
            bridge = CvBridge()
            ros_image = bridge.cv2_to_imgmsg(cv_image, "bgr8")
            
            return ros_image
        except Exception as e:
            self.log_message(f"❌ Error converting PIL to ROS image: {e}")
            return None
    
    def cleanup(self):
        """Cleanup ROS2 resources and shutdown entire system"""
        try:
            logger.info("🔄 Shutting down Robot GUI and entire system...")
            
            # Stop VILA server first
            self._cleanup_vila_server()
            
            # Send shutdown signal to other nodes via ROS2 topic
            try:
                shutdown_msg = Bool()
                shutdown_msg.data = True
                # Create a temporary publisher for shutdown signal
                shutdown_pub = self.ros_node.create_publisher(Bool, '/system/shutdown_request', 10)
                shutdown_pub.publish(shutdown_msg)
                logger.info("📡 Sent shutdown signal to other nodes")
            except Exception as e:
                logger.warning(f"Could not send shutdown signal: {e}")
            
            # Stop ROS2 spinning if it's running
            if hasattr(self, 'ros_thread') and self.ros_thread.is_alive():
                logger.info("Stopping ROS2 thread...")
                
            # Destroy ROS2 node
            if hasattr(self, 'ros_node') and self.ros_node:
                logger.info("Destroying ROS2 node...")
                try:
                    self.ros_node.destroy_node()
                except Exception as e:
                    logger.warning(f"Error destroying node: {e}")
                
            # Shutdown ROS2
            if rclpy.ok():
                logger.info("Shutting down ROS2...")
                try:
                    rclpy.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down ROS2: {e}")
            
            # Try to kill the launch process that started us
            try:
                import os
                import signal
                # Find and kill the launch process
                os.system("pkill -f 'client_system.launch.py' 2>/dev/null")
                logger.info("🛑 Attempted to stop launch process")
            except Exception as e:
                logger.warning(f"Could not stop launch process: {e}")
                
            logger.info("✅ Robot GUI shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
            # Don't re-raise the exception to avoid hanging
    
    def _create_system_status_section(self, parent):
        """Create system status section"""
        try:
            # Create a three-column layout for status items
            info_frame = ttk.Frame(parent)
            info_frame.pack(fill=tk.X)
            
            # Left column: Robot and VILA status
            left_frame = ttk.Frame(info_frame)
            left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Create composite Robot status
            robot_status_frame = ttk.Frame(left_frame)
            robot_status_frame.pack(anchor=tk.W, pady=2)
            
            robot_prefix = ttk.Label(robot_status_frame, text="Robot:", foreground="black")
            robot_prefix.pack(side=tk.LEFT)
            
            self.ros_status_label = ttk.Label(robot_status_frame, text="Offline", foreground="red", font=("Arial", 10, "bold"))
            self.ros_status_label.pack(side=tk.LEFT, padx=(5, 0))
            
            # Create composite VILA status
            vila_status_frame = ttk.Frame(left_frame)
            vila_status_frame.pack(anchor=tk.W, pady=2)
            
            vila_prefix = ttk.Label(vila_status_frame, text="VILA:", foreground="black")
            vila_prefix.pack(side=tk.LEFT)
            
            self.vila_status_label = ttk.Label(vila_status_frame, text="Offline", foreground="red", font=("Arial", 10, "bold"))
            self.vila_status_label.pack(side=tk.LEFT, padx=(5, 0))
            
            # Middle column: LiDAR readings (each on separate lines)
            middle_frame = ttk.Frame(info_frame)
            middle_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
            
            # LiDAR title
            ttk.Label(middle_frame, text="LiDAR:", foreground="black", font=("Arial", 10, "bold")).pack(anchor=tk.CENTER)
            
            # Front distance
            lidar_front_frame = ttk.Frame(middle_frame)
            lidar_front_frame.pack(anchor=tk.CENTER, pady=1)
            ttk.Label(lidar_front_frame, text="Front: ", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
            self.lidar_front_label = ttk.Label(lidar_front_frame, text="x.xm", foreground="blue", font=("Arial", 10, "bold"))
            self.lidar_front_label.pack(side=tk.LEFT)
            
            # Left distance
            lidar_left_frame = ttk.Frame(middle_frame)
            lidar_left_frame.pack(anchor=tk.CENTER, pady=1)
            ttk.Label(lidar_left_frame, text="Left: ", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
            self.lidar_left_label = ttk.Label(lidar_left_frame, text="x.xm", foreground="blue", font=("Arial", 10, "bold"))
            self.lidar_left_label.pack(side=tk.LEFT)
            
            # Right distance
            lidar_right_frame = ttk.Frame(middle_frame)
            lidar_right_frame.pack(anchor=tk.CENTER, pady=1)
            ttk.Label(lidar_right_frame, text="Right: ", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
            self.lidar_right_label = ttk.Label(lidar_right_frame, text="x.xm", foreground="blue", font=("Arial", 10, "bold"))
            self.lidar_right_label.pack(side=tk.LEFT)
            
            # Right column: Battery, CPU %, CPU temp
            right_frame = ttk.Frame(info_frame)
            right_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            
            self.battery_label = ttk.Label(right_frame, text="Battery: x.xxV", font=("Arial", 10, "bold"))
            self.battery_label.pack(anchor=tk.E)
            
            self.cpu_label = ttk.Label(right_frame, text="CPU: xx%", font=("Arial", 10, "bold"))
            self.cpu_label.pack(anchor=tk.E)
            
            # Add CPU temperature under CPU usage
            self.cpu_temp_label = ttk.Label(right_frame, text="CPU: xx.x°C", font=("Arial", 10, "bold"))
            self.cpu_temp_label.pack(anchor=tk.E)
            
            print("🔧 DEBUG: System status section created")
        except Exception as e:
            print(f"🔧 DEBUG: Error in system status section: {e}")
    
    def _create_movement_section(self, parent):
        """Create movement controls section"""
        try:
            # Safety toggle
            self.safety_var = tk.BooleanVar(value=False)
            safety_cb = ttk.Checkbutton(parent, text="Movement Enable", 
                                       variable=self.safety_var,
                                       command=self._toggle_movement_safety)
            safety_cb.pack(pady=5)
            
            # Movement buttons in grid
            button_frame = ttk.Frame(parent)
            button_frame.pack(pady=10)
            
            # Store button references for enable/disable functionality
            self.movement_buttons = []
            
            # Forward (very compact buttons to fit frame)
            forward_btn = ttk.Button(button_frame, text="↑", width=3,
                                    command=lambda: self._send_movement_command("move_forward"))
            forward_btn.grid(row=0, column=1, pady=1, padx=1)
            self.movement_buttons.append(forward_btn)
            
            # Left and Right
            left_btn = ttk.Button(button_frame, text="←", width=3,
                                 command=lambda: self._send_movement_command("turn_left"))
            left_btn.grid(row=1, column=0, padx=1, pady=1)
            self.movement_buttons.append(left_btn)
            
            right_btn = ttk.Button(button_frame, text="→", width=3,
                                  command=lambda: self._send_movement_command("turn_right"))
            right_btn.grid(row=1, column=2, padx=1, pady=1)
            self.movement_buttons.append(right_btn)
            
            # Backward
            backward_btn = ttk.Button(button_frame, text="↓", width=3,
                                     command=lambda: self._send_movement_command("move_backward"))
            backward_btn.grid(row=2, column=1, pady=1, padx=1)
            self.movement_buttons.append(backward_btn)
            
            # Stop button (center) - always enabled for safety
            stop_btn = ttk.Button(button_frame, text="⏹", width=3,
                                 command=lambda: self._send_movement_command("stop"))
            stop_btn.grid(row=1, column=1, pady=1, padx=1)
            
            # Emergency stop - always enabled for safety (compact for small frame)
            emergency_btn = ttk.Button(parent, text="STOP", width=8,
                                      command=self._emergency_stop)
            emergency_btn.pack(pady=2)
            
            # Initially disable movement buttons (safety starts as False)
            self._update_button_states(False)
            
            print("🔧 DEBUG: Movement section created")
        except Exception as e:
            print(f"🔧 DEBUG: Error in movement section: {e}")
    
    def _create_camera_section(self, parent):
        """Create camera feed section"""
        try:
            # Camera source selection
            source_frame = ttk.Frame(parent)
            source_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(source_frame, text="Source:").pack(side=tk.LEFT)
            
            self.camera_source_var = tk.StringVar(value="robot")
            ttk.Radiobutton(source_frame, text="Robot", variable=self.camera_source_var,
                           value="robot", command=self._on_camera_source_change).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(source_frame, text="Loaded", variable=self.camera_source_var,
                           value="loaded", command=self._on_camera_source_change).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(source_frame, text="📁 Load", 
                      command=self._load_image_file).pack(side=tk.RIGHT)
            
            # Camera display (sized for native camera feed)
            self.camera_label = ttk.Label(parent, text="No camera feed", 
                                         relief=tk.SUNKEN, anchor=tk.CENTER)
            self.camera_label.pack(fill=tk.BOTH, expand=True, pady=5)
            
            print("🔧 DEBUG: Camera section created")
        except Exception as e:
            print(f"🔧 DEBUG: Error in camera section: {e}")
    
    def _create_vila_section(self, parent):
        """Create VILA analysis section"""
        try:
            # VILA status moved to System Status frame - no longer needed here
            # Store reference to parent for greying out when offline
            self.vila_analysis_frame = parent
            
            # Auto analysis toggle
            self.vila_auto_var = tk.BooleanVar(value=False)
            auto_cb = ttk.Checkbutton(parent, text="Auto Analysis (5s)",
                                     variable=self.vila_auto_var,
                                     command=self._toggle_vila_auto_analysis)
            auto_cb.pack()
            
            # Quick analysis buttons
            quick_frame = ttk.Frame(parent)
            quick_frame.pack(pady=10)
            
            ttk.Label(quick_frame, text="Quick:").pack(anchor=tk.W)
            ttk.Button(quick_frame, text="🚦 Navigation", width=15,
                      command=lambda: self._quick_vila_analysis("navigation")).pack(pady=2)
            ttk.Button(quick_frame, text="🔍 Objects", width=15,
                      command=lambda: self._quick_vila_analysis("objects")).pack(pady=2)
            
            # Custom prompt
            ttk.Label(parent, text="Custom Prompt:").pack(anchor=tk.W, pady=(10, 0))
            self.vila_prompt_text = tk.Text(parent, height=3, width=30, wrap=tk.WORD)
            self.vila_prompt_text.pack(fill=tk.X, pady=5)
            self.vila_prompt_text.insert(tk.END, "Analyze this image for navigation.")
            
            ttk.Button(parent, text="Analyze", 
                      command=self._request_vila_analysis).pack(pady=5)
            
            # Results (compact)
            ttk.Label(parent, text="Results:").pack(anchor=tk.W)
            self.vila_results_text = scrolledtext.ScrolledText(parent, height=8, width=30, wrap=tk.WORD)
            self.vila_results_text.pack(fill=tk.BOTH, expand=True, pady=5)
            
            print("🔧 DEBUG: VILA section created")
        except Exception as e:
            print(f"🔧 DEBUG: Error in VILA section: {e}")
    
    def _create_log_section(self, parent):
        """Create activity log section"""
        try:
            self.log_text = scrolledtext.ScrolledText(parent, height=6, wrap=tk.WORD)
            self.log_text.pack(fill=tk.BOTH, expand=True)
            
            # Add initial message
            # Move initialization messages to debug
            logger.debug("🤖 Robot GUI initialized - Single page design")
            logger.debug("🔗 Connected to standalone VILA server")
            # Show a single user-friendly startup message
            self.log_message("🚀 Robot Control System Ready")
            
            print("🔧 DEBUG: Log section created")
        except Exception as e:
            print(f"🔧 DEBUG: Error in log section: {e}")
    
    def _test_vila_connection(self):
        """Test connection to VILA server"""
        try:
            logger.debug("🔍 Testing VILA server connection...")
            # Show user-friendly message instead
            self.log_message("🔍 Testing VILA connection...")
            if self.ros_node and hasattr(self.ros_node, 'request_vila_analysis'):
                success = self.ros_node.request_vila_analysis("Test connection")
                if success:
                    self.log_message("✅ VILA connection test sent")
                else:
                    self.log_message("❌ VILA connection test failed")
            else:
                self.log_message("⚠️ ROS node not available for testing")
        except Exception as e:
            self.log_message(f"❌ Error testing VILA connection: {e}")

def main():
    """Main application entry point"""
    import signal
    import sys

    print("🔧 DEBUG: main() started")
    root = tk.Tk()
    print("🔧 DEBUG: Tkinter root created")
    app = None
    
    def signal_handler(signum, frame):
        """Handle Ctrl-C (SIGINT) gracefully"""
        logger.info("⚠️ Received interrupt signal (Ctrl-C)")
        try:
            if app:
                app.cleanup()
            root.quit()
            root.destroy()
        except Exception as e:
            logger.error(f"Error during signal cleanup: {e}")
        finally:
            sys.exit(0)
    
    # Register signal handler for Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("🔧 DEBUG: About to create RobotGUIROS2")
        # Test with minimal RobotGUIROS2
        print("🔧 DEBUG: Testing minimal RobotGUIROS2")
        app = RobotGUIROS2(root)
        print("🔧 DEBUG: RobotGUIROS2 created successfully")
        
        # Handle window closing
        def on_closing():
            logger.info("🚪 Window close requested")
            try:
                app.cleanup()
                root.quit()  # Exit mainloop
                root.destroy()  # Destroy window
            except Exception as e:
                logger.error(f"Error during window close: {e}")
                # Force quit even if cleanup fails
                try:
                    root.quit()
                    root.destroy()
                except:
                    pass
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start GUI main loop
        print("🔧 DEBUG: About to start mainloop")
        logger.info("🖥️ Starting Robot GUI main loop...")
        root.mainloop()
        print("🔧 DEBUG: mainloop completed")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Application interrupted by user")
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
    finally:
        try:
            if app:
                app.cleanup()
        except Exception as e:
            logger.error(f"❌ Final cleanup error: {e}")
        
        # Force exit if ROS2 is still running
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except:
                pass
                
        logger.info("👋 Robot GUI application exited")

if __name__ == '__main__':
    main()

