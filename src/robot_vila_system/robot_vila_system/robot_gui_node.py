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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

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
            '/robot/camera/image_raw',
            self._image_callback,
            self.best_effort_qos
        )
        
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
            '/vila/analyze'
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
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Convert to PIL Image for GUI
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Send to GUI
            self.gui_callback('camera_image', pil_image)
            
        except Exception as e:
            self.get_logger().error(f"Error processing camera image: {e}")
    
    def _vila_analysis_callback(self, msg: VILAAnalysis):
        """Handle VILA analysis results"""
        try:
            # Track VILA activity for model state display
            self.last_vila_update = time.time()
            
            analysis_data = {
                'robot_id': msg.robot_id,
                'prompt': msg.prompt,
                'analysis_result': msg.analysis_result,
                'navigation_commands': json.loads(msg.navigation_commands_json) if msg.navigation_commands_json else {},
                'confidence': msg.confidence,
                'timestamp': msg.timestamp_ns,
                'success': msg.success,
                'error_message': msg.error_message
            }
            
            # Send to GUI
            self.gui_callback('vila_analysis', analysis_data)
            
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
                'parameters': json.loads(msg.parameters_json) if msg.parameters_json else {},
                'timestamp': msg.timestamp_ns,
                'source_node': msg.source_node
            }
            
            self.gui_callback('command_ack', ack_data)
            
        except Exception as e:
            self.get_logger().error(f"Error processing command ack: {e}")
    
    def send_robot_command(self, command_type: str, parameters: Dict = None, safety_confirmed: bool = False):
        """Send robot command through single gateway [[memory:5366669]]"""
        try:
            if not self.execute_command_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("Execute command service not available")
                return False
            
            # Create command message
            command_msg = RobotCommand()
            command_msg.robot_id = self.robot_id
            command_msg.command_type = command_type
            
            # Set movement parameters based on command type
            if command_type == "move_forward":
                command_msg.linear_x = parameters.get('speed', 0.2)
                command_msg.duration = parameters.get('duration', 1.0)
            elif command_type == "move_backward":
                command_msg.linear_x = -parameters.get('speed', 0.2)
                command_msg.duration = parameters.get('duration', 1.0)
            elif command_type == "turn_left":
                command_msg.angular_z = parameters.get('speed', 0.5)
                command_msg.duration = parameters.get('duration', 1.0)
            elif command_type == "turn_right":
                command_msg.angular_z = -parameters.get('speed', 0.5)
                command_msg.duration = parameters.get('duration', 1.0)
            elif command_type == "stop":
                command_msg.linear_x = 0.0
                command_msg.linear_y = 0.0
                command_msg.angular_z = 0.0
                command_msg.duration = 0.0
            
            command_msg.parameters_json = json.dumps(parameters or {})
            command_msg.timestamp_ns = self.get_clock().now().nanoseconds
            command_msg.source_node = "robot_gui_node"
            
            # Create service request
            request = ExecuteCommand.Request()
            request.command = command_msg
            
            # Call service
            future = self.execute_command_client.call_async(request)
            future.add_done_callback(lambda f: self._command_response_callback(f, command_type))
            
            self.get_logger().info(f"🎯 GUI command sent through gateway: {command_type}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error sending robot command: {e}")
            return False
    
    def _command_response_callback(self, future, command_type: str):
        """Handle command response from gateway"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"✅ Gateway accepted GUI command: {command_type}")
                self.gui_callback('command_success', {'command': command_type, 'message': response.message})
            else:
                self.get_logger().warn(f"🚫 Gateway rejected GUI command: {command_type} - {response.message}")
                self.gui_callback('command_error', {'command': command_type, 'message': response.message})
        except Exception as e:
            self.get_logger().error(f"Error in command response: {e}")
    
    def request_vila_analysis(self, prompt: str, image: RosImage = None):
        """Request VILA analysis"""
        try:
            if not self.vila_analysis_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("VILA analysis service not available")
                return False
            
            request = RequestVILAAnalysis.Request()
            request.robot_id = self.robot_id
            request.prompt = prompt
            if image:
                request.image = image
            
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

class RobotGUIROS2:
    """Main GUI class using ROS2 communication"""
    
    def __init__(self, root):
        self.root = root
        self.base_title = "Robot Control Hub - ROS2 VILA Integration"
        self.root.title(self.base_title)
        self.root.geometry("1200x800")
        
        # Initialize ROS2
        rclpy.init()
        self.ros_node = RobotGUIROS2Node(self._ros_callback)
        
        # GUI state
        self.robot_id = "yahboomcar_x3_01"
        self.robot_data = {}
        self.sensor_data = {}
        self.current_image = None
        self.loaded_image = None  # For manually loaded images
        self.vila_analysis = {}
        self.safety_enabled = True
        self.movement_enabled = True
        self.vila_analysis_enabled = True  # New: VILA analysis switch
        self.camera_source = "robot"  # "robot", "sim", or "loaded"
        self.update_queue = queue.Queue()
        
        # Create GUI
        self._create_gui()
        
        # Start ROS2 spinning in background
        self.ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.ros_thread.start()
        
        # Start GUI update processing
        self._process_updates()
        
        # Start system status updates (every 1 second)
        self._update_system_status()
        
        self.log_message("🤖 Robot GUI ROS2 initialized")
        self.log_message("   └── All communication via ROS2 topics and services")
        self.log_message("   └── Single command gateway maintained [[memory:5366669]]")
    
    def _ros_callback(self, message_type: str, data):
        """Callback from ROS2 node to update GUI"""
        self.update_queue.put((message_type, data))
    
    def _spin_ros(self):
        """Spin ROS2 node in background thread"""
        try:
            rclpy.spin(self.ros_node)
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
                self.sensor_data['vila_model_state'] = self._get_vila_model_state()
                
                # Update robot status based on current sensor data
                self.sensor_data['robot_status'] = self._determine_robot_status(self.sensor_data)
                
                # Update system status labels
                for key, (label, unit) in self.sensor_labels.items():
                    if key in self.sensor_data:
                        value = self.sensor_data[key]
                        if isinstance(value, (int, float)):
                            label.config(text=f"{value:.2f} {unit}")
                        else:
                            # Special formatting for robot status with colors
                            if key == 'robot_status':
                                status_text = f"{value} {unit}".strip()
                                if value == 'Running':
                                    label.config(text=status_text, foreground="green")
                                elif value in ['Offline', 'Error']:
                                    label.config(text=status_text, foreground="red")
                                elif value in ['Stale', 'Low Battery', 'High Voltage']:
                                    label.config(text=status_text, foreground="orange")
                                else:
                                    label.config(text=status_text, foreground="black")
                            else:
                                label.config(text=f"{value} {unit}")
        except Exception as e:
            logger.error(f"Error updating system status: {e}")
        
        # Schedule next system status update (1000ms = 1 second)
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
        except Exception as e:
            logger.error(f"Error handling ROS update {message_type}: {e}")
    
    def _create_gui(self):
        """Create the main GUI"""
        # Main container with tabs on left and system status on right
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs (left side)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # System Status frame (always visible on right)
        self.system_status_frame = ttk.LabelFrame(main_container, text="System Status", padding=10)
        self.system_status_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        self._create_system_status_panel()
        
        # Control tab
        self.control_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.control_frame, text="Robot Control")
        self._create_control_tab()
        
        # Monitoring tab
        self.monitoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.monitoring_frame, text="Monitoring")
        self._create_monitoring_tab()
        
        # VILA tab
        self.vila_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.vila_frame, text="VILA Analysis")
        self._create_vila_tab()
        
        # Status bar
        self._create_status_bar()
    
    def _create_system_status_panel(self):
        """Create the always-visible system status panel"""
        # Configure grid column weights to prevent resizing
        self.system_status_frame.grid_columnconfigure(0, weight=0, minsize=120)
        self.system_status_frame.grid_columnconfigure(1, weight=1, minsize=100)
        
        # System status displays
        self.sensor_labels = {}
        sensor_names = [
            ("Robot Status", "robot_status", ""),
            ("Battery Voltage", "battery_voltage", "V"),
            ("CPU Temp", "cpu_temp", "°C"),
            ("CPU Usage", "cpu_usage", "%"),
            ("Distance Front", "distance_front", "m"),
            ("Distance Left", "distance_left", "m"),
            ("Distance Right", "distance_right", "m"),
            ("IMU Accel X", "imu_accel_x", "m/s²"),
            ("IMU Accel Y", "imu_accel_y", "m/s²"),
            ("IMU Accel Z", "imu_accel_z", "m/s²"),
            ("VILA Model", "vila_model_state", "")
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
        """Create robot control tab"""
        # Robot info frame
        info_frame = ttk.LabelFrame(self.control_frame, text="Robot Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Robot ID
        ttk.Label(info_frame, text="Robot ID:").grid(row=0, column=0, sticky=tk.W)
        self.robot_id_label = ttk.Label(info_frame, text=self.robot_id, font=("Arial", 10, "bold"))
        self.robot_id_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        # Connection status
        ttk.Label(info_frame, text="Connection:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.connection_label = ttk.Label(info_frame, text="ROS2 Active", foreground="green")
        self.connection_label.grid(row=0, column=3, sticky=tk.W, padx=10)
        
        # Movement controls frame
        movement_frame = ttk.LabelFrame(self.control_frame, text="Movement Controls", padding=10)
        movement_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Safety toggle - Start with movement disabled
        self.safety_var = tk.BooleanVar(value=False)
        self.safety_checkbox = ttk.Checkbutton(
            movement_frame, 
            text="Movement Enabled", 
            variable=self.safety_var,
            command=self._toggle_movement_safety
        )
        self.safety_checkbox.pack(anchor=tk.W)
        
        # Movement buttons
        button_frame = ttk.Frame(movement_frame)
        button_frame.pack(pady=10)
        
        # Forward button
        ttk.Button(button_frame, text="↑ Forward", 
                  command=lambda: self._send_movement_command("move_forward")).grid(row=0, column=1, padx=5, pady=5)
        
        # Left and right buttons
        ttk.Button(button_frame, text="← Left", 
                  command=lambda: self._send_movement_command("turn_left")).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="→ Right", 
                  command=lambda: self._send_movement_command("turn_right")).grid(row=1, column=2, padx=5, pady=5)
        
        # Stop button
        stop_btn = ttk.Button(button_frame, text="⏹ STOP", 
                             command=lambda: self._send_movement_command("stop"))
        stop_btn.grid(row=1, column=1, padx=5, pady=5)
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
        """Create monitoring tab"""
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
        """Create VILA analysis tab"""
        # VILA controls frame
        controls_frame = ttk.LabelFrame(self.vila_frame, text="VILA Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # VILA analysis enable/disable switch
        self.vila_analysis_var = tk.BooleanVar(value=True)
        vila_switch = ttk.Checkbutton(controls_frame, text="🔍 Enable VILA Analysis", 
                                     variable=self.vila_analysis_var,
                                     command=self._on_vila_analysis_toggle)
        vila_switch.pack(side=tk.LEFT, padx=5)
        
        # Status indicator
        self.vila_status_label = ttk.Label(controls_frame, text="🟢 VILA Active", foreground="green")
        self.vila_status_label.pack(side=tk.LEFT, padx=10)
        
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
        
        # ROS2 status indicator
        self.ros_status = ttk.Label(self.status_bar, text="🟢 ROS2", foreground="green")
        self.ros_status.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def _send_movement_command(self, command_type: str):
        """Send movement command through ROS2"""
        if not self.safety_var.get():
            messagebox.showwarning("Safety", "Movement is disabled. Enable movement first.")
            return
        
        parameters = {
            'speed': 0.3,
            'duration': 1.0
        }
        
        success = self.ros_node.send_robot_command(
            command_type, 
            parameters, 
            safety_confirmed=True
        )
        
        if success:
            self.log_message(f"🎯 Command sent: {command_type}")
        else:
            self.log_message(f"❌ Failed to send command: {command_type}")
    
    def _toggle_movement_safety(self):
        """Toggle movement safety"""
        enabled = self.safety_var.get()
        self.ros_node.set_safety_enabled(enabled)
        self.log_message(f"🛡️ Movement {'ENABLED' if enabled else 'DISABLED'}")
    
    def _emergency_stop(self):
        """Trigger emergency stop"""
        self.ros_node.emergency_stop()
        self.safety_var.set(False)
        self.log_message("🚨 EMERGENCY STOP ACTIVATED")
    
    def _request_vila_analysis(self):
        """Request VILA analysis"""
        prompt = self.vila_prompt_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("Input Error", "Please enter a prompt for analysis.")
            return
        
        success = self.ros_node.request_vila_analysis(prompt)
        if success:
            self.log_message(f"📝 VILA analysis requested: {prompt[:50]}...")
        else:
            self.log_message("❌ Failed to request VILA analysis")
    
    def _update_robot_status(self, data):
        """Update robot status display"""
        self.robot_data = data
        
        # Update connection status
        self.connection_label.config(text="ROS2 Connected", foreground="green")
        
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
            # Check if we have recent VILA analysis (within last 30 seconds)
            if hasattr(self.ros_node, 'last_vila_update') and self.ros_node.last_vila_update:
                time_diff = time.time() - self.ros_node.last_vila_update
                if time_diff < 30:
                    return "Active"
                else:
                    return "Idle"
            else:
                return "Ready"
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
            
            # Only display if camera source is set to robot
            if self.camera_source == "robot":
                self._update_camera_display(pil_image)
            
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
        """Update safety status"""
        self.safety_enabled = enabled
        self.safety_var.set(enabled)
        self.log_message(f"🛡️ Safety status: {'ENABLED' if enabled else 'DISABLED'}")
    
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
        timestamp = datetime.fromtimestamp(data['timestamp'] / 1e9).strftime("%H:%M:%S")
        analysis_text = f"[{timestamp}] Custom VILA Analysis:\n"
        analysis_text += f"Confidence: {data['confidence']:.2f}\n"
        analysis_text += f"Result: {data['analysis_result']}\n"
        analysis_text += "-" * 50 + "\n"
        
        self.vila_results_text.insert(tk.END, analysis_text)
        self.vila_results_text.see(tk.END)
    
    def _handle_vila_analysis_error(self, data):
        """Handle VILA analysis error"""
        error_text = f"❌ VILA Analysis Error: {data['error']}\n"
        error_text += "-" * 50 + "\n"
        
        self.vila_results_text.insert(tk.END, error_text)
        self.vila_results_text.see(tk.END)
    
    def log_message(self, message):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
    
    def _on_vila_analysis_toggle(self):
        """Handle VILA analysis enable/disable toggle"""
        self.vila_analysis_enabled = self.vila_analysis_var.get()
        status_text = "🟢 VILA Active" if self.vila_analysis_enabled else "🔴 VILA Disabled"
        status_color = "green" if self.vila_analysis_enabled else "red"
        self.vila_status_label.config(text=status_text, foreground=status_color)
        
        action = "enabled" if self.vila_analysis_enabled else "disabled"
        self.log_message(f"🔍 VILA analysis {action}")
        
        # TODO: Send ROS2 message to vila_vision_node to enable/disable analysis
    
    def _on_camera_source_change(self):
        """Handle camera source change"""
        self.camera_source = self.camera_source_var.get()
        self.log_message(f"📷 Camera source changed to: {self.camera_source}")
        
        # Update display based on source
        if self.camera_source == "loaded" and self.loaded_image:
            self._update_camera_display(self.loaded_image)
        elif self.camera_source == "robot" and self.current_image:
            self._update_camera_display(self.current_image)
        elif self.camera_source == "sim":
            # TODO: Load simulator image if available
            self.log_message("📷 Simulator camera not yet implemented")
    
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
                self.log_message(f"📁 Loaded image: {file_path}")
                
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
            # Resize image to fit display (max 400x300)
            display_image = pil_image.copy()
            display_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            
            # Convert to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(display_image)
            
            # Update label
            self.camera_label.config(image=photo, text="")
            self.camera_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.log_message(f"❌ Error updating camera display: {e}")
    
    def cleanup(self):
        """Cleanup ROS2 resources and shutdown entire system"""
        try:
            logger.info("🔄 Shutting down Robot GUI and entire system...")
            
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

def main():
    """Main application entry point"""
    import signal
    import sys
    
    root = tk.Tk()
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
        app = RobotGUIROS2(root)
        
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
        logger.info("🖥️ Starting Robot GUI main loop...")
        root.mainloop()
        
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
