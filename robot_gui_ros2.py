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
from robot_msgs.msg import RobotCommand, SensorData, VILAAnalysis, RobotStatus
from robot_msgs.srv import ExecuteCommand, RequestVILAAnalysis
from sensor_msgs.msg import Image as RosImage
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
        # Robot status
        self.status_subscriber = self.create_subscription(
            RobotStatus,
            f'/robot/{self.robot_id}/status',
            self._robot_status_callback,
            self.reliable_qos
        )
        
        # Sensor data
        self.sensor_subscriber = self.create_subscription(
            SensorData,
            f'/robot/{self.robot_id}/sensors',
            self._sensor_data_callback,
            self.best_effort_qos
        )
        
        # Camera images
        self.image_subscriber = self.create_subscription(
            RosImage,
            f'/robot/{self.robot_id}/camera/image_raw',
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
            '/vila/navigation_commands',
            self._navigation_commands_callback,
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
            f'/robot/{self.robot_id}/command_ack',
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
            '/vila/analyze_image'
        )
    
    def _robot_status_callback(self, msg: RobotStatus):
        """Handle robot status updates"""
        try:
            status_data = {
                'robot_id': msg.robot_id,
                'name': msg.name,
                'last_seen': msg.last_seen_ns,
                'battery_level': msg.battery_level,
                'status': msg.status,
                'capabilities': msg.capabilities,
                'connection_type': msg.connection_type,
                'last_command': msg.last_command,
                'command_history': msg.command_history
            }
            
            if msg.sensor_data.robot_id:  # Check if sensor data is present
                status_data['sensor_data'] = {
                    'battery_voltage': msg.sensor_data.battery_voltage,
                    'battery_percentage': msg.sensor_data.battery_percentage,
                    'temperature': msg.sensor_data.temperature,
                    'humidity': msg.sensor_data.humidity,
                    'distance_front': msg.sensor_data.distance_front,
                    'distance_left': msg.sensor_data.distance_left,
                    'distance_right': msg.sensor_data.distance_right,
                    'wifi_signal': msg.sensor_data.wifi_signal,
                    'cpu_usage': msg.sensor_data.cpu_usage,
                    'memory_usage': msg.sensor_data.memory_usage,
                    'camera_status': msg.sensor_data.camera_status,
                    'imu_values': {
                        'x': msg.sensor_data.imu_values.x,
                        'y': msg.sensor_data.imu_values.y,
                        'z': msg.sensor_data.imu_values.z
                    }
                }
            
            # Send to GUI
            self.gui_callback('robot_status', status_data)
            
        except Exception as e:
            self.get_logger().error(f"Error processing robot status: {e}")
    
    def _sensor_data_callback(self, msg: SensorData):
        """Handle sensor data updates"""
        try:
            sensor_data = {
                'robot_id': msg.robot_id,
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
                'camera_status': msg.camera_status,
                'imu_values': {
                    'x': msg.imu_values.x,
                    'y': msg.imu_values.y,
                    'z': msg.imu_values.z
                },
                'timestamp': msg.timestamp_ns
            }
            
            # Send to GUI
            self.gui_callback('sensor_data', sensor_data)
            
        except Exception as e:
            self.get_logger().error(f"Error processing sensor data: {e}")
    
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
                'parameters': json.loads(msg.parameters_json) if msg.parameters_json else {},
                'timestamp': msg.timestamp_ns,
                'success': msg.safety_confirmed,  # Using safety_confirmed as success indicator
                'source': msg.source
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
            command_msg.parameters_json = json.dumps(parameters or {})
            command_msg.timestamp_ns = self.get_clock().now().nanoseconds
            command_msg.priority = 1
            command_msg.safety_confirmed = safety_confirmed
            command_msg.gui_movement_enabled = True  # GUI-initiated command
            command_msg.source = "GUI"
            
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
        self.vila_analysis = {}
        self.safety_enabled = True
        self.movement_enabled = True
        self.update_queue = queue.Queue()
        
        # Create GUI
        self._create_gui()
        
        # Start ROS2 spinning in background
        self.ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.ros_thread.start()
        
        # Start GUI update processing
        self._process_updates()
        
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
        """Process updates from ROS2 callbacks"""
        try:
            while not self.update_queue.empty():
                message_type, data = self.update_queue.get_nowait()
                self._handle_ros_update(message_type, data)
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing updates: {e}")
        
        # Schedule next update
        self.root.after(50, self._process_updates)
    
    def _handle_ros_update(self, message_type: str, data):
        """Handle different types of ROS2 updates"""
        try:
            if message_type == 'robot_status':
                self._update_robot_status(data)
            elif message_type == 'sensor_data':
                self._update_sensor_data(data)
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
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
        
        # Battery status
        ttk.Label(info_frame, text="Battery:").grid(row=1, column=0, sticky=tk.W)
        self.battery_label = ttk.Label(info_frame, text="---%")
        self.battery_label.grid(row=1, column=1, sticky=tk.W, padx=10)
        
        # Robot status
        ttk.Label(info_frame, text="Status:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0))
        self.status_label = ttk.Label(info_frame, text="Unknown")
        self.status_label.grid(row=1, column=3, sticky=tk.W, padx=10)
        
        # Movement controls frame
        movement_frame = ttk.LabelFrame(self.control_frame, text="Movement Controls", padding=10)
        movement_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Safety toggle
        self.safety_var = tk.BooleanVar(value=True)
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
        # Camera frame
        camera_frame = ttk.LabelFrame(self.monitoring_frame, text="Camera Feed", padding=10)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.camera_label = ttk.Label(camera_frame, text="No camera feed")
        self.camera_label.pack(expand=True)
        
        # Sensor frame
        sensor_frame = ttk.LabelFrame(self.monitoring_frame, text="Sensor Data", padding=10)
        sensor_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Sensor displays
        self.sensor_labels = {}
        sensor_names = [
            ("Battery Voltage", "battery_voltage", "V"),
            ("Battery %", "battery_percentage", "%"),
            ("Temperature", "temperature", "°C"),
            ("Humidity", "humidity", "%"),
            ("Distance Front", "distance_front", "m"),
            ("Distance Left", "distance_left", "m"),
            ("Distance Right", "distance_right", "m"),
            ("CPU Usage", "cpu_usage", "%"),
            ("Memory Usage", "memory_usage", "%"),
            ("WiFi Signal", "wifi_signal", "dBm")
        ]
        
        for i, (name, key, unit) in enumerate(sensor_names):
            ttk.Label(sensor_frame, text=f"{name}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            label = ttk.Label(sensor_frame, text="---")
            label.grid(row=i, column=1, sticky=tk.W, padx=10, pady=2)
            self.sensor_labels[key] = (label, unit)
    
    def _create_vila_tab(self):
        """Create VILA analysis tab"""
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
        
        # Update labels
        if 'battery_level' in data:
            self.battery_label.config(text=f"{data['battery_level']:.1f}%")
        
        if 'status' in data:
            self.status_label.config(text=data['status'].title())
        
        # Update connection status
        self.connection_label.config(text="ROS2 Connected", foreground="green")
    
    def _update_sensor_data(self, data):
        """Update sensor data display"""
        self.sensor_data = data
        
        # Update sensor labels
        for key, (label, unit) in self.sensor_labels.items():
            if key in data:
                value = data[key]
                if isinstance(value, (int, float)):
                    label.config(text=f"{value:.2f} {unit}")
                else:
                    label.config(text=f"{value} {unit}")
    
    def _update_camera_image(self, pil_image):
        """Update camera image display"""
        try:
            self.current_image = pil_image
            
            # Resize for display
            display_size = (320, 240)
            display_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
            
            # Convert to Tkinter format
            photo = ImageTk.PhotoImage(display_image)
            
            # Update label
            self.camera_label.config(image=photo, text="")
            self.camera_label.image = photo  # Keep a reference
            
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
    
    def cleanup(self):
        """Cleanup ROS2 resources"""
        try:
            self.ros_node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    try:
        app = RobotGUIROS2(root)
        
        # Handle window closing
        def on_closing():
            app.cleanup()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
        
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        try:
            if 'app' in locals():
                app.cleanup()
        except:
            pass

if __name__ == '__main__':
    main()
