#!/usr/bin/env python3
"""
Robot Control GUI - Clean Component-Based Version
Tkinter application for monitoring and controlling robots via ROS2 topics
Uses standard ROS2 messages (cmd_vel) for robot control
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
from typing import Dict
import queue
from PIL import Image

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# ROS2 message imports
from robot_msgs.msg import SensorData
from sensor_msgs.msg import Image as RosImage, Imu
from std_msgs.msg import String
from geometry_msgs.msg import Twist

# GUI component imports
from .gui_config import GUIConfig
from .gui_components import (
    SystemStatusPanel,
    MovementControlPanel,
    CameraPanel,
    VLMAnalysisPanel,
    ActivityLogPanel
)


class RobotGUIROS2Node(Node):
    """ROS2 node for the robot GUI"""
    
    def __init__(self, gui_callback):
        super().__init__('robot_gui_internal')  # Use different name to avoid conflict with launch file

        self.gui_callback = gui_callback
        self.latest_imu_data = None
        self.current_camera_image = None
        self.current_sensor_data = None
        
        # Movement parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.5  # rad/s
        self.strafe_speed = 0.5  # m/s for lateral movement
        
        # Movement execution tracking
        self.movement_in_progress = False
        self.movement_timer = None
        self.movement_complete_callback = None
        
        # QoS profiles - use compatible settings
        self.image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        self.best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Setup ROS2 interfaces
        self._setup_subscribers()
        self._setup_publishers()
        
        self.get_logger().info("🖥️ Robot GUI ROS2 node initialized")
        self.get_logger().info("   └── Publishing cmd_vel on /cmd_vel topic")
    

    def _setup_subscribers(self):
        """Setup ROS2 subscribers"""
        # Sensor data
        self.sensor_subscriber = self.create_subscription(
            SensorData,
            '/robot/sensors',
            self._sensor_data_callback,
            self.best_effort_qos
        )
        
        # LiDAR scan data (separate subscription)
        from sensor_msgs.msg import LaserScan
        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self._lidar_scan_callback,
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
        
        # Safety status handled locally in GUI - no ROS subscription needed
        
        # VLM Model status updates
        self.model_status_subscriber = self.create_subscription(
            String,
            '/vlm/status',
            self._model_status_callback,
            self.reliable_qos
        )
    
    def _setup_publishers(self):
        """Setup ROS2 publishers"""
        # Standard cmd_vel publisher for robot movement control
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        self.get_logger().info("📡 Created /cmd_vel publisher for robot movement control")
    
    def _sensor_data_callback(self, msg: SensorData):
        """Handle sensor data messages"""
        sensor_data = {
            'battery_voltage': msg.battery_voltage,
            'cpu_temp': msg.cpu_temp,
            'cpu_usage': msg.cpu_usage,
            'timestamp': msg.timestamp_ns
        }
        self.gui_callback('sensor_data', sensor_data)
    
    def _lidar_scan_callback(self, msg):
        """Handle LiDAR scan messages"""
        lidar_data = {
            'ranges': list(msg.ranges),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        }
        self.gui_callback('lidar_data', lidar_data)
    
    def _imu_callback(self, msg: Imu):
        """Handle IMU data messages"""
        imu_data = {
            'accel_x': msg.linear_acceleration.x,
            'accel_y': msg.linear_acceleration.y,
            'accel_z': msg.linear_acceleration.z,
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        }
        self.gui_callback('imu_data', imu_data)
    
    def _image_callback(self, msg: RosImage):
        """Handle camera image messages"""
        self.current_camera_image = msg
        self.gui_callback('camera_image', msg)

    def _navigation_commands_callback(self, msg: String):
        """Handle navigation commands"""
        self.gui_callback('navigation_commands', msg.data)

    def _model_status_callback(self, msg: String):
        """Handle VLM Model status updates"""
        try:
            import json
            status_data = json.loads(msg.data)
            self.gui_callback('model_status', status_data)
        except Exception as e:
            self.get_logger().error(f"Error parsing Model status: {e}")
    
    def send_robot_command(self, command_type: str, parameters: Dict = None, safety_confirmed: bool = False, completion_callback=None):
        """Send robot movement command via standard cmd_vel topic
        
        Executes timed movements (0.5m linear, 45° angular) with automatic stop.
        
        Args:
            command_type: Type of movement command
            parameters: Optional parameters (can include 'angle' for turns in degrees)
            safety_confirmed: Safety confirmation flag
            completion_callback: Function to call when movement completes
        """
        # Don't queue multiple movements
        if self.movement_in_progress and command_type not in ['stop', 'emergency_stop']:
            self.get_logger().warn(f"⏸️ Movement already in progress, ignoring: {command_type}")
            return False
            
        self.get_logger().info(f"🚀 Sending robot command: {command_type}")

        # Create Twist message for velocity control
        twist = Twist()
        duration_sec = None  # Duration for timed movements

        # Parse command type and set velocities
        if command_type == 'move_forward':
            speed = parameters.get('speed', self.linear_speed) if parameters else self.linear_speed
            distance = parameters.get('distance', 0.5) if parameters else 0.5  # Default 0.5m
            twist.linear.x = speed
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            # Calculate duration: time = distance / speed
            duration_sec = distance / speed
            self.get_logger().info(f"   └── Moving forward {distance}m at {speed}m/s over {duration_sec:.2f}s")

        elif command_type == 'move_backward':
            speed = parameters.get('speed', self.linear_speed) if parameters else self.linear_speed
            distance = parameters.get('distance', 0.5) if parameters else 0.5  # Default 0.5m
            twist.linear.x = -speed
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            # Calculate duration: time = distance / speed
            duration_sec = distance / speed
            self.get_logger().info(f"   └── Moving backward {distance}m at {speed}m/s over {duration_sec:.2f}s")

        elif command_type == 'turn_left':
            angular_speed = parameters.get('speed', self.angular_speed) if parameters else self.angular_speed
            import math
            angle_rad = parameters.get('angle', math.pi / 4.0) if parameters else math.pi / 4.0  # Default 45°
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = angular_speed
            # Calculate duration: time = angle / angular_speed
            duration_sec = angle_rad / abs(angular_speed)
            angle_deg = math.degrees(angle_rad)
            self.get_logger().info(f"   └── Turning left {angle_deg:.1f}° at {angular_speed}rad/s over {duration_sec:.2f}s")

        elif command_type == 'turn_right':
            angular_speed = parameters.get('speed', self.angular_speed) if parameters else self.angular_speed
            import math
            angle_rad = parameters.get('angle', math.pi / 4.0) if parameters else math.pi / 4.0  # Default 45°
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = -angular_speed
            # Calculate duration: time = angle / angular_speed
            duration_sec = angle_rad / abs(angular_speed)
            angle_deg = math.degrees(angle_rad)
            self.get_logger().info(f"   └── Turning right {angle_deg:.1f}° at {angular_speed}rad/s over {duration_sec:.2f}s")

        elif command_type == 'strafe_left':
            speed = parameters.get('speed', self.strafe_speed) if parameters else self.strafe_speed
            distance = parameters.get('distance', 0.5) if parameters else 0.5  # Default 0.5m
            twist.linear.x = 0.0
            twist.linear.y = speed
            twist.angular.z = 0.0
            # Calculate duration: time = distance / speed
            duration_sec = distance / speed
            self.get_logger().info(f"   └── Strafing left {distance}m at {speed}m/s over {duration_sec:.2f}s")

        elif command_type == 'strafe_right':
            speed = parameters.get('speed', self.strafe_speed) if parameters else self.strafe_speed
            distance = parameters.get('distance', 0.5) if parameters else 0.5  # Default 0.5m
            twist.linear.x = 0.0
            twist.linear.y = -speed
            twist.angular.z = 0.0
            # Calculate duration: time = distance / speed
            duration_sec = distance / speed
            self.get_logger().info(f"   └── Strafing right {distance}m at {speed}m/s over {duration_sec:.2f}s")
            
        elif command_type in ['stop', 'emergency_stop']:
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            # Cancel any pending movement
            if self.movement_timer:
                self.movement_timer.cancel()
                self.movement_timer = None
            self.movement_in_progress = False
        else:
            self.get_logger().warn(f"Unknown command type: {command_type}")
            return False

        # Publish velocity command
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info(f"📤 Published cmd_vel: linear=({twist.linear.x:.2f}, {twist.linear.y:.2f}), angular={twist.angular.z:.2f}")
        
        # Schedule automatic stop for timed movements
        if duration_sec and command_type not in ['stop', 'emergency_stop']:
            self.movement_in_progress = True
            self.movement_complete_callback = completion_callback
            
            # Create ROS2 timer to stop movement after duration
            self.movement_timer = self.create_timer(duration_sec, self._on_movement_complete)
            self.get_logger().info(f"⏱️ Movement will complete in {duration_sec:.2f}s")
        
        return True
    
    def _on_movement_complete(self):
        """Called when timed movement completes"""
        self.get_logger().info("✅ Movement complete - stopping robot")
        
        # Stop the robot
        stop_twist = Twist()
        self.cmd_vel_publisher.publish(stop_twist)
        
        # Cancel timer
        if self.movement_timer:
            self.movement_timer.cancel()
            self.movement_timer = None
        
        self.movement_in_progress = False
        
        # Call completion callback if provided
        if self.movement_complete_callback:
            callback = self.movement_complete_callback
            self.movement_complete_callback = None
            # Execute callback via GUI callback to ensure thread safety
            self.gui_callback('movement_complete', {})
        
        self.get_logger().info("🔓 Ready for next movement")


class RobotGUIROS2:
    """Main GUI class using component-based architecture"""
    
    def __init__(self, root, logger=None):
        self.root = root

        # Create logger if none provided
        if logger is None:
            import logging
            self.logger = logging.getLogger('RobotGUIROS2')
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
                self.logger.addHandler(handler)
        else:
            self.logger = logger

        self.logger.debug(f"🔧 DEBUG: RobotGUIROS2.__init__ entered")

        # Use configuration constants
        self.base_title = GUIConfig.WINDOW_TITLE

        # Defer ROS2 initialization until after GUI is created
        self.ros_node = None
        self.logger.debug(f"🔧 ROS2 initialization deferred until after GUI creation")

        # Initialize GUI state
        self.logger.debug(f"🔧 Initializing GUI state...")
        self._initialize_gui_state()
        self.logger.debug(f"✅ GUI state initialized")

        # Initialize GUI components
        self.logger.debug(f"🔧 Initializing GUI components...")
        self._initialize_gui_components()
        self.logger.debug(f"✅ GUI components initialized")

        # Create GUI
        self.logger.debug( f"🔧 Creating GUI...")
        self._create_gui()
        self.logger.debug( f"✅ GUI created successfully")

        # Start background processes
        self.logger.debug( f"🔧 Starting background processes...")
        self._start_background_processes()
        self.logger.debug( f"✅ Background processes started")

        # Set window title
        self.logger.debug( f"🔧 Setting window title...")
        self._set_window_title_safely()
        self.logger.debug( f"✅ Window title set")

        # Set up window close event handler
        self.logger.debug( f"🔧 Setting up window close handler...")
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self.logger.debug( f"✅ Window close handler set")

        # Log initialization
        self.logger.info( f"🤖 Robot GUI ROS2 initialized (debug mode)")
        self.logger.info( f"   └── All communication via ROS2 topics and services")
        self.logger.info( f"   └── Component-based architecture")

    def _initialize_gui_state(self):
        """Initialize GUI state variables"""
        self.logger.debug( f"🔧 DEBUG: Initializing GUI state variables")

        # Single robot setup - no robot_id variable needed
        self.robot_data = {}
        self.sensor_data = {}
        self.current_image = None
        self.movement_enabled = True
        self._cleanup_requested = False

        
        # Auto VLM analysis
        self.auto_vlm_enabled = False
        self.auto_vlm_timer = None
        self.auto_execute_enabled = False  # Default: do not auto-execute for safety
        self.pending_analysis_requests = []  # Track pending ROS2 service calls
        self.analysis_in_progress = False  # Track if an analysis is currently running

        # Threading and queues
        self.update_queue = queue.Queue()

    def _initialize_gui_components(self):
        """Initialize GUI component classes"""
        self.logger.debug( f"🔧 Initializing GUI components...")

        try:
            # Create component instances with proper callbacks
            self.logger.debug( f"🔧 Creating SystemStatusPanel...")
            self.system_panel = SystemStatusPanel(self.root, self._update_system_status, self.logger)
            self.logger.debug( f"✅ SystemStatusPanel created")

            self.logger.debug( f"🔧 Creating MovementControlPanel...")
            self.movement_panel = MovementControlPanel(self.root, self._handle_movement_command, self.logger)
            self.logger.debug( f"✅ MovementControlPanel created")

            self.logger.debug( f"🔧 Creating CameraPanel...")
            self.camera_panel = CameraPanel(self.root, self._handle_camera_update, self.logger)
            self.logger.debug( f"✅ CameraPanel created")

            self.logger.debug( f"🔧 Creating VLMAnalysisPanel...")
            self.vlm_panel = VLMAnalysisPanel(self.root, self._handle_vlm_analysis, self.log_message, self.logger)
            self.logger.debug( f"✅ VLMAnalysisPanel created")

            self.logger.debug( f"🔧 Creating ActivityLogPanel...")
            self.log_panel = ActivityLogPanel(self.root, self.logger)
            self.logger.debug( f"✅ ActivityLogPanel created")

        except Exception as e:
            self.logger.error( f"❌ Error initializing GUI components: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_gui(self):
        """Create GUI using component classes"""
        self.logger.debug( f"🔧 Creating component-based GUI design")
        
        try:
            # Configure root window first
            self.logger.debug( f"🔧 Configuring root window...")
            self.root.geometry(f"{GUIConfig.WINDOW_WIDTH}x{GUIConfig.WINDOW_HEIGHT}")
            self.root.title(self.base_title)
            self.logger.debug( f"✅ Root window configured")

            # Process any pending events
            self.logger.debug( f"🔧 Processing pending events...")
            self.root.update_idletasks()
            self.logger.debug( f"✅ Pending events processed")

            # Main container
            self.logger.debug( f"🔧 Creating main frame...")
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.logger.debug( f"✅ Main frame created")
            
            # Create component frames
            self.logger.debug( f"🔧 Creating component frames...")
            self._create_component_frames(main_frame)
            self.logger.debug( f"✅ Component frames created")

            self.logger.debug( f"✅ Component-based GUI created successfully")

        except Exception as e:
            self.logger.error( f"❌ Error creating GUI: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_component_frames(self, parent):
        """Create and layout component frames"""
        try:
            # Top section: System Status
            self.logger.debug( f"🔧 Creating system status panel...")
            self.system_panel.create(parent)
            self.logger.debug( f"✅ System status panel created")

            # Middle section: Main controls (3 columns)
            self.logger.debug( f"🔧 Creating middle frame...")
            middle_frame = ttk.Frame(parent)
            middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            self.logger.debug( f"✅ Middle frame created")

            # Left column: Movement Controls
            self.logger.debug( f"🔧 Creating movement control panel...")
            self.movement_panel.create(middle_frame)
            self.logger.debug( f"✅ Movement control panel created")

            # Center column: Camera Feed
            self.logger.debug( f"🔧 Creating camera panel...")
            self.camera_panel.create(middle_frame)
            self.logger.debug( f"✅ Camera panel created")

            # Right column: Analysis
            self.logger.debug( f"🔧 Creating VLM analysis panel...")
            self.vlm_panel.create(middle_frame)
            self.logger.debug( f"✅ VLM analysis panel created")

            # Bottom section: Activity Log
            self.logger.debug( f"🔧 Creating activity log panel...")
            self.log_panel.create(parent)
            self.logger.debug( f"✅ Activity log panel created")

        except Exception as e:
            self.logger.error( f"❌ Error creating component frames: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _set_window_title_safely(self):
        """Set window title safely"""
        try:
            self.root.title(self.base_title)
            self.logger.debug( f"🔧 DEBUG: Window title set to: {self.base_title}")
        except Exception as e:
            self.logger.debug( f"🔧 DEBUG: Failed to set window title: {e}")

    def _start_background_processes(self):
        """Start background processes"""
        self.logger.debug( f"🔧 Starting background processes...")

        # Initialize ROS2 now that GUI is created
        try:
            self.logger.debug( f"🔧 Initializing ROS2...")
            if not rclpy.ok():
                rclpy.init()
            self.logger.debug( f"✅ ROS2 initialized")

            self.logger.debug( f"🔧 Creating ROS2 node...")
            self.ros_node = RobotGUIROS2Node(self._ros_callback)
            self.logger.info( f"✅ ROS2 node created")

            # Start ROS2 spinning in background thread
            self.logger.debug( f"🔧 Starting ROS2 spinning thread...")
            self.ros_spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
            self.ros_spin_thread.start()
            self.logger.debug( f"✅ ROS2 spinning thread started")

        except Exception as e:
            self.logger.error( f"❌ Error initializing ROS2: {e}")
            raise

        # Start system status updates
        self.logger.debug( f"🔧 Starting system status updates...")
        self.root.after(1000, self._update_system_status)  # Start after 1 second
        self.logger.debug( f"✅ System status updates scheduled")

    def _spin_ros(self):
        """Spin ROS2 node in background thread"""
        try:
            self.logger.debug( f"🔧 ROS2 spinning started in background thread")
            # Use spin_once with timeout to allow checking for cleanup requests
            while rclpy.ok() and not self._cleanup_requested:
                try:
                    rclpy.spin_once(self.ros_node, timeout_sec=0.05)  # Shorter timeout for faster exit
                except Exception as e:
                    if self._cleanup_requested:
                        break
                    self.logger.warning( f"⚠️ ROS2 spin_once error: {e}")
                    # If there's an error, sleep briefly to avoid tight loop
                    import time
                    time.sleep(0.1)
            self.logger.debug( f"🛑 ROS2 spinning stopped")
        except Exception as e:
            self.logger.error( f"❌ ROS2 spin error: {e}")
        finally:
            self.logger.debug( f"🔚 ROS2 spin thread exiting")

    # _process_updates method removed - using direct ROS2 callbacks

    def _handle_movement_command(self, command):
        """Handle movement command from movement panel
        
        CRITICAL: Always queue ROS2 commands to avoid thread safety issues
        """
        if isinstance(command, str):
            # Direct command string - check safety for movement commands
            if command in ['move_forward', 'move_backward', 'turn_left', 'turn_right']:
                if not self.movement_enabled:
                    self.log_message("🚫 Movement blocked by safety lock")
                    return
            self.log_message(f"🤖 Sending movement command: {command}")
            # Queue command for execution on GUI thread (prevents segfault)
            # Use default argument to capture command value immediately
            self.root.after(0, lambda cmd=command: self.ros_node.send_robot_command(cmd))
        elif isinstance(command, tuple) and len(command) == 2 and command[0] == 'safety_toggle':
            # Safety toggle command
            command_type, is_enabled = command
            self.movement_enabled = is_enabled
            status_text = "ENABLED" if is_enabled else "DISABLED"
            self.log_message(f"🔒 Safety lock {status_text}")

            # Update system status panel
            if hasattr(self, 'system_panel'):
                self.system_panel.update_robot_status({'movement_enabled': self.movement_enabled})
        elif isinstance(command, tuple) and len(command) == 2 and command[0] == 'movement_toggle':
            # Movement toggle command
            command_type, is_enabled = command
            self.movement_enabled = is_enabled
            status_text = "ENABLED" if is_enabled else "DISABLED"
            self.log_message(f"🚶 Movement controls {status_text}")

            # Update system status panel
            if hasattr(self, 'system_panel'):
                self.system_panel.update_robot_status({'movement_enabled': self.movement_enabled})
        elif callable(command):
            # Legacy callback function
            self.logger.error( f"Unknown movement command type: {type(command)}")
        else:
            self.logger.error( f"Unknown movement command format: {command}")

    def _handle_camera_update(self, message_type: str, data):
        """Handle camera update from camera panel"""
        # No camera source changes to handle - only robot camera available
        self.logger.warning( f"Unknown camera update type: {message_type}")





    def _ros_callback(self, message_type: str, data):
        """Handle ROS2 messages from node"""
        try:
            if message_type == 'sensor_data':
                self._handle_sensor_data(data)
            elif message_type == 'imu_data':
                self._handle_imu_data(data)
            elif message_type == 'lidar_data':
                self._handle_lidar_data(data)
            elif message_type == 'camera_image':
                self._handle_camera_image(data)
            elif message_type == 'navigation_commands':
                self._handle_navigation_commands(data)
            elif message_type == 'movement_status':
                self._handle_movement_status(data)
            elif message_type == 'model_status':
                self._handle_model_status(data)
            elif message_type == 'movement_complete':
                self._handle_movement_complete(data)
            else:
                self.logger.warning( f"Unknown ROS message type: {message_type}")
        except Exception as e:
            self.logger.error( f"Error handling ROS callback {message_type}: {e}")

    def _handle_sensor_data(self, data):
        """Handle sensor data updates"""
        self.sensor_data.update(data)
        # Update system status panel with sensor data
        if hasattr(self, 'system_panel'):
            self.system_panel.update_hardware_status(data)
        # Update movement panel with lidar data
        if hasattr(self, 'movement_panel'):
            self.movement_panel.update_lidar_data(data)

    def _handle_imu_data(self, data):
        """Handle IMU data updates"""
        # Store IMU data for movement panel display
        self.sensor_data.update(data)
        # Update movement panel with IMU data
        if hasattr(self, 'movement_panel'):
            self.movement_panel.update_imu_data(data)

    def _handle_lidar_data(self, data):
        """Handle LiDAR scan data updates"""
        # Update movement panel with LiDAR data
        if hasattr(self, 'movement_panel'):
            self.movement_panel.update_lidar_data(data)

    def _handle_camera_image(self, data):
        """Handle camera image updates"""
        # Store the current camera image for potential use
        self.current_image = data
        
        # Always update camera display with robot camera feed
        if hasattr(self, 'camera_panel'):
            # Convert ROS image to PIL for display
            try:
                # Use the same conversion logic as before
                import numpy as np
                np_image = np.frombuffer(data.data, dtype=np.uint8)

                if data.encoding == "rgb8":
                    rgb_array = np_image.reshape((data.height, data.width, 3))
                elif data.encoding == "bgr8":
                    bgr_array = np_image.reshape((data.height, data.width, 3))
                    rgb_array = bgr_array[:, :, ::-1]
                else:
                    rgb_array = np_image.reshape((data.height, data.width, 3))

                pil_image = Image.fromarray(rgb_array, 'RGB')
                self.camera_panel.update_camera_image(pil_image)
            except Exception as e:
                self.logger.error( f"Error converting camera image: {e}")
                # Camera error will be handled by the canvas message system

    def _handle_navigation_commands(self, data):
        """Handle navigation commands"""
        self.log_message(f"🧭 Navigation: {data}")

    def _handle_movement_status(self, data):
        """Handle movement status updates"""
        self.movement_enabled = data
        if hasattr(self, 'movement_panel'):
            self.movement_panel.update_button_states(self.movement_enabled)
        if hasattr(self, 'system_panel'):
            self.system_panel.update_robot_status({'movement_enabled': self.movement_enabled})

    def _handle_model_status(self, data):
        """Handle VLM Model status updates"""
        if hasattr(self, 'system_panel'):
            self.system_panel.update_model_status(data)
    
    def _handle_movement_complete(self, data):
        """Handle movement completion - trigger next analysis if auto-execute enabled"""
        self.log_message("✅ Movement complete")
        
        # If auto-execute is enabled, request next analysis
        if self.auto_execute_enabled and self.auto_vlm_enabled:
            self.log_message("🔄 Requesting next analysis...")
            # Small delay to let sensors settle
            self.root.after(500, lambda: self._auto_vlm_analysis())

    def _update_system_status(self):
        """Update system status display"""
        try:
            if hasattr(self, 'sensor_data') and self.sensor_data:
                # Update robot status based on sensor data
                robot_status = self._determine_robot_status(self.sensor_data)
                self.sensor_data['robot_status'] = robot_status

                # Update system status panel
                if hasattr(self, 'system_panel'):
                    status_data = {
                        'connection': 'Online' if robot_status != 'Offline' else 'Offline',
                        'movement_enabled': self.movement_enabled
                    }
                    self.system_panel.update_robot_status(status_data)
        except Exception as e:
            self.logger.error( f"Error updating system status: {e}")

        # Schedule next update
        self.root.after(GUIConfig.UPDATE_INTERVAL_MS, self._update_system_status)

    def _determine_robot_status(self, sensor_data):
        """Determine robot status based on sensor data"""
        if not sensor_data:
            return 'offline'

        # Check if we have recent sensor data (within 5 seconds)
        current_time = time.time()
        if 'timestamp_ns' in sensor_data:
            timestamp_seconds = sensor_data['timestamp_ns'] / 1e9
            if current_time - timestamp_seconds > 5.0:
                return 'offline'

        return 'Online'

    def log_message(self, message: str):
        """Log message to activity log"""
        if hasattr(self, 'log_panel'):
            self.log_panel.log_message(message)
        else:
            self.logger.info( message)
    
    def _immediate_exit(self):
        """Immediate exit without any cleanup to prevent hanging"""
        import os
        print("🔒 Window close - immediate exit")
        os._exit(0)
    
    def _on_window_close(self):
        """Handle window close event with timeout protection"""
        self.logger.info( f"🔒 Window close event triggered")
        
        # Set up a timeout to force exit if hanging
        import threading
        import os
        
        def force_exit_after_timeout():
            import time
            time.sleep(2.0)  # Wait 2 seconds max
            print("⏰ Timeout reached - force exit")
            os._exit(1)
        
        # Start timeout thread
        timeout_thread = threading.Thread(target=force_exit_after_timeout, daemon=True)
        timeout_thread.start()
        
        try:
            # Set cleanup flag
            self._cleanup_requested = True
            
            # Destroy window immediately
            self.root.destroy()
            
        except Exception as e:
            self.logger.error( f"❌ Error during window close: {e}")
        
        # If we get here, exit normally
        self.logger.info( f"🚪 Normal exit")
        os._exit(0)
    
    def _cleanup_simple(self):
        """Simple cleanup without waiting for threads"""
        try:
            self.logger.info( f"🧹 Simple cleanup starting...")
            
            # Stop auto analysis timer and cancel pending requests
            self._stop_auto_vlm_timer()
            self._cancel_pending_analysis_requests()
            
            # Don't wait for ROS thread - it will stop on its own
            # Just destroy the node quickly
            if hasattr(self, 'ros_node') and self.ros_node is not None:
                try:
                    self.ros_node.destroy_node()
                except:
                    pass  # Ignore errors during cleanup
                self.ros_node = None
            
            # Quick ROS shutdown without waiting
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except:
                    pass  # Ignore errors during cleanup
                    
            self.logger.info( f"✅ Simple cleanup completed")
            
        except Exception as e:
            self.logger.warning( f"⚠️ Error during simple cleanup: {e}")

    def cleanup(self):
        """Cleanup resources (for Ctrl-C compatibility)"""
        # Use the same simple cleanup as window close
        self._cleanup_simple()

    def _handle_vlm_analysis(self, event_type: str, data):
        """Handle analysis events"""
        if event_type == 'request_analysis':
            self._request_vlm_analysis(data)
        elif event_type == 'auto_analysis_toggle':
            self._toggle_auto_vlm_analysis(data)
        elif event_type == 'auto_execute_toggle':
            self._toggle_auto_execute(data)
        else:
            self.logger.warning( f"Unknown VLM analysis event: {event_type}")
    
    def _toggle_auto_execute(self, enabled: bool):
        """Toggle auto execute VLM recommendations on/off"""
        self.auto_execute_enabled = enabled
        if enabled:
            self.log_message("🤖 Auto execute VLM recommendations enabled")
        else:
            self.log_message("🤖 Auto execute VLM recommendations disabled")


    def _request_vlm_analysis(self, prompt: str, is_auto_request: bool = False):
        """Request analysis
        
        Args:
            prompt: The analysis prompt text
            is_auto_request: True if this is an automatic periodic request, False if manually triggered
        """
        try:
            # Check if an analysis is already in progress
            if self.analysis_in_progress:
                self.log_message("⏳ Analysis already in progress - queuing request")
                return

            # Only block auto-triggered requests when auto analysis is disabled
            # Manual requests (from Analyze button) should always go through
            if is_auto_request and not self.auto_vlm_enabled:
                self.log_message("🛑 Auto analysis disabled - skipping automatic request")
                return

            # Mark that analysis is now in progress
            self.analysis_in_progress = True
            self.log_message("🚀 Starting VLM analysis...")

            # Use the ROS2 node to send real analysis request to VLM service
            if hasattr(self, 'ros_node'):
                # Call the actual VLM service
                # Create VLM service client if it doesn't exist
                if not hasattr(self.ros_node, 'vlm_client'):
                    from robot_msgs.srv import ExecuteCommand
                    self.ros_node.vlm_client = self.ros_node.create_client(ExecuteCommand, '/vlm/analyze_scene')

                if not self.ros_node.vlm_client.wait_for_service(timeout_sec=1.0):
                    self.log_message("❌ VLM analysis service not available")
                    self.analysis_in_progress = False
                    return

                # Create service request for VLM analysis
                from robot_msgs.msg import RobotCommand
                from robot_msgs.srv import ExecuteCommand

                command_msg = RobotCommand()
                # Encode prompt in command_type field (format: "vlm_analysis:PROMPT")
                command_msg.command_type = f"vlm_analysis:{prompt}"

                request = ExecuteCommand.Request()
                request.command = command_msg

                # Call service asynchronously
                self.log_message(f"📤 Sending VLM request: {command_msg.command_type}")
                future = self.ros_node.vlm_client.call_async(request)

                # Track this pending request
                self.pending_analysis_requests.append(future)
                self.log_message(f"✅ VLM request sent, waiting for response...")
                
                # Handle response in callback
                def handle_vlm_response(future):
                    self.log_message(f"📥 VLM response callback triggered")
                    try:
                        # Remove this future from pending requests
                        if future in self.pending_analysis_requests:
                            self.pending_analysis_requests.remove(future)

                        # Check if the future was cancelled
                        if future.cancelled():
                            self.log_message("🛑 VLM analysis request was cancelled")
                            self.analysis_in_progress = False
                            return

                        response = future.result()
                        if response.success:
                            # Parse the JSON response from VLM service
                            try:
                                import json
                                result_data = json.loads(response.result_message)
                                
                                analysis_text = result_data.get('analysis', 'Analysis complete')
                                reasoning = result_data.get('reasoning', 'No reasoning provided')
                                full_analysis = result_data.get('full_analysis', analysis_text)
                                navigation_commands = result_data.get('navigation_commands', {'action': 'stop', 'confidence': 0.0})
                                action = navigation_commands.get('action', 'stop')
                                confidence = navigation_commands.get('confidence', 0.0)

                                vlm_result = {
                                    'success': True,
                                    'analysis_result': analysis_text,
                                    'reasoning': reasoning,
                                    'full_analysis': full_analysis,
                                    'navigation_commands': navigation_commands,
                                    'confidence': confidence,
                                    'timestamp_ns': self.ros_node.get_clock().now().nanoseconds
                                }
                                
                                # Update the VLM panel with real results
                                if hasattr(self, 'vlm_panel'):
                                    self.vlm_panel.update_analysis_result(vlm_result)
                                
                                # Check if this is an informational response (no movement)
                                is_informational = result_data.get('is_informational', False)
                                # Extract movement parameters (distance, angle, speed) if present
                                movement_params = result_data.get('parameters', {})

                                if is_informational or action == 'none':
                                    # This is an informational query - just display the response
                                    self.log_message(f"ℹ️ VLM Response: {reasoning}")
                                    # Don't execute any movement for informational queries
                                else:
                                    # EXECUTE THE RECOMMENDED ACTION (if auto-execute enabled)
                                    # Synchronous execution: analyze → move → wait → repeat
                                    if self.auto_execute_enabled:
                                        param_str = f" with params {movement_params}" if movement_params else ""
                                        self.log_message(f"🤖 Checking execution conditions - action: {action}{param_str}, confidence: {confidence:.2f}")
                                        if action != "stop" and confidence > 0.4:
                                            if self.movement_enabled:
                                                # Check if movement is already in progress
                                                if self.ros_node.movement_in_progress:
                                                    self.log_message(f"⏸️ Movement in progress, will re-analyze after completion")
                                                else:
                                                    self.log_message(f"🤖 Executing: {action}{param_str} (confidence: {confidence:.2f})")
                                                    # Execute command with parameters (will block new commands until complete)
                                                    # Use default arguments to capture values immediately
                                                    self.root.after(0, lambda act=action, params=movement_params: self.ros_node.send_robot_command(act, parameters=params))
                                                    # Next analysis will be triggered by movement_complete callback
                                            else:
                                                self.log_message(f"🔒 Movement disabled - VLM recommended: {action}{param_str}")
                                        else:
                                            self.log_message(f"🛑 VLM recommends: {action} (confidence: {confidence:.2f})")
                                            if action == "stop":
                                                # "stop" is a string literal, safe to use directly
                                                self.root.after(0, lambda: self.ros_node.send_robot_command("stop"))
                                    else:
                                        self.log_message(f"📋 VLM recommends: {action} (confidence: {confidence:.2f}) - Auto-execute disabled")
                                
                                self.log_message(f"✅ VLM analysis: {analysis_text}")
                                if reasoning and reasoning != 'No reasoning provided':
                                    self.log_message(f"🤖 Model reasoning: {reasoning}")

                                # Clear the analysis in progress flag on successful completion
                                self.analysis_in_progress = False

                            except json.JSONDecodeError as e:
                                self.log_message(f"❌ Error parsing VLM JSON response: {e}")
                                # Fallback to text parsing
                                analysis_text = response.result_message
                                action = "stop"
                                if "move_forward" in analysis_text:
                                    action = "move_forward"
                                elif "turn_left" in analysis_text:
                                    action = "turn_left"
                                elif "turn_right" in analysis_text:
                                    action = "turn_right"
                                
                                vlm_result = {
                                    'success': True,
                                    'analysis_result': analysis_text,
                                    'navigation_commands': {'action': action, 'confidence': 0.5},
                                    'confidence': 0.5,
                                    'timestamp_ns': self.ros_node.get_clock().now().nanoseconds
                                }
                                
                                if hasattr(self, 'vlm_panel'):
                                    self.vlm_panel.update_analysis_result(vlm_result)

                                # Clear the analysis in progress flag on completion (fallback case)
                                self.analysis_in_progress = False

                            except Exception as e:
                                self.log_message(f"❌ Error processing VLM response: {e}")
                                vlm_result = {
                                    'success': False,
                                    'analysis_result': f"Error: {str(e)}",
                                    'navigation_commands': {'action': 'stop', 'confidence': 0.0},
                                    'confidence': 0.0,
                                    'timestamp_ns': self.ros_node.get_clock().now().nanoseconds
                                }
                                # Clear the analysis in progress flag on error
                                self.analysis_in_progress = False
                        else:
                            self.log_message(f"❌ VLM analysis failed: {response.result_message}")
                            # Clear the analysis in progress flag on failure
                            self.analysis_in_progress = False
                    except Exception as e:
                        self.log_message(f"❌ VLM service response error: {e}")
                        # Clear the analysis in progress flag on error
                        self.analysis_in_progress = False

                # Add callback for when service call completes
                future.add_done_callback(handle_vlm_response)

        except Exception as e:
            self.log_message(f"❌ VLM analysis error: {e}")
            # Clear the analysis in progress flag on error
            self.analysis_in_progress = False

    def _toggle_auto_vlm_analysis(self, enabled: bool):
        """Toggle auto VLM analysis on/off"""
        self.auto_vlm_enabled = enabled

        if enabled:
            self.log_message("🔄 Auto VLM analysis enabled")
            self._start_auto_vlm_timer()
        else:
            self.log_message("🔄 Auto VLM analysis disabled - cancelling pending requests")
            self._stop_auto_vlm_timer()
            # Cancel all pending analysis requests
            self._cancel_pending_analysis_requests()

    def _cancel_pending_analysis_requests(self):
        """Cancel all pending VLM analysis requests"""
        cancelled_count = 0
        for future in self.pending_analysis_requests:
            try:
                if not future.done():
                    future.cancel()
                    cancelled_count += 1
            except Exception as e:
                self.logger.warning(f"⚠️ Error cancelling analysis request: {e}")

        if cancelled_count > 0:
            self.log_message(f"🛑 Cancelled {cancelled_count} pending analysis request(s)")

        # Clear the pending requests list and reset analysis flag
        self.pending_analysis_requests.clear()
        self.analysis_in_progress = False

    def _start_auto_vlm_timer(self):
        """Start the auto VLM analysis timer"""
        if self.auto_vlm_timer:
            self.root.after_cancel(self.auto_vlm_timer)
        
        # Start timer for automatic analysis using configurable interval
        self._schedule_auto_vlm_analysis()
    
    def _stop_auto_vlm_timer(self):
        """Stop the auto VLM analysis timer"""
        if self.auto_vlm_timer:
            self.root.after_cancel(self.auto_vlm_timer)
            self.auto_vlm_timer = None
    
    def _schedule_auto_vlm_analysis(self):
        """Schedule the next auto VLM analysis"""
        if self.auto_vlm_enabled:
            # Perform analysis
            self._auto_vlm_analysis()
            # Double-check auto analysis is still enabled before scheduling next
            if self.auto_vlm_enabled:
                # Schedule next analysis using configurable interval
                # Use VLM_AUTO_INTERVAL if available
                interval_seconds = getattr(GUIConfig, 'VLM_AUTO_INTERVAL', getattr(GUIConfig, 'VLM_AUTO_INTERVAL', 1.0))
                interval_ms = int(interval_seconds * 1000)  # Convert seconds to milliseconds
                self.auto_vlm_timer = self.root.after(interval_ms, self._schedule_auto_vlm_analysis)
            else:
                self.log_message("🛑 Auto analysis disabled during analysis - not rescheduling")
    
    def _auto_vlm_analysis(self):
        """Perform automatic VLM analysis"""
        if self.auto_vlm_enabled:
            # Use default prompt for auto analysis
            default_prompt = "Analyze the current camera view for navigation."
            status_msg = f"🤖 Auto analysis: {len(self.pending_analysis_requests)} pending"
            if self.analysis_in_progress:
                status_msg += " (analysis in progress)"
            else:
                status_msg += " (ready for new analysis)"
            self.log_message(status_msg)
            # Mark this as an automatic request
            self._request_vlm_analysis(default_prompt, is_auto_request=True)
        else:
            self.log_message("🛑 Auto analysis called but disabled - skipping")
    
    def _on_window_close(self):
        """Handle window close event (X button clicked)"""
        self.logger.info(f"🚪 Window close requested")
        try:
            # Stop auto-analysis timer
            if hasattr(self, 'auto_vlm_timer') and self.auto_vlm_timer:
                self.root.after_cancel(self.auto_vlm_timer)
                self.auto_vlm_timer = None

            # Stop ROS2 spinning thread gracefully
            if hasattr(self, 'ros_spin_thread') and self.ros_spin_thread and self.ros_spin_thread.is_alive():
                self.logger.info("🛑 Stopping ROS2 spinning thread...")
                # Set a flag to stop the spin thread
                self._running = False

            # Cleanup ROS2 resources
            self.logger.info(f"🧹 Cleaning up GUI resources...")
            if hasattr(self, 'ros_node') and self.ros_node:
                try:
                    # Publish a stop command before closing
                    try:
                        from geometry_msgs.msg import Twist
                        stop_msg = Twist()
                        self.ros_node.cmd_vel_publisher.publish(stop_msg)
                        self.logger.info("🛑 Published stop command before shutdown")
                    except Exception:
                        pass
                    
                    self.ros_node.destroy_node()
                    self.logger.info("✅ ROS2 node destroyed")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error destroying ROS2 node: {e}")

            # Shutdown ROS2 if we initialized it
            try:
                if rclpy.ok():
                    self.logger.info("🛑 Shutting down ROS2...")
                    rclpy.shutdown()
                    self.logger.info("✅ ROS2 shutdown complete")
            except Exception as e:
                self.logger.warning(f"⚠️ Error shutting down ROS2: {e}")

            self.logger.info(f"✅ GUI cleanup complete")
            
            # Destroy the window
            self.logger.info("🔒 Destroying window...")
            try:
                self.root.quit()
                self.root.destroy()
            except Exception as e:
                self.logger.warning(f"⚠️ Error destroying window: {e}")

            # Signal the launch process to terminate all nodes
            self.logger.info("🛑 Requesting launch system shutdown...")
            import os
            import signal
            try:
                # Send SIGINT (Ctrl-C) to the parent process group to terminate launch cleanly
                # This is different from SIGTERM - it triggers ROS2 launch's shutdown handlers
                parent_pid = os.getppid()
                self.logger.info(f"   └── Sending SIGINT to parent process {parent_pid}")
                os.kill(parent_pid, signal.SIGINT)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not signal parent process: {e}")

            self.logger.info("🚪 Window close cleanup finished")
            
        except Exception as e:
            self.logger.error(f"❌ Error during window close: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    import signal

    app = None

    def signal_handler(signum, frame):
        """Handle Ctrl-C and other signals consistently with window close"""
        print(f"🛑 Received signal {signum} (Ctrl-C)")
        if app:
            print("🧹 Initiating graceful shutdown...")
            app._on_window_close()  # Use the same cleanup path as window close
        else:
            print("🚪 No app to cleanup, exiting...")
            import sys
            sys.exit(0)

    # Set up signal handlers for consistent shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl-C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    try:
        # Import required modules
        import sys
        import rclpy

        # Check if ROS2 is already initialized (from launch file)
        if not rclpy.ok():
            rclpy.init()

        # Try to use ROS2 logging if available
        ros_logger = None

        try:
            if rclpy.ok():
                # ROS2 is already initialized, we can use ROS2 logging
                from rclpy.logging import get_logger
                ros_logger = get_logger('robot_gui_main')
                print("✅ Using ROS2 logging in main()")
            else:
                # ROS2 not initialized, use stderr
                raise Exception("ROS2 not initialized")
        except Exception as log_error:
            # Fallback to stderr logging
            print(f"⚠️ Falling back to stderr logging: {log_error}")
            def ros_logger_fallback(level, message):
                level_name = "INFO" if level == "info" else level.upper()
                sys.stderr.write(f"[robot_gui_main] [{level_name}] {message}\n")
                sys.stderr.flush()
            ros_logger = type('FallbackLogger', (), {
                'info': lambda msg: ros_logger_fallback('info', msg),
                'warn': lambda msg: ros_logger_fallback('warn', msg),
                'error': lambda msg: ros_logger_fallback('error', msg)
            })()

        # Create Tkinter root
        ros_logger.info("🔧 Creating Tkinter root...")
        root = tk.Tk()
        ros_logger.info("✅ Tkinter root created")

        # Create GUI application with the ROS logger
        ros_logger.info("🔧 Creating RobotGUIROS2 application...")
        app = RobotGUIROS2(root, ros_logger)
        ros_logger.info("✅ Robot GUI application ready")

        # Start Tkinter main loop
        ros_logger.info("🔧 Starting Tkinter main loop...")
        root.mainloop()

        # If we reach here, the window was closed
        ros_logger.info("🔒 Main loop exited")

    except KeyboardInterrupt:
        print("🛑 Received keyboard interrupt")
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup already done in _on_window_close
        print("👋 Robot GUI application exited")
        # Let Python exit naturally instead of forcing with os._exit
        import sys
        sys.exit(0)


if __name__ == '__main__':
    main()
