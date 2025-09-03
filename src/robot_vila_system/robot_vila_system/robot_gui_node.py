#!/usr/bin/env python3
"""
Robot Control GUI - Clean Component-Based Version
Tkinter application for monitoring and controlling robots via ROS2 topics and services
Replaces HTTP/WebSocket communication with ROS2 messaging
"""

import tkinter as tk
from tkinter import ttk
import json
import threading
import time
from typing import Dict
import logging
import queue
from PIL import Image
import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# ROS2 message imports
from robot_msgs.msg import RobotCommand, SensorData
from robot_msgs.srv import ExecuteCommand
from sensor_msgs.msg import Image as RosImage, Imu
from std_msgs.msg import String

# GUI component imports
from .gui_config import GUIConfig
from .gui_components import (
    SystemStatusPanel,
    MovementControlPanel,
    CameraPanel,
    VLMAnalysisPanel,
    ActivityLogPanel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RobotGUIROS2')


class RobotGUIROS2Node(Node):
    """ROS2 node for the robot GUI"""
    
    def __init__(self, gui_callback):
        super().__init__('robot_gui')
        
        self.gui_callback = gui_callback
        self.robot_id = "yahboomcar_x3_01"
        self.last_vila_update = None
        self.latest_imu_data = None
        self.current_camera_image = None
        self.current_sensor_data = None
        
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
        
        # Command acknowledgments
        self.command_ack_subscriber = self.create_subscription(
            RobotCommand,
            '/robot/command_ack',
            self._command_ack_callback,
            self.reliable_qos
        )

        # VLM Model status updates
        self.model_status_subscriber = self.create_subscription(
            String,
            '/vlm/status',
            self._model_status_callback,
            self.reliable_qos
        )
    
    def _setup_publishers(self):
        """Setup ROS2 publishers"""
        pass  # Add publishers if needed
    
    def _setup_service_clients(self):
        """Setup ROS2 service clients"""
        # Robot command service client
        self.robot_client = self.create_client(ExecuteCommand, '/robot/execute_command')
    
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
    
    # Safety status callback removed - handled locally in GUI
    
    def _command_ack_callback(self, msg: RobotCommand):
        """Handle command acknowledgments"""
        ack_data = {
            'command_type': msg.command_type,
            'parameters': msg.parameters,
            'success': msg.success,
            'message': msg.message,
            'timestamp': msg.timestamp_ns
        }
        self.gui_callback('command_ack', ack_data)

    def _model_status_callback(self, msg: String):
        """Handle VLM Model status updates"""
        try:
            import json
            status_data = json.loads(msg.data)
            self.gui_callback('model_status', status_data)
        except Exception as e:
            self.get_logger().error(f"Error parsing Model status: {e}")
    
    def send_robot_command(self, command_type: str, parameters: Dict = None, safety_confirmed: bool = False):
        """Send robot command"""
        self.get_logger().info(f"🚀 Attempting to send robot command: {command_type}")

        if not self.robot_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Robot command service not available")
            return False

        self.get_logger().info(f"✅ Robot command service available, sending: {command_type}")

        # Create RobotCommand message for the request
        command_msg = RobotCommand()
        command_msg.robot_id = self.robot_id
        command_msg.command_type = command_type
        command_msg.timestamp_ns = self.get_clock().now().nanoseconds
        command_msg.source_node = self.get_name()

        # Set command parameters based on type
        # Convert GUI command types to robot interface command types
        if command_type == 'move_forward':
            command_msg.command_type = 'move'
            command_msg.linear_x = parameters.get('speed', 0.5) if parameters else 0.5
            command_msg.duration = parameters.get('duration', 1.0) if parameters else 1.0
        elif command_type == 'move_backward':
            command_msg.command_type = 'move'
            command_msg.linear_x = -(parameters.get('speed', 0.5) if parameters else 0.5)
            command_msg.duration = parameters.get('duration', 1.0) if parameters else 1.0
        elif command_type == 'turn_left':
            command_msg.command_type = 'turn'
            command_msg.angular = parameters.get('angle', 90.0) if parameters else 90.0  # Positive = left
            command_msg.angular_speed = parameters.get('speed', 0.5) if parameters else 0.5
            command_msg.duration = parameters.get('duration', 8.0) if parameters else 8.0
        elif command_type == 'turn_right':
            command_msg.command_type = 'turn'
            command_msg.angular = -(parameters.get('angle', 90.0) if parameters else 90.0)  # Negative = right
            command_msg.angular_speed = parameters.get('speed', 0.5) if parameters else 0.5
            command_msg.duration = parameters.get('duration', 8.0) if parameters else 8.0
        elif command_type == 'strafe_left':
            command_msg.command_type = 'move'
            command_msg.linear_y = parameters.get('speed', 0.8) if parameters else 0.8
            command_msg.duration = parameters.get('duration', 1.0) if parameters else 1.0
        elif command_type == 'strafe_right':
            command_msg.command_type = 'move'
            command_msg.linear_y = -(parameters.get('speed', 0.8) if parameters else 0.8)
            command_msg.duration = parameters.get('duration', 1.0) if parameters else 1.0
        elif command_type == 'stop':
            command_msg.command_type = 'stop'
            command_msg.linear_x = 0.0
            command_msg.linear_y = 0.0
            command_msg.angular = 0.0
            command_msg.angular_speed = 0.0
        elif command_type == 'emergency_stop':
            command_msg.command_type = 'stop'
            command_msg.linear_x = 0.0
            command_msg.linear_y = 0.0
            command_msg.angular = 0.0
            command_msg.angular_speed = 0.0
        elif command_type == 'safety_toggle':
            command_msg.command_type = 'safety_toggle'

        request = ExecuteCommand.Request()
        request.command = command_msg

        self.get_logger().info(f"📡 Calling robot command service for: {command_type}")
        future = self.robot_client.call_async(request)
        self.get_logger().info(f"📤 Command sent to robot: {command_type}")
        return future


class RobotGUIROS2:
    """Main GUI class using component-based architecture"""
    
    def __init__(self, root):
        logger.debug("🔧 DEBUG: RobotGUIROS2.__init__ entered")
        self.root = root

        # Use configuration constants
        self.base_title = GUIConfig.WINDOW_TITLE
        
        # Defer ROS2 initialization until after GUI is created
        self.ros_node = None
        logger.debug("🔧 ROS2 initialization deferred until after GUI creation")

        # Initialize GUI state
        logger.debug("🔧 Initializing GUI state...")
        self._initialize_gui_state()
        logger.debug("✅ GUI state initialized")

        # Initialize GUI components
        logger.debug("🔧 Initializing GUI components...")
        self._initialize_gui_components()
        logger.debug("✅ GUI components initialized")

        # Create GUI
        logger.debug("🔧 Creating GUI...")
        self._create_gui()
        logger.debug("✅ GUI created successfully")

        # Start background processes
        logger.debug("🔧 Starting background processes...")
        self._start_background_processes()
        logger.debug("✅ Background processes started")

        # Set window title
        logger.debug("🔧 Setting window title...")
        self._set_window_title_safely()
        logger.debug("✅ Window title set")

        # Set up window close event handler
        logger.debug("🔧 Setting up window close handler...")
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        logger.debug("✅ Window close handler set")

        # Log initialization
        logger.info("🤖 Robot GUI ROS2 initialized (debug mode)")
        logger.info("   └── All communication via ROS2 topics and services")
        logger.info("   └── Component-based architecture")

    def _initialize_gui_state(self):
        """Initialize GUI state variables"""
        logger.debug("🔧 DEBUG: Initializing GUI state variables")

        self.robot_id = "yahboomcar_x3_01"
        self.robot_data = {}
        self.sensor_data = {}
        self.current_image = None
        self.loaded_image = None
        self.movement_enabled = True
        self._cleanup_requested = False

        self.camera_source = GUIConfig.DEFAULT_CAMERA_SOURCE
        
        # Auto VLM analysis
        self.auto_cosmos_enabled = False
        self.auto_cosmos_timer = None
        self.auto_execute_enabled = False  # Default: do not auto-execute for safety

        # Threading and queues
        self.update_queue = queue.Queue()

    def _initialize_gui_components(self):
        """Initialize GUI component classes"""
        logger.debug("🔧 Initializing GUI components...")

        try:
            # Create component instances with proper callbacks
            logger.debug("🔧 Creating SystemStatusPanel...")
            self.system_panel = SystemStatusPanel(self.root, self._update_system_status)
            logger.debug("✅ SystemStatusPanel created")

            logger.debug("🔧 Creating MovementControlPanel...")
            self.movement_panel = MovementControlPanel(self.root, self._handle_movement_command)
            logger.debug("✅ MovementControlPanel created")

            logger.debug("🔧 Creating CameraPanel...")
            self.camera_panel = CameraPanel(self.root, self._handle_camera_update)
            logger.debug("✅ CameraPanel created")

            logger.debug("🔧 Creating VLMAnalysisPanel...")
            self.vlm_panel = VLMAnalysisPanel(self.root, self._handle_vlm_analysis, self.log_message)
            logger.debug("✅ VLMAnalysisPanel created")

            logger.debug("🔧 Creating ActivityLogPanel...")
            self.log_panel = ActivityLogPanel(self.root)
            logger.debug("✅ ActivityLogPanel created")

        except Exception as e:
            logger.error(f"❌ Error initializing GUI components: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_gui(self):
        """Create GUI using component classes"""
        logger.debug("🔧 Creating component-based GUI design")
        
        try:
            # Configure root window first
            logger.debug("🔧 Configuring root window...")
            self.root.geometry(f"{GUIConfig.WINDOW_WIDTH}x{GUIConfig.WINDOW_HEIGHT}")
            self.root.title(self.base_title)
            logger.debug("✅ Root window configured")

            # Process any pending events
            logger.debug("🔧 Processing pending events...")
            self.root.update_idletasks()
            logger.debug("✅ Pending events processed")

            # Main container
            logger.debug("🔧 Creating main frame...")
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            logger.debug("✅ Main frame created")
            
            # Create component frames
            logger.debug("🔧 Creating component frames...")
            self._create_component_frames(main_frame)
            logger.debug("✅ Component frames created")

            logger.debug("✅ Component-based GUI created successfully")

        except Exception as e:
            logger.error(f"❌ Error creating GUI: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_component_frames(self, parent):
        """Create and layout component frames"""
        try:
            # Top section: System Status
            logger.debug("🔧 Creating system status panel...")
            self.system_panel.create(parent)
            logger.debug("✅ System status panel created")

            # Middle section: Main controls (3 columns)
            logger.debug("🔧 Creating middle frame...")
            middle_frame = ttk.Frame(parent)
            middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            logger.debug("✅ Middle frame created")

            # Left column: Movement Controls
            logger.debug("🔧 Creating movement control panel...")
            self.movement_panel.create(middle_frame)
            logger.debug("✅ Movement control panel created")

            # Center column: Camera Feed
            logger.debug("🔧 Creating camera panel...")
            self.camera_panel.create(middle_frame)
            logger.debug("✅ Camera panel created")

            # Right column: Analysis
            logger.debug("🔧 Creating VLM analysis panel...")
            self.vlm_panel.create(middle_frame)
            logger.debug("✅ VLM analysis panel created")

            # Bottom section: Activity Log
            logger.debug("🔧 Creating activity log panel...")
            self.log_panel.create(parent)
            logger.debug("✅ Activity log panel created")

        except Exception as e:
            logger.error(f"❌ Error creating component frames: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _set_window_title_safely(self):
        """Set window title safely"""
        try:
            self.root.title(self.base_title)
            logger.debug(f"🔧 DEBUG: Window title set to: {self.base_title}")
        except Exception as e:
            logger.debug(f"🔧 DEBUG: Failed to set window title: {e}")

    def _start_background_processes(self):
        """Start background processes"""
        logger.debug("🔧 Starting background processes...")

        # Initialize ROS2 now that GUI is created
        try:
            logger.debug("🔧 Initializing ROS2...")
            if not rclpy.ok():
                rclpy.init()
            logger.debug("✅ ROS2 initialized")

            logger.debug("🔧 Creating ROS2 node...")
            self.ros_node = RobotGUIROS2Node(self._ros_callback)
            logger.info("✅ ROS2 node created")

            # Start ROS2 spinning in background thread
            logger.debug("🔧 Starting ROS2 spinning thread...")
            self.ros_spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
            self.ros_spin_thread.start()
            logger.debug("✅ ROS2 spinning thread started")

        except Exception as e:
            logger.error(f"❌ Error initializing ROS2: {e}")
            raise

        # Start system status updates
        logger.debug("🔧 Starting system status updates...")
        self.root.after(1000, self._update_system_status)  # Start after 1 second
        logger.debug("✅ System status updates scheduled")

    def _spin_ros(self):
        """Spin ROS2 node in background thread"""
        try:
            logger.debug("🔧 ROS2 spinning started in background thread")
            # Use spin_once with timeout to allow checking for cleanup requests
            while rclpy.ok() and not self._cleanup_requested:
                try:
                    rclpy.spin_once(self.ros_node, timeout_sec=0.05)  # Shorter timeout for faster exit
                except Exception as e:
                    if self._cleanup_requested:
                        break
                    logger.warning(f"⚠️ ROS2 spin_once error: {e}")
                    # If there's an error, sleep briefly to avoid tight loop
                    import time
                    time.sleep(0.1)
            logger.debug("🛑 ROS2 spinning stopped")
        except Exception as e:
            logger.error(f"❌ ROS2 spin error: {e}")
        finally:
            logger.debug("🔚 ROS2 spin thread exiting")

    # _process_updates method removed - using direct ROS2 callbacks

    def _handle_movement_command(self, command):
        """Handle movement command from movement panel"""
        if isinstance(command, str):
            # Direct command string - check safety for movement commands
            if command in ['move_forward', 'move_backward', 'turn_left', 'turn_right']:
                if not self.movement_enabled:
                    self.log_message("🚫 Movement blocked by safety lock")
                    return
            self.log_message(f"🤖 Sending movement command: {command}")
            self.ros_node.send_robot_command(command)
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
            logger.error(f"Unknown movement command type: {type(command)}")
        else:
            logger.error(f"Unknown movement command format: {command}")

    def _handle_camera_update(self, message_type: str, data):
        """Handle camera update from camera panel"""
        if message_type == 'camera_source_changed':
            self.camera_source = data
            self._on_camera_source_change()
            self.log_message(f"📷 Camera source switched to: {data}")
        elif message_type == 'image_loaded':
            self._load_image_file_from_path(data)
            # Only switch to loaded source if user hasn't explicitly selected robot source
            current_source = self.camera_source
            if current_source != "robot":
                self.camera_source = "loaded"
                if hasattr(self, 'camera_panel'):
                    self.camera_panel.set_source("loaded")
                self.log_message(f"📷 Auto-switched to loaded image source")
            else:
                self.log_message(f"📷 Image loaded but keeping robot source as selected")
        else:
            logger.warning(f"Unknown camera update type: {message_type}")

    def _load_image_file_from_path(self, file_path: str):
        """Load image from file path"""
        try:
            from PIL import Image
            self.loaded_image = Image.open(file_path)
            filename = file_path.split('/')[-1] if '/' in file_path else file_path
            self.log_message(f"📁 Image loaded: {filename}")
            
            # Only update camera panel display if loaded source is currently selected
            if hasattr(self, 'camera_panel') and self.camera_source == "loaded":
                self.camera_panel.update_camera_image(self.loaded_image)
                
        except Exception as e:
            self.log_message(f"❌ Failed to load image: {e}")
            self.loaded_image = None

    def _on_camera_source_change(self):
        """Handle camera source change"""
        if self.camera_source == "loaded":
            if hasattr(self, 'loaded_image') and self.loaded_image:
                # Show loaded image immediately
                if hasattr(self, 'camera_panel'):
                    self.camera_panel.update_camera_image(self.loaded_image)
                self.log_message(f"📷 Displaying loaded image")
            else:
                # No loaded image available, show message
                if hasattr(self, 'camera_panel') and hasattr(self.camera_panel, 'camera_label'):
                    self.camera_panel.camera_label.config(text="📷 No image loaded\nClick 'Load Image' to select one", image="")
                self.log_message(f"📷 Loaded image source selected but no image available")
        elif self.camera_source == "robot":
            # Clear display and wait for next robot image
            if hasattr(self, 'camera_panel') and hasattr(self.camera_panel, 'camera_label'):
                self.camera_panel.camera_label.config(text="📷 Waiting for robot camera feed...", image="")
            self.log_message(f"📷 Robot camera source selected")
        else:
            self.log_message(f"📷 Unknown camera source: {self.camera_source}")



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
            elif message_type == 'command_ack':
                self._handle_command_ack(data)
            else:
                logger.warning(f"Unknown ROS message type: {message_type}")
        except Exception as e:
            logger.error(f"Error handling ROS callback {message_type}: {e}")

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
        
        # Only update camera display if source is set to robot
        if hasattr(self, 'camera_panel') and self.camera_source == "robot":
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
                logger.error(f"Error converting camera image: {e}")
                if hasattr(self, 'camera_panel') and hasattr(self.camera_panel, 'camera_label'):
                    self.camera_panel.camera_label.config(text=f"❌ Camera error: {str(e)}", image="")

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

    def _handle_command_ack(self, data):
        """Handle command acknowledgments"""
        status = "✅ Success" if data['success'] else "❌ Failed"
        self.log_message(f"🤖 Command {data['command_type']}: {status}")

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
            logger.error(f"Error updating system status: {e}")

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
            logger.info(message)
    
    def _immediate_exit(self):
        """Immediate exit without any cleanup to prevent hanging"""
        import os
        print("🔒 Window close - immediate exit")
        os._exit(0)
    
    def _on_window_close(self):
        """Handle window close event with timeout protection"""
        logger.info("🔒 Window close event triggered")
        
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
            logger.error(f"❌ Error during window close: {e}")
        
        # If we get here, exit normally
        logger.info("🚪 Normal exit")
        os._exit(0)
    
    def _cleanup_simple(self):
        """Simple cleanup without waiting for threads"""
        try:
            logger.info("🧹 Simple cleanup starting...")
            
            # Stop auto analysis timer
            self._stop_auto_cosmos_timer()
            
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
                    
            logger.info("✅ Simple cleanup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during simple cleanup: {e}")

    def cleanup(self):
        """Cleanup resources (for Ctrl-C compatibility)"""
        # Use the same simple cleanup as window close
        self._cleanup_simple()

    def _handle_vlm_analysis(self, event_type: str, data):
        """Handle analysis events"""
        if event_type == 'load_image':
            self._load_image_file_for_cosmos(data)
        elif event_type == 'request_analysis':
            self._request_cosmos_analysis(data)
        elif event_type == 'auto_analysis_toggle':
            self._toggle_auto_cosmos_analysis(data)
        elif event_type == 'auto_execute_toggle':
            self._toggle_auto_execute(data)
        else:
            logger.warning(f"Unknown VLM analysis event: {event_type}")
    
    def _toggle_auto_execute(self, enabled: bool):
        """Toggle auto execute VLM recommendations on/off"""
        self.auto_execute_enabled = enabled
        if enabled:
            self.log_message("🤖 Auto execute VLM recommendations enabled")
        else:
            self.log_message("🤖 Auto execute VLM recommendations disabled")

    def _load_image_file_for_cosmos(self, file_path: str):
        """Load image from file path for VLM analysis"""
        try:
            from PIL import Image
            self.loaded_image = Image.open(file_path)
            filename = file_path.split('/')[-1] if '/' in file_path else file_path
            self.log_message(f"📁 Image loaded for VLM analysis: {filename}")

            # Update camera panel display only if loaded source is selected
            if hasattr(self, 'camera_panel') and self.camera_source == "loaded":
                self.camera_panel.update_camera_image(self.loaded_image)

        except Exception as e:
            self.log_message(f"❌ Failed to load image for VLM analysis: {e}")
            self.loaded_image = None

    def _request_cosmos_analysis(self, prompt: str):
        """Request analysis"""
        try:
            # Use the ROS2 node to send real analysis request to VLM service
            if hasattr(self, 'ros_node'):
                self.log_message(f"🔍 VLM analysis requested: {prompt[:50]}...")
                
                # Call the actual VLM service
                # Create VLM service client if it doesn't exist
                if not hasattr(self.ros_node, 'vlm_client'):
                    from robot_msgs.srv import ExecuteCommand
                    self.ros_node.vlm_client = self.ros_node.create_client(ExecuteCommand, '/vlm/analyze_scene')
                
                if not self.ros_node.vlm_client.wait_for_service(timeout_sec=1.0):
                    self.log_message("❌ VLM analysis service not available")
                    return
                
                # Create service request for VLM analysis
                from robot_msgs.msg import RobotCommand
                from robot_msgs.srv import ExecuteCommand
                
                command_msg = RobotCommand()
                command_msg.robot_id = self.robot_id
                command_msg.command_type = "vlm_analysis"
                # Use source_node field to pass the prompt (temporary workaround)
                command_msg.source_node = f"{self.ros_node.get_name()}|{prompt}"
                command_msg.timestamp_ns = self.ros_node.get_clock().now().nanoseconds

                request = ExecuteCommand.Request()
                request.command = command_msg

                self.log_message(f"📡 Calling VLM analysis service...")
                
                # Call service asynchronously
                future = self.ros_node.vlm_client.call_async(request)
                
                # Handle response in callback
                def handle_cosmos_response(future):
                    try:
                        response = future.result()
                        if response.success:
                            # Parse the JSON response from VLM service
                            try:
                                import json
                                result_data = json.loads(response.result_message)
                                
                                analysis_text = result_data.get('analysis', 'Analysis complete')
                                navigation_commands = result_data.get('navigation_commands', {'action': 'stop', 'confidence': 0.0})
                                action = navigation_commands.get('action', 'stop')
                                confidence = navigation_commands.get('confidence', 0.0)
                                
                                cosmos_result = {
                                    'success': True,
                                    'analysis_result': analysis_text,
                                    'navigation_commands': navigation_commands,
                                    'confidence': confidence,
                                    'timestamp_ns': self.ros_node.get_clock().now().nanoseconds
                                }
                                
                                # Update the VLM panel with real results
                                if hasattr(self, 'vlm_panel'):
                                    self.vlm_panel.update_analysis_result(cosmos_result)
                                
                                # EXECUTE THE RECOMMENDED ACTION (if auto-execute enabled)
                                if self.auto_execute_enabled:
                                    if action != "stop" and confidence > 0.6:
                                        if self.movement_enabled:
                                            self.log_message(f"🤖 Executing VLM recommendation: {action} (confidence: {confidence:.2f})")
                                            self.ros_node.send_robot_command(action)
                                        else:
                                            self.log_message(f"🔒 Movement disabled - VLM recommended: {action}")
                                    else:
                                        self.log_message(f"🛑 VLM recommends: {action} (confidence: {confidence:.2f})")
                                        if action == "stop":
                                            self.ros_node.send_robot_command("stop")
                                else:
                                    self.log_message(f"📋 VLM recommends: {action} (confidence: {confidence:.2f}) - Auto-execute disabled")
                                
                                self.log_message(f"✅ VLM analysis: {analysis_text[:80]}...")
                                
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
                                
                                cosmos_result = {
                                    'success': True,
                                    'analysis_result': analysis_text,
                                    'navigation_commands': {'action': action, 'confidence': 0.5},
                                    'confidence': 0.5,
                                    'timestamp_ns': self.ros_node.get_clock().now().nanoseconds
                                }
                                
                                if hasattr(self, 'vlm_panel'):
                                    self.vlm_panel.update_analysis_result(cosmos_result)
                                
                            except Exception as e:
                                self.log_message(f"❌ Error processing VLM response: {e}")
                                cosmos_result = {
                                    'success': False,
                                    'analysis_result': f"Error: {str(e)}",
                                    'navigation_commands': {'action': 'stop', 'confidence': 0.0},
                                    'confidence': 0.0,
                                    'timestamp_ns': self.ros_node.get_clock().now().nanoseconds
                                }
                        else:
                            self.log_message(f"❌ VLM analysis failed: {response.result_message}")
                            
                    except Exception as e:
                        self.log_message(f"❌ VLM service response error: {e}")
                
                # Add callback for when service call completes
                future.add_done_callback(handle_cosmos_response)

        except Exception as e:
            self.log_message(f"❌ VLM analysis error: {e}")

    def _toggle_auto_cosmos_analysis(self, enabled: bool):
        """Toggle auto VLM analysis on/off"""
        self.auto_cosmos_enabled = enabled
        
        if enabled:
            self.log_message("🔄 Auto VLM analysis enabled")
            self._start_auto_cosmos_timer()
        else:
            self.log_message("🔄 Auto VLM analysis disabled")
            self._stop_auto_cosmos_timer()
    
    def _start_auto_cosmos_timer(self):
        """Start the auto VLM analysis timer"""
        if self.auto_cosmos_timer:
            self.root.after_cancel(self.auto_cosmos_timer)
        
        # Start timer for automatic analysis using configurable interval
        self._schedule_auto_cosmos_analysis()
    
    def _stop_auto_cosmos_timer(self):
        """Stop the auto VLM analysis timer"""
        if self.auto_cosmos_timer:
            self.root.after_cancel(self.auto_cosmos_timer)
            self.auto_cosmos_timer = None
    
    def _schedule_auto_cosmos_analysis(self):
        """Schedule the next auto VLM analysis"""
        if self.auto_cosmos_enabled:
            # Perform analysis
            self._auto_cosmos_analysis()
            # Schedule next analysis using configurable interval
            # Use VLM_AUTO_INTERVAL if available
            interval_seconds = getattr(GUIConfig, 'VLM_AUTO_INTERVAL', getattr(GUIConfig, 'VLM_AUTO_INTERVAL', 1.0))
            interval_ms = int(interval_seconds * 1000)  # Convert seconds to milliseconds
            self.auto_cosmos_timer = self.root.after(interval_ms, self._schedule_auto_cosmos_analysis)
    
    def _auto_cosmos_analysis(self):
        """Perform automatic VLM analysis"""
        if self.auto_cosmos_enabled:
            # Use default prompt for auto analysis
            default_prompt = "Analyze the current camera view for navigation."
            self._request_cosmos_analysis(default_prompt)
    
    def _on_window_close(self):
        """Handle window close event (X button clicked)"""
        logger.info("🚪 Window close requested")
        try:
            # Stop auto-analysis timer
            if hasattr(self, 'auto_cosmos_timer') and self.auto_cosmos_timer:
                self.root.after_cancel(self.auto_cosmos_timer)
                self.auto_cosmos_timer = None
            
            # Cleanup ROS2 resources
            logger.info("🧹 Cleaning up GUI resources...")
            if hasattr(self, 'ros_node') and self.ros_node:
                self.ros_node.destroy_node()
            
            # Shutdown ROS2
            if rclpy.ok():
                rclpy.shutdown()
            
            logger.info("✅ GUI cleanup complete")
            
            # Send SIGTERM to parent process group to terminate launch
            import os
            import signal
            try:
                # Get the process group ID and send SIGTERM to terminate launch
                pgid = os.getpgid(0)
                logger.info(f"🛑 Sending SIGTERM to process group {pgid}")
                os.killpg(pgid, signal.SIGTERM)
            except Exception as e:
                logger.warning(f"⚠️ Could not terminate process group: {e}")
            
            # Destroy the window and exit
            self.root.quit()
            self.root.destroy()
            
            # Final exit
            logger.info("🚪 Exiting application...")
            os._exit(0)
            
        except Exception as e:
            logger.error(f"❌ Error during window close: {e}")
            # Force exit if cleanup fails
            import os
            os._exit(0)


def main():
    """Main entry point"""
    import signal
    
    app = None
    
    def signal_handler(signum, frame):
        """Handle Ctrl-C and other signals consistently with window close"""
        logger.info(f"🛑 Received signal {signum} (Ctrl-C)")
        if app:
            logger.info("🧹 Initiating graceful shutdown...")
            app._on_window_close()  # Use the same cleanup path as window close
        else:
            logger.info("🚪 No app to cleanup, exiting...")
            import os
            os._exit(0)
    
    # Set up signal handlers for consistent shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl-C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    try:
        logger.info("🚀 Starting Robot GUI application...")

        # Create Tkinter root
        logger.debug("🔧 Creating Tkinter root...")
        root = tk.Tk()
        logger.debug("✅ Tkinter root created")

        # Create GUI application
        logger.debug("🔧 Creating RobotGUIROS2 application...")
        app = RobotGUIROS2(root)
        logger.info("✅ Robot GUI application ready")

        # Start Tkinter main loop
        logger.debug("🔧 Starting Tkinter main loop...")
        root.mainloop()
        
        # If we reach here, the window was closed
        logger.info("🔒 Main loop exited")

    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Don't do cleanup here - it was already done in _on_window_close
        logger.info("👋 Robot GUI application exited")
        # Use os._exit to avoid any remaining cleanup issues
        import os
        os._exit(0)


if __name__ == '__main__':
    main()
