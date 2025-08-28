#!/usr/bin/env python3
"""
Robot Control GUI - Clean Component-Based Version
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
from robot_msgs.msg import RobotCommand, SensorData
from robot_msgs.srv import ExecuteCommand
from sensor_msgs.msg import Image as RosImage, Imu
from std_msgs.msg import String

# GUI component imports
from .gui_config import GUIConfig
from .gui_utils import GUIUtils
from .gui_components import (
    SystemStatusPanel,
    MovementControlPanel,
    CameraPanel,
    CosmosAnalysisPanel,
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
        
        # Camera images
        self.image_subscriber = self.create_subscription(
            RosImage,
            '/realsense/camera/color/image_raw',
            self._image_callback,
            self.image_qos
        )
        self.get_logger().info(f"📸 Created camera subscription with QoS: reliability={self.image_qos.reliability}, durability={self.image_qos.durability}")
        
        # Cosmos-Transfer1 results (if available)
        # Analysis handled directly by Cosmos model
        
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

        # Cosmos status updates
        self.cosmos_status_subscriber = self.create_subscription(
            String,
            '/cosmos/status',
            self._cosmos_status_callback,
            self.reliable_qos
        )
    
        # Cosmos-Transfer1 status (if available)
        # Status handled by Cosmos model directly
    
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
            'distance_front': msg.distance_front,
            'distance_left': msg.distance_left,
            'distance_right': msg.distance_right,
            'timestamp': msg.timestamp_ns
        }
        self.gui_callback('sensor_data', sensor_data)
    
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

    # VILA analysis removed - using Cosmos-Transfer1 directly
    
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

    def _cosmos_status_callback(self, msg: String):
        """Handle Cosmos status updates"""
        try:
            import json
            status_data = json.loads(msg.data)
            self.gui_callback('cosmos_status', status_data)
        except Exception as e:
            self.get_logger().error(f"Error parsing Cosmos status: {e}")
    # VILA status removed - using Cosmos-Transfer1 directly
    
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
            command_msg.angular_z = -(parameters.get('speed', 0.8) if parameters else 0.8)
            command_msg.duration = parameters.get('duration', 1.0) if parameters else 1.0
        elif command_type == 'turn_right':
            command_msg.command_type = 'turn'
            command_msg.angular_z = parameters.get('speed', 0.8) if parameters else 0.8
            command_msg.duration = parameters.get('duration', 1.0) if parameters else 1.0
        elif command_type == 'strafe_left':
            command_msg.command_type = 'move'
            command_msg.linear_y = parameters.get('speed', 0.3) if parameters else 0.3
            command_msg.duration = parameters.get('duration', 1.0) if parameters else 1.0
        elif command_type == 'strafe_right':
            command_msg.command_type = 'move'
            command_msg.linear_y = -(parameters.get('speed', 0.3) if parameters else 0.3)
            command_msg.duration = parameters.get('duration', 1.0) if parameters else 1.0
        elif command_type == 'stop':
            command_msg.command_type = 'stop'
            command_msg.linear_x = 0.0
            command_msg.angular_z = 0.0
        elif command_type == 'emergency_stop':
            command_msg.command_type = 'stop'
            command_msg.linear_x = 0.0
            command_msg.angular_z = 0.0
        elif command_type == 'safety_toggle':
            command_msg.command_type = 'safety_toggle'

        request = ExecuteCommand.Request()
        request.command = command_msg

        self.get_logger().info(f"📡 Calling robot command service for: {command_type}")
        future = self.robot_client.call_async(request)
        self.get_logger().info(f"📤 Command sent to robot: {command_type}")
        return future

    # VILA functionality removed - using Cosmos-Transfer1 directly


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
        # VILA functionality removed - using Cosmos-Transfer1 directly
        self.camera_source = GUIConfig.DEFAULT_CAMERA_SOURCE

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

            logger.debug("🔧 Creating CosmosAnalysisPanel...")
            self.cosmos_panel = CosmosAnalysisPanel(self.root, self._handle_cosmos_analysis, self.log_message)
            logger.debug("✅ CosmosAnalysisPanel created")

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

            # Right column: Cosmos-Transfer1 Analysis
            logger.debug("🔧 Creating Cosmos analysis panel...")
            self.cosmos_panel.create(middle_frame)
            logger.debug("✅ Cosmos analysis panel created")

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
                    rclpy.spin_once(self.ros_node, timeout_sec=0.1)
                except Exception as e:
                    if self._cleanup_requested:
                        break
                    logger.warning(f"⚠️ ROS2 spin_once error: {e}")
            logger.debug("🛑 ROS2 spinning stopped")
        except Exception as e:
            logger.error(f"❌ ROS2 spin error: {e}")

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
            self._on_camera_source_change()
        elif message_type == 'image_loaded':
            self._load_image_file_from_path(data)
        else:
            logger.warning(f"Unknown camera update type: {message_type}")

    def _load_image_file_from_path(self, file_path: str):
        """Load image from file path"""
        try:
            from PIL import Image
            self.loaded_image = Image.open(file_path)
            self.log_message(f"📁 Image loaded: {file_path}")
            # Update camera panel
            if hasattr(self, 'camera_panel'):
                self.camera_panel.update_camera_image(self.loaded_image)
        except Exception as e:
            self.log_message(f"❌ Failed to load image: {e}")

    # VILA functionality removed - using Cosmos-Transfer1 directly

    def _on_camera_source_change(self):
        """Handle camera source change"""
        # Update camera source logic here
        pass



    def _ros_callback(self, message_type: str, data):
        """Handle ROS2 messages from node"""
        try:
            if message_type == 'sensor_data':
                self._handle_sensor_data(data)
            elif message_type == 'imu_data':
                self._handle_imu_data(data)
            elif message_type == 'camera_image':
                self._handle_camera_image(data)
            elif message_type == 'navigation_commands':
                self._handle_navigation_commands(data)
            elif message_type == 'movement_status':
                self._handle_movement_status(data)
            elif message_type == 'cosmos_status':
                self._handle_cosmos_status(data)
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

    def _handle_imu_data(self, data):
        """Handle IMU data updates"""
        # Store IMU data for movement panel display
        self.sensor_data.update(data)
        # Update movement panel with IMU data
        if hasattr(self, 'movement_panel'):
            self.movement_panel.update_imu_data(data)

    def _handle_camera_image(self, data):
        """Handle camera image updates"""
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
                logger.error(f"Error converting camera image: {e}")

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

    def _handle_cosmos_status(self, data):
        """Handle Cosmos status updates"""
        if hasattr(self, 'system_panel'):
            self.system_panel.update_cosmos_status(data)

    def _handle_command_ack(self, data):
        """Handle command acknowledgments"""
        status = "✅ Success" if data['success'] else "❌ Failed"
        self.log_message(f"🤖 Command {data['command_type']}: {status}")

    # VILA functionality removed - using Cosmos-Transfer1 directly

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

    # VILA functionality removed - using Cosmos-Transfer1 directly

    # Legacy update methods removed - using direct ROS2 callbacks

    def log_message(self, message: str):
        """Log message to activity log"""
        if hasattr(self, 'log_panel'):
            self.log_panel.log_message(message)
        else:
            logger.info(message)
    
    def _on_window_close(self):
        """Handle window close event"""
        logger.info("🔒 Window close event triggered")
        
        # Immediately close the window first
        try:
            self.root.quit()  # Exit mainloop immediately
        except Exception as e:
            logger.error(f"❌ Error quitting mainloop: {e}")
        
        # Start cleanup in background thread to avoid blocking
        import threading
        cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        cleanup_thread.start()
        
        # Force window destruction
        try:
            self.root.destroy()  # Destroy window
        except Exception as e:
            logger.error(f"❌ Error destroying window: {e}")
    
    def _background_cleanup(self):
        """Perform cleanup in background thread"""
        try:
            logger.info("🧹 Starting background cleanup...")
            self.cleanup()
        except Exception as e:
            logger.error(f"❌ Error during background cleanup: {e}")
        finally:
            # Force exit after cleanup attempt
            import os
            logger.info("🚪 Force exiting application...")
            os._exit(0)

    def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("🧹 Cleaning up GUI resources...")

            # Set cleanup flag to stop ROS2 spinning thread
            if hasattr(self, '_cleanup_requested'):
                self._cleanup_requested = True
                logger.info("🛑 Cleanup requested")

            # Don't wait for ROS thread - just signal it to stop
            if hasattr(self, 'ros_spin_thread') and self.ros_spin_thread.is_alive():
                logger.info("🔄 ROS thread still running, signaling to stop...")
                # Thread will stop on its own when it checks _cleanup_requested

            # Close ROS2 node with timeout
            if hasattr(self, 'ros_node') and self.ros_node is not None:
                logger.info("🔌 Destroying ROS2 node...")
                try:
                    import signal
                    import threading
                    
                    def timeout_handler():
                        logger.warning("⚠️ ROS2 node destruction timeout, skipping...")
                        
                    timer = threading.Timer(0.5, timeout_handler)
                    timer.start()
                    try:
                        self.ros_node.destroy_node()
                        timer.cancel()
                    except Exception as e:
                        timer.cancel()
                        logger.warning(f"⚠️ Error destroying ROS2 node: {e}")
                except Exception as e:
                    logger.warning(f"⚠️ Error setting up node destruction: {e}")
                self.ros_node = None
                
            # Shutdown ROS2 with timeout
            if rclpy.ok():
                logger.info("🔌 Shutting down ROS2...")
                try:
                    import threading
                    
                    def shutdown_with_timeout():
                        try:
                            rclpy.shutdown()
                        except Exception as e:
                            logger.warning(f"⚠️ Error shutting down ROS2: {e}")
                    
                    shutdown_thread = threading.Thread(target=shutdown_with_timeout, daemon=True)
                    shutdown_thread.start()
                    shutdown_thread.join(timeout=0.5)
                    
                    if shutdown_thread.is_alive():
                        logger.warning("⚠️ ROS2 shutdown timeout, proceeding anyway...")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error during ROS2 shutdown: {e}")

            logger.info("✅ GUI cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def _handle_cosmos_analysis(self, event_type: str, data):
        """Handle Cosmos-Transfer1 analysis events"""
        if event_type == 'load_image':
            self._load_image_file_from_path(data)
        elif event_type == 'request_analysis':
            self._request_cosmos_analysis(data)
        else:
            logger.warning(f"Unknown Cosmos analysis event: {event_type}")

    def _load_image_file_from_path(self, file_path: str):
        """Load image from file path for Cosmos analysis"""
        try:
            from PIL import Image
            self.loaded_image = Image.open(file_path)
            self.log_message(f"📁 Image loaded for Cosmos analysis: {file_path}")

            # Update camera panel if it exists
            if hasattr(self, 'camera_panel'):
                self.camera_panel.update_camera_image(self.loaded_image)

        except Exception as e:
            self.log_message(f"❌ Failed to load image: {e}")

    def _request_cosmos_analysis(self, prompt: str):
        """Request Cosmos-Transfer1 analysis"""
        try:
            # Use the ROS2 node to send analysis request
            if hasattr(self, 'ros_node'):
                # For now, just log the request since Cosmos integration is pending
                self.log_message(f"🔍 Cosmos analysis requested: {prompt[:50]}...")
                self.log_message("📝 Cosmos-Transfer1 integration pending - would analyze image here")

                # TODO: Implement actual Cosmos-Transfer1 model integration
                # For now, show a placeholder response
                placeholder_result = {
                    'success': True,
                    'analysis_result': f'Placeholder analysis for: {prompt[:30]}...',
                    'navigation_commands': '{"action": "forward", "distance": 1.0}',
                    'confidence': 0.85,
                    'timestamp_ns': self.get_clock().now().nanoseconds
                }

                # Update the Cosmos panel with results
                if hasattr(self, 'cosmos_panel'):
                    self.cosmos_panel.update_analysis_result(placeholder_result)

        except Exception as e:
            self.log_message(f"❌ Cosmos analysis error: {e}")


def main():
    """Main entry point"""
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

    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'app' in locals():
                app.cleanup()
        except Exception as e:
            logger.error(f"❌ Final cleanup error: {e}")

        logger.info("👋 Robot GUI application exited")


if __name__ == '__main__':
    main()
