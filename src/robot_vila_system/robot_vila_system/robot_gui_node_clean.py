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
from robot_msgs.msg import RobotCommand, SensorData, VILAAnalysis
from robot_msgs.srv import ExecuteCommand, RequestVILAAnalysis
from sensor_msgs.msg import Image as RosImage, Imu
from std_msgs.msg import String, Bool

# GUI component imports
from gui_config import GUIConfig
from gui_utils import GUIUtils
from gui_components import (
    SystemStatusPanel,
    MovementControlPanel,
    CameraPanel,
    VILAAnalysisPanel,
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

        # QoS profiles
        self.image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=GUIConfig.IMAGE_QOS_DEPTH
        )

        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=GUIConfig.RELIABLE_QOS_DEPTH
        )

        self.best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=GUIConfig.BEST_EFFORT_QOS_DEPTH
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

        # VILA server status
        self.vila_status_subscriber = self.create_subscription(
            String,
            '/vila/server_status',
            self._vila_status_callback,
            self.reliable_qos
        )

    def _setup_publishers(self):
        """Setup ROS2 publishers"""
        pass  # Add publishers if needed

    def _setup_service_clients(self):
        """Setup ROS2 service clients"""
        # VILA analysis service client
        self.vila_client = self.create_client(RequestVILAAnalysis, '/vila/request_analysis')

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

    def _vila_analysis_callback(self, msg: VILAAnalysis):
        """Handle VILA analysis results"""
        analysis_data = {
            'success': msg.success,
            'analysis_result': msg.analysis_result,
            'navigation_commands': msg.navigation_commands_json,
            'confidence': msg.confidence,
            'error_message': msg.error_message,
            'timestamp': msg.timestamp_ns
        }
        self.gui_callback('vila_analysis', analysis_data)

    def _navigation_commands_callback(self, msg: String):
        """Handle navigation commands"""
        self.gui_callback('navigation_commands', msg.data)

    def _safety_status_callback(self, msg: Bool):
        """Handle safety status updates"""
        self.gui_callback('safety_status', msg.data)

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

    def _vila_status_callback(self, msg: String):
        """Handle VILA server status updates"""
        try:
            status_data = json.loads(msg.data)
            self.gui_callback('vila_status', status_data)
        except Exception as e:
            self.get_logger().debug(f"Error parsing VILA status: {e}")

    def send_robot_command(self, command_type: str, parameters: Dict = None, safety_confirmed: bool = False):
        """Send robot command"""
        if not self.robot_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Robot command service not available")
            return False

        request = ExecuteCommand.Request()
        request.robot_id = self.robot_id
        request.command_type = command_type
        request.parameters_json = json.dumps(parameters or {})
        request.safety_confirmed = safety_confirmed

        future = self.robot_client.call_async(request)
        return future

    def request_vila_analysis(self, prompt: str, image: RosImage = None):
        """Request VILA analysis"""
        if not self.vila_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("VILA analysis service not available")
            return False

        request = RequestVILAAnalysis.Request()
        request.robot_id = self.robot_id
        request.prompt = prompt
        request.image = image or RosImage()

        future = self.vila_client.call_async(request)
        return future


class RobotGUIROS2:
    """Main GUI class using component-based architecture"""

    def __init__(self, root):
        logger.debug("🔧 DEBUG: RobotGUIROS2.__init__ entered")
        self.root = root

        # Use configuration constants
        self.base_title = GUIConfig.WINDOW_TITLE

        # Initialize ROS2
        logger.debug("🔧 DEBUG: Initializing ROS2...")
        rclpy.init()
        logger.debug("🔧 DEBUG: ROS2 initialized")

        # Create ROS2 node
        logger.debug("🔧 DEBUG: Creating ROS2 node...")
        self.ros_node = RobotGUIROS2Node(self._ros_callback)
        logger.debug("🔧 DEBUG: ROS2 node created successfully")

        # Initialize GUI state
        self._initialize_gui_state()

        # Initialize GUI components
        self._initialize_gui_components()

        # Create GUI
        logger.debug("🔧 DEBUG: About to call _create_gui()")
        self._create_gui()
        logger.debug("🔧 DEBUG: _create_gui() completed")

        # Start background processes
        self._start_background_processes()

        # Set window title
        self._set_window_title_safely()

        # Log initialization
        logger.debug("🤖 Robot GUI ROS2 initialized")
        logger.debug("   └── All communication via ROS2 topics and services")
        logger.debug("   └── Component-based architecture")

    def _initialize_gui_state(self):
        """Initialize GUI state variables"""
        logger.debug("🔧 DEBUG: Initializing GUI state variables")

        self.robot_id = "yahboomcar_x3_01"
        self.robot_data = {}
        self.sensor_data = {}
        self.current_image = None
        self.loaded_image = None
        self.vila_analysis = {}
        self.vila_server_status = {}
        self.safety_enabled = False
        self.movement_enabled = False

        # VILA automatic analysis state
        self.vila_auto_analysis = False
        self.vila_auto_interval = GUIConfig.VILA_AUTO_INTERVAL
        self.vila_last_auto_analysis = 0
        self.camera_source = GUIConfig.DEFAULT_CAMERA_SOURCE

        # Threading and queues
        self.update_queue = queue.Queue()

    def _initialize_gui_components(self):
        """Initialize GUI component classes"""
        logger.debug("🔧 DEBUG: Initializing GUI components")

        # Create component instances
        self.system_panel = SystemStatusPanel(self.root, self._update_system_status)
        self.movement_panel = MovementControlPanel(self.root, self._handle_movement_command)
        self.camera_panel = CameraPanel(self.root, self._handle_camera_update)
        self.vila_panel = VILAAnalysisPanel(self.root, self._handle_vila_request, self.log_message)
        self.log_panel = ActivityLogPanel(self.root)

    def _create_gui(self):
        """Create GUI using component classes"""
        logger.debug("🔧 DEBUG: Creating component-based GUI design")

        try:
            # Main container
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Create component frames
            self._create_component_frames(main_frame)

            logger.debug("🔧 DEBUG: Component-based GUI created successfully")

        except Exception as e:
            logger.error(f"🔧 DEBUG: Error creating GUI: {e}")
            import traceback
            traceback.print_exc()

    def _create_component_frames(self, parent):
        """Create and layout component frames"""
        # Top section: System Status
        logger.debug("🔧 DEBUG: Creating system status panel...")
        self.system_panel.create(parent)

        # Middle section: Main controls (3 columns)
        middle_frame = ttk.Frame(parent)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Left column: Movement Controls
        logger.debug("🔧 DEBUG: Creating movement control panel...")
        self.movement_panel.create(middle_frame)

        # Center column: Camera Feed
        logger.debug("🔧 DEBUG: Creating camera panel...")
        self.camera_panel.create(middle_frame)

        # Right column: VILA Analysis
        logger.debug("🔧 DEBUG: Creating VILA analysis panel...")
        self.vila_panel.create(middle_frame)

        # Bottom section: Activity Log
        logger.debug("🔧 DEBUG: Creating activity log panel...")
        self.log_panel.create(parent)

    def _set_window_title_safely(self):
        """Set window title safely"""
        try:
            self.root.title(self.base_title)
            logger.debug(f"🔧 DEBUG: Window title set to: {self.base_title}")
        except Exception as e:
            logger.debug(f"🔧 DEBUG: Failed to set window title: {e}")

    def _start_background_processes(self):
        """Start background processes"""
        logger.debug("🔧 DEBUG: Starting background processes")

        # Start GUI update processing
        self._process_updates()

        # Start ROS2 spinning in background
        self.root.after(GUIConfig.ROS_START_DELAY_MS, self._delayed_ros_start)

        # Start system status updates
        self._update_system_status()

        # Start automatic VILA analysis timer
        self.root.after(GUIConfig.AUTO_ANALYSIS_DELAY_MS, self._check_auto_analysis)

    def _delayed_ros_start(self):
        """Start ROS2 spinning after GUI is ready"""
        try:
            self.ros_spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
            self.ros_spin_thread.start()
            logger.debug("🔧 DEBUG: ROS2 spinning started")
        except Exception as e:
            logger.error(f"Error starting ROS2 spinning: {e}")

    def _spin_ros(self):
        """Spin ROS2 node"""
        try:
            rclpy.spin(self.ros_node)
        except Exception as e:
            logger.error(f"ROS2 spin error: {e}")

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

    def _handle_movement_command(self, command):
        """Handle movement command from movement panel"""
        if isinstance(command, str):
            # Direct command string
            self.ros_node.send_robot_command(command)
        elif callable(command):
            # Callback function (for safety toggle)
            result = command()
            if result == 'safety_toggle':
                self._toggle_safety()
        else:
            logger.error(f"Unknown movement command type: {type(command)}")

    def _handle_camera_update(self, message_type: str, data):
        """Handle camera update from camera panel"""
        if message_type == 'camera_source_changed':
            self._on_camera_source_change()
        elif message_type == 'image_loaded':
            self._load_image_file_from_path(data)
        else:
            logger.warning(f"Unknown camera update type: {message_type}")

    def _handle_vila_request(self, request_type: str, data):
        """Handle VILA request from VILA panel"""
        if request_type == 'request_analysis':
            self._request_vila_analysis_from_text(data)
        elif request_type == 'quick_analysis':
            self._quick_vila_analysis(data['type'], auto=False, prompt=data['prompt'])
        elif request_type == 'toggle_auto_analysis':
            self._toggle_vila_auto_analysis_from_panel(data)
        else:
            logger.warning(f"Unknown VILA request type: {request_type}")

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

    def _request_vila_analysis_from_text(self, prompt: str):
        """Request VILA analysis from text prompt"""
        if hasattr(self.ros_node, 'request_vila_analysis'):
            image = self.loaded_image if self.camera_source == "loaded" and self.loaded_image else None
            self.ros_node.request_vila_analysis(prompt, image)

    def _toggle_vila_auto_analysis_from_panel(self, enabled: bool):
        """Toggle auto analysis from panel"""
        self.vila_auto_analysis = enabled
        status = "ON" if enabled else "OFF"
        self.log_message(f"🔄 VILA Auto Analysis: {status}")

    def _on_camera_source_change(self):
        """Handle camera source change"""
        # Update camera source logic here
        pass

    def _toggle_safety(self):
        """Toggle safety status"""
        self.safety_enabled = not self.safety_enabled
        # Update movement panel
        if hasattr(self, 'movement_panel'):
            self.movement_panel.update_button_states(self.safety_enabled, self.movement_enabled)
        # Update system status panel
        if hasattr(self, 'system_panel'):
            self.system_panel.update_robot_status({'safety_enabled': self.safety_enabled})

    def _ros_callback(self, message_type: str, data):
        """Handle ROS2 messages from node"""
        try:
            if message_type == 'sensor_data':
                self._handle_sensor_data(data)
            elif message_type == 'imu_data':
                self._handle_imu_data(data)
            elif message_type == 'camera_image':
                self._handle_camera_image(data)
            elif message_type == 'vila_analysis':
                self._handle_vila_analysis(data)
            elif message_type == 'navigation_commands':
                self._handle_navigation_commands(data)
            elif message_type == 'safety_status':
                self._handle_safety_status(data)
            elif message_type == 'command_ack':
                self._handle_command_ack(data)
            elif message_type == 'vila_status':
                self._handle_vila_status(data)
            else:
                logger.warning(f"Unknown ROS message type: {message_type}")
        except Exception as e:
            logger.error(f"Error handling ROS callback {message_type}: {e}")

    def _handle_sensor_data(self, data):
        """Handle sensor data updates"""
        self.sensor_data.update(data)
        # Update system status panel with sensor data
        if hasattr(self, 'system_panel'):
            self.system_panel.update_sensor_data(data)

    def _handle_imu_data(self, data):
        """Handle IMU data updates"""
        # Handle IMU data if needed
        pass

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

    def _handle_vila_analysis(self, data):
        """Handle VILA analysis results"""
        if hasattr(self, 'vila_panel'):
            self.vila_panel.update_analysis_result(data)
        self.log_message(f"🧠 VILA Analysis: {data.get('analysis_result', 'N/A')}")

    def _handle_navigation_commands(self, data):
        """Handle navigation commands"""
        self.log_message(f"🧭 Navigation: {data}")

    def _handle_safety_status(self, data):
        """Handle safety status updates"""
        self.safety_enabled = data
        if hasattr(self, 'movement_panel'):
            self.movement_panel.update_button_states(self.safety_enabled, self.movement_enabled)
        if hasattr(self, 'system_panel'):
            self.system_panel.update_robot_status({'safety_enabled': self.safety_enabled})

    def _handle_command_ack(self, data):
        """Handle command acknowledgments"""
        status = "✅ Success" if data['success'] else "❌ Failed"
        self.log_message(f"🤖 Command {data['command_type']}: {status}")

    def _handle_vila_status(self, data):
        """Handle VILA status updates"""
        if hasattr(self, 'vila_panel'):
            self.vila_panel.update_status(data)
        if hasattr(self, 'system_panel'):
            self.system_panel.update_vila_status(data)

    def _update_system_status(self):
        """Update system status display"""
        try:
            # Update VILA model state
            if hasattr(self, 'sensor_data') and self.sensor_data:
                # Update robot status based on sensor data
                robot_status = self._determine_robot_status(self.sensor_data)
                self.sensor_data['robot_status'] = robot_status

                # Update system status panel
                if hasattr(self, 'system_panel'):
                    status_data = {
                        'connection': 'online' if robot_status != 'offline' else 'offline',
                        'safety_enabled': self.safety_enabled,
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
        if 'timestamp' in sensor_data:
            timestamp_seconds = sensor_data['timestamp'] / 1e9
            if current_time - timestamp_seconds > 5.0:
                return 'offline'

        return 'online'

    def _check_auto_analysis(self):
        """Check if automatic VILA analysis should run"""
        try:
            if self.vila_auto_analysis:
                current_time = time.time()
                if current_time - self.vila_last_auto_analysis >= self.vila_auto_interval:
                    self._run_auto_analysis()
                    self.vila_last_auto_analysis = current_time
        except Exception as e:
            logger.error(f"Error in auto analysis: {e}")

        # Schedule next check
        self.root.after(1000, self._check_auto_analysis)

    def _run_auto_analysis(self):
        """Run automatic VILA analysis"""
        if hasattr(self, 'vila_panel'):
            self.vila_panel._quick_analysis("navigation", auto=True)

    def _quick_vila_analysis(self, analysis_type: str, auto: bool = False, prompt: str = None):
        """Perform quick VILA analysis"""
        if prompt is None:
            prompts = {
                'navigation': "Analyze this image for safe navigation paths and obstacles.",
                'objects': "Identify and describe the objects visible in this image.",
                'scene': "Describe the overall scene and environment shown in this image."
            }
            prompt = prompts.get(analysis_type, "Analyze this image.")

        if hasattr(self.ros_node, 'request_vila_analysis'):
            image = self.loaded_image if self.camera_source == "loaded" and self.loaded_image else None
            self.ros_node.request_vila_analysis(prompt, image)

    def _handle_ros_update(self, message_type: str, data):
        """Handle ROS2 update (legacy method for compatibility)"""
        # This method maintains compatibility with any remaining old code
        # In the component-based architecture, updates are handled directly by components
        pass

    def log_message(self, message: str):
        """Log message to activity log"""
        if hasattr(self, 'log_panel'):
            self.log_panel.log_message(message)
        else:
            logger.info(message)

    def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("🧹 Cleaning up GUI resources...")

            # Stop ROS2 spinning
            if hasattr(self, 'ros_spin_thread') and self.ros_spin_thread.is_alive():
                # Note: In a real implementation, you'd want a cleaner way to stop the ROS thread
                pass

            # Close ROS2 node
            if hasattr(self, 'ros_node'):
                self.ros_node.destroy_node()

            # Shutdown ROS2
            if rclpy.ok():
                rclpy.shutdown()

            logger.info("✅ GUI cleanup completed")

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def main():
    """Main entry point"""
    try:
        # Create Tkinter root
        root = tk.Tk()

        # Create GUI application
        app = RobotGUIROS2(root)

        # Start Tkinter main loop
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
