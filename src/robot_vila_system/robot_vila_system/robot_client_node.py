#!/usr/bin/env python3
"""
ROS2 Robot Client
Replaces HTTP-based robot communication with ROS2 topics and services
Runs on the robot (Jetson Orin Nano) and communicates with the client PC via ROS2
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image as PILImage

# ROS2 message imports
from robot_msgs.msg import RobotCommand, SensorData, VILAAnalysis, RobotStatus
from robot_msgs.srv import ExecuteCommand, RequestVILAAnalysis
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge

class RobotClientROS2(Node):
    """
    ROS2 Robot Client
    Runs on the robot and handles:
    - Publishing sensor data
    - Publishing camera images
    - Receiving and executing commands
    - Replacing all HTTP communication with ROS2 topics
    """
    
    def __init__(self):
        super().__init__('robot_client')
        
        # Robot configuration
        self.robot_id = "yahboomcar_x3_01"  # Hardcoded for single robot system
        self.robot_name = "YahBoom Car X3"
        
        # Initialize camera and sensors
        self.bridge = CvBridge()
        self.camera = None
        self._init_camera()
        
        # Sensor simulation (replace with real sensor interfaces)
        self.sensor_data = {
            "battery_voltage": 12.6,
            "battery_percentage": 85.0,
            "temperature": 23.5,
            "humidity": 45.0,
            "distance_front": 1.2,
            "distance_left": 0.8,
            "distance_right": 1.5,
            "wifi_signal": -45,
            "cpu_usage": 25.0,
            "memory_usage": 60.0,
            "camera_status": "Active"
        }
        
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
        
        # Start periodic publishing
        self._start_periodic_publishing()
        
        self.get_logger().info(f"🤖 ROS2 Robot Client initialized for {self.robot_id}")
        self.get_logger().info("   └── Publishing sensor data and camera images via ROS2")
        self.get_logger().info("   └── Receiving commands via ROS2 topics")
        self.get_logger().info("   └── HTTP communication replaced with ROS2 messaging")
    
    def _init_camera(self):
        """Initialize camera (replace with actual camera interface)"""
        try:
            # For simulation, we'll create a placeholder
            # In real implementation, initialize actual camera here
            self.camera = cv2.VideoCapture(0)  # Try to open default camera
            if not self.camera.isOpened():
                self.get_logger().warn("Camera not available, using simulation")
                self.camera = None
        except Exception as e:
            self.get_logger().error(f"Failed to initialize camera: {e}")
            self.camera = None
    
    def _setup_publishers(self):
        """Setup ROS2 publishers"""
        # Sensor data publishing
        self.sensor_publisher = self.create_publisher(
            SensorData,
            f'/robot/{self.robot_id}/sensors',
            self.best_effort_qos
        )
        
        # Camera image publishing
        self.image_publisher = self.create_publisher(
            Image,
            f'/robot/{self.robot_id}/camera/image_raw',
            self.best_effort_qos
        )
        
        # Compressed image publishing (for efficiency)
        self.compressed_image_publisher = self.create_publisher(
            CompressedImage,
            f'/robot/{self.robot_id}/camera/image_raw/compressed',
            self.best_effort_qos
        )
        
        # Command acknowledgment
        self.command_ack_publisher = self.create_publisher(
            RobotCommand,
            f'/robot/{self.robot_id}/command_ack',
            self.reliable_qos
        )
        
        # Robot status
        self.status_publisher = self.create_publisher(
            RobotStatus,
            f'/robot/{self.robot_id}/status',
            self.reliable_qos
        )
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers"""
        # Command reception from server
        self.command_subscriber = self.create_subscription(
            RobotCommand,
            f'/robot/{self.robot_id}/commands',
            self._command_callback,
            self.reliable_qos
        )
        
        # Emergency stop
        self.emergency_stop_subscriber = self.create_subscription(
            Bool,
            '/robot/emergency_stop',
            self._emergency_stop_callback,
            self.reliable_qos
        )
    
    def _start_periodic_publishing(self):
        """Start periodic publishing of sensor data and camera images"""
        # Sensor data timer (10 Hz)
        self.sensor_timer = self.create_timer(0.1, self._publish_sensor_data)
        
        # Camera image timer (5 Hz to avoid overwhelming the network)
        self.camera_timer = self.create_timer(0.2, self._publish_camera_image)
        
        # Status timer (1 Hz)
        self.status_timer = self.create_timer(1.0, self._publish_status)
    
    def _publish_sensor_data(self):
        """Publish current sensor data (replaces HTTP /sensors endpoint)"""
        try:
            # Update sensor data (simulate or read from real sensors)
            self._update_sensor_data()
            
            # Create sensor message
            sensor_msg = SensorData()
            sensor_msg.robot_id = self.robot_id
            sensor_msg.battery_voltage = self.sensor_data["battery_voltage"]
            sensor_msg.battery_percentage = self.sensor_data["battery_percentage"]
            sensor_msg.temperature = self.sensor_data["temperature"]
            sensor_msg.humidity = self.sensor_data["humidity"]
            sensor_msg.distance_front = self.sensor_data["distance_front"]
            sensor_msg.distance_left = self.sensor_data["distance_left"]
            sensor_msg.distance_right = self.sensor_data["distance_right"]
            sensor_msg.wifi_signal = self.sensor_data["wifi_signal"]
            sensor_msg.cpu_usage = self.sensor_data["cpu_usage"]
            sensor_msg.memory_usage = self.sensor_data["memory_usage"]
            sensor_msg.camera_status = self.sensor_data["camera_status"]
            sensor_msg.timestamp_ns = self.get_clock().now().nanoseconds
            
            # Simulate IMU data
            sensor_msg.imu_values.x = -0.007
            sensor_msg.imu_values.y = 0.132
            sensor_msg.imu_values.z = 9.984
            
            # Publish sensor data
            self.sensor_publisher.publish(sensor_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing sensor data: {e}")
    
    def _update_sensor_data(self):
        """Update sensor data (simulate or read from real sensors)"""
        # Simulate sensor variations
        import random
        
        # Battery slowly decreases
        self.sensor_data["battery_voltage"] = max(10.0, self.sensor_data["battery_voltage"] - 0.001)
        self.sensor_data["battery_percentage"] = (self.sensor_data["battery_voltage"] - 10.0) / 2.6 * 100
        
        # Temperature varies slightly
        self.sensor_data["temperature"] += random.uniform(-0.1, 0.1)
        self.sensor_data["temperature"] = max(20.0, min(30.0, self.sensor_data["temperature"]))
        
        # Distance sensors vary
        self.sensor_data["distance_front"] += random.uniform(-0.1, 0.1)
        self.sensor_data["distance_front"] = max(0.1, min(3.0, self.sensor_data["distance_front"]))
        
        # CPU and memory usage vary
        self.sensor_data["cpu_usage"] += random.uniform(-2.0, 2.0)
        self.sensor_data["cpu_usage"] = max(10.0, min(90.0, self.sensor_data["cpu_usage"]))
        
        self.sensor_data["memory_usage"] += random.uniform(-1.0, 1.0)
        self.sensor_data["memory_usage"] = max(30.0, min(95.0, self.sensor_data["memory_usage"]))
    
    def _publish_camera_image(self):
        """Publish camera image (replaces HTTP /image endpoint)"""
        try:
            if self.camera and self.camera.isOpened():
                # Read from real camera
                ret, frame = self.camera.read()
                if ret:
                    # Convert to ROS2 Image message
                    image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    image_msg.header.stamp = self.get_clock().now().to_msg()
                    image_msg.header.frame_id = f"{self.robot_id}_camera"
                    
                    # Publish image
                    self.image_publisher.publish(image_msg)
                    
                    # Also publish compressed version
                    compressed_msg = CompressedImage()
                    compressed_msg.header = image_msg.header
                    compressed_msg.format = "jpeg"
                    
                    # Compress image
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    compressed_msg.data = buffer.tobytes()
                    
                    self.compressed_image_publisher.publish(compressed_msg)
                    
            else:
                # Generate simulated image
                self._publish_simulated_image()
                
        except Exception as e:
            self.get_logger().error(f"Error publishing camera image: {e}")
    
    def _publish_simulated_image(self):
        """Publish simulated camera image when no real camera available"""
        try:
            # Create a simple test pattern
            height, width = 480, 640
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add some pattern
            cv2.rectangle(image, (50, 50), (width-50, height-50), (100, 150, 200), 2)
            cv2.putText(image, f"Robot {self.robot_id}", (100, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Time: {time.strftime('%H:%M:%S')}", (100, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert to ROS2 Image message
            image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = f"{self.robot_id}_camera"
            
            # Publish image
            self.image_publisher.publish(image_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing simulated image: {e}")
    
    def _publish_status(self):
        """Publish robot status"""
        try:
            status_msg = RobotStatus()
            status_msg.robot_id = self.robot_id
            status_msg.name = self.robot_name
            status_msg.last_seen_ns = self.get_clock().now().nanoseconds
            status_msg.battery_level = self.sensor_data["battery_percentage"]
            status_msg.status = "active"
            status_msg.capabilities = ["navigation", "vision", "sensors"]
            status_msg.connection_type = "ROS2"
            status_msg.last_command = ""
            status_msg.command_history = []
            
            # Add current sensor data
            sensor_msg = SensorData()
            sensor_msg.robot_id = self.robot_id
            sensor_msg.battery_voltage = self.sensor_data["battery_voltage"]
            sensor_msg.battery_percentage = self.sensor_data["battery_percentage"]
            sensor_msg.temperature = self.sensor_data["temperature"]
            sensor_msg.humidity = self.sensor_data["humidity"]
            sensor_msg.distance_front = self.sensor_data["distance_front"]
            sensor_msg.distance_left = self.sensor_data["distance_left"]
            sensor_msg.distance_right = self.sensor_data["distance_right"]
            sensor_msg.wifi_signal = self.sensor_data["wifi_signal"]
            sensor_msg.cpu_usage = self.sensor_data["cpu_usage"]
            sensor_msg.memory_usage = self.sensor_data["memory_usage"]
            sensor_msg.camera_status = self.sensor_data["camera_status"]
            sensor_msg.timestamp_ns = self.get_clock().now().nanoseconds
            
            status_msg.sensor_data = sensor_msg
            
            self.status_publisher.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing status: {e}")
    
    def _command_callback(self, msg: RobotCommand):
        """
        Handle incoming robot commands (replaces HTTP command polling)
        
        This is where the robot receives commands from the single gateway
        """
        try:
            self.get_logger().info(f"📨 Received command: {msg.command_type} from {msg.source}")
            
            # Parse parameters
            parameters = json.loads(msg.parameters_json) if msg.parameters_json else {}
            
            # Execute command based on type
            success = self._execute_robot_command(msg.command_type, parameters)
            
            # Send acknowledgment
            ack_msg = RobotCommand()
            ack_msg.robot_id = msg.robot_id
            ack_msg.command_type = msg.command_type
            ack_msg.parameters_json = msg.parameters_json
            ack_msg.timestamp_ns = self.get_clock().now().nanoseconds
            ack_msg.priority = msg.priority
            ack_msg.safety_confirmed = success
            ack_msg.gui_movement_enabled = msg.gui_movement_enabled
            ack_msg.source = f"ACK_{msg.source}"
            
            self.command_ack_publisher.publish(ack_msg)
            
            if success:
                self.get_logger().info(f"✅ Command executed: {msg.command_type}")
            else:
                self.get_logger().warn(f"❌ Command failed: {msg.command_type}")
                
        except Exception as e:
            self.get_logger().error(f"Error handling command: {e}")
    
    def _execute_robot_command(self, command_type: str, parameters: Dict[str, Any]) -> bool:
        """
        Execute robot command (replace with actual robot control interface)
        
        This would interface with the actual robot hardware/ROS2 control nodes
        """
        try:
            self.get_logger().info(f"🤖 Executing: {command_type} with {parameters}")
            
            if command_type == "move_forward":
                # Replace with actual forward movement
                speed = parameters.get('speed', 0.2)
                duration = parameters.get('duration', 1.0)
                self.get_logger().info(f"   └── Moving forward at {speed} m/s for {duration}s")
                # Here you would publish to /cmd_vel or interface with motor controllers
                return True
                
            elif command_type == "turn_left":
                # Replace with actual left turn
                speed = parameters.get('speed', 0.5)
                duration = parameters.get('duration', 1.0)
                self.get_logger().info(f"   └── Turning left at {speed} rad/s for {duration}s")
                return True
                
            elif command_type == "turn_right":
                # Replace with actual right turn
                speed = parameters.get('speed', 0.5)
                duration = parameters.get('duration', 1.0)
                self.get_logger().info(f"   └── Turning right at {speed} rad/s for {duration}s")
                return True
                
            elif command_type == "stop":
                # Replace with actual stop
                self.get_logger().info("   └── Stopping robot")
                return True
                
            elif command_type == "move":
                # Generic move command
                direction = parameters.get('direction', 'forward')
                speed = parameters.get('speed', 0.2)
                duration = parameters.get('duration', 1.0)
                self.get_logger().info(f"   └── Moving {direction} at {speed} for {duration}s")
                return True
                
            else:
                self.get_logger().warn(f"Unknown command type: {command_type}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error executing command {command_type}: {e}")
            return False
    
    def _emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop"""
        if msg.data:
            self.get_logger().warn("🚨 EMERGENCY STOP RECEIVED")
            # Immediately stop all robot movement
            # In real implementation, this would stop motors, etc.
            self.get_logger().warn("   └── All robot movement stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.camera:
            self.camera.release()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        robot_client = RobotClientROS2()
        rclpy.spin(robot_client)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'robot_client' in locals():
            robot_client.cleanup()
            robot_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
