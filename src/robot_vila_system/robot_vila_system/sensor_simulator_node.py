#!/usr/bin/env python3
"""
ROS2 Sensor Simulator Node
Provides realistic sensor data simulation for testing when robot is not available
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time
import random
import math
from typing import Dict, Any

# ROS2 message imports
from robot_msgs.msg import SensorData
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

class SensorSimulatorNode(Node):
    """
    ROS2 node that simulates realistic sensor data for testing
    """
    
    def __init__(self):
        super().__init__('sensor_simulator')
        
        # Declare parameters
        self.declare_parameter('robot_id', 'simulator_robot_001')
        self.declare_parameter('initial_battery_level', 85.0)
        self.declare_parameter('sensor_publish_rate', 5.0)
        self.declare_parameter('camera_enabled', True)
        
        # Get parameters
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.initial_battery = self.get_parameter('initial_battery_level').get_parameter_value().double_value
        self.sensor_rate = self.get_parameter('sensor_publish_rate').get_parameter_value().double_value
        self.camera_enabled = self.get_parameter('camera_enabled').get_parameter_value().bool_value
        
        self.robot_name = f"Sensor Simulator ({self.robot_id})"
        
        # QoS profiles for reliable communication
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Publishers
        self.sensor_publisher = self.create_publisher(
            SensorData,
            '/robot/sensors',
            self.reliable_qos
        )
        

        
        if self.camera_enabled:
            self.image_publisher = self.create_publisher(
                RosImage,
                '/robot/camera/image_raw',
                self.reliable_qos
            )
            # CV Bridge for image conversion
            self.bridge = CvBridge()
        
        # Simulation state
        self.simulation_time = 0.0
        self.battery_drain_rate = 0.1  # %/minute
        self.robot_status = "active"
        
        # Initialize sensor data
        self._initialize_sensor_data()
        
        # Start publishing timers
        self._start_timers()
        
        self.get_logger().info(f"🤖 Sensor Simulator Node started")
        self.get_logger().info(f"   Robot ID: {self.robot_id}")
        self.get_logger().info(f"   Initial Battery: {self.initial_battery}%")
        self.get_logger().info(f"   Sensor Rate: {self.sensor_rate} Hz")
        self.get_logger().info(f"   Camera Enabled: {self.camera_enabled}")
    
    def _initialize_sensor_data(self):
        """Initialize all sensor data"""
        self.sensor_data = {
            "battery_voltage": 12.4,
            "temperature": 45.0,  # Jetson CPU temperature
            "distance_front": 1.5,
            "distance_left": 0.8,
            "distance_right": 1.2,
            "cpu_usage": 25.0,
            "camera_status": "Active" if self.camera_enabled else "Disabled",
            "imu_values": {"x": 0.0, "y": 0.0, "z": 0.0}
        }
        
        # Keep battery percentage for internal simulation logic
        self.battery_percentage = self.initial_battery
    
    def _start_timers(self):
        """Start periodic publishing timers"""
        # Sensor data timer
        sensor_period = 1.0 / self.sensor_rate
        self.sensor_timer = self.create_timer(sensor_period, self._publish_sensor_data)
        

        
        # Camera image timer (10 Hz) if enabled
        if self.camera_enabled:
            self.image_timer = self.create_timer(0.1, self._publish_simulated_image)
        
        # Simulation update timer (10 Hz)
        self.update_timer = self.create_timer(0.1, self._update_simulation)
    
    def _update_simulation(self):
        """Update simulation state"""
        self.simulation_time += 0.1
        
        # Simulate battery drain
        time_minutes = self.simulation_time / 60.0
        self.battery_percentage = max(
            0.0, 
            self.initial_battery - (self.battery_drain_rate * time_minutes)
        )
        
        # Update battery voltage based on percentage
        voltage_range = (10.8, 12.6)  # Typical Li-ion range
        voltage_ratio = self.battery_percentage / 100.0
        self.sensor_data["battery_voltage"] = (
            voltage_range[0] + 
            (voltage_range[1] - voltage_range[0]) * voltage_ratio
        )
        
        # Simulate varying sensor readings
        base_time = self.simulation_time
        
        # Jetson CPU temperature (30-65°C range)
        self.sensor_data["temperature"] = 45.0 + 10.0 * math.sin(base_time * 0.1)
        
        # Distance sensors
        self.sensor_data["distance_front"] = 1.5 + 0.5 * math.sin(base_time * 0.3)
        self.sensor_data["distance_left"] = 0.8 + 0.3 * math.cos(base_time * 0.2)
        self.sensor_data["distance_right"] = 1.2 + 0.4 * math.sin(base_time * 0.25)
        
        # CPU usage
        self.sensor_data["cpu_usage"] = 30.0 + 10.0 * math.sin(base_time * 0.15)
        
        # IMU simulation
        self.sensor_data["imu_values"]["x"] = 0.1 * math.sin(base_time * 0.5)
        self.sensor_data["imu_values"]["y"] = 0.1 * math.cos(base_time * 0.4)
        self.sensor_data["imu_values"]["z"] = 9.8 + 0.2 * math.sin(base_time * 0.1)
        
        # Update robot status based on battery level
        if self.battery_percentage < 10.0:
            self.robot_status = "error"
        elif self.battery_percentage < 20.0:
            self.robot_status = "idle"
        else:
            self.robot_status = "active"
    
    def _publish_sensor_data(self):
        """Publish sensor data with all fields populated directly"""
        try:
            from geometry_msgs.msg import Vector3
            
            # Create sensor message with all fields
            sensor_msg = SensorData()
            sensor_msg.robot_id = self.robot_id
            sensor_msg.battery_voltage = self.sensor_data["battery_voltage"]
            sensor_msg.temperature = self.sensor_data["temperature"]
            sensor_msg.distance_front = self.sensor_data["distance_front"]
            sensor_msg.distance_left = self.sensor_data["distance_left"]
            sensor_msg.distance_right = self.sensor_data["distance_right"]
            sensor_msg.cpu_usage = self.sensor_data["cpu_usage"]
            sensor_msg.camera_status = self.sensor_data["camera_status"]
            
            # IMU values
            sensor_msg.imu_values = Vector3()
            sensor_msg.imu_values.x = self.sensor_data["imu_values"]["x"]
            sensor_msg.imu_values.y = self.sensor_data["imu_values"]["y"]
            sensor_msg.imu_values.z = self.sensor_data["imu_values"]["z"]
            
            sensor_msg.timestamp_ns = self.get_clock().now().nanoseconds
            
            # Publish sensor data
            self.sensor_publisher.publish(sensor_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing sensor data: {e}")
    

    
    def _publish_simulated_image(self):
        """Publish simulated camera images"""
        if not self.camera_enabled:
            return
            
        try:
            # Create a simple test pattern image
            height, width = 480, 640
            
            # Create a colorful test pattern
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add gradient background
            for y in range(height):
                for x in range(width):
                    image[y, x] = [
                        int(128 + 127 * math.sin(x * 0.01 + self.simulation_time)),
                        int(128 + 127 * math.sin(y * 0.01 + self.simulation_time * 0.5)),
                        int(128 + 127 * math.sin((x + y) * 0.005 + self.simulation_time * 0.3))
                    ]
            
            # Add moving circle (simulated object)
            center_x = int(320 + 200 * math.sin(self.simulation_time * 0.5))
            center_y = int(240 + 100 * math.cos(self.simulation_time * 0.3))
            cv2.circle(image, (center_x, center_y), 30, (255, 255, 255), -1)
            cv2.circle(image, (center_x, center_y), 30, (0, 0, 0), 2)
            
            # Add timestamp text
            timestamp_text = f"Sim Time: {self.simulation_time:.1f}s"
            cv2.putText(image, timestamp_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add sensor info overlay
            battery_text = f"Battery: {self.battery_percentage:.1f}%"
            cv2.putText(image, battery_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            status_text = f"Status: {self.robot_status}"
            cv2.putText(image, status_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Convert to ROS Image message
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "camera_link"
            
            ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
            ros_image.header = header
            
            # Publish image
            self.image_publisher.publish(ros_image)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing simulated image: {e}")

def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = SensorSimulatorNode()
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f"Error running sensor simulator: {e}")
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except:
                pass
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
