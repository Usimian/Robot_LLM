#!/usr/bin/env python3
"""
ROS2 Sensor Simulator Node
Provides realistic sensor data simulation for testing when robot is not available
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import time
import random
import math
from typing import Dict, Any

# ROS2 message imports
from robot_msgs.msg import SensorData
from sensor_msgs.msg import Image as RosImage, CameraInfo, PointCloud2
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
        
        # QoS profile for image streams (matching subscribers - RELIABLE for consistency)
        self.image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # QoS profile for camera info (to match RealSense driver - VOLATILE durability)
        self.camera_info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.sensor_publisher = self.create_publisher(
            SensorData,
            '/robot/sensors',
            self.reliable_qos
        )
        

        
        if self.camera_enabled:
            # Color image publisher (matching RealSense namespace and QoS)
            self.image_publisher = self.create_publisher(
                RosImage,
                '/realsense/camera/color/image_raw',
                self.image_qos
            )
            
            # Depth image publisher
            self.depth_publisher = self.create_publisher(
                RosImage,
                '/realsense/camera/depth/image_rect_raw',
                self.image_qos
            )
            
            # Aligned depth to color publisher (new topic from documentation)
            self.aligned_depth_publisher = self.create_publisher(
                RosImage,
                '/realsense/camera/aligned_depth_to_color/image_raw',
                self.image_qos
            )
            
            # Color camera info publisher
            self.color_info_publisher = self.create_publisher(
                CameraInfo,
                '/realsense/camera/color/camera_info',
                self.camera_info_qos
            )
            
            # Depth camera info publisher
            self.depth_info_publisher = self.create_publisher(
                CameraInfo,
                '/realsense/camera/depth/camera_info',
                self.camera_info_qos
            )
            
            # Point cloud publisher (keeping original topic as per documentation)
            self.pointcloud_publisher = self.create_publisher(
                PointCloud2,
                '/camera/depth/points',
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
            "cpu_temp": 45.0,  # Jetson CPU temperature
            "distance_front": 1.5,
            "distance_left": 0.8,
            "distance_right": 1.2,
            "cpu_usage": 25.0,
            "camera_status": "Active" if self.camera_enabled else "Disabled"
        }
        
        # Keep battery percentage for internal simulation logic
        self.battery_percentage = self.initial_battery
    
    def _start_timers(self):
        """Start periodic publishing timers"""
        # Sensor data timer
        sensor_period = 1.0 / self.sensor_rate
        self.sensor_timer = self.create_timer(sensor_period, self._publish_sensor_data)
        

        
        # Camera timers (30 Hz for responsive video) if enabled
        if self.camera_enabled:
            self.image_timer = self.create_timer(0.033, self._publish_simulated_image)  # ~30 Hz
            self.depth_timer = self.create_timer(0.033, self._publish_simulated_depth)  # ~30 Hz
            self.aligned_depth_timer = self.create_timer(0.033, self._publish_simulated_aligned_depth)  # ~30 Hz
            self.camera_info_timer = self.create_timer(1.0, self._publish_camera_info)  # 1 Hz for camera info
            self.pointcloud_timer = self.create_timer(0.2, self._publish_simulated_pointcloud)  # 5 Hz for point cloud
        
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
        
        # Update battery voltage based on percentage (extended range for color testing)
        voltage_range = (10.0, 12.6)  # Extended range to test all color thresholds
        voltage_ratio = self.battery_percentage / 100.0
        self.sensor_data["battery_voltage"] = (
            voltage_range[0] + 
            (voltage_range[1] - voltage_range[0]) * voltage_ratio
        )
        
        # Simulate varying sensor readings
        base_time = self.simulation_time
        
        # Jetson CPU temperature (30-65°C range)
        self.sensor_data["cpu_temp"] = 45.0 + 10.0 * math.sin(base_time * 0.1)
        
        # Distance sensors
        self.sensor_data["distance_front"] = 1.5 + 0.5 * math.sin(base_time * 0.3)
        self.sensor_data["distance_left"] = 0.8 + 0.3 * math.cos(base_time * 0.2)
        self.sensor_data["distance_right"] = 1.2 + 0.4 * math.sin(base_time * 0.25)
        
        # CPU usage
        self.sensor_data["cpu_usage"] = 30.0 + 10.0 * math.sin(base_time * 0.15)
        
        # IMU simulation removed - now using /imu/data_raw topic
        
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
            sensor_msg.cpu_temp = self.sensor_data["cpu_temp"]
            sensor_msg.distance_front = self.sensor_data["distance_front"]
            sensor_msg.distance_left = self.sensor_data["distance_left"]
            sensor_msg.distance_right = self.sensor_data["distance_right"]
            sensor_msg.cpu_usage = self.sensor_data["cpu_usage"]
            sensor_msg.camera_status = self.sensor_data["camera_status"]
            
            # IMU values removed - now using /imu/data_raw topic
            
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
            
            # Log publishing info every 30 frames (~1 second at 30 Hz)
            if int(self.simulation_time * 30) % 30 == 0:
                self.get_logger().debug(f"📸 Published camera image: {width}x{height}, encoding: bgr8")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing simulated image: {e}")
    
    def _publish_simulated_depth(self):
        """Publish simulated depth images"""
        if not self.camera_enabled:
            return
            
        try:
            # Create a simple depth image (16-bit)
            height, width = 480, 640
            
            # Create depth image with simulated distance values (in mm)
            depth_image = np.zeros((height, width), dtype=np.uint16)
            
            # Add gradient depth pattern (closer objects have smaller values)
            for y in range(height):
                for x in range(width):
                    # Simulate depth from 500mm to 3000mm
                    base_depth = 1500  # 1.5m base distance
                    variation = 500 * math.sin(x * 0.01 + self.simulation_time * 0.3)
                    depth_image[y, x] = max(500, min(3000, base_depth + variation))
            
            # Add simulated object (closer depth)
            center_x = int(320 + 200 * math.sin(self.simulation_time * 0.5))
            center_y = int(240 + 100 * math.cos(self.simulation_time * 0.3))
            cv2.circle(depth_image, (center_x, center_y), 30, 800, -1)  # 80cm object
            
            # Convert to ROS Image message
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "camera_depth_optical_frame"
            
            ros_depth = self.bridge.cv2_to_imgmsg(depth_image, "16UC1")
            ros_depth.header = header
            
            # Publish depth image
            self.depth_publisher.publish(ros_depth)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing simulated depth: {e}")
    
    def _publish_simulated_aligned_depth(self):
        """Publish simulated aligned depth to color images"""
        if not self.camera_enabled:
            return
            
        try:
            # Create aligned depth image (same as depth but aligned to color frame)
            height, width = 480, 640
            
            # Create aligned depth image with simulated distance values (in mm)
            aligned_depth_image = np.zeros((height, width), dtype=np.uint16)
            
            # Add gradient depth pattern (similar to regular depth but aligned)
            for y in range(height):
                for x in range(width):
                    # Simulate depth from 500mm to 3000mm
                    base_depth = 1400  # Slightly different base for aligned
                    variation = 400 * math.sin(x * 0.01 + self.simulation_time * 0.25)
                    aligned_depth_image[y, x] = max(500, min(3000, base_depth + variation))
            
            # Add simulated object (closer depth, aligned to color)
            center_x = int(320 + 200 * math.sin(self.simulation_time * 0.5))
            center_y = int(240 + 100 * math.cos(self.simulation_time * 0.3))
            cv2.circle(aligned_depth_image, (center_x, center_y), 35, 750, -1)  # 75cm object
            
            # Convert to ROS Image message
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "camera_color_optical_frame"  # Aligned to color frame
            
            ros_aligned_depth = self.bridge.cv2_to_imgmsg(aligned_depth_image, "16UC1")
            ros_aligned_depth.header = header
            
            # Publish aligned depth image
            self.aligned_depth_publisher.publish(ros_aligned_depth)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing simulated aligned depth: {e}")
    
    def _publish_camera_info(self):
        """Publish camera calibration info for both color and depth cameras"""
        if not self.camera_enabled:
            return
            
        try:
            # Create camera info message with typical RealSense D435i parameters
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            
            # Color camera info
            color_info = CameraInfo()
            color_info.header = header
            color_info.header.frame_id = "camera_color_optical_frame"
            color_info.width = 640
            color_info.height = 480
            color_info.distortion_model = "plumb_bob"
            # Typical D435i color camera intrinsics (approximate)
            color_info.k = [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0]
            color_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion for simulation
            color_info.p = [615.0, 0.0, 320.0, 0.0, 0.0, 615.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            
            # Depth camera info
            depth_info = CameraInfo()
            depth_info.header = header
            depth_info.header.frame_id = "camera_depth_optical_frame"
            depth_info.width = 640
            depth_info.height = 480
            depth_info.distortion_model = "plumb_bob"
            # Typical D435i depth camera intrinsics (approximate)
            depth_info.k = [385.0, 0.0, 320.0, 0.0, 385.0, 240.0, 0.0, 0.0, 1.0]
            depth_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion for simulation
            depth_info.p = [385.0, 0.0, 320.0, 0.0, 0.0, 385.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            
            # Publish camera info
            self.color_info_publisher.publish(color_info)
            self.depth_info_publisher.publish(depth_info)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing camera info: {e}")
    
    def _publish_simulated_pointcloud(self):
        """Publish simulated point cloud data"""
        if not self.camera_enabled:
            return
            
        try:
            # Create a simple point cloud message
            # For now, just create the message structure - actual point cloud generation
            # would require more complex 3D geometry
            
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "camera_depth_optical_frame"
            
            pointcloud = PointCloud2()
            pointcloud.header = header
            pointcloud.height = 1  # Unorganized point cloud
            pointcloud.width = 100  # Simplified with 100 points
            pointcloud.is_dense = False
            
            # TODO: Generate actual point cloud data
            # For simulation, we'll publish an empty point cloud structure
            self.pointcloud_publisher.publish(pointcloud)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing simulated point cloud: {e}")

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
