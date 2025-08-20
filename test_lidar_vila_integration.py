#!/usr/bin/env python3
"""
Test script to demonstrate LiDAR + VILA integration
Shows how LiDAR distances are provided directly to VILA for decision making
"""

import rclpy
from rclpy.node import Node
from robot_msgs.srv import RequestVILAAnalysis
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge

class LiDARVILATestNode(Node):
    """Test node to demonstrate LiDAR + VILA integration"""
    
    def __init__(self):
        super().__init__('lidar_vila_test')
        
        # Create service client for VILA analysis
        self.vila_client = self.create_client(RequestVILAAnalysis, '/vila/request_analysis')
        self.bridge = CvBridge()
        
        self.get_logger().info("🧪 LiDAR + VILA Integration Test Node started")
        
    def create_test_image(self, width=640, height=480):
        """Create a test image"""
        # Create a simple test image with some objects
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some "obstacles" as colored rectangles
        cv2.rectangle(img, (100, 200), (200, 400), (0, 0, 255), -1)  # Red obstacle (left)
        cv2.rectangle(img, (450, 200), (550, 400), (255, 0, 0), -1)  # Blue obstacle (right)
        
        # Add some "clear path" areas
        cv2.rectangle(img, (250, 300), (400, 350), (0, 255, 0), -1)  # Green path (center)
        
        # Convert to ROS Image message
        return self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
    
    def test_scenario(self, scenario_name, front_dist, left_dist, right_dist, expected_action=None):
        """Test a specific LiDAR scenario"""
        self.get_logger().info(f"\n🔬 Testing Scenario: {scenario_name}")
        self.get_logger().info(f"   LiDAR: Front={front_dist:.1f}m, Left={left_dist:.1f}m, Right={right_dist:.1f}m")
        
        if not self.vila_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("❌ VILA service not available")
            return False
        
        # Create request
        request = RequestVILAAnalysis.Request()
        request.robot_id = "test_robot"
        request.prompt = ""  # Use default navigation prompt
        request.image = self.create_test_image()
        request.distance_front = front_dist
        request.distance_left = left_dist
        request.distance_right = right_dist
        
        # Send request
        future = self.vila_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        
        if future.result():
            response = future.result()
            self.get_logger().info(f"✅ VILA Response:")
            self.get_logger().info(f"   Success: {response.success}")
            if response.success:
                self.get_logger().info(f"   Analysis: {response.analysis_result[:200]}...")
                self.get_logger().info(f"   Commands: {response.navigation_commands_json}")
                self.get_logger().info(f"   Confidence: {response.confidence:.2f}")
                
                if expected_action:
                    import json
                    nav_commands = json.loads(response.navigation_commands_json)
                    actual_action = nav_commands.get('action', 'unknown')
                    if actual_action == expected_action:
                        self.get_logger().info(f"✅ Expected action '{expected_action}' matches actual '{actual_action}'")
                    else:
                        self.get_logger().warn(f"⚠️ Expected '{expected_action}' but got '{actual_action}'")
            else:
                self.get_logger().error(f"❌ Analysis failed: {response.error_message}")
        else:
            self.get_logger().error("❌ Service call failed")
        
        self.get_logger().info("=" * 60)
        return True

def main():
    rclpy.init()
    test_node = LiDARVILATestNode()
    
    try:
        # Test various scenarios
        scenarios = [
            ("Clear Path Ahead", 2.5, 1.0, 1.2, "move_forward"),
            ("Obstacle Too Close", 0.15, 1.0, 1.2, "stop"),
            ("Left Side Blocked", 1.0, 0.2, 1.5, "turn_right"),
            ("Right Side Blocked", 1.0, 1.5, 0.2, "turn_left"),
            ("Front Blocked, Both Sides Open", 0.4, 1.2, 1.3, "turn_left"),
            ("All Directions Tight", 0.3, 0.3, 0.3, "stop"),
            ("Moderate Front, Good Sides", 0.7, 2.0, 1.8, None),  # Let VILA decide
        ]
        
        test_node.get_logger().info("🚀 Starting LiDAR + VILA Integration Tests")
        test_node.get_logger().info("=" * 60)
        
        for scenario_name, front, left, right, expected in scenarios:
            if not test_node.test_scenario(scenario_name, front, left, right, expected):
                break
            
            # Brief pause between tests
            import time
            time.sleep(1)
        
        test_node.get_logger().info("🎉 All tests completed!")
        
    except KeyboardInterrupt:
        test_node.get_logger().info("Test interrupted by user")
    finally:
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
