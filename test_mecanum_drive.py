#!/usr/bin/env python3

"""
Test script for mecanum drive robot in Gazebo simulation.
Demonstrates forward, backward, strafing, and rotation movements.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time


class MecanumDriveTest(Node):
    def __init__(self):
        super().__init__('mecanum_drive_test')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Mecanum Drive Test Node Started')
        
    def send_velocity(self, linear_x=0.0, linear_y=0.0, angular_z=0.0, duration=3.0):
        """Send velocity command for specified duration"""
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        
        start_time = time.time()
        publish_rate = 20  # Hz
        sleep_time = 1.0 / publish_rate
        
        while (time.time() - start_time) < duration:
            self.publisher.publish(twist)
            rclpy.spin_once(self, timeout_sec=sleep_time)
            
    def stop(self, duration=1.0):
        """Stop the robot"""
        self.send_velocity(0.0, 0.0, 0.0, duration)
        
    def run_test(self):
        """Run the complete drive test sequence"""
        self.get_logger().info('Starting mecanum drive test sequence...')
        self.get_logger().info('Make sure Gazebo is running and robot is visible!')
        time.sleep(2)
        
        # Test 1: Forward
        self.get_logger().info('TEST 1: Moving FORWARD at 1.0 m/s for 3 seconds')
        self.send_velocity(linear_x=1.0, duration=3.0)
        self.stop()
        self.get_logger().info('Forward test complete')
        time.sleep(1)
        
        # Test 2: Backward
        self.get_logger().info('TEST 2: Moving BACKWARD at 1.0 m/s for 3 seconds')
        self.send_velocity(linear_x=-1.0, duration=3.0)
        self.stop()
        self.get_logger().info('Backward test complete')
        time.sleep(1)
        
        # Test 3: Strafe Left
        self.get_logger().info('TEST 3: STRAFE LEFT at 1.0 m/s for 3 seconds')
        self.send_velocity(linear_y=1.0, duration=3.0)
        self.stop()
        self.get_logger().info('Strafe left test complete')
        time.sleep(1)
        
        # Test 4: Strafe Right
        self.get_logger().info('TEST 4: STRAFE RIGHT at 1.0 m/s for 3 seconds')
        self.send_velocity(linear_y=-1.0, duration=3.0)
        self.stop()
        self.get_logger().info('Strafe right test complete')
        time.sleep(1)
        
        # Test 5: Rotate Left (Counter-clockwise)
        self.get_logger().info('TEST 5: ROTATE LEFT at 1.0 rad/s for 3 seconds')
        self.send_velocity(angular_z=1.0, duration=3.0)
        self.stop()
        self.get_logger().info('Rotate left test complete')
        time.sleep(1)
        
        # Test 6: Rotate Right (Clockwise)
        self.get_logger().info('TEST 6: ROTATE RIGHT at 1.0 rad/s for 3 seconds')
        self.send_velocity(angular_z=-1.0, duration=3.0)
        self.stop()
        self.get_logger().info('Rotate right test complete')
        time.sleep(1)
        
        # Test 7: Diagonal movement (Forward + Strafe Left)
        self.get_logger().info('TEST 7: DIAGONAL (Forward + Left) at 0.7 m/s for 3 seconds')
        self.send_velocity(linear_x=0.7, linear_y=0.7, duration=3.0)
        self.stop()
        self.get_logger().info('Diagonal test complete')
        time.sleep(1)
        
        # Test 8: Forward while rotating
        self.get_logger().info('TEST 8: FORWARD + ROTATE (spiral) for 3 seconds')
        self.send_velocity(linear_x=0.5, angular_z=0.5, duration=3.0)
        self.stop()
        self.get_logger().info('Spiral test complete')
        time.sleep(1)
        
        self.get_logger().info('===================================')
        self.get_logger().info('ALL TESTS COMPLETE!')
        self.get_logger().info('===================================')


def main(args=None):
    rclpy.init(args=args)
    
    test_node = MecanumDriveTest()
    
    try:
        test_node.run_test()
    except KeyboardInterrupt:
        test_node.get_logger().info('Test interrupted by user')
    finally:
        test_node.stop(duration=0.5)
        test_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

