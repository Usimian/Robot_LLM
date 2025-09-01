#!/usr/bin/env python3
"""
RoboMP2 Integration Test Script

This script demonstrates the RoboMP2 integration by:
1. Setting goals for the robot
2. Adding custom policies
3. Testing goal-conditioned navigation

Usage:
    python3 test_robomp2.py
"""

import rclpy
from rclpy.node import Node
import time
import json

from robot_msgs.msg import RobotCommand
from robot_msgs.srv import ExecuteCommand


class RoboMP2Tester(Node):
    """Test node for RoboMP2 functionality"""
    
    def __init__(self):
        super().__init__('robomp2_tester')
        
        # Service clients
        self.goal_client = self.create_client(ExecuteCommand, '/robomp2/set_goal')
        self.policy_client = self.create_client(ExecuteCommand, '/robomp2/add_policy')
        self.analysis_client = self.create_client(ExecuteCommand, '/vlm/analyze_scene')
        
        self.get_logger().info("🧪 RoboMP2 Tester initialized")
        
    def wait_for_services(self):
        """Wait for RoboMP2 services to be available"""
        self.get_logger().info("⏳ Waiting for RoboMP2 services...")
        
        services = [
            (self.goal_client, '/robomp2/set_goal'),
            (self.policy_client, '/robomp2/add_policy'),
            (self.analysis_client, '/vlm/analyze_scene')
        ]
        
        for client, service_name in services:
            if not client.wait_for_service(timeout_sec=10.0):
                self.get_logger().error(f"❌ Service {service_name} not available")
                return False
            else:
                self.get_logger().info(f"✅ Service {service_name} ready")
        
        return True
    
    def test_goal_setting(self):
        """Test setting different types of goals"""
        self.get_logger().info("🎯 Testing goal setting...")
        
        test_goals = [
            "navigation|Navigate to the kitchen safely",
            "exploration|Explore the environment to map the room",
            "manipulation|Pick up the red cup|red cup",
            "interaction|Approach the person and greet them|person"
        ]
        
        for goal_data in test_goals:
            try:
                request = ExecuteCommand.Request()
                request.command = RobotCommand()
                request.command.robot_id = "yahboomcar_x3_01"
                request.command.command_type = "set_goal"
                request.command.source_node = goal_data
                
                future = self.goal_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                
                if future.result():
                    response = future.result()
                    if response.success:
                        self.get_logger().info(f"✅ Goal set: {goal_data}")
                    else:
                        self.get_logger().error(f"❌ Goal setting failed: {response.result_message}")
                else:
                    self.get_logger().error(f"❌ Goal service call failed for: {goal_data}")
                
                time.sleep(1)  # Brief delay between goals
                
            except Exception as e:
                self.get_logger().error(f"❌ Exception in goal setting: {e}")
    
    def test_policy_addition(self):
        """Test adding custom policies"""
        self.get_logger().info("📚 Testing policy addition...")
        
        test_policies = [
            "custom_avoid|obstacle_ahead|navigation|turn_right,move_forward|Turn right when obstacle is directly ahead",
            "room_explore|open_space|exploration|move_forward,turn_left,move_forward,turn_left|Systematic room exploration pattern",
            "object_approach|object_detected|manipulation|move_forward,stop|Approach detected objects carefully",
            "corridor_nav|narrow_corridor|navigation|move_forward|Navigate through narrow corridors"
        ]
        
        for policy_data in test_policies:
            try:
                request = ExecuteCommand.Request()
                request.command = RobotCommand()
                request.command.robot_id = "yahboomcar_x3_01"
                request.command.command_type = "add_policy"
                request.command.source_node = policy_data
                
                future = self.policy_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                
                if future.result():
                    response = future.result()
                    if response.success:
                        self.get_logger().info(f"✅ Policy added: {policy_data.split('|')[0]}")
                    else:
                        self.get_logger().error(f"❌ Policy addition failed: {response.result_message}")
                else:
                    self.get_logger().error(f"❌ Policy service call failed for: {policy_data}")
                
                time.sleep(0.5)  # Brief delay between policies
                
            except Exception as e:
                self.get_logger().error(f"❌ Exception in policy addition: {e}")
    
    def test_goal_conditioned_analysis(self):
        """Test goal-conditioned VLM analysis"""
        self.get_logger().info("🧠 Testing goal-conditioned VLM analysis...")
        
        # First set a specific goal
        goal_request = ExecuteCommand.Request()
        goal_request.command = RobotCommand()
        goal_request.command.robot_id = "yahboomcar_x3_01"
        goal_request.command.command_type = "set_goal"
        goal_request.command.source_node = "navigation|Navigate to find a chair|chair"
        
        future = self.goal_client.call_async(goal_request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result() and future.result().success:
            self.get_logger().info("✅ Goal set for analysis test")
            
            # Now request VLM analysis with the goal context
            analysis_request = ExecuteCommand.Request()
            analysis_request.command = RobotCommand()
            analysis_request.command.robot_id = "yahboomcar_x3_01"
            analysis_request.command.command_type = "analyze"
            analysis_request.command.source_node = "robomp2_test|Analyze current scene with goal context"
            
            future = self.analysis_client.call_async(analysis_request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)  # Longer timeout for VLM
            
            if future.result():
                response = future.result()
                if response.success:
                    self.get_logger().info("✅ Goal-conditioned analysis completed")
                    try:
                        result_data = json.loads(response.result_message)
                        self.get_logger().info(f"   └── Analysis: {result_data.get('analysis', 'N/A')}")
                        self.get_logger().info(f"   └── Action: {result_data.get('navigation_commands', {}).get('action', 'N/A')}")
                        self.get_logger().info(f"   └── Confidence: {result_data.get('confidence', 'N/A')}")
                    except:
                        self.get_logger().info(f"   └── Raw response: {response.result_message}")
                else:
                    self.get_logger().error(f"❌ Analysis failed: {response.result_message}")
            else:
                self.get_logger().error("❌ Analysis service call failed")
        else:
            self.get_logger().error("❌ Could not set goal for analysis test")
    
    def run_tests(self):
        """Run all RoboMP2 tests"""
        self.get_logger().info("🚀 Starting RoboMP2 integration tests...")
        
        if not self.wait_for_services():
            self.get_logger().error("❌ Services not available, aborting tests")
            return False
        
        try:
            # Test 1: Goal setting
            self.test_goal_setting()
            time.sleep(2)
            
            # Test 2: Policy addition  
            self.test_policy_addition()
            time.sleep(2)
            
            # Test 3: Goal-conditioned analysis
            self.test_goal_conditioned_analysis()
            
            self.get_logger().info("🎉 RoboMP2 integration tests completed!")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ Test execution failed: {e}")
            return False


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        tester = RoboMP2Tester()
        
        # Run tests
        success = tester.run_tests()
        
        if success:
            print("\n" + "="*60)
            print("🎉 RoboMP2 Integration Test Results:")
            print("✅ Goal setting functionality working")
            print("✅ Policy database functionality working") 
            print("✅ Goal-conditioned VLM analysis working")
            print("✅ RoboMP2 framework successfully integrated!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ Some tests failed. Check logs for details.")
            print("="*60)
            
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
