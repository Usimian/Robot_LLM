#!/usr/bin/env python3
"""
VILA ROS Node
Integrates VILA vision-language model with ROS for robotic applications
"""

import rospy
import sys
import os
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import numpy as np
import threading
import json

# Add VILA paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VILA'))
from main_vila import VILAModel

class VILARosNode:
    def __init__(self):
        rospy.init_node('vila_vision_node', anonymous=True)
        
        # Initialize VILA model
        self.vila_model = VILAModel()
        self.model_loaded = False
        self.bridge = CvBridge()
        
        # ROS Publishers
        self.analysis_pub = rospy.Publisher('/vila/analysis', String, queue_size=10)
        self.navigation_pub = rospy.Publisher('/vila/navigation_command', String, queue_size=10)
        # ⚠️ DISABLED: Direct cmd_vel publishing bypasses safety system
        # self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.status_pub = rospy.Publisher('/vila/status', Bool, queue_size=10)
        
        # 🛡️ SAFETY: Disable direct robot control to enforce single command gateway
        self.direct_robot_control_enabled = False
        rospy.logwarn("🚫 SAFETY: Direct robot control DISABLED - all commands must go through unified gateway")
        rospy.logwarn("   └── This ROS node will NOT publish to /cmd_vel")
        rospy.logwarn("   └── Use the robot_vila_server.py gateway for all robot commands")
        
        # ROS Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.query_sub = rospy.Subscriber('/vila/query', String, self.query_callback)
        
        # Load model
        self.load_model_async()
        
        rospy.loginfo("🤖 VILA ROS Node initialized")
    
    def load_model_async(self):
        """Load VILA model in background thread"""
        def load_model():
            rospy.loginfo("🚀 Loading VILA model...")
            success = self.vila_model.load_model()
            if success:
                self.model_loaded = True
                rospy.loginfo("✅ VILA model loaded successfully")
                self.status_pub.publish(Bool(data=True))
            else:
                rospy.logerr("❌ Failed to load VILA model")
                self.status_pub.publish(Bool(data=False))
        
        threading.Thread(target=load_model, daemon=True).start()
    
    def image_callback(self, msg):
        """Process incoming camera images"""
        if not self.model_loaded:
            return
            
        try:
            # Convert ROS image to PIL
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # Analyze for navigation (customize prompt based on your robot's needs)
            navigation_prompt = """You are a robot's vision system. Analyze this camera view and provide:
1. Can I move forward safely?
2. Are there obstacles ahead?
3. What should I do next (move_forward, turn_left, turn_right, stop)?
4. Describe what you see briefly.
Keep it concise for real-time navigation."""
            
            # Generate response
            response = self.vila_model.generate_response(
                prompt=navigation_prompt,
                image=pil_image
            )
            
            # Publish analysis
            self.analysis_pub.publish(String(data=response))
            
            # Extract navigation commands
            nav_commands = self.parse_navigation_commands(response)
            self.navigation_pub.publish(String(data=json.dumps(nav_commands)))
            
            # 🛡️ SAFETY: All robot commands must go through the single gateway
            if self.direct_robot_control_enabled:
                rospy.logwarn("🚫 SAFETY VIOLATION: Direct robot control attempted but blocked")
                rospy.logwarn(f"   └── Commands blocked: {nav_commands}")
                rospy.logwarn("   └── Use robot_vila_server.py gateway for ALL robot control")
            else:
                rospy.loginfo("✅ SAFETY: Commands routed to gateway instead of direct /cmd_vel")
                rospy.loginfo(f"   └── Commands: {nav_commands}")
                rospy.loginfo("   └── Send these to robot_vila_server.py for proper safety validation")
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def query_callback(self, msg):
        """Handle custom queries from other ROS nodes"""
        if not self.model_loaded:
            return
            
        # This would need the latest camera image - implement image buffering
        # For now, just acknowledge the query
        rospy.loginfo(f"Received query: {msg.data}")
    
    def parse_navigation_commands(self, response):
        """Parse VILA response for navigation commands"""
        response_lower = response.lower()
        
        commands = {
            'action': 'stop',  # default safe action
            'confidence': 0.0,
            'reason': response
        }
        
        if 'move forward' in response_lower or ('clear' in response_lower and 'safe' in response_lower):
            commands['action'] = 'move_forward'
            commands['confidence'] = 0.8
        elif 'turn left' in response_lower:
            commands['action'] = 'turn_left'
            commands['confidence'] = 0.7
        elif 'turn right' in response_lower:
            commands['action'] = 'turn_right'
            commands['confidence'] = 0.7
        elif 'stop' in response_lower or 'obstacle' in response_lower:
            commands['action'] = 'stop'
            commands['confidence'] = 0.9
            
        return commands
    
    def send_velocity_commands(self, commands):
        """
        🚫 DISABLED: Direct velocity commands bypass safety system
        
        This method is disabled to enforce single command gateway architecture.
        All robot commands must go through robot_vila_server.py for proper safety validation.
        """
        rospy.logwarn("🚫 DISABLED: send_velocity_commands() called but blocked for safety")
        rospy.logwarn(f"   └── Commands that would have been sent: {commands}")
        rospy.logwarn("   └── Use robot_vila_server.py gateway for ALL robot commands")
        rospy.logwarn("   └── This ensures proper safety validation and single point of control")
        
        # Do NOT publish to /cmd_vel - this would bypass our safety system
    
    def run(self):
        """Main run loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            # Publish status
            self.status_pub.publish(Bool(data=self.model_loaded))
            rate.sleep()

if __name__ == '__main__':
    try:
        node = VILARosNode()
        node.run()
    except rospy.ROSInterruptException:
        pass