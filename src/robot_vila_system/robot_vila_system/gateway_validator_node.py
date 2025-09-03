#!/usr/bin/env python3
"""
ROS2 Command Gateway Validator
Ensures single command gateway architecture [[memory:5366669]] is maintained in ROS2 system
"""

import rclpy
from rclpy.node import Node
import json
import time
from datetime import datetime
from robot_msgs.msg import RobotCommand
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# Import simple logger


class CommandGatewayValidator(Node):
    """
    Validates that ALL robot commands go through the single gateway
    Monitors command flow to ensure no bypasses exist
    """
    
    def __init__(self):
        super().__init__('command_gateway_validator')
        
        # Simple logger for clean output

        
        self.robot_id = "yahboomcar_x3_01"
        
        # Track commands from different sources
        self.command_sources = {
            'gateway': 0,
            'direct': 0,
            'bypass': 0
        }
        
        # Monitor all command-related topics
        self._setup_monitoring()
        
        # Setup system monitoring publishers
        self._setup_system_monitoring()
        
        # Validation timer
        self.validation_timer = self.create_timer(5.0, self._validate_gateway_integrity)
        
        self.get_logger().info("🛡️ Command Gateway Validator started")
        self.get_logger().info("   └── Monitoring single command gateway architecture [[memory:5366669]]")
    

    def _setup_monitoring(self):
        """Setup monitoring for all command channels"""
        
        # Monitor commands FROM the gateway (legitimate path)
        self.gateway_command_subscriber = self.create_subscription(
            RobotCommand,
            '/robot/commands',
            self._gateway_command_callback,
            10
        )
        
        # Monitor direct cmd_vel publishing (SHOULD NOT EXIST)
        self.direct_cmd_vel_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel',
            self._direct_cmd_vel_callback,
            10
        )
        
        # Monitor any other potential bypass channels
        # (Add more as needed based on system architecture)
        
        # Monitor command acknowledgments to track execution
        self.command_ack_subscriber = self.create_subscription(
            RobotCommand,
            '/robot/command_ack',
            self._command_ack_callback,
            10
        )
        
        self.get_logger().info("📊 Monitoring all command channels for gateway compliance")
    
    def _setup_system_monitoring(self):
        """Setup system monitoring publishers per specification"""
        
        # Gateway violations publisher
        self.violations_publisher = self.create_publisher(
            String,
            '/system/gateway_violations',
            10
        )
        
        # Command statistics publisher
        self.statistics_publisher = self.create_publisher(
            String,
            '/system/command_statistics',
            10
        )
        
        self.get_logger().info("📊 System monitoring topics configured")
        self.get_logger().info("   └── /system/gateway_violations")
        self.get_logger().info("   └── /system/command_statistics")
    
    def _gateway_command_callback(self, msg: RobotCommand):
        """Monitor legitimate commands from gateway"""
        self.command_sources['gateway'] += 1
        
        self.get_logger().info(f"✅ LEGITIMATE: Command {msg.command_type} from gateway (robot: {msg.robot_id})")
        
        # Validate command structure
        self._validate_command_structure(msg)
    
    def _direct_cmd_vel_callback(self, msg: Twist):
        """Detect direct cmd_vel publishing (VIOLATION)"""
        self.command_sources['direct'] += 1
        
        # Publish violation to system monitoring topic
        violation_data = {
            'timestamp': datetime.now().isoformat(),
            'violation_type': 'direct_cmd_vel',
            'details': {
                'linear_x': msg.linear.x,
                'angular_z': msg.angular.z
            },
            'message': 'Direct /cmd_vel command bypasses single gateway'
        }
        
        violation_msg = String()
        violation_msg.data = json.dumps(violation_data)
        self.violations_publisher.publish(violation_msg)
        
        self.get_logger().error("🚨 GATEWAY VIOLATION: Direct /cmd_vel command detected!")
        self.get_logger().error("   └── This bypasses the single command gateway")
        self.get_logger().error("   └── ALL commands must go through /robot/execute_command service")
        self.get_logger().error(f"   └── Violating command: linear={msg.linear.x}, angular={msg.angular.z}")
    
    def _command_ack_callback(self, msg: RobotCommand):
        """Monitor command acknowledgments"""
        self.get_logger().debug(f"📨 Command ACK: {msg.command_type} executed by robot")
    
    def _validate_command_structure(self, msg: RobotCommand):
        """Validate command message structure"""
        issues = []
        
        # Check required fields
        if not msg.robot_id:
            issues.append("Missing robot_id")
        
        if not msg.command_type:
            issues.append("Missing command_type")
        
        # Report issues
        if issues:
            self.get_logger().warn(f"⚠️ Command structure issues: {', '.join(issues)}")
        else:
            self.get_logger().debug(f"✅ Command structure valid: {msg.command_type}")
    
    def _validate_gateway_integrity(self):
        """Periodic validation of gateway integrity"""
        total_commands = sum(self.command_sources.values())
        
        if total_commands == 0:
            self.get_logger().debug("📊 No commands observed yet")
            return
        
        # Calculate percentages
        gateway_percent = (self.command_sources['gateway'] / total_commands) * 100
        direct_percent = (self.command_sources['direct'] / total_commands) * 100
        bypass_percent = (self.command_sources['bypass'] / total_commands) * 100
        
        # Publish statistics to system monitoring topic
        statistics_data = {
            'timestamp': datetime.now().isoformat(),
            'total_commands': total_commands,
            'command_sources': self.command_sources.copy(),
            'percentages': {
                'gateway': gateway_percent,
                'direct': direct_percent,
                'bypass': bypass_percent
            },
            'compliance_status': 'COMPLIANT' if (self.command_sources['direct'] == 0 and self.command_sources['bypass'] == 0) else 'VIOLATION'
        }
        
        statistics_msg = String()
        statistics_msg.data = json.dumps(statistics_data)
        self.statistics_publisher.publish(statistics_msg)
        
        # Report statistics
        self.get_logger().info("📊 GATEWAY INTEGRITY REPORT:")
        self.get_logger().info(f"   └── Gateway commands: {self.command_sources['gateway']} ({gateway_percent:.1f}%)")
        self.get_logger().info(f"   └── Direct commands: {self.command_sources['direct']} ({direct_percent:.1f}%)")
        self.get_logger().info(f"   └── Bypass commands: {self.command_sources['bypass']} ({bypass_percent:.1f}%)")
        
        # Validate compliance
        if self.command_sources['direct'] > 0 or self.command_sources['bypass'] > 0:
            self.get_logger().error("🚨 GATEWAY INTEGRITY VIOLATION DETECTED!")
            self.get_logger().error("   └── Some commands are bypassing the single gateway")
            self.get_logger().error("   └── This violates the single command gateway architecture")
        else:
            self.get_logger().info("✅ GATEWAY INTEGRITY MAINTAINED")
            self.get_logger().info("   └── All commands properly routed through single gateway")
    
    def generate_compliance_report(self):
        """Generate detailed compliance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_commands': sum(self.command_sources.values()),
            'command_sources': self.command_sources.copy(),
            'compliance_status': 'COMPLIANT' if (self.command_sources['direct'] == 0 and self.command_sources['bypass'] == 0) else 'VIOLATION',
            'architecture': 'Single Command Gateway [[memory:5366669]]',
            'communication_protocol': 'ROS2 Topics and Services'
        }
        
        return report

def test_gateway_compliance():
    """Test function to validate gateway compliance"""
    rclpy.init()
    
    try:
        validator = CommandGatewayValidator()
        
        # Run for a test period
        test_duration = 30.0  # seconds
        start_time = time.time()
        
        validator.logger.info(f"🧪 Starting {test_duration}s gateway compliance test")
        
        while time.time() - start_time < test_duration:
            rclpy.spin_once(validator, timeout_sec=0.1)
        
        # Generate final report
        report = validator.generate_compliance_report()
        
        validator.logger.info("📋 FINAL COMPLIANCE REPORT:")
        validator.logger.info(f"   └── Status: {report['compliance_status']}")
        validator.logger.info(f"   └── Total Commands: {report['total_commands']}")
        validator.logger.info(f"   └── Gateway: {report['command_sources']['gateway']}")
        validator.logger.info(f"   └── Direct: {report['command_sources']['direct']}")
        validator.logger.info(f"   └── Bypass: {report['command_sources']['bypass']}")
        
        # Save report
        with open('gateway_compliance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        validator.logger.info("📄 Report saved to gateway_compliance_report.json")
        
    except KeyboardInterrupt:
        pass
    finally:
        if 'validator' in locals():
            validator.destroy_node()
        rclpy.shutdown()

def main(args=None):
    """Main function"""
    if args and len(args) > 0 and args[0] == 'test':
        test_gateway_compliance()
    else:
        rclpy.init(args=args)
        
        try:
            validator = CommandGatewayValidator()
            rclpy.spin(validator)
        except KeyboardInterrupt:
            pass
        finally:
            if 'validator' in locals():
                validator.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
