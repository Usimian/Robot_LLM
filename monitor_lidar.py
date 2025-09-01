#!/usr/bin/env python3
"""
LiDAR Stability Monitor

This script monitors the LiDAR data to check for infinite values and display stability.
It helps diagnose flickering issues in the display.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
import time

class LiDARMonitor(Node):
    def __init__(self):
        super().__init__('lidar_monitor')
        
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        self.scan_count = 0
        self.inf_count = 0
        self.nan_count = 0
        self.last_report_time = time.time()
        
        self.get_logger().info("🔍 LiDAR Monitor started - checking for stability issues")
    
    def lidar_callback(self, msg):
        self.scan_count += 1
        ranges = np.array(msg.ranges)
        
        # Count problematic values
        inf_values = np.isinf(ranges).sum()
        nan_values = np.isnan(ranges).sum()
        
        self.inf_count += inf_values
        self.nan_count += nan_values
        
        # Report every 10 scans (about every 5 seconds at 2Hz)
        if self.scan_count % 10 == 0:
            current_time = time.time()
            elapsed = current_time - self.last_report_time
            
            self.get_logger().info(f"📊 LiDAR Stability Report (Scan #{self.scan_count}):")
            self.get_logger().info(f"   └── Total points per scan: {len(ranges)}")
            self.get_logger().info(f"   └── Infinite values in this scan: {inf_values}")
            self.get_logger().info(f"   └── NaN values in this scan: {nan_values}")
            self.get_logger().info(f"   └── Front distance: {ranges[0]:.3f}m")
            self.get_logger().info(f"   └── Min distance: {np.min(ranges[np.isfinite(ranges)]):.3f}m")
            self.get_logger().info(f"   └── Max distance: {np.max(ranges[np.isfinite(ranges)]):.3f}m")
            
            # Check for stability issues
            if inf_values > 0 or nan_values > 0:
                self.get_logger().warn(f"⚠️  Stability issue detected!")
                self.get_logger().warn(f"   └── Infinite values: {inf_values}")
                self.get_logger().warn(f"   └── NaN values: {nan_values}")
                
                # Show which angles have problems
                problem_angles = []
                for i, val in enumerate(ranges):
                    if not np.isfinite(val):
                        angle = i * (360.0 / len(ranges))
                        problem_angles.append(f"{angle:.0f}°")
                
                if problem_angles:
                    self.get_logger().warn(f"   └── Problem angles: {', '.join(problem_angles[:10])}")
                    if len(problem_angles) > 10:
                        self.get_logger().warn(f"   └── ... and {len(problem_angles) - 10} more")
            else:
                self.get_logger().info("✅ Scan is stable (no inf/nan values)")
            
            self.last_report_time = current_time
            
            # Overall statistics
            if self.scan_count >= 10:
                inf_rate = (self.inf_count / (self.scan_count * len(ranges))) * 100
                nan_rate = (self.nan_count / (self.scan_count * len(ranges))) * 100
                
                self.get_logger().info(f"📈 Overall Statistics:")
                self.get_logger().info(f"   └── Infinite value rate: {inf_rate:.2f}%")
                self.get_logger().info(f"   └── NaN value rate: {nan_rate:.2f}%")
                
                if inf_rate > 1.0 or nan_rate > 0.1:
                    self.get_logger().warn("⚠️  High rate of problematic values detected!")
                    self.get_logger().warn("   └── This could cause display flickering")
                else:
                    self.get_logger().info("✅ LiDAR data quality is good")
            
            self.get_logger().info("─" * 60)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        monitor = LiDARMonitor()
        print("\n🔍 LiDAR Stability Monitor")
        print("This tool monitors LiDAR data for stability issues that could cause display flickering.")
        print("Press Ctrl+C to stop monitoring.\n")
        
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
