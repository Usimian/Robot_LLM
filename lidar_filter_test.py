#!/usr/bin/env python3
"""
LiDAR Filter Test

This script tests different filtering approaches to eliminate display flickering
caused by infinite values in LiDAR data.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
import time

class LiDARFilterTest(Node):
    def __init__(self):
        super().__init__('lidar_filter_test')
        
        # Create filtered LiDAR publisher
        self.filtered_publisher = self.create_publisher(
            LaserScan,
            '/scan_filtered',
            10
        )
        
        # Subscribe to raw LiDAR
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        self.previous_ranges = None
        self.filter_stats = {
            'total_scans': 0,
            'inf_values_filtered': 0,
            'nan_values_filtered': 0
        }
        
        self.get_logger().info("🔧 LiDAR Filter Test started")
        self.get_logger().info("   └── Publishing filtered data to /scan_filtered")
    
    def advanced_filter(self, ranges, msg):
        """Apply advanced filtering to eliminate flickering"""
        
        # Convert to numpy array for efficient processing
        ranges = np.array(ranges, dtype=np.float64)
        
        # Count problematic values before filtering
        inf_count = np.isinf(ranges).sum()
        nan_count = np.isnan(ranges).sum()
        
        self.filter_stats['inf_values_filtered'] += inf_count
        self.filter_stats['nan_values_filtered'] += nan_count
        
        # Step 1: Handle infinite and NaN values
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        
        # Step 2: Clamp to reasonable sensor limits
        ranges = np.where(ranges > msg.range_max, msg.range_max, ranges)
        ranges = np.where(ranges < msg.range_min, msg.range_min, ranges)
        
        # Step 3: Additional range validation (prevent extreme values)
        ranges = np.where(ranges > 10.0, 10.0, ranges)  # Cap at 10m
        ranges = np.where(ranges < 0.05, 0.05, ranges)  # Minimum 5cm
        
        # Step 4: Temporal smoothing to reduce flickering
        if self.previous_ranges is not None and len(self.previous_ranges) == len(ranges):
            # Exponential moving average with outlier detection
            alpha = 0.2  # Smoothing factor
            
            # Calculate change rate to detect outliers
            change_rate = np.abs(ranges - self.previous_ranges)
            outlier_threshold = 2.0  # 2m change threshold
            
            # Apply smoothing, but allow rapid changes for real obstacles
            smooth_mask = change_rate < outlier_threshold
            ranges = np.where(smooth_mask, 
                             alpha * ranges + (1 - alpha) * self.previous_ranges,
                             ranges)  # Keep rapid changes as-is
        
        # Step 5: Median filtering for noise reduction
        if len(ranges) > 5:
            # Apply median filter with window size 5
            filtered_ranges = np.copy(ranges)
            for i in range(2, len(ranges) - 2):
                window = ranges[i-2:i+3]
                filtered_ranges[i] = np.median(window)
            ranges = filtered_ranges
        
        # Store for next iteration
        self.previous_ranges = ranges.copy()
        
        return ranges.tolist()
    
    def lidar_callback(self, msg):
        self.filter_stats['total_scans'] += 1
        
        # Apply advanced filtering
        filtered_ranges = self.advanced_filter(msg.ranges, msg)
        
        # Create filtered message
        filtered_msg = LaserScan()
        filtered_msg.header = msg.header
        filtered_msg.angle_min = msg.angle_min
        filtered_msg.angle_max = msg.angle_max
        filtered_msg.angle_increment = msg.angle_increment
        filtered_msg.time_increment = msg.time_increment
        filtered_msg.scan_time = msg.scan_time
        filtered_msg.range_min = msg.range_min
        filtered_msg.range_max = msg.range_max
        filtered_msg.ranges = filtered_ranges
        filtered_msg.intensities = msg.intensities
        
        # Publish filtered data
        self.filtered_publisher.publish(filtered_msg)
        
        # Report statistics every 50 scans
        if self.filter_stats['total_scans'] % 50 == 0:
            total_points = self.filter_stats['total_scans'] * len(msg.ranges)
            inf_rate = (self.filter_stats['inf_values_filtered'] / total_points) * 100
            nan_rate = (self.filter_stats['nan_values_filtered'] / total_points) * 100
            
            self.get_logger().info(f"📊 Filter Statistics (Scan #{self.filter_stats['total_scans']}):")
            self.get_logger().info(f"   └── Infinite values filtered: {inf_rate:.2f}%")
            self.get_logger().info(f"   └── NaN values filtered: {nan_rate:.2f}%")
            self.get_logger().info(f"   └── Filtered LiDAR published to /scan_filtered")
            
            if inf_rate < 1.0 and nan_rate < 0.1:
                self.get_logger().info("✅ LiDAR filtering working well!")
            else:
                self.get_logger().warn("⚠️  High rate of problematic values still detected")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        filter_test = LiDARFilterTest()
        print("\n🔧 LiDAR Filter Test")
        print("This tool applies advanced filtering to eliminate display flickering.")
        print("Filtered data is published to /scan_filtered topic.")
        print("Press Ctrl+C to stop.\n")
        
        rclpy.spin(filter_test)
    except KeyboardInterrupt:
        print("\n🛑 Filter test stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
