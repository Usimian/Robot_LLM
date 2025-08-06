#!/usr/bin/env python3
"""
Simple test script to continuously read sensor values from robot
Tests the pull-based sensor system by making direct HTTP requests
"""

import requests
import time
import json
from datetime import datetime

class RobotSensorReader:
    def __init__(self, robot_ip="192.168.1.166", robot_port=8080):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.robot_url = f"http://{robot_ip}:{robot_port}"
        self.sensor_url = f"{self.robot_url}/sensors"
        self.image_url = f"{self.robot_url}/image"
        
    def read_sensors(self):
        """Read sensor data from robot"""
        try:
            response = requests.get(self.sensor_url, timeout=3)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def read_image_info(self):
        """Read image metadata from robot (not the full image)"""
        try:
            response = requests.get(self.image_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # Return only metadata, not the full base64 image
                return {
                    "format": data.get("format", "Unknown"),
                    "width": data.get("width", 0),
                    "height": data.get("height", 0),
                    "image_size_bytes": len(data.get("image", "")),
                    "timestamp": data.get("timestamp", "Unknown")
                }
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def format_sensor_display(self, sensor_data):
        """Format sensor data for display"""
        if "error" in sensor_data:
            return f"❌ ERROR: {sensor_data['error']}"
        
        lines = []
        lines.append("📊 ROBOT SENSOR DATA:")
        lines.append(f"  🔋 Battery Voltage: {sensor_data.get('battery_voltage', 'N/A')} V")
        
        # Handle IMU values (acceleration x, y, z)
        imu_values = sensor_data.get('imu_values', {})
        if isinstance(imu_values, dict):
            x = imu_values.get('x', 'N/A')
            y = imu_values.get('y', 'N/A')
            z = imu_values.get('z', 'N/A')
            lines.append(f"  🧭 IMU Acceleration: X:{x} Y:{y} Z:{z} m/s²")
        else:
            lines.append(f"  🧭 IMU Acceleration: {imu_values}")
            
        lines.append(f"  📷 Camera Status: {sensor_data.get('camera_status', 'N/A')}")
        lines.append(f"  🌡️ Temperature: {sensor_data.get('temperature', 'N/A')}°C")
        lines.append(f"  ⏰ Timestamp: {sensor_data.get('timestamp', 'N/A')}")
        
        return "\n".join(lines)
    
    def format_image_display(self, image_info):
        """Format image info for display"""
        if "error" in image_info:
            return f"❌ IMAGE ERROR: {image_info['error']}"
        
        lines = []
        lines.append("📸 ROBOT IMAGE INFO:")
        lines.append(f"  📐 Size: {image_info.get('width', 0)}x{image_info.get('height', 0)}")
        lines.append(f"  📄 Format: {image_info.get('format', 'Unknown')}")
        lines.append(f"  💾 Data Size: {image_info.get('image_size_bytes', 0):,} bytes")
        lines.append(f"  ⏰ Timestamp: {image_info.get('timestamp', 'N/A')}")
        
        return "\n".join(lines)
    
    def run_continuous_test(self, interval=2.0, test_images=False):
        """Run continuous sensor reading test"""
        print(f"🤖 Starting continuous sensor reading from {self.robot_url}")
        print(f"📊 Reading sensors every {interval} seconds")
        if test_images:
            print(f"📸 Also testing image endpoint")
        print(f"🛑 Press Ctrl+C to stop\n")
        
        try:
            while True:
                start_time = time.time()
                current_time = datetime.now().strftime("%H:%M:%S")
                
                print(f"\n{'='*60}")
                print(f"🕒 {current_time} - Reading robot data...")
                print(f"{'='*60}")
                
                # Read sensors
                sensor_data = self.read_sensors()
                print(self.format_sensor_display(sensor_data))
                
                # Optionally read image info
                if test_images:
                    print("\n" + "-"*40)
                    image_info = self.read_image_info()
                    print(self.format_image_display(image_info))
                
                # Calculate timing
                elapsed = time.time() - start_time
                print(f"\n⏱️ Request completed in {elapsed:.2f}s")
                
                # Wait for next interval
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Stopping sensor reader...")
            print(f"✅ Test completed successfully!")

def main():
    """Main test function"""
    print("🔧 Robot Sensor Reader Test")
    print("=" * 50)
    
    # Configuration
    robot_ip = "192.168.1.166"  # Default robot IP
    robot_port = 8080           # Default robot port
    
    # Allow user to override IP if needed
    import sys
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]
        print(f"📡 Using custom robot IP: {robot_ip}")
    
    reader = RobotSensorReader(robot_ip, robot_port)
    
    print(f"🎯 Target robot: {robot_ip}:{robot_port}")
    print(f"📊 Sensor endpoint: {reader.sensor_url}")
    print(f"📸 Image endpoint: {reader.image_url}")
    
    # Test connection first
    print(f"\n🔍 Testing connection...")
    sensor_data = reader.read_sensors()
    
    if "error" in sensor_data:
        print(f"❌ Connection failed: {sensor_data['error']}")
        print(f"💡 Make sure the robot server is running on {robot_ip}:{robot_port}")
        return
    
    print(f"✅ Connection successful!")
    print(reader.format_sensor_display(sensor_data))
    
    # Ask user for test options
    print(f"\n🚀 Starting continuous monitoring...")
    print(f"   - Sensor data every 2 seconds")
    print(f"   - Image info every 10 readings (optional)")
    
    try:
        # Run continuous test
        reader.run_continuous_test(interval=2.0, test_images=True)
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()