#!/usr/bin/env python3
"""
Test script to send simulated sensor data to the VILA Robot Hub
This simulates what the real robot should be doing
"""

import requests
import time
import random
import json

def send_sensor_data():
    """Send simulated sensor data to the server"""
    
    # Simulate realistic sensor values that change over time
    battery_voltage = 12.0 + random.uniform(-0.5, 0.5)  # 11.5 - 12.5V
    imu_heading = random.uniform(0, 360)  # 0-360 degrees
    camera_status = random.choice(["Active", "Recording", "Idle", "Error"])
    lidar_distance = random.uniform(0.5, 5.0)  # 0.5 - 5.0 meters
    
    sensor_data = {
        "sensor_data": {
            "battery_voltage": round(battery_voltage, 2),
            "imu_heading": round(imu_heading, 1),
            "camera_status": camera_status,
            "lidar_distance": round(lidar_distance, 2)
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/robots/yahboomcar_x3_01/sensors",
            headers={"Content-Type": "application/json"},
            json=sensor_data,
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✅ Sensor data sent: Battery={battery_voltage:.2f}V, Heading={imu_heading:.1f}°, Camera={camera_status}, Lidar={lidar_distance:.2f}m")
        else:
            print(f"❌ Failed to send sensor data: HTTP {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending sensor data: {e}")

def main():
    """Main loop to send sensor data every 2 seconds"""
    print("🤖 Starting sensor data simulation...")
    print("📊 Sending sensor data to yahboomcar_x3_01 every 2 seconds")
    print("🛑 Press Ctrl+C to stop")
    print()
    
    try:
        while True:
            send_sensor_data()
            time.sleep(2)  # Send every 2 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Sensor simulation stopped")

if __name__ == "__main__":
    main()