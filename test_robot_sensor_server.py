#!/usr/bin/env python3
"""
Test HTTP sensor server to simulate robot behavior
This simulates what the robot should implement on port 8080
"""

from flask import Flask, jsonify
import time
import random
import threading

app = Flask(__name__)

@app.route('/sensors', methods=['GET'])
def get_sensors():
    """Return simulated sensor readings"""
    # Simulate realistic sensor values that change over time
    sensor_data = {
        "battery_voltage": round(11.5 + random.uniform(0, 1.0), 2),  # 11.5 - 12.5V
        "imu_values": {
            "x": round(random.uniform(-0.1, 0.1), 3),  # Acceleration X (m/s²)
            "y": round(random.uniform(-0.2, 0.2), 3),  # Acceleration Y (m/s²)  
            "z": round(9.8 + random.uniform(-0.2, 0.2), 3)  # Acceleration Z (~9.8 m/s²)
        },
        "camera_status": random.choice(["Active", "Recording", "Idle", "Error"]),
        "temperature": round(40 + random.uniform(0, 20), 1),  # 40-60°C
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    }
    
    print(f"📊 Sensor request received, returning: {sensor_data}")
    return jsonify(sensor_data)

@app.route('/image', methods=['GET'])
def get_camera_image():
    """Return current camera image as base64"""
    import base64
    import io
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a test image with timestamp and sensor info
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Add timestamp and robot info
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    draw.text((10, 10), f"Robot Camera Feed", fill='black')
    draw.text((10, 40), f"Time: {timestamp}", fill='black')
    draw.text((10, 70), f"Robot: yahboomcar_x3_01", fill='black')
    draw.text((10, 100), f"Status: Active", fill='green')
    
    # Add some dynamic elements
    battery = round(11.5 + random.uniform(0, 1.0), 2)
    accel_z = round(9.8 + random.uniform(-0.2, 0.2), 3)
    draw.text((10, 130), f"Battery: {battery}V", fill='red' if battery < 12.0 else 'green')
    draw.text((10, 160), f"IMU Z: {accel_z} m/s²", fill='blue')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    print(f"📸 Image request received, returning {len(image_b64)} bytes")
    return jsonify({
        "image": image_b64,
        "format": "JPEG",
        "width": 640,
        "height": 480,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "robot_id": "yahboomcar_x3_01",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    })

def main():
    """Start the test robot sensor server"""
    print("🤖 Starting test robot sensor server on port 8080")
    print("📊 Simulating sensor data for yahboomcar_x3_01")
    print("🔗 Endpoints:")
    print("   GET http://localhost:8080/sensors - Get sensor data")
    print("   GET http://localhost:8080/image - Get camera image")
    print("   GET http://localhost:8080/health - Health check")
    print("🛑 Press Ctrl+C to stop")
    print()
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Test robot sensor server stopped")

if __name__ == "__main__":
    main()