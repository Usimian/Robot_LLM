#!/usr/bin/env python3
"""
Robot Client Examples
Demonstrates how robots can communicate with the VILA Robot Hub
"""

import requests
import json
import base64
import time
import cv2
import numpy as np
from PIL import Image
import io
import socket
import struct
import threading
import socketio

class HTTPRobotClient:
    """HTTP REST API client for robot communication"""
    
    def __init__(self, server_url="http://localhost:5000", robot_id="test_robot_01"):
        self.server_url = server_url
        self.robot_id = robot_id
        self.registered = False
        
    def register(self, robot_info=None):
        """Register robot with the hub"""
        if robot_info is None:
            robot_info = {
                'robot_id': self.robot_id,
                'name': f'Test Robot {self.robot_id}',
                'capabilities': ['navigation', 'camera'],
                'connection_type': 'http',
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'heading': 0.0},
                'battery_level': 95.0
            }
        
        try:
            response = requests.post(f"{self.server_url}/robots/register", json=robot_info)
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    self.registered = True
                    print(f"✅ Robot {self.robot_id} registered successfully")
                    return True
            
            print(f"❌ Registration failed: {response.text}")
            return False
            
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False
    
    def send_image_for_analysis(self, image_path_or_array):
        """Send image to hub for analysis"""
        if not self.registered:
            print("❌ Robot not registered. Call register() first.")
            return None
        
        try:
            # Handle different image inputs
            if isinstance(image_path_or_array, str):
                # File path
                with open(image_path_or_array, 'rb') as f:
                    image_data = f.read()
            elif isinstance(image_path_or_array, np.ndarray):
                # OpenCV array
                _, buffer = cv2.imencode('.jpg', image_path_or_array)
                image_data = buffer.tobytes()
            else:
                print("❌ Unsupported image format")
                return None
            
            # Encode to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Send for analysis
            payload = {
                'image': image_b64,
                'prompt': 'Analyze this environment for safe robot navigation. What should I do next?',
                'generate_commands': True
            }
            
            response = requests.post(
                f"{self.server_url}/robots/{self.robot_id}/analyze",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"🤖 VILA Analysis: {result['analysis']}")
                    print(f"📋 Commands: {result['commands']}")
                    return result
            
            print(f"❌ Analysis failed: {response.text}")
            return None
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None
    
    def get_pending_commands(self):
        """Get pending commands from hub"""
        try:
            response = requests.get(f"{self.server_url}/robots/{self.robot_id}/commands")
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result['commands']
            return []
        except Exception as e:
            print(f"❌ Error getting commands: {e}")
            return []
    
    def update_status(self, status_updates):
        """Update robot status"""
        try:
            response = requests.post(
                f"{self.server_url}/robots/{self.robot_id}/status",
                json=status_updates
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Status update error: {e}")
            return False

    def send_sensor_data(self, sensor_data):
        """Send sensor readings to the hub"""
        if not self.registered:
            print("❌ Robot not registered. Call register() first.")
            return False
        
        try:
            payload = {
                'sensor_data': sensor_data
            }
            
            response = requests.post(
                f"{self.server_url}/robots/{self.robot_id}/sensors",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"📊 Sensor data sent successfully")
                    return True
            
            print(f"❌ Sensor update failed: {response.text}")
            return False
            
        except Exception as e:
            print(f"❌ Sensor update error: {e}")
            return False

class TCPRobotClient:
    """TCP client for direct binary communication"""
    
    def __init__(self, server_host="localhost", server_port=9999, robot_id="tcp_robot_01"):
        self.server_host = server_host
        self.server_port = server_port
        self.robot_id = robot_id
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to TCP server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            self.connected = True
            print(f"✅ Connected to TCP server at {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            print(f"❌ TCP connection failed: {e}")
            return False
    
    def send_message(self, message):
        """Send JSON message via TCP"""
        if not self.connected:
            print("❌ Not connected to server")
            return None
        
        try:
            # Convert to JSON bytes
            message_data = json.dumps(message).encode('utf-8')
            message_length = struct.pack('!I', len(message_data))
            
            # Send message
            self.socket.send(message_length + message_data)
            
            # Receive response
            length_data = self.socket.recv(4)
            if not length_data:
                return None
            
            response_length = struct.unpack('!I', length_data)[0]
            response_data = b''
            
            while len(response_data) < response_length:
                chunk = self.socket.recv(response_length - len(response_data))
                if not chunk:
                    break
                response_data += chunk
            
            return json.loads(response_data.decode('utf-8'))
            
        except Exception as e:
            print(f"❌ TCP message error: {e}")
            return None
    
    def register(self):
        """Register robot via TCP"""
        message = {
            'type': 'register',
            'robot_info': {
                'robot_id': self.robot_id,
                'name': f'TCP Robot {self.robot_id}',
                'capabilities': ['navigation', 'camera', 'tcp_direct'],
                'connection_type': 'tcp',
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'heading': 0.0},
                'battery_level': 90.0
            }
        }
        
        response = self.send_message(message)
        if response and response.get('success'):
            print(f"✅ TCP Robot {self.robot_id} registered")
            return True
        else:
            print(f"❌ TCP registration failed: {response}")
            return False
    
    def analyze_image_tcp(self, image_array):
        """Send image for analysis via TCP"""
        try:
            # Encode image
            _, buffer = cv2.imencode('.jpg', image_array)
            image_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            message = {
                'type': 'analyze_image',
                'robot_id': self.robot_id,
                'image': image_b64,
                'prompt': 'TCP robot requesting navigation analysis. What should I do?'
            }
            
            response = self.send_message(message)
            if response and response.get('success'):
                print(f"🤖 TCP Analysis: {response['analysis']}")
                return response
            else:
                print(f"❌ TCP analysis failed: {response}")
                return None
                
        except Exception as e:
            print(f"❌ TCP image analysis error: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("👋 Disconnected from TCP server")

class WebSocketRobotClient:
    """WebSocket client for real-time communication"""
    
    def __init__(self, server_url="http://localhost:5000", robot_id="ws_robot_01"):
        self.server_url = server_url
        self.robot_id = robot_id
        self.sio = socketio.Client()
        self.connected = False
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.sio.event
        def connect():
            print("✅ WebSocket connected to VILA Hub")
            self.connected = True
            
        @self.sio.event
        def disconnect():
            print("👋 WebSocket disconnected")
            self.connected = False
            
        @self.sio.event
        def registration_response(data):
            if data.get('success'):
                print(f"✅ WebSocket robot {data.get('robot_id')} registered")
            else:
                print(f"❌ WebSocket registration failed: {data.get('error')}")
                
        @self.sio.event
        def analysis_result(data):
            if data.get('success'):
                print(f"🤖 WebSocket Analysis: {data.get('analysis')}")
                print(f"📋 Commands: {data.get('commands')}")
            else:
                print(f"❌ WebSocket analysis error: {data.get('error')}")
    
    def connect_ws(self):
        """Connect to WebSocket server"""
        try:
            self.sio.connect(self.server_url)
            return True
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return False
    
    def register(self):
        """Register robot via WebSocket"""
        if not self.connected:
            print("❌ Not connected to WebSocket server")
            return False
        
        robot_info = {
            'robot_id': self.robot_id,
            'name': f'WebSocket Robot {self.robot_id}',
            'capabilities': ['navigation', 'camera', 'realtime'],
            'connection_type': 'websocket',
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'heading': 0.0},
            'battery_level': 88.0
        }
        
        self.sio.emit('register_robot', robot_info)
        return True
    
    def analyze_image_ws(self, image_array):
        """Send image for real-time analysis"""
        if not self.connected:
            print("❌ Not connected to WebSocket server")
            return False
        
        try:
            # Encode image
            _, buffer = cv2.imencode('.jpg', image_array)
            image_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            data = {
                'robot_id': self.robot_id,
                'image': image_b64,
                'prompt': 'Real-time navigation analysis needed. What should I do next?'
            }
            
            self.sio.emit('analyze_image', data)
            return True
            
        except Exception as e:
            print(f"❌ WebSocket image analysis error: {e}")
            return False
    
    def disconnect_ws(self):
        """Disconnect from WebSocket"""
        if self.connected:
            self.sio.disconnect()

def demo_http_client():
    """Demonstrate HTTP client usage"""
    print("\n🚀 === HTTP Client Demo ===")
    
    client = HTTPRobotClient()
    
    # Register robot
    if not client.register():
        return
    
    # Create a test image (simulated camera capture)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Test Robot View", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Send for analysis
    result = client.send_image_for_analysis(test_image)
    
    if result:
        # Check for pending commands
        time.sleep(1)
        commands = client.get_pending_commands()
        if commands:
            print(f"📬 Pending commands: {len(commands)}")
            for cmd in commands:
                print(f"  - {cmd['command_type']}: {cmd['parameters']}")
        
        # Update status
        client.update_status({
            'position': {'x': 1.0, 'y': 0.5, 'z': 0.0, 'heading': 45.0},
            'battery_level': 92.0,
            'status': 'active'
        })

def demo_tcp_client():
    """Demonstrate TCP client usage"""
    print("\n🚀 === TCP Client Demo ===")
    
    client = TCPRobotClient()
    
    if not client.connect():
        return
    
    try:
        # Register robot
        if not client.register():
            return
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "TCP Robot View", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Send for analysis
        result = client.analyze_image_tcp(test_image)
        
    finally:
        client.disconnect()

def demo_websocket_client():
    """Demonstrate WebSocket client usage"""
    print("\n🚀 === WebSocket Client Demo ===")
    
    client = WebSocketRobotClient()
    
    if not client.connect_ws():
        return
    
    try:
        # Wait for connection
        time.sleep(1)
        
        # Register robot
        client.register()
        time.sleep(1)
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "WebSocket Robot", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Send for real-time analysis
        client.analyze_image_ws(test_image)
        
        # Wait for response
        time.sleep(3)
        
    finally:
        client.disconnect_ws()

def demo_sensor_data():
    """Demo sending REAL sensor data to the hub (for testing only)"""
    print("\n📊 Testing Sensor Data Transmission...")
    print("⚠️  This demo sends simulated sensor data for testing purposes only")
    print("⚠️  Real robots should send actual sensor readings")
    
    # Create HTTP client
    client = HTTPRobotClient(robot_id="sensor_demo_robot")
    
    # Register with sensor capabilities
    robot_info = {
        'robot_id': 'sensor_demo_robot',
        'name': 'Sensor Demo Robot',
        'capabilities': ['navigation', 'camera', 'sensors'],
        'connection_type': 'http',
        'position': {'x': 1.5, 'y': 2.0, 'z': 0.0, 'heading': 45.0},
        'battery_level': 87.5,
        'sensor_data': {
            'battery_voltage': 12.4,
            'temperature': 35.2,
            'lidar_distance': 2.8,
            'imu_heading': 45.0,
            'camera_status': 'active'
        }
    }
    
    if not client.register(robot_info):
        print("❌ Failed to register robot")
        return
    
    # Simulate sending periodic sensor updates
    print("📊 Sending sensor data updates...")
    import random
    
    for i in range(5):
        # Simulate changing sensor values
        sensor_data = {
            'battery_voltage': 12.4 - (i * 0.1) + random.uniform(-0.1, 0.1),
            'temperature': 35.2 + random.uniform(-2.0, 3.0),
            'lidar_distance': 2.8 + random.uniform(-1.0, 1.0),
            'imu_heading': (45.0 + i * 10) % 360,
            'camera_status': 'active' if i % 2 == 0 else 'processing',
            'gps_lat': 37.7749 + random.uniform(-0.001, 0.001),
            'gps_lon': -122.4194 + random.uniform(-0.001, 0.001)
        }
        
        if client.send_sensor_data(sensor_data):
            print(f"  Update {i+1}/5: Battery {sensor_data['battery_voltage']:.2f}V, "
                  f"Temp {sensor_data['temperature']:.1f}°C, "
                  f"Lidar {sensor_data['lidar_distance']:.2f}m")
        
        time.sleep(2)  # Wait 2 seconds between updates
    
    print("✅ Sensor data demo completed")

def main():
    """Main demo function"""
    print("🤖 VILA Robot Hub Client Examples")
    print("Make sure the robot hub server is running on localhost:5000")
    
    # Check if server is available
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server is healthy, VILA loaded: {health.get('model_loaded', False)}")
        else:
            print("❌ Server is not responding properly")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Start robot_vila_server.py first!")
        return
    
    print("\nChoose demo:")
    print("1. HTTP REST API client")
    print("2. TCP direct client")
    print("3. WebSocket real-time client")
    print("4. Sensor Data Demo")
    print("5. Run all demos")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        demo_http_client()
    elif choice == "2":
        demo_tcp_client()
    elif choice == "3":
        demo_websocket_client()
    elif choice == "4":
        demo_sensor_data()
    elif choice == "5":
        demo_http_client()
        demo_tcp_client()
        demo_websocket_client()
        demo_sensor_data()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()