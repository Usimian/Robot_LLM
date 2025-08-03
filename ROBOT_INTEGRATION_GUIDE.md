# VILA Robot Integration Guide

## 🤖 Complete Ubuntu Application for Mobile Robot Vision Processing

This guide explains how to use your VILA vision-language model with mobile robots through a comprehensive Ubuntu application that provides multiple communication protocols and robust robot management.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   Mobile Robot  │◄──►│  VILA Robot Hub      │◄──►│  VILA 3B Model  │
│                 │    │  (Ubuntu Server)     │    │                 │
│ • Camera        │    │ • HTTP REST API      │    │ • Vision        │
│ • Sensors       │    │ • WebSocket          │    │ • Language      │
│ • Motors        │    │ • TCP Server         │    │ • Reasoning     │
│ • Control       │    │ • Robot Manager      │    │                 │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Start the VILA Robot Hub

```bash
# Make sure all dependencies are installed
pip3 install flask flask-socketio python-socketio eventlet

# Start the hub (loads VILA automatically)
python3 robot_vila_server.py
```

The hub provides:
- **HTTP REST API** on port 5000
- **WebSocket** for real-time communication  
- **TCP Server** on port 9999 for direct binary communication
- **Robot Management** with state tracking
- **Command Queuing** and control

### 2. Test with Example Clients

```bash
# Run client examples to test different communication methods
python3 robot_client_examples.py
```

## 📡 Communication Protocols

### HTTP REST API (Most Common)

Perfect for standard robots with HTTP capability:

```python
import requests
import base64

# Register robot
robot_info = {
    'robot_id': 'my_robot_01',
    'name': 'My Mobile Robot',
    'capabilities': ['navigation', 'camera'],
    'position': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'heading': 0.0},
    'battery_level': 95.0
}

response = requests.post('http://localhost:5000/robots/register', json=robot_info)

# Send image for analysis
with open('camera_image.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    'image': image_b64,
    'prompt': 'Analyze this for navigation. What should I do?'
}

response = requests.post(
    'http://localhost:5000/robots/my_robot_01/analyze',
    json=payload
)

result = response.json()
print(f"VILA says: {result['analysis']}")
print(f"Commands: {result['commands']}")
```

### WebSocket (Real-time)

For robots needing real-time, low-latency communication:

```python
import socketio

sio = socketio.Client()

@sio.event
def analysis_result(data):
    print(f"Real-time analysis: {data['analysis']}")
    # Execute commands immediately

sio.connect('http://localhost:5000')
sio.emit('register_robot', robot_info)
sio.emit('analyze_image', {'robot_id': 'my_robot', 'image': image_b64})
```

### TCP Direct (Embedded Systems)

For embedded systems, microcontrollers, or custom protocols:

```python
import socket
import json
import struct

# Connect to TCP server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 9999))

# Send structured message
message = {
    'type': 'analyze_image',
    'robot_id': 'embedded_robot',
    'image': image_b64,
    'prompt': 'Navigation analysis needed'
}

# Send with length prefix
data = json.dumps(message).encode('utf-8')
sock.send(struct.pack('!I', len(data)) + data)

# Receive response
length = struct.unpack('!I', sock.recv(4))[0]
response = json.loads(sock.recv(length).decode('utf-8'))
```

## 🎯 API Endpoints

### Robot Management

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check hub status and VILA model |
| `/robots` | GET | List all registered robots |
| `/robots/register` | POST | Register new robot |
| `/robots/{id}/status` | GET/POST | Get/update robot status |
| `/robots/{id}/analyze` | POST | Send image for analysis |
| `/robots/{id}/commands` | GET/POST | Get pending commands or send new |

### Example Responses

**Health Check:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "registered_robots": 3,
  "timestamp": 1703123456.789
}
```

**Analysis Result:**
```json
{
  "success": true,
  "robot_id": "my_robot_01",
  "analysis": "I can see a clear hallway ahead with good lighting. The path appears safe for forward movement. There are no visible obstacles in the immediate path, though I notice a doorway on the left side. The floor looks smooth and suitable for robot navigation.",
  "commands": {
    "move_forward": true,
    "stop": false,
    "turn_left": false,
    "turn_right": false,
    "hazard_detected": false
  },
  "timestamp": 1703123456.789
}
```

## 🤖 Robot Integration Examples

### Raspberry Pi Robot

```python
from picamera2 import Picamera2
import requests
import base64

class PiRobot:
    def __init__(self):
        self.camera = Picamera2()
        self.robot_id = "pi_robot_001"
        self.hub_url = "http://192.168.1.100:5000"  # Hub IP
        
    def navigate_autonomously(self):
        while True:
            # Capture image
            image_path = "/tmp/nav_image.jpg"
            self.camera.capture_file(image_path)
            
            # Send to VILA hub
            with open(image_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            response = requests.post(
                f"{self.hub_url}/robots/{self.robot_id}/analyze",
                json={'image': image_b64}
            )
            
            if response.status_code == 200:
                result = response.json()
                commands = result['commands']
                
                # Execute movement
                if commands['move_forward']:
                    self.move_forward()
                elif commands['turn_left']:
                    self.turn_left()  
                elif commands['turn_right']:
                    self.turn_right()
                else:
                    self.stop()
            
            time.sleep(2)  # Analysis every 2 seconds
```

### Arduino Robot (via Serial Bridge)

```python
import serial
import cv2

class ArduinoRobotBridge:
    def __init__(self):
        self.arduino = serial.Serial('/dev/ttyUSB0', 9600)
        self.camera = cv2.VideoCapture(0)
        self.hub_url = "http://localhost:5000"
        
    def control_loop(self):
        while True:
            # Capture from USB camera
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            # Get VILA analysis
            _, buffer = cv2.imencode('.jpg', frame)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            response = requests.post(
                f"{self.hub_url}/robots/arduino_bot/analyze",
                json={'image': image_b64}
            )
            
            if response.status_code == 200:
                commands = response.json()['commands']
                
                # Send to Arduino
                if commands['move_forward']:
                    self.arduino.write(b'FORWARD\n')
                elif commands['turn_left']:
                    self.arduino.write(b'LEFT\n')
                elif commands['turn_right']:
                    self.arduino.write(b'RIGHT\n')
                else:
                    self.arduino.write(b'STOP\n')
```

### ROS Robot Integration

```python
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import requests

class ROSVILANode:
    def __init__(self):
        rospy.init_node('vila_navigation_node')
        self.bridge = CvBridge()
        
        # Publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Subscribers  
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        self.hub_url = rospy.get_param('~vila_hub_url', 'http://localhost:5000')
        self.robot_id = rospy.get_param('~robot_id', 'ros_robot')
        
    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Send to VILA hub
        _, buffer = cv2.imencode('.jpg', cv_image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        response = requests.post(
            f"{self.hub_url}/robots/{self.robot_id}/analyze",
            json={'image': image_b64}
        )
        
        if response.status_code == 200:
            commands = response.json()['commands']
            
            # Convert to ROS Twist message
            twist = Twist()
            if commands['move_forward']:
                twist.linear.x = 0.3
            elif commands['turn_left']:
                twist.angular.z = 0.5
            elif commands['turn_right']:
                twist.angular.z = -0.5
            # else: stop (twist remains zero)
            
            self.cmd_pub.publish(twist)
```

## ⚙️ Configuration

The hub creates a configuration file `robot_hub_config.ini`:

```ini
[DEFAULT]
http_port = 5000
tcp_port = 9999
log_level = INFO
max_robots = 10

[VILA]
model_name = Efficient-Large-Model/VILA1.5-3b
device = auto
```

## 📊 Monitoring and Logging

### Log Files
- `robot_hub.log` - Complete application logs
- Console output with timestamps and severity levels

### WebSocket Monitoring
```python
import socketio

# Connect as monitor
sio = socketio.Client()
sio.connect('http://localhost:5000')
sio.emit('join_monitors')

@sio.event
def robot_analysis(data):
    print(f"Robot {data['robot_id']} analysis: {data['analysis']}")
```

### REST API Monitoring
```python
# Check hub health
response = requests.get('http://localhost:5000/health')
print(response.json())

# List all robots
response = requests.get('http://localhost:5000/robots')
for robot in response.json()['robots']:
    print(f"Robot {robot['robot_id']}: {robot['status']}")
```

## 🔧 Troubleshooting

### Common Issues

**1. VILA Model Not Loading**
```bash
# Check GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Check disk space (model is ~6GB)
df -h
```

**2. Connection Issues**
```bash
# Check if ports are available
netstat -tulpn | grep :5000
netstat -tulpn | grep :9999

# Test hub connectivity
curl http://localhost:5000/health
```

**3. Image Analysis Failures**
- Ensure images are valid JPEG/PNG
- Check base64 encoding
- Verify image size (recommend < 2MB)

**4. Robot Registration Issues**
- Ensure unique robot_id for each robot
- Check required fields in registration payload
- Verify JSON format

### Performance Optimization

**For High-Frequency Analysis:**
- Use WebSocket instead of HTTP
- Reduce image resolution (640x480 recommended)
- Batch multiple robots on same hub

**For Multiple Robots:**
- Scale horizontally with multiple hub instances
- Use load balancer for HTTP endpoints
- Consider robot priority queuing

## 🎯 Use Cases

### Autonomous Navigation
- **Hallway navigation** - "Is the path clear ahead?"
- **Obstacle avoidance** - "What obstacles do you see?"
- **Door detection** - "Is there a door I can go through?"

### Object Manipulation  
- **Object location** - "Where is the red box?"
- **Grasping assessment** - "Can I pick up this object?"
- **Scene understanding** - "What objects are on the table?"

### Safety and Monitoring
- **Hazard detection** - "Are there any safety hazards?"
- **Person detection** - "Are there people in this area?"
- **Environment assessment** - "Is this area safe for operation?"

### Inspection Tasks
- **Quality control** - "Is this assembly correct?"
- **Damage assessment** - "Do you see any damage?"
- **Maintenance checks** - "What is the condition of this equipment?"

## 📈 Scaling and Production

### Multi-Robot Deployment
```python
# Register multiple robots
robots = [
    {'robot_id': 'warehouse_01', 'capabilities': ['navigation', 'lifting']},
    {'robot_id': 'security_02', 'capabilities': ['navigation', 'monitoring']},
    {'robot_id': 'cleaning_03', 'capabilities': ['navigation', 'cleaning']}
]

for robot_info in robots:
    requests.post('http://localhost:5000/robots/register', json=robot_info)
```

### Load Balancing
```python
import random

# Multiple hub instances
hub_urls = [
    'http://hub1.example.com:5000',
    'http://hub2.example.com:5000', 
    'http://hub3.example.com:5000'
]

# Round-robin or random selection
hub_url = random.choice(hub_urls)
```

### Database Integration
```python
# Add to robot_vila_server.py for persistence
import sqlite3

class RobotDatabase:
    def __init__(self):
        self.conn = sqlite3.connect('robots.db')
        self.create_tables()
    
    def store_analysis(self, robot_id, analysis, timestamp):
        # Store analysis history
        pass
    
    def get_robot_history(self, robot_id):
        # Retrieve robot interaction history
        pass
```

## 🛡️ Security Considerations

### Authentication
```python
# Add API key authentication
@app.before_request
def check_auth():
    if request.endpoint != 'health':
        api_key = request.headers.get('X-API-Key')
        if not validate_api_key(api_key):
            return jsonify({'error': 'Unauthorized'}), 401
```

### Network Security
- Use HTTPS in production
- Implement rate limiting
- Add firewall rules for ports 5000 and 9999
- Consider VPN for robot communication

## 🎓 Advanced Features

### Custom VILA Prompts
```python
# Specialized prompts for different robot types
navigation_prompt = "You are a navigation system for an autonomous mobile robot..."
manipulation_prompt = "You are a vision system for a robotic arm..."
security_prompt = "You are monitoring this area for security purposes..."
```

### Command Scheduling
```python
# Schedule delayed commands
command = RobotCommand(
    robot_id='robot_01',
    command_type='move',
    parameters={'direction': 'forward', 'delay': 5.0},
    timestamp=datetime.now(),
    priority=1
)
```

### Multi-Modal Integration
```python
# Combine vision with other sensors
payload = {
    'image': image_b64,
    'sensor_data': {
        'lidar': lidar_data,
        'imu': imu_data,
        'gps': gps_coordinates
    },
    'prompt': 'Analyze visual and sensor data for navigation'
}
```

---

## 🚀 Ready to Deploy!

Your VILA Robot Hub is now ready to handle mobile robot vision processing at scale. The system provides:

✅ **Multiple Communication Protocols** (HTTP, WebSocket, TCP)  
✅ **Robot State Management** with persistent tracking  
✅ **Real-time Vision Analysis** powered by VILA 3B  
✅ **Command Generation and Queuing**  
✅ **Comprehensive Logging and Monitoring**  
✅ **Production-Ready Architecture**  

Start with the example clients, then adapt the communication protocol that best fits your robot hardware and requirements!