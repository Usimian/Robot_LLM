# 🤖 Robot Integration Guide - Single Robot System

## Overview for Robot Developers

This guide is for developers working on the **robot side** (Jetson Orin Nano) who need to integrate with the **Single Robot VILA System**. The system has been completely redesigned for **one robot** with maximum efficiency and security.

> **🎯 Critical**: This is a **SINGLE ROBOT SYSTEM** - all complexity around multiple robots has been removed. Your robot ID is hardcoded as `yahboomcar_x3_01`.

## 🏗️ System Architecture

```
┌─────────────────────────────────────┐
│         Client PC (Ubuntu 22.04)    │
│         IP: 192.168.1.XXX           │
│    ┌─────────────────────────────┐   │
│    │   Single Robot Controller   │   │
│    │   • One VILA Model          │   │
│    │   • HTTP + WebSocket        │   │
│    │   • Single Command Gateway  │   │
│    │   • GUI Safety Control      │   │
│    └─────────────────────────────┘   │
│              Port 5000              │
└─────────────────┬───────────────────┘
                  │ HTTP Only
                  │ (Simplified)
┌─────────────────▼───────────────────┐
│    Jetson Orin Nano (Your Robot)    │
│         IP: 192.168.1.166           │
│  ┌─────────────────────────────────┐ │
│  │     yahboomcar_x3_01 System     │ │
│  │  • Sends Images → Server        │ │
│  │  • Polls for Commands ← Server  │ │
│  │  • Executes Movement Commands   │ │
│  │  • Reports Sensor Data          │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

## 🔄 What Changed from Old System

### **Old Multi-Robot System (Removed)**
- ❌ Complex robot registration
- ❌ Multiple communication protocols (TCP/HTTP/WebSocket)
- ❌ IP address discovery
- ❌ Robot management overhead

### **New Single Robot System**
- ✅ **Hardcoded robot configuration** (yahboomcar_x3_01)
- ✅ **HTTP-only communication** (simplified)
- ✅ **Single command gateway** (no bypasses)
- ✅ **Config file setup** (easy to change)

## 📡 Communication Protocol

### **🔄 Client → Robot (Request Camera Image)**
```http
GET http://192.168.1.166:8080/image
```

**Robot Response:**
```json
{
  "image": "base64_encoded_jpeg_data",
  "format": "JPEG",
  "width": 640,
  "height": 480,
  "timestamp": "2025-08-06T11:45:00.000Z"
}
```

### **🔄 Client → Server (Request VILA Analysis)**
```http
GET http://192.168.1.XXX:5000/robots/yahboomcar_x3_01/analyze?prompt=Analyze%20environment
```

**Server Response:**
```json
{
  "success": true,
  "message": "Analysis complete - commands will be sent separately if approved",
  "timestamp": 1754495000.123
}
```

> **🎯 Architecture Change**: All communication is now **CLIENT-INITIATED** (pull-based). The client requests images and analysis when needed, rather than the robot continuously pushing data.

> **🛡️ Security Note**: Server **NEVER** returns VILA responses or parsed commands to prevent autonomous execution bypassing safety.

### **🔄 Robot → Server (Get Commands)**
```http
GET http://192.168.1.XXX:5000/robots/yahboomcar_x3_01/commands
```

**Response:**
```json
{
  "success": true,
  "robot_id": "yahboomcar_x3_01",
  "commands": [
    {
      "robot_id": "yahboomcar_x3_01",
      "command_type": "move",
      "parameters": {
        "direction": "forward",
        "speed": 0.3,
        "duration": 2.0
      },
      "timestamp": "2025-08-06T11:45:09.123456",
      "priority": 1
    }
  ],
  "count": 1
}
```

### **🔄 Client → Robot (Request Sensor Data)**
```http
GET http://192.168.1.166:8080/sensors
```

**Robot Response:**
```json
{
  "battery_voltage": 12.4,
  "imu_values": {
    "x": -0.007,
    "y": 0.132,
    "z": 9.984
  },
  "camera_status": "Active",
  "temperature": 45.2,
  "timestamp": "2025-08-06T11:45:00.000Z"
}
```

> **🎯 Architecture Change**: Sensor data is now **CLIENT-INITIATED** (pull-based) rather than robot-initiated (push-based). The client requests sensor data when needed, making the system more efficient.

## 🤖 Robot Implementation Requirements

### **1. Robot Configuration**

Your robot is **hardcoded** in the server config:
```ini
[ROBOT]
robot_id = yahboomcar_x3_01
robot_name = YahBoom Car X3  
robot_ip = 192.168.1.166
robot_port = 8080
```

> **📝 Note**: The `robot_port` (8080) is now used for the robot's HTTP sensor server.

### **2. Required Robot Behavior**

#### **A. HTTP Server for Data Requests**
```python
from flask import Flask, jsonify
import time
import base64
import cv2

app = Flask(__name__)

@app.route('/sensors', methods=['GET'])
def get_sensors():
    """Return current sensor readings"""
    sensor_data = {
        "battery_voltage": get_battery_voltage(),
        "imu_values": get_imu_values(),
        "camera_status": get_camera_status(),
        "temperature": get_temperature(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    }
    return jsonify(sensor_data)

@app.route('/image', methods=['GET'])
def get_camera_image():
    """Return current camera image as base64"""
    try:
        # Capture image from camera
        cap = cv2.VideoCapture(0)  # Adjust camera index as needed
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({"error": "Failed to capture image"}), 500
        
        # Encode image as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "image": image_b64,
            "format": "JPEG",
            "width": frame.shape[1],
            "height": frame.shape[0],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start HTTP server on port 8080
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
```

#### **B. Command Polling Loop**
```python
def poll_for_commands(self):
    """Poll server for movement commands"""
    try:
        response = requests.get(
            f"{self.server_url}/robots/{self.robot_id}/commands",
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            commands = result.get('commands', [])
            
            for command in commands:
                self.execute_command(command)
                
            return len(commands)
        else:
            print(f"❌ Command poll failed: {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"❌ Command poll error: {e}")
        return 0

def execute_command(self, command):
    """Execute a movement command"""
    cmd_type = command.get('command_type')
    params = command.get('parameters', {})
    
    print(f"🤖 Executing: {cmd_type} with params: {params}")
    
    if cmd_type == 'move':
        direction = params.get('direction')
        speed = params.get('speed', 0.3)
        duration = params.get('duration', 2.0)
        
        if direction == 'forward':
            self.move_forward(speed, duration)
        elif direction == 'backward':
            self.move_backward(speed, duration)
        elif direction == 'left':
            self.turn_left(speed, duration)
        elif direction == 'right':
            self.turn_right(speed, duration)
            
    elif cmd_type == 'turn':
        direction = params.get('direction')
        angle = params.get('angle', 45)
        speed = params.get('speed', 0.2)
        
        if direction == 'left':
            self.turn_left(speed, angle/45.0)  # Convert angle to duration
        elif direction == 'right':
            self.turn_right(speed, angle/45.0)
            
    elif cmd_type == 'stop':
        self.stop_robot()
    
    print(f"✅ Command {cmd_type} completed")
```

#### **C. Sensor Reading Functions**
```python
def get_battery_voltage():
    """Read battery voltage from ADC or power monitor"""
    # Implement actual battery voltage reading
    # Example for ADC reading:
    # import Adafruit_ADS1x15
    # adc = Adafruit_ADS1x15.ADS1115()
    # raw_value = adc.read_adc(0, gain=1)
    # voltage = raw_value * 4.096 / 32767.0 * voltage_divider_ratio
    return 12.4  # Replace with actual reading

def get_imu_values():
    """Read acceleration values from IMU"""
    # Implement actual IMU reading for acceleration
    # Example for MPU9250:
    # import mpu9250
    # sensor = mpu9250.MPU9250()
    # accel = sensor.readAccel()
    # return {"x": accel['x'], "y": accel['y'], "z": accel['z']}
    return {"x": -0.007, "y": 0.132, "z": 9.984}  # Replace with actual reading

def get_camera_status():
    """Check camera operational status"""
    # Implement actual camera status check
    # Example:
    # try:
    #     cap = cv2.VideoCapture(0)
    #     if cap.isOpened():
    #         return "Active"
    #     else:
    #         return "Error"
    # except:
    #     return "Offline"
    return "Active"  # Replace with actual check

def get_temperature():
    """Read system temperature"""
    # Read from system temperature sensor
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0  # Convert from millicelsius
            return temp
    except:
        return 45.2  # Default if can't read
```

### **3. Main Robot Loop**
```python
def main_robot_loop(self):
    """Main robot control loop - PULL-BASED SYSTEM"""
    print(f"🤖 Starting robot {self.robot_id}")
    
    # Start HTTP server for data requests in background thread
    server_thread = threading.Thread(target=self.start_data_server, daemon=True)
    server_thread.start()
    
    last_command_time = 0
    
    while True:
        current_time = time.time()
        
        try:
            # NOTE: No image sending loop needed - client will request images via GET /image
            
            # Poll for commands (every 0.5 seconds)
            if current_time - last_command_time > 0.5:
                command_count = self.poll_for_commands()
                last_command_time = current_time
                
        except KeyboardInterrupt:
            print("🛑 Robot stopping...")
            self.stop_robot()
            break
        except Exception as e:
            print(f"❌ Robot loop error: {e}")
            time.sleep(1)  # Brief pause before retrying
        
        time.sleep(0.1)  # Small delay to prevent CPU spinning

def start_data_server(self):
    """Start HTTP server for sensor data and image requests"""
    print("📊 Starting data HTTP server on port 8080")
    print("   GET /sensors - Sensor data")
    print("   GET /image - Camera image")
    app.run(host='0.0.0.0', port=8080, debug=False)
```

## 🛡️ Safety Requirements

### **Critical Safety Rules**

1. **🚫 NEVER execute VILA responses directly**
   - Server will not send VILA analysis text
   - Only execute explicit movement commands from command queue

2. **🛡️ Always check for stop commands first**
   - Process stop commands immediately
   - Cancel any ongoing movement

3. **⏱️ Implement command timeouts**
   - Stop if no commands received for 30+ seconds
   - Prevent runaway robot behavior

4. **🔋 Monitor battery levels**
   - Stop robot if battery < 20%
   - Send accurate battery data to server

### **Example Safety Implementation**
```python
def execute_command_safely(self, command):
    """Execute command with safety checks"""
    
    # 1. Always check stop first
    if command.get('command_type') == 'stop':
        self.emergency_stop()
        return
    
    # 2. Check battery level
    if self.get_battery_percentage() < 20:
        print("🔋 Low battery - stopping robot")
        self.emergency_stop()
        return
    
    # 3. Check for obstacles (if you have sensors)
    if self.detect_obstacle():
        print("🚧 Obstacle detected - stopping")
        self.emergency_stop()
        return
    
    # 4. Execute command with timeout
    self.execute_command_with_timeout(command, timeout=5.0)

def emergency_stop(self):
    """Immediately stop all robot movement"""
    # Stop motors
    self.stop_all_motors()
    
    # Clear any movement queues
    self.clear_movement_queue()
    
    print("🛑 Emergency stop executed")
```

## 🔧 Network Configuration

### **Required Network Setup**
- **Server IP**: `192.168.1.XXX` (replace with actual)
- **Robot IP**: `192.168.1.166` (your Jetson)
- **Server Port**: `5000` (HTTP + WebSocket)
- **Robot Port**: Not required (robot is client-only)

### **Firewall Rules**
```bash
# On robot (Jetson), allow outbound to server
sudo ufw allow out 5000

# On server (Client PC), allow inbound from robot  
sudo ufw allow from 192.168.1.166 to any port 5000
```

## 🧪 Testing Your Integration

### **1. Test Image Requests**
```bash
# From client, test robot image endpoint
curl http://192.168.1.166:8080/image

# From client, test VILA analysis (pull-based)
curl "http://192.168.1.XXX:5000/robots/yahboomcar_x3_01/analyze?prompt=test"
```

### **2. Test Command Polling**
```bash
# From robot, test command retrieval
curl http://192.168.1.XXX:5000/robots/yahboomcar_x3_01/commands
```

### **3. Test Sensor Data**
```bash
# From client, test sensor request
curl http://192.168.1.166:8080/sensors

# From client through server
curl http://192.168.1.XXX:5000/robots/yahboomcar_x3_01/sensors
```

## 📋 Implementation Checklist

### **Phase 1: Basic Communication**
- [ ] **HTTP server** for sensor data (port 8080)
- [ ] **HTTP client** for server communication
- [ ] **Image capture** and base64 encoding
- [ ] **Command polling** every 0.5 seconds
- [ ] **Basic movement** functions (forward, backward, left, right, stop)

### **Phase 2: Safety Integration**
- [ ] **Emergency stop** function
- [ ] **Battery monitoring** and reporting
- [ ] **Command timeouts** (stop if no commands)
- [ ] **Obstacle detection** (if available)

### **Phase 3: Sensor Integration**
- [ ] **Battery voltage** reading from ADC
- [ ] **IMU acceleration values** (x, y, z) from accelerometer
- [ ] **Camera status** monitoring
- [ ] **Orin CPU Temperature** monitoring
- [ ] **HTTP sensor endpoint** (/sensors)

### **Phase 4: Optimization**
- [ ] **Error handling** and retry logic
- [ ] **Connection recovery** if server unavailable
- [ ] **Logging** for debugging
- [ ] **Performance tuning** (image compression, polling frequency)

## 🚨 Troubleshooting

### **Common Issues**

1. **"Connection refused" errors**
   - Check server IP address in robot code
   - Verify server is running on port 5000
   - Test with `curl` from robot

2. **"No commands received"**
   - Verify robot ID is exactly `yahboomcar_x3_01`
   - Check command polling frequency
   - Enable GUI movement in server

3. **"Analysis failed" errors**
   - Check image format (JPEG/PNG work best)
   - Verify base64 encoding is correct
   - Check image size (< 1MB recommended)

4. **Robot doesn't move**
   - Verify movement functions work locally
   - Check for safety blocks in server logs
   - Ensure GUI "Movement ENABLED" is set

### **Debug Commands**
```bash
# Check server status from robot
curl http://192.168.1.XXX:5000/health

# Check robot registration
curl http://192.168.1.XXX:5000/robots

# Monitor server logs
tail -f /path/to/robot_hub.log | grep yahboomcar_x3_01
```

## 📞 Support

If you encounter issues:

1. **Check server logs** for error messages
2. **Test network connectivity** between robot and server
3. **Verify robot ID** matches exactly: `yahboomcar_x3_01`
4. **Test each component** separately (image, commands, sensors)

The system is designed to be simple and reliable for your single robot setup. Focus on getting basic movement working first, then add sensors and safety features.

---

**🎯 Remember**: This is a **single robot system** - all the complexity of multi-robot management has been removed. Your robot just needs to send images, poll for commands, and report sensor data. Keep it simple!