#!/usr/bin/env python3
"""
VILA Robot Vision Hub
Comprehensive Ubuntu application for mobile robot vision processing
Supports multiple communication protocols and robot management
"""

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import io
import json
import logging
import configparser
import queue
import socket
import struct
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import threading
import time
import torch
import sys
import os

# Add VILA paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VILA'))

# Import your existing VILA model class
from main_vila import VILAModel

# Initialize Flask app with WebSocket support
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vila_robot_hub_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robot_hub.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('VILARobotHub')

@dataclass
class RobotStatus:
    """Robot status information"""
    robot_id: str
    name: str
    last_seen: datetime
    position: Dict[str, float]  # x, y, z, heading
    battery_level: float
    status: str  # active, idle, error, offline
    capabilities: List[str]  # navigation, manipulation, etc.
    connection_type: str  # http, websocket, tcp

@dataclass
class RobotCommand:
    """Robot control command"""
    robot_id: str
    command_type: str  # move, turn, stop, manipulate
    parameters: Dict[str, Any]
    timestamp: datetime
    priority: int = 1

class RobotManager:
    """Manages connected robots and their states"""
    
    def __init__(self):
        self.robots: Dict[str, RobotStatus] = {}
        self.command_queues: Dict[str, queue.Queue] = {}
        self.lock = threading.Lock()
        
    def register_robot(self, robot_info: Dict) -> bool:
        """Register a new robot"""
        try:
            robot_id = robot_info['robot_id']
            with self.lock:
                # Get battery level, default to unknown if not provided
                provided_battery = robot_info.get('battery_level')
                if provided_battery is None:
                    battery_level = 0.0  # Unknown battery - force robot to send real data
                    logger.warning(f"⚠️ Robot {robot_id} registered without battery level - defaulting to 0% (please send real battery data)")
                else:
                    battery_level = float(provided_battery)
                    logger.info(f"🔋 Robot {robot_id} registered with battery level: {battery_level:.1f}%")
                
                self.robots[robot_id] = RobotStatus(
                    robot_id=robot_id,
                    name=robot_info.get('name', f'Robot_{robot_id}'),
                    last_seen=datetime.now(),
                    position=robot_info.get('position', {'x': 0, 'y': 0, 'z': 0, 'heading': 0}),
                    battery_level=battery_level,
                    status='active',
                    capabilities=robot_info.get('capabilities', ['navigation']),
                    connection_type=robot_info.get('connection_type', 'http')
                )
                self.command_queues[robot_id] = queue.Queue()
            
            logger.info(f"Robot {robot_id} registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register robot: {e}")
            return False
    
    def update_robot_status(self, robot_id: str, updates: Dict) -> bool:
        """Update robot status"""
        try:
            with self.lock:
                if robot_id in self.robots:
                    robot = self.robots[robot_id]
                    robot.last_seen = datetime.now()
                    
                    # Log battery level updates specifically
                    if 'battery_level' in updates:
                        old_battery = robot.battery_level
                        logger.info(f"🔋 Robot {robot_id} battery update: {old_battery:.1f}% → {updates['battery_level']:.1f}%")
                    
                    for key, value in updates.items():
                        if hasattr(robot, key):
                            setattr(robot, key, value)
                    
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to update robot {robot_id}: {e}")
            return False
    
    def get_robot(self, robot_id: str) -> Optional[RobotStatus]:
        """Get robot by ID"""
        with self.lock:
            return self.robots.get(robot_id)
    
    def get_all_robots(self) -> List[RobotStatus]:
        """Get all registered robots"""
        with self.lock:
            return list(self.robots.values())
    
    def send_command(self, robot_id: str, command: RobotCommand) -> bool:
        """Send command to robot"""
        try:
            if robot_id in self.command_queues:
                self.command_queues[robot_id].put(command)
                logger.info(f"Command sent to robot {robot_id}: {command.command_type}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to send command to robot {robot_id}: {e}")
            return False
    
    def get_pending_commands(self, robot_id: str) -> List[RobotCommand]:
        """Get pending commands for robot"""
        commands = []
        if robot_id in self.command_queues:
            try:
                while not self.command_queues[robot_id].empty():
                    commands.append(self.command_queues[robot_id].get_nowait())
            except queue.Empty:
                pass
        return commands

class RobotVILAService:
    """Enhanced VILA service with robot management"""
    
    def __init__(self):
        self.vila_model = VILAModel()
        self.model_loaded = False
        self.robot_manager = RobotManager()
        self.tcp_server = None
        self.tcp_thread = None
        
    def initialize(self):
        """Initialize VILA model and services"""
        logger.info("🚀 Loading VILA model for robot service...")
        success = self.vila_model.load_model()
        if success:
            self.model_loaded = True
            logger.info("✅ VILA model ready for robot requests")
            
            # Start TCP server for direct robot communication
            self.start_tcp_server()
        return success
    
    def start_tcp_server(self, port=9999):
        """Start TCP server for direct robot communication"""
        def tcp_server_worker():
            try:
                self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.tcp_server.bind(('0.0.0.0', port))
                self.tcp_server.listen(5)
                logger.info(f"🌐 TCP server listening on port {port}")
                
                while True:
                    client_socket, address = self.tcp_server.accept()
                    logger.info(f"TCP connection from {address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_tcp_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
            except Exception as e:
                logger.error(f"TCP server error: {e}")
        
        self.tcp_thread = threading.Thread(target=tcp_server_worker, daemon=True)
        self.tcp_thread.start()
    
    def handle_tcp_client(self, client_socket, address):
        """Handle TCP client connections"""
        try:
            while True:
                # Receive message length (4 bytes)
                length_data = client_socket.recv(4)
                if not length_data:
                    break
                
                message_length = struct.unpack('!I', length_data)[0]
                
                # Receive message
                message_data = b''
                while len(message_data) < message_length:
                    chunk = client_socket.recv(message_length - len(message_data))
                    if not chunk:
                        break
                    message_data += chunk
                
                # Process message
                try:
                    message = json.loads(message_data.decode('utf-8'))
                    response = self.process_tcp_message(message)
                    
                    # Send response
                    response_data = json.dumps(response).encode('utf-8')
                    response_length = struct.pack('!I', len(response_data))
                    client_socket.send(response_length + response_data)
                    
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received via TCP")
                    
        except Exception as e:
            logger.error(f"TCP client error: {e}")
        finally:
            client_socket.close()
            logger.info(f"TCP connection closed: {address}")
    
    def process_tcp_message(self, message: Dict) -> Dict:
        """Process TCP message from robot"""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'register':
                success = self.robot_manager.register_robot(message.get('robot_info', {}))
                return {'success': success, 'message': 'Robot registered'}
            
            elif msg_type == 'analyze_image':
                # Handle image analysis
                robot_id = message.get('robot_id')
                image_data = base64.b64decode(message.get('image', ''))
                image = Image.open(io.BytesIO(image_data))
                prompt = message.get('prompt', 'Analyze this environment for robot navigation')
                
                response = self.vila_model.generate_response(prompt=prompt, image=image)
                commands = self.parse_navigation_response(response)
                
                return {
                    'success': True,
                    'analysis': response,
                    'commands': commands,
                    'timestamp': time.time()
                }
            
            elif msg_type == 'get_commands':
                robot_id = message.get('robot_id')
                commands = self.robot_manager.get_pending_commands(robot_id)
                return {
                    'success': True,
                    'commands': [asdict(cmd) for cmd in commands]
                }
            
            else:
                return {'success': False, 'error': 'Unknown message type'}
                
        except Exception as e:
            logger.error(f"Error processing TCP message: {e}")
            return {'success': False, 'error': str(e)}
    
    def parse_navigation_response(self, response):
        """Parse VILA response to extract robot commands"""
        response_lower = response.lower()
        commands = {
            'move_forward': 'forward' in response_lower and 'clear' in response_lower,
            'stop': 'stop' in response_lower or 'obstacle' in response_lower,
            'turn_left': 'left' in response_lower,
            'turn_right': 'right' in response_lower,
            'hazard_detected': any(word in response_lower for word in ['danger', 'hazard', 'unsafe', 'obstacle'])
        }
        return commands

# Global service instance
vila_service = RobotVILAService()

# ===== HTTP REST API ENDPOINTS =====

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': vila_service.model_loaded,
        'gpu_available': torch.cuda.is_available(),
        'registered_robots': len(vila_service.robot_manager.robots),
        'timestamp': time.time()
    })

@app.route('/robots', methods=['GET'])
def list_robots():
    """List all registered robots"""
    robots = vila_service.robot_manager.get_all_robots()
    return jsonify({
        'success': True,
        'robots': [asdict(robot) for robot in robots],
        'count': len(robots)
    })

@app.route('/robots/register', methods=['POST'])
def register_robot():
    """Register a new robot"""
    try:
        data = request.get_json()
        success = vila_service.robot_manager.register_robot(data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Robot registered successfully',
                'robot_id': data.get('robot_id')
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to register robot'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/robots/<robot_id>/status', methods=['GET', 'POST'])
def robot_status(robot_id):
    """Get or update robot status"""
    if request.method == 'GET':
        robot = vila_service.robot_manager.get_robot(robot_id)
        if robot:
            return jsonify({
                'success': True,
                'robot': asdict(robot)
            })
        else:
            return jsonify({'success': False, 'error': 'Robot not found'}), 404
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            success = vila_service.robot_manager.update_robot_status(robot_id, data)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Robot status updated'
                })
            else:
                return jsonify({'success': False, 'error': 'Robot not found'}), 404
                
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/robots/<robot_id>/analyze', methods=['POST'])
def analyze_for_robot(robot_id):
    """Analyze image for specific robot"""
    try:
        if not vila_service.model_loaded:
            return jsonify({'error': 'VILA model not loaded'}), 503
            
        data = request.get_json()
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Get robot info for context
        robot = vila_service.robot_manager.get_robot(robot_id)
        robot_context = ""
        if robot:
            capabilities = ", ".join(robot.capabilities)
            robot_context = f"Robot capabilities: {capabilities}. "
        
        # Get prompt with robot context
        prompt = data.get('prompt', 
            f"{robot_context}Analyze this image for robot navigation and provide specific actionable advice.")
        
        # Generate response
        response = vila_service.vila_model.generate_response(
            prompt=prompt,
            image=image
        )
        
        # Parse commands
        commands = vila_service.parse_navigation_response(response)
        
        # SAFETY: DISABLE auto-generate control commands to prevent autonomous movement
        # This was bypassing the GUI safety toggle and sending unauthorized movement commands
        if data.get('generate_commands', False):  # Changed default to False for safety
            logger.warning(f"🚫 VILA auto-command generation DISABLED for safety")
            logger.warning(f"   └── Robot {robot_id} autonomous movement blocked - use manual control only")
            # control_cmd = vila_service.generate_control_command(robot_id, commands, response)
            # if control_cmd:
            #     vila_service.robot_manager.send_command(robot_id, control_cmd)
        
        return jsonify({
            'success': True,
            'robot_id': robot_id,
            'analysis': response,
            'commands': commands,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error analyzing for robot {robot_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/robots/<robot_id>/commands', methods=['GET', 'POST'])
def robot_commands(robot_id):
    """Get pending commands or send new command to robot"""
    if request.method == 'GET':
        commands = vila_service.robot_manager.get_pending_commands(robot_id)
        return jsonify({
            'success': True,
            'robot_id': robot_id,
            'commands': [asdict(cmd) for cmd in commands],
            'count': len(commands)
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            command_type = data['command_type']
            
            # CRITICAL SAFETY CHECK: Block all movement commands except STOP
            # This protects against autonomous VILA commands when movement is disabled
            movement_commands = ['move', 'turn', 'forward', 'backward', 'left', 'right']
            if command_type in movement_commands:
                # Get movement parameters to check direction
                direction = data.get('parameters', {}).get('direction', command_type)
                
                # ONLY allow STOP commands - block all other movement
                if direction != 'stop':
                    logger.warning(f"🚫 SERVER SAFETY: Blocking {command_type} command '{direction}' to robot {robot_id}")
                    logger.warning(f"   └── All movement blocked except STOP - check GUI movement toggle")
                    return jsonify({
                        'success': False, 
                        'error': f'Movement command "{direction}" blocked by server safety system. Only STOP commands allowed when movement disabled.',
                        'safety_block': True
                    }), 403
            
            command = RobotCommand(
                robot_id=robot_id,
                command_type=command_type,
                parameters=data.get('parameters', {}),
                timestamp=datetime.now(),
                priority=data.get('priority', 1)
            )
            
            success = vila_service.robot_manager.send_command(robot_id, command)
            
            if success:
                logger.info(f"Command sent to robot {robot_id}: {data.get('parameters', {}).get('direction', command_type)}")
                return jsonify({
                    'success': True,
                    'message': 'Command sent successfully'
                })
            else:
                return jsonify({'success': False, 'error': 'Robot not found'}), 404
                
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

# ===== WEBSOCKET ENDPOINTS =====

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"WebSocket client connected")
    emit('status', {'message': 'Connected to VILA Robot Hub'})

@socketio.on('register_robot')
def handle_register_robot(data):
    """Handle robot registration via WebSocket"""
    try:
        success = vila_service.robot_manager.register_robot(data)
        if success:
            emit('registration_response', {
                'success': True,
                'robot_id': data.get('robot_id'),
                'message': 'Robot registered successfully'
            })
            
            # Broadcast to all clients
            socketio.emit('robot_registered', {
                'robot_id': data.get('robot_id'),
                'name': data.get('name', 'Unknown')
            })
        else:
            emit('registration_response', {
                'success': False,
                'error': 'Failed to register robot'
            })
    except Exception as e:
        emit('registration_response', {
            'success': False,
            'error': str(e)
        })

@socketio.on('analyze_image')
def handle_analyze_image(data):
    """Handle real-time image analysis via WebSocket"""
    try:
        if not vila_service.model_loaded:
            emit('analysis_result', {'error': 'VILA model not loaded'})
            return
        
        robot_id = data.get('robot_id')
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        prompt = data.get('prompt', 'Analyze this environment for robot navigation')
        
        # Generate response
        response = vila_service.vila_model.generate_response(prompt=prompt, image=image)
        commands = vila_service.parse_navigation_response(response)
        
        result = {
            'success': True,
            'robot_id': robot_id,
            'analysis': response,
            'commands': commands,
            'timestamp': time.time()
        }
        
        # Send to requesting client
        emit('analysis_result', result)
        
        # Broadcast to monitoring clients
        socketio.emit('robot_analysis', result, room='monitors')
        
    except Exception as e:
        emit('analysis_result', {'error': str(e)})

@socketio.on('join_monitors')
def handle_join_monitors():
    """Join monitoring room for robot updates"""
    from flask_socketio import join_room
    join_room('monitors')
    emit('status', {'message': 'Joined monitoring room'})

# ===== ENHANCED SERVICE METHODS =====

def add_service_methods():
    """Add enhanced methods to the service"""
    
    def generate_control_command(self, robot_id: str, parsed_commands: Dict, analysis: str) -> Optional[RobotCommand]:
        """Generate control command from analysis"""
        try:
            # Determine primary action
            if parsed_commands.get('hazard_detected') or parsed_commands.get('stop'):
                action = 'stop'
                params = {'reason': 'hazard_detected'}
            elif parsed_commands.get('move_forward'):
                action = 'move'
                params = {'direction': 'forward', 'speed': 0.3, 'duration': 2.0}
            elif parsed_commands.get('turn_left'):
                action = 'turn'
                params = {'direction': 'left', 'angle': 45, 'speed': 0.2}
            elif parsed_commands.get('turn_right'):
                action = 'turn'
                params = {'direction': 'right', 'angle': 45, 'speed': 0.2}
            else:
                action = 'stop'
                params = {'reason': 'unclear_instruction'}
            
            return RobotCommand(
                robot_id=robot_id,
                command_type=action,
                parameters=params,
                timestamp=datetime.now(),
                priority=2 if parsed_commands.get('hazard_detected') else 1
            )
            
        except Exception as e:
            logger.error(f"Error generating control command: {e}")
            return None
    
    # Add method to service instance
    vila_service.generate_control_command = generate_control_command.__get__(vila_service, RobotVILAService)

add_service_methods()

# ===== MAIN APPLICATION =====

def load_config():
    """Load configuration from file"""
    config = configparser.ConfigParser()
    config_file = 'robot_hub_config.ini'
    
    # Create default config if it doesn't exist
    if not Path(config_file).exists():
        config['DEFAULT'] = {
            'http_port': '5000',
            'tcp_port': '9999',
            'log_level': 'INFO',
            'max_robots': '10'
        }
        
        config['VILA'] = {
            'model_name': 'Efficient-Large-Model/VILA1.5-3b',
            'device': 'auto'
        }
        
        with open(config_file, 'w') as f:
            config.write(f)
        
        logger.info(f"Created default config file: {config_file}")
    
    config.read(config_file)
    return config

def main():
    """Main application entry point"""
    logger.info("🚀 Starting VILA Robot Vision Hub...")
    
    # Load configuration
    config = load_config()
    http_port = int(config.get('DEFAULT', 'http_port', fallback=5000))
    tcp_port = int(config.get('DEFAULT', 'tcp_port', fallback=9999))
    
    # Initialize VILA service
    logger.info("Initializing VILA service...")
    if vila_service.initialize():
        logger.info("✅ VILA Robot Hub ready!")
        logger.info(f"🌐 HTTP/WebSocket server starting on port {http_port}")
        logger.info(f"🔌 TCP server listening on port {tcp_port}")
        logger.info("📡 WebSocket enabled for real-time communication")
        
        # Start the server
        socketio.run(app, host='0.0.0.0', port=http_port, debug=False)
    else:
        logger.error("❌ Failed to initialize VILA model")
        return 1

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code or 0)
    except KeyboardInterrupt:
        logger.info("👋 Shutting down VILA Robot Hub...")
        sys.exit(0)