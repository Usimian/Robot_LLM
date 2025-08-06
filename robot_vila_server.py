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
    sensor_data: Dict[str, Any] = None  # Real-time sensor readings

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
                    connection_type=robot_info.get('connection_type', 'http'),
                    sensor_data=robot_info.get('sensor_data', {})
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
        """Send command to robot with safety validation"""
        try:
            # CRITICAL SAFETY: Block movement commands at the robot manager level
            movement_commands = ['move', 'turn', 'forward', 'backward', 'left', 'right']
            if command.command_type in movement_commands:
                logger.warning(f"🚫 ROBOT MANAGER SAFETY BLOCK: Movement command '{command.command_type}' rejected")
                logger.warning(f"   └── All movement commands are now blocked at robot manager level")
                logger.warning(f"   └── Only GUI with proper safety confirmation can send movement commands")
                return False
            
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
        self.model_enabled = False  # New: separate enabled state from loaded
        self.robot_manager = RobotManager()
        self.tcp_server = None
        self.tcp_thread = None
        
    def initialize(self):
        """Initialize VILA model and services (but don't auto-enable)"""
        logger.info("🚀 VILA Robot Hub starting...")
        # Don't auto-load model on initialization - wait for enable command
        logger.info("✅ VILA service ready (model not loaded - use GUI to enable)")
        
        # Start TCP server for direct robot communication
        self.start_tcp_server()
        return True
    
    def enable_vila_model(self):
        """Enable and load VILA model"""
        try:
            if not self.model_loaded:
                logger.info("🚀 Loading VILA model...")
                success = self.vila_model.load_model()
                if not success:
                    return False, "Failed to load VILA model"
                self.model_loaded = True
                logger.info("✅ VILA model loaded successfully")
            
            self.model_enabled = True
            logger.info("✅ VILA model enabled and ready")
            return True, "VILA model enabled successfully"
            
        except Exception as e:
            logger.error(f"❌ Error enabling VILA model: {e}")
            return False, str(e)
    
    def disable_vila_model(self):
        """Disable VILA model (keep loaded but mark as disabled)"""
        try:
            self.model_enabled = False
            logger.info("⏸️ VILA model disabled")
            return True, "VILA model disabled successfully"
            
        except Exception as e:
            logger.error(f"❌ Error disabling VILA model: {e}")
            return False, str(e)
    
    def get_vila_status(self):
        """Get current VILA model status"""
        # SYNC: Check the actual VILA model status, not just server flag
        actual_model_loaded = hasattr(self.vila_model, 'model_loaded') and self.vila_model.model_loaded
        
        # Update server flag to match actual model status
        if self.model_loaded != actual_model_loaded:
            logger.warning(f"🔄 VILA status sync: server={self.model_loaded} vs actual={actual_model_loaded}")
            self.model_loaded = actual_model_loaded
        
        if self.model_loaded and self.model_enabled:
            status = "Ready"
        elif self.model_loaded and not self.model_enabled:
            status = "Loaded but Disabled"
        else:
            status = "Not Loaded"
        
        return {
            'model_loaded': self.model_loaded,
            'enabled': self.model_enabled,
            'status': status,
            'device': self.vila_model.device if self.model_loaded else 'N/A'
        }
    
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
        
        # Enhanced forward movement detection with multiple patterns
        forward_keywords = [
            'move forward', 'go forward', 'proceed forward', 'continue forward',
            'move ahead', 'go ahead', 'proceed ahead', 'continue ahead',
            'move straight', 'go straight', 'continue straight',
            'advance', 'proceed', 'continue moving', 'move in', 'keep moving',
            'forward', 'ahead'  # Single word fallbacks
        ]
        
        # Check for forward movement indicators
        move_forward = any(keyword in response_lower for keyword in forward_keywords)
        
        # Safety override: Don't move forward if explicit hazards mentioned
        hazard_keywords = ['obstacle', 'blocked', 'danger', 'hazard', 'unsafe', 'collision', 'wall', 'barrier']
        has_hazard = any(keyword in response_lower for keyword in hazard_keywords)
        
        # Don't move forward if VILA explicitly says not to or mentions hazards
        explicit_stop = any(phrase in response_lower for phrase in ['should not', 'cannot', 'do not', 'avoid'])
        
        commands = {
            'move_forward': move_forward and not has_hazard and not explicit_stop,
            'stop': 'stop' in response_lower or has_hazard,
            'turn_left': 'left' in response_lower,
            'turn_right': 'right' in response_lower,
            'hazard_detected': has_hazard
        }
        
        return commands

# Global service instance
vila_service = RobotVILAService()

# ===== HTTP REST API ENDPOINTS =====

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    vila_status = vila_service.get_vila_status()
    return jsonify({
        'status': 'healthy',
        'model_loaded': vila_service.model_loaded,
        'model_enabled': vila_service.model_enabled,
        'vila_status': vila_status['status'],
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

@app.route('/robots/<robot_id>/sensors', methods=['POST'])
def update_robot_sensors(robot_id):
    """Update robot sensor data"""
    try:
        data = request.get_json()
        sensor_data = data.get('sensor_data', {})
        
        # Update robot sensor data
        robot = vila_service.robot_manager.get_robot(robot_id)
        if not robot:
            return jsonify({'success': False, 'error': 'Robot not found'}), 404
        
        # Log significant sensor updates
        if 'battery_voltage' in sensor_data:
            logger.info(f"🔋 Robot {robot_id} battery voltage: {sensor_data['battery_voltage']:.2f}V")
        
        # Update sensor data and last seen time
        updates = {
            'sensor_data': sensor_data,
            'last_seen': datetime.now()
        }
        
        success = vila_service.robot_manager.update_robot_status(robot_id, updates)
        
        if success:
            # Broadcast sensor update to monitoring clients
            socketio.emit('robot_sensors', {
                'robot_id': robot_id,
                'sensor_data': sensor_data,
                'timestamp': time.time()
            }, room='monitors')
            
            return jsonify({
                'success': True,
                'message': 'Sensor data updated'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to update sensors'}), 500
            
    except Exception as e:
        logger.error(f"Error updating sensors for robot {robot_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/robots/<robot_id>/analyze', methods=['POST'])
def analyze_for_robot(robot_id):
    """Analyze image for specific robot"""
    try:
        if not vila_service.model_loaded:
            return jsonify({'error': 'VILA model not loaded'}), 503
        
        if not vila_service.model_enabled:
            return jsonify({'error': 'VILA model is disabled - please enable it first'}), 503
            
        data = request.get_json()
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # AUTO-REGISTER: Register robot if not already registered (for GUI visibility)
        robot = vila_service.robot_manager.get_robot(robot_id)
        if not robot:
            logger.info(f"🤖 Auto-registering robot {robot_id} for GUI visibility")
            auto_robot_info = {
                'robot_id': robot_id,
                'name': f'Auto-Robot {robot_id}',
                'capabilities': ['navigation', 'camera', 'autonomous'],
                'connection_type': 'http_analyze',
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'heading': 0.0},
                'battery_level': 85.0  # Default for auto-registered robots
            }
            vila_service.robot_manager.register_robot(auto_robot_info)
            robot = vila_service.robot_manager.get_robot(robot_id)
            
            # Broadcast robot registration to GUI clients
            socketio.emit('robot_registered', {
                'robot_id': robot_id,
                'name': auto_robot_info['name'],
                'auto_registered': True
            })
        
        # Update robot's last seen time
        vila_service.robot_manager.update_robot_status(robot_id, {'last_seen': datetime.now()})
        
        # Get robot info for context
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
        
        # LOG ROBOT ACTIVITY: Broadcast to GUI for visibility
        # Create small thumbnail for GUI display (to save bandwidth)
        thumbnail = image.copy()
        thumbnail.thumbnail((120, 80), Image.Resampling.LANCZOS)
        thumbnail_buffer = io.BytesIO()
        thumbnail.save(thumbnail_buffer, format='PNG')
        thumbnail_b64 = base64.b64encode(thumbnail_buffer.getvalue()).decode('utf-8')
        
        activity_data = {
            'robot_id': robot_id,
            'activity_type': 'analysis_request',
            'vila_prompt': prompt,
            'vila_response': response,
            'parsed_commands': commands,
            'timestamp': time.time(),
            'image_size': len(image_data),
            'thumbnail': thumbnail_b64  # Include small thumbnail for GUI
        }
        
        # Broadcast robot activity to GUI monitoring clients
        socketio.emit('robot_activity', activity_data, room='monitors')
        logger.info(f"🤖 Robot {robot_id}: VILA analysis request processed (image: {len(image_data):,} bytes)")
        
        # ARCHITECTURE CHANGE: No autonomous robot movement allowed
        # ALL movement commands must go through GUI for proper control and logging
        
        # Notify GUI about VILA analysis completion for potential command generation
        if any(commands.values()):
            gui_notification = {
                'robot_id': robot_id,
                'analysis_complete': True,
                'vila_response': response,
                'parsed_commands': commands,
                'timestamp': time.time(),
                'requires_gui_processing': True
            }
            socketio.emit('vila_analysis_complete', gui_notification, room='monitors')
            logger.info(f"📝 VILA analysis complete for robot {robot_id} - awaiting GUI command processing")
        
        # QUEUE FOR GUI PROCESSING: Store VILA analysis for GUI command processing
        if any(commands.values()):
            # Store analysis result for GUI to pick up and process
            analysis_key = f"vila_analysis_{robot_id}_{int(time.time())}"
            if not hasattr(vila_service, 'pending_analyses'):
                vila_service.pending_analyses = {}
            
            vila_service.pending_analyses[analysis_key] = {
                'robot_id': robot_id,
                'vila_response': response,
                'parsed_commands': commands,
                'timestamp': time.time(),
                'processed': False
            }
            
            # Clean old entries (older than 2 minutes)
            current_time = time.time()
            keys_to_remove = [k for k, v in vila_service.pending_analyses.items() 
                             if current_time - v['timestamp'] > 120]
            for key in keys_to_remove:
                del vila_service.pending_analyses[key]
        
        # STRICT POLICY: Server never generates robot movement commands
        # Only GUI can send movement commands to ensure proper logging and safety
        logger.info(f"🔒 Robot {robot_id}: VILA analysis queued for GUI processing")
        
        return jsonify({
            'success': True,
            'robot_id': robot_id,
            'analysis': response,
            'commands': commands,
            'timestamp': time.time(),
            'gui_processing_required': any(commands.values())
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
            
            # CRITICAL SAFETY: Server-side movement validation
            # Even though GUI should control movement, server must validate for safety
            movement_commands = ['move', 'turn', 'forward', 'backward', 'left', 'right']
            if command_type in movement_commands:
                direction = data.get('parameters', {}).get('direction', command_type)
                
                # SERVER-SIDE SAFETY CHECK: Refuse movement commands if safety not confirmed
                # This prevents ANY external process from bypassing GUI safety
                safety_confirmed = data.get('safety_confirmed', False)
                gui_movement_enabled = data.get('gui_movement_enabled', False)
                
                if not safety_confirmed or not gui_movement_enabled:
                    logger.warning(f"🚫 SERVER SAFETY BLOCK: Movement command '{direction}' rejected - missing safety confirmation")
                    logger.warning(f"   └── safety_confirmed: {safety_confirmed}, gui_movement_enabled: {gui_movement_enabled}")
                    return jsonify({
                        'success': False, 
                        'error': f'Movement command rejected by server safety system',
                        'details': 'GUI must confirm movement is enabled and safe'
                    }), 403
                
                logger.info(f"✅ SAFE COMMAND: Accepting {command_type} command '{direction}' to robot {robot_id}")
                logger.info(f"   └── Command from GUI with safety confirmation - server allows movement")
            
            command = RobotCommand(
                robot_id=robot_id,
                command_type=command_type,
                parameters=data.get('parameters', {}),
                timestamp=datetime.now(),
                priority=data.get('priority', 1)
            )
            
            # ENSURE ROBOT EXISTS: Double-check robot is registered before sending command
            robot = vila_service.robot_manager.get_robot(robot_id)
            if not robot:
                logger.warning(f"❌ Robot {robot_id} not found in registry - cannot send command")
                logger.info(f"📝 Available robots: {list(vila_service.robot_manager.robots.keys())}")
                return jsonify({'success': False, 'error': f'Robot {robot_id} not found in registry'}), 404
            
            success = vila_service.robot_manager.send_command(robot_id, command)
            
            if success:
                logger.info(f"✅ Command sent to robot {robot_id}: {data.get('parameters', {}).get('direction', command_type)}")
                return jsonify({
                    'success': True,
                    'message': 'Command sent successfully'
                })
            else:
                logger.error(f"❌ Failed to send command to robot {robot_id} - robot exists but command failed")
                return jsonify({'success': False, 'error': f'Failed to send command to robot {robot_id}'}), 500
                
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

# ===== VILA MODEL CONTROL ENDPOINTS =====

@app.route('/vila/status', methods=['GET'])
def vila_status():
    """Get VILA model status"""
    try:
        status = vila_service.get_vila_status()
        return jsonify({
            'success': True,
            **status
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/vila/diagnostic', methods=['GET'])
def vila_diagnostic():
    """Get detailed VILA diagnostic information"""
    try:
        import torch
        diagnostic = {
            'server_model_loaded': vila_service.model_loaded,
            'server_model_enabled': vila_service.model_enabled,
            'actual_model_loaded': hasattr(vila_service.vila_model, 'model_loaded') and vila_service.vila_model.model_loaded,
            'model_loading': hasattr(vila_service.vila_model, 'loading') and vila_service.vila_model.loading,
            'model_device': vila_service.vila_model.device if hasattr(vila_service.vila_model, 'device') else 'Unknown',
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0,
            'model_attributes': {
                'has_model': hasattr(vila_service.vila_model, 'model') and vila_service.vila_model.model is not None,
                'has_tokenizer': hasattr(vila_service.vila_model, 'tokenizer') and vila_service.vila_model.tokenizer is not None,
                'has_processor': hasattr(vila_service.vila_model, 'image_processor') and vila_service.vila_model.image_processor is not None,
            },
            'status_sync': vila_service.model_loaded == (hasattr(vila_service.vila_model, 'model_loaded') and vila_service.vila_model.model_loaded)
        }
        
        return jsonify({
            'success': True,
            'diagnostic': diagnostic
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/vila/pending_analyses', methods=['GET'])
def get_pending_analyses():
    """Get VILA analyses waiting for GUI command processing"""
    try:
        if not hasattr(vila_service, 'pending_analyses'):
            vila_service.pending_analyses = {}
        
        # Get unprocessed analyses
        pending = {k: v for k, v in vila_service.pending_analyses.items() if not v.get('processed', False)}
        
        return jsonify({
            'success': True,
            'pending_count': len(pending),
            'analyses': pending
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/vila/mark_processed', methods=['POST'])
def mark_analysis_processed():
    """Mark a VILA analysis as processed by GUI"""
    try:
        data = request.get_json()
        analysis_key = data.get('analysis_key')
        
        if not hasattr(vila_service, 'pending_analyses'):
            vila_service.pending_analyses = {}
        
        if analysis_key in vila_service.pending_analyses:
            vila_service.pending_analyses[analysis_key]['processed'] = True
            logger.info(f"✅ VILA analysis {analysis_key} marked as processed by GUI")
            return jsonify({'success': True, 'message': 'Analysis marked as processed'})
        else:
            return jsonify({'success': False, 'error': 'Analysis key not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/vila/enable', methods=['POST'])
def enable_vila():
    """Enable/Load VILA model"""
    try:
        success, message = vila_service.enable_vila_model()
        status_code = 200 if success else 500
        
        return jsonify({
            'success': success,
            'message': message,
            'status': vila_service.get_vila_status()
        }), status_code
        
    except Exception as e:
        logger.error(f"Error enabling VILA: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to enable VILA model'
        }), 500

@app.route('/vila/disable', methods=['POST'])
def disable_vila():
    """Disable VILA model"""
    try:
        success, message = vila_service.disable_vila_model()
        status_code = 200 if success else 500
        
        return jsonify({
            'success': success,
            'message': message,
            'status': vila_service.get_vila_status()
        }), status_code
        
    except Exception as e:
        logger.error(f"Error disabling VILA: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to disable VILA model'
        }), 500

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
        
        if not vila_service.model_enabled:
            emit('analysis_result', {'error': 'VILA model is disabled - please enable it first'})
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