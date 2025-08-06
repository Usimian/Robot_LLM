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
import requests
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
    
    def execute_command(self, robot_id: str, command: RobotCommand, safety_confirmed: bool = False, gui_movement_enabled: bool = False, source: str = "unknown") -> tuple[bool, str]:
        """
        🛡️ SINGLE COMMAND GATEWAY - ALL robot commands MUST go through this method
        
        This is the ONLY method that can send commands to robots. All other methods
        must call this one to ensure consistent safety validation.
        
        Args:
            robot_id: Target robot ID
            command: Command to execute
            safety_confirmed: True if GUI has confirmed safety
            gui_movement_enabled: True if GUI movement is enabled
            source: Source of the command (for logging)
            
        Returns:
            (success: bool, message: str)
        """
        try:
            # 🚨 CRITICAL SAFETY VALIDATION - This is the ONLY place commands are validated
            movement_commands = ['move', 'turn', 'forward', 'backward', 'left', 'right']
            
            if command.command_type in movement_commands:
                # Movement command - requires explicit safety confirmation
                if not safety_confirmed:
                    error_msg = f"Movement command '{command.command_type}' rejected - no safety confirmation"
                    logger.warning(f"🚫 COMMAND GATEWAY BLOCK: {error_msg}")
                    logger.warning(f"   └── Source: {source}")
                    logger.warning(f"   └── Robot: {robot_id}")
                    return False, error_msg
                    
                if not gui_movement_enabled:
                    error_msg = f"Movement command '{command.command_type}' rejected - GUI movement disabled"
                    logger.warning(f"🚫 COMMAND GATEWAY BLOCK: {error_msg}")
                    logger.warning(f"   └── Source: {source}")
                    logger.warning(f"   └── Robot: {robot_id}")
                    return False, error_msg
                    
                logger.info(f"✅ SAFE MOVEMENT: Executing {command.command_type} command for robot {robot_id}")
                logger.info(f"   └── Source: {source} (with safety confirmation)")
            else:
                # Non-movement command - always allowed
                logger.info(f"✅ NON-MOVEMENT: Executing {command.command_type} command for robot {robot_id}")
                logger.info(f"   └── Source: {source}")
            
            # Check if robot exists 
            if robot_id not in self.command_queues:
                error_msg = f"Robot {robot_id} not found in command queues"
                logger.error(f"❌ COMMAND GATEWAY ERROR: {error_msg}")
                return False, error_msg
                
            # 🎯 SINGLE POINT OF ROBOT COMMUNICATION
            # This is THE ONLY place in the entire system that communicates with robots
            success = self._send_command_to_physical_robot(robot_id, command)
            
            if success:
                success_msg = f"Command {command.command_type} sent directly to robot {robot_id}"
                logger.info(f"📤 DIRECT COMMAND SENT: {success_msg}")
                return True, success_msg
            else:
                error_msg = f"Failed to send command {command.command_type} to robot {robot_id}"
                logger.error(f"❌ DIRECT COMMAND FAILED: {error_msg}")
                return False, error_msg
            
        except Exception as e:
            error_msg = f"Command gateway error: {str(e)}"
            logger.error(f"💥 COMMAND GATEWAY EXCEPTION: {error_msg}")
            logger.error(f"   └── Robot: {robot_id}, Command: {command.command_type}, Source: {source}")
            return False, error_msg

    def _send_command_to_physical_robot(self, robot_id: str, command: RobotCommand) -> bool:
        """
        🎯 THE SINGLE POINT where ALL robot communication happens
        
        This is the ONLY method in the entire system that actually sends data to robots.
        Every command from GUI, API, ROS, WebSocket - EVERYTHING goes through here.
        """
        try:
            robot = self.get_robot(robot_id)
            if not robot:
                logger.error(f"❌ Robot {robot_id} not found for physical communication")
                return False
                
            # Convert our command format to robot's expected format
            robot_command = {
                'type': command.command_type,
                'parameters': command.parameters,
                'timestamp': command.timestamp.isoformat(),
                'priority': command.priority
            }
            
            # 🎯 SINGLE ROBOT SYSTEM: Queue-based communication
            # Robot polls for commands - we don't send directly to robot
            
            # Add command to robot's queue for polling
            if robot_id not in self.command_queues:
                logger.error(f"❌ Robot {robot_id} not found in command queues")
                return False
                
            self.command_queues[robot_id].put(command)
            logger.info(f"✅ COMMAND QUEUED: {command.command_type} for robot {robot_id}")
            logger.info(f"   └── Robot will receive this command when it polls GET /commands")
            logger.info(f"   └── Queue now has {self.command_queues[robot_id].qsize()} pending commands")
            
            return True
                
        except Exception as e:
            logger.error(f"💥 PHYSICAL ROBOT: Communication error with {robot_id}: {e}")
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
        
    def initialize(self):
        """Initialize VILA model and services (but don't auto-enable)"""
        logger.info("🚀 VILA Robot Hub starting...")
        # Don't auto-load model on initialization - wait for enable command
        logger.info("✅ VILA service ready (model not loaded - use GUI to enable)")
        
        # Single robot system uses HTTP-only communication
        logger.info("🎯 Single robot system initialized - HTTP/WebSocket communication only")
        
        # Pre-register the single robot from config
        self._register_configured_robot()
        return True
    
    def _register_configured_robot(self):
        """Pre-register the single robot from configuration"""
        try:
            config = configparser.ConfigParser()
            config.read('robot_hub_config.ini')
            
            robot_info = {
                'robot_id': config.get('ROBOT', 'robot_id', fallback='yahboomcar_x3_01'),
                'name': config.get('ROBOT', 'robot_name', fallback='YahBoom Car X3'),
                'capabilities': ['navigation', 'camera', 'autonomous'],
                'connection_type': 'http_polling',
                'position': {
                    'x': 0.0, 'y': 0.0, 'z': 0.0, 'heading': 0.0,
                    'ip': config.get('ROBOT', 'robot_ip', fallback='192.168.1.166')
                },
                'battery_level': 100.0  # Will be updated when robot sends real data
            }
            
            success = self.robot_manager.register_robot(robot_info)
            if success:
                logger.info(f"✅ Pre-registered robot: {robot_info['robot_id']} @ {robot_info['position']['ip']}")
            else:
                logger.error("❌ Failed to pre-register configured robot")
                
        except Exception as e:
            logger.error(f"❌ Error pre-registering robot: {e}")
    
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

@app.route('/robot', methods=['GET'])
def get_robot():
    """Get the single robot status"""
    config = configparser.ConfigParser()
    config.read('robot_hub_config.ini')
    robot_id = config.get('ROBOT', 'robot_id', fallback='yahboomcar_x3_01')
    
    robot = vila_service.robot_manager.get_robot(robot_id)
    if robot:
        return jsonify({
            'success': True,
            'robot': asdict(robot)
        })
    else:
        return jsonify({'success': False, 'error': 'Robot not found'}), 404

# Robot registration endpoint removed - single robot system uses hardcoded configuration

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

@app.route('/robots/<robot_id>/sensors', methods=['GET', 'POST'])
def robot_sensors(robot_id):
    """Get or update robot sensor data"""
    if request.method == 'GET':
        # CLIENT-INITIATED: Get current sensor data from robot
        try:
            robot = vila_service.robot_manager.get_robot(robot_id)
            if not robot:
                return jsonify({'success': False, 'error': 'Robot not found'}), 404
            
            # For single robot system, request sensor data directly from robot
            config = configparser.ConfigParser()
            config.read('robot_hub_config.ini')
            robot_ip = config.get('ROBOT', 'robot_ip', fallback='192.168.1.166')
            robot_port = config.get('ROBOT', 'robot_port', fallback='8080')
            
            try:
                # Request sensor data from robot
                response = requests.get(
                    f"http://{robot_ip}:{robot_port}/sensors",
                    timeout=3
                )
                
                if response.status_code == 200:
                    sensor_data = response.json()
                    
                    # Update our stored sensor data
                    updates = {
                        'sensor_data': sensor_data,
                        'last_seen': datetime.now()
                    }
                    vila_service.robot_manager.update_robot_status(robot_id, updates)
                    
                    return jsonify({
                        'success': True,
                        'robot_id': robot_id,
                        'sensor_data': sensor_data,
                        'timestamp': time.time()
                    })
                else:
                    # Fallback to stored data if robot doesn't respond
                    stored_data = robot.sensor_data or {}
                    return jsonify({
                        'success': True,
                        'robot_id': robot_id,
                        'sensor_data': stored_data,
                        'timestamp': time.time(),
                        'note': 'Using cached data - robot not responding'
                    })
                    
            except requests.exceptions.RequestException:
                # Robot not responding, return stored data
                stored_data = robot.sensor_data or {}
                return jsonify({
                    'success': True,
                    'robot_id': robot_id,
                    'sensor_data': stored_data,
                    'timestamp': time.time(),
                    'note': 'Using cached data - robot not reachable'
                })
                
        except Exception as e:
            logger.error(f"Error getting sensors for robot {robot_id}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    elif request.method == 'POST':
        # ROBOT-INITIATED: Robot reporting sensor data (legacy support)
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
                return jsonify({
                    'success': True,
                    'message': 'Sensor data updated'
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to update sensors'}), 500
                
        except Exception as e:
            logger.error(f"Error updating sensors for robot {robot_id}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/robots/<robot_id>/image', methods=['GET'])
def get_robot_image(robot_id):
    """Get current image from robot"""
    try:
        robot = vila_service.robot_manager.get_robot(robot_id)
        if not robot:
            return jsonify({'success': False, 'error': 'Robot not found'}), 404
        
        # For single robot system, request image directly from robot
        config = configparser.ConfigParser()
        config.read('robot_hub_config.ini')
        robot_ip = config.get('ROBOT', 'robot_ip', fallback='192.168.1.166')
        robot_port = config.get('ROBOT', 'robot_port', fallback='8080')
        
        try:
            # Request image from robot
            response = requests.get(
                f"http://{robot_ip}:{robot_port}/image",
                timeout=5
            )
            
            if response.status_code == 200:
                image_data = response.json()
                return jsonify({
                    'success': True,
                    'robot_id': robot_id,
                    'image': image_data.get('image'),
                    'format': image_data.get('format', 'JPEG'),
                    'width': image_data.get('width', 640),
                    'height': image_data.get('height', 480),
                    'timestamp': time.time()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Robot image request failed: HTTP {response.status_code}'
                }), 502
                
        except requests.exceptions.RequestException as e:
            return jsonify({
                'success': False,
                'error': f'Robot not reachable: {str(e)}'
            }), 503
            
    except Exception as e:
        logger.error(f"Error getting image from robot {robot_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/robots/<robot_id>/analyze', methods=['POST', 'GET'])
def analyze_for_robot(robot_id):
    """Analyze image for specific robot - supports both push and pull modes"""
    try:
        if not vila_service.model_loaded:
            return jsonify({'error': 'VILA model not loaded'}), 503
        
        if not vila_service.model_enabled:
            return jsonify({'error': 'VILA model is disabled - please enable it first'}), 503
        
        if request.method == 'POST':
            # LEGACY: Robot pushing image data (still supported)
            data = request.get_json()
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            prompt = data.get('prompt', 'Analyze this environment for robot navigation')
            
        elif request.method == 'GET':
            # NEW: Client-initiated analysis - request image from robot first
            try:
                # Get image from robot
                image_response = requests.get(
                    f"{request.url_root}robots/{robot_id}/image",
                    timeout=5
                )
                
                if image_response.status_code != 200:
                    return jsonify({'error': 'Failed to get image from robot'}), 502
                
                image_json = image_response.json()
                if not image_json.get('success'):
                    return jsonify({'error': 'Robot image request failed'}), 502
                
                # Decode the image
                image_data = base64.b64decode(image_json['image'])
                image = Image.open(io.BytesIO(image_data))
                prompt = request.args.get('prompt', 'Analyze this environment for robot navigation')
                
            except Exception as e:
                return jsonify({'error': f'Failed to get image from robot: {str(e)}'}), 502
        
        # Get pre-registered robot (registered at startup)
        robot = vila_service.robot_manager.get_robot(robot_id)
        if not robot:
            logger.error(f"❌ Robot {robot_id} not found - should be pre-registered at startup")
            return jsonify({'error': f'Robot {robot_id} not configured'}), 404
        
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
        
        # 🛡️ SECURITY: Never send VILA responses or commands to robots
        # Robots should only receive explicit movement commands via the command queue
        return jsonify({
            'success': True,
            'robot_id': robot_id,
            'message': 'Analysis complete - commands will be sent separately if approved',
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
            
            # Extract safety parameters from the request
            safety_confirmed = data.get('safety_confirmed', False)
            gui_movement_enabled = data.get('gui_movement_enabled', False)
            
            # Create the command object
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
            
            # 🛡️ USE SINGLE COMMAND GATEWAY - All safety validation happens here
            success, message = vila_service.robot_manager.execute_command(
                robot_id=robot_id,
                command=command,
                safety_confirmed=safety_confirmed,
                gui_movement_enabled=gui_movement_enabled,
                source="HTTP_API"
            )
            
            if success:
                logger.info(f"✅ HTTP API: Command executed successfully - {message}")
                return jsonify({
                    'success': True,
                    'message': message
                })
            else:
                logger.error(f"❌ HTTP API: Command rejected - {message}")
                # Return 403 for safety blocks, 500 for other errors
                status_code = 403 if "rejected" in message else 500
                return jsonify({
                    'success': False, 
                    'error': message
                }), status_code
                
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

# WebSocket robot registration removed - single robot system uses hardcoded configuration

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
        
        # 🛡️ SECURITY: Never send VILA responses or commands to robots
        # Robots should only receive explicit movement commands via the command queue
        result = {
            'success': True,
            'robot_id': robot_id,
            'message': 'Analysis complete - commands will be sent separately if approved',
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
    
    # Initialize VILA service
    logger.info("Initializing VILA service...")
    if vila_service.initialize():
        logger.info("✅ VILA Robot Hub ready!")
        logger.info(f"🌐 HTTP/WebSocket server starting on port {http_port}")
        logger.info("📡 WebSocket enabled for real-time communication")
        logger.info("🎯 SINGLE ROBOT SYSTEM: Streamlined HTTP/WebSocket communication")
        logger.info("🤖 Robot: yahboomcar_x3_01 @ 192.168.1.166 (command polling pattern)")
        
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