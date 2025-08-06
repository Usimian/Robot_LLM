#!/usr/bin/env python3
"""
Unified Robot Controller
High-efficiency single-process robot control system with integrated VILA
Eliminates duplicate model loading and communication overhead
"""

import asyncio
import json
import logging
import base64
import io
import time
import threading
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import configparser

# Core libraries
import numpy as np
from PIL import Image
import cv2
import torch

# Web interface
from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO, emit
import requests

# Add VILA paths
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VILA'))
from main_vila import VILAModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_robot_controller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('UnifiedRobotController')

@dataclass
class RobotStatus:
    """Comprehensive robot status"""
    robot_id: str
    name: str
    last_seen: datetime
    position: Dict[str, float]  # x, y, z, heading
    battery_level: float
    status: str  # active, idle, error, offline
    capabilities: List[str]
    connection_type: str
    sensor_data: Dict[str, Any] = None
    last_command: Optional[str] = None
    command_history: List[str] = None

    def __post_init__(self):
        if self.command_history is None:
            self.command_history = []

@dataclass
class SafetyStatus:
    """System-wide safety status"""
    movement_enabled: bool = False
    autonomous_mode: bool = False
    emergency_stop: bool = False
    last_safety_check: datetime = None
    safety_violations: List[str] = None

    def __post_init__(self):
        if self.safety_violations is None:
            self.safety_violations = []
        if self.last_safety_check is None:
            self.last_safety_check = datetime.now()

class UnifiedSafetyController:
    """Centralized safety system - single source of truth"""
    
    def __init__(self):
        self.status = SafetyStatus()
        self.lock = asyncio.Lock()
        
    async def check_movement_allowed(self, robot_id: str = None) -> tuple[bool, str]:
        """
        Single source of truth for movement permissions
        Returns: (allowed, reason)
        """
        async with self.lock:
            self.status.last_safety_check = datetime.now()
            
            if self.status.emergency_stop:
                return False, "Emergency stop active"
                
            if not self.status.movement_enabled:
                return False, "Movement globally disabled"
                
            # Robot-specific checks could go here
            if robot_id:
                # Add robot-specific safety checks
                pass
                
            return True, "Movement allowed"
    
    async def enable_movement(self) -> bool:
        """Enable movement system-wide"""
        async with self.lock:
            self.status.movement_enabled = True
            logger.info("🟢 Movement enabled system-wide")
            return True
    
    async def disable_movement(self) -> bool:
        """Disable movement system-wide"""
        async with self.lock:
            self.status.movement_enabled = False
            logger.info("🔴 Movement disabled system-wide")
            return True
    
    async def emergency_stop(self) -> bool:
        """Activate emergency stop"""
        async with self.lock:
            self.status.emergency_stop = True
            self.status.movement_enabled = False
            logger.critical("🚨 EMERGENCY STOP ACTIVATED")
            return True
    
    async def clear_emergency_stop(self) -> bool:
        """Clear emergency stop (requires manual re-enable of movement)"""
        async with self.lock:
            self.status.emergency_stop = False
            logger.info("✅ Emergency stop cleared (movement still disabled)")
            return True

class RobotManager:
    """Efficient robot management with direct communication"""
    
    def __init__(self, safety_controller: UnifiedSafetyController):
        self.robots: Dict[str, RobotStatus] = {}
        self.safety = safety_controller
        self.lock = asyncio.Lock()
        
    async def register_robot(self, robot_info: Dict) -> bool:
        """Register robot with validation"""
        try:
            robot_id = robot_info['robot_id']
            async with self.lock:
                self.robots[robot_id] = RobotStatus(
                    robot_id=robot_id,
                    name=robot_info.get('name', f'Robot_{robot_id}'),
                    last_seen=datetime.now(),
                    position=robot_info.get('position', {'x': 0, 'y': 0, 'z': 0, 'heading': 0}),
                    battery_level=robot_info.get('battery_level', 0.0),
                    status='active',
                    capabilities=robot_info.get('capabilities', ['navigation']),
                    connection_type=robot_info.get('connection_type', 'http'),
                    sensor_data=robot_info.get('sensor_data', {})
                )
            
            logger.info(f"✅ Robot {robot_id} registered: {robot_info.get('name', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register robot: {e}")
            return False
    
    async def update_robot_sensor_data(self, robot_id: str, sensor_data: Dict[str, Any]) -> bool:
        """Update robot sensor data efficiently"""
        async with self.lock:
            if robot_id in self.robots:
                self.robots[robot_id].sensor_data = sensor_data
                self.robots[robot_id].last_seen = datetime.now()
                return True
            return False
    
    async def send_command_to_robot(self, robot_id: str, command: Dict[str, Any]) -> tuple[bool, str]:
        """
        🚫 DISABLED: This unified controller bypasses the single command gateway
        
        This method is disabled to enforce single command gateway architecture.
        All robot commands must go through robot_vila_server.py for proper safety validation.
        """
        logger.error("🚫 ARCHITECTURAL VIOLATION: Unified controller attempted direct robot communication")
        logger.error(f"   └── Robot: {robot_id}, Command: {command}")
        logger.error("   └── This bypasses the single command gateway in robot_vila_server.py")
        logger.error("   └── USE ONLY robot_vila_server.py for ALL robot communication")
        
        return False, "DISABLED: Use robot_vila_server.py single command gateway instead"

class VILAProcessor:
    """Single VILA instance for all processing"""
    
    def __init__(self):
        self.vila_model = VILAModel()
        self.model_loaded = False
        self.processing_queue = asyncio.Queue()
        self.results_callbacks: Dict[str, Callable] = {}
        
    async def initialize(self) -> bool:
        """Initialize VILA model once"""
        try:
            logger.info("🚀 Loading single VILA model instance...")
            success = await asyncio.get_event_loop().run_in_executor(
                None, self.vila_model.load_model
            )
            if success:
                self.model_loaded = True
                logger.info("✅ VILA model loaded successfully (single instance)")
                # Start processing worker
                asyncio.create_task(self._process_queue())
                return True
            else:
                logger.error("❌ Failed to load VILA model")
                return False
        except Exception as e:
            logger.error(f"❌ Error initializing VILA: {e}")
            return False
    
    async def analyze_image(self, image_data: bytes, prompt: str, callback_id: str = None) -> Optional[str]:
        """Queue image for analysis"""
        if not self.model_loaded:
            return None
            
        # Add to processing queue
        await self.processing_queue.put({
            'image_data': image_data,
            'prompt': prompt,
            'callback_id': callback_id,
            'timestamp': datetime.now()
        })
        
        return "queued"
    
    async def _process_queue(self):
        """Process VILA requests sequentially (avoid model conflicts)"""
        while True:
            try:
                task = await self.processing_queue.get()
                
                # Process image
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._analyze_image_sync, task['image_data'], task['prompt']
                )
                
                # Call callback if provided
                if task['callback_id'] and task['callback_id'] in self.results_callbacks:
                    await self.results_callbacks[task['callback_id']](result, task)
                
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Error processing VILA queue: {e}")
                await asyncio.sleep(1)
    
    def _analyze_image_sync(self, image_data: bytes, prompt: str) -> str:
        """Synchronous VILA processing"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Use VILA model
            result = self.vila_model.analyze_image(image, prompt)
            return result
            
        except Exception as e:
            logger.error(f"❌ VILA processing error: {e}")
            return f"Error: {str(e)}"

class UnifiedRobotController:
    """Main unified controller - single process, maximum efficiency"""
    
    def __init__(self, config_file: str = "robot_hub_config.ini"):
        self.config = self._load_config(config_file)
        
        # Core components
        self.safety = UnifiedSafetyController()
        self.robot_manager = RobotManager(self.safety)
        self.vila_processor = VILAProcessor()
        
        # Web interface
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'unified_robot_controller'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes and events
        self._setup_routes()
        self._setup_socketio_events()
        
        # State
        self.running = False
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration"""
        config = configparser.ConfigParser()
        if Path(config_file).exists():
            config.read(config_file)
            return {
                'http_port': config.getint('server', 'http_port', fallback=5000),
                'host': config.get('server', 'host', fallback='0.0.0.0')
            }
        else:
            return {'http_port': 5000, 'host': '0.0.0.0'}
    
    def _setup_routes(self):
        """Setup HTTP API routes"""
        
        @self.app.route('/api/robots/register', methods=['POST'])
        async def register_robot():
            data = request.get_json()
            success = await self.robot_manager.register_robot(data)
            return jsonify({'success': success})
        
        @self.app.route('/api/robots/<robot_id>/sensors', methods=['POST'])
        async def update_sensors(robot_id):
            data = request.get_json()
            success = await self.robot_manager.update_robot_sensor_data(robot_id, data)
            
            # Broadcast to web interface
            self.socketio.emit('robot_sensors', {
                'robot_id': robot_id,
                'sensor_data': data,
                'timestamp': datetime.now().isoformat()
            })
            
            return jsonify({'success': success})
        
        @self.app.route('/api/robots/<robot_id>/command', methods=['POST'])
        async def send_command(robot_id):
            command = request.get_json()
            success, message = await self.robot_manager.send_command_to_robot(robot_id, command)
            return jsonify({'success': success, 'message': message})
        
        @self.app.route('/api/safety/status', methods=['GET'])
        def get_safety_status():
            return jsonify(asdict(self.safety.status))
        
        @self.app.route('/api/safety/enable_movement', methods=['POST'])
        async def enable_movement():
            success = await self.safety.enable_movement()
            self.socketio.emit('safety_update', asdict(self.safety.status))
            return jsonify({'success': success})
        
        @self.app.route('/api/safety/disable_movement', methods=['POST'])
        async def disable_movement():
            success = await self.safety.disable_movement()
            self.socketio.emit('safety_update', asdict(self.safety.status))
            return jsonify({'success': success})
        
        @self.app.route('/api/safety/emergency_stop', methods=['POST'])
        async def emergency_stop():
            success = await self.safety.emergency_stop()
            self.socketio.emit('safety_update', asdict(self.safety.status))
            return jsonify({'success': success})
        
        @self.app.route('/api/vila/analyze', methods=['POST'])
        async def analyze_image():
            data = request.get_json()
            image_data = base64.b64decode(data['image'])
            prompt = data.get('prompt', 'Describe what you see and suggest robot actions.')
            
            result = await self.vila_processor.analyze_image(image_data, prompt)
            return jsonify({'result': result})
        
        @self.app.route('/api/robots', methods=['GET'])
        def get_robots():
            robots_dict = {}
            for robot_id, robot in self.robot_manager.robots.items():
                robots_dict[robot_id] = asdict(robot)
                # Convert datetime to string for JSON serialization
                robots_dict[robot_id]['last_seen'] = robot.last_seen.isoformat()
            return jsonify(robots_dict)
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(DASHBOARD_HTML)
    
    def _setup_socketio_events(self):
        """Setup WebSocket events for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Web client connected")
            emit('status', {'connected': True})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("Web client disconnected")
        
        @self.socketio.on('get_robots')
        def handle_get_robots():
            robots_dict = {}
            for robot_id, robot in self.robot_manager.robots.items():
                robots_dict[robot_id] = asdict(robot)
                robots_dict[robot_id]['last_seen'] = robot.last_seen.isoformat()
            emit('robots_update', robots_dict)
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("🚀 Starting Unified Robot Controller...")
        
        # Initialize VILA processor
        vila_success = await self.vila_processor.initialize()
        if not vila_success:
            logger.error("❌ Failed to initialize VILA processor")
            return False
        
        logger.info("✅ Unified Robot Controller initialized")
        return True
    
    def run(self):
        """Run the unified controller"""
        self.running = True
        
        # Run Flask-SocketIO with asyncio support
        logger.info(f"🌐 Starting web interface on http://{self.config['host']}:{self.config['http_port']}")
        self.socketio.run(
            self.app,
            host=self.config['host'],
            port=self.config['http_port'],
            debug=False,
            use_reloader=False
        )

# Lightweight web dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Unified Robot Controller</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-good { color: green; font-weight: bold; }
        .status-bad { color: red; font-weight: bold; }
        .status-warning { color: orange; font-weight: bold; }
        button { padding: 10px 20px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-warning { background: #ffc107; color: black; }
        .robot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .sensor-data { font-family: monospace; background: #f8f9fa; padding: 10px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Unified Robot Controller</h1>
        
        <div class="card">
            <h2>System Status</h2>
            <p>VILA Model: <span id="vila-status" class="status-good">Ready</span></p>
            <p>Movement: <span id="movement-status">Unknown</span></p>
            <p>Connected Robots: <span id="robot-count">0</span></p>
            
            <div style="margin-top: 15px;">
                <button class="btn-success" onclick="enableMovement()">Enable Movement</button>
                <button class="btn-warning" onclick="disableMovement()">Disable Movement</button>
                <button class="btn-danger" onclick="emergencyStop()">EMERGENCY STOP</button>
            </div>
        </div>
        
        <div class="card">
            <h2>Connected Robots</h2>
            <div id="robots-container" class="robot-grid">
                <p>No robots connected</p>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to server');
            socket.emit('get_robots');
        });
        
        socket.on('robots_update', function(robots) {
            updateRobotsDisplay(robots);
        });
        
        socket.on('robot_sensors', function(data) {
            updateRobotSensors(data.robot_id, data.sensor_data);
        });
        
        socket.on('safety_update', function(safety_status) {
            updateSafetyStatus(safety_status);
        });
        
        function updateRobotsDisplay(robots) {
            const container = document.getElementById('robots-container');
            const robotCount = Object.keys(robots).length;
            document.getElementById('robot-count').textContent = robotCount;
            
            if (robotCount === 0) {
                container.innerHTML = '<p>No robots connected</p>';
                return;
            }
            
            let html = '';
            for (const [robotId, robot] of Object.entries(robots)) {
                html += `
                    <div class="card" id="robot-${robotId}">
                        <h3>${robot.name} (${robotId})</h3>
                        <p><strong>Status:</strong> <span class="status-${robot.status === 'active' ? 'good' : 'warning'}">${robot.status}</span></p>
                        <p><strong>Battery:</strong> ${robot.battery_level}%</p>
                        <p><strong>Last Seen:</strong> ${new Date(robot.last_seen).toLocaleString()}</p>
                        <p><strong>Last Command:</strong> ${robot.last_command || 'None'}</p>
                        <div class="sensor-data" id="sensors-${robotId}">
                            <strong>Sensors:</strong><br>
                            ${formatSensorData(robot.sensor_data)}
                        </div>
                    </div>
                `;
            }
            container.innerHTML = html;
        }
        
        function updateRobotSensors(robotId, sensorData) {
            const sensorsDiv = document.getElementById(`sensors-${robotId}`);
            if (sensorsDiv) {
                sensorsDiv.innerHTML = `<strong>Sensors:</strong><br>${formatSensorData(sensorData)}`;
            }
        }
        
        function formatSensorData(data) {
            if (!data || Object.keys(data).length === 0) {
                return 'No sensor data';
            }
            return Object.entries(data)
                .map(([key, value]) => `${key}: ${value}`)
                .join('<br>');
        }
        
        function updateSafetyStatus(safety) {
            const movementStatus = document.getElementById('movement-status');
            if (safety.emergency_stop) {
                movementStatus.textContent = 'EMERGENCY STOP';
                movementStatus.className = 'status-bad';
            } else if (safety.movement_enabled) {
                movementStatus.textContent = 'Enabled';
                movementStatus.className = 'status-good';
            } else {
                movementStatus.textContent = 'Disabled';
                movementStatus.className = 'status-warning';
            }
        }
        
        function enableMovement() {
            fetch('/api/safety/enable_movement', {method: 'POST'})
                .then(response => response.json())
                .then(data => console.log('Movement enabled:', data));
        }
        
        function disableMovement() {
            fetch('/api/safety/disable_movement', {method: 'POST'})
                .then(response => response.json())
                .then(data => console.log('Movement disabled:', data));
        }
        
        function emergencyStop() {
            if (confirm('Are you sure you want to activate EMERGENCY STOP?')) {
                fetch('/api/safety/emergency_stop', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => console.log('Emergency stop activated:', data));
            }
        }
        
        // Refresh robot data every 5 seconds
        setInterval(() => {
            socket.emit('get_robots');
        }, 5000);
    </script>
</body>
</html>
"""

async def main():
    """Main entry point"""
    controller = UnifiedRobotController()
    
    # Initialize components
    success = await controller.initialize()
    if not success:
        logger.error("❌ Failed to initialize unified controller")
        return
    
    # Run the controller
    controller.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Unified Robot Controller shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")