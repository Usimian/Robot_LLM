#!/usr/bin/env python3
"""
VILA Model Interface for ROS2
Provides a simplified interface to the VILA vision-language model for ROS2 integration
Uses HTTP client approach to communicate with VILA server
"""

import os
import sys
import logging
import numpy as np
import requests
import json
import base64
import io
import time
import subprocess
import threading
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image

# Configure logging
logger = logging.getLogger('VILAModel')

# VILA Server Configuration
VILA_SERVER_URL = "http://localhost:8000"
VILA_MODEL_NAME = "VILA1.5-3B"
VILA_SERVER_TIMEOUT = 30  # seconds

class VILAModel:
    """
    VILA Model Interface using HTTP client approach
    Communicates with VILA server via REST API and manages server lifecycle
    """
    
    def __init__(self, auto_start_server=True):
        """Initialize the VILA HTTP client with optional server management"""
        self.server_url = VILA_SERVER_URL
        self.model_name = VILA_MODEL_NAME
        self.server_ready = False
        self.auto_start_server = auto_start_server
        self.server_process = None
        self.server_monitor_thread = None
        self.server_status = "stopped"  # stopped, starting, running, error
        self.server_logs = []
        self.max_log_lines = 100
        
        logger.info(f"🚀 Initializing VILA HTTP client for server: {self.server_url}")
        
        if auto_start_server:
            self.start_server()
    
    def start_server(self):
        """Start the VILA server subprocess"""
        if self.server_process and self.server_process.poll() is None:
            logger.info("🔄 VILA server already running")
            return
        
        try:
            logger.info("🚀 Starting VILA server...")
            self.server_status = "starting"
            
            # Find the simple_vila_server.py script
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            server_script = os.path.join(workspace_root, "simple_vila_server.py")
            
            if not os.path.exists(server_script):
                # Try alternative locations
                server_script = os.path.join(os.getcwd(), "simple_vila_server.py")
                if not os.path.exists(server_script):
                    raise FileNotFoundError(f"VILA server script not found at {server_script}")
            
            # Start the server process
            self.server_process = subprocess.Popen(
                [sys.executable, server_script, "--port", "8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Start monitoring thread
            self.server_monitor_thread = threading.Thread(
                target=self._monitor_server_output,
                daemon=True
            )
            self.server_monitor_thread.start()
            
            logger.info("✅ VILA server startup initiated")
            
        except Exception as e:
            logger.error(f"❌ Failed to start VILA server: {e}")
            self.server_status = "error"
            self.server_logs.append(f"ERROR: Failed to start server: {e}")
    
    def stop_server(self):
        """Stop the VILA server subprocess"""
        if self.server_process:
            logger.info("🛑 Stopping VILA server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Force killing VILA server...")
                self.server_process.kill()
            self.server_process = None
            self.server_status = "stopped"
            logger.info("✅ VILA server stopped")
    
    def restart_server(self):
        """Restart the VILA server"""
        logger.info("🔄 Restarting VILA server...")
        self.stop_server()
        time.sleep(2)
        self.start_server()
    
    def _monitor_server_output(self):
        """Monitor server output in background thread"""
        if not self.server_process:
            return
            
        try:
            for line in iter(self.server_process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    self.server_logs.append(line)
                    
                    # Keep log size manageable
                    if len(self.server_logs) > self.max_log_lines:
                        self.server_logs = self.server_logs[-self.max_log_lines:]
                    
                    # Update status based on log messages
                    if "✅ Simple VILA Server ready!" in line:
                        self.server_status = "running"
                        logger.info("✅ VILA server is ready")
                    elif "ERROR" in line or "Traceback" in line:
                        self.server_status = "error"
                
                # Check if process is still running
                if self.server_process and self.server_process.poll() is not None:
                    break
                    
        except Exception as e:
            logger.error(f"Error monitoring server: {e}")
        
        # Process ended
        if self.server_status == "starting":
            self.server_status = "error"
        elif self.server_status == "running":
            self.server_status = "stopped"
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status and recent logs"""
        is_running = self.server_process and self.server_process.poll() is None
        return {
            'status': self.server_status,
            'process_running': is_running,
            'recent_logs': self.server_logs[-10:],  # Last 10 log lines
            'server_ready': self.server_ready,
            'server_url': self.server_url
        }
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Check if VILA server is available and ready
        
        Args:
            model_path: Ignored in HTTP client mode
            
        Returns:
            bool: True if server is ready, False otherwise
        """
        try:
            logger.info(f"🔗 Checking VILA server availability at {self.server_url}")
            
            # Try to connect to the server
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                self.server_ready = True
                logger.info("✅ VILA server is ready")
                return True
            else:
                logger.warning(f"⚠️ VILA server responded with status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ VILA server not available: {e}")
            logger.info("💡 Make sure to start VILA server with: cd VILA && python server.py --model-path VILA1.5-3B")
            return False
    
    def generate_response(self, prompt: str, image: Image.Image) -> Dict[str, Any]:
        """
        Generate a response using VILA server (wrapper for ROS2 compatibility)
        
        Args:
            prompt: Text prompt for analysis
            image: PIL Image to analyze
            
        Returns:
            Dict with success flag and analysis result
        """
        try:
            analysis_result = self.analyze_image(image, prompt)
            return {
                'success': True,
                'analysis': analysis_result
            }
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return {
                'success': False,
                'analysis': f"Error: {str(e)}"
            }
    
    def analyze_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Analyze an image using VILA server
        
        Args:
            image: PIL Image to analyze
            prompt: Optional text prompt for analysis
            
        Returns:
            str: Analysis result from VILA
        """
        if not self.server_ready:
            if not self.load_model():
                return "VILA server not available"
        
        try:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Default prompt for robot navigation
            if prompt is None:
                prompt = ("Analyze this image from a robot's camera. Describe what you see and suggest "
                         "a navigation command (forward, backward, turn_left, turn_right, or stop) "
                         "to help the robot navigate safely.")
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.1
            }
            
            logger.info("📤 Sending image analysis request to VILA server")
            
            # Send request to VILA server
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=VILA_SERVER_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                logger.info(f"✅ VILA analysis received: {analysis[:100]}...")
                return analysis
            else:
                error_msg = f"VILA server error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "VILA server request timed out"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error communicating with VILA server: {e}"
            logger.error(error_msg)
            return error_msg
    
    def generate_robot_command(self, image: Image.Image, context: str = "") -> Dict[str, Any]:
        """
        Generate robot navigation command based on image analysis
        
        Args:
            image: PIL Image from robot camera
            context: Additional context about robot state
            
        Returns:
            Dict containing navigation command and confidence
        """
        try:
            # Create navigation-specific prompt
            nav_prompt = (
                "You are a robot navigation assistant. Analyze this camera image and provide "
                "a navigation command. Respond with ONLY one of these commands: "
                "forward, backward, turn_left, turn_right, or stop. "
                "Consider obstacles, paths, and safety. "
                f"Additional context: {context}"
            )
            
            # Get analysis from VILA
            analysis = self.analyze_image(image, nav_prompt)
            
            # Parse the response to extract command
            command = self._parse_navigation_command(analysis)
            
            result = {
                'command': command,
                'confidence': 0.8,  # Default confidence for HTTP client mode
                'reasoning': analysis,
                'timestamp': time.time()
            }
            
            logger.info(f"🎯 Generated robot command: {command}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating robot command: {e}")
            return {
                'command': 'stop',
                'confidence': 0.0,
                'reasoning': f"Error: {e}",
                'timestamp': time.time()
            }
    
    def _parse_navigation_command(self, analysis: str) -> str:
        """
        Parse VILA analysis to extract navigation command
        
        Args:
            analysis: Text analysis from VILA
            
        Returns:
            str: Navigation command (forward, backward, turn_left, turn_right, stop)
        """
        analysis_lower = analysis.lower()
        
        # Priority order for command detection
        if any(word in analysis_lower for word in ['stop', 'halt', 'wait', 'danger', 'obstacle']):
            return 'stop'
        elif any(word in analysis_lower for word in ['forward', 'ahead', 'straight', 'continue']):
            return 'forward'
        elif any(word in analysis_lower for word in ['backward', 'back', 'reverse']):
            return 'backward'
        elif any(word in analysis_lower for word in ['turn left', 'left', 'turn_left']):
            return 'turn_left'
        elif any(word in analysis_lower for word in ['turn right', 'right', 'turn_right']):
            return 'turn_right'
        else:
            # Default to stop if unclear
            logger.warning(f"⚠️ Unclear navigation command in analysis: {analysis[:100]}")
            return 'stop'
    
    def is_model_loaded(self) -> bool:
        """Check if VILA server is ready"""
        return self.server_ready
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the VILA model/server"""
        status = self.get_server_status()
        return {
            'model_type': 'VILA HTTP Client with Server Management',
            'server_url': self.server_url,
            'model_name': self.model_name,
            'server_ready': self.server_ready,
            'timeout': VILA_SERVER_TIMEOUT,
            'server_status': status['status'],
            'process_running': status['process_running'],
            'auto_start_enabled': self.auto_start_server
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self.server_process:
                self.stop_server()
        except:
            pass


# For backward compatibility
def create_vila_model() -> VILAModel:
    """Create and return a VILAModel instance"""
    return VILAModel()


if __name__ == "__main__":
    # Test the VILA HTTP client
    logging.basicConfig(level=logging.INFO)
    
    vila = VILAModel()
    print(f"Model info: {vila.get_model_info()}")
    
    # Test server connection
    if vila.load_model():
        print("✅ VILA server is ready for testing")
        
        # Create a test image
        test_image = Image.new('RGB', (640, 480), color='blue')
        
        # Test analysis
        result = vila.generate_robot_command(test_image, "Testing robot navigation")
        print(f"Test result: {result}")
    else:
        print("❌ VILA server not available. Start with: cd VILA && python server.py --model-path VILA1.5-3B")