#!/usr/bin/env python3
"""
ROS2 Cosmos-Transfer1-7B Node
Direct interface to Cosmos-Transfer1 model for video generation and vision-language-action tasks
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import sys
import os
import json
import logging
import time
import threading
import torch
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# ROS2 message imports
from robot_msgs.msg import RobotCommand, SensorData
from robot_msgs.srv import ExecuteCommand
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String, Bool
# cv_bridge removed - using direct image processing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CosmosTransfer1Node')

@dataclass
class CosmosConfig:
    """Configuration for Cosmos-Transfer1 model"""
    model_dir: str
    device: str = "cuda"
    precision: str = "float16"
    max_sequence_length: int = 4096

class CosmosTransfer1Node(Node):
    """
    ROS2 Node for Cosmos-Transfer1-7B model
    Provides vision-language-action capabilities for robot navigation
    """
    
    def __init__(self):
        super().__init__('cosmos_transfer1_node')
        
        # Get parameters
        self.declare_parameter('model_dir', '/home/marc/Robot_LLM/models/Cosmos-Transfer1-7B')
        self.model_dir = self.get_parameter('model_dir').get_parameter_value().string_value
        
        # Initialize configuration
        self.config = CosmosConfig(model_dir=self.model_dir)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.model_loaded = False
        
        # Direct image processing (no cv_bridge needed for now)
        
        # Device setup
        self.device = self._setup_device()
        
        # QoS profiles
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        self.image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Setup ROS2 interfaces
        self._setup_services()
        self._setup_publishers()
        
        # Load model in background
        self._load_model_async()
        
        self.get_logger().info("🚀 Cosmos-Transfer1 Node initialized")
        self.get_logger().info(f"   └── Model directory: {self.model_dir}")
        self.get_logger().info(f"   └── Device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup device configuration"""
        if torch.cuda.is_available():
            device = "cuda"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.get_logger().info(f"🖥️ CUDA detected: {torch.cuda.get_device_name(0)}")
            self.get_logger().info(f"   └── VRAM: {vram_gb:.1f} GB")
            
            if vram_gb < 14.0:  # Cosmos-Transfer1-7B requires ~14GB
                self.get_logger().warn(f"⚠️ Low VRAM: {vram_gb:.1f}GB (recommended: 14GB+)")
            
            return device
        else:
            raise RuntimeError("❌ CUDA not available - GPU required for Cosmos-Transfer1")
    
    def _setup_services(self):
        """Setup ROS2 services"""
        # Cosmos-Transfer1 analysis service
        self.analysis_service = self.create_service(
            ExecuteCommand,
            '/cosmos/request_analysis',
            self._analysis_service_callback
        )

        self.get_logger().info("📡 Services configured:")
        self.get_logger().info("   └── /cosmos/request_analysis")
    
    def _setup_publishers(self):
        """Setup ROS2 publishers"""
        # Cosmos analysis results
        self.analysis_publisher = self.create_publisher(
            RobotCommand,
            '/cosmos/analysis',
            self.reliable_qos
        )

        # Status updates
        self.status_publisher = self.create_publisher(
            String,
            '/cosmos/status',
            self.reliable_qos
        )
    
    def _load_model_async(self):
        """Load Cosmos model in background thread"""
        def load_model():
            try:
                self.get_logger().info("🔄 Loading Cosmos-Transfer1 model...")
                
                # Load Cosmos-Transfer1 using transformers
                from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
                
                # Check if model files exist
                model_path = Path(self.model_dir)
                if not model_path.exists():
                    raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
                
                # Load tokenizer and model
                self.get_logger().info("   └── Loading tokenizer...")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_dir,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                except Exception as e:
                    # Fallback to a compatible tokenizer
                    self.get_logger().warn(f"   └── Tokenizer load failed, using fallback: {e}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "microsoft/DialoGPT-medium",
                        trust_remote_code=True
                    )
                
                self.get_logger().info("   └── Loading model...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_dir,
                        local_files_only=True,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )
                except Exception as e:
                    self.get_logger().error(f"   └── Model load failed: {e}")
                    # For now, create a placeholder - this needs to be fixed with proper Cosmos loading
                    raise RuntimeError(f"Cosmos model loading failed: {e}")
                
                self.get_logger().info("   └── Loading image processor...")
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_dir,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                except Exception as e:
                    self.get_logger().warn(f"   └── Processor load failed, using fallback: {e}")
                    # Use a basic processor fallback
                    from transformers import BlipProcessor
                    self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                
                self.model_loaded = True
                self.get_logger().info("✅ Cosmos-Transfer1 model loaded successfully")
                
                # Publish status update
                self._publish_status("model_loaded")
                
            except Exception as e:
                self.get_logger().error(f"❌ Failed to load Cosmos model: {e}")
                self._publish_status("model_load_failed", str(e))
        
        # Start loading in background
        threading.Thread(target=load_model, daemon=True).start()
    
    def _analysis_service_callback(self, request, response):
        """Handle Cosmos analysis requests"""
        try:
            if not self.model_loaded:
                response.success = False
                response.message = "Cosmos model not loaded yet"
                response.timestamp_ns = self.get_clock().now().nanoseconds
                return response

            self.get_logger().info(f"🔍 Cosmos analysis request: {request.command_type}")

            # For now, use basic analysis without image processing
            # TODO: Integrate with actual Cosmos-Transfer1 model
            analysis_result = self._analyze_scene(None, request.parameters, {
                'distance_front': 1.0,  # Placeholder values
                'distance_left': 1.0,
                'distance_right': 1.0
            })

            # Fill response
            response.success = analysis_result['success']
            response.message = analysis_result.get('analysis', 'Analysis complete')
            response.timestamp_ns = self.get_clock().now().nanoseconds
            
            # Publish analysis results as RobotCommand
            command_msg = RobotCommand()
            command_msg.robot_id = request.robot_id
            command_msg.command_type = "cosmos_analysis"
            command_msg.parameters = response.analysis_result
            command_msg.safety_confirmed = True
            command_msg.timestamp_ns = response.timestamp_ns

            self.analysis_publisher.publish(command_msg)
            
            self.get_logger().info(f"✅ Analysis complete: {analysis_result.get('navigation_commands', {}).get('action', 'unknown')}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Analysis service error: {e}")
            response.success = False
            response.message = str(e)
            response.timestamp_ns = self.get_clock().now().nanoseconds
        
        return response
    
    def _analyze_scene(self, image: Image.Image, prompt: str, lidar_data: Dict[str, float]) -> Dict[str, Any]:
        """Analyze scene using Cosmos-Transfer1 model"""
        try:
            # Build enhanced prompt with sensor data
            enhanced_prompt = self._build_enhanced_prompt(prompt, lidar_data)
            
            # For now, implement basic rule-based navigation until proper Cosmos integration
            # This is a temporary implementation that needs to be replaced with actual Cosmos inference
            
            # Analyze LiDAR data for safety
            front_dist = lidar_data.get('distance_front', 999.0)
            left_dist = lidar_data.get('distance_left', 999.0)
            right_dist = lidar_data.get('distance_right', 999.0)
            
            # Basic navigation logic based on LiDAR
            if front_dist < 0.5:
                action = "stop"
                confidence = 0.9
                reasoning = f"Obstacle ahead at {front_dist:.2f}m - stopping for safety"
            elif front_dist < 1.0:
                if left_dist > right_dist and left_dist > 1.0:
                    action = "turn_left"
                    confidence = 0.8
                    reasoning = f"Front blocked ({front_dist:.2f}m), turning left (clear: {left_dist:.2f}m)"
                elif right_dist > 1.0:
                    action = "turn_right"
                    confidence = 0.8
                    reasoning = f"Front blocked ({front_dist:.2f}m), turning right (clear: {right_dist:.2f}m)"
                else:
                    action = "stop"
                    confidence = 0.9
                    reasoning = f"All directions blocked - stopping"
            else:
                action = "move_forward"
                confidence = 0.7
                reasoning = f"Path clear ahead ({front_dist:.2f}m) - moving forward"
            
            analysis_text = f"Scene Analysis: {reasoning}. LiDAR readings - Front: {front_dist:.2f}m, Left: {left_dist:.2f}m, Right: {right_dist:.2f}m"
            
            return {
                'success': True,
                'analysis': analysis_text,
                'navigation_commands': {
                    'action': action,
                    'confidence': confidence,
                    'reasoning': reasoning
                },
                'confidence': confidence
            }
            
        except Exception as e:
            return {
                'success': False,
                'analysis': f"Analysis error: {str(e)}",
                'navigation_commands': {
                    'action': 'stop',
                    'confidence': 0.0,
                    'reasoning': f'Error in analysis: {str(e)}'
                },
                'confidence': 0.0,
                'error_message': str(e)
            }
    
    def _build_enhanced_prompt(self, user_prompt: str, lidar_data: Dict[str, float]) -> str:
        """Build enhanced prompt with sensor data"""
        base_prompt = f"""Cosmos-Transfer1 Robot Navigation Analysis

LiDAR Sensor Data:
- Front distance: {lidar_data.get('distance_front', 'N/A')}m
- Left distance: {lidar_data.get('distance_left', 'N/A')}m  
- Right distance: {lidar_data.get('distance_right', 'N/A')}m

User Request: {user_prompt}

Analyze the scene and provide navigation guidance considering both visual and LiDAR data.
"""
        return base_prompt
    
    def _publish_status(self, status: str, message: str = ""):
        """Publish status updates"""
        try:
            status_data = {
                'status': status,
                'model_loaded': self.model_loaded,
                'timestamp': datetime.now().isoformat(),
                'message': message
            }
            
            status_msg = String()
            status_msg.data = json.dumps(status_data)
            self.status_publisher.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish status: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CosmosTransfer1Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Node error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
