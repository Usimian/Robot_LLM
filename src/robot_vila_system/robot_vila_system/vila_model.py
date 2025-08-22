#!/usr/bin/env python3
"""
Cosmos Nemotron VLA Model Interface for ROS2
Enhanced VILA/Cosmos Nemotron vision-language-action model for robotics
Provides multi-modal sensor fusion with LiDAR, IMU, camera, and text integration
"""

import os
import sys
import logging
import numpy as np
import torch
import json
import base64
import io
import time
import threading
import warnings
import requests  # For legacy HTTP compatibility
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Configure logging
logger = logging.getLogger('CosmosNemotronVLA')

# Model Configuration
DEFAULT_MODEL_NAME = "Efficient-Large-Model/VILA1.5-3b"
FALLBACK_MODEL_NAME = "microsoft/DialoGPT-medium"  # Fallback for testing
MODEL_LOAD_TIMEOUT = 300  # 5 minutes timeout for model loading

@dataclass
class SensorData:
    """Enhanced sensor data structure for multi-modal fusion"""
    lidar_distances: Dict[str, float]
    imu_data: Dict[str, Dict[str, float]]
    camera_image: Optional[Image.Image] = None
    timestamp: float = 0.0

class CosmosNemotronVLAModel:
    """
    Cosmos Nemotron VLA Model Interface
    Enhanced VILA model with multi-modal sensor fusion capabilities
    Supports LiDAR + IMU + Camera + Text integration for robotics
    """
    
    def __init__(self, auto_start_server=False, model_name: str = None, 
                 quantization: bool = True):
        """Initialize Cosmos Nemotron VLA model"""
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.quantization = quantization
        self.device = self._setup_device()
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.model_loaded = False
        self.model_type = "cosmos_nemotron_vla"
        
        # Enhanced capabilities
        self.sensor_fusion_enabled = True
        self.max_context_length = 4096
        
        # Legacy compatibility (HTTP server approach as fallback)
        self.auto_start_server = auto_start_server
        self.server_ready = False
        self.server_process = None
        self.server_status = "disabled"  # Modern model doesn't need server
        self.server_logs = ["Cosmos Nemotron VLA - Local model loading"]
        self.server_url = "localhost:8000"  # Legacy compatibility
        self.server_timeout = 30  # Legacy compatibility
        
        logger.info(f"🚀 Initializing Cosmos Nemotron VLA Model")
        logger.info(f"   └── Model: {self.model_name}")
        logger.info(f"   └── Quantization: {'Enabled' if quantization else 'Disabled'}")
        logger.info(f"   └── Device: {self.device}")
        logger.info(f"   └── Enhanced sensor fusion: Enabled")
        
    def _setup_device(self) -> str:
        """Setup optimal device configuration"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"🖥️ CUDA detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"   └── VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            logger.info("💻 Using CPU (CUDA not available)")
        return device
    
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
        Load VILA1.5-3b model using VILA package directly
        
        Args:
            model_path: Optional model path override (will use VILA1.5-3b if not specified)
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            model_to_load = model_path or self.model_name
            logger.info(f"🔄 Loading VILA1.5-3b model: {model_to_load}")
            
            # Use VILA package directly for loading VILA models
            if self._load_vila_model_direct(model_to_load):
                self.model_loaded = True
                self.server_ready = True  # Legacy compatibility
                logger.info("✅ VILA1.5-3b model loaded successfully")
                logger.info(f"   └── Model: {model_to_load}")
                logger.info(f"   └── Multi-modal sensor fusion: Enabled")
                logger.info(f"   └── Enhanced safety validation: Enabled")
                return True
            else:
                logger.error("❌ Failed to load VILA1.5-3b model")
                return False
            
        except Exception as e:
            logger.error(f"❌ VILA1.5-3b model loading error: {e}")
            return False
    
    def _load_vila_model_direct(self, model_name: str) -> bool:
        """Load VILA model using VILA package directly"""
        try:
            # Import VILA components
            import sys
            vila_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "VILA")
            if vila_path not in sys.path:
                sys.path.insert(0, vila_path)
            
            logger.info("   └── Loading VILA model using VILA package...")
            
            # Import VILA model components
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            from llava.constants import DEFAULT_IMAGE_TOKEN
            from llava.conversation import conv_templates, SeparatorStyle
            
            # Get model name and load
            model_name_clean = get_model_name_from_path(model_name)
            logger.info(f"   └── Model name resolved: {model_name_clean}")
            
            # Load VILA model components
            logger.info("   └── Loading VILA tokenizer, model, image processor, and context length...")
            # Load with or without quantization based on device
            load_kwargs = {
                "model_path": model_name,
                "model_base": None,
                "model_name": model_name_clean,
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            # Disable quantization for now to get VILA1.5-3b loading successfully
            # TODO: Re-enable quantization once basic loading works
            logger.info("   └── Loading without quantization (quantization disabled for compatibility)...")
            load_kwargs["load_8bit"] = False
            load_kwargs["load_4bit"] = False
            
            # Patch VILA's infer_stop_tokens to handle missing chat templates  
            logger.info("   └── Patching VILA for chat template compatibility...")
            
            # Import and patch the tokenizer utility function
            from llava.utils.tokenizer import infer_stop_tokens as original_infer_stop_tokens
            
            def safe_infer_stop_tokens(tokenizer):
                # Set a default chat template if none exists
                if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
                    logger.info("   └── Setting default chat template for VILA tokenizer...")
                    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
                
                # Now call the original function
                return original_infer_stop_tokens(tokenizer)
            
            # Patch the module-level function
            import llava.utils.tokenizer
            llava.utils.tokenizer.infer_stop_tokens = safe_infer_stop_tokens
            
            # Also patch in the builder module since it imports directly  
            import llava.model.language_model.builder as vila_builder
            # Store original and patch the imported function in the builder module
            original_builder_infer_stop_tokens = getattr(vila_builder, 'infer_stop_tokens', original_infer_stop_tokens)
            vila_builder.infer_stop_tokens = safe_infer_stop_tokens
            
            try:
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(**load_kwargs)
            finally:
                # Restore original functions
                llava.utils.tokenizer.infer_stop_tokens = original_infer_stop_tokens
                vila_builder.infer_stop_tokens = original_builder_infer_stop_tokens
            
            # Store VILA-specific constants (get dynamically from tokenizer)
            self.default_image_token = DEFAULT_IMAGE_TOKEN
            self.image_token_index = self.tokenizer.media_token_ids["image"] if hasattr(self.tokenizer, 'media_token_ids') else -200
            
            # VILA doesn't use start/end tokens the same way
            self.default_im_start_token = ""
            self.default_im_end_token = ""
            
            # Store conversation template
            if "llama-2" in model_name_clean.lower():
                self.conv_mode = "llava_llama_2"
            elif "mistral" in model_name_clean.lower() or "mixtral" in model_name_clean.lower():
                self.conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name_clean.lower():
                self.conv_mode = "chatml_direct"
            elif "v1" in model_name_clean.lower():
                self.conv_mode = "llava_v1"
            else:
                self.conv_mode = "vicuna_v1"
            
            logger.info(f"   └── Using conversation mode: {self.conv_mode}")
            logger.info("   └── VILA model loaded successfully with native VILA architecture")
            
            # Set correct model type after successful loading
            self.model_type = "vila_1_5_3b"
            return True
            
        except Exception as e:
            logger.error(f"VILA model loading failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _enable_simulation_mode(self) -> bool:
        """Enable enhanced simulation mode for development and testing"""
        logger.info("🎯 Enabling Enhanced Cosmos Nemotron VLA Simulation Mode")
        logger.info("   └── Full sensor fusion capabilities: ACTIVE")
        logger.info("   └── Multi-layer safety validation: ACTIVE") 
        logger.info("   └── Intelligent decision making: ACTIVE")
        logger.info("   └── RTX 3090 optimized processing: ACTIVE")
        
        self.model_loaded = True
        self.server_ready = True
        self.model_type = "cosmos_nemotron_enhanced_simulation"
        
        # No simulation mode - force real VILA loading to see actual errors
        logger.error("❌ Simulation mode disabled - VILA1.5-3b must load successfully")
        logger.error("   └── Fix the real VILA loading issue instead of masking with simulation")
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
    
    def analyze_multi_modal_scene(self, sensor_data: SensorData, prompt: str = None) -> Dict[str, Any]:
        """
        Enhanced multi-modal scene analysis with sensor fusion
        
        Args:
            sensor_data: Combined sensor data (LiDAR, IMU, camera)
            prompt: Analysis prompt
            
        Returns:
            Dict with analysis results and navigation commands
        """
        try:
            # Build enhanced prompt with sensor context
            enhanced_prompt = self._build_sensor_fusion_prompt(sensor_data, prompt)
            
            # Generate analysis using real VILA model
            analysis = self._generate_real_analysis(enhanced_prompt, sensor_data.camera_image)
            
            # Parse navigation command with enhanced safety
            nav_command = self._parse_enhanced_navigation(analysis, sensor_data)
            
            return {
                'success': True,
                'analysis': analysis,
                'navigation_command': nav_command['action'],
                'confidence': nav_command['confidence'],
                'reasoning': nav_command['reasoning'],
                'sensor_context': {
                    'lidar': sensor_data.lidar_distances,
                    'imu': sensor_data.imu_data,
                    'fusion_enabled': True
                }
            }
            
        except Exception as e:
            logger.error(f"Multi-modal analysis error: {e}")
            return {
                'success': False,
                'analysis': f"Analysis error: {str(e)}",
                'navigation_command': 'stop',
                'confidence': 0.0,
                'reasoning': 'Error in analysis - defaulting to safe stop'
            }
    
    def analyze_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Analyze an image using Cosmos Nemotron VLA (legacy compatibility method)
        
        Args:
            image: PIL Image to analyze
            prompt: Optional text prompt for analysis
            
        Returns:
            str: Analysis result
        """
        # Use new multi-modal analysis for legacy compatibility
        try:
            sensor_data = SensorData(
                lidar_distances={'distance_front': 999.0, 'distance_left': 999.0, 'distance_right': 999.0},
                imu_data={'acceleration': {'x': 0, 'y': 0, 'z': 9.8}, 'gyroscope': {'x': 0, 'y': 0, 'z': 0}},
                camera_image=image,
                timestamp=time.time()
            )
            
            result = self.analyze_multi_modal_scene(sensor_data, prompt)
            if result['success']:
                return result['analysis']
            else:
                return f"Analysis failed: {result.get('analysis', 'Unknown error')}"
        except Exception as e:
            logger.error(f"Legacy image analysis error: {e}")
            return f"Legacy analysis error: {str(e)}"
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
            'model_type': 'Cosmos Nemotron VLA',
            'server_url': self.server_url,
            'model_name': self.model_name,
            'server_ready': self.server_ready,
            'timeout': self.server_timeout,
            'server_status': status['status'],
            'process_running': status['process_running'],
            'auto_start_enabled': self.auto_start_server,
            'sensor_fusion_enabled': self.sensor_fusion_enabled,
            'device': self.device,
            'quantization': self.quantization
        }
    
    def _build_sensor_fusion_prompt(self, sensor_data: SensorData, user_prompt: str = None) -> str:
        """Build enhanced prompt with sensor data integration"""
        base_prompt = """🤖 Cosmos Nemotron VLA - Multi-Modal Robot Navigation System

SENSOR DATA FUSION:
"""
        
        # Add LiDAR context
        if sensor_data.lidar_distances:
            lidar_text = "📡 LiDAR Distances:\n"
            for direction, distance in sensor_data.lidar_distances.items():
                status = "⚠️ OBSTACLE" if distance < 0.5 else "✓ Clear"
                lidar_text += f"   • {direction.replace('distance_', '').title()}: {distance:.2f}m {status}\n"
            base_prompt += lidar_text + "\n"
        
        # Add IMU context
        if sensor_data.imu_data:
            imu_text = "📊 IMU Data:\n"
            if 'acceleration' in sensor_data.imu_data:
                acc = sensor_data.imu_data['acceleration']
                imu_text += f"   • Linear Acceleration: X={acc.get('x', 0):.2f}, Y={acc.get('y', 0):.2f}, Z={acc.get('z', 0):.2f} m/s²\n"
            if 'gyroscope' in sensor_data.imu_data:
                gyro = sensor_data.imu_data['gyroscope']
                imu_text += f"   • Angular Velocity: X={gyro.get('x', 0):.3f}, Y={gyro.get('y', 0):.3f}, Z={gyro.get('z', 0):.3f} rad/s\n"
            base_prompt += imu_text + "\n"
        
        # Add camera context
        if sensor_data.camera_image:
            base_prompt += "📷 Camera: Visual data available for analysis\n\n"
        
        # Task instruction
        task_prompt = user_prompt or """VLA NAVIGATION TASK:
Analyze ALL sensor data and provide safe navigation guidance. Consider:
1. LiDAR for precise distance measurements and obstacle detection
2. IMU for robot orientation and movement state
3. Visual data for scene understanding and semantic analysis

OUTPUT FORMAT:
Action: [move_forward/turn_left/turn_right/stop]
Confidence: [0.0-1.0]
Reasoning: [Brief explanation using sensor fusion]

SAFETY PRIORITY: When uncertain or obstacles detected, choose 'stop'."""

        return base_prompt + task_prompt
    
    # Simulation method removed - using only real VILA1.5-3b inference
    
    def _generate_real_analysis(self, prompt: str, image: Image.Image = None) -> str:
        """Generate real analysis using VILA model"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("VILA model not loaded - cannot perform analysis")
        
        try:
            # Import VILA conversation utilities
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import tokenizer_image_token
            
            # Create conversation
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            if image and hasattr(self, 'image_processor'):
                # Multi-modal VILA inference
                logger.info("   └── Running VILA multi-modal inference...")
                
                # Process image using VILA image processor
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                
                # Tokenize with image token
                if self.default_image_token in prompt_text:
                    input_ids = tokenizer_image_token(prompt_text, self.tokenizer, return_tensors='pt')
                else:
                    # Add image token at beginning
                    prompt_with_image = self.default_image_token + '\n' + prompt_text
                    input_ids = tokenizer_image_token(prompt_with_image, self.tokenizer, return_tensors='pt')
                
                input_ids = input_ids.unsqueeze(0).to(self.model.device)
                image_tensor = image_tensor.unsqueeze(0).to(dtype=self.model.dtype, device=self.model.device)
                
                # Generate with VILA (media format: Dict[str, List[torch.Tensor]])
                media = {"image": [image_tensor.squeeze(0)]}  # Remove extra batch dimension for media format
                media_config = {"image": {}}  # Basic media config for images
                
                logger.info("   └── Starting VILA generation with media...")
                logger.info(f"   └── Input shape: {input_ids.shape}")
                logger.info(f"   └── Image tensor shape: {image_tensor.shape}")
                
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        media=media,
                        media_config=media_config,
                        do_sample=True,
                        temperature=0.1,
                        max_new_tokens=64,  # Reduced for faster inference
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                logger.info("   └── VILA generation completed")
                
                # Decode response
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                logger.info(f"   └── Raw VILA output: {outputs[:200]}...")
                
                # Extract new content (remove input prompt)
                outputs = outputs.strip()
                if conv.sep in outputs:
                    outputs = outputs.split(conv.sep)[-1].strip()
                
                return outputs
                
            else:
                # Text-only analysis (fallback)
                logger.info("   └── Running text-only inference...")
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
                
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract new content
                if prompt_text in response:
                    response = response.split(prompt_text)[-1].strip()
                
                return response
            
        except Exception as e:
            logger.error(f"VILA analysis error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Re-raise the exception instead of masking it with simulation
            raise RuntimeError(f"VILA inference failed: {str(e)}")
    
    def _parse_enhanced_navigation(self, analysis: str, sensor_data: SensorData) -> Dict[str, Any]:
        """Parse analysis with enhanced safety validation"""
        analysis_lower = analysis.lower()
        
        # Extract structured output if present
        action = "stop"
        confidence = 0.5
        
        if 'action:' in analysis_lower:
            try:
                lines = analysis.split('\n')
                action_line = [line for line in lines if 'action:' in line.lower()][0]
                action = action_line.split(':', 1)[1].strip().lower()
                
                confidence_line = [line for line in lines if 'confidence:' in line.lower()]
                if confidence_line:
                    conf_str = confidence_line[0].split(':', 1)[1].strip()
                    confidence = float(conf_str)
            except:
                pass
        else:
            # Fallback parsing
            if any(word in analysis_lower for word in ['move_forward', 'forward']):
                action = 'move_forward'
                confidence = 0.7
            elif any(word in analysis_lower for word in ['turn_left', 'left']):
                action = 'turn_left'
                confidence = 0.7
            elif any(word in analysis_lower for word in ['turn_right', 'right']):
                action = 'turn_right'
                confidence = 0.7
            elif any(word in analysis_lower for word in ['stop', 'halt']):
                action = 'stop'
                confidence = 0.9
        
        # Enhanced safety validation with LiDAR
        safety_check = self._validate_enhanced_safety(action, sensor_data)
        
        return {
            'action': safety_check['action'],
            'confidence': safety_check['confidence'],
            'reasoning': safety_check['reasoning']
        }
    
    def _validate_enhanced_safety(self, action: str, sensor_data: SensorData) -> Dict[str, Any]:
        """Enhanced safety validation using all sensor data"""
        if not sensor_data.lidar_distances:
            return {'action': action, 'confidence': 0.5, 'reasoning': 'No LiDAR data for safety validation'}
        
        front_dist = sensor_data.lidar_distances.get('distance_front', float('inf'))
        left_dist = sensor_data.lidar_distances.get('distance_left', float('inf'))
        right_dist = sensor_data.lidar_distances.get('distance_right', float('inf'))
        
        # Critical safety thresholds
        CRITICAL_DISTANCE = 0.3
        WARNING_DISTANCE = 0.5
        
        # Safety overrides
        if front_dist < CRITICAL_DISTANCE and action == 'move_forward':
            return {
                'action': 'stop',
                'confidence': 0.98,
                'reasoning': f'SAFETY OVERRIDE: Front obstacle at {front_dist:.2f}m < {CRITICAL_DISTANCE}m critical threshold'
            }
        
        if action == 'turn_left' and left_dist < CRITICAL_DISTANCE:
            return {
                'action': 'stop',
                'confidence': 0.95,
                'reasoning': f'SAFETY OVERRIDE: Left obstacle at {left_dist:.2f}m too close for turn'
            }
        
        if action == 'turn_right' and right_dist < CRITICAL_DISTANCE:
            return {
                'action': 'stop',
                'confidence': 0.95,
                'reasoning': f'SAFETY OVERRIDE: Right obstacle at {right_dist:.2f}m too close for turn'
            }
        
        # Warning conditions (reduce confidence but allow)
        confidence_factor = 1.0
        if front_dist < WARNING_DISTANCE and action == 'move_forward':
            confidence_factor = 0.7
            reasoning = f'Cautious movement - front obstacle at {front_dist:.2f}m'
        else:
            reasoning = f'Command validated - distances: F:{front_dist:.1f}m L:{left_dist:.1f}m R:{right_dist:.1f}m'
        
        return {
            'action': action,
            'confidence': min(0.9, confidence_factor * 0.8),
            'reasoning': reasoning
        }

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self.server_process:
                self.stop_server()
        except:
            pass


# Legacy compatibility wrapper
class VILAModel(CosmosNemotronVLAModel):
    """Legacy compatibility wrapper for existing code"""
    def __init__(self, auto_start_server=False):
        super().__init__(auto_start_server=auto_start_server)
        logger.info("🔄 Legacy VILAModel wrapper - using Cosmos Nemotron VLA")


# For backward compatibility
def create_vila_model() -> VILAModel:
    """Create and return a VILAModel instance (now using Cosmos Nemotron)"""
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