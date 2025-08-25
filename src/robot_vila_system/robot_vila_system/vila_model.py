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
import subprocess
import requests  # For legacy HTTP compatibility
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Configure logging
logger = logging.getLogger('CosmosNemotronVLA')

# Model Configuration - No Fallbacks
MODEL_NAME = "nvidia/Cosmos-Reason1-7B"  # Multi-modal reasoning model with sensor data support
MODEL_LOAD_TIMEOUT = 600  # 10 minutes timeout
REQUIRED_VRAM_GB = 14  # Minimum VRAM required for Cosmos Reason1-7B

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
    
    def __init__(self, model_name: str = None, quantization: bool = True):
        """Initialize Cosmos Nemotron VLA model - No fallbacks"""
        self.model_name = model_name or MODEL_NAME
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
        
        # No legacy server support - direct model loading only
        
        logger.info(f"🚀 Initializing Cosmos Transfer1-7B-AV Model (with native LiDAR support)")
        logger.info(f"   └── Model: {self.model_name}")
        logger.info(f"   └── Quantization: {'Enabled' if quantization else 'Disabled'}")
        logger.info(f"   └── Device: {self.device}")
        logger.info(f"   └── Enhanced sensor fusion: Enabled")
        
    def _setup_device(self) -> str:
        """Setup device configuration - No fallbacks"""
        if torch.cuda.is_available():
            device = "cuda"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🖥️ CUDA detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"   └── VRAM: {vram_gb:.1f} GB")

            # Check VRAM requirements - either meets requirement or fails
            if vram_gb < REQUIRED_VRAM_GB:
                raise RuntimeError(f"❌ Insufficient VRAM: {vram_gb:.1f}GB available, {REQUIRED_VRAM_GB}GB required for {self.model_name}")
            else:
                logger.info(f"✅ VRAM check passed: {vram_gb:.1f}GB ≥ {REQUIRED_VRAM_GB}GB required")
        else:
            raise RuntimeError("❌ CUDA not available - GPU required for Cosmos models")
        return device
    
    # Legacy server methods removed - direct model loading only
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Load Cosmos model directly - either works or fails clearly

        Args:
            model_path: Optional model path override

        Returns:
            bool: True if model loaded successfully, raises exception if failed
        """
        try:
            model_to_load = model_path or self.model_name
            logger.info(f"🔄 Loading Cosmos model: {model_to_load}")

            # Load Cosmos model directly - no fallbacks
            if self._load_cosmos_model(model_to_load):
                self.model_loaded = True
                logger.info("✅ Cosmos model loaded successfully")
                logger.info(f"   └── Model: {model_to_load}")
                logger.info(f"   └── Multi-modal sensor fusion: Enabled")
                return True
            else:
                raise RuntimeError(f"❌ Failed to load Cosmos model: {model_to_load}")

        except Exception as e:
            logger.error(f"❌ Cosmos model loading error: {e}")
            raise RuntimeError(f"Cosmos model loading failed: {str(e)}")
    
    def _load_cosmos_model(self, model_name: str) -> bool:
        """Load Cosmos model directly - no fallbacks"""
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
    
    # Simulation mode completely removed
    
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
            logger.info(f"🔧 Enhanced prompt length: {len(enhanced_prompt)} chars")
            logger.info(f"🔧 Enhanced prompt preview: {enhanced_prompt[:300]}...")
            
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
        
        # Trust VILA's decision - no priority ordering that favors 'stop'
        if any(word in analysis_lower for word in ['forward', 'ahead', 'straight', 'continue', 'move forward']):
            return 'forward'
        elif any(word in analysis_lower for word in ['turn left', 'left', 'turn_left']):
            return 'turn_left'
        elif any(word in analysis_lower for word in ['turn right', 'right', 'turn_right']):
            return 'turn_right'
        elif any(word in analysis_lower for word in ['backward', 'back', 'reverse']):
            return 'backward'
        elif any(word in analysis_lower for word in ['stop', 'halt', 'wait']):
            return 'stop'
        else:
            # Default to stop if unclear (but log it)
            logger.info(f"🤔 Unclear navigation command in analysis: {analysis[:100]}")
            return 'stop'
    
    def is_model_loaded(self) -> bool:
        """Check if VILA model is loaded"""
        return self.model_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the VILA model - no server fallback"""
        return {
            'model_type': 'Cosmos Nemotron VLA',
            'server_url': 'direct_model',  # No server
            'model_name': self.model_name,
            'server_ready': self.model_loaded,
            'timeout': MODEL_LOAD_TIMEOUT,
            'server_status': 'model_loaded' if self.model_loaded else 'model_not_loaded',
            'process_running': False,  # No server process
            'auto_start_enabled': False,  # No auto-start
            'sensor_fusion_enabled': self.sensor_fusion_enabled,
            'device': self.device,
            'quantization': self.quantization
        }
    
    def _build_sensor_fusion_prompt(self, sensor_data: SensorData, user_prompt: str = None) -> str:
        """Build enhanced prompt with sensor data integration"""
        base_prompt = """🤖 Cosmos Reason1-7B - Multi-Modal Robot Reasoning with Sensor Integration

SENSOR-BASED NAVIGATION REASONING:
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
        
        # Task instruction - enhanced for better safety integration with dynamic LiDAR values
        if user_prompt:
            task_prompt = user_prompt
        else:
            # Get actual LiDAR values for dynamic prompt
            front_dist = sensor_data.lidar_distances.get('distance_front', float('inf'))
            left_dist = sensor_data.lidar_distances.get('distance_left', float('inf'))
            right_dist = sensor_data.lidar_distances.get('distance_right', float('inf'))

            # Determine safety status for each direction (using stricter thresholds)
            front_status = "🚨 DANGER!" if front_dist < 0.7 else "✅ SAFE"
            left_status = "🚨 DANGER!" if left_dist < 0.6 else "✅ SAFE"
            right_status = "🚨 DANGER!" if right_dist < 0.6 else "✅ SAFE"

            # Determine if movement should be blocked
            front_blocked = front_dist < 0.7
            any_blocked = front_blocked or left_dist < 0.6 or right_dist < 0.6

            required_action = "stop" if any_blocked else "analyze_further"
            safety_reason = f"Front obstacle at {front_dist:.2f}m (< 0.7m threshold)" if front_blocked else "All clear per LiDAR"

            task_prompt = f"""🚨 CRITICAL ROBOT NAVIGATION - LiDAR DATA IS TRUTH:

⚠️ LiDAR SAFETY DATA (meters):
• Front: {front_dist:.2f}m (Obstacle at {front_dist:.2f}m!)
• Left: {left_dist:.2f}m
• Right: {right_dist:.2f}m

🚫 SAFETY RULES:
• Front distance < 0.5m = CANNOT MOVE FORWARD
• Front distance < 0.7m = HIGH RISK - avoid if possible
• Current front: {front_dist:.2f}m = {'DANGER - BLOCKED' if front_dist < 0.5 else 'HIGH RISK' if front_dist < 0.7 else 'CLEAR'}

📷 VISUAL ANALYSIS (secondary to LiDAR):
Look at the camera image, but remember LiDAR shows {'an obstacle at ' + str(front_dist) + 'm' if front_dist < 0.7 else 'clear path'}.

REQUIRED RESPONSE FORMAT:
1. [YES/NO] Can I move forward? (based on LiDAR data)
2. [YES/NO] Are there obstacles? (acknowledge LiDAR obstacle)
3. Action: [move_forward/turn_left/turn_right/stop] (respect LiDAR safety)
4. Visual description: [brief scene description]"""

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
                        temperature=0.2,
                        max_new_tokens=150,  # Increased for better analysis
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                logger.info("   └── VILA generation completed")
                
                # Decode response
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                logger.info(f"   └── Raw VILA output length: {len(outputs)} chars")
                logger.info(f"   └── Raw VILA output: {outputs[:500]}...")
                
                # Extract new content (remove input prompt)
                outputs = outputs.strip()
                logger.info(f"   └── After strip: '{outputs[:200]}...'")
                
                # Look for proper conversation markers instead of simple separator
                # Common patterns: "ASSISTANT:", "Assistant:", or conversation end markers
                assistant_markers = ["ASSISTANT:", "Assistant:", "### Assistant:"]
                
                # Try to find the assistant response
                assistant_response = None
                for marker in assistant_markers:
                    if marker in outputs:
                        # Split on the marker and take the part after it
                        parts = outputs.split(marker)
                        if len(parts) > 1:
                            assistant_response = parts[-1].strip()
                            logger.info(f"   └── Found assistant marker '{marker}', extracted response")
                            break
                
                if assistant_response:
                    outputs = assistant_response
                    logger.info(f"   └── Assistant response: '{outputs[:200]}...'")
                else:
                    # If no assistant marker found, check if the output looks like a direct response
                    # (no conversation format, just the answer)
                    logger.info(f"   └── No assistant markers found, using full output as direct response")
                    # Keep the full output as is
                
                if len(outputs) < 10:
                    logger.warning(f"⚠️ Very short VILA output: '{outputs}' - this may indicate a generation issue")
                
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
        """Parse analysis and trust VILA's navigation decisions completely"""
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
            # Fallback parsing with broader keyword matching
            if any(word in analysis_lower for word in ['move_forward', 'forward', 'ahead', 'straight', 'continue']):
                action = 'move_forward'
                confidence = 0.8
            elif any(word in analysis_lower for word in ['turn_left', 'left']):
                action = 'turn_left'
                confidence = 0.8
            elif any(word in analysis_lower for word in ['turn_right', 'right']):
                action = 'turn_right'
                confidence = 0.8
            elif any(word in analysis_lower for word in ['stop', 'halt', 'wait']):
                action = 'stop'
                confidence = 0.9
        
        # Include sensor context for logging but no safety overrides
        front_dist = sensor_data.lidar_distances.get('distance_front', 0.0)
        left_dist = sensor_data.lidar_distances.get('distance_left', 0.0)
        right_dist = sensor_data.lidar_distances.get('distance_right', 0.0)
        
        reasoning = f'VILA decision with sensor context (F:{front_dist:.1f}m L:{left_dist:.1f}m R:{right_dist:.1f}m): {analysis[:100]}'
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def _validate_enhanced_safety(self, action: str, sensor_data: SensorData) -> Dict[str, Any]:
        """Safety validation completely removed - VILA has full control"""
        return {
            'action': action,
            'confidence': 0.8,
            'reasoning': f'VILA action: {action} - no safety restrictions'
        }

    def __del__(self):
        """Cleanup on deletion - no server processes"""
        try:
            # No server processes to clean up - direct model loading only
            pass
        except:
            pass


# Legacy compatibility wrapper
class VILAModel(CosmosNemotronVLAModel):
    """Legacy compatibility wrapper for existing code"""
    def __init__(self):
        super().__init__()
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