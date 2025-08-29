#!/usr/bin/env python3
"""
Local Vision-Language Model Navigation Node

This ROS2 node uses a local VLM (Qwen2-VL-7B-Instruct) running on RTX 3090 
for intelligent robot navigation decisions based on camera images and sensor data.

Author: Robot LLM System
"""

import json
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# ROS2 Messages
from robot_msgs.msg import RobotCommand, SensorData
from robot_msgs.srv import ExecuteCommand
from sensor_msgs.msg import Image as ROSImage, LaserScan
from std_msgs.msg import String

# Computer Vision and ML
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import cv2

class LocalVLMNavigationNode(Node):
    """ROS2 Node for local VLM-based robot navigation"""
    
    def __init__(self):
        super().__init__('local_vlm_navigation_node')
        
        # Configuration - Optimized for speed
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_memory_gb = 16  # Leave 8GB free on RTX 3090 for stability
        
        # Aggressive speed optimization settings
        self.max_image_size = 112  # Reduce to 112x112 for maximum speed (16x fewer pixels than 448x448)
        self.max_new_tokens = 16   # Reduce to 16 tokens for ultra-fast generation
        self.use_cache = True      # Enable KV cache for faster inference
        self.skip_chat_template = True  # Skip expensive chat template processing
        
        # Model components (initialized later)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Data storage with thread safety
        self.current_camera_image = None
        self.current_sensor_data = None
        self.current_lidar_data = None
        self.camera_lock = threading.Lock()
        self.sensor_data_lock = threading.Lock()
        self.lidar_lock = threading.Lock()
        
        # Request management - prevent multiple simultaneous analysis requests
        self.analysis_in_progress = False
        self.analysis_lock = threading.Lock()
        self.request_queue = []
        self.max_queue_size = 1  # Only keep the most recent request
        
        # Initialize ROS2 components
        self._setup_ros2_components()
        
        # Start model loading in background
        self._start_model_loading()
        
        self.get_logger().info("🤖 Local VLM Navigation Node initialized")
    
    def _setup_ros2_components(self):
        """Setup ROS2 publishers, subscribers, and services"""
        
        # QoS Profiles
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        self.image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )
        
        # Publishers
        self.analysis_publisher = self.create_publisher(
            RobotCommand,
            '/vlm/analysis', 
            self.reliable_qos
        )

        self.status_publisher = self.create_publisher(
            String,
            '/vlm/status',
            self.reliable_qos
        )
        
        # Subscribers
        self.sensor_subscriber = self.create_subscription(
            SensorData,
            '/robot/sensors',
            self._sensor_data_callback,
            self.sensor_qos
        )
        
        self.camera_subscriber = self.create_subscription(
            ROSImage,
            '/realsense/camera/color/image_raw',
            self._camera_image_callback,
            self.image_qos
        )
        
        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self._lidar_scan_callback,
            self.sensor_qos
        )
        
        # Services
        self.analysis_service = self.create_service(
            ExecuteCommand,
            '/vlm/analyze_scene',
            self._analysis_service_callback
        )
        
        self.get_logger().info("📡 ROS2 components configured")
        self.get_logger().info("   └── Publishers: /vlm/analysis, /vlm/status")
        self.get_logger().info("   └── Subscribers: /robot/sensors, /realsense/camera/color/image_raw, /scan")
        self.get_logger().info("   └── Services: /vlm/analyze_scene")
    
    def _start_model_loading(self):
        """Start model loading in background thread"""
        def load_model():
            try:
                self.get_logger().info("🔄 Loading Qwen2-VL-7B-Instruct model for navigation...")
                self._publish_status("loading", "Initializing local VLM model...")
                
                # Load processor and tokenizer with speed optimizations
                self.get_logger().info("   └── Loading processor and tokenizer...")
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    padding_side="left"  # Optimize for batch processing
                )
                
                # Load model with speed and memory optimization
                self.get_logger().info("   └── Loading Qwen2-VL model with speed optimization...")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,  # Use FP16 for speed and memory efficiency
                    device_map="auto",          # Automatic device placement
                    trust_remote_code=True,
                    max_memory={0: f"{self.max_memory_gb}GB"},  # Limit GPU memory usage
                    use_cache=True  # Enable KV caching for faster generation
                )
                
                # Set to evaluation mode and optimize for inference
                self.model.eval()
                
                # Compile model for faster inference (PyTorch 2.0+)
                try:
                    if hasattr(torch, 'compile'):
                        self.get_logger().info("   └── Compiling model for faster inference...")
                        self.model = torch.compile(self.model, mode="reduce-overhead")
                        self.get_logger().info("   └── Model compiled successfully")
                except Exception as e:
                    self.get_logger().warn(f"   └── Model compilation failed (not critical): {e}")
                
                # Enable CUDA optimizations
                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    self.get_logger().info("   └── CUDA optimizations enabled")
                
                # Log memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    self.get_logger().info(f"   └── GPU memory: {memory_used:.2f}GB / {memory_total:.2f}GB")
                
                self.model_loaded = True
                self.get_logger().info("✅ Qwen2-VL-7B-Instruct loaded successfully for navigation")
                self._publish_status("ready", "Local VLM navigation system ready", model_name=self.model_name)
                
            except Exception as e:
                error_msg = f"Model loading failed: {e}"
                self.get_logger().error(f"❌ {error_msg}")
                self._publish_status("model_load_failed", error_msg)
        
        # Start loading in background
        loading_thread = threading.Thread(target=load_model, daemon=True)
        loading_thread.start()
    
    def _publish_status(self, status: str, message: str, model_name: str = None):
        """Publish status updates"""
        status_msg = String()
        status_data = {
            'status': status,
            'message': message,
            'model_loaded': self.model_loaded,
            'timestamp': time.time()
        }
        if model_name:
            status_data['model_name'] = model_name
        status_msg.data = json.dumps(status_data)
        self.status_publisher.publish(status_msg)
    
    def _sensor_data_callback(self, msg: SensorData):
        """Handle incoming sensor data"""
        with self.sensor_data_lock:
            self.current_sensor_data = {
                'distance_front': msg.distance_front,
                'distance_left': msg.distance_left,
                'distance_right': msg.distance_right,
                'battery_voltage': msg.battery_voltage,
                'cpu_temp': msg.cpu_temp,
                'cpu_usage': msg.cpu_usage,
                'timestamp': msg.timestamp_ns
            }
    
    def _camera_image_callback(self, msg: ROSImage):
        """Handle incoming camera images"""
        try:
            with self.camera_lock:
                pil_image = self._ros_image_to_pil(msg)
                self.current_camera_image = pil_image
                self.get_logger().debug(f"📷 Camera image received: {pil_image.size} pixels, {msg.encoding}")
        except Exception as e:
            self.get_logger().warn(f"Camera image conversion failed: {e}")
    
    def _lidar_scan_callback(self, msg: LaserScan):
        """Handle incoming LiDAR scans"""
        try:
            with self.lidar_lock:
                self.current_lidar_data = self._process_lidar_scan(msg)
        except Exception as e:
            self.get_logger().warn(f"LiDAR processing failed: {e}")
    
    def _ros_image_to_pil(self, ros_image: ROSImage) -> Image.Image:
        """Convert ROS Image message to PIL Image"""
        np_image = np.frombuffer(ros_image.data, dtype=np.uint8)
        
        if ros_image.encoding == "rgb8":
            rgb_array = np_image.reshape((ros_image.height, ros_image.width, 3))
        elif ros_image.encoding == "bgr8":
            bgr_array = np_image.reshape((ros_image.height, ros_image.width, 3))
            rgb_array = bgr_array[:, :, ::-1]  # Convert BGR to RGB
        else:
            # Default to RGB8 interpretation
            rgb_array = np_image.reshape((ros_image.height, ros_image.width, 3))
        
        return Image.fromarray(rgb_array, 'RGB')
    
    def _process_lidar_scan(self, msg: LaserScan) -> Dict[str, Any]:
        """Process LiDAR scan into navigation-relevant format"""
        ranges = np.array(msg.ranges)
        
        # Replace invalid readings with max range
        ranges = np.where(
            (ranges < msg.range_min) | (ranges > msg.range_max), 
            msg.range_max, 
            ranges
        )
        
        # Extract key directions for navigation
        num_readings = len(ranges)
        front_idx = 0
        left_idx = num_readings // 4
        right_idx = 3 * num_readings // 4
        
        return {
            'front_distance': ranges[front_idx],
            'left_distance': ranges[left_idx],
            'right_distance': ranges[right_idx],
            'min_distance': np.min(ranges),
            'ranges': ranges.tolist(),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment
        }
    
    def _analysis_service_callback(self, request, response):
        """Handle VLM analysis requests with queuing to prevent simultaneous requests"""
        try:
            if not self.model_loaded:
                response.success = False
                response.result_message = "VLM model not loaded yet"
                return response

            # Check if analysis is already in progress
            with self.analysis_lock:
                if self.analysis_in_progress:
                    self.get_logger().warn(f"🚫 Analysis already in progress, rejecting request")
                    response.success = False
                    response.result_message = "Analysis already in progress, please wait"
                    return response
                
                # Mark analysis as in progress
                self.analysis_in_progress = True

            self.get_logger().info(f"🔍 VLM analysis request: {request.command.command_type}")

            try:
                # Get current multimodal data
                current_image = self._get_current_camera_image()
                sensor_data = self._get_current_sensor_data()
                
                # Extract prompt from source_node field
                source_parts = request.command.source_node.split('|', 1)
                prompt = source_parts[1] if len(source_parts) > 1 else "Analyze the current scene for robot navigation"
                
                # Run VLM inference
                navigation_result = self._run_vlm_navigation_inference(current_image, prompt, sensor_data)
                
                # Format response
                analysis_result = {
                'success': True,
                'analysis': f"Local VLM analysis: {navigation_result['reasoning']}",
                'navigation_commands': {
                    'action': navigation_result['action'],
                    'confidence': navigation_result['confidence']
                },
                'confidence': navigation_result['confidence']
                }

                # Pack response
                result_data = {
                    'analysis': analysis_result['analysis'],
                    'navigation_commands': analysis_result['navigation_commands'],
                    'confidence': analysis_result['confidence']
                }
                
                response.success = True
                response.result_message = json.dumps(result_data)
                
                # Publish analysis results
                command_msg = RobotCommand()
                command_msg.robot_id = request.command.robot_id
                command_msg.command_type = "vlm_analysis"
                command_msg.source_node = f"vlm_analysis_result|{analysis_result['analysis']}"
                command_msg.timestamp_ns = self.get_clock().now().nanoseconds

                self.analysis_publisher.publish(command_msg)
                
                self.get_logger().info(f"✅ Analysis complete: {navigation_result['action']} (confidence: {navigation_result['confidence']:.2f})")
                
            except Exception as e:
                self.get_logger().error(f"❌ Analysis service error: {e}")
                response.success = False
                response.result_message = str(e)
            finally:
                # Always release the analysis lock
                with self.analysis_lock:
                    self.analysis_in_progress = False
                    self.get_logger().debug("🔓 Analysis lock released")
        
        except Exception as e:
            self.get_logger().error(f"❌ Analysis service error: {e}")
            response.success = False
            response.result_message = str(e)
            # Release lock on error
            with self.analysis_lock:
                self.analysis_in_progress = False
        
        return response
    
    def _get_current_camera_image(self) -> Optional[Image.Image]:
        """Get current camera image thread-safely"""
        with self.camera_lock:
            return self.current_camera_image.copy() if self.current_camera_image else None
    
    def _get_current_sensor_data(self) -> Dict[str, float]:
        """Get current sensor data thread-safely"""
        with self.sensor_data_lock:
            if self.current_sensor_data is not None:
                return self.current_sensor_data.copy()
            else:
                # Return safe defaults
                return {
                    'distance_front': 999.0,
                    'distance_left': 999.0,
                    'distance_right': 999.0,
                    'battery_voltage': 12.0,
                    'cpu_temp': 50.0,
                    'cpu_usage': 10.0,
                    'timestamp': self.get_clock().now().nanoseconds
                }
    
    def _run_vlm_navigation_inference(self, image: Optional[Image.Image], prompt: str, sensor_data: Dict[str, float]) -> Dict[str, Any]:
        """Run local VLM inference for navigation decisions"""
        start_time = time.time()
        inference_timeout = 2.0  # 2-second timeout for ultra-fast response
        try:
            if image is None:
                # Use sensor-only navigation if no image
                self.get_logger().warn("   └── No camera image available, using sensor-only navigation")
                return self._sensor_only_navigation(sensor_data, "No camera image available")
            
            self.get_logger().info(f"   └── Processing image: {image.size} pixels")
            
            # Ultra-concise prompt for maximum speed
            navigation_prompt = f"Navigate robot: F={sensor_data.get('distance_front', 0):.1f}m L={sensor_data.get('distance_left', 0):.1f}m R={sensor_data.get('distance_right', 0):.1f}m. Reply: ACTION:move_forward/turn_left/turn_right/stop"
            
            self.get_logger().info("   └── Running Qwen2-VL inference for navigation...")
            
            # Use proper chat template for compatibility
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": navigation_prompt}
                    ]
                }
            ]
            
            # Apply chat template
            self.get_logger().info("   └── Applying chat template...")
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs with speed optimization
            self.get_logger().info("   └── Processing inputs with Qwen2-VL processor...")
            
            # Resize image for faster processing
            if image.size[0] > self.max_image_size or image.size[1] > self.max_image_size:
                image = image.resize((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
                self.get_logger().debug(f"   └── Resized image to {self.max_image_size}x{self.max_image_size} for speed")
            
            inputs = self.processor(
                text=[text], 
                images=[image], 
                return_tensors="pt"
            )
            
            self.get_logger().info(f"   └── Input keys: {list(inputs.keys())}")
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with aggressive speed optimizations and timeout
            generation_start = time.time()
            with torch.no_grad():
                # Check if we're already approaching timeout
                if time.time() - start_time > inference_timeout * 0.8:
                    self.get_logger().warn(f"   └── Pre-generation timeout, using sensor fallback")
                    return self._sensor_only_navigation(sensor_data, "Pre-generation timeout")
                
                # Use torch.compile and optimized settings
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,  # Ultra-reduced tokens
                    do_sample=False,  # Deterministic for navigation safety
                    temperature=0.0,  # Set to 0 for fastest greedy decoding
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=self.use_cache,  # Enable KV caching
                    num_beams=1,  # Greedy search
                    early_stopping=True,  # Stop early
                    repetition_penalty=1.0,  # No repetition penalty for speed
                    length_penalty=1.0,      # No length penalty for speed
                    no_repeat_ngram_size=0,  # Disable n-gram checking for speed
                    output_scores=False,     # Don't compute scores for speed
                    return_dict_in_generate=False  # Simple output for speed
                )
                
                generation_time = time.time() - generation_start
                self.get_logger().info(f"   └── Generation completed in {generation_time:.3f}s")
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            response_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # Parse navigation decision from response
            navigation_result = self._parse_vlm_response(response_text, sensor_data)
            
            # Log timing performance
            total_time = time.time() - start_time
            self.get_logger().info(f"   └── VLM decision: {navigation_result['action']} (confidence: {navigation_result['confidence']:.2f}) in {total_time:.3f}s")
            
            # Add timing to result
            navigation_result['inference_time'] = total_time
            
            return navigation_result
            
        except Exception as e:
            import traceback
            error_msg = f"VLM inference failed: {e}"
            full_traceback = traceback.format_exc()
            self.get_logger().error(f"   └── {error_msg}")
            self.get_logger().error(f"   └── Full traceback: {full_traceback}")
            return self._sensor_only_navigation(sensor_data, f"{error_msg}. Using sensor-only navigation.")
    
    def _parse_vlm_response(self, response_text: str, sensor_data: Dict[str, float]) -> Dict[str, Any]:
        """Parse VLM response to extract navigation decision"""
        try:
            response_upper = response_text.upper()
            
            # Extract action
            action = "stop"  # Default safe action
            if "ACTION:" in response_upper:
                action_part = response_upper.split("ACTION:")[1].split("|")[0].strip()
                if "MOVE_FORWARD" in action_part:
                    action = "move_forward"
                elif "TURN_LEFT" in action_part:
                    action = "turn_left"
                elif "TURN_RIGHT" in action_part:
                    action = "turn_right"
                elif "STOP" in action_part:
                    action = "stop"
            
            # Extract confidence
            confidence = 0.5  # Default confidence
            if "CONFIDENCE:" in response_upper:
                try:
                    conf_part = response_upper.split("CONFIDENCE:")[1].split("|")[0].strip()
                    confidence = float(conf_part)
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except:
                    confidence = 0.5
            
            # Extract reasoning
            reasoning = "VLM navigation decision"
            if "REASON:" in response_upper:
                reasoning = response_text.split("REASON:")[1].split("|")[0].strip()
            
            # Safety check with sensors
            front_dist = sensor_data.get('distance_front', 999.0)
            if front_dist < 0.5 and action == "move_forward":
                action = "stop"
                reasoning += " (Override: obstacle detected by sensors)"
                confidence = 0.95
                
            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'vlm_response': response_text
            }
                
        except Exception as e:
            self.get_logger().warn(f"Failed to parse VLM response: {e}")
            return self._sensor_only_navigation(sensor_data, f"VLM response parsing failed: {e}")
    
    def _sensor_only_navigation(self, sensor_data: Dict[str, float], reason: str) -> Dict[str, Any]:
        """Fallback sensor-only navigation when VLM fails"""
        front_dist = sensor_data.get('distance_front', 999.0)
        left_dist = sensor_data.get('distance_left', 999.0)
        right_dist = sensor_data.get('distance_right', 999.0)
        
        if front_dist < 0.8:  # Obstacle ahead
            if left_dist > right_dist and left_dist > 1.0:
                return {
                    'action': 'turn_left',
                    'confidence': 0.85,
                    'reasoning': f'Sensor-only navigation: obstacle ahead, turning left. {reason}'
                }
            elif right_dist > 1.0:
                return {
                    'action': 'turn_right',
                    'confidence': 0.85,
                    'reasoning': f'Sensor-only navigation: obstacle ahead, turning right. {reason}'
                }
            else:
                return {
                    'action': 'stop',
                    'confidence': 0.95,
                    'reasoning': f'Sensor-only navigation: obstacles detected, stopping. {reason}'
                }
        elif front_dist > 2.0:  # Clear path
            return {
                'action': 'move_forward',
                'confidence': 0.80,
                'reasoning': f'Sensor-only navigation: clear path ahead. {reason}'
            }
        else:  # Moderate distance
            return {
                'action': 'move_forward',
                'confidence': 0.70,
                'reasoning': f'Sensor-only navigation: proceeding cautiously. {reason}'
            }

def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = LocalVLMNavigationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if 'node' in locals():
            node.get_logger().error(f"Node error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
