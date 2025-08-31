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
import transformers
transformers.logging.set_verbosity_error()  # Suppress generation warnings
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
        self.max_new_tokens = 128  # Sufficient tokens for detailed reasoning
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
        
        # No enhanced sensor publisher needed - GUI will subscribe to /scan directly
        
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
                    trust_remote_code=True,
                    use_fast=False  # Use slow processor for consistent outputs
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
                    dtype=torch.float16,        # Use FP16 for speed and memory efficiency (fixed deprecated torch_dtype)
                    device_map="cuda:0",        # Force GPU placement to avoid meta device issues
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
                    use_cache=True              # Enable KV caching for faster generation
                )
                
                # Ensure all model parameters are on GPU (fix meta device issue)
                self.get_logger().info("   └── Moving model to GPU and setting eval mode...")
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Verify no parameters are on meta device
                meta_params = sum(1 for p in self.model.parameters() if p.device.type == 'meta')
                if meta_params > 0:
                    self.get_logger().warn(f"   └── Warning: {meta_params} parameters still on meta device")
                else:
                    self.get_logger().info("   └── All parameters successfully loaded to GPU")
                
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
                # Store both processed data and raw ranges for VLM analysis
                processed_data = self._process_lidar_scan(msg)
                processed_data['ranges'] = list(msg.ranges)  # Add raw ranges for VLM
                processed_data['angle_min'] = msg.angle_min
                processed_data['angle_max'] = msg.angle_max
                processed_data['angle_increment'] = msg.angle_increment
                self.current_lidar_data = processed_data
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
                lidar_data = self._get_current_lidar_data()
                
                # Extract prompt from source_node field
                source_parts = request.command.source_node.split('|', 1)
                prompt = source_parts[1] if len(source_parts) > 1 else "Analyze the current scene for robot navigation"
            
                # Run VLM inference with LiDAR data
                navigation_result = self._run_vlm_navigation_inference(current_image, prompt, sensor_data, lidar_data)
            
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
            self.get_logger().error(f"❌ Outer analysis service error: {e}")
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
                    'battery_voltage': 12.0,
                    'cpu_temp': 50.0,
                    'cpu_usage': 10.0,
                    'timestamp': self.get_clock().now().nanoseconds
                }
    
    def _get_current_lidar_data(self) -> Optional[Dict[str, Any]]:
        """Get current LiDAR data thread-safely"""
        with self.lidar_lock:
            if self.current_lidar_data is not None:
                return self.current_lidar_data.copy()
            else:
                return None
    
    def _run_vlm_navigation_inference(self, image: Optional[Image.Image], prompt: str, sensor_data: Dict[str, float], lidar_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run local VLM inference for navigation decisions"""
        start_time = time.time()
        inference_timeout = 2.0  # 2-second timeout for ultra-fast response
        try:
            if image is None:
                # Cannot navigate without camera image
                self.get_logger().error("   └── No camera image available - cannot perform navigation")
                return {
                    'action': 'stop',
                    'confidence': 0.0,
                    'reasoning': 'No camera image available - navigation requires both camera and LiDAR data'
                }
            
            self.get_logger().debug(f"   └── Processing image: {image.size} pixels")
            
            # Check for LiDAR data availability
            if not lidar_data or 'ranges' not in lidar_data or not lidar_data['ranges']:
                self.get_logger().error("   └── No LiDAR data available - cannot perform navigation")
                return {
                    'action': 'stop',
                    'confidence': 0.0,
                    'reasoning': 'No LiDAR data available - navigation requires both camera and LiDAR data'
                }
            
            # Build sensor information from LiDAR data
            sensor_info = "Sensors: Using LiDAR_360° scan for all distance measurements"
            
            # Process full 360° LiDAR scan (1° resolution, rear 90° obstructed)
            ranges = lidar_data['ranges']
            num_points = len(ranges)
            
            # DEBUG: Log LiDAR data details
            self.get_logger().info(f"   └── DEBUG: LiDAR has {num_points} points")
            self.get_logger().info(f"   └── DEBUG: Front distance (ranges[0]) = {ranges[0]:.3f}m")
            if num_points > 100:
                self.get_logger().info(f"   └── DEBUG: ranges[10] = {ranges[10]:.3f}m, ranges[50] = {ranges[50]:.3f}m")
            
            # Create 360° distance profile (every 10° for VLM readability)
            lidar_profile = []
            for i in range(0, 360, 10):  # Every 10 degrees
                if i < num_points:
                    angle_desc = f"{i}°"
                    if 135 <= i <= 225:  # Rear 90° obstructed
                        lidar_profile.append(f"{angle_desc}:obstructed")
                    else:
                        dist = ranges[i] if i < len(ranges) else 0.0
                        lidar_profile.append(f"{angle_desc}:{dist:.1f}m")
                        # DEBUG: Log first few values
                        if i <= 20:
                            self.get_logger().info(f"   └── DEBUG: ranges[{i}] = {dist:.3f}m")
            
            lidar_360 = ", ".join(lidar_profile)
            
            # Extract front distance explicitly for VLM clarity
            front_distance = ranges[0] if len(ranges) > 0 else 0.0
            sensor_info += f", FRONT_DISTANCE: {front_distance:.1f}m, LiDAR_360°: [{lidar_360}]"
            
            # DEBUG: Log what data we're sending to VLM
            self.get_logger().info(f"   └── VLM INPUT DATA: Camera + 360° LiDAR scan")
            self.get_logger().info(f"   └── DEBUG SENSOR_INFO: {sensor_info[:200]}...")  # First 200 chars
            
            # Navigation prompt with 360° LiDAR awareness
            navigation_prompt = f"""You are a robot navigation assistant. Analyze this camera image and 360° LiDAR scan to make a safe navigation decision.

⚠️ CRITICAL: The FRONT_DISTANCE in sensor data shows the exact distance directly ahead. Use ONLY this value for front distance decisions. DO NOT make up or hallucinate any other distance values.

SENSOR DATA: {sensor_info}
TASK: {prompt}

LIDAR EXPLANATION:
- 0° = directly ahead, 90° = left side, 180° = behind, 270° = right side
- 135°-225° = rear blind spot (obstructed)
- Values show distance to nearest obstacle in each direction

CRITICAL MATH CHECK - MEMORIZE THESE FACTS:
- ANY number less than 1.0 means STOP: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
- ANY number greater than 1.0 means SAFE: 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 5.0

EXPLICIT MATH TRAINING - THESE ARE FACTS:
- 0.4 < 1.0 = TRUE (0.4 is LESS than 1.0) → STOP
- 0.8 < 1.0 = TRUE (0.8 is LESS than 1.0) → STOP  
- 1.2 > 1.0 = TRUE (1.2 is GREATER than 1.0) → SAFE
- 2.5 > 2.0 = TRUE (2.5 is GREATER than 2.0) → MOVE FORWARD

WRONG EXAMPLES TO AVOID:
- NEVER say "0.4m > 1.0m" - this is mathematically FALSE
- NEVER say "0.8m > 1.0m" - this is mathematically FALSE
- If you see 0.4m, you MUST say "0.4 < 1.0" and choose STOP

DECISION LOGIC (FOLLOW EXACTLY):
1. ALWAYS use the FRONT_DISTANCE value from sensor data - THIS IS THE DISTANCE DIRECTLY AHEAD
2. If FRONT_DISTANCE < 1.0m → MUST stop (obstacle too close)
3. If FRONT_DISTANCE > 2.0m → MUST move forward (LiDAR is accurate, camera may show distant objects)
4. If FRONT_DISTANCE 1.0m-2.0m → Use camera to decide if path is navigable
5. NEVER make up or hallucinate distance values - ONLY use the provided FRONT_DISTANCE

CRITICAL: LiDAR measures ACTUAL distance to obstacles. Camera shows visual appearance but may make distant objects look closer than they are.

NAVIGATION PHILOSOPHY:
- LiDAR distance > 2.0m = ALWAYS SAFE to move forward (trust the distance sensor)
- Camera is for context only when LiDAR shows 1.0m-2.0m range
- Furniture/objects >2.0m away = SAFE to approach (can navigate around them)
- The goal is navigation, not stopping for distant furniture
- Turns are for exploration and finding clearer paths, not just obstacle avoidance

MECANUM MOVEMENT SCENARIOS:
- Front 1.5m, Left 3.0m → move_forward OR strafe_left OR turn_left (multiple options)
- Front 0.8m, Right 2.5m → strafe_right (slide sideways to avoid obstacle)
- Front 0.6m, Left 2.0m → strafe_left (front blocked, strafe to clear space)
- Front 3.0m → move_forward (clear path ahead)
- Tight hallway with sides at 1.0m → move_forward (can fit)
- All directions < 1.0m → stop (completely surrounded)

MECANUM WHEEL CAPABILITIES:
This robot has mecanum wheels and can:
- Move forward/backward
- Strafe left/right (sideways movement)
- Rotate in place around its center
- Combine movements (diagonal, etc.)

Choose ONE action:
- stop: ONLY if all directions < 1.0m (completely surrounded)
- move_forward: If front (0°) distance > 1.0m
- turn_left: Rotate in place to face left direction (useful for exploration)
- turn_right: Rotate in place to face right direction (useful for exploration)  
- strafe_left: Slide sideways to the left if left (90°) > 1.5m (great for tight spaces)
- strafe_right: Slide sideways to the right if right (270°) > 1.5m (great for tight spaces)

CRITICAL: Your ACTION must match your REASONING. If you say "robot must stop", then ACTION must be "stop".

RESPONSE FORMAT (follow exactly):
1. First, state the front distance: "Front distance is X.Xm"
2. Then, do the math: "Since X.X < 1.0m" OR "X.X > 1.0m" OR "X.X > 2.0m"  
3. Then, state what this means: "the robot must [stop/move_forward/turn_left/etc.]"
4. Finally, choose ACTION that matches step 3

EXAMPLES:
- "Front distance is 0.4m. Since 0.4 < 1.0m, the robot must stop." → ACTION: stop
- "Front distance is 2.5m. Since 2.5 > 2.0m, the robot must move forward." → ACTION: move_forward
- "Front distance is 1.5m. Since 1.5 > 1.0m but camera shows obstacle, the robot must stop." → ACTION: stop

Respond with: ACTION: [action] CONFIDENCE: [0.1-0.9] REASONING: [Follow the 4-step format above exactly]"""
            
            self.get_logger().debug("   └── Running Qwen2-VL inference for navigation...")
            
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
            self.get_logger().debug("   └── Applying chat template...")
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs with speed optimization
            self.get_logger().debug("   └── Processing inputs with Qwen2-VL processor...")
            
            # Resize image for faster processing
            if image.size[0] > self.max_image_size or image.size[1] > self.max_image_size:
                image = image.resize((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
                self.get_logger().debug(f"   └── Resized image to {self.max_image_size}x{self.max_image_size} for speed")
            
            inputs = self.processor(
                text=[text], 
                images=[image], 
                return_tensors="pt"
            )
            
            self.get_logger().debug(f"   └── Input keys: {list(inputs.keys())}")
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with aggressive speed optimizations and timeout
            generation_start = time.time()
            with torch.no_grad():
                # Check if we're already approaching timeout
                if time.time() - start_time > inference_timeout * 0.8:
                    self.get_logger().warn(f"   └── Pre-generation timeout, using sensor fallback")
                    self.get_logger().error("   └── VLM inference timeout - cannot navigate safely")
                    return {
                        'action': 'stop',
                        'confidence': 0.0,
                        'reasoning': 'VLM inference timeout - navigation requires successful analysis'
                    }
                
                # Use torch.compile and optimized settings (only supported parameters)
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,  # Ultra-reduced tokens
                    do_sample=False,  # Deterministic for navigation safety
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=self.use_cache,  # Enable KV caching
                    num_beams=1,  # Greedy search
                    repetition_penalty=1.0,  # No repetition penalty for speed
                    length_penalty=1.0,      # No length penalty for speed
                    no_repeat_ngram_size=0,  # Disable n-gram checking for speed
                    output_scores=False,     # Don't compute scores for speed
                    return_dict_in_generate=False  # Simple output for speed
                )
                
                generation_time = time.time() - generation_start
                self.get_logger().debug(f"   └── Generation completed in {generation_time:.3f}s")
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            response_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # DEBUG: Log the actual VLM response
            self.get_logger().info(f"   └── RAW VLM Response: '{response_text}'")
            self.get_logger().debug(f"   └── Response length: {len(response_text)} characters")
            
            # Parse navigation decision from response
            navigation_result = self._parse_vlm_response(response_text, sensor_data)
            
            # Safety check - ensure we got a valid result
            if navigation_result is None:
                self.get_logger().debug("   └── VLM parsing failed, using sensor-only navigation")
                self.get_logger().error("   └── VLM parsing failed - cannot navigate safely")
                navigation_result = {
                    'action': 'stop',
                    'confidence': 0.0,
                    'reasoning': 'VLM parsing failed - navigation requires successful analysis'
                }
            
            # Mathematical validation - catch VLM math errors using LiDAR data (same as VLM sees)
            lidar_front_distance = 999.0
            if lidar_data and 'ranges' in lidar_data and len(lidar_data['ranges']) > 0:
                ranges = lidar_data['ranges']
                # Front is at index 0 in LiDAR scan
                lidar_front_distance = ranges[0] if len(ranges) > 0 else 999.0
            
            if navigation_result['action'] == 'move_forward' and lidar_front_distance < 1.0:
                self.get_logger().error(f"   └── VLM MATH ERROR DETECTED: Wants to move forward with LiDAR front distance {lidar_front_distance:.1f}m < 1.0m")
                self.get_logger().error(f"   └── OVERRIDING VLM decision for safety")
                self.get_logger().error(f"   └── Overriding dangerous VLM decision with STOP")
                navigation_result = {
                    'action': 'stop',
                    'confidence': 0.9,
                    'reasoning': f'Safety override: VLM wanted to move forward but obstacle at {lidar_front_distance:.1f}m < 1.0m'
                }
            
            # Log timing performance
            total_time = time.time() - start_time
            self.get_logger().debug(f"   └── VLM decision: {navigation_result['action']} (confidence: {navigation_result['confidence']:.2f}) in {total_time:.3f}s")
            
            # Add timing to result
            navigation_result['inference_time'] = total_time
            
            return navigation_result
                
        except Exception as e:
            import traceback
            error_msg = f"VLM inference failed: {e}"
            full_traceback = traceback.format_exc()
            self.get_logger().error(f"   └── {error_msg}")
            self.get_logger().error(f"   └── Full traceback: {full_traceback}")
            self.get_logger().error(f"   └── VLM inference failed - cannot navigate safely")
            return {
                'action': 'stop',
                'confidence': 0.0,
                'reasoning': f'VLM inference failed: {error_msg} - navigation requires successful analysis'
            }
    
    def _parse_vlm_response(self, response_text: str, sensor_data: Dict[str, float]) -> Dict[str, Any]:
        """Parse VLM response to extract navigation decision"""
        try:
            # DEBUG: Log parsing details
            self.get_logger().debug(f"   └── PARSING: Input text: '{response_text}'")
            response_upper = response_text.upper()
            self.get_logger().debug(f"   └── PARSING: Uppercase: '{response_upper}'")
            
            # Extract action
            action = "stop"  # Default safe action
            if "ACTION:" in response_upper:
                action_part = response_upper.split("ACTION:")[1].split("|")[0].strip()
                self.get_logger().debug(f"   └── PARSING: Found ACTION: '{action_part}'")
                if "MOVE_FORWARD" in action_part:
                    action = "move_forward"
                elif "TURN_LEFT" in action_part:
                    action = "turn_left"
                elif "TURN_RIGHT" in action_part:
                    action = "turn_right"
                elif "STRAFE_LEFT" in action_part:
                    action = "strafe_left"
                elif "STRAFE_RIGHT" in action_part:
                    action = "strafe_right"
                elif "STOP" in action_part:
                    action = "stop"
            else:
                # Try simple keyword detection
                self.get_logger().info("   └── PARSING: No ACTION: found, trying keyword detection")
                if "MOVE_FORWARD" in response_upper or "FORWARD" in response_upper:
                    action = "move_forward"
                elif "STRAFE_LEFT" in response_upper:
                    action = "strafe_left"
                elif "STRAFE_RIGHT" in response_upper:
                    action = "strafe_right"
                elif "TURN_LEFT" in response_upper or "LEFT" in response_upper:
                    action = "turn_left"
                elif "TURN_RIGHT" in response_upper or "RIGHT" in response_upper:
                    action = "turn_right"
                elif "STOP" in response_upper:
                    action = "stop"
            
            self.get_logger().debug(f"   └── PARSING: Final action: '{action}'")
            
            # Extract confidence
            confidence = 0.5  # Default confidence
            if "CONFIDENCE:" in response_upper:
                try:
                    conf_part = response_upper.split("CONFIDENCE:")[1].strip()
                    # Extract just the number (handle spaces and other text)
                    import re
                    conf_match = re.search(r'(\d+\.?\d*)', conf_part)
                    if conf_match:
                        confidence = float(conf_match.group(1))
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                        self.get_logger().debug(f"   └── PARSING: Extracted confidence: {confidence}")
                except Exception as e:
                    self.get_logger().debug(f"   └── PARSING: Confidence extraction failed: {e}")
                    confidence = 0.5
            
            # Extract reasoning
            reasoning = "VLM navigation decision"
            if "REASONING:" in response_upper:
                reasoning = response_text.split("REASONING:")[1].split("|")[0].strip()
            elif "REASON:" in response_upper:
                reasoning = response_text.split("REASON:")[1].split("|")[0].strip()
            
            # Check for reasoning-action consistency
            reasoning_lower = reasoning.lower()
            if action == 'move_forward' and ('must stop' in reasoning_lower or 'should stop' in reasoning_lower):
                self.get_logger().error(f"   └── REASONING-ACTION MISMATCH: Says 'must stop' but ACTION is 'move_forward'")
                self.get_logger().error(f"   └── Correcting action to match reasoning")
                action = 'stop'
                confidence = max(0.8, confidence)  # High confidence in safety correction
            elif action == 'stop' and ('must move' in reasoning_lower or 'should move' in reasoning_lower):
                self.get_logger().warn(f"   └── REASONING-ACTION MISMATCH: Says 'must move' but ACTION is 'stop'")
                # Keep stop for safety - don't override stop with movement
            
            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'vlm_response': response_text
            }
            
        except Exception as e:
            self.get_logger().warn(f"Failed to parse VLM response (response_length={len(response_text)}): {e}")
            self.get_logger().warn(f"Response was: '{response_text[:200]}{'...' if len(response_text) > 200 else ''}'")
            self.get_logger().error(f"   └── VLM response parsing failed - cannot navigate safely")
            return {
                'action': 'stop',
                'confidence': 0.0,
                'reasoning': f'VLM response parsing failed: {e} - navigation requires successful analysis'
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
