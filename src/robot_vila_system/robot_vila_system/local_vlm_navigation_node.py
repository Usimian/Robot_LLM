#!/usr/bin/env python3
"""
RoboMP2-Enhanced VLM Navigation Node

This ROS2 node integrates RoboMP2 framework components:
- Goal-Conditioned Multimodal Perceptor (GCMP) for environment state understanding
- Retrieval-Augmented Multimodal Planner (RAMP) for enhanced planning capabilities
- Uses local VLM (Qwen2.5-VL-7B-Instruct) running on RTX 3090

Author: Robot LLM System with RoboMP2 Integration
"""

import json
import threading
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from collections import deque
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import transformers
transformers.logging.set_verbosity_error()  # Suppress generation warnings
import cv2

# Import RoboMP2 components
from .robomp2_components import GoalConditionedMultimodalPerceptor, RetrievalAugmentedMultimodalPlanner

# Import configuration
from .gui_config import GUIConfig

# RoboMP2 Data Structures
@dataclass
class EnvironmentState:
    """Represents the current environment state for GCMP"""
    visual_features: Dict[str, Any]
    spatial_features: Dict[str, Any]
    temporal_features: Dict[str, Any]
    goal_context: Optional[str]
    timestamp: float
    state_hash: str = ""
    
    def __post_init__(self):
        """Generate hash for state comparison"""
        state_data = {
            'visual': str(self.visual_features),
            'spatial': str(self.spatial_features),
            'goal': self.goal_context
        }
        self.state_hash = hashlib.md5(str(state_data).encode()).hexdigest()[:8]

@dataclass
class PolicyEntry:
    """Represents a policy entry in the RAMP database"""
    policy_id: str
    state_signature: str
    goal_type: str
    action_sequence: List[str]
    success_rate: float
    context_description: str
    usage_count: int = 0
    last_used: float = 0.0
    
    def update_usage(self):
        """Update usage statistics"""
        self.usage_count += 1
        self.last_used = time.time()

@dataclass
class GoalSpecification:
    """Represents a goal for the robot to achieve"""
    goal_id: str
    goal_type: str  # "navigation", "manipulation", "exploration", "interaction"
    description: str
    target_object: Optional[str] = None
    target_location: Optional[Dict[str, float]] = None
    constraints: Optional[Dict[str, Any]] = None
    priority: float = 1.0
    timeout: float = 60.0

class RoboMP2NavigationNode(Node):
    """ROS2 Node for RoboMP2-enhanced VLM navigation with GCMP and RAMP"""
    
    def __init__(self):
        super().__init__('robomp2_navigation_node')
        
        # Configuration - Optimized for speed
        self.model_name = GUIConfig.DEFAULT_VLM_MODEL
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
        self.previous_ranges = None  # For LiDAR smoothing to reduce flickering
        self.camera_lock = threading.Lock()
        self.sensor_data_lock = threading.Lock()
        self.lidar_lock = threading.Lock()
        
        # Request management - prevent multiple simultaneous analysis requests
        self.analysis_in_progress = False
        self.analysis_lock = threading.Lock()
        self.request_queue = []
        self.max_queue_size = 1  # Only keep the most recent request
        
        # RoboMP2 Components
        self.gcmp = None  # Goal-Conditioned Multimodal Perceptor (initialized after logger)
        self.ramp = None  # Retrieval-Augmented Multimodal Planner (initialized after logger)
        self.current_goal: Optional[GoalSpecification] = None
        self.goal_lock = threading.Lock()
        self.active_policies: List[PolicyEntry] = []
        self.policy_execution_history = deque(maxlen=50)
        
        # Initialize ROS2 components
        self._setup_ros2_components()
        
        # Initialize RoboMP2 components
        self._initialize_robomp2_components()
        
        # Start model loading in background
        self._start_model_loading()
        
        self.get_logger().info("🤖 RoboMP2 Navigation Node initialized with GCMP and RAMP")
    
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
        
        # Filtered LiDAR publisher for stable display
        self.filtered_lidar_publisher = self.create_publisher(
            LaserScan,
            '/scan_filtered',
            self.sensor_qos
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
        
        # RoboMP2 Services
        self.goal_service = self.create_service(
            ExecuteCommand,
            '/robomp2/set_goal',
            self._goal_service_callback
        )
        
        self.policy_service = self.create_service(
            ExecuteCommand,
            '/robomp2/add_policy',
            self._policy_service_callback
        )
        
        self.get_logger().info("📡 ROS2 components configured")
        self.get_logger().info("   └── Publishers: /vlm/analysis, /vlm/status")
        self.get_logger().info("   └── Subscribers: /robot/sensors, /realsense/camera/color/image_raw, /scan")
        self.get_logger().info("   └── Services: /vlm/analyze_scene, /robomp2/set_goal, /robomp2/add_policy")
    
    def _initialize_robomp2_components(self):
        """Initialize RoboMP2 GCMP and RAMP components"""
        try:
            # Initialize Goal-Conditioned Multimodal Perceptor
            self.gcmp = GoalConditionedMultimodalPerceptor(self.get_logger())
            self.get_logger().info("🎯 GCMP (Goal-Conditioned Multimodal Perceptor) initialized")
            
            # Initialize Retrieval-Augmented Multimodal Planner
            policy_db_path = str(Path.home() / "Robot_LLM" / "robomp2_policies.pkl")
            self.ramp = RetrievalAugmentedMultimodalPlanner(self.get_logger(), policy_db_path)
            self.get_logger().info("🧠 RAMP (Retrieval-Augmented Multimodal Planner) initialized")
            
            self.get_logger().info("✅ RoboMP2 components ready for goal-conditioned navigation")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to initialize RoboMP2 components: {e}")
            # Continue without RoboMP2 - fallback to basic VLM navigation
            self.gcmp = None
            self.ramp = None
    
    def _start_model_loading(self):
        """Start model loading in background thread"""
        def load_model():
            try:
                self.get_logger().info(f"🔄 Loading {self.model_name.split('/')[-1]} model for navigation...")
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
                self.get_logger().info("   └── Loading Qwen2.5-VL model with speed optimization...")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
                self.get_logger().info(f"✅ {self.model_name.split('/')[-1]} loaded successfully for navigation")
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
        """Handle incoming LiDAR scans with improved filtering"""
        try:
            with self.lidar_lock:
                # Store both processed data and cleaned ranges for VLM analysis
                processed_data = self._process_lidar_scan(msg)
                
                # Use processed ranges instead of raw ranges to prevent inf values
                if hasattr(self, 'previous_ranges') and self.previous_ranges is not None:
                    processed_data['ranges'] = list(self.previous_ranges)
                    
                    # Publish filtered LiDAR data for GUI display
                    filtered_msg = LaserScan()
                    filtered_msg.header = msg.header
                    filtered_msg.angle_min = msg.angle_min
                    filtered_msg.angle_max = msg.angle_max
                    filtered_msg.angle_increment = msg.angle_increment
                    filtered_msg.time_increment = msg.time_increment
                    filtered_msg.scan_time = msg.scan_time
                    filtered_msg.range_min = msg.range_min
                    filtered_msg.range_max = msg.range_max
                    filtered_msg.ranges = list(self.previous_ranges)
                    filtered_msg.intensities = msg.intensities
                    
                    self.filtered_lidar_publisher.publish(filtered_msg)
                else:
                    # Fallback: clean raw ranges
                    clean_ranges = np.array(msg.ranges)
                    clean_ranges = np.where(np.isfinite(clean_ranges), clean_ranges, msg.range_max)
                    processed_data['ranges'] = list(clean_ranges)
                
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
        """Process LiDAR scan with advanced filtering to eliminate display flickering"""
        ranges = np.array(msg.ranges, dtype=np.float64)
        
        # Step 1: Handle infinite and NaN values aggressively
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        
        # Step 2: Clamp to sensor limits
        ranges = np.where(ranges > msg.range_max, msg.range_max, ranges)
        ranges = np.where(ranges < msg.range_min, msg.range_min, ranges)
        
        # Step 3: Additional range validation to prevent extreme values
        ranges = np.where(ranges > 10.0, 10.0, ranges)  # Cap at 10m
        ranges = np.where(ranges < 0.05, 0.05, ranges)  # Minimum 5cm
        
        # Step 4: Advanced temporal smoothing with outlier detection
        if hasattr(self, 'previous_ranges') and self.previous_ranges is not None and len(self.previous_ranges) == len(ranges):
            alpha = 0.2  # More conservative smoothing
            
            # Calculate change rate to detect outliers
            change_rate = np.abs(ranges - self.previous_ranges)
            outlier_threshold = 2.0  # 2m change threshold
            
            # Apply smoothing, but preserve rapid changes for real obstacles
            smooth_mask = change_rate < outlier_threshold
            ranges = np.where(smooth_mask, 
                             alpha * ranges + (1 - alpha) * self.previous_ranges,
                             ranges)  # Keep rapid changes as-is
        
        # Step 5: Median filtering for noise reduction (only for large scans)
        if len(ranges) > 5:
            filtered_ranges = np.copy(ranges)
            for i in range(2, len(ranges) - 2):
                window = ranges[i-2:i+3]
                filtered_ranges[i] = np.median(window)
            ranges = filtered_ranges
        
        # Store for next iteration
        self.previous_ranges = ranges.copy()
        
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
        """Run RoboMP2-enhanced VLM inference for navigation decisions"""
        start_time = time.time()
        inference_timeout = 2.0  # 2-second timeout for ultra-fast response
        try:
            # RoboMP2 Enhancement: Extract environment state using GCMP
            current_state = None
            relevant_policies = []
            goal_enhanced_prompt = prompt
            
            if self.gcmp and self.ramp and image and lidar_data:
                try:
                    # Extract current environment state using GCMP
                    with self.goal_lock:
                        current_goal = self.current_goal
                    
                    current_state = self.gcmp.extract_state_features(
                        image, lidar_data, sensor_data, current_goal, EnvironmentState
                    )
                    
                    # Retrieve relevant policies using RAMP
                    relevant_policies = self.ramp.retrieve_relevant_policies(
                        current_state, current_goal, top_k=3
                    )
                    
                    # Enhance prompt with goal context and retrieved policies
                    if current_goal:
                        goal_enhanced_prompt = f"GOAL: {current_goal.description}\n{prompt}"
                    
                    if relevant_policies:
                        policy_context = "RELEVANT POLICIES:\n"
                        for i, policy in enumerate(relevant_policies, 1):
                            if hasattr(policy, 'context_description') and hasattr(policy, 'action_sequence'):
                                policy_context += f"{i}. {policy.context_description}: {' -> '.join(policy.action_sequence)}\n"
                        goal_enhanced_prompt = f"{policy_context}\n{goal_enhanced_prompt}"
                    
                    self.get_logger().debug(f"🎯 RoboMP2: State extracted, {len(relevant_policies)} policies retrieved")
                    
                except Exception as e:
                    self.get_logger().warn(f"RoboMP2 enhancement failed, falling back to basic VLM: {e}")
                    # Continue with basic VLM inference
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
            raw_ranges = lidar_data['ranges']
            ranges = np.array(raw_ranges)
            
            # Clean up infinite and invalid values for stable display
            ranges = np.where(np.isfinite(ranges), ranges, 10.0)  # Replace inf with reasonable max
            ranges = np.where(ranges > 10.0, 10.0, ranges)  # Cap at 10m
            ranges = np.where(ranges < 0.1, 0.1, ranges)   # Minimum 10cm
            
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
3. If FRONT_DISTANCE > 2.0m → MUST move_forward (path is clear, safe to proceed)
4. If FRONT_DISTANCE 1.0m-2.0m → Use camera to decide if path is navigable
5. NEVER make up or hallucinate distance values - ONLY use the provided FRONT_DISTANCE

EXPLICIT DECISION EXAMPLES (MEMORIZE THESE):
- FRONT_DISTANCE: 0.5m → ACTION: stop (0.5 < 1.0 = obstacle too close)
- FRONT_DISTANCE: 0.8m → ACTION: stop (0.8 < 1.0 = obstacle too close)  
- FRONT_DISTANCE: 1.5m → ACTION: move_forward or turn (1.0-2.0 range, check camera)
- FRONT_DISTANCE: 2.5m → ACTION: move_forward (2.5 > 2.0 = clear path ahead)
- FRONT_DISTANCE: 3.2m → ACTION: move_forward (3.2 > 2.0 = clear path ahead)
- FRONT_DISTANCE: 5.0m → ACTION: move_forward (5.0 > 2.0 = clear path ahead)

CRITICAL SAFETY RULE: LARGE DISTANCES = SAFE TO MOVE FORWARD!

❌ WRONG LOGIC TO AVOID:
- "3.2m > 1.0m, so stop" ← THIS IS BACKWARDS! 3.2m means CLEAR PATH!
- "Distance is large, so dangerous" ← NO! Large distance = SAFE!

✅ CORRECT LOGIC:
- "3.2m > 2.0m, so move_forward" ← CORRECT! Large distance = SAFE!
- "0.5m < 1.0m, so stop" ← CORRECT! Small distance = DANGEROUS!

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
            
            self.get_logger().debug("   └── Running Qwen2.5-VL inference for navigation...")
            
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
            self.get_logger().debug("   └── Processing inputs with Qwen2.5-VL processor...")
            
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
            
            # COMPREHENSIVE VLM LOGIC VALIDATION - catch and correct reasoning errors
            lidar_front_distance = 999.0
            if lidar_data and 'ranges' in lidar_data and len(lidar_data['ranges']) > 0:
                ranges = lidar_data['ranges']
                # Front is at index 0 in LiDAR scan
                lidar_front_distance = ranges[0] if len(ranges) > 0 else 999.0
            
            # Check for dangerous move_forward commands
            if navigation_result['action'] == 'move_forward' and lidar_front_distance < 1.0:
                self.get_logger().error(f"   └── VLM MATH ERROR DETECTED: Wants to move forward with LiDAR front distance {lidar_front_distance:.1f}m < 1.0m")
                self.get_logger().error(f"   └── OVERRIDING VLM decision for safety")
                self.get_logger().error(f"   └── Overriding dangerous VLM decision with STOP")
                navigation_result = {
                    'action': 'stop',
                    'confidence': 0.9,
                    'reasoning': f'SAFETY OVERRIDE: VLM wanted move_forward but front distance {lidar_front_distance:.1f}m < 1.0m is too close'
                }
            
            # Check for overly cautious stop commands when path is clear
            elif navigation_result['action'] == 'stop' and lidar_front_distance > 2.5:
                self.get_logger().warning(f"   └── VLM LOGIC ERROR DETECTED: Wants to stop with clear front distance {lidar_front_distance:.1f}m > 2.5m")
                self.get_logger().warning(f"   └── CORRECTING overly cautious VLM decision")
                self.get_logger().warning(f"   └── Overriding stop with MOVE_FORWARD (path is clear)")
                navigation_result = {
                    'action': 'move_forward',
                    'confidence': 0.8,
                    'reasoning': f'LOGIC CORRECTION: VLM wanted stop but front distance {lidar_front_distance:.1f}m > 2.5m is safe to proceed'
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
    
    def _goal_service_callback(self, request, response):
        """Handle RoboMP2 goal setting requests"""
        try:
            # Parse goal from request
            goal_data = request.command.source_node  # Goal data passed in source_node field
            
            if '|' in goal_data:
                goal_parts = goal_data.split('|')
                goal_type = goal_parts[0] if len(goal_parts) > 0 else "navigation"
                goal_description = goal_parts[1] if len(goal_parts) > 1 else "Navigate safely"
                target_object = goal_parts[2] if len(goal_parts) > 2 else None
            else:
                goal_type = "navigation"
                goal_description = goal_data
                target_object = None
            
            # Create goal specification
            new_goal = GoalSpecification(
                goal_id=f"goal_{int(time.time())}",
                goal_type=goal_type,
                description=goal_description,
                target_object=target_object,
                priority=1.0,
                timeout=60.0
            )
            
            # Set current goal
            with self.goal_lock:
                self.current_goal = new_goal
            
            self.get_logger().info(f"🎯 Goal set: {goal_type} - {goal_description}")
            if target_object:
                self.get_logger().info(f"   └── Target: {target_object}")
            
            response.success = True
            response.result_message = f"Goal set: {goal_description}"
            
        except Exception as e:
            self.get_logger().error(f"Failed to set goal: {e}")
            response.success = False
            response.result_message = str(e)
        
        return response
    
    def _policy_service_callback(self, request, response):
        """Handle RoboMP2 policy addition requests"""
        try:
            if not self.ramp:
                response.success = False
                response.result_message = "RAMP not initialized"
                return response
            
            # Parse policy from request
            policy_data = request.command.source_node  # Policy data passed in source_node field
            
            if '|' in policy_data:
                parts = policy_data.split('|')
                policy_id = parts[0] if len(parts) > 0 else f"policy_{int(time.time())}"
                state_signature = parts[1] if len(parts) > 1 else "unknown"
                goal_type = parts[2] if len(parts) > 2 else "navigation"
                actions_str = parts[3] if len(parts) > 3 else "stop"
                description = parts[4] if len(parts) > 4 else "User-defined policy"
                
                action_sequence = actions_str.split(',') if ',' in actions_str else [actions_str]
                
                # Create policy entry
                new_policy = PolicyEntry(
                    policy_id=policy_id,
                    state_signature=state_signature,
                    goal_type=goal_type,
                    action_sequence=action_sequence,
                    success_rate=0.5,  # Default success rate
                    context_description=description
                )
                
                # Add to database
                self.ramp.add_policy(new_policy)
                
                self.get_logger().info(f"➕ Policy added: {policy_id}")
                response.success = True
                response.result_message = f"Policy added: {policy_id}"
            else:
                response.success = False
                response.result_message = "Invalid policy format. Use: policy_id|state_signature|goal_type|actions|description"
            
        except Exception as e:
            self.get_logger().error(f"Failed to add policy: {e}")
            response.success = False
            response.result_message = str(e)
        
        return response


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = RoboMP2NavigationNode()
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
