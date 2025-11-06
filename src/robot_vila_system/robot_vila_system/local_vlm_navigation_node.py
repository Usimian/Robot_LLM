#!/usr/bin/env python3
"""
RoboMP2-Enhanced VLM Navigation Node

This ROS2 node integrates RoboMP2 framework components:
- Goal-Conditioned Multimodal Perceptor (GCMP) for environment state understanding
- Retrieval-Augmented Multimodal Planner (RAMP) for enhanced planning capabilities
- Uses local VLM for vision-based navigation (model configured in GUIConfig)

Author: Robot LLM System with RoboMP2 Integration
"""

import json
import threading
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
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
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import transformers
transformers.logging.set_verbosity_error()  # Suppress generation warnings

# Import RoboMP2 components
from .robomp2_components import GoalConditionedMultimodalPerceptor, RetrievalAugmentedMultimodalPlanner

# Import NLP command parser
from .nlp_command_parser import NLPCommandParser

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
        self.max_new_tokens = 512  # Further increased to ensure complete VLM responses
        self.use_cache = True      # Enable KV cache for faster inference
        self.skip_chat_template = True  # Skip expensive chat template processing
        
        # Model components (initialized later)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_loaded = False

        # NLP command parser (initialized later)
        self.nlp_parser = None
        self.nlp_parser_enabled = GUIConfig.ENABLE_NLP_PARSER
        self.nlp_model_name = GUIConfig.DEFAULT_NLP_MODEL
        
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

        # Initialize NLP command parser
        self._initialize_nlp_parser()

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
        
        # Service clients
        self.robot_service_client = self.create_client(
            ExecuteCommand,
            '/robot/execute_command'
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
            self.get_logger().error("❌ RoboMP2 is required for navigation - cannot proceed")
            raise RuntimeError(f"RoboMP2 initialization failed: {e}")

    def _initialize_nlp_parser(self):
        """Initialize NLP command parser for natural language understanding"""
        if not self.nlp_parser_enabled:
            self.get_logger().info("🔇 NLP command parser disabled via configuration")
            return

        try:
            self.get_logger().info(f"🔄 Initializing NLP command parser: {self.nlp_model_name}")

            # Create NLP parser instance
            self.nlp_parser = NLPCommandParser(
                model_name=self.nlp_model_name,
                device=self.device,
                logger=self.get_logger()
            )

            # Load model in background thread to avoid blocking
            def load_nlp_model():
                try:
                    self.nlp_parser.load_model()
                    self.get_logger().info("✅ NLP command parser ready for natural language understanding")
                except Exception as e:
                    self.get_logger().error(f"❌ Failed to load NLP parser model: {e}")
                    self.get_logger().warn("   └── Falling back to keyword-based command parsing")
                    self.nlp_parser = None

            nlp_loading_thread = threading.Thread(target=load_nlp_model, daemon=True)
            nlp_loading_thread.start()

        except Exception as e:
            self.get_logger().error(f"❌ Failed to initialize NLP parser: {e}")
            self.get_logger().warn("   └── Continuing with keyword-based command parsing")
            self.nlp_parser = None

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
                model_short_name = self.model_name.split('/')[-1]
                self.get_logger().info(f"   └── Loading {model_short_name} model with speed optimization...")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    dtype=torch.float16,        # Use FP16 for speed and memory efficiency (fixed deprecated torch_dtype)
                    device_map="cuda:0",        # Force GPU placement to avoid meta device issues
                    trust_remote_code=True,
                    low_cpu_mem_usage=True      # Reduce CPU memory usage during loading
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
                # Store RAW LiDAR message for VLM analysis (unfiltered)
                self.latest_raw_lidar_msg = msg

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
        
        # Step 4: Light temporal smoothing for noise only (NOT for obstacle detection)
        # WARNING: Heavy smoothing is DANGEROUS - it makes obstacles appear farther than they are!
        if hasattr(self, 'previous_ranges') and self.previous_ranges is not None and len(self.previous_ranges) == len(ranges):
            alpha = 0.9  # Light smoothing - trust new data (90% new, 10% old)
            
            # Only smooth when change is very small (likely noise, not real obstacle movement)
            change_rate = np.abs(ranges - self.previous_ranges)
            noise_threshold = 0.05  # Only smooth changes < 5cm (sensor noise)
            
            # Apply minimal smoothing ONLY to noise, preserve all real changes
            smooth_mask = change_rate < noise_threshold
            ranges = np.where(smooth_mask, 
                             alpha * ranges + (1 - alpha) * self.previous_ranges,
                             ranges)  # Keep all real changes as-is
        
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

            try:
                # Get current multimodal data
                current_image = self._get_current_camera_image()
                sensor_data = self._get_current_sensor_data()
                lidar_data = self._get_current_lidar_data()
                
                # Extract custom prompt from request if provided
                # Prompt is encoded in command_type as "vlm_analysis:PROMPT"
                prompt = "Analyze the current scene for robot navigation"  # Default
                if hasattr(request, 'command') and hasattr(request.command, 'command_type'):
                    cmd_type = request.command.command_type
                    if ':' in cmd_type:
                        # Extract prompt after the colon
                        parts = cmd_type.split(':', 1)
                        if len(parts) == 2 and parts[0] == 'vlm_analysis':
                            prompt = parts[1].strip()
                            self.get_logger().info(f"📝 Using custom prompt: {prompt}")
            
                # Run VLM inference with RAW LiDAR data (not processed/filtered)
                navigation_result = self._run_vlm_navigation_inference(current_image, prompt, sensor_data, lidar_data)

                # Check if this is an informational query response
                is_informational = navigation_result.get('is_informational', False)

                # Format response with enhanced reasoning display
                if is_informational:
                    # For informational queries, format as description
                    analysis_result = {
                        'success': True,
                        'analysis': navigation_result['reasoning'],
                        'reasoning': navigation_result['reasoning'],
                        'full_analysis': f"VLM Description:\n{navigation_result['reasoning']}",
                        'navigation_commands': {
                            'action': 'none',
                            'confidence': 1.0
                        },
                        'confidence': 1.0,
                        'is_informational': True
                    }
                else:
                    # For navigation queries, format as navigation decision
                    analysis_result = {
                        'success': True,
                        'analysis': f"VLM Decision: {navigation_result['action']} (confidence: {navigation_result['confidence']:.2f})",
                        'reasoning': navigation_result['reasoning'],
                        'full_analysis': f"Local VLM Analysis:\n• Action: {navigation_result['action']}\n• Confidence: {navigation_result['confidence']:.2f}\n• Reasoning: {navigation_result['reasoning']}\n• Raw Response: {navigation_result.get('vlm_response', 'N/A')}",
                        'navigation_commands': {
                            'action': navigation_result['action'],
                            'confidence': navigation_result['confidence']
                        },
                        'confidence': navigation_result['confidence'],
                        'is_informational': False
                    }

                # Pack response with enhanced reasoning
                result_data = {
                    'analysis': analysis_result['analysis'],
                    'reasoning': analysis_result['reasoning'],
                    'full_analysis': analysis_result['full_analysis'],
                    'navigation_commands': analysis_result['navigation_commands'],
                    'confidence': analysis_result['confidence'],
                    'is_informational': is_informational,
                    'parameters': navigation_result.get('parameters', {})  # Include parsed parameters if present
                }
                
                response.success = True
                response.result_message = json.dumps(result_data)
            
                # Publish analysis results
                command_msg = RobotCommand()
                command_msg.command_type = "vlm_analysis"

                self.analysis_publisher.publish(command_msg)

                # VLM analysis complete - execution will be handled by GUI if enabled
                action = navigation_result['action']
                confidence = navigation_result['confidence']
                
                self.get_logger().info(f"📋 VLM ANALYSIS COMPLETE: {action} (confidence: {confidence:.2f})")
                self.get_logger().info(f"🤖 MODEL REASONING: {navigation_result['reasoning']}")
                self.get_logger().info("   └── Execution control handled by GUI - not auto-executing")
            
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
    
    def _parse_movement_parameters(self, prompt: str, command_type: str) -> Dict[str, Any]:
        """Parse movement parameters (distance, angle) from command string

        Args:
            prompt: Command string (e.g., "turn right 90 deg", "move forward 2m")
            command_type: Command type (e.g., 'turn_right', 'move_forward')

        Returns:
            Dictionary with parsed parameters (distance, angle, speed)
        """
        import re

        params = {}
        prompt_lower = prompt.lower()

        # Parse distance for linear movements (move_forward, move_backward, strafe)
        if command_type in ['move_forward', 'move_backward', 'strafe_left', 'strafe_right']:
            # Look for patterns like: "2m", "1.5 meters", "0.5 m", "50cm", "50 cm"
            distance_patterns = [
                r'(\d+\.?\d*)\s*m(?:eters?)?(?:\s|$)',  # 2m, 2 meters, 1.5m
                r'(\d+\.?\d*)\s*cm(?:entimeters?)?(?:\s|$)',  # 50cm, 50 centimeters
            ]

            for pattern in distance_patterns:
                match = re.search(pattern, prompt_lower)
                if match:
                    value = float(match.group(1))
                    # Convert cm to meters
                    if 'cm' in pattern:
                        value = value / 100.0
                    params['distance'] = value
                    self.get_logger().info(f"   └── Parsed distance: {value:.2f}m")
                    break

        # Parse angle for turns (turn_left, turn_right)
        elif command_type in ['turn_left', 'turn_right']:
            # Look for patterns like: "90 deg", "45 degrees", "180°", "1.57 rad"
            angle_patterns = [
                r'(\d+\.?\d*)\s*(?:deg|degree)s?(?:\s|$)',  # 90 deg, 45 degrees
                r'(\d+\.?\d*)\s*°',  # 90°
                r'(\d+\.?\d*)\s*rad(?:ians?)?(?:\s|$)',  # 1.57 rad, 1.57 radians
            ]

            for pattern in angle_patterns:
                match = re.search(pattern, prompt_lower)
                if match:
                    value = float(match.group(1))
                    # Convert to radians if needed
                    if 'rad' in pattern:
                        angle_rad = value
                    else:
                        # Degrees to radians
                        import math
                        angle_rad = math.radians(value)
                    params['angle'] = angle_rad
                    self.get_logger().info(f"   └── Parsed angle: {value}° = {angle_rad:.3f} rad")
                    break

        # Parse speed if specified (optional for all movement types)
        speed_patterns = [
            r'(\d+\.?\d*)\s*m/s(?:\s|$)',  # 0.5 m/s
            r'at\s+(\d+\.?\d*)\s*(?:m/s|meters?\s+per\s+second)',  # at 0.5 m/s
        ]

        for pattern in speed_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                params['speed'] = float(match.group(1))
                self.get_logger().info(f"   └── Parsed speed: {params['speed']:.2f} m/s")
                break

        return params

    def _is_direct_command(self, prompt: str) -> tuple[bool, str, Dict[str, Any]]:
        """Check if prompt is a direct movement command (e.g., 'move forward', 'turn left')

        Returns:
            (is_direct_command, command_type, parameters)
            e.g., (True, 'move_forward', {'distance': 2.0})
        """
        prompt_lower = prompt.lower().strip()

        # Direct command mappings - ordered from most specific to least specific
        # This prevents "right" from matching "turn right 90 deg"
        direct_commands = [
            ('move forward', 'move_forward'),
            ('go forward', 'move_forward'),
            ('move backward', 'move_backward'),
            ('move back', 'move_backward'),
            ('go back', 'move_backward'),
            ('turn left', 'turn_left'),
            ('rotate left', 'turn_left'),
            ('turn right', 'turn_right'),
            ('rotate right', 'turn_right'),
            ('strafe left', 'strafe_left'),
            ('strafe right', 'strafe_right'),
            # Single word commands only match if they're the ENTIRE prompt (prevent false matches)
            ('forward', 'move_forward'),
            ('backward', 'move_backward'),
            ('reverse', 'move_backward'),
            ('left', 'turn_left'),
            ('right', 'turn_right'),
            ('stop', 'stop'),
            ('halt', 'stop')
        ]

        for phrase, command in direct_commands:
            # For single-word commands, only match if it's the entire prompt
            if len(phrase.split()) == 1:
                if prompt_lower == phrase:
                    params = self._parse_movement_parameters(prompt, command)
                    return True, command, params
            else:
                # For multi-word commands, match if phrase is in prompt
                if phrase in prompt_lower:
                    params = self._parse_movement_parameters(prompt, command)
                    return True, command, params

        return False, None, {}

    def _is_informational_query(self, prompt: str) -> bool:
        """Detect if the query is informational (description) vs navigational (movement)

        Strategy: Default to informational UNLESS explicit movement/navigation keywords are found.
        This is safer and more intuitive - user must explicitly request movement.
        """
        # First check if it's a direct command
        is_direct, _, _ = self._is_direct_command(prompt)
        if is_direct:
            return False  # Direct commands are not informational

        # Navigation/movement keywords for goal-based navigation (e.g., "navigate to the door")
        navigation_keywords = [
            'navigate to', 'go to', 'drive to', 'travel to', 'head to',
            'approach', 'reach', 'get to'
        ]

        prompt_lower = prompt.lower()

        # If ANY navigation keyword is found, it's a navigation query
        has_navigation = any(keyword in prompt_lower for keyword in navigation_keywords)

        if has_navigation:
            self.get_logger().debug(f"   └── Navigation keyword detected in: '{prompt}'")
            return False  # Not informational - it's navigation
        else:
            self.get_logger().debug(f"   └── No navigation keywords - treating as informational: '{prompt}'")
            return True   # Default to informational

    def _run_vlm_navigation_inference(self, image: Optional[Image.Image], prompt: str, sensor_data: Dict[str, float], lidar_data: Optional[Dict[str, Any]] = None, raw_lidar_msg: Optional[object] = None) -> Dict[str, Any]:
        """Run RoboMP2-enhanced VLM inference for navigation decisions"""
        start_time = time.time()
        inference_timeout = 2.0  # 2-second timeout for ultra-fast response

        # NEW: Try NLP parser first for natural language understanding (if enabled and loaded)
        if self.nlp_parser is not None and hasattr(self.nlp_parser, 'model_loaded') and self.nlp_parser.model_loaded:
            try:
                self.get_logger().info(f"🧠 Using NLP parser for: '{prompt}'")
                parsed_cmd = self.nlp_parser.parse_command(prompt)

                # If command doesn't need vision, execute directly
                if not parsed_cmd.needs_vision:
                    self.get_logger().info(f"   └── NLP parsed to {parsed_cmd.action} (no vision needed)")
                    return {
                        'action': parsed_cmd.action,
                        'confidence': parsed_cmd.confidence,
                        'reasoning': f'NLP parsed: {prompt}',
                        'vlm_response': f'NLP command: {parsed_cmd.action}',
                        'is_informational': False,
                        'parameters': parsed_cmd.parameters
                    }
                else:
                    # Command needs vision - continue to VLM processing below
                    self.get_logger().info(f"   └── NLP parsed to {parsed_cmd.action} (needs vision: {parsed_cmd.parameters.get('target', 'N/A')})")
                    # Store parsed command for later use with vision
                    nlp_action = parsed_cmd.action
                    nlp_params = parsed_cmd.parameters

            except Exception as e:
                self.get_logger().warn(f"⚠️ NLP parsing failed, falling back to keyword matching: {e}")
                # Continue with fallback methods below

        # FALLBACK: Check if this is a direct command (e.g., "move forward", "turn left")
        is_direct, direct_command, params = self._is_direct_command(prompt)
        if is_direct:
            self.get_logger().info(f"🎮 Detected DIRECT COMMAND: {prompt} → {direct_command}")
            if params:
                self.get_logger().info(f"   └── Parameters: {params}")
            # Return direct command without VLM processing
            return {
                'action': direct_command,
                'confidence': 1.0,
                'reasoning': f'Direct user command: {prompt}',
                'vlm_response': f'Executing direct command: {direct_command}',
                'is_informational': False,
                'parameters': params  # Include parsed parameters
            }

        # Check if this is an informational query
        is_informational = self._is_informational_query(prompt)
        if is_informational:
            self.get_logger().info(f"📝 Detected INFORMATIONAL query: {prompt}")
        else:
            self.get_logger().info(f"🧭 Detected NAVIGATION query: {prompt}")

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
            
            # Build sensor information from RAW LiDAR data
            sensor_info = "Sensors: Using RAW LiDAR_360° scan for all distance measurements"

            # Use RAW LiDAR data directly from ROS message to avoid filtering artifacts
            # Get the raw data from the most recent LiDAR message stored in callback
            with self.lidar_lock:
                if hasattr(self, 'latest_raw_lidar_msg') and self.latest_raw_lidar_msg:
                    raw_msg = self.latest_raw_lidar_msg
                    raw_ranges = np.array(raw_msg.ranges, dtype=np.float64)
                else:
                    # No raw LiDAR data available - this is an error
                    self.get_logger().error("❌ No raw LiDAR message available for VLM analysis")
                    return {
                        'action': 'stop',
                        'confidence': 0.0,
                        'reasoning': 'ERROR: No LiDAR data available - cannot navigate safely'
                    }

            # For VLM analysis, use minimally processed sensor data
            # Only handle infinite/NaN values, preserve ALL actual distances including very close ones
            vlm_ranges = np.where(np.isfinite(raw_ranges), raw_ranges, 30.0)  # Replace inf/NaN with 30m
            vlm_ranges = np.where(vlm_ranges > 30.0, 30.0, vlm_ranges)  # Cap at 30m
            # CRITICAL: Do NOT artificially set minimum distances - let model see actual close obstacles

            # DEBUG: Check LiDAR message properties
            if hasattr(self, 'latest_raw_lidar_msg') and self.latest_raw_lidar_msg:
                msg = self.latest_raw_lidar_msg
                self.get_logger().info(f"   └── LiDAR MSG CHECK: angle_min={msg.angle_min:.3f}, angle_max={msg.angle_max:.3f}, angle_increment={msg.angle_increment:.6f}")
                self.get_logger().info(f"   └── LiDAR MSG CHECK: range_min={msg.range_min:.3f}, range_max={msg.range_max:.3f}")

            # DEBUG: Log the actual data being sent to VLM
            self.get_logger().info(f"   └── VLM DATA CHECK: Front distance to VLM: {vlm_ranges[0]:.3f}m")
            if len(vlm_ranges) > 90:
                left_idx = 90 if 90 < len(vlm_ranges) else len(vlm_ranges)//4
                right_idx = 270 if 270 < len(vlm_ranges) else 3*len(vlm_ranges)//4
                self.get_logger().info(f"   └── VLM DATA CHECK: Left({left_idx}): {vlm_ranges[left_idx]:.3f}m, Right({right_idx}): {vlm_ranges[right_idx]:.3f}m")
            self.get_logger().info(f"   └── VLM DATA CHECK: Raw ranges sample: {vlm_ranges[:10] if len(vlm_ranges) > 10 else vlm_ranges}")

            # Check if there's a unit conversion issue
            if vlm_ranges[0] > 0:
                potential_meters = vlm_ranges[0]
                potential_mm = potential_meters * 1000
                potential_cm = potential_meters * 100
                self.get_logger().info(f"   └── UNIT CHECK: {potential_meters:.3f}m could be {potential_mm:.0f}mm or {potential_cm:.1f}cm")
            
            num_points = len(vlm_ranges)
            
            # DEBUG: Log LiDAR data details - USE SAME DATA AS VLM
            current_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(f"   └── DEBUG: LiDAR has {num_points} points (timestamp: {current_time:.3f})")
            self.get_logger().info(f"   └── DEBUG: Front distance (vlm_ranges[0]) = {vlm_ranges[0]:.3f}m")
            if num_points > 100:
                self.get_logger().info(f"   └── DEBUG: vlm_ranges[10] = {vlm_ranges[10]:.3f}m, vlm_ranges[50] = {vlm_ranges[50]:.3f}m")

            # EXTRA VERIFICATION: Will be logged after quantization

            # Create quantized LiDAR scan for VLM using configurable quantization
            quantization_degrees = GUIConfig.LIDAR_SCAN_QUANTIZATION_DEGREES
            num_points = int(360 / quantization_degrees)  # Calculate number of points
            quantized_ranges = np.zeros(num_points, dtype=np.float64)  # Initialize with zeros

            if len(vlm_ranges) > 0:
                # Calculate the angular resolution of the original scan
                if hasattr(self, 'latest_raw_lidar_msg') and self.latest_raw_lidar_msg:
                    msg = self.latest_raw_lidar_msg
                    angle_range = msg.angle_max - msg.angle_min
                    original_resolution = angle_range / (len(vlm_ranges) - 1) if len(vlm_ranges) > 1 else 0
                    self.get_logger().info(f"   └── LiDAR: {len(vlm_ranges)} points, angle_range: {np.degrees(angle_range):.1f}°, resolution: {np.degrees(original_resolution):.2f}°")

                    # Create proper angle mapping based on actual LiDAR angles
                    original_angles_rad = np.linspace(msg.angle_min, msg.angle_max, len(vlm_ranges))
                    original_angles_deg = np.degrees(original_angles_rad)

                    # Target angles for quantization (configurable degrees per point)
                    target_angles = np.arange(0, 360, quantization_degrees)

                    # Map target angles to original angle range
                    # Handle LiDARs that scan from -180° to +180° (need to reorder for 0-360° target)
                    if msg.angle_min < 0:
                        # LiDAR scans from -180° to +180°, but we want output as 0° to 360°
                        # Find the split point where angle crosses from negative to positive (0°/forward)
                        split_idx = np.argmin(np.abs(original_angles_deg))
                        
                        # Reorder data: [0° to 180°] + [180° to 360° (which was -180° to 0°)]
                        reordered_angles = np.concatenate([
                            original_angles_deg[split_idx:] + (360 if original_angles_deg[split_idx] < 0 else 0),
                            original_angles_deg[:split_idx] + 360
                        ])
                        reordered_ranges = np.concatenate([vlm_ranges[split_idx:], vlm_ranges[:split_idx]])
                        
                        # Now interpolate on properly ordered data (0° to 360°)
                        quantized_ranges = np.interp(target_angles, reordered_angles, reordered_ranges,
                                                   period=360)
                    else:
                        # Standard case for LiDARs starting at 0°
                        quantized_ranges = np.interp(target_angles, original_angles_deg, vlm_ranges,
                                                   left=vlm_ranges[0], right=vlm_ranges[-1])
                else:
                    # No LiDAR message metadata available - cannot proceed
                    self.get_logger().error("❌ No LiDAR message metadata available for angle calculation")
                    return {
                        'action': 'stop',
                        'confidence': 0.0,
                        'reasoning': 'ERROR: Cannot determine LiDAR angles - invalid sensor data'
                    }

                # Handle obstructed rear area (135-225 degrees)
                obstructed_start_idx = int(135 / quantization_degrees)
                obstructed_end_idx = int(225 / quantization_degrees)
                quantized_ranges[obstructed_start_idx:obstructed_end_idx] = -1  # Mark as obstructed

            # Create distance profile for VLM prompt - show all quantized points
            lidar_profile = []
            for i in range(num_points):  # Show all actual quantized points
                angle_deg = i * quantization_degrees  # Convert index back to angle
                angle_desc = f"{angle_deg}°"
                if quantized_ranges[i] < 0:  # Obstructed
                    lidar_profile.append(f"{angle_desc}:obstructed")
                else:
                    dist = quantized_ranges[i]
                    lidar_profile.append(f"{angle_desc}:{dist:.2f}m")

            lidar_360 = ", ".join(lidar_profile)

            # Log the quantization
            self.get_logger().info(f"   └── LiDAR quantized: {len(vlm_ranges)} -> {num_points} points ({quantization_degrees}° resolution)")
            # Calculate sample indices based on quantization
            idx_90 = int(90 / quantization_degrees)
            idx_180 = int(180 / quantization_degrees)
            idx_270 = int(270 / quantization_degrees)
            self.get_logger().info(f"   └── Sample quantized: [0°]={quantized_ranges[0]:.3f}m, [90°]={quantized_ranges[min(idx_90, num_points-1)]:.3f}m, [180°]={quantized_ranges[min(idx_180, num_points-1)]:.3f}m, [270°]={quantized_ranges[min(idx_270, num_points-1)]:.3f}m")

            # EXTRA VERIFICATION: Log what we're actually sending to VLM
            current_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(f"   └── VERIFICATION: Data sent to VLM (timestamp: {current_time:.3f}) - Front: {quantized_ranges[0]:.3f}m")
            self.get_logger().info(f"   └── VERIFICATION: Left: {quantized_ranges[min(idx_90, num_points-1)]:.3f}m, Right: {quantized_ranges[min(idx_270, num_points-1)]:.3f}m")

            # Extract key distances from quantized data
            front_distance = quantized_ranges[0] if len(quantized_ranges) > 0 else 30.0
            left_distance = quantized_ranges[min(idx_90, num_points-1)] if len(quantized_ranges) > idx_90 else front_distance
            right_distance = quantized_ranges[min(idx_270, num_points-1)] if len(quantized_ranges) > idx_270 else front_distance

            # Build comprehensive sensor data - ONLY provide full scan, no summary to force analysis
            sensor_info += f"FULL_{num_points}_LIDAR_SCAN_DATA: [{lidar_360}]"
            sensor_info += f"\n\nMANDATORY: You MUST analyze EVERY angle in the FULL_{num_points}_LIDAR_SCAN_DATA above. Do NOT use any summary values. Find the angle ranges with maximum clearance."
            
            # DEBUG: Log what data we're sending to VLM
            self.get_logger().info(f"   └── VLM INPUT DATA: Camera + 360° LiDAR scan")
            self.get_logger().info(f"   └── DEBUG SENSOR_INFO: {sensor_info}")

            # Generate different prompts based on query type
            if is_informational:
                # Informational prompt - focus on description, no movement required
                navigation_prompt = f"""You are a robot vision system. Analyze what you see in the camera image and answer the user's question.

USER QUESTION: {prompt}

Provide a clear, concise description of what you observe. Focus on:
- Objects and their locations
- Colors, sizes, and distinctive features
- Spatial relationships
- Any relevant details to answer the question

DO NOT provide movement commands. Just describe what you see.

Your response should be conversational and informative."""
            else:
                # Navigation prompt - CONCISE format to avoid token waste
                navigation_prompt = f"""Robot navigation decision required. Camera + LiDAR scan provided.

SENSOR DATA: {sensor_info}
TASK: {prompt}

CRITICAL FORMAT: Start your response IMMEDIATELY with "ACTION: [action]" where action is one of: stop, move_forward, turn_left, turn_right, strafe_left, strafe_right

LIDAR ANGLES:
- 0° = directly ahead (forward)
- 90° = hard left
- 180° = directly behind
- 270° = hard right
- Angles increase clockwise: 0°→90°→180°→270°→360°(0°)
- 135°-225° = rear blind spot (obstructed)
- Each angle shows distance to nearest obstacle in meters
- FULL_{num_points}_LIDAR_SCAN_DATA contains the complete {num_points}-point scan (every {quantization_degrees}°)
- You must examine the actual angle-by-angle data above
- Find angle ranges with maximum clearance for navigation

TURNING DIRECTIONS:
- If best clearance is at angles 315°-45° (forward-right) → turn_right or move_forward
- If best clearance is at angles 45°-135° (forward-left) → turn_left or move_forward
- If best clearance is at angles 225°-315° (back-right) → turn_right to face it
- If best clearance is at angles 135°-225° (back-left) → turn_left to face it

⚠️ CRITICAL SAFETY RULES - FOLLOW EXACTLY:
- Distance < 0.5m = CRITICAL DANGER - STOP or TURN AWAY immediately
- Distance 0.5m-1.5m = TOO CLOSE - DO NOT move_forward, TURN instead  
- Distance > 1.5m = SAFE - OK to move_forward
- Distance > 3.0m = VERY SAFE - clear path

🚨 MANDATORY DECISION RULES (CHECK FRONT FIRST):
1. **Front < 1.5m** → NEVER move_forward, MUST turn_left or turn_right toward clearer side
2. **Front >= 1.5m** → move_forward is SAFE and PREFERRED
3. **All directions < 1.0m** → stop (rare)

EXAMPLES WITH FRONT DISTANCE CHECK:
- Front: 0.5m → ❌ DO NOT move_forward (< 1.5m) → turn_left or turn_right
- Front: 1.0m → ❌ DO NOT move_forward (< 1.5m) → turn_left or turn_right  
- Front: 1.4m → ❌ DO NOT move_forward (< 1.5m) → turn_left or turn_right
- Front: 1.5m → ✅ SAFE to move_forward (>= 1.5m)
- Front: 2.0m → ✅ SAFE to move_forward (>= 1.5m)
- Front: 3.0m → ✅ SAFE to move_forward (>= 1.5m)

Choose ONE action:
- **move_forward**: ONLY if front >= 1.5m (check 0° angle first!)
- **turn_left**: If front < 1.5m AND left side clearer  
- **turn_right**: If front < 1.5m AND right side clearer
- strafe_left: If front < 1.0m AND left > 1.5m
- strafe_right: If front < 1.0m AND right > 1.5m
- stop: If all directions < 1.0m

🚨 MANDATORY 3-LINE FORMAT - YOU MUST OUTPUT EXACTLY THIS:
Line 1: CHECK: 0° is [distance]m, [< or >=] 1.5m
Line 2: ACTION: [move_forward if >= 1.5m, otherwise turn_left or turn_right]
Line 3: CONFIDENCE: 0.8

EXAMPLES (COPY THIS FORMAT EXACTLY):
CHECK: 0° is 2.5m, >= 1.5m
ACTION: move_forward
CONFIDENCE: 0.8

CHECK: 0° is 0.93m, < 1.5m
ACTION: turn_right
CONFIDENCE: 0.8

CHECK: 0° is 1.2m, < 1.5m  
ACTION: turn_left
CONFIDENCE: 0.8

DO NOT WRITE ANYTHING ELSE. JUST THESE 3 LINES."""
            
            model_short_name = self.model_name.split('/')[-1]
            self.get_logger().debug(f"   └── Running {model_short_name} inference for navigation...")
            
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
            self.get_logger().debug(f"   └── Processing inputs with {model_short_name} processor...")
            
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
            self.get_logger().info(f"   └── DEBUG: Full VLM response for parsing: '{response_text}'")

            # Handle informational vs navigation responses differently
            if is_informational:
                # For informational queries, return the description without movement
                self.get_logger().info(f"   └── Processing INFORMATIONAL response")
                navigation_result = {
                    'action': 'none',  # Special action type for informational queries
                    'confidence': 1.0,
                    'reasoning': response_text,  # The full description
                    'vlm_response': response_text,
                    'is_informational': True
                }
            else:
                # DEBUG: Check for turn-related keywords in response
                turn_keywords = ['TURN_LEFT', 'TURN_RIGHT', 'turn_left', 'turn_right', 'STRAFE_LEFT', 'STRAFE_RIGHT', 'strafe_left', 'strafe_right']
                found_turn_keywords = [kw for kw in turn_keywords if kw in response_text]
                if found_turn_keywords:
                    self.get_logger().info(f"   └── TURN KEYWORDS FOUND: {found_turn_keywords}")
                else:
                    self.get_logger().warn("   └── NO TURN KEYWORDS FOUND in VLM response - this explains why no turns!")

                # Parse navigation decision from response
                navigation_result = self._parse_vlm_response(response_text, sensor_data)
                navigation_result['is_informational'] = False
            
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
                # Use raw data for validation (same as what VLM sees)
                vlm_ranges = np.array(ranges)
                vlm_ranges = np.where(np.isfinite(vlm_ranges), vlm_ranges, 30.0)
                vlm_ranges = np.where(vlm_ranges > 30.0, 30.0, vlm_ranges)
                # Front is at index 0 in LiDAR scan
                lidar_front_distance = vlm_ranges[0] if len(vlm_ranges) > 0 else 999.0
            
            
            # Check for overly cautious stop commands when path is clear
            elif navigation_result['action'] == 'stop' and lidar_front_distance > 2.5:
                self.get_logger().warn(f"   └── VLM LOGIC ERROR DETECTED: Wants to stop with clear front distance {lidar_front_distance:.1f}m > 2.5m")
                self.get_logger().warn(f"   └── CORRECTING overly cautious VLM decision")
                self.get_logger().warn(f"   └── Overriding stop with MOVE_FORWARD (path is clear)")
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
            
            # Extract action - support both old ACTION: format and new format
            action = "stop"  # Default safe action
            self.get_logger().debug(f"   └── PARSING: Looking for action in response")

            # First try the old format (ACTION:)
            if "ACTION:" in response_upper:
                try:
                    action_split = response_upper.split("ACTION:")
                    if len(action_split) > 1:
                        # CRITICAL FIX: Only get the FIRST LINE after ACTION:, not the entire rest of the text
                        # Split by newline to isolate the action line from reasoning
                        first_line = action_split[1].split("\n")[0].split("\r")[0].strip()
                        # Also split by CONFIDENCE: in case it's all on one line
                        action_part = first_line.split("CONFIDENCE")[0].strip()
                        self.get_logger().info(f"   └── PARSING: Found ACTION: '{action_part}' (first line only)")
                    else:
                        action_part = ""
                        self.get_logger().warn("   └── PARSING: ACTION: found but no content after it")
                except (IndexError, AttributeError) as e:
                    self.get_logger().warn(f"   └── PARSING: Error parsing ACTION: {e}")
                    action_part = ""
                if "MOVE_FORWARD" in action_part or "MOVE FORWARD" in action_part:
                    action = "move_forward"
                    self.get_logger().info("   └── PARSING: Detected MOVE_FORWARD action")
                elif "TURN_LEFT" in action_part or "TURN LEFT" in action_part:
                    action = "turn_left"
                    self.get_logger().info("   └── PARSING: Detected TURN_LEFT action")
                elif "TURN_RIGHT" in action_part or "TURN RIGHT" in action_part:
                    action = "turn_right"
                    self.get_logger().info("   └── PARSING: Detected TURN_RIGHT action")
                elif "STRAFE_LEFT" in action_part or "STRAFE LEFT" in action_part:
                    action = "strafe_left"
                    self.get_logger().info("   └── PARSING: Detected STRAFE_LEFT action")
                elif "STRAFE_RIGHT" in action_part or "STRAFE RIGHT" in action_part:
                    action = "strafe_right"
                    self.get_logger().info("   └── PARSING: Detected STRAFE_RIGHT action")
                elif "STOP" in action_part:
                    action = "stop"
                    self.get_logger().info("   └── PARSING: Detected STOP action")
                else:
                    self.get_logger().warn(f"   └── PARSING: Unknown action in ACTION: '{action_part}'")
            else:
                # Try new format: "Best action is [action]"
                if "BEST ACTION IS" in response_upper:
                    try:
                        best_action_split = response_upper.split("BEST ACTION IS")
                        if len(best_action_split) > 1:
                            action_part = best_action_split[1].split("BECAUSE")[0].strip()
                            self.get_logger().info(f"   └── PARSING: Found BEST ACTION IS: '{action_part}'")
                        else:
                            action_part = ""
                            self.get_logger().warn("   └── PARSING: BEST ACTION IS found but no content after it")
                    except (IndexError, AttributeError) as e:
                        self.get_logger().warn(f"   └── PARSING: Error parsing BEST ACTION IS: {e}")
                        action_part = ""

                    # Clean up the action part
                    action_part = action_part.replace(".", "").replace(",", "").strip()

                    if "MOVE_FORWARD" in action_part or "FORWARD" in action_part:
                        action = "move_forward"
                        self.get_logger().info("   └── PARSING: Detected MOVE_FORWARD action")
                    elif "TURN_LEFT" in action_part or "LEFT" in action_part:
                        action = "turn_left"
                        self.get_logger().info("   └── PARSING: Detected TURN_LEFT action")
                    elif "TURN_RIGHT" in action_part or "RIGHT" in action_part:
                        action = "turn_right"
                        self.get_logger().info("   └── PARSING: Detected TURN_RIGHT action")
                    elif "STRAFE_LEFT" in action_part:
                        action = "strafe_left"
                        self.get_logger().info("   └── PARSING: Detected STRAFE_LEFT action")
                    elif "STRAFE_RIGHT" in action_part:
                        action = "strafe_right"
                        self.get_logger().info("   └── PARSING: Detected STRAFE_RIGHT action")
                    elif "STOP" in action_part:
                        action = "stop"
                        self.get_logger().info("   └── PARSING: Detected STOP action")
                    else:
                        self.get_logger().error(f"   └── PARSING ERROR: Unknown action in BEST ACTION IS: '{action_part}'")
                        self.get_logger().error("   └── VLM response must use proper ACTION: format")
                        return {
                            'action': 'stop',
                            'confidence': 0.0,
                            'reasoning': f"ERROR: Invalid action format in VLM response: {action_part}"
                        }
                else:
                    # No ACTION: or BEST ACTION IS found - this is an error
                    self.get_logger().error("   └── PARSING ERROR: VLM response missing required ACTION: or BEST ACTION IS prefix")
                    self.get_logger().error(f"   └── Response was: {response_text[:200]}")
                    self.get_logger().error("   └── VLM must respond with proper format: ACTION: [action] or BEST ACTION IS [action]")
                    return {
                        'action': 'stop',
                        'confidence': 0.0,
                        'reasoning': "ERROR: VLM response missing required ACTION: format"
                    }
            
            self.get_logger().debug(f"   └── PARSING: Final action: '{action}'")
            
            # Extract confidence - support both old CONFIDENCE: format and new format
            confidence = 0.5  # Default confidence
            self.get_logger().info(f"   └── DEBUG: Starting confidence parsing with default: {confidence}")
            if "CONFIDENCE:" in response_upper:
                try:
                    conf_split = response_upper.split("CONFIDENCE:")
                    if len(conf_split) > 1:
                        conf_part = conf_split[1].strip()
                    else:
                        conf_part = ""
                    self.get_logger().debug(f"   └── PARSING: Confidence part: '{conf_part}'")
                    # Extract just the number after CONFIDENCE: (handle spaces and other text)
                    import re
                    # Look for number immediately after CONFIDENCE: or at start of conf_part
                    conf_match = re.search(r'^(\d+\.?\d*)', conf_part.strip()) or re.search(r'(\d+\.?\d*)', conf_part[:20])
                    if conf_match:
                        confidence = float(conf_match.group(1))
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                        self.get_logger().info(f"   └── PARSING: Extracted confidence from CONFIDENCE:: {confidence}")
                    else:
                        self.get_logger().debug(f"   └── PARSING: No numeric confidence found in: '{conf_part}' - checking full response")
                        # Fall through to check for confidence level indicators in full response
                except Exception as e:
                    self.get_logger().debug(f"   └── PARSING: Confidence extraction failed: {e}")
                    # Fall through to check for confidence level indicators in full response
            # Check for confidence level indicators in the full response (both CONFIDENCE: and non-CONFIDENCE: cases)
            if confidence == 0.5:  # Only check if we haven't found a better confidence yet
                if "confidence" in response_text.lower():
                    self.get_logger().info("   └── PARSING: Found 'confidence' in response")
                    # Look for confidence level indicators
                    if "high confidence" in response_text.lower():
                        confidence = 0.8
                        self.get_logger().info("   └── PARSING: Detected HIGH confidence -> 0.8")
                    elif "low confidence" in response_text.lower():
                        confidence = 0.3
                        self.get_logger().info("   └── PARSING: Detected LOW confidence -> 0.3")
                    elif "very high" in response_text.lower() or "certain" in response_text.lower():
                        confidence = 0.9
                        self.get_logger().info("   └── PARSING: Detected VERY HIGH confidence -> 0.9")
                    elif "very low" in response_text.lower() or "uncertain" in response_text.lower():
                        confidence = 0.2
                        self.get_logger().info("   └── PARSING: Detected VERY LOW confidence -> 0.2")
                    else:
                        self.get_logger().info("   └── PARSING: Found confidence but no level indicator, keeping default 0.5")
                else:
                    # Try to extract numeric confidence from reasoning
                    import re
                    conf_matches = re.findall(r'(\d+\.?\d*)', response_text)
                    if conf_matches:
                        for match in conf_matches:
                            conf_val = float(match)
                            if 0.0 <= conf_val <= 1.0:
                                confidence = conf_val
                                self.get_logger().debug(f"   └── PARSING: Extracted numeric confidence: {confidence}")
                                break
            
            # Extract reasoning - support multiple formats
            reasoning = "VLM navigation decision"
            try:
                self.get_logger().info(f"   └── PARSING: Raw response for reasoning extraction: '{response_text[:200]}...'")
                self.get_logger().info(f"   └── PARSING: response_upper starts with: '{response_upper[:100]}...'")

                # Try different reasoning formats
                if "ANALYZE SITUATION" in response_upper:
                    self.get_logger().info("   └── PARSING: Found ANALYZE SITUATION pattern in response_upper")
                    analyze_split = response_text.split("ANALYZE SITUATION:")
                    if len(analyze_split) > 1:
                        reasoning = analyze_split[1].strip()
                        # Clean up quotes and extract just the assessment part
                        reasoning = reasoning.strip('"').split('"')[0] if '"' in reasoning else reasoning
                        self.get_logger().info(f"   └── PARSING: Found ANALYZE SITUATION format: '{reasoning}'")
                elif "REASON:" in response_upper:
                    reason_split = response_text.split("REASON:")
                    if len(reason_split) > 1:
                        reasoning = reason_split[1].split("|")[0].strip()
                        self.get_logger().info(f"   └── PARSING: Found REASON format: '{reasoning}'")
                elif "STATE CONFIDENCE" in response_upper:
                    # Handle the actual VLM format: "3. State confidence: ..."
                    confidence_split = response_text.split("STATE CONFIDENCE:")
                    if len(confidence_split) > 1:
                        reasoning = confidence_split[1].strip()
                        # Remove quotes if present
                        reasoning = reasoning.strip('"')
                        self.get_logger().info(f"   └── PARSING: Found STATE CONFIDENCE format: '{reasoning}'")
                elif "CONFIDENCE:" in response_upper:
                    confidence_split = response_text.split("CONFIDENCE:")
                    if len(confidence_split) > 1:
                        reasoning = confidence_split[1].strip()
                        reasoning = reasoning.strip('"')
                        self.get_logger().info(f"   └── PARSING: Found CONFIDENCE format: '{reasoning}'")
                elif "BECAUSE" in response_upper:
                    because_split = response_text.split("BECAUSE")
                    if len(because_split) > 1:
                        reasoning = because_split[1].strip()
                        reasoning = reasoning.split(".")[0] if "." in reasoning else reasoning
                        self.get_logger().info(f"   └── PARSING: Found BECAUSE format: '{reasoning}'")
                else:
                    # Try to find any reasoning-like content
                    self.get_logger().info(f"   └── PARSING: No standard reasoning format found in response")
                    # Check if there's a CHECK: line for the new simplified format
                    if "CHECK:" in response_upper:
                        # Extract the CHECK line as reasoning
                        check_line = response_text.split("CHECK:")[1].split("\n")[0].strip() if "CHECK:" in response_text else ""
                        reasoning = f"CHECK: {check_line}" if check_line else "Navigation decision based on sensor data"
                        self.get_logger().info(f"   └── PARSING: Using CHECK line as reasoning: '{reasoning}'")
                    # Look for common reasoning keywords
                    elif "because" in response_text.lower():
                        because_idx = response_text.lower().find("because")
                        reasoning = response_text[because_idx:].strip()
                        reasoning = reasoning.split(".")[0] if "." in reasoning else reasoning
                        self.get_logger().info(f"   └── PARSING: Extracted reasoning from 'because': '{reasoning}'")
                    else:
                        # Accept responses without explicit reasoning for simplified format
                        reasoning = "Navigation decision based on LiDAR analysis"
                        self.get_logger().info(f"   └── PARSING: No reasoning provided, using default")
            except (IndexError, AttributeError) as e:
                self.get_logger().error(f"   └── PARSING ERROR: Failed to extract reasoning: {e}")
                return {
                    'action': 'stop',
                    'confidence': 0.0,
                    'reasoning': f"ERROR: Reasoning extraction failed: {e}"
                }
            
            # Check for reasoning-action consistency
            reasoning_lower = reasoning.lower()
            if action == 'move_forward' and ('must stop' in reasoning_lower or 'should stop' in reasoning_lower):
                self.get_logger().warn(f"   └── REASONING-ACTION MISMATCH: Says 'must stop' but ACTION is 'move_forward'")
                self.get_logger().warn(f"   └── Correcting action to match reasoning")
                action = 'stop'
                confidence = max(0.8, confidence)  # High confidence in safety correction
            elif action == 'stop' and ('must move' in reasoning_lower or 'should move' in reasoning_lower):
                self.get_logger().warn(f"   └── REASONING-ACTION MISMATCH: Says 'must move' but ACTION is 'stop'")
                # Keep stop for safety - don't override stop with movement
            
            self.get_logger().info(f"   └── PARSING: FINAL RESULT - Action: {action}, Confidence: {confidence:.2f}, Reasoning: {reasoning}")
            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'vlm_response': response_text
            }
            
        except Exception as e:
            self.get_logger().warn(f"Failed to parse VLM response (response_length={len(response_text)}): {e}")
            self.get_logger().warn(f"Response was: '{response_text}'")
            self.get_logger().error(f"   └── VLM response parsing failed - cannot navigate safely")
            return {
                'action': 'stop',
                'confidence': 0.0,
                'reasoning': f'VLM response parsing failed: {e} - navigation requires successful analysis'
            }
    
    def _goal_service_callback(self, request, response):
        """Handle RoboMP2 goal setting requests"""
        try:
            # Parse goal from request - single robot setup
            goal_data = "navigation|Navigate safely"  # Default goal for single robot
            
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
            
            # Parse policy from request - single robot setup
            policy_data = f"policy_{int(time.time())}|unknown|navigation|stop|Default policy"  # Default policy data
            
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
