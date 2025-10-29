# VLM Sensor Data Processing Flow

**Date**: October 29, 2025

## Overview

The VLM (Vision-Language Model) Navigation Node processes multimodal sensor data to generate intelligent navigation commands for the robot. It combines camera images, LiDAR scans, and robot sensor data using a vision-language model (configured in GUIConfig) enhanced with **RoboMP2** components.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SENSOR INPUTS                             │
├─────────────────────────────────────────────────────────────┤
│  1. Camera: /realsense/camera/color/image_raw (RGB)        │
│  2. LiDAR:  /scan (180° laser scan, 0°=front)             │
│  3. Sensors: /robot/sensors (battery, CPU, temp)           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│         RoboMP2NavigationNode (VLM Processing)              │
├─────────────────────────────────────────────────────────────┤
│  • Subscribes to sensor topics                              │
│  • Stores latest data in thread-safe buffers                │
│  • Processes requests via /vlm/analyze_scene service        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              VLM INFERENCE PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│  1. Data Collection                                          │
│  2. Image Preprocessing (resize to 112x112 for speed)       │
│  3. LiDAR Processing (quantize to 180 points)               │
│  4. Prompt Engineering (include all sensor context)         │
│  5. VLM Generation (with timeout protection)                │
│  6. Response Parsing (extract ACTION/CONFIDENCE/REASONING)  │
│  7. Safety Validation (check for logical errors)            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  NAVIGATION OUTPUT                           │
├─────────────────────────────────────────────────────────────┤
│  • Action: move_forward, turn_left, turn_right, stop        │
│  • Confidence: 0.1 - 0.9                                     │
│  • Reasoning: Detailed explanation of decision               │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow Details

### 1. **Sensor Data Collection** (Continuous)

The VLM node subscribes to three main data streams:

#### A. Camera Images (`/realsense/camera/color/image_raw`)
```python
def _camera_image_callback(self, msg: ROSImage):
    """Store latest camera image in thread-safe buffer"""
    with self.camera_lock:
        self.current_camera_image = msg
```
- **Format**: ROS Image message (RGB8 encoding)
- **Resolution**: Variable (resized to 112x112 for VLM processing)
- **Update Rate**: ~30 Hz from Gazebo
- **Storage**: Latest image only (overwritten)

#### B. LiDAR Scans (`/scan`)
```python
def _lidar_scan_callback(self, msg: LaserScan):
    """Process and store LiDAR scan data"""
    processed_data = self._process_lidar_scan(msg)
    with self.lidar_lock:
        self.current_lidar_data = processed_data
```
- **Format**: LaserScan message with ranges array
- **Coverage**: 180° arc (front hemisphere)
- **Angle Convention**: 
  - Index 0 = 0° (front)
  - Increases counterclockwise
  - Index mid = 90° (left)
  - Index max = 180° (back)
- **Processing**: 
  - Invalid values (inf, nan) → 30.0m
  - Quantized to 180 points for VLM context
  - Smoothing applied to reduce noise

#### C. Robot Sensors (`/robot/sensors`)
```python
def _sensor_data_callback(self, msg: SensorData):
    """Store robot telemetry"""
    with self.sensor_data_lock:
        self.current_sensor_data = {
            'battery_voltage': msg.battery_voltage,
            'cpu_temp': msg.cpu_temp,
            'cpu_usage': msg.cpu_usage
        }
```
- **Data**: Battery, CPU temperature, CPU usage
- **Purpose**: Context for decision-making (low battery → conservative)

---

### 2. **Analysis Request** (On-Demand)

When the GUI sends a VLM analysis request:

```python
# GUI calls the service
self.ros_node.vlm_client.call_async(request)

# VLM node receives request
def _analysis_service_callback(self, request, response):
    # 1. Collect current sensor data
    current_image = self._get_current_camera_image()
    sensor_data = self._get_current_sensor_data()
    lidar_data = self._get_current_lidar_data()
    
    # 2. Run VLM inference
    navigation_result = self._run_vlm_navigation_inference(
        current_image, prompt, sensor_data, lidar_data
    )
    
    # 3. Return result
    response.success = True
    response.result_message = json.dumps(navigation_result)
```

---

### 3. **VLM Inference Pipeline**

The core processing happens in `_run_vlm_navigation_inference()`:

#### Step 1: Image Preprocessing
```python
# Convert ROS Image to PIL Image
np_image = np.frombuffer(ros_image.data, dtype=np.uint8)
rgb_array = np_image.reshape((ros_image.height, ros_image.width, 3))
image = Image.fromarray(rgb_array, 'RGB')

# Resize for speed (112x112)
image = image.resize((112, 112), Image.Resampling.LANCZOS)
```

#### Step 2: LiDAR Data Formatting
```python
# Quantize to 180 points
lidar_ranges = np.array(raw_ranges)
lidar_ranges = np.where(np.isfinite(lidar_ranges), lidar_ranges, 30.0)

# Format for VLM prompt
lidar_text = "FULL_180_LIDAR_SCAN_DATA:\n"
for i, dist in enumerate(lidar_ranges):
    angle = i  # 0° to 180°
    lidar_text += f"Angle {angle:3d}°: {dist:.2f}m\n"
```

#### Step 3: Prompt Construction
```python
navigation_prompt = f"""ROBOT NAVIGATION ANALYSIS:

Current Camera View: [Image shows the robot's perspective]

{lidar_text}

CRITICAL: Analyze the COMPLETE LiDAR scan from 0° to 180° to find the BEST path.
Don't just look at front (0°) - explore ALL angles for maximum clearance!

NAVIGATION DECISION STEPS:
1. Environment analysis: "[Describe what you see in ALL parts of scan]"
2. Strategy selection: "[Based on COMPLETE scan, recommend best direction]"
3. Confidence: "[Justify based on full scan analysis]"

Respond with: ACTION: [action] CONFIDENCE: [0.1-0.9] REASONING: [analysis]
"""
```

**Key aspects**:
- Includes complete 180° LiDAR data with angles and distances
- Camera image provides visual context
- Explicit instructions to analyze ALL angles
- Structured response format requested

#### Step 4: Model Generation
```python
# Apply chat template
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": navigation_prompt}
    ]
}]

# Process inputs
inputs = self.processor(text=[text], images=[image], return_tensors="pt")
inputs = {k: v.to('cuda') for k, v in inputs.items()}

# Generate with optimizations
with torch.no_grad():
    generated_ids = self.model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,  # Deterministic
        use_cache=True,   # KV cache for speed
        num_beams=1       # Greedy search
    )

# Decode response
response_text = self.processor.batch_decode(generated_ids_trimmed)[0]
```

**Optimizations**:
- Image resized to 112x112 (16x fewer pixels than 448x448)
- FP16 precision for speed
- Greedy decoding (no sampling)
- KV cache enabled
- Typical inference time: 2-5 seconds

#### Step 5: Response Parsing
```python
def _parse_vlm_response(self, response_text, sensor_data):
    # Extract structured information
    # Expected format: "ACTION: move_forward CONFIDENCE: 0.8 REASONING: ..."
    
    action = self._extract_action(response_text)
    confidence = self._extract_confidence(response_text)
    reasoning = self._extract_reasoning(response_text)
    
    return {
        'action': action,
        'confidence': confidence,
        'reasoning': reasoning,
        'vlm_response': response_text
    }
```

**Extracted Actions**:
- `move_forward` - Clear path ahead
- `turn_left` - Better clearance to the left
- `turn_right` - Better clearance to the right
- `stop` - No safe path / obstacle too close

#### Step 6: Safety Validation
```python
# Check for logical errors
lidar_front_distance = lidar_data['ranges'][0]

if action == 'stop' and lidar_front_distance > 2.5:
    # VLM being overly cautious - override
    self.get_logger().warn("VLM logic error - correcting")
    return {
        'action': 'move_forward',
        'confidence': 0.8,
        'reasoning': 'CORRECTED: Front clear at {dist}m'
    }
```

---

### 4. **Result Delivery**

The VLM analysis result is sent back to the GUI:

```python
# Response format
{
    'success': True,
    'analysis': "VLM Decision: move_forward (confidence: 0.85)",
    'reasoning': "Front sector clear at 3.2m, best path forward",
    'navigation_commands': {
        'action': 'move_forward',
        'confidence': 0.85
    },
    'inference_time': 3.45  # seconds
}
```

The GUI displays:
- **Action** in the analysis panel
- **Reasoning** showing the VLM's thought process
- **Confidence** level indicator

---

## RoboMP2 Components

### GCMP (Goal-Conditioned Multimodal Perceptor)
- Processes environment state with goal context
- Extracts visual, spatial, and temporal features
- Maintains state history for temporal reasoning

### RAMP (Retrieval-Augmented Multimodal Planner)
- Stores successful navigation policies
- Retrieves similar past situations
- Enhances planning with learned experience

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Image Processing | ~100ms |
| LiDAR Processing | ~50ms |
| VLM Inference | 2-5 seconds |
| Total Latency | 2-6 seconds |
| GPU Memory | ~8-10GB (FP16) |
| Throughput | ~0.2-0.5 requests/sec |

---

## Key Design Decisions

### 1. **Small Image Size (112x112)**
- **Why**: 16x faster than 448x448
- **Trade-off**: Less visual detail, but LiDAR provides spatial precision
- **Result**: 2-3 second inference vs 10-15 seconds

### 2. **180-Point LiDAR Quantization**
- **Why**: Fits in VLM context window, provides complete coverage
- **Format**: Text-based angle/distance pairs (easy for VLM to parse)
- **Benefit**: VLM can reason about specific angles

### 3. **Thread-Safe Data Buffers**
- **Why**: Sensors update at different rates (30 Hz camera, 10 Hz LiDAR)
- **Mechanism**: Locks protect concurrent access
- **Result**: Always get latest synchronized data

### 4. **Safety Validation Layer**
- **Why**: VLMs can make logical errors or be overly cautious
- **Checks**: Compare VLM decision to sensor thresholds
- **Override**: When VLM contradicts clear sensor data

### 5. **Deterministic Generation**
- **Why**: Reproducible navigation behavior
- **Method**: `do_sample=False`, greedy decoding
- **Benefit**: Same input → same output

---

## Troubleshooting

### VLM Returns "stop" Too Often
- Check LiDAR data quality (run `ros2 topic echo /scan`)
- Verify safety validation thresholds
- Increase confidence requirements

### Slow Inference (>10 seconds)
- Check GPU utilization (`nvidia-smi`)
- Verify FP16 mode is enabled
- Ensure image is resized to 112x112

### Inconsistent Actions
- Enable deterministic mode (`do_sample=False`)
- Check if RAMP is retrieving conflicting policies
- Review LiDAR data smoothing parameters

---

## Future Enhancements

1. **Dynamic Image Sizing**: Adjust resolution based on scene complexity
2. **Temporal Reasoning**: Use GCMP to track obstacle motion
3. **Active Learning**: Update RAMP database with successful runs
4. **Multi-Goal Planning**: Optimize for battery, speed, and safety simultaneously
5. **Attention Visualization**: Show which image regions influenced decision

---

**For more details, see:**
- `local_vlm_navigation_node.py` - Main VLM processing logic
- `robomp2_components.py` - GCMP and RAMP implementations
- `robot_gui_node.py` - GUI integration and service calls

