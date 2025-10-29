# Autonomous Navigation Guide

**Date**: October 29, 2025

## Overview

The Robot VLM system supports **autonomous exploration** where the VLM (Vision-Language Model) analyzes the environment and directs the robot to navigate and map objects automatically.

---

## How It Works

### System Components

1. **VLM Navigation Node** (`local_vlm_navigation_node.py`)
   - Analyzes camera images, LiDAR scans, and sensor data
   - Uses Qwen3-VL-8B-Instruct model enhanced with RoboMP2 components
   - Generates navigation commands based on visual and spatial understanding
   - Publishes commands directly to `/cmd_vel` topic

2. **GUI Controls** (`robot_gui_node.py`)
   - **Auto Analysis Toggle** (`🔄 Auto: OFF/ON`): Enables continuous VLM analysis
   - **Auto Execute Toggle** (`🤖 Execute: OFF/ON`): Enables automatic execution of VLM commands
   - **Movement Enable Toggle** (`✅ ENABLED/🔒 DISABLED`): Emergency stop override

3. **RoboMP2 Framework**
   - **GCMP** (Goal-Conditioned Multimodal Perceptor): Understands environment state
   - **RAMP** (Retrieval-Augmented Multimodal Planner): Retrieves and applies successful navigation policies

---

## Starting Autonomous Navigation

### Step 1: Launch the System

```bash
cd /home/marc/Robot_LLM
source install/setup.bash
ros2 launch robot_sim gazebo_vila_integration.launch.py
```

This will start:
- Gazebo simulation with the mecanum robot
- ROS-Gazebo bridge
- GUI with movement controls
- VLM Navigation Node (RoboMP2)

### Step 2: Wait for Model Loading

Watch the terminal output for:
```
[local_vlm_navigation_node.py-9] [INFO] Loading Qwen3-VL-8B-Instruct model with speed optimization...
[local_vlm_navigation_node.py-9] [INFO] ✅ Model loaded successfully
```

The GUI status panel will show:
- **VLM Model Status**: `RoboMP2 + Qwen3-VL-8B-Instruct` (green)

### Step 3: Enable Autonomous Navigation

In the GUI, **in this order**:

1. **Enable Auto Analysis**:
   - Click the `🔄 Auto: OFF` button
   - It will change to `🔄 Auto: ON` (green background)
   - The VLM will start analyzing the environment every 3 seconds

2. **Enable Auto Execute**:
   - Click the `🤖 Execute: OFF` button
   - It will change to `🤖 Execute: ON` (green background)
   - The robot will automatically execute VLM navigation commands

3. **Verify Movement is Enabled**:
   - Check that the movement toggle shows `✅ ENABLED`
   - If it shows `🔒 DISABLED`, click it to enable

### Step 4: Monitor the Robot

The robot will now:
- ✅ Continuously analyze its surroundings
- ✅ Detect obstacles using LiDAR and camera
- ✅ Make navigation decisions
- ✅ Move autonomously to explore the environment
- ✅ Avoid collisions

---

## What the Robot Will Do

### Autonomous Behaviors

1. **Obstacle Detection**:
   - Uses 360° LiDAR to detect obstacles
   - Analyzes camera feed for visual identification
   - Maintains safe distances from objects

2. **Path Planning**:
   - Identifies clear paths using LiDAR data
   - Prioritizes forward movement when clear
   - Turns toward open spaces when obstructed

3. **Exploration Strategy**:
   - Moves toward unexplored areas
   - Rotates to scan surroundings
   - Records object locations in policy database

4. **Object Mapping**:
   - VLM identifies objects from camera view (boxes, cylinders, walls)
   - GCMP records spatial relationships
   - RAMP stores successful navigation patterns

---

## Monitoring and Control

### GUI Displays

1. **Camera Feed** (center panel):
   - Live view from robot's perspective
   - Shows what the VLM is analyzing

2. **LiDAR Visualization** (left panel):
   - Green dots show obstacle distances
   - Triangle points forward (robot orientation)
   - Dead zone at rear (90° arc)

3. **VLM Analysis** (right panel):
   - **Current Command**: Shows VLM's current action (e.g., `move_forward`, `turn_left`)
   - **Analysis Prompt**: Customizable prompt for VLM
   - **Result**: Full VLM response with reasoning

4. **Activity Log** (bottom panel):
   - Real-time log of all actions
   - VLM analysis results
   - Navigation decisions
   - Error messages

### Manual Override

**To stop autonomous navigation**:

1. **Emergency Stop**: Click `✅ ENABLED` → `🔒 DISABLED`
   - Immediately disables all movement
   - VLM continues analyzing but won't execute

2. **Disable Auto Execute**: Click `🤖 Execute: ON` → `OFF`
   - Stops automatic command execution
   - VLM still provides recommendations
   - You can manually review and execute

3. **Disable Auto Analysis**: Click `🔄 Auto: ON` → `OFF`
   - Stops continuous VLM analysis
   - Manual analysis still available via `🔍 Analyze` button

4. **Manual Control**: Use arrow buttons
   - Override autonomous navigation
   - Direct control: Forward, Backward, Left, Right, Strafe
   - Click `⏹ Stop` to halt movement

---

## Advanced Features

### Custom Navigation Goals

Send custom goals via ROS2 service:

```bash
ros2 service call /robomp2/set_goal robot_msgs/srv/ExecuteCommand "{
  command: '{
    \"goal_type\": \"exploration\",
    \"description\": \"Find and photograph all red objects\",
    \"timeout\": 120.0
  }'
}"
```

**Goal Types**:
- `exploration`: General environment exploration
- `navigation`: Navigate to specific location
- `interaction`: Approach specific object
- `inspection`: Detailed examination of area

### Custom Analysis Prompts

In the GUI's **Analysis Prompt** field, try:
- `"Find the nearest red object and approach it"`
- `"Explore the room and identify all obstacles"`
- `"Navigate to the open area on the left"`
- `"Search for the blue cylinder"`

Then click `🔍 Analyze` or enable `🔄 Auto` + `🤖 Execute`.

### Policy Database

The system learns from successful navigation:
- Stores successful action sequences in `robomp2_policies.pkl`
- Retrieves similar situations using RAMP
- Adapts behavior based on past experience

---

## Troubleshooting

### Robot Not Moving

**Check**:
1. Movement toggle shows `✅ ENABLED`
2. Auto Execute is `ON` (green)
3. Auto Analysis is `ON` (green)
4. VLM Model Status shows `RoboMP2 + Qwen3-VL-8B-Instruct` (green)

**If model status shows "Loading..."**:
- Wait 30-60 seconds for model to load
- Check terminal for errors
- Verify GPU is available: `nvidia-smi`

### Robot Moving Erratically

**Possible causes**:
1. **LiDAR range too low**: Increase slider to 5.0m
2. **VLM analysis interval too short**: Adjust `VLM_AUTO_INTERVAL` in `gui_config.py`
3. **Conflicting commands**: Disable auto execute, review VLM output

### VLM Not Analyzing

**Check terminal for**:
- Model loading errors
- Memory errors (GPU out of memory)
- Service connection issues

**Solutions**:
- Restart the system
- Check GPU memory: `nvidia-smi`
- Reduce image size in `local_vlm_navigation_node.py`

---

## Performance Optimization

### Speed vs. Accuracy

Edit `/home/marc/Robot_LLM/src/robot_vila_system/robot_vila_system/gui_config.py`:

```python
# Faster analysis (less detail)
VLM_AUTO_INTERVAL = 2.0  # Analyze every 2 seconds

# More detailed analysis (slower)
VLM_AUTO_INTERVAL = 5.0  # Analyze every 5 seconds
```

### GPU Memory

If you encounter GPU out-of-memory errors:

1. **Reduce image size** in `local_vlm_navigation_node.py`:
   ```python
   self.max_image_size = 112  # Default: 112, try 96 or 80
   ```

2. **Use smaller batch size**:
   ```python
   max_new_tokens=100  # Default: 150
   ```

---

## ROS2 Topics Reference

### Published by VLM Node

- `/vlm/analysis` (String): VLM analysis results
- `/vlm/status` (String): Model status updates
- `/robot/navigation_commands` (String): Generated navigation commands

### Subscribed by VLM Node

- `/realsense/camera/color/image_raw` (Image): Camera feed
- `/scan` (LaserScan): LiDAR data
- `/robot/sensors` (SensorData): Robot sensor data

### Command Topic

- `/cmd_vel` (Twist): Robot velocity commands (published by VLM when auto-executing)

### Services

- `/vlm/analyze_scene`: Request manual VLM analysis
- `/robomp2/set_goal`: Set navigation goal
- `/robomp2/add_policy`: Add policy to database

---

## Example Session

```bash
# Terminal 1: Launch system
cd /home/marc/Robot_LLM
source install/setup.bash
ros2 launch robot_sim gazebo_vila_integration.launch.py

# Wait for model to load (30-60 seconds)
# Watch for: "[INFO] ✅ Model loaded successfully"

# In GUI:
# 1. Click "🔄 Auto: OFF" → turns ON
# 2. Click "🤖 Execute: OFF" → turns ON
# 3. Watch robot explore!

# Terminal 2: Monitor topics (optional)
ros2 topic echo /vlm/analysis
ros2 topic echo /robot/navigation_commands
ros2 topic echo /cmd_vel

# Terminal 3: Monitor robot state (optional)
ros2 topic echo /scan
ros2 topic echo /imu
```

---

## Safety Notes

⚠️ **Important Safety Features**:

1. **Movement Enable Toggle**: Acts as emergency stop
2. **LiDAR-based Collision Avoidance**: Built into VLM decision-making
3. **Manual Override**: Always available via GUI buttons
4. **Velocity Limits**: Configured in code for safe operation

⚠️ **When testing on real hardware**:
- Start with low speed settings
- Keep emergency stop within reach
- Test in open area first
- Monitor battery levels

---

## Next Steps

Once autonomous navigation is working:

1. **Adjust exploration prompts** to focus on specific tasks
2. **Monitor policy database** to see learned behaviors
3. **Export navigation logs** for analysis
4. **Tune VLM parameters** for better performance
5. **Add custom goals** for specific missions

---

## Support

For issues or questions:
- Check terminal logs for errors
- Review Activity Log in GUI
- Check ROS2 topic data: `ros2 topic list` and `ros2 topic echo`
- Verify GPU status: `nvidia-smi`

