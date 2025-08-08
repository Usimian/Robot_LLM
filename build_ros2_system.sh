#!/bin/bash
"""
ROS2 System Build Script
Builds the ROS2 message packages and sets up the environment
"""

set -e  # Exit on any error

echo "🔧 Building ROS2 Robot System"
echo "================================"

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "❌ ROS2 environment not sourced!"
    echo "Please run: source /opt/ros/<distro>/setup.bash"
    exit 1
fi

echo "✅ ROS2 environment detected: $ROS_DISTRO"

# Create workspace structure if it doesn't exist
WORKSPACE_DIR="/home/marc/Robot_LLM"
cd "$WORKSPACE_DIR"

echo "📁 Workspace: $WORKSPACE_DIR"

# Build custom messages
echo "🔨 Building robot_msgs package..."
if [ -d "robot_msgs" ]; then
    colcon build --packages-select robot_msgs
    
    if [ $? -eq 0 ]; then
        echo "✅ robot_msgs built successfully"
    else
        echo "❌ Failed to build robot_msgs"
        exit 1
    fi
else
    echo "❌ robot_msgs directory not found!"
    exit 1
fi

# Source the built packages
echo "📦 Sourcing built packages..."
source install/setup.bash

# Verify the build
echo "🧪 Verifying build..."
python3 -c "
try:
    import robot_msgs.msg
    import robot_msgs.srv
    print('✅ robot_msgs import successful')
except ImportError as e:
    print(f'❌ robot_msgs import failed: {e}')
    exit(1)
"

# Check dependencies
echo "🔍 Checking Python dependencies..."
python3 -c "
import sys
missing = []

# Required packages
required = [
    'rclpy',
    'cv_bridge', 
    'PIL',
    'cv2',
    'numpy'
]

for pkg in required:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'❌ {pkg} - MISSING')

if missing:
    print(f'\\n⚠️  Missing packages: {missing}')
    print('Install with: pip install pillow opencv-python numpy')
    print('For ROS2 packages: sudo apt install ros-$ROS_DISTRO-cv-bridge')
else:
    print('\\n✅ All dependencies satisfied')
"

# Create launch aliases
echo "⚙️ Creating launch aliases..."
cat > launch_aliases.sh << 'EOF'
#!/bin/bash
# ROS2 Robot System Launch Aliases

# Source ROS2 environment
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash

# Launch aliases
alias ros2_robot_complete="python3 launch_ros2_system.py complete"
alias ros2_robot_minimal="python3 launch_ros2_system.py minimal"
alias ros2_robot_gui="python3 launch_ros2_system.py gui"
alias ros2_robot_interactive="python3 launch_ros2_system.py interactive"

# Monitoring aliases
alias ros2_robot_validate="python3 ros2_command_gateway_validator.py"
alias ros2_robot_topics="ros2 topic list | grep robot"
alias ros2_robot_services="ros2 service list | grep robot"

echo "🤖 ROS2 Robot System aliases loaded"
echo "Available commands:"
echo "  ros2_robot_complete    - Launch complete system"
echo "  ros2_robot_minimal     - Launch minimal system"  
echo "  ros2_robot_gui         - Launch GUI only"
echo "  ros2_robot_interactive - Interactive launcher"
echo "  ros2_robot_validate    - Validate gateway compliance"
echo "  ros2_robot_topics      - List robot topics"
echo "  ros2_robot_services    - List robot services"
EOF

chmod +x launch_aliases.sh

echo "📋 Build Summary:"
echo "=================="
echo "✅ robot_msgs package built"
echo "✅ Dependencies verified"
echo "✅ Launch scripts created"
echo ""
echo "🚀 To use the system:"
echo "1. Source the environment:"
echo "   source launch_aliases.sh"
echo ""
echo "2. Launch the system:"
echo "   ros2_robot_complete     # Complete system"
echo "   ros2_robot_minimal      # Minimal system"
echo "   ros2_robot_gui          # GUI only"
echo ""
echo "3. Monitor the system:"
echo "   ros2_robot_validate     # Check gateway compliance"
echo "   ros2_robot_topics       # List robot topics"
echo ""
echo "🎉 ROS2 Robot System ready!"
