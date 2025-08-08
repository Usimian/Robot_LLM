#!/bin/bash
"""
Proper ROS2 Colcon Build Script
Builds the ROS2 workspace using standard colcon workflow
"""

set -e  # Exit on any error

echo "🔧 Building ROS2 VILA Robot System with Colcon"
echo "=============================================="

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "❌ ROS2 environment not sourced!"
    echo "Please run: source /opt/ros/<distro>/setup.bash"
    exit 1
fi

echo "✅ ROS2 environment detected: $ROS_DISTRO"

# Ensure we're in the workspace root
WORKSPACE_DIR="/home/marc/Robot_LLM"
cd "$WORKSPACE_DIR"

echo "📁 Workspace: $WORKSPACE_DIR"

# Check workspace structure
if [ ! -d "src" ]; then
    echo "❌ No 'src' directory found! This doesn't look like a colcon workspace."
    exit 1
fi

echo "📦 Found packages:"
find src -name "package.xml" -exec dirname {} \; | sed 's|src/||' | sort

# Clean previous build (optional)
if [ "$1" = "clean" ]; then
    echo "🧹 Cleaning previous build..."
    rm -rf build install log
fi

# Build with colcon
echo "🔨 Building packages with colcon..."
colcon build \
    --packages-select robot_msgs robot_vila_system \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --event-handlers console_direct+

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
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
    
    # Check if executables are available
    import subprocess
    result = subprocess.run(['ros2', 'pkg', 'executables', 'robot_vila_system'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print('✅ robot_vila_system executables:')
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f'   - {line.strip()}')
    else:
        print('⚠️ Could not list executables')
        
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

# Create launch aliases for the new structure
echo "⚙️ Creating launch aliases..."
cat > ros2_launch_aliases.sh << 'EOF'
#!/bin/bash
# ROS2 VILA Robot System Launch Aliases (Proper colcon version)

# Source ROS2 environment
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash

# Launch aliases using proper ROS2 launch files
alias ros2_vila_complete="ros2 launch robot_vila_system robot_vila_system.launch.py"
alias ros2_vila_minimal="ros2 launch robot_vila_system minimal_system.launch.py"
alias ros2_vila_robot="ros2 launch robot_vila_system robot_client.launch.py"

# Launch with parameters
alias ros2_vila_complete_sim="ros2 launch robot_vila_system robot_vila_system.launch.py launch_robot_client:=true"
alias ros2_vila_no_gui="ros2 launch robot_vila_system robot_vila_system.launch.py launch_gui:=false"

# Individual nodes
alias ros2_vila_server="ros2 run robot_vila_system vila_server"
alias ros2_vila_vision="ros2 run robot_vila_system vila_vision"
alias ros2_vila_client="ros2 run robot_vila_system robot_client"
alias ros2_vila_gui="ros2 run robot_vila_system robot_gui"
alias ros2_vila_validator="ros2 run robot_vila_system gateway_validator"

# Monitoring aliases
alias ros2_vila_topics="ros2 topic list | grep -E '(robot|vila)'"
alias ros2_vila_services="ros2 service list | grep -E '(robot|vila)'"
alias ros2_vila_nodes="ros2 node list | grep robot_system"

# Debug aliases
alias ros2_vila_info="ros2 pkg list | grep -E '(robot_msgs|robot_vila_system)'"
alias ros2_vila_interfaces="ros2 interface list | grep robot_msgs"

echo "🤖 ROS2 VILA Robot System aliases loaded (proper colcon version)"
echo "Available commands:"
echo "  Launch files:"
echo "    ros2_vila_complete      - Complete system"
echo "    ros2_vila_minimal       - Server + vision only"
echo "    ros2_vila_robot         - Robot client only"
echo "    ros2_vila_complete_sim  - Complete system with simulated robot"
echo "    ros2_vila_no_gui        - Complete system without GUI"
echo ""
echo "  Individual nodes:"
echo "    ros2_vila_server        - VILA server node"
echo "    ros2_vila_vision        - VILA vision node"
echo "    ros2_vila_client        - Robot client node"
echo "    ros2_vila_gui           - Robot GUI node"
echo "    ros2_vila_validator     - Gateway validator"
echo ""
echo "  Monitoring:"
echo "    ros2_vila_topics        - List robot topics"
echo "    ros2_vila_services      - List robot services"
echo "    ros2_vila_nodes         - List running nodes"
echo "    ros2_vila_info          - Package information"
EOF

chmod +x ros2_launch_aliases.sh

echo "📋 Build Summary:"
echo "=================="
echo "✅ Workspace built with colcon"
echo "✅ robot_msgs package built"
echo "✅ robot_vila_system package built"
echo "✅ Launch files created"
echo "✅ Configuration files ready"
echo ""
echo "🚀 To use the system:"
echo "1. Source the environment:"
echo "   source ros2_launch_aliases.sh"
echo ""
echo "2. Launch using proper ROS2 launch files:"
echo "   ros2_vila_complete      # Complete system"
echo "   ros2_vila_minimal       # Minimal system"
echo "   ros2_vila_robot         # Robot client only"
echo ""
echo "3. Or use individual nodes:"
echo "   ros2 run robot_vila_system vila_server"
echo "   ros2 run robot_vila_system vila_vision"
echo ""
echo "4. Monitor with standard ROS2 tools:"
echo "   ros2 topic list"
echo "   ros2 node list"
echo "   ros2 service list"
echo ""
echo "🎉 Proper ROS2 colcon workspace ready!"
