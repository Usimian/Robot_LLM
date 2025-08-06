#!/bin/bash
# Unified Robot Controller Startup Script

echo "🚀 Starting Unified Robot Controller..."

# Check if Python dependencies are installed
python3 -c "import aiohttp, flask_socketio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip3 install aiohttp flask-socketio
fi

# Start unified controller
cd "$(dirname "$0")"
python3 unified_robot_controller.py

echo "👋 Unified Robot Controller stopped"