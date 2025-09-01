#!/usr/bin/env python3
"""
GUI Configuration Constants
Centralized configuration for the robot GUI
"""


class GUIConfig:
    """Centralized GUI configuration constants"""

    # Window settings
    WINDOW_TITLE = "Robot Nemotron Client"
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT =1000

    # Colors
    COLORS = {
        'success': 'green',
        'error': 'red',
        'warning': 'orange',
        'info': 'blue',
        'disabled': 'gray',
        'background': 'white'
    }

    # Panel dimensions
    MOVEMENT_PANEL_WIDTH = 200
    CAMERA_PANEL_WIDTH = 440
    CAMERA_PANEL_HEIGHT = 400
    STATUS_PANEL_HEIGHT = 120
    
    # LiDAR display settings
    LIDAR_MAX_RANGE = 2.5  # Maximum LiDAR range in meters

    # Font settings
    FONT_FAMILY = "Arial"
    FONT_SIZE_NORMAL = 10
    FONT_SIZE_HEADER = 12
    FONT_SIZE_LARGE = 14

    # Timing settings
    UPDATE_INTERVAL_MS = 1000  # 1 second
    ROS_START_DELAY_MS = 500   # 0.5 seconds

    # VILA settings
    DEFAULT_VILA_PROMPT = "Analyze the current camera view for navigation."
    COSMOS_AUTO_INTERVAL = 3.0  # seconds

    # QoS settings
    IMAGE_QOS_DEPTH = 10
    RELIABLE_QOS_DEPTH = 10
    BEST_EFFORT_QOS_DEPTH = 10

    # Button text templates
    MOVEMENT_BUTTON_TEMPLATE = "⬆️ Forward\n⬇️ Backward\n⬅️ Left\n➡️ Right\n⏹️ Stop"
    SAFETY_BUTTON_TEMPLATE = "🛡️ Safety: {}"
    VILA_AUTO_BUTTON_TEMPLATE = "🔄 Auto Analysis: {}"

    # Status text templates
    OFFLINE_TEXT = "Offline"
    ONLINE_TEXT = "Online"
    PROCESSING_TEXT = "Processing..."
    CONNECTING_TEXT = "Connecting..."

    # Camera settings
    CAMERA_SOURCES = ["robot", "loaded"]
    DEFAULT_CAMERA_SOURCE = "robot"

    # Log settings
    MAX_LOG_LINES = 1000
    LOG_HEIGHT = 8

    # Debug settings
    DEBUG_MODE = True
