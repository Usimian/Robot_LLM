#!/usr/bin/env python3
"""
GUI Utility Functions
Common utility functions for the robot GUI
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import Tuple, Optional

from .gui_config import GUIConfig


class GUIUtils:
    """Utility functions for GUI operations"""

    @staticmethod
    def format_distance(distance: float) -> str:
        """Format distance for display"""
        if distance is None or distance < 0:
            return "--"
        return f"{distance:.1f}"

    @staticmethod
    def format_voltage(voltage: float) -> str:
        """Format voltage for display"""
        if voltage is None or voltage <= 0:
            return "--"
        return f"{voltage:.1f}"

    @staticmethod
    def format_temperature(temp: float) -> str:
        """Format temperature for display"""
        if temp is None:
            return "--"
        return f"{temp:.1f}"

    @staticmethod
    def format_cpu_usage(usage: float) -> str:
        """Format CPU usage for display"""
        if usage is None or usage < 0:
            return "--"
        return f"{usage:.1f}"

    @staticmethod
    def get_distance_color(distance: float) -> str:
        """Get color based on distance"""
        if distance is None:
            return GUIConfig.COLORS['disabled']
        elif distance < 0.5:
            return GUIConfig.COLORS['error']  # Very close - danger
        elif distance < 1.0:
            return GUIConfig.COLORS['warning']  # Close - caution
        else:
            return GUIConfig.COLORS['success']  # Safe distance

    @staticmethod
    def get_status_color(status: str) -> str:
        """Get color based on status"""
        color_map = {
            'online': GUIConfig.COLORS['success'],
            'offline': GUIConfig.COLORS['error'],
            'connecting': GUIConfig.COLORS['warning'],
            'processing': GUIConfig.COLORS['warning'],
            'error': GUIConfig.COLORS['error'],
            'success': GUIConfig.COLORS['success']
        }
        return color_map.get(status.lower(), GUIConfig.COLORS['disabled'])

    @staticmethod
    def get_battery_color(voltage: float) -> str:
        """Get color based on battery voltage"""
        if voltage is None:
            return GUIConfig.COLORS['disabled']
        elif voltage < 10.0:
            return GUIConfig.COLORS['error']  # Very low
        elif voltage < 11.5:
            return GUIConfig.COLORS['warning']  # Low
        else:
            return GUIConfig.COLORS['success']  # Good

    @staticmethod
    def get_temperature_color(temp: float) -> str:
        """Get color based on temperature"""
        if temp is None:
            return GUIConfig.COLORS['disabled']
        elif temp > 80:
            return GUIConfig.COLORS['error']  # Too hot
        elif temp > 60:
            return GUIConfig.COLORS['warning']  # Warm
        else:
            return GUIConfig.COLORS['success']  # Normal

    @staticmethod
    def create_colored_label(parent, text: str, color: str) -> ttk.Label:
        """Create a colored label"""
        return ttk.Label(parent, text=text, foreground=color)

    @staticmethod
    def create_status_label(parent, prefix: str, status: str, color: str) -> Tuple[ttk.Label, ttk.Label]:
        """Create a composite status label (prefix + colored status)"""
        frame = ttk.Frame(parent)

        prefix_label = ttk.Label(frame, text=f"{prefix}:", foreground="black")
        prefix_label.pack(side=tk.LEFT)

        status_label = ttk.Label(frame, text=status, foreground=color)
        status_label.pack(side=tk.LEFT, padx=(5, 0))

        return prefix_label, status_label

    @staticmethod
    def update_status_label(status_label: ttk.Label, text: str, color: str):
        """Update a status label with new text and color"""
        status_label.config(text=text, foreground=color)

    @staticmethod
    def set_widget_state(widget, enabled: bool):
        """Set widget enabled/disabled state recursively"""
        state = tk.NORMAL if enabled else tk.DISABLED

        if hasattr(widget, 'config'):
            widget.config(state=state)

        # Handle children for containers
        if hasattr(widget, 'winfo_children'):
            for child in widget.winfo_children():
                GUIUtils.set_widget_state(child, enabled)

    @staticmethod
    def get_current_timestamp() -> str:
        """Get current timestamp for logging"""
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def validate_prompt(prompt: str) -> bool:
        """Validate VILA analysis prompt"""
        if not prompt or not prompt.strip():
            return False
        if len(prompt.strip()) < 5:
            return False
        return True

    @staticmethod
    def truncate_text(text: str, max_length: int = 100) -> str:
        """Truncate text to max length with ellipsis"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."

    @staticmethod
    def format_log_message(level: str, message: str) -> str:
        """Format log message with timestamp and level"""
        timestamp = GUIUtils.get_current_timestamp()
        return f"[{timestamp}] {level}: {message}"

    @staticmethod
    def safe_callback(callback, logger, *args, **kwargs):
        """Safely execute callback with error handling"""
        try:
            return callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Callback error: {e}")
            return None
