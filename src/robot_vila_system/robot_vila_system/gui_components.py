#!/usr/bin/env python3
"""
GUI Component Classes
Separated GUI components for better maintainability
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from PIL import Image, ImageTk
import cv2
import numpy as np
import logging

from .gui_config import GUIConfig
from .gui_utils import GUIUtils

logger = logging.getLogger('GUIComponents')


class SystemStatusPanel:
    """Handles system status display panel"""

    def __init__(self, parent, update_callback: Callable):
        self.parent = parent
        self.update_callback = update_callback
        self.status_labels = {}

    def create(self, parent):
        """Create the system status panel"""
        # Main status frame
        self.frame = ttk.LabelFrame(parent, text="🖥️ System Status", padding=10, labelanchor='n')
        self.frame.pack(fill=tk.X, pady=(0, 10))

        # Create status grid
        self._create_status_grid(self.frame)

        return self.frame

    def _create_status_grid(self, parent):
        """Create grid layout for status items - 2 columns of 3 values"""
        # Left Column
        # Robot status
        ttk.Label(parent, text="🤖 Robot:", font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.status_labels['robot'] = ttk.Label(parent, text=GUIConfig.OFFLINE_TEXT, foreground=GUIConfig.COLORS['error'], font=('TkDefaultFont', 10, 'bold'))
        self.status_labels['robot'].grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # CPU Temperature
        ttk.Label(parent, text="🌡️ CPU Temp:", font=('TkDefaultFont', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.status_labels['cpu_temp'] = ttk.Label(parent, text="--°C", foreground=GUIConfig.COLORS['warning'], font=('TkDefaultFont', 10, 'bold'))
        self.status_labels['cpu_temp'].grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # CPU Load (under CPU Temp, no lightning bolt)
        ttk.Label(parent, text="💻 CPU Load:", font=('TkDefaultFont', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.status_labels['cpu_usage'] = ttk.Label(parent, text="--%", foreground=GUIConfig.COLORS['warning'], font=('TkDefaultFont', 10, 'bold'))
        self.status_labels['cpu_usage'].grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        # Right Column
        # Cosmos-Transfer1 status
        ttk.Label(parent, text="🌌 Cosmos:", font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=15, pady=2)
        self.status_labels['cosmos'] = ttk.Label(parent, text=GUIConfig.OFFLINE_TEXT, foreground=GUIConfig.COLORS['error'], font=('TkDefaultFont', 10, 'bold'))
        self.status_labels['cosmos'].grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        # Movement status
        ttk.Label(parent, text="🚶 Movement:", font=('TkDefaultFont', 10, 'bold')).grid(row=1, column=2, sticky=tk.W, padx=15, pady=2)
        self.status_labels['movement'] = ttk.Label(parent, text="Disabled", foreground=GUIConfig.COLORS['error'], font=('TkDefaultFont', 10, 'bold'))
        self.status_labels['movement'].grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)

        # Battery Voltage
        ttk.Label(parent, text="🔋 Battery:", font=('TkDefaultFont', 10, 'bold')).grid(row=2, column=2, sticky=tk.W, padx=15, pady=2)
        self.status_labels['battery'] = ttk.Label(parent, text="--V", foreground=GUIConfig.COLORS['warning'], font=('TkDefaultFont', 10, 'bold'))
        self.status_labels['battery'].grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)

        # Configure grid weights for proper spacing
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(3, weight=1)

    def update_sensor_data(self, sensor_data: Dict[str, Any]):
        """Update sensor data display (alias for update_hardware_status)"""
        self.update_hardware_status(sensor_data)

    def update_hardware_status(self, sensor_data: Dict[str, Any]):
        """Update hardware status with real sensor data"""
        try:
            # Update CPU temperature
            if 'cpu_temp' in sensor_data:
                temp = sensor_data['cpu_temp']
                color = GUIConfig.COLORS['error'] if temp > 70 else GUIConfig.COLORS['success']
                self.status_labels['cpu_temp'].config(text=f"{temp:.0f}°C", foreground=color)

            # Update CPU load/usage
            if 'cpu_usage' in sensor_data:
                usage = sensor_data['cpu_usage']
                color = GUIConfig.COLORS['error'] if usage > 80 else GUIConfig.COLORS['success']
                self.status_labels['cpu_usage'].config(text=f"{usage:.0f}%", foreground=color)

            # Update battery voltage
            if 'battery_voltage' in sensor_data:
                voltage = sensor_data['battery_voltage']
                color = GUIConfig.COLORS['error'] if voltage < 11.0 else GUIConfig.COLORS['success']
                self.status_labels['battery'].config(text=f"{voltage:.2f}V", foreground=color)



        except Exception as e:
            logger.error(f"Error updating hardware status: {e}")

    def update_robot_status(self, status_data: Dict[str, Any]):
        """Update robot status display"""
        if 'connection' in status_data:
            color = GUIUtils.get_status_color(status_data['connection'])
            self.status_labels['robot'].config(text=status_data['connection'], foreground=color)

        if 'movement_enabled' in status_data:
            color = GUIConfig.COLORS['success'] if status_data['movement_enabled'] else GUIConfig.COLORS['error']
            text = "Enabled" if status_data['movement_enabled'] else "Disabled"
            self.status_labels['movement'].config(text=text, foreground=color)

    def update_sensor_data(self, sensor_data: Dict[str, Any]):
        """Update sensor data display"""
        # This method will be expanded to handle all sensor data updates
        # that were previously done by the old monolithic methods
        pass

    def update_cosmos_status(self, status_data: Dict[str, Any]):
        """Update Cosmos status display"""
        try:
            # Determine status based on model_loaded flag
            if status_data.get('model_loaded', False):
                status_text = "Online"
                color = GUIConfig.COLORS['success']
            elif status_data.get('status') == 'model_load_failed':
                status_text = "Error"
                color = GUIConfig.COLORS['error']
            elif status_data.get('status') == 'loading':
                status_text = "Loading..."
                color = GUIConfig.COLORS['warning']
            else:
                status_text = "Offline"
                color = GUIConfig.COLORS['error']

            self.status_labels['cosmos'].config(text=status_text, foreground=color)

        except Exception as e:
            logger.error(f"Error updating Cosmos status: {e}")
            self.status_labels['cosmos'].config(text="Error", foreground=GUIConfig.COLORS['error'])


class MovementControlPanel:
    """Handles movement control panel"""

    def __init__(self, parent, command_callback: Callable):
        self.parent = parent
        self.command_callback = command_callback
        self.buttons = {}

    def create(self, parent):
        """Create the movement control panel"""
        # Main movement frame
        movement_frame = ttk.LabelFrame(parent, text="🎮 Movement Controls", padding=10)
        movement_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        movement_frame.pack_propagate(False)
        movement_frame.configure(width=GUIConfig.MOVEMENT_PANEL_WIDTH)

        # Create movement buttons
        self._create_movement_buttons(movement_frame)

        return movement_frame

    def _create_movement_buttons(self, parent):
        """Create movement control buttons"""
        # Movement buttons frame
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))

        # Forward button
        self.buttons['forward'] = ttk.Button(
            buttons_frame,
            text="⬆️",
            command=lambda: self.command_callback('move_forward')
        )
        self.buttons['forward'].pack(fill=tk.X, pady=2)

        # Left/Right buttons frame
        lr_frame = ttk.Frame(buttons_frame)
        lr_frame.pack(fill=tk.X, pady=2)

        # Configure grid weights for equal sizing
        lr_frame.grid_columnconfigure(0, weight=1)
        lr_frame.grid_columnconfigure(1, weight=1)

        self.buttons['left'] = ttk.Button(
            lr_frame,
            text="⬅️",
            command=lambda: self.command_callback('turn_left')
        )
        self.buttons['left'].grid(row=0, column=0, sticky=tk.EW, padx=(0, 1))

        self.buttons['right'] = ttk.Button(
            lr_frame,
            text="➡️",
            command=lambda: self.command_callback('turn_right')
        )
        self.buttons['right'].grid(row=0, column=1, sticky=tk.EW, padx=(1, 0))

        # Backward button
        self.buttons['backward'] = ttk.Button(
            buttons_frame,
            text="⬇️",
            command=lambda: self.command_callback('move_backward')
        )
        self.buttons['backward'].pack(fill=tk.X, pady=2)

        # Strafe buttons frame
        strafe_frame = ttk.Frame(buttons_frame)
        strafe_frame.pack(fill=tk.X, pady=2)

        # Configure grid weights for equal sizing
        strafe_frame.grid_columnconfigure(0, weight=1)
        strafe_frame.grid_columnconfigure(1, weight=1)

        self.buttons['strafe_left'] = ttk.Button(
            strafe_frame,
            text="↖️",
            command=lambda: self.command_callback('strafe_left')
        )
        self.buttons['strafe_left'].grid(row=0, column=0, sticky=tk.EW, padx=(0, 1))

        self.buttons['strafe_right'] = ttk.Button(
            strafe_frame,
            text="↗️",
            command=lambda: self.command_callback('strafe_right')
        )
        self.buttons['strafe_right'].grid(row=0, column=1, sticky=tk.EW, padx=(1, 0))

        # Stop button (emphasized)
        self.buttons['stop'] = ttk.Button(
            buttons_frame,
            text="⏹️ STOP",
            command=lambda: self.command_callback('stop')
        )
        self.buttons['stop'].pack(fill=tk.X, pady=2)

        # IMU display section (under Stop button)
        imu_frame = ttk.Frame(buttons_frame)
        imu_frame.pack(fill=tk.X, pady=(10, 0))

        # IMU header
        ttk.Label(imu_frame, text="IMU:", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)
        
        # IMU values (X, Y, Z)
        self.imu_labels = {}
        for axis in ['X', 'Y', 'Z']:
            imu_row = ttk.Frame(imu_frame)
            imu_row.pack(fill=tk.X, padx=(20, 0))
            ttk.Label(imu_row, text=f"{axis}:", font=('TkDefaultFont', 10, 'bold'), width=2).pack(side=tk.LEFT)
            self.imu_labels[axis] = ttk.Label(imu_row, text="--.- m/s²   ", foreground=GUIConfig.COLORS['warning'], font=('TkDefaultFont', 10, 'bold'))
            self.imu_labels[axis].pack(side=tk.RIGHT, padx=(5, 0))

        # LiDAR distance display section (below IMU)
        # LiDAR header
        ttk.Label(imu_frame, text="LiDAR:", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        
        # LiDAR values (Front, Left, Right)
        self.lidar_labels = {}
        for direction in ['F', 'L', 'R']:  # Front, Left, Right
            lidar_row = ttk.Frame(imu_frame)
            lidar_row.pack(fill=tk.X, padx=(20, 0))
            ttk.Label(lidar_row, text=f"{direction}:", font=('TkDefaultFont', 10, 'bold'), width=2).pack(side=tk.LEFT)
            self.lidar_labels[direction] = ttk.Label(lidar_row, text="--.- m", foreground=GUIConfig.COLORS['warning'], font=('TkDefaultFont', 10, 'bold'))
            self.lidar_labels[direction].pack(side=tk.LEFT, padx=(5, 0))

        # Movement Enable Toggle Button (below LiDAR)
        self.movement_enabled = True
        self.movement_toggle = ttk.Button(
            imu_frame,
            text="✅ ENABLED - Click to Disable Movement",
            command=self._toggle_movement,
            style='Toggle.TButton'
        )
        self.movement_toggle.pack(fill=tk.X, pady=(15, 0))

        # Configure toggle button style
        style = ttk.Style()
        style.configure('Toggle.TButton', font=('TkDefaultFont', 10, 'bold'), padding=10)



    def _toggle_movement(self):
        """Handle movement enable/disable toggle button"""
        self.movement_enabled = not self.movement_enabled

        if self.movement_enabled:
            self.movement_toggle.config(text="✅ ENABLED - Click to Disable Movement")
        else:
            self.movement_toggle.config(text="🔒 DISABLED - Click to Enable Movement")

        # Update button states
        self.update_button_states(self.movement_enabled)

        # Notify parent of movement status change
        if self.command_callback:
            self.command_callback(('movement_toggle', self.movement_enabled))

    def _on_safety_toggle(self):
        """Handle safety checkbox toggle"""
        is_enabled = self.safety_var.get()
        self.update_button_states(is_enabled)

        # Notify parent of safety status change
        if self.command_callback:
            # Send safety toggle command with current state
            self.command_callback(('safety_toggle', is_enabled))

    def update_button_states(self, movement_enabled: bool):
        """Update button states based on movement status"""
        # Update movement enabled state
        self.movement_enabled = movement_enabled

        # Update toggle button appearance
        if hasattr(self, 'movement_toggle'):
            if movement_enabled:
                self.movement_toggle.config(text="✅ ENABLED - Click to Disable Movement")
            else:
                self.movement_toggle.config(text="🔒 DISABLED - Click to Enable Movement")

        # Enable/disable movement buttons based on movement status
        # Stop button is always enabled
        movement_buttons = ['forward', 'backward', 'left', 'right', 'strafe_left', 'strafe_right']
        for button_name in movement_buttons:
            if button_name in self.buttons:
                GUIUtils.set_widget_state(self.buttons[button_name], movement_enabled)

        # Stop button is always enabled for safety
        if 'stop' in self.buttons:
            GUIUtils.set_widget_state(self.buttons['stop'], True)

    def set_command_callback(self, callback: Callable):
        """Update the command callback"""
        self.command_callback = callback

    def update_imu_data(self, sensor_data: Dict[str, Any]):
        """Update IMU display with sensor data"""
        try:
            # Update IMU acceleration data (x, y, z)
            if 'accel_x' in sensor_data:
                self.imu_labels['X'].config(text=f"{sensor_data['accel_x']:.2f} m/s²", foreground=GUIConfig.COLORS['success'])
            if 'accel_y' in sensor_data:
                self.imu_labels['Y'].config(text=f"{sensor_data['accel_y']:.2f} m/s²", foreground=GUIConfig.COLORS['success'])
            if 'accel_z' in sensor_data:
                self.imu_labels['Z'].config(text=f"{sensor_data['accel_z']:.2f} m/s²", foreground=GUIConfig.COLORS['success'])
        except Exception as e:
            logger.error(f"Error updating IMU data: {e}")
            # Set error states for IMU labels
            for axis in ['X', 'Y', 'Z']:
                if axis in self.imu_labels:
                    self.imu_labels[axis].config(text="Error", foreground=GUIConfig.COLORS['error'])

    def update_lidar_data(self, sensor_data: Dict[str, Any]):
        """Update LiDAR display with sensor data"""
        try:
            # Update LiDAR distance data (front, left, right)
            if 'distance_front' in sensor_data:
                distance = sensor_data['distance_front']
                color = GUIConfig.COLORS['error'] if distance < 0.5 else GUIConfig.COLORS['success']
                self.lidar_labels['F'].config(text=f"{distance:.2f} m", foreground=color)
            if 'distance_left' in sensor_data:
                distance = sensor_data['distance_left']
                color = GUIConfig.COLORS['error'] if distance < 0.5 else GUIConfig.COLORS['success']
                self.lidar_labels['L'].config(text=f"{distance:.2f} m", foreground=color)
            if 'distance_right' in sensor_data:
                distance = sensor_data['distance_right']
                color = GUIConfig.COLORS['error'] if distance < 0.5 else GUIConfig.COLORS['success']
                self.lidar_labels['R'].config(text=f"{distance:.2f} m", foreground=color)
        except Exception as e:
            logger.error(f"Error updating LiDAR data: {e}")
            # Set error states for LiDAR labels
            for direction in ['F', 'L', 'R']:
                if direction in self.lidar_labels:
                    self.lidar_labels[direction].config(text="Error", foreground=GUIConfig.COLORS['error'])


class CameraPanel:
    """Handles camera display panel"""

    def __init__(self, parent, image_callback: Callable):
        self.parent = parent
        self.image_callback = image_callback
        self.camera_label = None
        self.source_var = None
        self.load_button = None

    def create(self, parent):
        """Create the camera panel"""
        # Main camera frame
        camera_frame = ttk.LabelFrame(parent, text="📹 Camera Feed", padding=10)
        camera_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        camera_frame.pack_propagate(False)
        camera_frame.configure(width=GUIConfig.CAMERA_PANEL_WIDTH, height=GUIConfig.CAMERA_PANEL_HEIGHT)

        # Camera display area with no border
        self.camera_label = ttk.Label(
            camera_frame, 
            text="📹 No camera feed",
            anchor='center',
            justify='center'
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Camera controls
        self._create_camera_controls(camera_frame)

        return camera_frame

    def _create_camera_controls(self, parent):
        """Create camera control buttons"""
        # Source selection
        source_frame = ttk.Frame(parent)
        source_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(source_frame, text="Source:").pack(side=tk.LEFT, padx=(0, 5))

        self.source_var = tk.StringVar(value=GUIConfig.DEFAULT_CAMERA_SOURCE)
        robot_radio = ttk.Radiobutton(
            source_frame,
            text="Robot",
            variable=self.source_var,
            value="robot",
            command=self._on_source_change
        )
        robot_radio.pack(side=tk.LEFT, padx=(0, 10))

        loaded_radio = ttk.Radiobutton(
            source_frame,
            text="Loaded",
            variable=self.source_var,
            value="loaded",
            command=self._on_source_change
        )
        loaded_radio.pack(side=tk.LEFT)

        # Load image button
        self.load_button = ttk.Button(
            parent,
            text="📁 Load Image",
            command=self._load_image_file
        )
        self.load_button.pack(fill=tk.X, pady=(5, 0))

    def _on_source_change(self):
        """Handle camera source change"""
        source = self.source_var.get()
        if self.image_callback:
            self.image_callback('camera_source_changed', source)

    def _load_image_file(self):
        """Load image file from disk"""
        file_path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )

        if file_path and self.image_callback:
            self.image_callback('image_loaded', file_path)

    def update_camera_image(self, pil_image):
        """Update camera display with new image"""
        if pil_image and self.camera_label:
            try:
                # Calculate available space for image display
                available_width = GUIConfig.CAMERA_PANEL_WIDTH - 20
                available_height = GUIConfig.CAMERA_PANEL_HEIGHT - 100
                
                # Resize image to fit display area while maintaining aspect ratio
                display_image = pil_image.copy()
                
                # Calculate scaling factor to fit image in available space
                img_width, img_height = display_image.size
                scale_w = available_width / img_width
                scale_h = available_height / img_height
                scale = min(scale_w, scale_h)
                
                # Resize image with proper scaling
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(display_image)
                self.camera_label.config(image=photo, text="")
                self.camera_label.image = photo  # Keep reference

            except Exception as e:
                self.camera_label.config(text=f"❌ Error displaying image: {str(e)}", image="")

    def set_source(self, source: str):
        """Set camera source"""
        if self.source_var and source in GUIConfig.CAMERA_SOURCES:
            self.source_var.set(source)


# VILAAnalysisPanel removed - using Cosmos-Transfer1 directly


class ActivityLogPanel:
    """Handles activity log panel"""

    def __init__(self, parent):
        self.parent = parent
        self.log_text = None

    def create(self, parent):
        """Create the activity log panel"""
        # Log frame
        log_frame = ttk.LabelFrame(parent, text="📋 Activity Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        # Scrolled text widget for log
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            height=GUIConfig.LOG_HEIGHT,
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        return log_frame

    def log_message(self, message: str):
        """Add message to log"""
        if not self.log_text:
            return

        # Get current timestamp
        timestamp = GUIUtils.get_current_timestamp()

        # Format message
        formatted_message = f"[{timestamp}] {message}\n"

        # Add to log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)  # Scroll to bottom
        self.log_text.config(state=tk.DISABLED)

        # Limit log size
        if float(self.log_text.index('end-1c').split('.')[0]) > GUIConfig.MAX_LOG_LINES:
            self._trim_log()

    def _trim_log(self):
        """Trim log to prevent excessive memory usage"""
        if not self.log_text:
            return

        # Keep only the last MAX_LOG_LINES lines
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', f'{GUIConfig.MAX_LOG_LINES}.0')
        self.log_text.config(state=tk.DISABLED)

    def clear_log(self):
        """Clear the log"""
        if self.log_text:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete('1.0', tk.END)
            self.log_text.config(state=tk.DISABLED)


class CosmosAnalysisPanel:
    """Handles Cosmos-Transfer1 analysis panel"""

    def __init__(self, parent, analysis_callback: Callable, log_callback: Callable):
        self.parent = parent
        self.analysis_callback = analysis_callback
        self.log_callback = log_callback
        self.prompt_text = None
        self.result_text = None
        self.load_button = None

    def create(self, parent):
        """Create the Cosmos-Transfer1 analysis panel"""
        # Main Cosmos frame
        cosmos_frame = ttk.LabelFrame(parent, text="🌌 Cosmos-Transfer1 Analysis", padding=10)
        cosmos_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Create analysis controls
        self._create_analysis_controls(cosmos_frame)

        # Create result display
        self._create_result_display(cosmos_frame)

        return cosmos_frame

    def _create_analysis_controls(self, parent):
        """Create analysis control buttons and prompt input"""
        # Controls frame
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Load image button
        self.load_button = ttk.Button(
            controls_frame,
            text="📁 Load Image",
            command=self._load_image_file
        )
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Analyze button
        ttk.Button(
            controls_frame,
            text="🔍 Analyze",
            command=self._request_analysis
        ).pack(side=tk.RIGHT, padx=5)

        # Prompt input area
        prompt_frame = ttk.Frame(parent)
        prompt_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(prompt_frame, text="Analysis Prompt:").pack(anchor=tk.W)
        self.prompt_text = tk.Text(prompt_frame, height=3, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.X, pady=5)
        self.prompt_text.insert(tk.END, GUIConfig.DEFAULT_VILA_PROMPT)

    def _create_result_display(self, parent):
        """Create analysis result display area"""
        # Result frame
        result_frame = ttk.LabelFrame(parent, text="Analysis Results", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Result text area
        self.result_text = tk.Text(result_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.config(yscrollcommand=scrollbar.set)

        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _load_image_file(self):
        """Load image from file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
                    ("All files", "*.*")
                ]
            )

            if file_path and self.analysis_callback:
                self.analysis_callback('load_image', file_path)
                if self.log_callback:
                    self.log_callback(f"📁 Image loaded: {file_path}")

        except Exception as e:
            if self.log_callback:
                self.log_callback(f"❌ Failed to load image: {e}")

    def _request_analysis(self):
        """Request Cosmos-Transfer1 analysis"""
        if not hasattr(self, 'prompt_text') or not self.prompt_text:
            return

        prompt = self.prompt_text.get("1.0", tk.END).strip()

        if not GUIUtils.validate_prompt(prompt):
            if self.log_callback:
                self.log_callback("❌ Please enter a valid prompt (minimum 5 characters)")
            return

        if self.analysis_callback:
            self.analysis_callback('request_analysis', prompt)
            if self.log_callback:
                self.log_callback(f"🔍 Sending analysis request: {prompt[:50]}...")

    def update_analysis_result(self, result_data: Dict[str, Any]):
        """Update analysis result display"""
        if self.result_text:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete('1.0', tk.END)

            # Format the result
            result_text = f"Analysis Result:\n"
            result_text += f"Success: {result_data.get('success', False)}\n"
            result_text += f"Analysis: {result_data.get('analysis_result', 'N/A')}\n"

            if result_data.get('navigation_commands'):
                result_text += f"\nNavigation: {result_data['navigation_commands']}\n"

            result_text += f"\nConfidence: {result_data.get('confidence', 0.0)}\n"
            result_text += f"Timestamp: {result_data.get('timestamp_ns', 0)}\n"

            self.result_text.insert(tk.END, result_text)
            self.result_text.config(state=tk.DISABLED)

    def clear_results(self):
        """Clear analysis results"""
        if self.result_text:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete('1.0', tk.END)
            self.result_text.config(state=tk.DISABLED)
