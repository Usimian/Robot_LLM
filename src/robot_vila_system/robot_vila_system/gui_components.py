#!/usr/bin/env python3
"""
GUI Component Classes
Separated GUI components for better maintainability
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from typing import Dict, Any, Callable, Optional
from PIL import Image, ImageTk
import math
import time

from .gui_config import GUIConfig
from .gui_utils import GUIUtils


class SystemStatusPanel:
    """Handles system status display panel"""

    def __init__(self, parent, update_callback: Callable, logger=None):
        self.parent = parent
        self.update_callback = update_callback
        self.logger = logger
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
        # VLM Model status
        ttk.Label(parent, text="🤖 Model:", font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=15, pady=2)
        self.status_labels['model'] = ttk.Label(parent, text="Loading...", foreground=GUIConfig.COLORS['warning'], font=('TkDefaultFont', 10, 'bold'))
        self.status_labels['model'].grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

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
            self.logger.error(f"Error updating hardware status: {e}")

    def update_robot_status(self, status_data: Dict[str, Any]):
        """Update robot status display"""
        if 'connection' in status_data:
            color = GUIUtils.get_status_color(status_data['connection'])
            self.status_labels['robot'].config(text=status_data['connection'], foreground=color)

        if 'movement_enabled' in status_data:
            color = GUIConfig.COLORS['success'] if status_data['movement_enabled'] else GUIConfig.COLORS['error']
            text = "Enabled" if status_data['movement_enabled'] else "Disabled"
            self.status_labels['movement'].config(text=text, foreground=color)


    def update_model_status(self, status_data: Dict[str, Any]):
        """Update VLM Model status display"""
        try:
            # Determine status based on model_loaded flag
            if status_data.get('model_loaded', False):
                # Show RoboMP2-enhanced model name when loaded
                model_name = status_data.get('model_name', GUIConfig.DEFAULT_VLM_MODEL)
                # Display as RoboMP2-enhanced system
                if GUIConfig.DEFAULT_VLM_MODEL.split('/')[-1] in model_name:
                    status_text = "RoboMP2 + Qwen2.5-VL-7B"
                else:
                    status_text = f"RoboMP2 + {model_name.split('/')[-1]}"  # Get last part after slash
                color = GUIConfig.COLORS['success']
            elif status_data.get('status') == 'model_load_failed':
                status_text = "Error"
                color = GUIConfig.COLORS['error']
            elif status_data.get('status') == 'loading':
                status_text = "Loading..."
                color = GUIConfig.COLORS['warning']
            else:
                status_text = "Loading..."
                color = GUIConfig.COLORS['warning']

            self.status_labels['model'].config(text=status_text, foreground=color)

        except Exception as e:
            self.logger.error(f"Error updating Model status: {e}")
            self.status_labels['model'].config(text="Error", foreground=GUIConfig.COLORS['error'])


class MovementControlPanel:
    """Handles movement control panel"""

    def __init__(self, parent, command_callback: Callable, logger=None):
        self.parent = parent
        self.command_callback = command_callback
        self.logger = logger
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
        self.imu_frame = ttk.Frame(buttons_frame)
        self.imu_frame.pack(fill=tk.X, pady=(10, 0))

        # IMU header
        ttk.Label(self.imu_frame, text="IMU:", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)
        
        # IMU values (X, Y, Z)
        self.imu_labels = {}
        for axis in ['X', 'Y', 'Z']:
            imu_row = ttk.Frame(self.imu_frame)
            imu_row.pack(fill=tk.X, padx=(20, 0))
            ttk.Label(imu_row, text=f"{axis}:", font=('TkDefaultFont', 10, 'bold'), width=2).pack(side=tk.LEFT)
            self.imu_labels[axis] = ttk.Label(imu_row, text="--.- m/s²   ", foreground=GUIConfig.COLORS['warning'], font=('TkDefaultFont', 10, 'bold'))
            self.imu_labels[axis].pack(side=tk.RIGHT, padx=(5, 0))

        # LiDAR distance display section (below IMU)
        # LiDAR header
        ttk.Label(self.imu_frame, text="LiDAR:", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        
        # LiDAR 270° arc visualization - SIMPLE and STABLE approach
        self.lidar_canvas = tk.Canvas(self.imu_frame, width=180, height=120, bg='black', highlightthickness=1, highlightbackground='gray')
        self.lidar_canvas.pack(pady=(5, 0))
        
        # Data change tracking to prevent unnecessary redraws
        self.last_lidar_hash = None
        # Pre-computed values for performance
        self._lidar_angles_rad = None
        self._lidar_angles_deg = None
        self._canvas_width = 180
        self._canvas_height = 120
        self._center_x = self._canvas_width // 2
        self._center_y = self._canvas_height - (self._canvas_height // 4)
        
        # Initialize LiDAR data storage
        self.max_range = GUIConfig.LIDAR_MAX_RANGE
        self.lidar_ranges = [self.max_range] * 360  # Default to max range
        self.raw_lidar_ranges = None  # Will store actual scan data

        # Rate limiting for LiDAR updates
        self._last_lidar_update_time = 0
        self._min_lidar_update_interval = 0.1  # 10 FPS max for LiDAR display

        # Draw initial arc ONCE
        self._draw_lidar_arc()
        
        # LiDAR range selection dropdown
        range_frame = ttk.Frame(self.imu_frame)
        range_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(range_frame, text="Range:", font=('TkDefaultFont', 9)).pack(side=tk.LEFT)
        
        # Range value display
        self.range_var = tk.DoubleVar(value=3.0)
        self.range_label = ttk.Label(range_frame, text="3.0m", font=('TkDefaultFont', 9))
        self.range_label.pack(side=tk.RIGHT)
        
        # Range slider (0.4m to 5.0m)
        self.range_slider = ttk.Scale(
            range_frame,
            from_=0.4,
            to=5.0,
            variable=self.range_var,
            orient=tk.HORIZONTAL,
            length=80,
            command=self._on_range_slider_changed
        )
        self.range_slider.pack(side=tk.RIGHT, padx=(5, 5))

        # Movement Enable Toggle Button (moved lower to accommodate LiDAR display)
        self.movement_enabled = True
        self.movement_toggle = ttk.Button(
            self.imu_frame,
            text="✅ ENABLED",
            command=self._toggle_movement,
            style='Toggle.TButton'
        )
        self.movement_toggle.pack(fill=tk.X, pady=(10, 0))

        # Configure toggle button style
        style = ttk.Style()
        style.configure('Toggle.TButton', font=('TkDefaultFont', 10, 'bold'), padding=10)



    def _toggle_movement(self):
        """Handle movement enable/disable toggle button"""
        self.movement_enabled = not self.movement_enabled

        if self.movement_enabled:
            self.movement_toggle.config(text="✅ ENABLED")
        else:
            self.movement_toggle.config(text="🔒 DISABLED")

        # Update button states
        self.update_button_states(self.movement_enabled)

        # Notify parent of movement status change
        if self.command_callback:
            self.command_callback(('movement_toggle', self.movement_enabled))
    
    def _on_range_slider_changed(self, value):
        """Handle LiDAR range slider change"""
        try:
            new_range = float(value)
            self.max_range = new_range
            # Update the display label
            self.range_label.config(text=f"{new_range:.1f}m")
            # Redraw the LiDAR display with new range immediately
            self._draw_lidar_arc()
        except (ValueError, AttributeError):
            # Invalid range value, revert to default
            self.max_range = 3.0
            if hasattr(self, 'range_label'):
                self.range_label.config(text="3.0m")
    
    # REMOVED: _schedule_lidar_update - no more periodic updates!
    # LiDAR display now only updates when data actually changes

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
                self.movement_toggle.config(text="✅ ENABLED")
            else:
                self.movement_toggle.config(text="🔒 DISABLED")

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
            self.logger.error(f"Error updating IMU data: {e}")
            # Set error states for IMU labels
            for axis in ['X', 'Y', 'Z']:
                if axis in self.imu_labels:
                    self.imu_labels[axis].config(text="Error", foreground=GUIConfig.COLORS['error'])

    def update_lidar_data(self, lidar_data: Dict[str, Any]):
        """Update LiDAR arc visualization with LiDAR scan data"""
        try:
            # Use LiDAR scan data from /scan topic
            if 'ranges' in lidar_data and lidar_data['ranges']:
                # Store the raw scan data - we'll interpret it correctly in _draw_lidar_arc
                self.raw_lidar_ranges = lidar_data['ranges']

                # Rate limiting to prevent excessive updates
                current_time = time.time()
                if current_time - self._last_lidar_update_time < self._min_lidar_update_interval:
                    return  # Skip this update to maintain smooth performance
                self._last_lidar_update_time = current_time

                # Debug: Log some sample data points with correct indexing (only if debug logging enabled)
                if self.logger and len(self.raw_lidar_ranges) >= 100:
                    mid_idx = len(self.raw_lidar_ranges) // 2
                    quarter_idx = len(self.raw_lidar_ranges) // 4
                    three_quarter_idx = 3 * len(self.raw_lidar_ranges) // 4
                    self.logger.debug(f"LiDAR scan received: {len(self.raw_lidar_ranges)} points, "
                               f"Back(idx=0)={self.raw_lidar_ranges[0]:.2f}m, "
                               f"Front(idx={mid_idx})={self.raw_lidar_ranges[mid_idx]:.2f}m, "
                               f"Left(idx={quarter_idx})={self.raw_lidar_ranges[quarter_idx]:.2f}m, "
                               f"Right(idx={three_quarter_idx})={self.raw_lidar_ranges[three_quarter_idx]:.2f}m")

                # Update display with rate limiting
                self._draw_lidar_arc()
            else:
                # No LiDAR data available
                self.logger.debug("No LiDAR ranges data available")
                self.raw_lidar_ranges = None
        except Exception as e:
            self.logger.error(f"Error updating LiDAR data: {e}")
            self.raw_lidar_ranges = None
    
    def _draw_lidar_arc(self, error=False):
        """Draw LiDAR visualization - SIMPLE approach, only update when data changes"""
        try:
            # Only redraw if data has actually changed
            if hasattr(self, 'raw_lidar_ranges') and self.raw_lidar_ranges:
                # Efficient hash calculation using tuple of rounded values
                sample_data = tuple(round(x, 2) for x in self.raw_lidar_ranges[::10])  # Sample every 10th point, round to 2 decimals
                data_hash = hash(sample_data)
                if data_hash == self.last_lidar_hash:
                    return  # No change, don't redraw
                self.last_lidar_hash = data_hash
            
            # Clear canvas only when we need to redraw
            self.lidar_canvas.delete("all")
            
            # Use pre-computed canvas dimensions
            center_x = self._center_x
            center_y = self._center_y

            # Show error state
            if error:
                self.lidar_canvas.create_text(center_x, self._canvas_height//2, text="LiDAR Error", fill="red", font=('Arial', 10))
                return

            # Draw LiDAR data points - optimized approach
            if hasattr(self, 'raw_lidar_ranges') and self.raw_lidar_ranges:
                num_points = len(self.raw_lidar_ranges)

                # Pre-compute angles if not already done or if scan size changed
                if (self._lidar_angles_rad is None or
                    len(self._lidar_angles_rad) != num_points):
                    self._lidar_angles_rad = []
                    self._lidar_angles_deg = []
                    for i in range(num_points):
                        scan_angle_rad = -math.pi + (i / (num_points - 1)) * (2 * math.pi)
                        self._lidar_angles_rad.append(scan_angle_rad)
                        self._lidar_angles_deg.append(math.degrees(scan_angle_rad))

                # Batch draw points for better performance
                points_to_draw = []

                for i in range(0, num_points, 5):  # Sample every 5th point for performance
                    distance = self.raw_lidar_ranges[i]

                    # Handle invalid/infinite distances
                    if not math.isfinite(distance) or distance <= 0 or distance > self.max_range:
                        continue

                    # Use pre-computed angles
                    scan_angle_rad = self._lidar_angles_rad[i]
                    scan_angle_deg = self._lidar_angles_deg[i]

                    # Skip rear 90° blind spot
                    if -45 <= scan_angle_deg <= 45:
                        continue

                    # Calculate point position - optimized calculations
                    scaled_distance = distance / self.max_range
                    screen_angle_rad = scan_angle_rad - math.pi/2

                    # Use pre-computed canvas boundaries
                    dx = scaled_distance * center_x * math.cos(screen_angle_rad)
                    dy = scaled_distance * center_y * math.sin(screen_angle_rad)

                    # Map to screen coordinates
                    point_x = center_x + dx
                    point_y = center_y - dy

                    # Collect points for batch drawing
                    points_to_draw.append((point_x, point_y))

                # Batch draw all points at once for better performance
                for point_x, point_y in points_to_draw:
                    self.lidar_canvas.create_rectangle(point_x-1, point_y-1, point_x+1, point_y+1,
                                                     fill="red", outline="red", width=0)
            
            # Draw robot as upward-pointing triangle (front = up) at center - using pre-computed values
            triangle_size = 6
            triangle_points = [
                center_x, center_y - triangle_size,      # Top point (front)
                center_x - triangle_size//2, center_y + triangle_size//2,  # Bottom left
                center_x + triangle_size//2, center_y + triangle_size//2   # Bottom right
            ]
            self.lidar_canvas.create_polygon(triangle_points, fill="cyan", outline="blue", width=1)
            
        except Exception as e:
            self.logger.error(f"Error drawing LiDAR arc: {e}")
            self.lidar_canvas.delete("all")
            self.lidar_canvas.create_text(90, 60, text="LiDAR Error", fill="red", font=('Arial', 10))
    

class CameraPanel:
    """Handles camera display panel"""

    def __init__(self, parent, image_callback: Callable, logger=None):
        self.parent = parent
        self.image_callback = image_callback
        self.logger = logger
        self.camera_canvas = None
        
        # Rate limiting for camera updates
        self._last_update_time = 0
        self._min_update_interval = 0.033  # ~30 FPS max to reduce flickering

    def create(self, parent):
        """Create the camera panel"""
        # Main camera frame
        camera_frame = ttk.LabelFrame(parent, text="📹 Camera Feed", padding=10)
        camera_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        camera_frame.pack_propagate(False)
        camera_frame.configure(width=GUIConfig.CAMERA_PANEL_WIDTH, height=GUIConfig.CAMERA_PANEL_HEIGHT)

        # Camera display area with canvas for overlay support
        self.camera_canvas = tk.Canvas(
            camera_frame,
            bg='black',
            highlightthickness=0,
            width=GUIConfig.CAMERA_PANEL_WIDTH - 20,
            height=GUIConfig.CAMERA_PANEL_HEIGHT - 100
        )
        self.camera_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Initial text when no image (will be positioned after canvas is mapped)
        self._show_no_feed_message()

        # Camera controls
        self._create_camera_controls(camera_frame)

        return camera_frame
    
    def _show_no_feed_message(self):
        """Show 'no camera feed' message on canvas"""
        self._show_canvas_message("📷 Waiting for camera...")
    
    def _show_canvas_message(self, message, color='white'):
        """Show a message on the camera canvas"""
        if self.camera_canvas:
            # Clear ALL canvas content when showing a message
            self.camera_canvas.delete("all")

            # Use actual canvas dimensions instead of winfo (which can return 0)
            width = GUIConfig.CAMERA_PANEL_WIDTH - 20  # 420px
            height = GUIConfig.CAMERA_PANEL_HEIGHT - 100  # 300px
            self.camera_canvas.create_text(
                width//2,
                height//2,
                text=message,
                tags="message",
                fill=color,
                font=('TkDefaultFont', 10),
                anchor='center'
            )

    def _create_camera_controls(self, parent):
        """Create camera control buttons"""
        # No source selection needed - only robot camera available





    def update_camera_image(self, pil_image):
        """Update camera display with new image and red line overlay"""
        if pil_image and self.camera_canvas:
            # Rate limiting to reduce flickering
            import time
            current_time = time.time()
            if current_time - self._last_update_time < self._min_update_interval:
                return  # Skip this update
            self._last_update_time = current_time
            
            try:
                # Clear any existing message text before displaying image
                self.camera_canvas.delete("message")

                # Calculate available space for image display
                available_width = GUIConfig.CAMERA_PANEL_WIDTH - 20
                available_height = GUIConfig.CAMERA_PANEL_HEIGHT - 100
                
                # Resize image to fit display area while maintaining aspect ratio
                display_image = pil_image.copy()
                
                # Calculate scaling factor to fit image in available space
                img_width, img_height = display_image.size
                scale_w = available_width / img_width if img_width > 0 else 1
                scale_h = available_height / img_height if img_height > 0 else 1
                scale = min(scale_w, scale_h, 1.0)  # Don't upscale
                
                # Only resize if scaling is needed
                if scale < 1.0:
                    new_width = max(1, int(img_width * scale))
                    new_height = max(1, int(img_height * scale))
                    display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    new_width, new_height = img_width, img_height

                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(display_image)
                
                # Center the image on canvas
                canvas_width = self.camera_canvas.winfo_width() or available_width
                canvas_height = self.camera_canvas.winfo_height() or available_height
                x_offset = (canvas_width - new_width) // 2
                y_offset = (canvas_height - new_height) // 2
                
                # Use configure instead of delete/create to reduce flickering
                if not hasattr(self, '_camera_image_id'):
                    # First time - create the image
                    self._camera_image_id = self.camera_canvas.create_image(
                        x_offset, y_offset, anchor=tk.NW, image=photo, tags="camera_image"
                    )
                    # Create red line overlay at midpoint (50%) of image
                    line_y = y_offset + (new_height * 0.5)
                    line_x1 = x_offset
                    line_x2 = x_offset + new_width
                    
                    self._overlay_line_id = self.camera_canvas.create_line(
                        line_x1, line_y, line_x2, line_y,
                        fill='#800000',  # Dark red (half brightness)
                        width=1,
                        tags="overlay_line"
                    )
                else:
                    # Update existing image and line positions
                    self.camera_canvas.coords(self._camera_image_id, x_offset, y_offset)
                    self.camera_canvas.itemconfig(self._camera_image_id, image=photo)
                    
                    # Update red line position
                    line_y = y_offset + (new_height * 0.5)
                    line_x1 = x_offset
                    line_x2 = x_offset + new_width
                    self.camera_canvas.coords(self._overlay_line_id, line_x1, line_y, line_x2, line_y)
                
                # Keep reference to prevent garbage collection
                self.camera_canvas.image = photo

            except Exception as e:
                # Clear canvas and show error
                error_msg = f"❌ Error displaying image: {str(e)[:50]}"
                self._show_canvas_message(error_msg, color='red')



class ActivityLogPanel:
    """Handles activity log panel"""

    def __init__(self, parent, logger=None):
        self.parent = parent
        self.logger = logger
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


class VLMAnalysisPanel:
    """Handles analysis panel"""

    def __init__(self, parent, analysis_callback: Callable, log_callback: Callable, logger=None):
        self.parent = parent
        self.analysis_callback = analysis_callback
        self.log_callback = log_callback
        self.logger = logger
        self.prompt_text = None
        self.result_text = None
        self.command_label = None
        self.auto_analysis_enabled = False
        self.auto_toggle_button = None

    def create(self, parent):
        """Create the analysis panel"""
        vlm_frame = ttk.LabelFrame(parent, text="🤖 VLM Analysis", padding=10)
        vlm_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Create analysis controls
        self._create_analysis_controls(vlm_frame)

        # Create result display
        self._create_result_display(vlm_frame)

        return vlm_frame

    def _create_analysis_controls(self, parent):
        """Create analysis control buttons and prompt input"""
        # Controls frame
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Auto analysis toggle button (left side)
        self.auto_toggle_button = ttk.Button(
            controls_frame,
            text="🔄 Auto: OFF",
            command=self._toggle_auto_analysis,
            style='AutoToggle.TButton'
        )
        self.auto_toggle_button.pack(side=tk.LEFT, padx=5)
        
        # Auto execute toggle button
        self.auto_execute_enabled = False  # Default disabled for safety
        self.auto_execute_button = ttk.Button(
            controls_frame,
            text="🤖 Execute: OFF",  # Start with OFF to match default state
            command=self._toggle_auto_execute,
            style='AutoToggle.TButton'
        )
        self.auto_execute_button.pack(side=tk.LEFT, padx=5)

        # Analyze button
        ttk.Button(
            controls_frame,
            text="🔍 Analyze",
            command=self._request_analysis
        ).pack(side=tk.RIGHT, padx=5)

        # Configure auto toggle button style
        style = ttk.Style()
        style.configure('AutoToggle.TButton', font=('TkDefaultFont', 10, 'bold'))

        # Prompt input area
        prompt_frame = ttk.Frame(parent)
        prompt_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(prompt_frame, text="Analysis Prompt:").pack(anchor=tk.W)
        self.prompt_text = tk.Text(prompt_frame, height=3, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.X, pady=5)
        self.prompt_text.insert(tk.END, GUIConfig.DEFAULT_VILA_PROMPT)
        
        # Add single-line command display
        command_frame = ttk.Frame(parent)
        command_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(command_frame, text="Current Command:").pack(anchor=tk.W)
        self.command_label = ttk.Label(
            command_frame, 
            text="No command yet", 
            font=('TkDefaultFont', 10, 'bold'),
            foreground='blue'
        )
        self.command_label.pack(anchor=tk.W, pady=(2, 0))

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



    def _toggle_auto_analysis(self):
        """Toggle auto analysis on/off"""
        self.auto_analysis_enabled = not self.auto_analysis_enabled
        
        if self.auto_analysis_enabled:
            self.auto_toggle_button.config(text="🔄 Auto: ON")
            if self.log_callback:
                self.log_callback("🔄 Auto VLM analysis ENABLED")
        else:
            self.auto_toggle_button.config(text="🔄 Auto: OFF")
            if self.log_callback:
                self.log_callback("🔄 Auto VLM analysis DISABLED")
        
        # Notify the parent about auto analysis state change
        if self.analysis_callback:
            self.analysis_callback('auto_analysis_toggle', self.auto_analysis_enabled)
    
    def _toggle_auto_execute(self):
        """Toggle auto execute on/off"""
        self.auto_execute_enabled = not self.auto_execute_enabled
        
        if self.auto_execute_enabled:
            self.auto_execute_button.config(text="🤖 Execute: ON")
            if self.log_callback:
                self.log_callback("🤖 Auto execute VLM commands ENABLED")
        else:
            self.auto_execute_button.config(text="🤖 Execute: OFF")
            if self.log_callback:
                self.log_callback("🤖 Auto execute VLM commands DISABLED")
        
        # Notify the parent about auto execute state change
        if self.analysis_callback:
            self.analysis_callback('auto_execute_toggle', self.auto_execute_enabled)

    def _request_analysis(self):
        """Request analysis"""
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
                self.log_callback(f"🔍 Sending analysis request: {prompt}")

    def update_analysis_result(self, result_data: Dict[str, Any]):
        """Update analysis result display"""
        # Update single-line command display
        if self.command_label and result_data.get('navigation_commands'):
            nav_commands = result_data['navigation_commands']
            action = nav_commands.get('action', 'unknown')
            confidence = nav_commands.get('confidence', 0.0)

            # Format the action nicely
            action_display = {
                'move_forward': 'Move Forward',
                'turn_left': 'Turn Left',
                'turn_right': 'Turn Right',
                'stop': 'Stop',
                'move_backward': 'Move Backward',
                'strafe_left': 'Strafe Left',
                'strafe_right': 'Strafe Right'
            }.get(action, action.title())

            command_text = f"{action_display}, Confidence: {confidence:.2f}"
            self.command_label.config(text=command_text)
        
        if self.result_text:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete('1.0', tk.END)
            
            # Configure bold tag for movement commands
            self.result_text.tag_configure("bold", font=("TkDefaultFont", 10, "bold"))

            # Insert formatted result with enhanced reasoning display
            self.result_text.insert(tk.END, f"Success: {result_data.get('success', False)}\n")
            self.result_text.insert(tk.END, f"{result_data.get('analysis_result', 'N/A')}\n")

            # Display model reasoning prominently
            if result_data.get('reasoning'):
                self.result_text.insert(tk.END, "\n🤖 Model Reasoning:\n")
                self.result_text.insert(tk.END, f"{result_data['reasoning']}\n")

            # Display full analysis if available
            if result_data.get('full_analysis'):
                self.result_text.insert(tk.END, "\n📋 Complete Analysis:\n")
                self.result_text.insert(tk.END, f"{result_data['full_analysis']}\n")

            # Handle navigation commands with bold formatting
            if result_data.get('navigation_commands'):
                self.result_text.insert(tk.END, "\nNavigation: ")
                
                # Parse navigation commands to find movement commands
                nav_commands = result_data['navigation_commands']
                nav_text = str(nav_commands)
                
                # Look for common movement commands and make them bold
                movement_commands = ['move_forward', 'move_backward', 'turn_left', 'turn_right', 'stop', 'strafe_left', 'strafe_right']
                
                # Insert navigation text with bold movement commands
                remaining_text = nav_text
                while remaining_text:
                    # Find the next movement command
                    earliest_pos = len(remaining_text)
                    earliest_cmd = None
                    
                    for cmd in movement_commands:
                        pos = remaining_text.find(cmd)
                        if pos != -1 and pos < earliest_pos:
                            earliest_pos = pos
                            earliest_cmd = cmd
                    
                    if earliest_cmd:
                        # Insert text before the command normally
                        if earliest_pos > 0:
                            self.result_text.insert(tk.END, remaining_text[:earliest_pos])
                        
                        # Insert the movement command in bold
                        self.result_text.insert(tk.END, earliest_cmd, "bold")
                        
                        # Continue with the rest of the text
                        remaining_text = remaining_text[earliest_pos + len(earliest_cmd):]
                    else:
                        # No more movement commands, insert the rest normally
                        self.result_text.insert(tk.END, remaining_text)
                        break
                
                self.result_text.insert(tk.END, "\n")

            self.result_text.insert(tk.END, f"\nConfidence: {result_data.get('confidence', 0.0)}\n")
            self.result_text.insert(tk.END, f"Timestamp: {result_data.get('timestamp_ns', 0)}\n")

            self.result_text.config(state=tk.DISABLED)
