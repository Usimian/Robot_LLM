#!/usr/bin/env python3
"""
Robot Control GUI
Tkinter application for monitoring and controlling robots via robot_vila_server.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import json
import requests
import socketio
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import base64
import io
from typing import Dict, List, Optional
import logging
import queue
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RobotGUI')

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "localhost"
    http_port: int = 5000
    tcp_port: int = 9999
    
    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.http_port}"
    
    @property
    def websocket_url(self) -> str:
        return f"http://{self.host}:{self.http_port}"

class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.base_title = "Robot Control Hub - VILA Integration"
        self.root.title(self.base_title)
        self.root.geometry("1200x800")
        
        # Configuration
        self.config = ServerConfig()
        
        # Communication components
        self.sio = socketio.Client()
        self.setup_websocket()
        
        # GUI state
        self.robots = {}
        self.selected_robot_id = None
        self.connection_status = "Disconnected"
        self.update_queue = queue.Queue()
        self.movement_enabled = True  # Movement toggle state
        self.title_reset_timer = None  # Timer for title reset
        
        # Create GUI
        self.create_gui()
        self.setup_periodic_updates()
        
        # Connect to server on startup
        self.root.after(1000, self.connect_to_server)
        
        # Start periodic robot status refresh for live battery updates
        self.root.after(2000, self.start_robot_status_refresh)
        
        # Initialize activity displays
        self.root.after(500, self.initialize_activity_displays)
        
        # Start periodic VILA status checking
        self.root.after(1500, self.start_vila_status_refresh)
        
        # Start robot list refresh for GUI visibility
        self.root.after(2500, self.start_robot_list_refresh)

    def setup_websocket(self):
        """Setup WebSocket event handlers"""
        @self.sio.event
        def connect():
            logger.info("Connected to robot server via WebSocket")
            self.update_connection_status("Connected (WebSocket)")
            self.sio.emit('join_monitors')

        @self.sio.event
        def disconnect():
            logger.info("Disconnected from robot server")
            self.update_connection_status("Disconnected")

        @self.sio.event
        def robot_registered(data):
            logger.info(f"Robot registered: {data}")
            self.update_queue.put(('robot_registered', data))

        @self.sio.event
        def robot_analysis(data):
            logger.info(f"Robot analysis received: {data.get('robot_id')}")
            self.update_queue.put(('robot_analysis', data))

        @self.sio.event
        def robot_activity(data):
            logger.info(f"Robot activity received: {data.get('robot_id')} - {data.get('activity_type')}")
            self.update_queue.put(('robot_activity', data))

        @self.sio.event
        def robot_sensors(data):
            logger.info(f"Robot sensors received: {data.get('robot_id')}")
            self.update_queue.put(('robot_sensors', data))

    def create_gui(self):
        """Create the main GUI interface"""
        # Top frame to hold notebook and controls
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control frame for system controls (right side)
        control_frame = ttk.LabelFrame(top_frame, text="System Controls", padding=5)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # VILA Model toggle button
        self.vila_enabled_var = tk.BooleanVar(value=False)
        self.vila_toggle_btn = ttk.Checkbutton(
            control_frame,
            text="VILA Model\nDisabled",
            variable=self.vila_enabled_var,
            command=self.toggle_vila_model
        )
        self.vila_toggle_btn.pack(pady=10, padx=5)
        
        # VILA Status display
        self.vila_model_status_var = tk.StringVar(value="Not Loaded")
        self.vila_status_display = ttk.Label(
            control_frame, 
            textvariable=self.vila_model_status_var,
            foreground="red",
            font=("Arial", 8)
        )
        self.vila_status_display.pack(pady=(0, 10), padx=5)
        
        # Movement toggle button
        self.movement_toggle_var = tk.BooleanVar(value=True)
        self.movement_toggle_btn = ttk.Checkbutton(
            control_frame,
            text="Movement\nEnabled",
            variable=self.movement_toggle_var,
            command=self.toggle_movement
        )
        self.movement_toggle_btn.pack(pady=10, padx=5)
        
        # VILA Image Thumbnail Display
        self.vila_thumbnail_frame = ttk.LabelFrame(control_frame, text="VILA Image", padding=5)
        self.vila_thumbnail_frame.pack(pady=(10, 0), padx=5, fill=tk.X)
        
        # Thumbnail label
        self.vila_thumbnail_label = ttk.Label(
            self.vila_thumbnail_frame,
            text="No image sent\nto VILA yet",
            background="lightgray",
            anchor="center",
            font=("Arial", 8),
            cursor="hand2"
        )
        self.vila_thumbnail_label.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
        # Bind click event to show full-size image
        self.vila_thumbnail_label.bind("<Button-1>", self.show_vila_image_fullsize)
        
        # Initialize thumbnail attributes
        self.vila_thumbnail_image = None
        self.last_vila_image = None
        
        # Ensure initial state is synchronized and call update immediately
        self.movement_enabled = self.movement_toggle_var.get()
        self.vila_enabled = self.vila_enabled_var.get()
        self.vila_autonomous = False  # Initialize autonomous mode
        print(f"🔧 INIT: Movement toggle initialized - enabled: {self.movement_enabled}")
        print(f"🤖 INIT: VILA toggle initialized - enabled: {self.vila_enabled}")
        print(f"🤖 INIT: VILA auto navigation mode initialized - enabled: {self.vila_autonomous}")
        
        # Force initial button state update
        self.root.after(100, self.update_movement_buttons_state)
        self.root.after(200, self.check_vila_status)
        
        # Main notebook for tabs
        self.notebook = ttk.Notebook(top_frame)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_status_tab()
        self.create_robots_tab()
        self.create_control_tab()
        self.create_vila_activity_tab()  # New VILA activity tab
        self.create_monitoring_tab()
        self.create_settings_tab()

    def update_vila_thumbnail(self, image):
        """Update the VILA thumbnail with the image being sent for analysis"""
        try:
            if image is None:
                return
            
            # Store the original image
            self.last_vila_image = image.copy()
            
            # Scale image to fill frame width (approximately 240px) while maintaining aspect ratio
            target_width = 240
            max_height = 180
            
            # Calculate scaling to fill width
            original_width, original_height = image.size
            scale_factor = target_width / original_width
            new_height = int(original_height * scale_factor)
            
            # If scaled height exceeds max, scale by height instead and crop width
            if new_height > max_height:
                scale_factor = max_height / original_height
                new_width = int(original_width * scale_factor)
                new_height = max_height
                
                # Scale and crop to center
                thumbnail = image.copy()
                try:
                    thumbnail = thumbnail.resize((new_width, new_height), Image.Resampling.LANCZOS)
                except AttributeError:
                    thumbnail = thumbnail.resize((new_width, new_height), Image.ANTIALIAS)
                
                # Crop to target width if needed (center crop)
                if new_width > target_width:
                    left = (new_width - target_width) // 2
                    right = left + target_width
                    thumbnail = thumbnail.crop((left, 0, right, new_height))
            else:
                # Scale to fill width exactly
                try:
                    thumbnail = image.resize((target_width, new_height), Image.Resampling.LANCZOS)
                except AttributeError:
                    thumbnail = image.resize((target_width, new_height), Image.ANTIALIAS)
            
            # Convert to PhotoImage for tkinter
            self.vila_thumbnail_image = ImageTk.PhotoImage(thumbnail)
            
            # Update the label
            self.vila_thumbnail_label.configure(
                image=self.vila_thumbnail_image,
                text="",  # Clear text when showing image
                background="white"
            )
            
            # Reset frame title for manual analysis
            self.vila_thumbnail_frame.configure(text="VILA Image")
            
            logger.info(f"Updated VILA thumbnail: {thumbnail.size} (scaled to fill frame width)")
            
        except Exception as e:
            logger.error(f"Error updating VILA thumbnail: {e}")
            self.vila_thumbnail_label.configure(
                text=f"Error loading\nthumbnail",
                background="lightgray"
            )

    def show_vila_image_fullsize(self, event):
        """Show the full-size image that was sent to VILA"""
        if self.last_vila_image is None:
            messagebox.showinfo("No Image", "No image has been sent to VILA yet.")
            return
        
        try:
            # Create a new window to display the full-size image
            image_window = tk.Toplevel(self.root)
            image_window.title("VILA Image - Full Size")
            image_window.geometry("800x600")
            
            # Create a frame with scrollbars for large images
            canvas = tk.Canvas(image_window)
            scrollbar_v = ttk.Scrollbar(image_window, orient="vertical", command=canvas.yview)
            scrollbar_h = ttk.Scrollbar(image_window, orient="horizontal", command=canvas.xview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
            
            # Display the full-size image
            display_image = ImageTk.PhotoImage(self.last_vila_image)
            image_label = ttk.Label(scrollable_frame, image=display_image)
            image_label.pack(padx=10, pady=10)
            
            # Keep a reference to prevent garbage collection
            image_label.image = display_image
            
            # Pack scrollbars and canvas
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar_v.pack(side="right", fill="y")
            scrollbar_h.pack(side="bottom", fill="x")
            
            # Center the window
            image_window.transient(self.root)
            image_window.grab_set()
            
        except Exception as e:
            logger.error(f"Error showing full-size VILA image: {e}")
            messagebox.showerror("Error", f"Could not display image: {e}")

    def update_vila_thumbnail_from_robot(self, thumbnail_image, robot_id):
        """Update VILA thumbnail from robot's automatic image analysis"""
        try:
            # Store the original thumbnail as the last VILA image
            self.last_vila_image = thumbnail_image.copy()
            
            # Scale image to fill frame width (same logic as manual analysis)
            target_width = 240
            max_height = 180
            
            # Calculate scaling to fill width
            original_width, original_height = thumbnail_image.size
            scale_factor = target_width / original_width
            new_height = int(original_height * scale_factor)
            
            # If scaled height exceeds max, scale by height instead and crop width
            if new_height > max_height:
                scale_factor = max_height / original_height
                new_width = int(original_width * scale_factor)
                new_height = max_height
                
                # Scale and crop to center
                scaled_thumbnail = thumbnail_image.copy()
                try:
                    scaled_thumbnail = scaled_thumbnail.resize((new_width, new_height), Image.Resampling.LANCZOS)
                except AttributeError:
                    scaled_thumbnail = scaled_thumbnail.resize((new_width, new_height), Image.ANTIALIAS)
                
                # Crop to target width if needed (center crop)
                if new_width > target_width:
                    left = (new_width - target_width) // 2
                    right = left + target_width
                    scaled_thumbnail = scaled_thumbnail.crop((left, 0, right, new_height))
            else:
                # Scale to fill width exactly
                try:
                    scaled_thumbnail = thumbnail_image.resize((target_width, new_height), Image.Resampling.LANCZOS)
                except AttributeError:
                    scaled_thumbnail = thumbnail_image.resize((target_width, new_height), Image.ANTIALIAS)
            
            # Convert to PhotoImage for tkinter
            self.vila_thumbnail_image = ImageTk.PhotoImage(scaled_thumbnail)
            
            # Update the label with robot info
            self.vila_thumbnail_label.configure(
                image=self.vila_thumbnail_image,
                text="",  # Clear text when showing image
                background="lightblue"  # Different color to indicate auto-analysis
            )
            
            # Update the frame title to show it's from a robot
            self.vila_thumbnail_frame.configure(text=f"VILA Image (from {robot_id})")
            
            logger.info(f"Updated VILA thumbnail from robot {robot_id}: {scaled_thumbnail.size} (scaled to fill frame width)")
            
        except Exception as e:
            logger.error(f"Error updating VILA thumbnail from robot: {e}")

    def flash_title_with_activity(self, message):
        """Flash the window title to show activity"""
        try:
            # Update title with activity message
            self.root.title(f"{self.base_title} - {message}")
            
            # Reset title after 3 seconds
            if self.title_reset_timer:
                self.root.after_cancel(self.title_reset_timer)
            
            self.title_reset_timer = self.root.after(3000, lambda: self.root.title(self.base_title))
            
        except Exception as e:
            logger.error(f"Error flashing title: {e}")

    def toggle_movement(self):
        """Toggle robot movement enable/disable"""
        try:
            # Update both variables to ensure sync
            toggle_value = self.movement_toggle_var.get()
            self.movement_enabled = toggle_value
            
            status = "Enabled" if self.movement_enabled else "DISABLED"
            
            # Debug logging
            print(f"🔄 TOGGLE: Movement {status} - toggle_var: {toggle_value}, movement_enabled: {self.movement_enabled}")
            
            # Log to movement activity display
            safety_msg = f"Movement safety toggle changed: {status} (enabled:{self.movement_enabled}, toggle:{toggle_value})"
            activity_type = "SAFETY" if not self.movement_enabled else "INFO"
            self.log_movement_activity(safety_msg, activity_type)
            
            # Update movement status in activity tab
            self.movement_status_var.set(f"Movement {status}")
            if self.movement_enabled:
                self.movement_status_label.configure(foreground="green")
            else:
                self.movement_status_label.configure(foreground="red")
            
            # Update button appearance with clearer text
            self.movement_toggle_btn.configure(
                text=f"Movement\n{status}"
            )
            
            # Enable/disable movement buttons based on toggle state
            self.update_movement_buttons_state()
            
            # Log the change
            self.log_message(f"🔄 Robot movement {status.lower()}")
            
            # Cancel any pending title reset
            if self.title_reset_timer:
                self.root.after_cancel(self.title_reset_timer)
            
            # Update window title with clean base title
            status_title = f"{self.base_title} - Movement {status}"
            self.root.title(status_title)
            
            # Schedule reset to base title after 3 seconds
            self.title_reset_timer = self.root.after(3000, self.reset_window_title)
            
        except Exception as e:
            self.log_message(f"❌ Toggle error: {str(e)}")
            self.log_movement_activity(f"Toggle error: {str(e)}", "ERROR")
            print(f"Toggle error: {e}")  # Debug print

    def update_movement_buttons_state(self):
        """Enable or disable movement buttons based on movement_enabled state"""
        try:
            # Double-check toggle state for safety
            toggle_state = self.movement_toggle_var.get()
            final_enabled = self.movement_enabled and toggle_state
            
            state = 'normal' if final_enabled else 'disabled'
            
            # Update all movement buttons except STOP
            movement_only_buttons = [self.btn_forward, self.btn_left, self.btn_right, self.btn_backward]
            for button in movement_only_buttons:
                button.configure(state=state)
            
            # Special handling for STOP button - always enabled for safety
            self.btn_stop.configure(state='normal')
            
            status = "enabled" if final_enabled else "disabled"
            print(f"🎛️ Movement buttons {status} (movement_enabled: {self.movement_enabled}, toggle: {toggle_state})")
            
        except Exception as e:
            print(f"Button state update error: {e}")

    def reset_window_title(self):
        """Reset window title to base title"""
        try:
            self.root.title(self.base_title)
            self.title_reset_timer = None
        except Exception as e:
            print(f"Title reset error: {e}")

    def create_status_tab(self):
        """Create the server status and communication tab"""
        status_frame = ttk.Frame(self.notebook)
        self.notebook.add(status_frame, text="Server Status")
        
        # Connection status section
        conn_frame = ttk.LabelFrame(status_frame, text="Server Connection", padding=10)
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Connection status display
        self.status_var = tk.StringVar(value="Disconnected")
        status_label = ttk.Label(conn_frame, text="Status:")
        status_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        
        self.status_display = ttk.Label(conn_frame, textvariable=self.status_var, 
                                       foreground="red", font=("Arial", 10, "bold"))
        self.status_display.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Server info
        server_info_frame = ttk.Frame(conn_frame)
        server_info_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        ttk.Label(server_info_frame, text="Server:").pack(side=tk.LEFT)
        self.server_label = ttk.Label(server_info_frame, text=self.config.http_url)
        self.server_label.pack(side=tk.LEFT, padx=5)
        
        # Connection buttons
        button_frame = ttk.Frame(conn_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        ttk.Button(button_frame, text="Connect", command=self.connect_to_server).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Disconnect", command=self.disconnect_from_server).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Test Connection", command=self.test_connection).pack(side=tk.LEFT, padx=2)
        
        # Server health section
        health_frame = ttk.LabelFrame(status_frame, text="Server Health", padding=10)
        health_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Health display
        self.health_text = scrolledtext.ScrolledText(health_frame, height=15, wrap=tk.WORD)
        self.health_text.pack(fill=tk.BOTH, expand=True)
        
        # Update health info button
        ttk.Button(health_frame, text="Refresh Health Info", 
                  command=self.update_health_info).pack(pady=5)

    def create_robots_tab(self):
        """Create the robots management tab"""
        robots_frame = ttk.Frame(self.notebook)
        self.notebook.add(robots_frame, text="Robots")
        
        # Robot list section  
        list_frame = ttk.LabelFrame(robots_frame, text="Robot Status", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Robot treeview
        columns = ('ID', 'Name', 'Status', 'Battery', 'Last Seen', 'Position')
        self.robot_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.robot_tree.heading(col, text=col)
            self.robot_tree.column(col, width=120)
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.robot_tree.yview)
        self.robot_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.robot_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Robot selection handling
        self.robot_tree.bind('<<TreeviewSelect>>', self.on_robot_select)
        
        # Robot details section
        details_frame = ttk.LabelFrame(robots_frame, text="Robot Information", padding=10)
        details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.robot_details_text = scrolledtext.ScrolledText(details_frame, height=8, wrap=tk.WORD)
        self.robot_details_text.pack(fill=tk.BOTH, expand=True)
        
        # Robot management
        mgmt_frame = ttk.Frame(robots_frame)
        mgmt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(mgmt_frame, text="Refresh Status", 
                  command=self.refresh_robot_list).pack(side=tk.LEFT, padx=2)
        
        # Status indicator
        self.robot_count_var = tk.StringVar(value="No robots connected")
        status_label = ttk.Label(mgmt_frame, textvariable=self.robot_count_var, 
                                font=("Arial", 9), foreground="gray")
        status_label.pack(side=tk.RIGHT, padx=10)

    def create_control_tab(self):
        """Create the robot control tab"""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="Robot Control")
        
        # Robot status for control
        select_frame = ttk.LabelFrame(control_frame, text="Robot Control", padding=10)
        select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(select_frame, text="Active Robot:").pack(side=tk.LEFT)
        self.selected_robot_var = tk.StringVar(value="No robot connected")
        self.selected_robot_label = ttk.Label(select_frame, textvariable=self.selected_robot_var,
                                             font=("Arial", 10, "bold"))
        self.selected_robot_label.pack(side=tk.LEFT, padx=10)
        
        # Movement controls
        movement_frame = ttk.LabelFrame(control_frame, text="Movement Control", padding=10)
        movement_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create movement button grid
        move_grid = ttk.Frame(movement_frame)
        move_grid.pack(pady=10)
        
        # Movement buttons - store as instance variables for enable/disable control
        self.btn_forward = ttk.Button(move_grid, text="↑ Forward", width=12,
                                     command=lambda: self.send_movement_command('forward'))
        self.btn_forward.grid(row=0, column=1, padx=2, pady=2)
        
        self.btn_left = ttk.Button(move_grid, text="← Left", width=12,
                                  command=lambda: self.send_movement_command('left'))
        self.btn_left.grid(row=1, column=0, padx=2, pady=2)
        
        self.btn_stop = ttk.Button(move_grid, text="STOP", width=12,
                                  command=lambda: self.send_movement_command('stop'))
        self.btn_stop.grid(row=1, column=1, padx=2, pady=2)
        
        self.btn_right = ttk.Button(move_grid, text="Right →", width=12,
                                   command=lambda: self.send_movement_command('right'))
        self.btn_right.grid(row=1, column=2, padx=2, pady=2)
        
        self.btn_backward = ttk.Button(move_grid, text="↓ Backward", width=12,
                                      command=lambda: self.send_movement_command('backward'))
        self.btn_backward.grid(row=2, column=1, padx=2, pady=2)
        
        # Store movement buttons for easy enable/disable
        self.movement_buttons = [self.btn_forward, self.btn_left, self.btn_stop, 
                               self.btn_right, self.btn_backward]
        
        # Initialize button states based on current movement setting
        self.update_movement_buttons_state()
        
        # Image analysis section
        analysis_frame = ttk.LabelFrame(control_frame, text="Vision Analysis", padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image upload
        img_upload_frame = ttk.Frame(analysis_frame)
        img_upload_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(img_upload_frame, text="Load Image", 
                  command=self.load_image_for_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(img_upload_frame, text="Analyze Image", 
                  command=self.analyze_image).pack(side=tk.LEFT, padx=2)
        
        # Image display placeholder
        self.image_label = ttk.Label(analysis_frame, text="No image loaded", 
                                   background="lightgray", width=40)
        self.image_label.pack(pady=5)
        
        # Analysis prompt
        prompt_frame = ttk.Frame(analysis_frame)
        prompt_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(prompt_frame, text="Prompt:").pack(side=tk.LEFT)
        self.analysis_prompt = tk.Entry(prompt_frame, width=50)
        self.analysis_prompt.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.analysis_prompt.insert(0, "Analyze this environment for robot navigation")
        
        # Analysis results
        self.analysis_results = scrolledtext.ScrolledText(analysis_frame, height=8, wrap=tk.WORD)
        self.analysis_results.pack(fill=tk.BOTH, expand=True, pady=5)

    def create_monitoring_tab(self):
        """Create the monitoring tab with placeholders for future features"""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="Monitoring")
        
        # SLAM Map placeholder
        slam_frame = ttk.LabelFrame(monitor_frame, text="SLAM Map", padding=10)
        slam_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.slam_placeholder = ttk.Label(slam_frame, text="SLAM Map Display\n(To be implemented)", 
                                         background="lightblue", font=("Arial", 14))
        self.slam_placeholder.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Sensor values section
        sensor_frame = ttk.LabelFrame(monitor_frame, text="Sensor Readings", padding=10)
        sensor_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create sensor value displays
        sensor_grid = ttk.Frame(sensor_frame)
        sensor_grid.pack(fill=tk.X)
        
        # Real sensor displays (no placeholders)
        sensors = [
            ("Lidar Distance", "meters"), ("IMU Heading", "degrees"), 
            ("GPS Position", "lat/lon"), ("Camera Status", "status"),
            ("Battery Voltage", "volts"), ("Motor Temperature", "°C")
        ]
        
        self.sensor_vars = {}
        self.sensor_labels = {}
        for i, (sensor, unit) in enumerate(sensors):
            row, col = i // 2, (i % 2) * 2
            
            # Label for sensor name
            name_label = ttk.Label(sensor_grid, text=f"{sensor}:")
            name_label.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            
            # Variable and label for sensor value
            var = tk.StringVar(value="----")
            self.sensor_vars[sensor] = var
            value_label = ttk.Label(sensor_grid, textvariable=var, foreground="red")
            value_label.grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=2)
            self.sensor_labels[sensor] = value_label
        
        # Add connection status indicator
        status_frame = ttk.Frame(sensor_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(status_frame, text="Sensor Status:").pack(side=tk.LEFT, padx=5)
        self.sensor_status_var = tk.StringVar(value="No robot selected")
        self.sensor_status_label = ttk.Label(status_frame, textvariable=self.sensor_status_var, foreground="gray")
        self.sensor_status_label.pack(side=tk.LEFT, padx=5)

    def create_settings_tab(self):
        """Create the settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Server settings
        server_frame = ttk.LabelFrame(settings_frame, text="Server Configuration", padding=10)
        server_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Host setting
        host_frame = ttk.Frame(server_frame)
        host_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(host_frame, text="Server Host:", width=15).pack(side=tk.LEFT)
        self.host_entry = tk.Entry(host_frame, width=20)
        self.host_entry.pack(side=tk.LEFT, padx=5)
        self.host_entry.insert(0, self.config.host)
        
        # HTTP Port setting
        http_port_frame = ttk.Frame(server_frame)
        http_port_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(http_port_frame, text="HTTP Port:", width=15).pack(side=tk.LEFT)
        self.http_port_entry = tk.Entry(http_port_frame, width=20)
        self.http_port_entry.pack(side=tk.LEFT, padx=5)
        self.http_port_entry.insert(0, str(self.config.http_port))
        
        # TCP Port setting
        tcp_port_frame = ttk.Frame(server_frame)
        tcp_port_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(tcp_port_frame, text="TCP Port:", width=15).pack(side=tk.LEFT)
        self.tcp_port_entry = tk.Entry(tcp_port_frame, width=20)
        self.tcp_port_entry.pack(side=tk.LEFT, padx=5)
        self.tcp_port_entry.insert(0, str(self.config.tcp_port))
        
        # Apply settings button
        ttk.Button(server_frame, text="Apply Settings", 
                  command=self.apply_settings).pack(pady=10)
        
        # Log display
        log_frame = ttk.LabelFrame(settings_frame, text="Application Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).pack(pady=5)
    
    def create_vila_activity_tab(self):
        """Create the VILA activity monitoring tab"""
        vila_frame = ttk.Frame(self.notebook)
        self.notebook.add(vila_frame, text="VILA Activity")
        
        # Create main container with vertical paned window for better layout
        main_paned = ttk.PanedWindow(vila_frame, orient=tk.VERTICAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # VILA Analysis Pipeline Section (top)
        pipeline_frame = ttk.Frame(main_paned)
        main_paned.add(pipeline_frame, weight=2)
        
        pipeline_section = ttk.LabelFrame(pipeline_frame, text="VILA → Robot Command Pipeline", padding=10)
        pipeline_section.pack(fill=tk.BOTH, expand=True)
        
        # Pipeline controls
        pipeline_controls = ttk.Frame(pipeline_section)
        pipeline_controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(pipeline_controls, text="Command Pipeline:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        # Add VILA Auto Nav Mode toggle
        self.vila_autonomous_var = tk.BooleanVar(value=False)
        self.vila_autonomous_btn = ttk.Checkbutton(
            pipeline_controls,
            text="VILA Auto Nav Mode",
            variable=self.vila_autonomous_var,
            command=self.toggle_vila_autonomous
        )
        self.vila_autonomous_btn.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(pipeline_controls, text="Clear Pipeline", 
                  command=self.clear_pipeline_display).pack(side=tk.RIGHT)
        
        # Pipeline display
        self.pipeline_text = scrolledtext.ScrolledText(
            pipeline_section, height=15, wrap=tk.WORD, font=("Consolas", 9)
        )
        self.pipeline_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Bottom container for side-by-side displays
        bottom_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
        main_paned.add(bottom_paned, weight=1)
        
        # VILA Model Activity Section (bottom left)
        vila_activity_frame = ttk.Frame(bottom_paned)
        bottom_paned.add(vila_activity_frame, weight=1)
        
        vila_section = ttk.LabelFrame(vila_activity_frame, text="VILA Model Activity", padding=10)
        vila_section.pack(fill=tk.BOTH, expand=True)
        
        # VILA status and controls
        vila_status_frame = ttk.Frame(vila_section)
        vila_status_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(vila_status_frame, text="Status:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.vila_status_var = tk.StringVar(value="Ready")
        self.vila_status_label = ttk.Label(vila_status_frame, textvariable=self.vila_status_var, 
                                          foreground="green", font=("Arial", 9))
        self.vila_status_label.pack(side=tk.LEFT, padx=5)
        
        # Verbose logging checkbox for VILA
        self.verbose_vila_var = tk.BooleanVar(value=True)  # Default to verbose mode
        verbose_vila_checkbox = ttk.Checkbutton(vila_status_frame, text="Verbose", 
                                               variable=self.verbose_vila_var)
        verbose_vila_checkbox.pack(side=tk.RIGHT, padx=(0, 10))
        
        ttk.Button(vila_status_frame, text="Clear", 
                  command=self.clear_vila_activity).pack(side=tk.RIGHT)
        
        # VILA activity display
        self.vila_activity_text = scrolledtext.ScrolledText(
            vila_section, height=8, wrap=tk.WORD, font=("Consolas", 8)
        )
        self.vila_activity_text.pack(fill=tk.BOTH, expand=True, pady=2)
        
        # Track autoscroll state for VILA activity
        self.vila_autoscroll_enabled = True
        
        # Bind scroll events to detect manual scrolling for VILA
        def on_vila_scroll(*args):
            """Detect manual scrolling and disable autoscroll"""
            try:
                # Get scrollbar position
                top, bottom = self.vila_activity_text.yview()
                # If not at bottom (with small tolerance), disable autoscroll
                if bottom < 0.98:  # 98% to allow for minor rounding
                    self.vila_autoscroll_enabled = False
                else:
                    self.vila_autoscroll_enabled = True
            except:
                pass
        
        # Bind to both scrollbar and mousewheel events for VILA
        self.vila_activity_text.bind('<MouseWheel>', lambda e: self.root.after(10, on_vila_scroll))
        self.vila_activity_text.bind('<Button-4>', lambda e: self.root.after(10, on_vila_scroll))
        self.vila_activity_text.bind('<Button-5>', lambda e: self.root.after(10, on_vila_scroll))
        
        # Also bind to scrollbar interactions for VILA
        vila_scrollbar = self.vila_activity_text.vbar
        if vila_scrollbar:
            vila_scrollbar.bind('<ButtonRelease-1>', lambda e: self.root.after(10, on_vila_scroll))
            vila_scrollbar.bind('<B1-Motion>', lambda e: self.root.after(10, on_vila_scroll))
        
        # Movement Commands Section (bottom right)
        movement_activity_frame = ttk.Frame(bottom_paned)
        bottom_paned.add(movement_activity_frame, weight=1)
        
        movement_section = ttk.LabelFrame(movement_activity_frame, text="Robot Movement Commands", padding=10)
        movement_section.pack(fill=tk.BOTH, expand=True)
        
        # Movement status and controls
        movement_status_frame = ttk.Frame(movement_section)
        movement_status_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(movement_status_frame, text="Status:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.movement_status_var = tk.StringVar(value="Ready")
        self.movement_status_label = ttk.Label(movement_status_frame, textvariable=self.movement_status_var,
                                             foreground="blue", font=("Arial", 9))
        self.movement_status_label.pack(side=tk.LEFT, padx=5)
        
        # Verbose logging checkbox
        self.verbose_movement_var = tk.BooleanVar(value=True)  # Default to verbose mode
        verbose_checkbox = ttk.Checkbutton(movement_status_frame, text="Verbose", 
                                         variable=self.verbose_movement_var)
        verbose_checkbox.pack(side=tk.RIGHT, padx=(0, 10))
        
        ttk.Button(movement_status_frame, text="Clear", 
                  command=self.clear_movement_activity).pack(side=tk.RIGHT)
        
        # Movement activity display
        self.movement_activity_text = scrolledtext.ScrolledText(
            movement_section, height=8, wrap=tk.WORD, font=("Consolas", 8)
        )
        self.movement_activity_text.pack(fill=tk.BOTH, expand=True, pady=2)
        
        # Track autoscroll state for movement activity
        self.movement_autoscroll_enabled = True
        
        # Bind scroll events to detect manual scrolling
        def on_movement_scroll(*args):
            """Detect manual scrolling and disable autoscroll"""
            try:
                # Get scrollbar position
                top, bottom = self.movement_activity_text.yview()
                # If not at bottom (with small tolerance), disable autoscroll
                if bottom < 0.98:  # 98% to allow for minor rounding
                    self.movement_autoscroll_enabled = False
                else:
                    self.movement_autoscroll_enabled = True
            except:
                pass
        
        # Bind to both scrollbar and mousewheel events
        self.movement_activity_text.bind('<MouseWheel>', lambda e: self.root.after(10, on_movement_scroll))
        self.movement_activity_text.bind('<Button-4>', lambda e: self.root.after(10, on_movement_scroll))
        self.movement_activity_text.bind('<Button-5>', lambda e: self.root.after(10, on_movement_scroll))
        
        # Also bind to scrollbar interactions
        scrollbar = self.movement_activity_text.vbar
        if scrollbar:
            scrollbar.bind('<ButtonRelease-1>', lambda e: self.root.after(10, on_movement_scroll))
            scrollbar.bind('<B1-Motion>', lambda e: self.root.after(10, on_movement_scroll))
        
        # Initialize with welcome messages
        self.pipeline_text.insert(tk.END,
            "🚀 VILA → Robot Command Pipeline\n"
            "=" * 70 + "\n"
            "This shows exactly how VILA moves the robot:\n\n"
            "1. 📷 Image Analysis    → VILA analyzes camera image\n"
            "2. 📝 Text Response     → VILA generates natural language response\n"
            "3. 🔍 Command Parsing   → Extract movement keywords from text\n"
            "4. ⚙️  Command Generation → Convert to robot movement commands\n"
            "5. 🤖 Robot Execution   → Send commands to robot (if autonomous mode enabled)\n\n"
            "💡 Enable 'VILA Auto Nav Mode' to see full pipeline in action!\n"
            "⚠️  Safety: Movement toggle must also be enabled for actual robot movement.\n\n"
            "Start an image analysis to see the pipeline...\n\n"
        )
        
        self.vila_activity_text.insert(tk.END, 
            "🤖 VILA Activity Monitor\n"
            "=" * 30 + "\n"
            "VILA model operations\n\n"
        )
        
        self.movement_activity_text.insert(tk.END,
            "🎮 Movement Commands\n"
            "=" * 30 + "\n"
            "Robot command execution\n\n"
        )

    def setup_periodic_updates(self):
        """Setup periodic GUI updates"""
        self.process_update_queue()
        self.root.after(100, self.setup_periodic_updates)

    def process_update_queue(self):
        """Process updates from background threads"""
        try:
            while True:
                update_type, data = self.update_queue.get_nowait()
                
                if update_type == 'robot_registered':
                    self.log_message(f"Robot registered: {data.get('robot_id', 'Unknown')}")
                    self.refresh_robot_list()
                    
                elif update_type == 'robot_analysis':
                    robot_id = data.get('robot_id')
                    analysis = data.get('analysis', 'No analysis')
                    self.log_message(f"Analysis for {robot_id}: {analysis[:100]}...")
                    
                elif update_type == 'robot_activity':
                    self.handle_robot_activity(data)
                    
                elif update_type == 'robot_sensors':
                    self.handle_robot_sensors(data)
                    
                elif update_type == 'connection_status':
                    self.update_connection_status(data)
                    
        except queue.Empty:
            pass

    def handle_robot_activity(self, data):
        """Handle robot activity events (including automatic image analysis)"""
        try:
            robot_id = data.get('robot_id')
            activity_type = data.get('activity_type')
            
            if activity_type == 'analysis_request':
                # Robot sent an image for analysis
                analysis = data.get('vila_response', 'No analysis')
                thumbnail_b64 = data.get('thumbnail')
                
                # Log the activity
                self.log_message(f"🤖 Auto-analysis from {robot_id}: {analysis[:100]}...")
                
                # Update thumbnail if we have image data
                if thumbnail_b64:
                    try:
                        # Decode thumbnail
                        thumbnail_data = base64.b64decode(thumbnail_b64)
                        thumbnail_image = Image.open(io.BytesIO(thumbnail_data))
                        
                        # Update VILA thumbnail display
                        self.update_vila_thumbnail_from_robot(thumbnail_image, robot_id)
                        
                    except Exception as img_error:
                        logger.error(f"Error processing robot thumbnail: {img_error}")
                
                # Update title to show activity
                self.flash_title_with_activity(f"VILA Analysis from {robot_id}")
                
        except Exception as e:
            logger.error(f"Error handling robot activity: {e}")

    def handle_robot_sensors(self, data):
        """Handle robot sensor data updates"""
        try:
            robot_id = data.get('robot_id')
            sensor_data = data.get('sensor_data', {})
            
            # Update sensor display if this is the selected robot
            if robot_id == self.selected_robot_id:
                if sensor_data:
                    self.update_sensor_display(sensor_data)
                else:
                    self.clear_sensor_display(f"Robot {robot_id} sent empty sensor data")
            
            # Log sensor updates
            sensor_summary = []
            if 'battery_voltage' in sensor_data:
                sensor_summary.append(f"Battery: {sensor_data['battery_voltage']:.2f}V")
            if 'temperature' in sensor_data:
                sensor_summary.append(f"Temp: {sensor_data['temperature']:.1f}°C")
            if 'lidar_distance' in sensor_data:
                sensor_summary.append(f"Lidar: {sensor_data['lidar_distance']:.2f}m")
            
            if sensor_summary:
                self.log_message(f"📊 Sensors from {robot_id}: {', '.join(sensor_summary)}")
            else:
                self.log_message(f"⚠️ Robot {robot_id} sent sensor update with no recognizable data")
                
        except Exception as e:
            logger.error(f"Error handling robot sensors: {e}")
            if self.selected_robot_id:
                self.clear_sensor_display("Error processing sensor data")

    def update_sensor_display(self, sensor_data):
        """Update the sensor display with real data only"""
        try:
            if not sensor_data:
                self.clear_sensor_display("No sensor data received")
                return
            
            # Mapping of sensor data keys to display names
            sensor_mapping = {
                'lidar_distance': ('Lidar Distance', 'meters'),
                'imu_heading': ('IMU Heading', 'degrees'),
                'gps_lat': ('GPS Position', 'lat/lon'),
                'camera_status': ('Camera Status', 'status'),
                'battery_voltage': ('Battery Voltage', 'volts'),
                'temperature': ('Motor Temperature', '°C')
            }
            
            # Track which sensors have data
            sensors_updated = 0
            
            # First, clear all displays to unavailable state
            for display_name in self.sensor_vars.keys():
                self.sensor_vars[display_name].set("----")
                self.sensor_labels[display_name].configure(foreground="red")
            
            # Update only sensors that have real data
            for key, value in sensor_data.items():
                if key in sensor_mapping:
                    display_name, unit = sensor_mapping[key]
                    if display_name in self.sensor_vars:
                        try:
                            if key == 'gps_lat' and 'gps_lon' in sensor_data:
                                # Special handling for GPS coordinates
                                formatted_value = f"{value:.6f}, {sensor_data['gps_lon']:.6f}"
                            elif isinstance(value, (int, float)):
                                formatted_value = f"{value:.2f} {unit}"
                            else:
                                formatted_value = f"{value}"
                            
                            self.sensor_vars[display_name].set(formatted_value)
                            # Change color to indicate live data
                            self.sensor_labels[display_name].configure(foreground="black")
                            sensors_updated += 1
                            
                        except (ValueError, TypeError) as e:
                            # Handle invalid sensor values
                            self.sensor_vars[display_name].set("Invalid data")
                            self.sensor_labels[display_name].configure(foreground="red")
                            logger.warning(f"Invalid sensor value for {key}: {value} ({e})")
            
            # Update status
            if sensors_updated > 0:
                self.sensor_status_var.set(f"Live data ({sensors_updated} sensors)")
                self.sensor_status_label.configure(foreground="green")
                logger.info(f"Updated {sensors_updated} sensors with live data")
            else:
                self.sensor_status_var.set("No valid sensor data")
                self.sensor_status_label.configure(foreground="orange")
            
        except Exception as e:
            logger.error(f"Error updating sensor display: {e}")
            self.sensor_status_var.set("Sensor update error")
            self.sensor_status_label.configure(foreground="red")

    # Communication methods
    def connect_to_server(self):
        """Connect to the robot server"""
        def connect_worker():
            try:
                # Test HTTP connection first
                response = requests.get(f"{self.config.http_url}/health", timeout=5)
                if response.status_code == 200:
                    # Connect WebSocket
                    self.sio.connect(self.config.websocket_url)
                    self.update_queue.put(('connection_status', 'Connected'))
                    self.log_message("Successfully connected to robot server")
                else:
                    self.update_queue.put(('connection_status', 'Connection Failed'))
                    self.log_message(f"HTTP connection failed: {response.status_code}")
                    
            except Exception as e:
                self.update_queue.put(('connection_status', 'Connection Failed'))
                self.log_message(f"Connection error: {str(e)}")
        
        threading.Thread(target=connect_worker, daemon=True).start()

    def disconnect_from_server(self):
        """Disconnect from the robot server"""
        try:
            if self.sio.connected:
                self.sio.disconnect()
            self.update_connection_status("Disconnected")
            self.log_message("Disconnected from robot server")
        except Exception as e:
            self.log_message(f"Disconnect error: {str(e)}")

    def test_connection(self):
        """Test connection to the server"""
        def test_worker():
            try:
                response = requests.get(f"{self.config.http_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    self.log_message(f"Connection test successful: {data}")
                else:
                    self.log_message(f"Connection test failed: HTTP {response.status_code}")
            except Exception as e:
                self.log_message(f"Connection test error: {str(e)}")
        
        threading.Thread(target=test_worker, daemon=True).start()

    def update_connection_status(self, status):
        """Update the connection status display"""
        self.connection_status = status
        
        # Update sensor display based on connection status
        if status == "Disconnected" and hasattr(self, 'sensor_status_var'):
            self.clear_sensor_display("Server disconnected")
        elif status.startswith("Connected") and hasattr(self, 'sensor_status_var'):
            # If we have a selected robot, try to refresh its sensor data
            if self.selected_robot_id and self.selected_robot_id in self.robots:
                robot = self.robots[self.selected_robot_id]
                if 'sensor_data' in robot and robot['sensor_data']:
                    self.update_sensor_display(robot['sensor_data'])
                else:
                    self.clear_sensor_display(f"Robot {self.selected_robot_id} - no sensor data available")
        self.status_var.set(status)
        
        if "Connected" in status:
            self.status_display.configure(foreground="green")
        else:
            self.status_display.configure(foreground="red")

    def update_health_info(self):
        """Update server health information"""
        def health_worker():
            try:
                response = requests.get(f"{self.config.http_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    health_info = json.dumps(data, indent=2)
                    
                    def update_health_display():
                        self.health_text.delete(1.0, tk.END)
                        self.health_text.insert(tk.END, f"Server Health (Updated: {datetime.now().strftime('%H:%M:%S')})\n")
                        self.health_text.insert(tk.END, "=" * 50 + "\n")
                        self.health_text.insert(tk.END, health_info)
                    
                    self.root.after(0, update_health_display)
                else:
                    self.log_message(f"Health check failed: HTTP {response.status_code}")
            except Exception as e:
                self.log_message(f"Health check error: {str(e)}")
        
        threading.Thread(target=health_worker, daemon=True).start()

    # Robot management methods
    def refresh_robot_list(self):
        """Refresh the robot list from server"""
        def refresh_worker():
            try:
                response = requests.get(f"{self.config.http_url}/robots", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    robots = data.get('robots', [])
                    
                    def update_robot_display():
                        # Clear existing items
                        for item in self.robot_tree.get_children():
                            self.robot_tree.delete(item)
                        
                        # Add robots to tree with better battery level formatting
                        for robot in robots:
                            battery_level = robot.get('battery_level', 0)
                            battery_color = ""
                            if battery_level < 20:
                                battery_color = " ⚠️"  # Warning for low battery
                            elif battery_level < 10:
                                battery_color = " 🔴"  # Critical battery
                            
                            self.robot_tree.insert('', 'end', values=(
                                robot.get('robot_id', 'Unknown'),
                                robot.get('name', 'Unknown'),
                                robot.get('status', 'Unknown'),
                                f"{battery_level:.1f}%{battery_color}",
                                robot.get('last_seen', 'Never'),
                                f"({robot.get('position', {}).get('x', 0):.1f}, {robot.get('position', {}).get('y', 0):.1f})"
                            ))
                        
                        self.robots = {r['robot_id']: r for r in robots}
                        
                        # Auto-select the single robot for control
                        if len(robots) == 1 and not self.selected_robot_id:
                            robot_id = robots[0]['robot_id']
                            self.selected_robot_id = robot_id
                            self.selected_robot_var.set(robot_id)
                            print(f"🤖 Auto-selected single robot: {robot_id}")
                    
                    self.root.after(0, update_robot_display)
                    
                    # Update status indicator
                    if len(robots) == 0:
                        self.robot_count_var.set("No robots connected")
                    elif len(robots) == 1:
                        self.robot_count_var.set("✅ 1 robot connected")
                    else:
                        self.robot_count_var.set(f"⚠️ {len(robots)} robots connected (expected 1)")
                    
                    self.log_message(f"Refreshed robot status: {len(robots)} robot{'s' if len(robots) != 1 else ''}")
                else:
                    self.log_message(f"Failed to refresh robot list: HTTP {response.status_code}")
            except Exception as e:
                self.log_message(f"Robot list refresh error: {str(e)}")
        
        threading.Thread(target=refresh_worker, daemon=True).start()

    def start_robot_status_refresh(self):
        """Start periodic robot status refresh for live battery updates"""
        def periodic_refresh():
            while True:
                try:
                    time.sleep(15)  # Refresh every 15 seconds
                    if hasattr(self, 'robots') and self.robots:
                        self.refresh_robot_list()
                except Exception as e:
                    print(f"Periodic refresh error: {e}")
                    time.sleep(30)  # Wait longer on error
        
        refresh_thread = threading.Thread(target=periodic_refresh, daemon=True)
        refresh_thread.start()
        print("🔄 Started periodic robot status refresh (every 15 seconds)")

    def on_robot_select(self, event):
        """Handle robot selection in the tree"""
        selection = self.robot_tree.selection()
        if selection:
            item = self.robot_tree.item(selection[0])
            robot_id = item['values'][0]
            self.selected_robot_id = robot_id
            self.selected_robot_var.set(robot_id)
            
            # Update robot details
            if robot_id in self.robots:
                robot = self.robots[robot_id]
                details = json.dumps(robot, indent=2, default=str)
                self.robot_details_text.delete(1.0, tk.END)
                self.robot_details_text.insert(tk.END, details)
                
                # Update sensor display with current robot's sensor data
                if 'sensor_data' in robot and robot['sensor_data']:
                    self.update_sensor_display(robot['sensor_data'])
                else:
                    # Clear sensor display with specific reason
                    robot_status = robot.get('status', 'unknown')
                    if robot_status == 'offline':
                        self.clear_sensor_display(f"Robot {robot_id} is offline")
                    elif robot_status == 'error':
                        self.clear_sensor_display(f"Robot {robot_id} has errors")
                    else:
                        self.clear_sensor_display(f"Robot {robot_id} - no sensor data available")

    def clear_sensor_display(self, reason="No robot selected"):
        """Clear all sensor displays with specific reason"""
        try:
            for sensor_name, var in self.sensor_vars.items():
                var.set("----")
                self.sensor_labels[sensor_name].configure(foreground="red")
            
            # Update status with reason
            self.sensor_status_var.set(reason)
            
            # Color code the status based on reason
            if "No robot selected" in reason:
                self.sensor_status_label.configure(foreground="gray")
            elif "error" in reason.lower() or "failed" in reason.lower():
                self.sensor_status_label.configure(foreground="red")
            elif "disconnected" in reason.lower() or "offline" in reason.lower():
                self.sensor_status_label.configure(foreground="orange")
            else:
                self.sensor_status_label.configure(foreground="gray")
                
        except Exception as e:
            logger.error(f"Error clearing sensor display: {e}")
            if hasattr(self, 'sensor_status_var'):
                self.sensor_status_var.set("Display error")
                self.sensor_status_label.configure(foreground="red")



    # Control methods
    def send_movement_command(self, direction):
        """Send movement command to selected robot"""
        # Debug logging
        toggle_state = self.movement_toggle_var.get()
        print(f"🔍 DEBUG: Movement command '{direction}' - Toggle state: {toggle_state}, movement_enabled: {self.movement_enabled}")
        
        # Log command attempt to activity display
        self.log_movement_activity(f"Attempting to send '{direction}' command to robot {self.selected_robot_id or 'None'}", "COMMAND")
        
        if not self.selected_robot_id:
            self.log_movement_activity("No robot connected - command cancelled", "ERROR")
            messagebox.showwarning("No Robot Connected", "No robot is currently connected. Please ensure your robot is registered with the server.")
            return
        
        # CRITICAL SAFETY CHECK: Block ALL movement commands when disabled (ONLY allow STOP for emergency)
        # Check BOTH variables to ensure safety
        toggle_state = self.movement_toggle_var.get()
        movement_allowed = self.movement_enabled and toggle_state
        
        # ONLY allow STOP commands when movement is disabled - block everything else
        if direction != 'stop' and not movement_allowed:
            safety_msg = f"Movement '{direction}' blocked by safety system (enabled:{self.movement_enabled}, toggle:{toggle_state})"
            print(f"🚫 CRITICAL SAFETY BLOCK: Movement command '{direction}' blocked")
            print(f"   └── movement_enabled: {self.movement_enabled}, toggle_state: {toggle_state}")
            print(f"   └── ONLY 'stop' commands allowed when movement disabled")
            self.log_message(f"🚫 BLOCKED: Movement '{direction}' - Safety disabled (enabled:{self.movement_enabled}, toggle:{toggle_state})")
            self.log_movement_activity(safety_msg, "BLOCKED")
            
            # Also log to server logs for debugging
            print(f"🔴 SERVER LOG: Movement command '{direction}' BLOCKED by safety system")
            return
        
        def send_command_worker():
            try:
                # DOUBLE SAFETY CHECK: Final verification before sending HTTP request
                # ABSOLUTE SAFETY: Block ANY command that is not 'stop' when movement disabled
                if direction != 'stop' and (not self.movement_enabled or not self.movement_toggle_var.get()):
                    error_msg = f"Command '{direction}' blocked at final safety layer - PRIMARY SAFETY FAILED!"
                    print(f"❌ FINAL SAFETY BLOCK: Movement command '{direction}' blocked in worker thread")
                    print(f"   └── This should NEVER happen - primary safety check failed!")
                    self.log_message(f"❌ CRITICAL: {error_msg}")
                    self.log_movement_activity(error_msg, "SAFETY")
                    return
                
                print(f"✅ SENDING: Command '{direction}' to robot {self.selected_robot_id}")
                self.log_movement_activity(f"Sending HTTP request: {direction} to {self.selected_robot_id}", "COMMAND")
                
                # Include safety confirmation for server-side validation
                toggle_state = self.movement_toggle_var.get()
                movement_allowed = self.movement_enabled and toggle_state
                
                command_data = {
                    'command_type': 'move' if direction != 'stop' else 'stop',
                    'parameters': {
                        'direction': direction,
                        'speed': 0.3 if direction != 'stop' else 0.0,
                        'duration': 2.0
                    },
                    'safety_confirmed': True,  # GUI confirms it has done safety checks
                    'gui_movement_enabled': movement_allowed  # Current safety state
                }
                
                # Log the exact command being sent
                self.log_movement_activity(f"Command payload: {json.dumps(command_data, indent=2)}", "INFO")
                
                response = requests.post(
                    f"{self.config.http_url}/robots/{self.selected_robot_id}/commands",
                    json=command_data, timeout=5
                )
                
                if response.status_code == 200:
                    success_msg = f"Command '{direction}' sent successfully to {self.selected_robot_id}"
                    self.log_message(f"✅ Command sent to {self.selected_robot_id}: {direction}")
                    self.log_movement_activity(success_msg, "SUCCESS")
                    print(f"✅ SUCCESS: Command '{direction}' sent successfully")
                    
                    # Log server response if available
                    try:
                        response_data = response.json()
                        if response_data:
                            self.log_movement_activity(f"Server response: {json.dumps(response_data, indent=2)}", "INFO")
                    except:
                        pass
                else:
                    error_msg = f"Command '{direction}' failed - HTTP {response.status_code}: {response.text}"
                    self.log_message(f"❌ Command failed: {response.text}")
                    self.log_movement_activity(error_msg, "ERROR")
                    print(f"❌ FAILED: Command '{direction}' failed - {response.text}")
                    
            except Exception as e:
                error_msg = f"Command '{direction}' error: {str(e)}"
                self.log_message(f"❌ Command error: {str(e)}")
                self.log_movement_activity(error_msg, "ERROR")
                print(f"❌ ERROR: Command '{direction}' error - {str(e)}")
        
        # Start command thread
        threading.Thread(target=send_command_worker, daemon=True).start()

    def load_image_for_analysis(self):
        """Load an image for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                # Resize for display
                image.thumbnail((300, 200))
                photo = ImageTk.PhotoImage(image)
                
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # Keep reference
                
                # Store full image for analysis
                self.current_image = Image.open(file_path)
                self.log_message(f"Image loaded: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def analyze_image(self):
        """Analyze the loaded image"""
        if not hasattr(self, 'current_image'):
            self.log_vila_activity("No image loaded - analysis cancelled", "ERROR")
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        if not self.selected_robot_id:
            self.log_vila_activity("No robot selected - analysis cancelled", "ERROR")
            messagebox.showwarning("No Robot Selected", "Please select a robot first")
            return
        
        # Log start of analysis
        prompt = self.analysis_prompt.get()
        self.log_vila_activity(f"Starting image analysis for robot {self.selected_robot_id}", "PROCESSING")
        self.log_vila_activity(f"Prompt: {prompt}", "PROMPT")
        
        # Update VILA status
        self.vila_status_var.set("Analyzing image...")
        self.vila_status_label.configure(foreground="orange")
        
        def analyze_worker():
            try:
                # STEP 1: Image Analysis Preparation
                self.log_pipeline_activity(f"🚀 Starting VILA → Robot Pipeline for robot {self.selected_robot_id}", "SYSTEM")
                self.log_pipeline_activity(f"📝 Prompt: {prompt}", "IMAGE")
                
                # Convert image to base64
                self.log_vila_activity("Converting image to base64 format", "PROCESSING")
                self.log_pipeline_activity("Preparing image for VILA analysis...", "IMAGE")
                
                # Update thumbnail in System Controls (on main thread)
                self.root.after(0, lambda: self.update_vila_thumbnail(self.current_image))
                
                buffer = io.BytesIO()
                self.current_image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Send analysis request
                analysis_data = {
                    'image': image_base64,
                    'prompt': prompt
                }
                
                self.log_vila_activity(f"Sending analysis request to server ({len(image_base64)} bytes)", "PROCESSING")
                self.log_pipeline_activity(f"Sending {len(image_base64):,} bytes to VILA model", "IMAGE")
                
                response = requests.post(
                    f"{self.config.http_url}/robots/{self.selected_robot_id}/analyze",
                    json=analysis_data, timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result.get('analysis', 'No analysis returned')
                    commands = result.get('commands', {})
                    
                    # STEP 2: VILA Response
                    self.log_vila_activity("Image analysis completed successfully", "SUCCESS")
                    self.log_vila_activity(f"Response: {analysis}", "RESPONSE")
                    
                    self.log_pipeline_activity("VILA analysis complete!", "RESPONSE")
                    self.log_pipeline_activity(f"📝 VILA says: \"{analysis}\"", "RESPONSE")
                    
                    # STEP 3: Command Parsing
                    self.log_pipeline_activity("Parsing VILA response for robot commands...", "PARSING")
                    
                    # Show the parsing logic in detail (matches server logic)
                    response_lower = analysis.lower()
                    parsing_details = []
                    
                    # Enhanced forward movement detection
                    forward_keywords = [
                        'move forward', 'go forward', 'proceed forward', 'continue forward',
                        'move ahead', 'go ahead', 'proceed ahead', 'continue ahead',
                        'move straight', 'go straight', 'continue straight',
                        'advance', 'proceed', 'continue moving', 'move in', 'keep moving',
                        'forward', 'ahead'
                    ]
                    
                    move_forward = any(keyword in response_lower for keyword in forward_keywords)
                    hazard_keywords = ['obstacle', 'blocked', 'danger', 'hazard', 'unsafe', 'collision', 'wall', 'barrier']
                    has_hazard = any(keyword in response_lower for keyword in hazard_keywords)
                    explicit_stop = any(phrase in response_lower for phrase in ['should not', 'cannot', 'do not', 'avoid'])
                    
                    # Check each command type with new logic
                    final_move_forward = move_forward and not has_hazard and not explicit_stop
                    
                    if final_move_forward:
                        found_keywords = [kw for kw in forward_keywords if kw in response_lower]
                        parsing_details.append(f"✓ Found forward keywords: {found_keywords[:2]} → move_forward = True")
                    else:
                        if not move_forward:
                            parsing_details.append("✗ No forward movement keywords → move_forward = False")
                        elif has_hazard:
                            parsing_details.append("✗ Hazard detected, blocking forward → move_forward = False")
                        elif explicit_stop:
                            parsing_details.append("✗ Explicit stop instruction → move_forward = False")
                        
                    if 'stop' in response_lower or has_hazard:
                        parsing_details.append("✓ Found 'stop' or hazard → stop = True")
                    else:
                        parsing_details.append("✗ No 'stop' or hazard → stop = False")
                        
                    if 'left' in response_lower:
                        parsing_details.append("✓ Found 'left' → turn_left = True")
                    else:
                        parsing_details.append("✗ No 'left' → turn_left = False")
                        
                    if 'right' in response_lower:
                        parsing_details.append("✓ Found 'right' → turn_right = True")
                    else:
                        parsing_details.append("✗ No 'right' → turn_right = False")
                        
                    hazard_words = ['danger', 'hazard', 'unsafe', 'obstacle']
                    hazard_found = any(word in response_lower for word in hazard_words)
                    if hazard_found:
                        parsing_details.append(f"⚠️ Found hazard keywords → hazard_detected = True")
                    else:
                        parsing_details.append("✓ No hazard keywords → hazard_detected = False")
                    
                    for detail in parsing_details:
                        self.log_pipeline_activity(detail, "PARSING")
                    
                    self.log_pipeline_activity(f"Parsed commands: {json.dumps(commands, indent=2)}", "PARSING")
                    
                    # STEP 4: Command Generation
                    if hasattr(self, 'vila_autonomous') and self.vila_autonomous:
                        self.log_pipeline_activity("Generating robot movement commands...", "GENERATION")
                        
                        # Simulate the generate_control_command logic
                        if commands.get('hazard_detected') or commands.get('stop'):
                            action = 'stop'
                            params = {'reason': 'hazard_detected'}
                            self.log_pipeline_activity("🛑 Safety priority: STOP command generated", "GENERATION")
                        elif commands.get('move_forward'):
                            action = 'move'
                            params = {'direction': 'forward', 'speed': 0.3, 'duration': 2.0}
                            self.log_pipeline_activity("➡️ Forward movement command generated", "GENERATION")
                        elif commands.get('turn_left'):
                            action = 'turn'
                            params = {'direction': 'left', 'angle': 45, 'speed': 0.2}
                            self.log_pipeline_activity("↪️ Left turn command generated", "GENERATION")
                        elif commands.get('turn_right'):
                            action = 'turn'
                            params = {'direction': 'right', 'angle': 45, 'speed': 0.2}
                            self.log_pipeline_activity("↩️ Right turn command generated", "GENERATION")
                        else:
                            action = 'stop'
                            params = {'reason': 'unclear_instruction'}
                            self.log_pipeline_activity("❓ Unclear instruction: STOP command generated for safety", "GENERATION")
                        
                        robot_command = {
                            'command_type': action,
                            'parameters': params,
                            'timestamp': datetime.now().isoformat(),
                            'priority': 2 if commands.get('hazard_detected') else 1,
                            'safety_confirmed': True,  # GUI confirms it has done safety checks
                            'gui_movement_enabled': movement_allowed  # Current safety state from above
                        }
                        
                        self.log_pipeline_activity(f"Generated robot command: {json.dumps(robot_command, indent=2)}", "GENERATION")
                        
                        # STEP 5: Robot Execution (if enabled AND toggle is on)
                        toggle_state = self.movement_toggle_var.get()
                        movement_allowed = self.movement_enabled and toggle_state
                        if movement_allowed:
                            self.log_pipeline_activity("Executing robot command...", "EXECUTION")
                            
                            # Send the generated command
                            try:
                                cmd_response = requests.post(
                                    f"{self.config.http_url}/robots/{self.selected_robot_id}/commands",
                                    json=robot_command, timeout=5
                                )
                                
                                if cmd_response.status_code == 200:
                                    self.log_pipeline_activity(f"✅ Robot command executed successfully: {action}", "EXECUTION")
                                    self.log_movement_activity(f"VILA autonomous command executed: {action}", "SUCCESS")
                                else:
                                    self.log_pipeline_activity(f"❌ Robot command failed: {cmd_response.text}", "EXECUTION")
                                    self.log_movement_activity(f"VILA autonomous command failed: {cmd_response.text}", "ERROR")
                                    
                            except Exception as cmd_e:
                                self.log_pipeline_activity(f"❌ Command execution error: {str(cmd_e)}", "EXECUTION")
                                self.log_movement_activity(f"VILA command error: {str(cmd_e)}", "ERROR")
                        else:
                            # Log specific reason why command was blocked
                            if not self.movement_enabled:
                                self.log_pipeline_activity("🚫 Movement system disabled - robot command NOT executed", "EXECUTION")
                                self.log_pipeline_activity("Enable movement system to allow robot execution", "EXECUTION")
                            elif not toggle_state:
                                self.log_pipeline_activity("🚫 Movement toggle disabled - robot command NOT executed", "EXECUTION")
                                self.log_pipeline_activity("Enable movement toggle to allow robot execution", "EXECUTION")
                            else:
                                self.log_pipeline_activity("🚫 Movement blocked - robot command NOT executed", "EXECUTION")
                                self.log_pipeline_activity("Check movement settings", "EXECUTION")
                    else:
                        self.log_pipeline_activity("🔒 Autonomous mode disabled - no robot commands generated", "GENERATION")
                        self.log_pipeline_activity("Enable 'VILA Auto Nav Mode' to see command generation", "GENERATION")
                    
                    if commands:
                        self.log_vila_activity(f"Parsed commands: {json.dumps(commands, indent=2)}", "INFO")
                    
                    def update_analysis_display():
                        self.analysis_results.delete(1.0, tk.END)
                        self.analysis_results.insert(tk.END, f"Analysis Result:\n{'-'*50}\n")
                        self.analysis_results.insert(tk.END, f"{analysis}\n\n")
                        self.analysis_results.insert(tk.END, f"Parsed Commands:\n{'-'*50}\n")
                        self.analysis_results.insert(tk.END, json.dumps(commands, indent=2))
                        
                        # Update VILA status
                        self.vila_status_var.set("Analysis complete")
                        self.vila_status_label.configure(foreground="green")
                    
                    self.root.after(0, update_analysis_display)
                    self.log_message(f"Image analysis completed for {self.selected_robot_id}")
                    
                    # Pipeline complete
                    self.log_pipeline_activity("🎉 VILA → Robot Pipeline complete!", "SYSTEM")
                    
                else:
                    error_msg = f"Analysis failed - HTTP {response.status_code}: {response.text}"
                    self.log_vila_activity(error_msg, "ERROR")
                    self.log_pipeline_activity(error_msg, "ERROR")
                    self.log_message(f"Analysis failed: {response.text}")
                    
                    def update_error_status():
                        self.vila_status_var.set("Analysis failed")
                        self.vila_status_label.configure(foreground="red")
                    
                    self.root.after(0, update_error_status)
                    
            except Exception as e:
                error_msg = f"Analysis error: {str(e)}"
                self.log_vila_activity(error_msg, "ERROR")
                self.log_pipeline_activity(error_msg, "ERROR")
                self.log_message(f"Analysis error: {str(e)}")
                
                def update_error_status():
                    self.vila_status_var.set("Analysis error")
                    self.vila_status_label.configure(foreground="red")
                
                self.root.after(0, update_error_status)
        
        threading.Thread(target=analyze_worker, daemon=True).start()

    # Settings methods
    def apply_settings(self):
        """Apply new server settings"""
        try:
            self.config.host = self.host_entry.get()
            self.config.http_port = int(self.http_port_entry.get())
            self.config.tcp_port = int(self.tcp_port_entry.get())
            
            # Update server label
            self.server_label.configure(text=self.config.http_url)
            
            # Disconnect and reconnect if connected
            if self.sio.connected:
                self.disconnect_from_server()
                self.root.after(1000, self.connect_to_server)
            
            self.log_message("Settings applied successfully")
            messagebox.showinfo("Settings", "Settings applied successfully!")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid port number")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {str(e)}")

    # Utility methods
    def log_message(self, message):
        """Log a message to the application log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        def update_log():
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
        
        if hasattr(self, 'log_text'):
            self.root.after(0, update_log)
        
        logger.info(message)

    def clear_log(self):
        """Clear the application log"""
        self.log_text.delete(1.0, tk.END)
    
    def clear_vila_activity(self):
        """Clear the VILA activity display"""
        self.vila_activity_text.delete(1.0, tk.END)
        self.vila_activity_text.insert(tk.END, f"🧹 VILA activity cleared at {datetime.now().strftime('%H:%M:%S')}\n\n")
    
    def clear_movement_activity(self):
        """Clear the movement activity display"""
        self.movement_activity_text.delete(1.0, tk.END)
        self.movement_activity_text.insert(tk.END, f"🧹 Movement commands cleared at {datetime.now().strftime('%H:%M:%S')}\n\n")
    
    def clear_pipeline_display(self):
        """Clear the pipeline display"""
        self.pipeline_text.delete(1.0, tk.END)
        self.pipeline_text.insert(tk.END, f"🧹 Pipeline cleared at {datetime.now().strftime('%H:%M:%S')}\n\n")
    
    def toggle_vila_autonomous(self):
        """Toggle VILA auto navigation mode"""
        try:
            autonomous_enabled = self.vila_autonomous_var.get()
            self.vila_autonomous = autonomous_enabled
            
            # Update button text - show current state
            if autonomous_enabled:
                self.vila_autonomous_btn.configure(text="VILA Auto Nav Mode\n✅ Enabled")
            else:
                self.vila_autonomous_btn.configure(text="VILA Auto Nav Mode\n❌ Disabled")
            
            # Log the change
            status = "Enabled" if autonomous_enabled else "Disabled"
            self.log_pipeline_activity(f"🤖 VILA Auto Nav Mode {status}", "SYSTEM")
            if autonomous_enabled:
                self.log_pipeline_activity(
                    "⚠️ VILA can now generate and execute robot navigation commands automatically\n"
                    "🛡️ Safety: Movement toggle must still be enabled for actual movement\n"
                    "🎯 Navigation commands will be executed when image analysis detects actionable scenarios", 
                    "SYSTEM"
                )
            else:
                self.log_pipeline_activity(
                    "🔒 VILA will only analyze images - no automatic navigation commands", 
                    "SYSTEM"
                )
                
            self.log_message(f"🤖 VILA auto navigation mode {status.lower()}")
            
        except Exception as e:
            self.log_message(f"❌ VILA auto nav toggle error: {str(e)}")
            print(f"VILA auto nav toggle error: {e}")
    
    def log_pipeline_activity(self, message, activity_type="INFO"):
        """Log activity to the pipeline display"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        
        # Color coding based on activity type
        if activity_type == "IMAGE":
            icon = "📷"
            prefix = "IMAGE ANALYSIS"
        elif activity_type == "RESPONSE":
            icon = "📝"
            prefix = "VILA RESPONSE"
        elif activity_type == "PARSING":
            icon = "🔍"
            prefix = "COMMAND PARSING"
        elif activity_type == "GENERATION":
            icon = "⚙️"
            prefix = "CMD GENERATION"
        elif activity_type == "EXECUTION":
            icon = "🤖"
            prefix = "ROBOT EXECUTION"
        elif activity_type == "SYSTEM":
            icon = "🔧"
            prefix = "SYSTEM"
        elif activity_type == "ERROR":
            icon = "❌"
            prefix = "ERROR"
        else:
            icon = "ℹ️"
            prefix = "INFO"
        
        log_entry = f"[{timestamp}] {icon} {prefix}: {message}\n"
        
        def update_pipeline_display():
            if hasattr(self, 'pipeline_text'):
                self.pipeline_text.insert(tk.END, log_entry)
                self.pipeline_text.see(tk.END)
        
        self.root.after(0, update_pipeline_display)
    
    def log_vila_activity(self, message, activity_type="INFO"):
        """Log VILA activity to the VILA activity display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding based on activity type
        if activity_type == "PROMPT":
            icon = "📤"
            prefix = "PROMPT"
        elif activity_type == "RESPONSE":
            icon = "📥"
            prefix = "RESPONSE"
        elif activity_type == "ERROR":
            icon = "❌"
            prefix = "ERROR"
        elif activity_type == "PROCESSING":
            icon = "⚙️"
            prefix = "PROCESSING"
        else:
            icon = "ℹ️"
            prefix = "INFO"
        
        # Check if verbose mode is enabled for VILA
        verbose_mode = getattr(self, 'verbose_vila_var', None)
        is_verbose = verbose_mode.get() if verbose_mode else True
        
        if is_verbose:
            # Verbose mode: show full message
            log_entry = f"[{timestamp}] {icon} {prefix}: {message}\n"
        else:
            # Compressed mode: show only VILA text responses
            compressed_message = self.compress_vila_message(message, activity_type)
            if compressed_message == "":
                # Empty string means suppress this message entirely in compressed mode
                log_entry = None
            elif compressed_message:
                log_entry = f"[{timestamp}] {icon} {compressed_message}\n"
            else:
                # Fallback to verbose if compression fails
                log_entry = f"[{timestamp}] {icon} {prefix}: {message}\n"
        
        def update_vila_display():
            if hasattr(self, 'vila_activity_text') and log_entry is not None:
                self.vila_activity_text.insert(tk.END, log_entry)
                # Only autoscroll if enabled (user hasn't manually scrolled up)
                if getattr(self, 'vila_autoscroll_enabled', True):
                    self.vila_activity_text.see(tk.END)
        
        self.root.after(0, update_vila_display)
    
    def compress_vila_message(self, message, activity_type):
        """Convert VILA messages to compressed format - only show VILA responses"""
        try:
            # In compressed mode, ONLY show VILA text responses
            if activity_type == "RESPONSE" and "VILA response:" in message:
                # Extract the VILA response text
                if "VILA response:" in message:
                    response_text = message.split("VILA response:", 1)[1].strip()
                    # Return just the response text without prefix
                    return response_text
            
            # Suppress all other messages in compressed mode (status updates, processing, etc.)
            return ""
            
        except Exception as e:
            print(f"VILA compression error: {e}")
            return ""
    
    def log_movement_activity(self, message, activity_type="INFO"):
        """Log movement activity to the movement activity display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding based on activity type
        if activity_type == "COMMAND":
            icon = "🎮"
            prefix = "COMMAND"
        elif activity_type == "SUCCESS":
            icon = "✅"
            prefix = "SUCCESS"
        elif activity_type == "BLOCKED":
            icon = "🚫"
            prefix = "BLOCKED"
        elif activity_type == "ERROR":
            icon = "❌"
            prefix = "ERROR"
        elif activity_type == "SAFETY":
            icon = "🛡️"
            prefix = "SAFETY"
        else:
            icon = "ℹ️"
            prefix = "INFO"
        
        # Check if verbose mode is enabled
        verbose_mode = getattr(self, 'verbose_movement_var', None)
        is_verbose = verbose_mode.get() if verbose_mode else True
        
        if is_verbose:
            # Verbose mode: show full message
            log_entry = f"[{timestamp}] {icon} {prefix}: {message}\n"
        else:
            # Compressed mode: convert to one-line format
            compressed_message = self.compress_movement_message(message, activity_type)
            if compressed_message == "":
                # Empty string means suppress this message entirely in compressed mode
                log_entry = None
            elif compressed_message:
                log_entry = f"[{timestamp}] {icon} {compressed_message}\n"
            else:
                # Fallback to verbose if compression fails
                log_entry = f"[{timestamp}] {icon} {prefix}: {message}\n"
        
        def update_movement_display():
            if hasattr(self, 'movement_activity_text') and log_entry is not None:
                self.movement_activity_text.insert(tk.END, log_entry)
                # Only autoscroll if enabled (user hasn't manually scrolled up)
                if getattr(self, 'movement_autoscroll_enabled', True):
                    self.movement_activity_text.see(tk.END)
        
        self.root.after(0, update_movement_display)
    
    def compress_movement_message(self, message, activity_type):
        """Convert verbose movement messages to compressed one-line format"""
        try:
            # SUPPRESS VERBOSE-ONLY MESSAGES: Filter out detailed info that shouldn't appear in compressed mode
            verbose_only_patterns = [
                "Command payload:",
                "Server response:",
                "Sending HTTP request:",
                "Command executed successfully",
                "Robot Control GUI initialized",
                "Server URL:",
                "Initial movement state:",
                "GUI Command Processor:",
                "Architecture: All robot movements"
            ]
            
            # Skip verbose-only messages entirely in compressed mode
            if any(pattern in message for pattern in verbose_only_patterns):
                return ""  # Return empty string to skip this message entirely
            
            # Skip INFO messages that aren't command-related
            if activity_type == "INFO" and not any(keyword in message.lower() for keyword in ["command", "move", "turn", "stop", "robot", "vila", "connected"]):
                return ""
            
            # Command execution patterns
            if "Sending GUI-processed command:" in message:
                # Extract command from "🚙 Sending GUI-processed command: right to robot yahboomcar_x3_01"
                if " to robot " in message:
                    command = message.split("command: ")[1].split(" to robot")[0].strip()
                    robot_id = message.split(" to robot ")[1].strip()
                    return self.format_command_compressed(command, robot_id)
            
            elif "Attempting to send" in message and "command to robot" in message:
                # SKIP in compressed mode - this is verbose detail
                return ""
            
            elif "Command sent to robot" in message or "'sent successfully" in message:
                # Extract success messages
                if ":" in message and "robot" in message:
                    parts = message.split(": ")
                    if len(parts) >= 2:
                        command = parts[-1].strip()
                        robot_part = parts[0].split("robot ")
                        robot_id = robot_part[1] if len(robot_part) > 1 else "unknown"
                        return f"✅ {self.format_command_compressed(command, robot_id, success=True)}"
                elif "'sent successfully" in message:
                    # Extract from success messages
                    if " to " in message:
                        parts = message.split(" to ")
                        robot_id = parts[-1].strip()
                        if "'" in message:
                            command = message.split("'")[1]
                            return f"✅ {self.format_command_compressed(command, robot_id, success=True)}"
            
            elif "Command" in message and "failed" in message:
                # Extract failed commands
                if "HTTP" in message:
                    error_code = message.split("HTTP ")[1].split(":")[0].strip() if "HTTP " in message else "error"
                    return f"❌ Command failed - {error_code}"
                else:
                    return f"❌ Command failed"
            
            # MOVEMENT COMMANDS ONLY: Suppress all non-movement messages in compressed mode
            # Only show actual robot movement commands: Stop, Move (forward/backward), Turn (left/right)
            
            # Return empty string if no compression pattern matches (suppress unknown messages)
            return ""
            
        except Exception as e:
            # Fallback to suppress on error
            print(f"Compression error: {e}")
            return ""
    
    def format_command_compressed(self, command, robot_id, success=False):
        """Format individual commands in compressed format"""
        try:
            # Debug logging to trace robot ID parsing
            # print(f"🔍 DEBUG: Formatting command '{command}' for robot_id '{robot_id}'")
            
            # Better robot ID shortening: keep more context
            if "_" in robot_id:
                parts = robot_id.split("_")
                if len(parts) >= 3:  # e.g., yahboomcar_x3_01
                    robot_short = f"{parts[-2]}_{parts[-1]}"  # x3_01
                else:
                    robot_short = parts[-1]  # fallback to last part
            else:
                robot_short = robot_id[:8]  # fallback for IDs without underscores
            
            # Debug logging
            # print(f"🔍 DEBUG: robot_id '{robot_id}' → robot_short '{robot_short}'")
            
            if command.lower() == "forward":
                return f"Move 30cm forward → {robot_short}"
            elif command.lower() == "backward":
                return f"Move 30cm backward → {robot_short}"
            elif command.lower() == "left":
                return f"Turn 45° CCW → {robot_short}"
            elif command.lower() == "right":
                return f"Turn 45° CW → {robot_short}"
            elif command.lower() == "stop":
                return f"Stop → {robot_short}"
            else:
                return f"{command.title()} → {robot_short}"
                
        except Exception as e:
            print(f"Error in format_command_compressed: {e}")
            return f"{command} → {robot_id}"
    
    def initialize_activity_displays(self):
        """Initialize the activity displays with startup information"""
        try:
            # Log startup information to movement activity
            self.log_movement_activity("Robot Control GUI initialized", "INFO")
            self.log_movement_activity(f"Server URL: {self.config.http_url}", "INFO")
            movement_status = "Enabled" if self.movement_enabled else "DISABLED"
            self.log_movement_activity(f"Initial movement state: {movement_status}", "SAFETY")
            
            # Log startup information to VILA activity
            self.log_vila_activity("VILA Activity Monitor initialized", "INFO")
            self.log_vila_activity(f"Ready to process image analysis requests", "INFO")
            
            # Start robot activity monitoring
            self.start_robot_activity_monitoring()
            
        except Exception as e:
            print(f"Error initializing activity displays: {e}")
    
    def toggle_vila_model(self):
        """Toggle VILA model enable/disable"""
        try:
            toggle_value = self.vila_enabled_var.get()
            self.vila_enabled = toggle_value
            
            status = "Enabled" if self.vila_enabled else "Disabled"
            
            # Debug logging
            print(f"🤖 VILA TOGGLE: {status} - toggle_var: {toggle_value}, vila_enabled: {self.vila_enabled}")
            
            # Log to VILA activity display
            vila_msg = f"VILA model toggle changed: {status} (enabled:{self.vila_enabled}, toggle:{toggle_value})"
            self.log_vila_activity(vila_msg, "PROCESSING")
            
            # Update button text
            self.vila_toggle_btn.configure(text=f"VILA Model\n{status}")
            
            # Update status display
            self.vila_model_status_var.set("Processing...")
            self.vila_status_display.configure(foreground="orange")
            
            # Send command to server
            def vila_toggle_worker():
                try:
                    if self.vila_enabled:
                        # Enable/Load VILA model
                        self.log_vila_activity("Sending VILA enable command to server", "PROCESSING")
                        response = requests.post(
                            f"{self.config.http_url}/vila/enable",
                            json={'action': 'enable'},
                            timeout=60  # Model loading can take time
                        )
                    else:
                        # Disable VILA model
                        self.log_vila_activity("Sending VILA disable command to server", "PROCESSING")
                        response = requests.post(
                            f"{self.config.http_url}/vila/disable",
                            json={'action': 'disable'},
                            timeout=10
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        success = result.get('success', False)
                        message = result.get('message', 'No message')
                        
                        def update_vila_status():
                            if success:
                                if self.vila_enabled:
                                    self.vila_model_status_var.set("Ready")
                                    self.vila_status_display.configure(foreground="green")
                                    self.log_vila_activity("VILA model enabled successfully", "SUCCESS")
                                else:
                                    self.vila_model_status_var.set("Disabled")
                                    self.vila_status_display.configure(foreground="gray")
                                    self.log_vila_activity("VILA model disabled successfully", "INFO")
                            else:
                                self.vila_model_status_var.set("Error")
                                self.vila_status_display.configure(foreground="red")
                                self.log_vila_activity(f"VILA toggle failed: {message}", "ERROR")
                                
                                # Reset toggle state on failure
                                self.vila_enabled_var.set(not self.vila_enabled)
                                self.vila_enabled = not self.vila_enabled
                                self.vila_toggle_btn.configure(text=f"VILA Model\n{'Enabled' if self.vila_enabled else 'Disabled'}")
                        
                        self.root.after(0, update_vila_status)
                    else:
                        error_msg = f"Server error: HTTP {response.status_code}"
                        self.log_vila_activity(error_msg, "ERROR")
                        
                        def update_error_status():
                            self.vila_model_status_var.set("Error")
                            self.vila_status_display.configure(foreground="red")
                            # Reset toggle state on failure
                            self.vila_enabled_var.set(not self.vila_enabled)
                            self.vila_enabled = not self.vila_enabled
                            self.vila_toggle_btn.configure(text=f"VILA Model\n{'Enabled' if self.vila_enabled else 'Disabled'}")
                        
                        self.root.after(0, update_error_status)
                        
                except Exception as e:
                    error_msg = f"VILA toggle error: {str(e)}"
                    self.log_vila_activity(error_msg, "ERROR")
                    
                    def update_error_status():
                        self.vila_model_status_var.set("Error")
                        self.vila_status_display.configure(foreground="red")
                        # Reset toggle state on failure
                        self.vila_enabled_var.set(not self.vila_enabled)
                        self.vila_enabled = not self.vila_enabled
                        self.vila_toggle_btn.configure(text=f"VILA Model\n{'Enabled' if self.vila_enabled else 'Disabled'}")
                    
                    self.root.after(0, update_error_status)
            
            # Start worker thread
            threading.Thread(target=vila_toggle_worker, daemon=True).start()
            
            # Log the change
            self.log_message(f"🤖 VILA model {status.lower()}")
            
        except Exception as e:
            self.log_message(f"❌ VILA toggle error: {str(e)}")
            self.log_vila_activity(f"Toggle error: {str(e)}", "ERROR")
            print(f"VILA toggle error: {e}")
    
    def check_vila_status(self):
        """Check VILA model status from server"""
        def status_worker():
            try:
                response = requests.get(f"{self.config.http_url}/vila/status", timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    loaded = result.get('model_loaded', False)
                    enabled = result.get('enabled', False)
                    status = result.get('status', 'Unknown')
                    
                    def update_status():
                        self.vila_enabled = enabled
                        self.vila_enabled_var.set(enabled)
                        
                        if loaded and enabled:
                            self.vila_model_status_var.set("Ready")
                            self.vila_status_display.configure(foreground="green")
                            self.vila_toggle_btn.configure(text="VILA Model\nEnabled")
                        elif loaded and not enabled:
                            self.vila_model_status_var.set("Loaded")
                            self.vila_status_display.configure(foreground="blue")
                            self.vila_toggle_btn.configure(text="VILA Model\nDisabled")
                        else:
                            self.vila_model_status_var.set("Not Loaded")
                            self.vila_status_display.configure(foreground="red")
                            self.vila_toggle_btn.configure(text="VILA Model\nDisabled")
                        
                        self.log_vila_activity(f"VILA status: {status} (loaded:{loaded}, enabled:{enabled})", "INFO")
                    
                    self.root.after(0, update_status)
                else:
                    def update_error():
                        self.vila_model_status_var.set("Server Error")
                        self.vila_status_display.configure(foreground="red")
                        self.log_vila_activity("Failed to get VILA status from server", "ERROR")
                    
                    self.root.after(0, update_error)
                    
            except Exception as status_error:
                error_msg = str(status_error)
                def update_error():
                    self.vila_model_status_var.set("Connection Error")
                    self.vila_status_display.configure(foreground="red")
                    print(f"VILA status check error: {error_msg}")
                
                self.root.after(0, update_error)
        
        threading.Thread(target=status_worker, daemon=True).start()
    
    def start_vila_status_refresh(self):
        """Start periodic VILA status refresh"""
        def periodic_vila_check():
            while True:
                try:
                    time.sleep(10)  # Check every 10 seconds
                    if hasattr(self, 'config') and self.config:
                        self.check_vila_status()
                except Exception as e:
                    print(f"Periodic VILA check error: {e}")
                    time.sleep(30)  # Wait longer on error
        
        vila_thread = threading.Thread(target=periodic_vila_check, daemon=True)
        vila_thread.start()
        print("🤖 Started periodic VILA status refresh (every 10 seconds)")
    
    def start_robot_activity_monitoring(self):
        """Start monitoring robot activities and processing VILA responses"""
        self.log_movement_activity("🎮 GUI Command Processor: Monitoring robot VILA analysis requests", "INFO")
        self.log_movement_activity("🔒 Architecture: All robot movements will be processed by GUI", "INFO")
        
        # Track processed analyses to avoid duplicates
        self.processed_analyses = set()
        
        def vila_response_processor():
            """Poll for VILA analyses that need GUI command processing"""
            while True:
                try:
                    time.sleep(3)  # Check every 3 seconds for pending VILA analyses
                    if hasattr(self, 'config') and self.config:
                        # Get pending VILA analyses from server
                        response = requests.get(f"{self.config.http_url}/vila/pending_analyses", timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            analyses = data.get('analyses', {})
                            
                            if analyses:
                                self.process_pending_vila_analyses(analyses)
                                    
                except Exception as e:
                    print(f"VILA response processor error: {e}")
                    time.sleep(10)  # Wait longer on error
        
        processor_thread = threading.Thread(target=vila_response_processor, daemon=True)
        processor_thread.start()
        print("🤖 Started VILA response processor - GUI will handle all robot movement commands")
    
    def process_pending_vila_analyses(self, analyses):
        """Process pending VILA analyses and generate movement commands"""
        try:
            for analysis_key, analysis_data in analyses.items():
                robot_id = analysis_data.get('robot_id')
                vila_response = analysis_data.get('vila_response', '')
                parsed_commands = analysis_data.get('parsed_commands', {})
                
                # Only process if GUI autonomous mode is enabled AND movement toggle is enabled
                toggle_state = self.movement_toggle_var.get()
                movement_allowed = self.movement_enabled and toggle_state
                if hasattr(self, 'vila_autonomous') and self.vila_autonomous and movement_allowed:
                    self.log_movement_activity(f"🎮 GUI Processing VILA analysis: {analysis_key}", "COMMAND")
                    self.log_vila_activity(f"Robot {robot_id}: VILA analysis intercepted by GUI", "PROCESSING")
                    self.log_vila_activity(f"VILA response: {vila_response[:100]}{'...' if len(vila_response) > 100 else ''}", "RESPONSE")
                    
                    # Generate movement command based on parsed VILA response
                    movement_command = self.generate_movement_from_vila_analysis(parsed_commands)
                    
                    if movement_command and robot_id:
                        # Set the robot as selected for command sending
                        original_selected = self.selected_robot_id
                        self.selected_robot_id = robot_id
                        
                        self.log_movement_activity(f"🚙 Sending GUI-processed command: {movement_command} to robot {robot_id}", "COMMAND")
                        
                        # Send movement command through normal GUI pathway (this ensures logging)
                        self.send_movement_command(movement_command)
                        
                        # SMART SELECTION RESTORE: Only restore to None if a robot was originally selected
                        # If no robot was selected before, keep the current robot selected for future commands
                        if original_selected is not None:
                            self.selected_robot_id = original_selected
                        else:
                            # Keep the robot that just processed the VILA command as selected
                            self.log_movement_activity(f"Auto-selected robot {robot_id} for future commands", "INFO")
                            # Update the GUI dropdown to reflect the selection
                            try:
                                if hasattr(self, 'robot_dropdown'):
                                    for i, (robot_id_option, _) in enumerate(self.robot_dropdown['values']):
                                        if robot_id_option == robot_id:
                                            self.robot_dropdown.current(i)
                                            break
                            except Exception as e:
                                print(f"Error updating robot dropdown: {e}")
                    
                    # Mark analysis as processed
                    self.mark_analysis_processed(analysis_key)
                    
                else:
                    # Log specific reason why VILA analysis was blocked
                    if not hasattr(self, 'vila_autonomous') or not self.vila_autonomous:
                        self.log_movement_activity(f"🔒 VILA analysis for robot {robot_id} - autonomous mode disabled", "INFO")
                    elif not self.movement_enabled:
                        self.log_movement_activity(f"🚫 VILA analysis for robot {robot_id} - movement system disabled", "BLOCKED")
                    elif not toggle_state:
                        self.log_movement_activity(f"🚫 VILA analysis for robot {robot_id} - movement toggle disabled", "BLOCKED")
                    else:
                        self.log_movement_activity(f"🔒 VILA analysis for robot {robot_id} - movement blocked (unknown reason)", "BLOCKED")
                    
                    # Still mark as processed to avoid reprocessing
                    self.mark_analysis_processed(analysis_key)
                    
        except Exception as e:
            print(f"Error processing pending VILA analyses: {e}")
            self.log_movement_activity(f"Error processing VILA analyses: {e}", "ERROR")
    
    def generate_movement_from_vila_analysis(self, parsed_commands):
        """Generate movement command from VILA parsed commands"""
        try:
            # Priority order: safety first, then movement
            if parsed_commands.get('hazard_detected', False) or parsed_commands.get('stop', False):
                return 'stop'
            elif parsed_commands.get('move_forward', False):
                return 'forward'
            elif parsed_commands.get('turn_left', False):
                return 'left'
            elif parsed_commands.get('turn_right', False):
                return 'right'
            else:
                # Default safe action
                return 'stop'
                
        except Exception as e:
            print(f"Error generating movement from VILA analysis: {e}")
            return 'stop'  # Safe default
    
    def mark_analysis_processed(self, analysis_key):
        """Mark VILA analysis as processed on server"""
        try:
            response = requests.post(
                f"{self.config.http_url}/vila/mark_processed",
                json={'analysis_key': analysis_key},
                timeout=5
            )
            if response.status_code == 200:
                self.log_vila_activity(f"Analysis {analysis_key} marked as processed", "INFO")
            else:
                print(f"Failed to mark analysis as processed: {response.text}")
        except Exception as e:
            print(f"Error marking analysis as processed: {e}")
    
    def start_robot_list_refresh(self):
        """Start periodic robot list refresh to show connected robots"""
        def refresh_robot_list():
            while True:
                try:
                    time.sleep(15)  # Check every 15 seconds
                    if hasattr(self, 'config') and self.config:
                        response = requests.get(f"{self.config.http_url}/robots", timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            robot_count = data.get('count', 0)
                            robots = data.get('robots', [])
                            
                            def update_robot_display():
                                if robot_count > 0:
                                    robot_names = [robot.get('name', robot.get('robot_id', 'Unknown')) for robot in robots]
                                    self.log_movement_activity(f"📡 Connected robots: {', '.join(robot_names)} ({robot_count} total)", "INFO")
                                    
                                    # Update robot dropdown if it exists
                                    if hasattr(self, 'robot_dropdown') and self.robot_dropdown:
                                        current_robots = [robot['robot_id'] for robot in robots]
                                        self.robot_dropdown['values'] = current_robots
                                        
                                        # Select first robot if none selected
                                        if not self.selected_robot_id and current_robots:
                                            self.robot_dropdown.set(current_robots[0])
                                            self.selected_robot_id = current_robots[0]
                                            self.log_movement_activity(f"Auto-selected robot: {current_robots[0]}", "INFO")
                            
                            self.root.after(0, update_robot_display)
                            
                except Exception as e:
                    print(f"Robot list refresh error: {e}")
                    time.sleep(30)  # Wait longer on error
        
        robot_list_thread = threading.Thread(target=refresh_robot_list, daemon=True)
        robot_list_thread.start()
        print("🤖 Started periodic robot list refresh (every 15 seconds)")


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = RobotGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    finally:
        if app.sio.connected:
            app.sio.disconnect()


if __name__ == '__main__':
    main()