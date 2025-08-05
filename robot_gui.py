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

    def create_gui(self):
        """Create the main GUI interface"""
        # Top frame to hold notebook and controls
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control frame for movement toggle (right side)
        control_frame = ttk.LabelFrame(top_frame, text="Safety Controls", padding=5)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Movement toggle button
        self.movement_toggle_var = tk.BooleanVar(value=True)
        self.movement_toggle_btn = ttk.Checkbutton(
            control_frame,
            text="Movement\nEnabled",
            variable=self.movement_toggle_var,
            command=self.toggle_movement
        )
        self.movement_toggle_btn.pack(pady=10, padx=5)
        
        # Ensure initial state is synchronized and call update immediately
        self.movement_enabled = self.movement_toggle_var.get()
        print(f"🔧 INIT: Movement toggle initialized - enabled: {self.movement_enabled}")
        
        # Force initial button state update
        self.root.after(100, self.update_movement_buttons_state)
        
        # Main notebook for tabs
        self.notebook = ttk.Notebook(top_frame)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_status_tab()
        self.create_robots_tab()
        self.create_control_tab()
        self.create_monitoring_tab()
        self.create_settings_tab()

    def toggle_movement(self):
        """Toggle robot movement enable/disable"""
        try:
            # Update both variables to ensure sync
            toggle_value = self.movement_toggle_var.get()
            self.movement_enabled = toggle_value
            
            status = "Enabled" if self.movement_enabled else "DISABLED"
            
            # Debug logging
            print(f"🔄 TOGGLE: Movement {status} - toggle_var: {toggle_value}, movement_enabled: {self.movement_enabled}")
            
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
        
        # Placeholder sensor displays
        sensors = [
            ("Lidar Distance", "meters"), ("IMU Heading", "degrees"), 
            ("GPS Position", "lat/lon"), ("Camera Status", "active/inactive"),
            ("Battery Voltage", "volts"), ("Motor Temperature", "°C")
        ]
        
        self.sensor_vars = {}
        for i, (sensor, unit) in enumerate(sensors):
            row, col = i // 2, (i % 2) * 2
            
            ttk.Label(sensor_grid, text=f"{sensor}:").grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=f"-- {unit}")
            self.sensor_vars[sensor] = var
            ttk.Label(sensor_grid, textvariable=var).grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=2)

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
                    
                elif update_type == 'connection_status':
                    self.update_connection_status(data)
                    
        except queue.Empty:
            pass

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



    # Control methods
    def send_movement_command(self, direction):
        """Send movement command to selected robot"""
        # Debug logging
        toggle_state = self.movement_toggle_var.get()
        print(f"🔍 DEBUG: Movement command '{direction}' - Toggle state: {toggle_state}, movement_enabled: {self.movement_enabled}")
        
        if not self.selected_robot_id:
            messagebox.showwarning("No Robot Connected", "No robot is currently connected. Please ensure your robot is registered with the server.")
            return
        
        # CRITICAL SAFETY CHECK: Block ALL movement commands when disabled (ONLY allow STOP for emergency)
        # Check BOTH variables to ensure safety
        toggle_state = self.movement_toggle_var.get()
        movement_allowed = self.movement_enabled and toggle_state
        
        # ONLY allow STOP commands when movement is disabled - block everything else
        if direction != 'stop' and not movement_allowed:
            print(f"🚫 CRITICAL SAFETY BLOCK: Movement command '{direction}' blocked")
            print(f"   └── movement_enabled: {self.movement_enabled}, toggle_state: {toggle_state}")
            print(f"   └── ONLY 'stop' commands allowed when movement disabled")
            self.log_message(f"🚫 BLOCKED: Movement '{direction}' - Safety disabled (enabled:{self.movement_enabled}, toggle:{toggle_state})")
            
            # Also log to server logs for debugging
            print(f"🔴 SERVER LOG: Movement command '{direction}' BLOCKED by safety system")
            return
        
        def send_command_worker():
            try:
                # DOUBLE SAFETY CHECK: Final verification before sending HTTP request
                # ABSOLUTE SAFETY: Block ANY command that is not 'stop' when movement disabled
                if direction != 'stop' and (not self.movement_enabled or not self.movement_toggle_var.get()):
                    print(f"❌ FINAL SAFETY BLOCK: Movement command '{direction}' blocked in worker thread")
                    print(f"   └── This should NEVER happen - primary safety check failed!")
                    self.log_message(f"❌ CRITICAL: Command '{direction}' blocked at final safety layer - PRIMARY SAFETY FAILED!")
                    return
                
                print(f"✅ SENDING: Command '{direction}' to robot {self.selected_robot_id}")
                
                command_data = {
                    'command_type': 'move' if direction != 'stop' else 'stop',
                    'parameters': {
                        'direction': direction,
                        'speed': 0.3 if direction != 'stop' else 0.0,
                        'duration': 2.0
                    }
                }
                
                response = requests.post(
                    f"{self.config.http_url}/robots/{self.selected_robot_id}/commands",
                    json=command_data, timeout=5
                )
                
                if response.status_code == 200:
                    self.log_message(f"✅ Command sent to {self.selected_robot_id}: {direction}")
                    print(f"✅ SUCCESS: Command '{direction}' sent successfully")
                else:
                    self.log_message(f"❌ Command failed: {response.text}")
                    print(f"❌ FAILED: Command '{direction}' failed - {response.text}")
                    
            except Exception as e:
                self.log_message(f"❌ Command error: {str(e)}")
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
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        if not self.selected_robot_id:
            messagebox.showwarning("No Robot Selected", "Please select a robot first")
            return
        
        def analyze_worker():
            try:
                # Convert image to base64
                buffer = io.BytesIO()
                self.current_image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Send analysis request
                analysis_data = {
                    'image': image_base64,
                    'prompt': self.analysis_prompt.get()
                }
                
                response = requests.post(
                    f"{self.config.http_url}/robots/{self.selected_robot_id}/analyze",
                    json=analysis_data, timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result.get('analysis', 'No analysis returned')
                    commands = result.get('commands', {})
                    
                    def update_analysis_display():
                        self.analysis_results.delete(1.0, tk.END)
                        self.analysis_results.insert(tk.END, f"Analysis Result:\n{'-'*50}\n")
                        self.analysis_results.insert(tk.END, f"{analysis}\n\n")
                        self.analysis_results.insert(tk.END, f"Parsed Commands:\n{'-'*50}\n")
                        self.analysis_results.insert(tk.END, json.dumps(commands, indent=2))
                    
                    self.root.after(0, update_analysis_display)
                    self.log_message(f"Image analysis completed for {self.selected_robot_id}")
                else:
                    self.log_message(f"Analysis failed: {response.text}")
                    
            except Exception as e:
                self.log_message(f"Analysis error: {str(e)}")
        
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