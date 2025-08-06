#!/usr/bin/env python3
"""
Robot System Launcher
Simple launcher for the robot server and GUI components
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import time
import sys
import os
import signal
from pathlib import Path

class RobotLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot System Launcher")
        self.root.geometry("600x400")
        
        # Process tracking
        self.server_process = None
        self.gui_process = None
        
        self.create_gui()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_gui(self):
        """Create the launcher GUI"""
        # Title
        title_label = ttk.Label(self.root, text="Robot System Launcher", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Server section
        server_frame = ttk.LabelFrame(self.root, text="VILA Robot Server", padding=10)
        server_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.server_status_var = tk.StringVar(value="Stopped")
        ttk.Label(server_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.server_status_label = ttk.Label(server_frame, textvariable=self.server_status_var,
                                           foreground="red", font=("Arial", 10, "bold"))
        self.server_status_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        server_button_frame = ttk.Frame(server_frame)
        server_button_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.start_server_btn = ttk.Button(server_button_frame, text="Start Server", 
                                          command=self.start_server)
        self.start_server_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_server_btn = ttk.Button(server_button_frame, text="Stop Server", 
                                         command=self.stop_server, state=tk.DISABLED)
        self.stop_server_btn.pack(side=tk.LEFT, padx=2)
        
        # GUI section
        gui_frame = ttk.LabelFrame(self.root, text="Robot Control GUI", padding=10)
        gui_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.gui_status_var = tk.StringVar(value="Stopped")
        ttk.Label(gui_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.gui_status_label = ttk.Label(gui_frame, textvariable=self.gui_status_var,
                                         foreground="red", font=("Arial", 10, "bold"))
        self.gui_status_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        gui_button_frame = ttk.Frame(gui_frame)
        gui_button_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.start_gui_btn = ttk.Button(gui_button_frame, text="Start GUI", 
                                       command=self.start_gui)
        self.start_gui_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_gui_btn = ttk.Button(gui_button_frame, text="Stop GUI", 
                                      command=self.stop_gui, state=tk.DISABLED)
        self.stop_gui_btn.pack(side=tk.LEFT, padx=2)
        
        # Quick start section
        quick_frame = ttk.LabelFrame(self.root, text="Quick Start", padding=10)
        quick_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(quick_frame, text="Start Both (Server + GUI)", 
                  command=self.start_both).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(quick_frame, text="Stop All", 
                  command=self.stop_all).pack(side=tk.LEFT, padx=5)
        
        # Log output
        log_frame = ttk.LabelFrame(self.root, text="System Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        from tkinter import scrolledtext
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).pack(pady=5)
        
        # Initial log message
        self.log_message("Robot System Launcher ready")

    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        """Clear the log"""
        self.log_text.delete(1.0, tk.END)

    def start_server(self):
        """Start the robot server"""
        if self.server_process and self.server_process.poll() is None:
            self.log_message("Server is already running")
            return
        
        try:
            # Check if server file exists
            if not Path("robot_vila_server.py").exists():
                messagebox.showerror("Error", "robot_vila_server.py not found!")
                return
            
            self.log_message("Starting VILA Robot Server...")
            self.server_process = subprocess.Popen(
                [sys.executable, "robot_vila_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Start thread to read server output
            threading.Thread(target=self.read_server_output, daemon=True).start()
            
            # Update UI
            self.server_status_var.set("Starting...")
            self.server_status_label.configure(foreground="orange")
            self.start_server_btn.configure(state=tk.DISABLED)
            self.stop_server_btn.configure(state=tk.NORMAL)
            
            # Check if server started successfully after a delay
            self.root.after(3000, self.check_server_status)
            
        except Exception as e:
            self.log_message(f"Error starting server: {str(e)}")
            messagebox.showerror("Error", f"Failed to start server: {str(e)}")

    def read_server_output(self):
        """Read server output in background thread"""
        try:
            while self.server_process and self.server_process.poll() is None:
                line = self.server_process.stdout.readline()
                if line:
                    self.root.after(0, lambda: self.log_message(f"SERVER: {line.strip()}"))
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"Server output error: {str(e)}"))

    def check_server_status(self):
        """Check if server is running properly"""
        if self.server_process and self.server_process.poll() is None:
            self.server_status_var.set("Running")
            self.server_status_label.configure(foreground="green")
            self.log_message("Server started successfully")
        else:
            self.server_status_var.set("Failed")
            self.server_status_label.configure(foreground="red")
            self.start_server_btn.configure(state=tk.NORMAL)
            self.stop_server_btn.configure(state=tk.DISABLED)
            self.log_message("Server failed to start")

    def stop_server(self):
        """Stop the robot server"""
        if self.server_process:
            try:
                self.log_message("Stopping server...")
                
                # Try graceful shutdown first
                self.server_process.terminate()
                
                # Wait a moment for graceful shutdown
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.server_process.kill()
                    self.log_message("Server force killed")
                
                self.server_process = None
                
                # Update UI
                self.server_status_var.set("Stopped")
                self.server_status_label.configure(foreground="red")
                self.start_server_btn.configure(state=tk.NORMAL)
                self.stop_server_btn.configure(state=tk.DISABLED)
                
                self.log_message("Server stopped")
                
            except Exception as e:
                self.log_message(f"Error stopping server: {str(e)}")

    def start_gui(self):
        """Start the robot GUI"""
        if self.gui_process and self.gui_process.poll() is None:
            self.log_message("GUI is already running")
            return
        
        try:
            # Check if GUI file exists
            if not Path("robot_gui.py").exists():
                messagebox.showerror("Error", "robot_gui.py not found!")
                return
            
            self.log_message("Starting Robot Control GUI...")
            self.gui_process = subprocess.Popen(
                [sys.executable, "robot_gui.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Start thread to read GUI output
            threading.Thread(target=self.read_gui_output, daemon=True).start()
            
            # Update UI
            self.gui_status_var.set("Running")
            self.gui_status_label.configure(foreground="green")
            self.start_gui_btn.configure(state=tk.DISABLED)
            self.stop_gui_btn.configure(state=tk.NORMAL)
            
            self.log_message("GUI started successfully")
            
        except Exception as e:
            self.log_message(f"Error starting GUI: {str(e)}")
            messagebox.showerror("Error", f"Failed to start GUI: {str(e)}")

    def read_gui_output(self):
        """Read GUI output in background thread"""
        try:
            while self.gui_process and self.gui_process.poll() is None:
                line = self.gui_process.stdout.readline()
                if line:
                    self.root.after(0, lambda: self.log_message(f"GUI: {line.strip()}"))
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"GUI output error: {str(e)}"))
        finally:
            # GUI process ended, update UI
            self.root.after(0, self.gui_process_ended)

    def gui_process_ended(self):
        """Handle GUI process ending"""
        self.gui_status_var.set("Stopped")
        self.gui_status_label.configure(foreground="red")
        self.start_gui_btn.configure(state=tk.NORMAL)
        self.stop_gui_btn.configure(state=tk.DISABLED)
        self.log_message("GUI stopped")

    def stop_gui(self):
        """Stop the robot GUI"""
        if self.gui_process:
            try:
                self.log_message("Stopping GUI...")
                
                # Try graceful shutdown first
                self.gui_process.terminate()
                
                # Wait a moment for graceful shutdown
                try:
                    self.gui_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.gui_process.kill()
                    self.log_message("GUI force killed")
                
                self.gui_process = None
                
                # Update UI
                self.gui_status_var.set("Stopped")
                self.gui_status_label.configure(foreground="red")
                self.start_gui_btn.configure(state=tk.NORMAL)
                self.stop_gui_btn.configure(state=tk.DISABLED)
                
                self.log_message("GUI stopped")
                
            except Exception as e:
                self.log_message(f"Error stopping GUI: {str(e)}")

    def start_both(self):
        """Start both server and GUI"""
        self.log_message("Starting complete robot system...")
        
        # Start server first
        self.start_server()
        
        # Start GUI after a delay to let server initialize
        self.root.after(5000, self.start_gui)

    def stop_all(self):
        """Stop all processes"""
        self.log_message("Stopping all processes...")
        self.stop_gui()
        self.stop_server()

    def on_closing(self):
        """Handle window closing"""
        self.stop_all()
        # Give processes time to stop
        time.sleep(2)
        self.root.destroy()


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = RobotLauncher(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    finally:
        # Cleanup any remaining processes
        if hasattr(app, 'server_process') and app.server_process:
            try:
                app.server_process.terminate()
            except:
                pass
        if hasattr(app, 'gui_process') and app.gui_process:
            try:
                app.gui_process.terminate()
            except:
                pass


if __name__ == '__main__':
    main()