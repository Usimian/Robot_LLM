#!/usr/bin/env python3
"""
ROS2 Robot System Launcher
Launch script for the complete ROS2-based robot system
Replaces HTTP-based launcher with ROS2 nodes
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ROS2SystemLauncher')

class ROS2SystemLauncher:
    """
    Launcher for the complete ROS2 robot system
    Manages all ROS2 nodes and ensures proper startup order
    """
    
    def __init__(self):
        self.processes = {}
        self.shutdown_requested = False
        
        # Check ROS2 environment
        self._check_ros2_environment()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _check_ros2_environment(self):
        """Check if ROS2 environment is properly set up"""
        if 'ROS_DISTRO' not in os.environ:
            logger.error("ROS2 environment not sourced. Please run:")
            logger.error("source /opt/ros/<distro>/setup.bash")
            sys.exit(1)
        
        logger.info(f"ROS2 environment detected: {os.environ['ROS_DISTRO']}")
        
        # Check if custom messages are built
        try:
            import robot_msgs
            logger.info("✅ Custom robot_msgs package found")
        except ImportError:
            logger.error("❌ robot_msgs package not found. Please build it first:")
            logger.error("cd /home/marc/Robot_LLM && colcon build --packages-select robot_msgs")
            logger.error("source install/setup.bash")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_requested = True
        self.shutdown_all()
    
    def start_node(self, name: str, script_path: str, args: list = None, delay: float = 0):
        """Start a ROS2 node"""
        if delay > 0:
            logger.info(f"Waiting {delay}s before starting {name}...")
            time.sleep(delay)
        
        if not Path(script_path).exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        try:
            logger.info(f"Starting {name}...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes[name] = process
            
            # Start output monitoring thread
            threading.Thread(
                target=self._monitor_process_output,
                args=(name, process),
                daemon=True
            ).start()
            
            logger.info(f"✅ {name} started (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start {name}: {e}")
            return False
    
    def _monitor_process_output(self, name: str, process: subprocess.Popen):
        """Monitor process output"""
        try:
            while process.poll() is None and not self.shutdown_requested:
                line = process.stdout.readline()
                if line:
                    logger.info(f"[{name}] {line.strip()}")
        except Exception as e:
            logger.error(f"Error monitoring {name}: {e}")
        finally:
            if process.poll() is not None:
                logger.warning(f"Process {name} terminated with code {process.returncode}")
    
    def stop_node(self, name: str):
        """Stop a specific node"""
        if name in self.processes:
            process = self.processes[name]
            try:
                logger.info(f"Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                    logger.info(f"✅ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}...")
                    process.kill()
                    process.wait()
                    logger.info(f"✅ {name} force stopped")
                
                del self.processes[name]
                
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
    
    def shutdown_all(self):
        """Shutdown all nodes"""
        logger.info("Shutting down all nodes...")
        
        # Stop nodes in reverse order
        node_names = list(self.processes.keys())
        for name in reversed(node_names):
            self.stop_node(name)
        
        logger.info("All nodes stopped")
    
    def check_node_health(self):
        """Check health of all running nodes"""
        healthy_nodes = []
        failed_nodes = []
        
        for name, process in self.processes.items():
            if process.poll() is None:
                healthy_nodes.append(name)
            else:
                failed_nodes.append(name)
        
        if failed_nodes:
            logger.warning(f"Failed nodes: {failed_nodes}")
        
        return healthy_nodes, failed_nodes
    
    def launch_complete_system(self):
        """Launch the complete ROS2 robot system"""
        logger.info("🚀 Launching complete ROS2 robot system...")
        logger.info("   └── All communication via ROS2 topics and services")
        logger.info("   └── Single command gateway maintained [[memory:5366669]]")
        
        # Start nodes in proper order
        startup_sequence = [
            # 1. Core VILA server (command gateway)
            {
                'name': 'vila_server',
                'script': 'robot_vila_server_ros2.py',
                'delay': 0
            },
            
            # 2. VILA vision node
            {
                'name': 'vila_vision',
                'script': 'vila_ros2_node.py',
                'delay': 2
            },
            
            # 3. Robot client (if running on same machine for testing)
            {
                'name': 'robot_client',
                'script': 'robot_client_ros2.py',
                'delay': 4
            },
            
            # 4. GUI (optional)
            {
                'name': 'robot_gui',
                'script': 'robot_gui_ros2.py',
                'delay': 6
            },
            
            # 5. Gateway validator (monitoring)
            {
                'name': 'gateway_validator',
                'script': 'ros2_command_gateway_validator.py',
                'delay': 8
            }
        ]
        
        # Start each node
        for node_config in startup_sequence:
            if self.shutdown_requested:
                break
                
            success = self.start_node(
                node_config['name'],
                node_config['script'],
                delay=node_config['delay']
            )
            
            if not success:
                logger.error(f"Failed to start {node_config['name']}, aborting launch")
                self.shutdown_all()
                return False
        
        logger.info("🎉 ROS2 robot system launch complete!")
        logger.info("📊 System Status:")
        
        # Wait a moment for nodes to initialize
        time.sleep(2)
        
        # Check system health
        healthy, failed = self.check_node_health()
        logger.info(f"   └── Healthy nodes: {len(healthy)}")
        logger.info(f"   └── Failed nodes: {len(failed)}")
        
        if failed:
            logger.warning(f"   └── Failed: {failed}")
        
        return len(failed) == 0
    
    def launch_minimal_system(self):
        """Launch minimal system (server + vision only)"""
        logger.info("🚀 Launching minimal ROS2 system...")
        
        startup_sequence = [
            {
                'name': 'vila_server',
                'script': 'robot_vila_server_ros2.py',
                'delay': 0
            },
            {
                'name': 'vila_vision',
                'script': 'vila_ros2_node.py',
                'delay': 2
            }
        ]
        
        for node_config in startup_sequence:
            if self.shutdown_requested:
                break
                
            success = self.start_node(
                node_config['name'],
                node_config['script'],
                delay=node_config['delay']
            )
            
            if not success:
                logger.error(f"Failed to start {node_config['name']}")
                return False
        
        logger.info("✅ Minimal ROS2 system launched")
        return True
    
    def run_interactive(self):
        """Run in interactive mode"""
        logger.info("🖥️ Interactive ROS2 System Controller")
        logger.info("Commands: start, stop <node>, status, health, quit")
        
        while not self.shutdown_requested:
            try:
                cmd = input("\nros2_system> ").strip().split()
                
                if not cmd:
                    continue
                
                if cmd[0] == 'start':
                    if len(cmd) > 1:
                        if cmd[1] == 'complete':
                            self.launch_complete_system()
                        elif cmd[1] == 'minimal':
                            self.launch_minimal_system()
                        else:
                            logger.info("Usage: start [complete|minimal]")
                    else:
                        logger.info("Usage: start [complete|minimal]")
                
                elif cmd[0] == 'stop':
                    if len(cmd) > 1:
                        self.stop_node(cmd[1])
                    else:
                        self.shutdown_all()
                
                elif cmd[0] == 'status':
                    logger.info(f"Running nodes: {list(self.processes.keys())}")
                
                elif cmd[0] == 'health':
                    healthy, failed = self.check_node_health()
                    logger.info(f"Healthy: {healthy}")
                    logger.info(f"Failed: {failed}")
                
                elif cmd[0] in ['quit', 'exit']:
                    break
                
                else:
                    logger.info("Unknown command. Available: start, stop, status, health, quit")
            
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        self.shutdown_all()
    
    def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        try:
            while not self.shutdown_requested and any(p.poll() is None for p in self.processes.values()):
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown_all()

def main():
    """Main function"""
    launcher = ROS2SystemLauncher()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'complete':
            # Launch complete system and wait
            if launcher.launch_complete_system():
                launcher.wait_for_shutdown()
        
        elif command == 'minimal':
            # Launch minimal system and wait
            if launcher.launch_minimal_system():
                launcher.wait_for_shutdown()
        
        elif command == 'interactive':
            # Run in interactive mode
            launcher.run_interactive()
        
        elif command == 'gui':
            # Launch just the GUI
            launcher.start_node('robot_gui', 'robot_gui_ros2.py')
            launcher.wait_for_shutdown()
        
        else:
            print("Usage: python launch_ros2_system.py [complete|minimal|interactive|gui]")
            print("  complete    - Launch complete system (server, vision, client, GUI)")
            print("  minimal     - Launch minimal system (server, vision only)")
            print("  interactive - Interactive mode")
            print("  gui         - Launch GUI only")
    
    else:
        # Default: launch complete system
        if launcher.launch_complete_system():
            launcher.wait_for_shutdown()

if __name__ == '__main__':
    main()
