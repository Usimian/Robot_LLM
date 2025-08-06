#!/usr/bin/env python3
"""
Unified Robot Client
Efficient client for the new unified robot controller system
Eliminates communication overhead and provides direct integration
"""

import asyncio
import aiohttp
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import base64
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('UnifiedRobotClient')

class UnifiedRobotClient:
    """
    High-efficiency robot client for the unified system
    Direct HTTP communication with minimal overhead
    """
    
    def __init__(self, robot_id: str, robot_name: str, server_host: str = "localhost", server_port: int = 5000):
        self.robot_id = robot_id
        self.robot_name = robot_name
        self.server_url = f"http://{server_host}:{server_port}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.registered = False
        
        # Robot capabilities
        self.capabilities = ["navigation", "vision", "sensors"]
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0, "heading": 0.0, "ip": "192.168.1.100"}
        
        # Sensor simulation (replace with real sensors)
        self.sensor_data = {
            "battery_voltage": 12.6,
            "battery_percentage": 85.0,
            "temperature": 23.5,
            "humidity": 45.0,
            "distance_front": 1.2,
            "distance_left": 0.8,
            "distance_right": 1.5,
            "wifi_signal": -45,
            "cpu_usage": 25.0,
            "memory_usage": 60.0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def register(self) -> bool:
        """Register robot with unified controller"""
        if not self.session:
            return False
            
        registration_data = {
            "robot_id": self.robot_id,
            "name": self.robot_name,
            "position": self.position,
            "battery_level": self.sensor_data["battery_percentage"],
            "capabilities": self.capabilities,
            "connection_type": "http",
            "sensor_data": self.sensor_data
        }
        
        try:
            async with self.session.post(
                f"{self.server_url}/api/robots/register",
                json=registration_data,
                timeout=10
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('success'):
                        self.registered = True
                        logger.info(f"✅ Robot {self.robot_id} registered successfully")
                        return True
                    else:
                        logger.error(f"❌ Registration failed: {result}")
                        return False
                else:
                    logger.error(f"❌ Registration failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Registration error: {e}")
            return False
    
    async def send_sensor_data(self, sensor_data: Optional[Dict[str, Any]] = None) -> bool:
        """Send sensor data to unified controller"""
        if not self.session or not self.registered:
            return False
        
        # Use provided data or default sensor data
        data_to_send = sensor_data or self.sensor_data
        
        # Add timestamp
        data_to_send["timestamp"] = datetime.now().isoformat()
        
        try:
            async with self.session.post(
                f"{self.server_url}/api/robots/{self.robot_id}/sensors",
                json=data_to_send,
                timeout=5
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('success', False)
                else:
                    logger.warning(f"⚠️ Sensor data failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Sensor data error: {e}")
            return False
    
    async def get_commands(self) -> List[Dict[str, Any]]:
        """Get pending commands from unified controller"""
        if not self.session or not self.registered:
            return []
        
        try:
            async with self.session.get(
                f"{self.server_url}/api/robots/{self.robot_id}/commands",
                timeout=5
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Command fetch error: {e}")
            return []
    
    async def send_image_for_analysis(self, image_data: bytes, prompt: str = "Analyze this image for robot navigation") -> Optional[str]:
        """Send image to unified VILA processor"""
        if not self.session or not self.registered:
            return None
        
        # Encode image
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        analysis_request = {
            "image": image_b64,
            "prompt": prompt,
            "robot_id": self.robot_id
        }
        
        try:
            async with self.session.post(
                f"{self.server_url}/api/vila/analyze",
                json=analysis_request,
                timeout=30  # VILA processing can take time
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('result')
                else:
                    logger.warning(f"⚠️ VILA analysis failed with status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ VILA analysis error: {e}")
            return None
    
    async def update_position(self, x: float, y: float, z: float = 0.0, heading: float = 0.0) -> bool:
        """Update robot position"""
        self.position.update({"x": x, "y": y, "z": z, "heading": heading})
        
        # Send as sensor data
        position_data = {
            **self.sensor_data,
            "position_x": x,
            "position_y": y,
            "position_z": z,
            "heading": heading
        }
        
        return await self.send_sensor_data(position_data)
    
    async def simulate_sensor_updates(self, duration: int = 60, interval: float = 2.0):
        """Simulate sensor data updates for testing"""
        logger.info(f"🔄 Starting sensor simulation for {duration} seconds...")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Simulate changing sensor values
            import random
            
            self.sensor_data.update({
                "battery_voltage": round(12.6 + random.uniform(-0.5, 0.2), 2),
                "battery_percentage": max(0, min(100, self.sensor_data["battery_percentage"] + random.uniform(-1, 0.5))),
                "temperature": round(23.5 + random.uniform(-2, 3), 1),
                "humidity": round(45.0 + random.uniform(-5, 5), 1),
                "distance_front": round(max(0.1, 1.2 + random.uniform(-0.3, 0.3)), 2),
                "distance_left": round(max(0.1, 0.8 + random.uniform(-0.2, 0.4)), 2),
                "distance_right": round(max(0.1, 1.5 + random.uniform(-0.4, 0.2)), 2),
                "wifi_signal": -45 + random.randint(-10, 5),
                "cpu_usage": round(max(5, min(95, 25.0 + random.uniform(-10, 20))), 1),
                "memory_usage": round(max(20, min(90, 60.0 + random.uniform(-15, 15))), 1)
            })
            
            # Send sensor data
            success = await self.send_sensor_data()
            if success:
                logger.info(f"📊 Sensor data sent: Battery {self.sensor_data['battery_percentage']:.1f}%, Temp {self.sensor_data['temperature']}°C")
            else:
                logger.warning("⚠️ Failed to send sensor data")
            
            await asyncio.sleep(interval)
        
        logger.info("✅ Sensor simulation completed")
    
    async def test_vila_analysis(self, test_image_path: str = None):
        """Test VILA analysis with a sample image"""
        logger.info("🧪 Testing VILA analysis...")
        
        # Create a simple test image if none provided
        if test_image_path and Path(test_image_path).exists():
            with open(test_image_path, 'rb') as f:
                image_data = f.read()
        else:
            # Create a simple test image
            from PIL import Image
            import io
            
            # Create a simple colored rectangle
            img = Image.new('RGB', (640, 480), color='blue')
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG')
            image_data = img_buffer.getvalue()
        
        # Send for analysis
        result = await self.send_image_for_analysis(
            image_data,
            "What do you see in this image? Suggest robot navigation actions."
        )
        
        if result:
            logger.info(f"✅ VILA Analysis Result: {result}")
        else:
            logger.warning("⚠️ VILA analysis failed")
        
        return result

async def demo_unified_client():
    """Demonstration of the unified robot client"""
    print("🤖 Unified Robot Client Demo")
    print("=" * 40)
    
    # Create client
    async with UnifiedRobotClient("demo_robot_001", "Demo Robot") as client:
        
        # Step 1: Register
        print("1️⃣ Registering robot...")
        success = await client.register()
        if not success:
            print("❌ Registration failed - is the unified controller running?")
            return
        
        # Step 2: Send initial sensor data
        print("2️⃣ Sending initial sensor data...")
        await client.send_sensor_data()
        
        # Step 3: Test VILA analysis
        print("3️⃣ Testing VILA analysis...")
        await client.test_vila_analysis()
        
        # Step 4: Simulate sensor updates
        print("4️⃣ Simulating sensor updates (30 seconds)...")
        await client.simulate_sensor_updates(duration=30, interval=3.0)
        
        # Step 5: Update position
        print("5️⃣ Updating position...")
        await client.update_position(1.5, 2.3, 0.0, 45.0)
        
        print("✅ Demo completed successfully!")

async def performance_test():
    """Performance test comparing old vs new system"""
    print("⚡ Performance Test: Unified System")
    print("=" * 40)
    
    async with UnifiedRobotClient("perf_test_robot", "Performance Test Robot") as client:
        
        # Register
        await client.register()
        
        # Test rapid sensor updates
        print("📊 Testing rapid sensor updates...")
        start_time = time.time()
        
        success_count = 0
        total_requests = 50
        
        for i in range(total_requests):
            success = await client.send_sensor_data({
                "test_sensor": i,
                "timestamp": datetime.now().isoformat()
            })
            if success:
                success_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"📈 Results:")
        print(f"  • Total requests: {total_requests}")
        print(f"  • Successful: {success_count}")
        print(f"  • Duration: {duration:.2f} seconds")
        print(f"  • Rate: {total_requests/duration:.1f} requests/second")
        print(f"  • Average latency: {(duration/total_requests)*1000:.1f} ms")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Robot Client")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--performance", action="store_true", help="Run performance test")
    parser.add_argument("--robot-id", default="test_robot_001", help="Robot ID")
    parser.add_argument("--robot-name", default="Test Robot", help="Robot name")
    parser.add_argument("--server-host", default="localhost", help="Server host")
    parser.add_argument("--server-port", type=int, default=5000, help="Server port")
    
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(demo_unified_client())
    elif args.performance:
        asyncio.run(performance_test())
    else:
        print("Unified Robot Client")
        print("Usage:")
        print("  --demo          Run demonstration")
        print("  --performance   Run performance test")
        print("  --robot-id      Set robot ID")
        print("  --robot-name    Set robot name")
        print("  --server-host   Set server host")
        print("  --server-port   Set server port")