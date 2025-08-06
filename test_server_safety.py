#!/usr/bin/env python3
"""
Test Server Safety Blocking
Verifies that the server properly blocks movement commands without safety confirmation
"""

import requests
import json

def test_server_safety_blocking():
    """Test that server blocks unauthorized movement commands"""
    server_url = "http://localhost:5000"
    robot_id = "test_robot_safety"
    
    print("🧪 Testing Server-Side Safety Blocking")
    print("=" * 50)
    
    # Test 1: Try to send movement command WITHOUT safety confirmation
    print("\n1️⃣ Testing movement command WITHOUT safety confirmation...")
    
    unsafe_command = {
        'command_type': 'move',
        'parameters': {
            'direction': 'forward',
            'speed': 0.3,
            'duration': 2.0
        }
        # NOTE: Missing 'safety_confirmed' and 'gui_movement_enabled'
    }
    
    try:
        response = requests.post(
            f"{server_url}/robots/{robot_id}/commands",
            json=unsafe_command,
            timeout=5
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Body: {response.json()}")
        
        if response.status_code == 403:
            print("✅ SERVER SAFETY WORKING: Movement command properly blocked!")
        else:
            print("❌ SERVER SAFETY FAILED: Movement command was not blocked!")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    # Test 2: Try to send movement command WITH safety confirmation but movement disabled
    print("\n2️⃣ Testing movement command with safety confirmation but movement disabled...")
    
    disabled_command = {
        'command_type': 'move',
        'parameters': {
            'direction': 'forward',
            'speed': 0.3,
            'duration': 2.0
        },
        'safety_confirmed': True,
        'gui_movement_enabled': False  # Movement disabled
    }
    
    try:
        response = requests.post(
            f"{server_url}/robots/{robot_id}/commands",
            json=disabled_command,
            timeout=5
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Body: {response.json()}")
        
        if response.status_code == 403:
            print("✅ SERVER SAFETY WORKING: Disabled movement properly blocked!")
        else:
            print("❌ SERVER SAFETY FAILED: Disabled movement was not blocked!")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    # Test 3: Try to send movement command WITH full safety confirmation
    print("\n3️⃣ Testing movement command with FULL safety confirmation...")
    
    safe_command = {
        'command_type': 'move',
        'parameters': {
            'direction': 'forward',
            'speed': 0.3,
            'duration': 2.0
        },
        'safety_confirmed': True,
        'gui_movement_enabled': True  # Movement enabled
    }
    
    try:
        response = requests.post(
            f"{server_url}/robots/{robot_id}/commands",
            json=safe_command,
            timeout=5
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Body: {response.json()}")
        
        if response.status_code == 200:
            print("✅ SERVER ALLOWS SAFE COMMANDS: Properly authorized movement allowed!")
        elif response.status_code == 404:
            print("ℹ️  Robot not found (expected if robot not registered)")
        else:
            print("⚠️  Unexpected response - check server logs")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    # Test 4: Try non-movement command (should always work)
    print("\n4️⃣ Testing non-movement command (should always work)...")
    
    non_movement_command = {
        'command_type': 'status_update',
        'parameters': {
            'battery_level': 85.0
        }
    }
    
    try:
        response = requests.post(
            f"{server_url}/robots/{robot_id}/commands",
            json=non_movement_command,
            timeout=5
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Body: {response.json()}")
        
        if response.status_code in [200, 404]:  # 404 if robot not registered
            print("✅ NON-MOVEMENT COMMANDS: Working as expected!")
        else:
            print("⚠️  Unexpected response for non-movement command")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("• Movement commands WITHOUT safety confirmation should be BLOCKED (403)")
    print("• Movement commands with safety_confirmed=True but gui_movement_enabled=False should be BLOCKED (403)")
    print("• Movement commands with FULL safety confirmation should be ALLOWED (200) or robot not found (404)")
    print("• Non-movement commands should always work (200) or robot not found (404)")

if __name__ == "__main__":
    test_server_safety_blocking()