#!/usr/bin/env python3
"""
Simple VILA debugging using our existing model wrapper
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('VILADebug')

# Add our module to path
sys.path.append('/home/marc/Robot_LLM/src/robot_vila_system')

def test_vila_model():
    """Test our VILA model wrapper"""
    print("🔧 VILA MODEL WRAPPER TEST")
    print("=" * 50)
    
    try:
        from robot_vila_system.vila_model import CosmosNemotronVLAModel, SensorData
        from PIL import Image
        import numpy as np
        
        print("✅ Successfully imported VILA model wrapper")
        
        # Initialize model
        print("🚀 Initializing VILA model...")
        vila_model = CosmosNemotronVLAModel(quantization=False)
        
        print("🔄 Loading VILA model...")
        success = vila_model.load_model()
        
        if not success:
            print("❌ Model loading failed")
            return False
        
        print("✅ Model loaded successfully!")
        
        # Create test image
        print("\n🖼️ Creating test image...")
        image_array = np.zeros((480, 640, 3), dtype=np.uint8)
        # Simple scene - clear path
        image_array[300:480, :] = [100, 100, 100]  # Floor
        image_array[200:400, 200:440] = [150, 150, 150]  # Clear path
        image = Image.fromarray(image_array)
        
        # Create sensor data
        print("📡 Creating sensor data...")
        sensor_data = SensorData(
            lidar_distances={
                'distance_front': 2.0,
                'distance_left': 2.0, 
                'distance_right': 2.0
            },
            imu_data={
                'acceleration': {'x': 0, 'y': 0, 'z': 9.8},
                'gyroscope': {'x': 0, 'y': 0, 'z': 0}
            },
            camera_image=image,
            timestamp=0.0
        )
        
        # Test simple analysis
        print("\n🧭 Testing navigation analysis...")
        prompt = "What should the robot do? The path looks clear ahead."
        
        result = vila_model.analyze_multi_modal_scene(sensor_data, prompt)
        
        print(f"\n📊 ANALYSIS RESULT:")
        print(f"Success: {result['success']}")
        print(f"Analysis: '{result['analysis']}'")
        print(f"Navigation Command: {result['navigation_command']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reasoning: {result['reasoning'][:200]}...")
        
        if result['success'] and len(result['analysis']) > 10:
            print("\n🎉 VILA model is working correctly!")
            print(f"Generated {len(result['analysis'])} characters of analysis")
            return True
        else:
            print("\n❌ VILA model generated short or failed response")
            return False
            
    except Exception as e:
        print(f"❌ Error testing VILA model: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_simple_generation():
    """Test the simplest possible generation"""
    print("\n🔤 SIMPLE TEXT GENERATION TEST")
    print("=" * 50)
    
    try:
        from robot_vila_system.vila_model import CosmosNemotronVLAModel
        from PIL import Image
        import numpy as np
        
        vila_model = CosmosNemotronVLAModel(quantization=False)
        
        if not vila_model.load_model():
            print("❌ Model loading failed")
            return False
        
        # Create minimal image
        image = Image.new('RGB', (64, 64), color='white')
        
        # Test with minimal prompt
        response = vila_model.analyze_image(image, "What do you see?")
        
        print(f"Response: '{response}'")
        print(f"Length: {len(response)} characters")
        
        if len(response) > 5:
            print("✅ Basic generation working")
            return True
        else:
            print("❌ Generation too short")
            return False
            
    except Exception as e:
        print(f"❌ Simple generation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Main debug function"""
    print("🔍 VILA SIMPLE DEBUG")
    print("=" * 60)
    
    # Test 1: Full model wrapper
    full_test = test_vila_model()
    
    # Test 2: Simple generation
    simple_test = test_simple_generation()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 DEBUG SUMMARY")
    print("=" * 60)
    print(f"Full Model Test: {'✅ PASS' if full_test else '❌ FAIL'}")
    print(f"Simple Generation: {'✅ PASS' if simple_test else '❌ FAIL'}")
    
    if full_test or simple_test:
        print("\n🎉 VILA model has basic functionality!")
        print("The issue may be in specific prompts or ROS integration.")
    else:
        print("\n❌ VILA model has fundamental issues.")
        print("Check model loading, GPU memory, or dependencies.")

if __name__ == '__main__':
    main()

