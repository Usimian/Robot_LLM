#!/usr/bin/env python3
"""
Comprehensive VILA Server Debugging Tool
"""

import os
import sys
import torch
import logging
from PIL import Image
import numpy as np

# Add VILA to path
sys.path.append('/home/marc/Robot_LLM/VILA')

def check_system_resources():
    """Check system resources and GPU availability"""
    print("🔍 SYSTEM RESOURCE CHECK")
    print("=" * 50)
    
    # GPU Check
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            memory_gb = gpu.total_memory / 1024**3
            print(f"GPU {i}: {gpu.name} ({memory_gb:.1f}GB)")
            
        # Memory usage
        current_memory = torch.cuda.memory_allocated() / 1024**3
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Current GPU Memory: {current_memory:.2f}GB")
        print(f"Max GPU Memory Used: {max_memory:.2f}GB")
    
    # CPU and RAM
    import psutil
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_percent}%")
    print(f"RAM Usage: {memory.percent}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
    print()

def test_vila_model_loading():
    """Test loading VILA model directly"""
    print("🚀 VILA MODEL LOADING TEST")
    print("=" * 50)
    
    try:
        # Import VILA components
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.constants import DEFAULT_IMAGE_TOKEN
        
        model_path = "Efficient-Large-Model/VILA1.5-3b"
        print(f"Loading model: {model_path}")
        
        # Load model
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            load_8bit=False,
            load_4bit=False,
            device_map="auto",
            device="cuda"
        )
        
        print("✅ Model loaded successfully!")
        print(f"Context length: {context_len}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        return tokenizer, model, image_processor, context_len
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None

def test_simple_generation(tokenizer, model, image_processor):
    """Test simple text generation without images"""
    print("\n📝 SIMPLE TEXT GENERATION TEST")
    print("=" * 50)
    
    if not model:
        print("❌ No model available for testing")
        return
    
    try:
        # Simple prompt
        prompt = "What is a robot?"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"Input tokens: {inputs.input_ids.shape}")
        
        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                inputs.input_ids,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=100,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        new_text = outputs[len(prompt):].strip()
        
        print(f"Prompt: {prompt}")
        print(f"Generated: {new_text}")
        print(f"Length: {len(new_text)} characters")
        
        if len(new_text) > 10:
            print("✅ Text generation working!")
            return True
        else:
            print("⚠️ Generated text is very short")
            return False
            
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a 640x480 image with a clear path
    image = Image.new('RGB', (640, 480), color='lightblue')
    
    # Draw a simple scene
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    
    # Draw floor
    draw.rectangle([0, 300, 640, 480], fill='gray')
    
    # Draw walls on sides
    draw.rectangle([0, 0, 100, 300], fill='darkgray')  # Left wall
    draw.rectangle([540, 0, 640, 300], fill='darkgray')  # Right wall
    
    # Draw clear path
    draw.rectangle([200, 200, 440, 400], fill='lightgray')
    
    # Add text
    try:
        font = ImageFont.load_default()
        draw.text((250, 100), "CLEAR PATH", fill='black', font=font)
    except:
        draw.text((250, 100), "CLEAR PATH", fill='black')
    
    return image

def test_image_generation(tokenizer, model, image_processor):
    """Test image + text generation"""
    print("\n🖼️ IMAGE + TEXT GENERATION TEST")
    print("=" * 50)
    
    if not model:
        print("❌ No model available for testing")
        return False
    
    try:
        # Import conversation utilities
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import DEFAULT_IMAGE_TOKEN
        
        # Create test image
        image = create_test_image()
        print(f"Created test image: {image.size}")
        
        # Process image
        images = [image]
        image_sizes = [image.size]
        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        
        print(f"Image tensor shape: {images_tensor.shape}")
        
        # Create conversation
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        
        # Simple navigation prompt
        prompt = "Look at this image. What do you see? Can the robot move forward safely?"
        
        # Add image token to prompt
        if model.config.mm_use_im_start_end:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + prompt
        
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        print(f"Full prompt length: {len(prompt_text)} chars")
        print(f"Prompt preview: {prompt_text[:200]}...")
        
        # Tokenize (VILA handles image token index automatically)
        input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)
        
        print(f"Input IDs shape: {input_ids.shape}")
        
        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=150,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        print("Generation completed")
        
        # Decode
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(f"Raw output length: {len(outputs)} chars")
        print(f"Raw output: {outputs}")
        
        # Extract new content
        outputs = outputs.strip()
        if conv.sep in outputs:
            parts = outputs.split(conv.sep)
            print(f"Split into {len(parts)} parts by separator '{conv.sep}'")
            for i, part in enumerate(parts):
                print(f"Part {i}: '{part[:100]}...'")
            new_response = parts[-1].strip()
        else:
            # Try to extract after the assistant token
            assistant_start = outputs.find("ASSISTANT:")
            if assistant_start != -1:
                new_response = outputs[assistant_start + 10:].strip()
            else:
                new_response = outputs
        
        print(f"\n🎯 FINAL RESPONSE:")
        print(f"'{new_response}'")
        print(f"Length: {len(new_response)} characters")
        
        if len(new_response) > 20:
            print("✅ Image generation working!")
            return True
        else:
            print("⚠️ Generated response is very short")
            return False
            
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_navigation_specific_prompt(tokenizer, model, image_processor):
    """Test with navigation-specific prompts"""
    print("\n🧭 NAVIGATION-SPECIFIC PROMPT TEST")
    print("=" * 50)
    
    if not model:
        print("❌ No model available for testing")
        return False
    
    try:
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import DEFAULT_IMAGE_TOKEN
        
        # Create test image
        image = create_test_image()
        
        # Process image
        images = [image]
        image_sizes = [image.size]
        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        
        # Navigation-specific prompts to test
        test_prompts = [
            "What should a robot do in this scene? Answer with: move_forward, turn_left, turn_right, or stop.",
            "Can the robot safely move forward in this image?",
            "Describe what you see and suggest a navigation action.",
        ]
        
        for i, nav_prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}: {nav_prompt[:50]}... ---")
            
            # Create conversation
            conv = conv_templates["llava_v1"].copy()
            
            # Add image token
            if model.config.mm_use_im_start_end:
                full_prompt = DEFAULT_IMAGE_TOKEN + '\n' + nav_prompt
            else:
                full_prompt = DEFAULT_IMAGE_TOKEN + nav_prompt
            
            conv.append_message(conv.roles[0], full_prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            # Tokenize and generate
            input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True,
                    temperature=0.1,  # Lower temperature for more consistent responses
                    max_new_tokens=100,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # Extract response
            if conv.sep in outputs:
                response = outputs.split(conv.sep)[-1].strip()
            else:
                assistant_start = outputs.find("ASSISTANT:")
                if assistant_start != -1:
                    response = outputs[assistant_start + 10:].strip()
                else:
                    response = outputs
            
            print(f"Response: '{response}'")
            print(f"Length: {len(response)} characters")
            
            # Check for navigation keywords
            response_lower = response.lower()
            nav_keywords = ['forward', 'left', 'right', 'stop', 'move', 'turn']
            found_keywords = [kw for kw in nav_keywords if kw in response_lower]
            print(f"Navigation keywords found: {found_keywords}")
            
    except Exception as e:
        print(f"❌ Navigation prompt test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Main debugging function"""
    print("🔧 VILA SERVER COMPREHENSIVE DEBUG")
    print("=" * 60)
    
    # Step 1: Check system resources
    check_system_resources()
    
    # Step 2: Test model loading
    tokenizer, model, image_processor, context_len = test_vila_model_loading()
    
    if not model:
        print("\n❌ Cannot proceed without model. Check model loading errors above.")
        return
    
    # Step 3: Test simple text generation
    text_works = test_simple_generation(tokenizer, model, image_processor)
    
    # Step 4: Test image generation
    image_works = test_image_generation(tokenizer, model, image_processor)
    
    # Step 5: Test navigation prompts
    test_navigation_specific_prompt(tokenizer, model, image_processor)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 DEBUG SUMMARY")
    print("=" * 60)
    print(f"✅ Model Loading: {'SUCCESS' if model else 'FAILED'}")
    print(f"✅ Text Generation: {'SUCCESS' if text_works else 'FAILED'}")
    print(f"✅ Image Generation: {'SUCCESS' if image_works else 'FAILED'}")
    
    if model and text_works and image_works:
        print("\n🎉 VILA model appears to be working!")
        print("The issue may be in the ROS2 integration or prompt formatting.")
    else:
        print("\n❌ VILA model has fundamental issues that need to be resolved.")

if __name__ == '__main__':
    main()
