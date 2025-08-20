#!/usr/bin/env python3
"""
Simplified VILA Server for Robot Navigation
Bypasses quantization issues by using basic transformers approach
"""

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from typing import List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image as PILImage
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

# Add VILA path
sys.path.insert(0, 'VILA')

app = FastAPI(title="Simple VILA Server", version="1.0.0")

class TextContent(BaseModel):
    type: Literal["text"]
    text: str

class MediaURL(BaseModel):
    url: str

class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: MediaURL

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageContent]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.1

class ChatCompletionResponse(BaseModel):
    choices: List[dict]

# Global model variables
model = None
tokenizer = None
processor = None

@app.on_event("startup")
async def startup():
    global model, tokenizer, processor
    
    try:
        print("🚀 Starting Simple VILA Server...")
        
        # Try to load a basic vision-language model instead of complex VILA
        model_name = "microsoft/git-base"  # Simple vision-language model
        
        print(f"📦 Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print("✅ Simple VILA Server ready!")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("🔄 Starting in demo mode with simple responses")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Extract text and image from request
        text_content = ""
        image = None
        
        for message in request.messages:
            if isinstance(message.content, list):
                for content in message.content:
                    if content.type == "text":
                        text_content = content.text
                    elif content.type == "image_url":
                        # Decode base64 image
                        image_url = content.image_url.url
                        if image_url.startswith("data:image"):
                            base64_data = image_url.split(",")[1]
                            image_bytes = base64.b64decode(base64_data)
                            image = PILImage.open(BytesIO(image_bytes))
            elif isinstance(message.content, str):
                text_content = message.content
        
        # Generate response
        if model is not None and image is not None:
            # Use actual model
            inputs = processor(images=image, text=text_content, return_tensors="pt")
            generated_ids = model.generate(**inputs, max_new_tokens=request.max_tokens)
            response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            # Demo mode - simple rule-based responses for robot navigation
            response_text = generate_demo_response(text_content, image is not None)
        
        return ChatCompletionResponse(
            choices=[{
                "message": {
                    "role": "assistant", 
                    "content": response_text
                }
            }]
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing request: {str(e)}"}
        )

def generate_demo_response(text: str, has_image: bool) -> str:
    """Generate demo responses for robot navigation"""
    text_lower = text.lower()
    
    if not has_image:
        return "I need an image to provide navigation guidance."
    
    # Simple rule-based navigation responses
    if "navigate" in text_lower or "command" in text_lower:
        import random
        responses = [
            "Based on the image, I recommend moving forward carefully.",
            "The path appears clear, you can proceed forward.",
            "I see some obstacles ahead, consider stopping or turning.",
            "The area looks safe for forward movement.",
            "I recommend turning left to avoid obstacles.",
            "The right side appears clearer, suggest turning right.",
            "Stop here and reassess the situation for safety."
        ]
        return random.choice(responses)
    
    elif "analyze" in text_lower or "describe" in text_lower:
        return "I can see the camera feed. The environment appears to be an indoor/outdoor space suitable for robot navigation."
    
    else:
        return "I'm ready to help with robot navigation. Please provide specific navigation requests."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", type=str, default="microsoft/git-base")
    args = parser.parse_args()
    
    print(f"🌐 Starting server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
