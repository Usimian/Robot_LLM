#!/usr/bin/env python3
"""
VILA Tkinter GUI Application
A webcam-based chatbot using VILA 3B vision-language model
No conda required - uses native VILA installation
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import os
import sys
import threading
import time
import io
import base64
import warnings
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import sounddevice as sd

# Suppress warnings
warnings.filterwarnings("ignore")

# Add VILA paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VILA'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PS3'))

# VILA Model Manager
class VILAModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.loading = False
        
    def load_model(self, progress_callback=None):
        """Load VILA model with progress updates"""
        if self.model_loaded or self.loading:
            return True
            
        self.loading = True
        
        try:
            if progress_callback:
                progress_callback("🚀 Loading VILA components...")
            
            # Import VILA native components
            from llava.model.builder import load_pretrained_model  # type: ignore
            from llava.mm_utils import get_model_name_from_path  # type: ignore
            
            if progress_callback:
                progress_callback("📦 Loading VILA 3B model...")
            
            model_path = "Efficient-Large-Model/VILA1.5-3b"
            
            # Handle chat template by patching transformers
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            
            # Store original method
            original_apply_chat_template = PreTrainedTokenizerBase.apply_chat_template
            
            def safe_apply_chat_template(self, conversation, **kwargs):
                """Safe version that handles missing chat templates"""
                try:
                    # Set a basic chat template if missing
                    if not hasattr(self, 'chat_template') or self.chat_template is None:
                        self.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
                    
                    return original_apply_chat_template(self, conversation, **kwargs)
                except Exception as e:
                    # If chat template fails, return a simple fallback
                    if 'add_generation_prompt' in kwargs and kwargs['add_generation_prompt']:
                        return "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
                    else:
                        return "<|im_start|>user\nHello<|im_end|>\n"
            
            # Temporarily replace the method
            PreTrainedTokenizerBase.apply_chat_template = safe_apply_chat_template
            
            if progress_callback:
                progress_callback("🔧 Loading with VILA native loader...")
            
            # Load the model using VILA's native loader
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path),
                load_8bit=False,
                load_4bit=False,  # Disable quantization for stability
                device_map="auto",
                device=self.device,
                torch_dtype=torch.float16
            )
            
            # Restore the original method
            PreTrainedTokenizerBase.apply_chat_template = original_apply_chat_template
            
            if progress_callback:
                progress_callback("✅ VILA loaded successfully!")
                progress_callback(f"🔥 Device: {next(self.model.parameters()).device}")
                progress_callback(f"💾 GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                progress_callback(f"📏 Context Length: {self.context_len}")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error loading VILA: {e}")
            print(f"VILA loading error: {e}")
            return False
        finally:
            self.loading = False
    
    def generate_response(self, prompt, image=None, progress_callback=None):
        """Generate response using VILA"""
        if not self.model_loaded:
            return "❌ VILA model not loaded"
        
        try:
            if progress_callback:
                progress_callback("🤖 VILA is thinking...")
            
            # Import required VILA components
            from llava.conversation import conv_templates  # type: ignore
            from llava.constants import DEFAULT_IMAGE_TOKEN  # type: ignore
            from llava.mm_utils import tokenizer_image_token, process_images  # type: ignore
            
            # Get IMAGE_TOKEN_INDEX from tokenizer (it's not a constant)
            IMAGE_TOKEN_INDEX = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            
            # Define image start/end tokens (not in constants)
            DEFAULT_IM_START_TOKEN = "<im_start>"
            DEFAULT_IM_END_TOKEN = "<im_end>"
            
            # Process image if provided
            image_tensor = None
            if image is not None:
                # Convert PIL image for VILA processing
                if self.image_processor is None:
                    raise ValueError("Image processor is not available")
                if self.model.config is None:
                    raise ValueError("Model config is not available")
                
                # Fix common "reversed" image issue by flipping horizontally
                # This corrects mirror-like effects that some vision models exhibit
                corrected_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    
                image_tensor = process_images([corrected_image], self.image_processor, self.model.config)
                
                if image_tensor is None:
                    raise ValueError("Image processing returned None")
                
                # Image processing successful
                if type(image_tensor) is list:
                    if len(image_tensor) == 0:
                        raise ValueError("Image processing returned empty list")
                    # VILA handles batching - remove our batch dimension if present
                    processed_tensors = []
                    for img in image_tensor:
                        if img.dim() == 4 and img.shape[0] == 1:  # (1, C, H, W) -> remove batch dim
                            img = img.squeeze(0)  # -> (C, H, W)
                        processed_tensors.append(img.to(self.model.device, dtype=torch.float16))
                    image_tensor = processed_tensors
                else:
                    # Handle single tensor - remove batch dim if present
                    if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:  # (1, C, H, W) -> remove batch dim
                        image_tensor = image_tensor.squeeze(0)  # -> (C, H, W)
                    image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
                
                # Image tensor processed successfully
            
            # Prepare conversation
            conv_mode = "vicuna_v1"
            if conv_mode not in conv_templates:
                # Fallback to a simple conversation format
                prompt_text = f"USER: {prompt}\nASSISTANT:"
            else:
                conv = conv_templates[conv_mode].copy()
                
                # Create prompt with image token if image is provided
                if image is not None:
                    if hasattr(self.model.config, 'mm_use_im_start_end') and self.model.config.mm_use_im_start_end:
                        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
                    else:
                        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                
                # Check if conv and its methods exist
                if hasattr(conv, 'roles') and conv.roles and hasattr(conv, 'append_message'):
                    if len(conv.roles) < 2:
                        # Fallback if not enough roles
                        prompt_text = f"USER: {prompt}\nASSISTANT:"
                    else:
                        conv.append_message(conv.roles[0], prompt)
                        conv.append_message(conv.roles[1], None)
                        try:
                            prompt_text = conv.get_prompt()
                        except Exception:
                            # Fallback if get_prompt fails
                            prompt_text = f"USER: {prompt}\nASSISTANT:"
                else:
                    # Fallback to simple format
                    prompt_text = f"USER: {prompt}\nASSISTANT:"
            
            # Tokenize - use different approach for text vs image
            if image is not None:
                # Use VILA's image tokenization
                try:
                    input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX)
                    if not torch.is_tensor(input_ids):
                        input_ids = torch.tensor(input_ids, dtype=torch.long)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    input_ids = input_ids.to(self.model.device)
                except Exception as e:
                    # Fallback to standard tokenization
                    input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.model.device)
            else:
                # Use standard tokenization for text-only
                input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.model.device)
            
            # Generate response
            with torch.inference_mode():
                generate_kwargs = {
                    'input_ids': input_ids,
                    'do_sample': True,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_new_tokens': 512,
                    'use_cache': True,
                    'pad_token_id': self.tokenizer.eos_token_id
                }
                
                # Always provide media and media_config parameters (VILA expects both)
                if image_tensor is not None:
                    # VILA expects media as a dictionary with image list
                    generate_kwargs['media'] = {'image': [image_tensor] if not isinstance(image_tensor, list) else image_tensor}
                    # Also provide media_config for the image
                    generate_kwargs['media_config'] = {'image': {}}
                else:
                    # For text-only, provide empty dictionaries for both
                    generate_kwargs['media'] = {}
                    generate_kwargs['media_config'] = {}
                
                output_ids = self.model.generate(**generate_kwargs)
            
            # Decode response - handle None case
            if output_ids is None:
                return "I'm having trouble generating a response. Please try again."
            
            try:
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                if not outputs:
                    return "I generated an empty response. Please try again."
                outputs = outputs[0]
            except Exception as e:
                return f"Error decoding response: {e}"
            
            # Extract assistant's response
            try:
                if 'conv' in locals() and hasattr(conv, 'sep2') and conv.sep2 is not None:
                    response = outputs.split(conv.sep2)[-1].strip()
                elif 'conv' in locals() and hasattr(conv, 'sep') and conv.sep is not None:
                    response = outputs.split(conv.sep)[-1].strip()
                else:
                    # Simple extraction for fallback format
                    if "ASSISTANT:" in outputs:
                        response = outputs.split("ASSISTANT:")[-1].strip()
                    else:
                        response = outputs.strip()
            except Exception:
                # Last resort: just use the raw output
                response = outputs.strip()
            
            return response if response else "I can see your request but I'm having trouble generating a response. Please try again."
            
        except Exception as e:
            error_msg = f"❌ VILA generation error: {e}"
            print(error_msg)
            
            # CRITICAL: Check if error indicates model failure
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['cuda', 'memory', 'model', 'tensor', 'device']):
                print(f"🚨 VILA model may have failed - resetting model_loaded flag")
                self.model_loaded = False
            
            return error_msg

# Add Silero TTS model loading (same as original)
def load_silero_tts():
    try:
        model = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='en',
            speaker='v3_en'
        )
        return model
    except Exception as e:
        print(f"Warning: Failed to load TTS model: {e}")
        return None

class VILAApp:
    def __init__(self, root):
        self.root = root
        
        # Configure root window
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize VILA model
        self.vila_model = VILAModel()
        
        self.root.title("VILA 3B - Vision Language Model")

        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Add sizegrip
        self.sizegrip = ttk.Sizegrip(self.root)
        self.sizegrip.grid(row=0, column=0, sticky="se")
        
        # Configure main frame grid
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        for i in range(8):
            self.main_frame.rowconfigure(i, weight=0)
        self.main_frame.rowconfigure(8, weight=1)
        
        # --- Model Status ---
        self.status_label = ttk.Label(self.main_frame, text="🔄 VILA Model: Not Loaded", font=("TkDefaultFont", 10, "bold"))
        self.status_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        # Load Model Button
        self.load_model_button = ttk.Button(self.main_frame, text="Load VILA Model", command=self.load_vila_model)
        self.load_model_button.grid(row=0, column=1, sticky="e", pady=(0, 10))
        
        # --- Webcam and Image section ---
        # Titles
        self.webcam_title = ttk.Label(self.main_frame, text="Webcam Preview", font=("TkDefaultFont", 10, "bold"))
        self.webcam_title.grid(row=1, column=0, padx=(0, 10), pady=(0, 2), sticky="s")
        self.image_title = ttk.Label(self.main_frame, text="Image Being Used", font=("TkDefaultFont", 10, "bold"))
        self.image_title.grid(row=1, column=1, padx=(10, 0), pady=(0, 2), sticky="s")

        # Webcam preview (left)
        self.webcam_preview_label = ttk.Label(self.main_frame, text="Webcam preview loading...")
        self.webcam_preview_label.grid(row=2, column=0, padx=(0, 10), pady=(0, 10), sticky="n")

        # Image preview (right)
        self.image_preview = ttk.Label(self.main_frame, text="No image selected")
        self.image_preview.grid(row=2, column=1, padx=(10, 0), pady=(0, 10), sticky="n")

        # --- Image controls ---
        self.image_group = ttk.Frame(self.main_frame)
        self.image_group.grid(row=3, column=1, sticky="s", padx=(10, 0), pady=(0, 10))
        self.image_group.columnconfigure(0, weight=1)
        self.image_group.columnconfigure(1, weight=1)
        self.image_group.columnconfigure(2, weight=0)
        
        self.image_button = ttk.Button(self.image_group, text="Open Image...", command=self.select_image)
        self.image_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.webcam_button = ttk.Button(self.image_group, text="Take Photo", command=self.capture_webcam_image)
        self.webcam_button.grid(row=0, column=1, padx=(5, 5), sticky="ew")
        self.rotate_button = ttk.Button(self.image_group, text="↻", command=self.rotate_image, width=3)
        self.rotate_button.grid(row=0, column=2, padx=(5, 0), sticky="e")
        
        self.clear_image_button = ttk.Button(self.image_group, text="Clear Image", command=self.clear_image)
        self.clear_image_button.grid(row=1, column=0, columnspan=3, pady=(5, 0), sticky="ew")
        
        # --- Prompt input ---
        self.label = ttk.Label(self.main_frame, text="Enter your prompt:", font=("TkDefaultFont", 10, "bold"))
        self.label.grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        
        self.text_input = ttk.Entry(self.main_frame, width=50)
        self.text_input.insert(0, "Describe this image in detail")
        self.text_input.grid(row=5, column=0, sticky="we", pady=(0, 10))
        self.text_input.bind("<Return>", lambda event: self.process_query())
        
        # Submit button
        self.submit_button = ttk.Button(self.main_frame, text="Ask VILA", command=self.process_query)
        self.submit_button.grid(row=5, column=1, padx=(5, 0), pady=(0, 10), sticky=tk.E)
        self.submit_button.config(state="disabled")  # Disabled until model loads
        
        # Output label and speak button
        self.output_label = ttk.Label(self.main_frame, text="VILA Response:")
        self.output_label.grid(row=6, column=0, sticky=tk.W, pady=(10, 5))
        
        self.speak_button = ttk.Button(self.main_frame, text="🔊 Speak", command=self.speak_response)
        self.speak_button.grid(row=6, column=1, sticky=tk.E, padx=(5, 0), pady=(10, 5))
        
        # Output display
        self.text_output = scrolledtext.ScrolledText(self.main_frame, width=60, height=15, wrap=tk.WORD)
        self.text_output.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(0, 10))

        # Variables for image handling
        self.image_path = None
        self.current_image = None  # PIL Image
        self.photo = None  # PhotoImage reference
        self.text_input.focus_set()

        # Webcam setup
        self.webcam_cap = None
        self.webcam_preview_running = True
        self.webcam_preview_imgtk = None
        self.start_webcam_preview()

        # TTS setup (same as original)
        self.tts_model = None
        self.tts_sample_rate = 48000
        self.tts_speaker = 'en_0'
        self.tts_loading = False
        
        # Welcome message
        welcome_msg = """🎉 Welcome to VILA 3B Vision-Language Model!

✅ Native Python installation (no conda required)
🔥 GPU acceleration on your RTX 3090
🖼️ Advanced vision understanding capabilities
💬 Natural language conversation

Click "Load VILA Model" to get started!
"""
        self.text_output.insert(tk.END, welcome_msg)

    def load_vila_model(self):
        """Load VILA model in background thread"""
        if self.vila_model.model_loaded or self.vila_model.loading:
            return
        
        # Disable button during loading
        self.load_model_button.config(state="disabled", text="Loading...")
        self.status_label.config(text="🔄 Loading VILA Model...")
        
        def progress_callback(message):
            """Update UI with loading progress"""
            self.root.after(0, lambda: self.text_output.insert(tk.END, f"{message}\n"))
            self.root.after(0, lambda: self.text_output.see(tk.END))
        
        def load_model_thread():
            """Load model in background"""
            success = self.vila_model.load_model(progress_callback)
            
            def update_ui():
                if success:
                    self.status_label.config(text="✅ VILA Model: Ready")
                    self.load_model_button.config(text="Model Loaded", state="disabled")
                    self.submit_button.config(state="normal")
                    self.text_output.insert(tk.END, "\n🚀 VILA is ready! You can now ask questions about images.\n\n")
                else:
                    self.status_label.config(text="❌ VILA Model: Failed to Load")
                    self.load_model_button.config(text="Retry Load", state="normal")
                    self.submit_button.config(state="disabled")
                
                self.text_output.see(tk.END)
            
            self.root.after(0, update_ui)
        
        # Start loading in background thread
        threading.Thread(target=load_model_thread, daemon=True).start()

    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Open image...",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.PNG *.JPG *.JPEG *.GIF *.BMP")]
        )
        
        if file_path:
            self.image_path = file_path
            try:
                img = Image.open(file_path)
                img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                
                self.current_image = img.copy()
                
                img_thumbnail = img.resize((150, 150), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(img_thumbnail)
                self.image_preview.config(image=self.photo, text="")
                    
            except Exception as e:
                self.text_output.insert(tk.END, f"Error loading image: {str(e)}\n")
                self.image_preview.config(text="Error loading image")
                self.current_image = None
                self.photo = None
        else:
            self.current_image = None
            self.image_preview.config(image='', text="No image selected")
            self.photo = None

    def capture_webcam_image(self):
        """Capture image from webcam"""
        try:
            if self.webcam_cap is None:
                self.text_output.insert(tk.END, "Webcam is not running.\n")
                return
            
            ret, frame = self.webcam_cap.read()
            if not ret:
                self.text_output.insert(tk.END, "Failed to capture image from webcam.\n")
                return
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            self.current_image = img.copy()
            
            img_thumbnail = img.resize((150, 150), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_thumbnail)
            self.image_preview.config(image=self.photo, text="")
            self.image_path = None  # No file path for webcam image
            
        except Exception as e:
            self.text_output.insert(tk.END, f"Error capturing webcam image: {str(e)}\n")
            self.image_preview.config(text="Error capturing image")
            self.current_image = None
            self.photo = None

    def process_query(self):
        """Process query using VILA"""
        if not self.vila_model.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Please load the VILA model first!")
            return
        
        prompt = self.text_input.get().strip()
        if not prompt:
            if self.current_image:
                prompt = "Describe this image in detail"
            else:
                return
        
        # Disable submit button
        self.submit_button.config(state="disabled")
        
        # Clear output and show thinking
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, "🤖 VILA is analyzing...\n")
        
        def progress_callback(message):
            self.root.after(0, lambda: self.text_output.insert(tk.END, f"{message}\n"))
            self.root.after(0, lambda: self.text_output.see(tk.END))
        
        def generate_thread():
            """Generate response in background"""
            response = self.vila_model.generate_response(
                prompt=prompt, 
                image=self.current_image,
                progress_callback=progress_callback
            )
            
            def update_output():
                self.text_output.delete(1.0, tk.END)
                self.text_output.insert(tk.END, f"🤖 VILA: {response}")
                self.submit_button.config(state="normal")
            
            self.root.after(0, update_output)
        
        threading.Thread(target=generate_thread, daemon=True).start()

    def start_webcam_preview(self):
        """Start webcam preview (same as original)"""
        try:
            if self.webcam_cap is None:
                self.webcam_cap = cv2.VideoCapture(0)
                self.webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
            ret, frame = self.webcam_cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((320, 240), Image.Resampling.LANCZOS)
                self.webcam_preview_imgtk = ImageTk.PhotoImage(img)
                self.webcam_preview_label.config(image=self.webcam_preview_imgtk, text="")
            else:
                self.webcam_preview_label.config(text="Webcam not available")
        except Exception as e:
            self.webcam_preview_label.config(text=f"Webcam error: {e}")
        
        if self.webcam_preview_running:
            self.root.after(100, self.start_webcam_preview)

    def stop_webcam_preview(self):
        """Stop webcam preview"""
        self.webcam_preview_running = False
        if self.webcam_cap is not None:
            self.webcam_cap.release()
            self.webcam_cap = None

    def clear_image(self):
        """Clear current image"""
        self.image_path = None
        self.current_image = None
        self.photo = None
        self.image_preview.config(image='', text="No image selected")

    def rotate_image(self):
        """Rotate current image 90 degrees"""
        if self.current_image is None:
            return
        
        try:
            self.current_image = self.current_image.rotate(-90, expand=True)
            
            img_thumbnail = self.current_image.resize((150, 150), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_thumbnail)
            self.image_preview.config(image=self.photo, text="")
            
        except Exception as e:
            self.text_output.insert(tk.END, f"Error rotating image: {str(e)}\n")

    def _load_tts_model_if_needed(self):
        """Lazy load TTS model (same as original)"""
        if self.tts_model is None and not self.tts_loading:
            self.tts_loading = True
            try:
                self.tts_model = load_silero_tts()
            except Exception as e:
                self.text_output.insert(tk.END, f"\nTTS model loading failed: {e}\n")
            finally:
                self.tts_loading = False

    def speak_response(self):
        """Speak response using TTS (same as original)"""
        text = self.text_output.get(1.0, tk.END).strip()
        if not text or "analyzing" in text.lower():
            return
            
        self._load_tts_model_if_needed()
        
        if self.tts_model is None:
            self.text_output.insert(tk.END, f"\nTTS not available\n")
            return
            
        try:
            self.speak_button.config(state="disabled")
            
            # Remove VILA prefix for cleaner speech
            clean_text = text.replace("🤖 VILA:", "").strip()
            
            audio = self.tts_model.apply_tts(
                text=clean_text,
                speaker=self.tts_speaker,
                sample_rate=self.tts_sample_rate
            )
            
            if hasattr(audio, 'detach'):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = np.asarray(audio)
            
            sd.play(audio_np, self.tts_sample_rate)
            self.root.after(1000, lambda: self.speak_button.config(state="normal"))
            
        except Exception as e:
            self.text_output.insert(tk.END, f"\nTTS error: {e}\n")
            self.speak_button.config(state="normal")

    def cleanup(self):
        """Clean up resources"""
        self.stop_webcam_preview()
        self.photo = None
        self.webcam_preview_imgtk = None
        try:
            sd.stop()
        except:
            pass

def main():
    """Main entry point"""
    root_window = tk.Tk()
    root_window.title("VILA 3B - Vision Language Model")
    root_window.geometry("900x900")
    root_window.resizable(width=True, height=True)
    root_window.minsize(850, 800)
    root_window.update_idletasks()
    
    # Create app
    app = VILAApp(root_window)
    
    # Handle window close
    def on_closing():
        app.cleanup()
        root_window.destroy()
    
    root_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start main loop
    try:
        root_window.mainloop()
    except KeyboardInterrupt:
        print("\nExiting on Ctrl-C")
        app.cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting on Ctrl-C")