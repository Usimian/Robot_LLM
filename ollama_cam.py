import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import requests
import base64
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import sounddevice as sd
import threading
import io
# import time

# Add Silero TTS model loading (initialize once)
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

class App:
    def __init__(self, root):
        self.root = root
        
        # Configure root window to use grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # API configuration
        self.api_url = "http://localhost:11434/api/generate"
        self.api_base = "http://localhost:11434"
        self.model = "Gemma3"
        self.available_models = []
        
        self.root.title(f"{self.model} - Model Test")

        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Add sizegrip to bottom-right corner
        self.sizegrip = ttk.Sizegrip(self.root)
        self.sizegrip.grid(row=0, column=0, sticky="se")
        
        # Configure main frame grid
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        for i in range(7):
            self.main_frame.rowconfigure(i, weight=0)
        self.main_frame.rowconfigure(7, weight=1)
        
        # --- Webcam and Image Used side by side ---
        # Titles
        self.webcam_title = ttk.Label(self.main_frame, text="Webcam Preview", font=("TkDefaultFont", 10, "bold"))
        self.webcam_title.grid(row=0, column=0, padx=(0, 10), pady=(0, 2), sticky="s")
        self.image_title = ttk.Label(self.main_frame, text="Image Being Used", font=("TkDefaultFont", 10, "bold"))
        self.image_title.grid(row=0, column=1, padx=(10, 0), pady=(0, 2), sticky="s")

        # Webcam preview (left)
        self.webcam_preview_label = ttk.Label(self.main_frame, text="Webcam preview loading...")
        self.webcam_preview_label.grid(row=1, column=0, padx=(0, 10), pady=(0, 10), sticky="n")

        # Image preview (right)
        self.image_preview = ttk.Label(self.main_frame, text="No image selected")
        self.image_preview.grid(row=1, column=1, padx=(10, 0), pady=(0, 10), sticky="n")

        # --- Image selection/capture group aligned with bottom of webcam preview ---
        self.image_group = ttk.Frame(self.main_frame)
        self.image_group.grid(row=2, column=1, sticky="s", padx=(10, 0), pady=(0, 10))
        self.image_group.columnconfigure(0, weight=1)
        self.image_group.columnconfigure(1, weight=1)
        self.image_group.columnconfigure(2, weight=0)  # For rotate button
        
        self.image_button = ttk.Button(self.image_group, text="Open Image...", command=self.select_image)
        self.image_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.webcam_button = ttk.Button(self.image_group, text="Take Photo", command=self.capture_webcam_image)
        self.webcam_button.grid(row=0, column=1, padx=(5, 5), sticky="ew")
        # Add small rotate button next to Take Photo
        self.rotate_button = ttk.Button(self.image_group, text="↻", command=self.rotate_image, width=3)
        self.rotate_button.grid(row=0, column=2, padx=(5, 0), sticky="e")
        
        # Add Clear Image button under all buttons
        self.clear_image_button = ttk.Button(self.image_group, text="Clear Image", command=self.clear_image)
        self.clear_image_button.grid(row=1, column=0, columnspan=3, pady=(5, 0), sticky="ew")
        
        # --- Model Selection ---
        self.model_label = ttk.Label(self.main_frame, text="Model:", font=("TkDefaultFont", 10, "bold"))
        self.model_label.grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
        
        self.model_var = tk.StringVar(value=self.model)
        self.model_combo = ttk.Combobox(self.main_frame, textvariable=self.model_var, state="readonly", width=30)
        self.model_combo.grid(row=3, column=1, sticky="ew", padx=(5, 0), pady=(10, 5))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_changed)
        
        # Add a simple label
        self.label = ttk.Label(self.main_frame, text="Enter your prompt:", font=("TkDefaultFont", 10, "bold"))
        self.label.grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        
        # Add text input box
        self.text_input = ttk.Entry(self.main_frame, width=50)
        self.text_input.insert(0, "Describe image")
        self.text_input.grid(row=5, column=0, sticky="we", pady=(0, 10))
        # Bind Enter key to submit
        self.text_input.bind("<Return>", lambda event: self.process_query())
        
        # Add submit button
        self.submit_button = ttk.Button(self.main_frame, text="Submit", command=self.process_query)
        self.submit_button.grid(row=5, column=1, padx=(5, 0), pady=(0, 10), sticky=tk.E)
        
        # Add output display label
        self.output_label = ttk.Label(self.main_frame, text=f"Response [{self.model}]:")
        self.output_label.grid(row=6, column=0, sticky=tk.W, pady=(10, 5))
        
        # Add Speak button (keep fixed)
        self.speak_button = ttk.Button(self.main_frame, text="🔊 Speak", command=self.speak_response)
        self.speak_button.grid(row=6, column=1, sticky=tk.E, padx=(5, 0), pady=(10, 5))
        
        # Add text output display (scrolled text widget)
        self.text_output = scrolledtext.ScrolledText(self.main_frame, width=60, height=15, wrap=tk.WORD)
        self.text_output.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(0, 10))

        # Variables to store the selected image
        self.image_path = None
        self.image_data = None  # Ensure image_data is always defined
        self.photo = None  # Keep reference to prevent garbage collection
        self.current_image = None  # Store PIL Image for rotation
        self.text_input.focus_set()

        # Start webcam preview
        self.webcam_cap = None
        self.webcam_preview_running = True
        self.webcam_preview_imgtk = None  # Keep reference to prevent garbage collection
        self.start_webcam_preview()

        # Load Silero TTS model (lazy loading)
        self.tts_model = None
        self.tts_sample_rate = 48000
        self.tts_speaker = 'en_0'
        self.tts_loading = False
        
        # Load available models from Ollama
        self.load_models()

    def _load_tts_model_if_needed(self):
        """Lazy load TTS model when first needed"""
        if self.tts_model is None and not self.tts_loading:
            self.tts_loading = True
            try:
                self.tts_model = load_silero_tts()
            except Exception as e:
                self.text_output.insert(tk.END, f"\nTTS model loading failed: {e}\n")
            finally:
                self.tts_loading = False

    def load_models(self):
        """Load available models from Ollama server"""
        def fetch_models():
            try:
                response = requests.get(f"{self.api_base}/api/tags", timeout=10)
                if response.status_code == 200:
                    models_data = response.json()
                    models = [model['name'] for model in models_data.get('models', [])]
                    self.root.after(0, self.update_model_list, models)
                else:
                    self.root.after(0, self.update_model_list, [])
            except Exception as e:
                print(f"Failed to fetch models: {e}")
                self.root.after(0, self.update_model_list, [])
        
        # Fetch models in background thread
        threading.Thread(target=fetch_models, daemon=True).start()

    def update_model_list(self, models):
        """Update the model dropdown with available models"""
        self.available_models = models
        if models:
            self.model_combo['values'] = models
            # If current model is not in the list, set to first available
            if self.model not in models and models:
                self.model = models[0]
                self.model_var.set(self.model)
                self.update_window_title()
        else:
            # If no models found, keep current model but show message
            self.model_combo['values'] = [self.model]
            print("No models found from Ollama server, using current model")

    def on_model_changed(self, event=None):
        """Handle model selection change"""
        selected_model = self.model_var.get()
        if selected_model and selected_model != self.model:
            self.model = selected_model
            self.update_window_title()
            self.output_label.config(text=f"Response [{self.model}]:")

    def update_window_title(self):
        """Update window title with current model"""
        self.root.title(f"{self.model} - Model Test")

    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Open image...",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.PNG *.JPG *.JPEG *.GIF *.BMP")]
        )
        
        if file_path:
            self.image_path = file_path
            # Load and display preview
            try:
                img = Image.open(file_path)
                # Limit image size to prevent memory issues
                img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                
                # Store the current image for rotation
                self.current_image = img.copy()
                
                img_thumbnail = img.resize((150, 150), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(img_thumbnail)
                self.image_preview.config(image=self.photo, text="")
                
                # Convert image to base64
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                img_bytes = buffer.getvalue()
                self.image_data = base64.b64encode(img_bytes).decode('utf-8')
                    
            except Exception as e:
                self.text_output.insert(tk.END, f"Error loading image: {str(e)}\n")
                self.image_preview.config(text="Error loading image")
                self.image_data = None
                self.photo = None
                self.current_image = None
        else:
            self.image_data = None
            self.image_preview.config(image='', text="No image selected")
            self.photo = None
            self.current_image = None

    def capture_webcam_image(self):
        """Capture an image from the webcam and use it as the selected image"""
        try:
            if self.webcam_cap is None:
                self.text_output.insert(tk.END, "Webcam is not running.\n")
                return
            ret, frame = self.webcam_cap.read()
            if not ret:
                self.text_output.insert(tk.END, "Failed to capture image from webcam.\n")
                return
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Store the current image for rotation
            self.current_image = img.copy()
            
            img_thumbnail = img.resize((150, 150), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_thumbnail)
            self.image_preview.config(image=self.photo, text="")
            # Encode as JPEG in memory
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_bytes = buffer.getvalue()
            self.image_data = base64.b64encode(img_bytes).decode('utf-8')
            self.image_path = None  # No file path for webcam image
        except Exception as e:
            self.text_output.insert(tk.END, f"Error capturing webcam image: {str(e)}\n")
            self.image_preview.config(text="Error capturing image")
            self.image_data = None
            self.photo = None
            self.current_image = None

    def process_query(self):
        """Process the query with Ollama Llava model"""
        prompt = self.text_input.get()
        if not prompt:
            if self.image_data:
                prompt = "Describe image"
            else:
                # Do nothing if no prompt and no image
                return
        
        # Disable submit button to prevent multiple requests
        self.submit_button.config(state="disabled")
        
        # Show 'Thinking...' immediately
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, "Thinking...\n")
        
        # Start request in separate thread
        threading.Thread(target=self.send_request, args=(prompt,), daemon=True).start()
    
    def send_request(self, prompt):
        """Send request to Ollama API"""
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            # Add image to payload if available and not empty
            if self.image_data and self.image_data.strip():
                payload["images"] = [self.image_data]
            
            # Send the request with timeout
            response = requests.post(self.api_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "No response")
                self.root.after(0, self.update_output, response_text)
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                self.root.after(0, self.update_output, error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "Error: Request timed out"
            self.root.after(0, self.update_output, error_msg)
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Could not connect to Ollama server"
            self.root.after(0, self.update_output, error_msg)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, self.update_output, error_msg)
        finally:
            # Re-enable the submit button
            self.root.after(0, lambda: self.submit_button.config(state="normal"))
    
    def update_output(self, text):
        """Update the output text widget with the response"""
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, text)

    def start_webcam_preview(self):
        try:
            if self.webcam_cap is None:
                self.webcam_cap = cv2.VideoCapture(0)
                # Set lower resolution for better performance
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
            # Reduced frequency from 30ms to 100ms for better performance
            self.root.after(100, self.start_webcam_preview)

    def stop_webcam_preview(self):
        self.webcam_preview_running = False
        if self.webcam_cap is not None:
            self.webcam_cap.release()
            self.webcam_cap = None

    def clear_image(self):
        """Clear the image being used and reset preview and data"""
        self.image_path = None
        self.image_data = None
        self.photo = None
        self.current_image = None
        self.image_preview.config(image='', text="No image selected")

    def rotate_image(self):
        """Rotate the current image by 90 degrees clockwise"""
        if self.current_image is None:
            return
        
        try:
            # Rotate the image 90 degrees clockwise
            self.current_image = self.current_image.rotate(-90, expand=True)
            
            # Update the preview
            img_thumbnail = self.current_image.resize((150, 150), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_thumbnail)
            self.image_preview.config(image=self.photo, text="")
            
            # Update the base64 data
            buffer = io.BytesIO()
            self.current_image.save(buffer, format="JPEG", quality=85)
            img_bytes = buffer.getvalue()
            self.image_data = base64.b64encode(img_bytes).decode('utf-8')
            
        except Exception as e:
            self.text_output.insert(tk.END, f"Error rotating image: {str(e)}\n")

    def speak_response(self):
        """Speak the response text using Silero TTS"""
        text = self.text_output.get(1.0, tk.END).strip()
        if not text or text == "Thinking...":
            return
            
        # Load TTS model if needed
        self._load_tts_model_if_needed()
        
        if self.tts_model is None:
            self.text_output.insert(tk.END, f"\nTTS not available\n")
            return
            
        try:
            # Disable speak button during TTS
            self.speak_button.config(state="disabled")
            
            audio = self.tts_model.apply_tts(  # type: ignore
                text=text,
                speaker=self.tts_speaker,
                sample_rate=self.tts_sample_rate
            )
            # Fix NumPy 2.0+ compatibility issue
            if hasattr(audio, 'detach'):
                # PyTorch tensor - convert properly
                audio_np = audio.detach().cpu().numpy()
            else:
                # Already a numpy array or other format
                audio_np = np.asarray(audio)
            
            sd.play(audio_np, self.tts_sample_rate)
            
            # Re-enable button after a short delay
            self.root.after(1000, lambda: self.speak_button.config(state="normal"))
            
        except Exception as e:
            self.text_output.insert(tk.END, f"\nTTS error: {e}\n")
            self.speak_button.config(state="normal")

    def cleanup(self):
        """Clean up resources before closing"""
        self.stop_webcam_preview()
        # Clear image references
        self.photo = None
        self.webcam_preview_imgtk = None
        # Stop any ongoing audio
        try:
            sd.stop()
        except:
            pass

def main():
    """Main entry point for the application"""
    # Initialize Tk with specific options
    root_window = tk.Tk()
    
    # Set window title
    root_window.title("Model Test")
    
    # Set window geometry
    root_window.geometry("900x800")
    
    # Make window resizable - explicitly set both dimensions
    root_window.resizable(width=True, height=True)
    
    # Set minimum window size
    root_window.minsize(850, 800)
    
    # Force update of window manager hints
    root_window.update_idletasks()
    
    # Create the app
    app = App(root_window)
    
    # Handle window close event
    def on_closing():
        app.cleanup()
        root_window.destroy()
    
    root_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the main loop
    try:
        root_window.mainloop()
    except KeyboardInterrupt:
        print("\nExiting on Ctrl-C (KeyboardInterrupt)")
        app.cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting on Ctrl-C (KeyboardInterrupt)")
