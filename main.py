import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import requests
import base64
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import sounddevice as sd

# Add Silero TTS model loading (initialize once)
def load_silero_tts():
    model, example_text = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='en',
        speaker='v3_en'
    )
    return model

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("LLAVA Model Test")
        
        # Configure root window to use grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # API configuration
        self.api_url = "http://localhost:11434/api/generate"
        self.model = "Gemma3"
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Add sizegrip to bottom-right corner
        self.sizegrip = ttk.Sizegrip(self.root)
        self.sizegrip.grid(row=0, column=0, sticky="se")
        
        # Configure main frame grid
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        for i in range(6):
            self.main_frame.rowconfigure(i, weight=0)
        self.main_frame.rowconfigure(6, weight=1)
        
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
        self.image_button = ttk.Button(self.image_group, text="Open Image...", command=self.select_image)
        self.image_button.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.webcam_button = ttk.Button(self.image_group, text="Take Photo", command=self.capture_webcam_image)
        self.webcam_button.grid(row=0, column=1, padx=(10, 0), sticky="ew")
        # Add Clear Image button under Take Photo
        self.clear_image_button = ttk.Button(self.image_group, text="Clear Image", command=self.clear_image)
        self.clear_image_button.grid(row=1, column=0, columnspan=2, pady=(5, 0), sticky="ew")
        
        # Add a simple label
        self.label = ttk.Label(self.main_frame, text="Enter your prompt:", font=("TkDefaultFont", 10, "bold"))
        self.label.grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        # Add text input box
        self.text_input = ttk.Entry(self.main_frame, width=50)
        self.text_input.insert(0, "Describe image")
        self.text_input.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        # Bind Enter key to submit
        self.text_input.bind("<Return>", lambda event: self.process_query())
        
        # Add submit button
        self.submit_button = ttk.Button(self.main_frame, text="Submit", command=self.process_query)
        self.submit_button.grid(row=4, column=1, padx=(5, 0), pady=(0, 10), sticky=tk.E)
        
        # Add output display label
        self.output_label = ttk.Label(self.main_frame, text="Response:")
        self.output_label.grid(row=5, column=0, sticky=tk.W, pady=(10, 5))
        
        # Add Speak button (keep fixed)
        self.speak_button = ttk.Button(self.main_frame, text="🔊 Speak", command=self.speak_response)
        self.speak_button.grid(row=5, column=1, sticky=tk.E, padx=(5, 0), pady=(10, 5))
        
        # Add text output display (scrolled text widget)
        self.text_output = scrolledtext.ScrolledText(self.main_frame, width=60, height=15, wrap=tk.WORD)
        self.text_output.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(0, 10))

        # Variables to store the selected image
        self.image_path = None
        self.image_data = None  # Ensure image_data is always defined
        self.text_input.focus_set()

        # Start webcam preview
        self.webcam_cap = None
        self.webcam_preview_running = True
        self.start_webcam_preview()

        # Load Silero TTS model
        self.tts_model = load_silero_tts()
        self.tts_sample_rate = 48000
        self.tts_speaker = 'en_0'

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
                img = img.resize((150, 150), Image.LANCZOS)  # Create larger thumbnail
                self.photo = ImageTk.PhotoImage(img)
                self.image_preview.config(image=self.photo, text="")
                
                # Convert image to base64
                with open(file_path, "rb") as img_file:
                    self.image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
            except Exception as e:
                self.text_output.insert(tk.END, f"Error loading image: {str(e)}\n")
                self.image_preview.config(text="Error loading image")
                self.image_data = None
        else:
            self.image_data = None
            self.image_preview.config(image='', text="No image selected")

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
            img_thumbnail = img.resize((150, 150), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_thumbnail)
            self.image_preview.config(image=self.photo, text="")
            # Encode as JPEG in memory
            import io
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()
            self.image_data = base64.b64encode(img_bytes).decode('utf-8')
            self.image_path = None  # No file path for webcam image
        except Exception as e:
            self.text_output.insert(tk.END, f"Error capturing webcam image: {str(e)}\n")
            self.image_preview.config(text="Error capturing image")
            self.image_data = None

    def process_query(self):
        """Process the query with Ollama Llava model"""
        prompt = self.text_input.get()
        if not prompt:
            if self.image_data:
                prompt = "Describe image"
            else:
                # Do nothing if no prompt and no image
                return
        # Show 'Thinking...' immediately
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, "Thinking...\n")
        import threading
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
            if self.image_data:
                if self.image_data.strip() != "":
                    payload["images"] = [self.image_data]
            # If no image, do not include 'images' key (already handled by above)
            
            # Send the request
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                self.root.after(0, self.update_output, result.get("response", "No response"))
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                self.root.after(0, self.update_output, error_msg)
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, self.update_output, error_msg)
        
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
            ret, frame = self.webcam_cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((320, 240), Image.LANCZOS)
                self.webcam_preview_imgtk = ImageTk.PhotoImage(img)
                self.webcam_preview_label.config(image=self.webcam_preview_imgtk, text="")
            else:
                self.webcam_preview_label.config(text="Webcam not available")
        except Exception as e:
            self.webcam_preview_label.config(text=f"Webcam error: {e}")
        if self.webcam_preview_running:
            self.root.after(30, self.start_webcam_preview)

    def stop_webcam_preview(self):
        self.webcam_preview_running = False
        if self.webcam_cap is not None:
            self.webcam_cap.release()
            self.webcam_cap = None

    def clear_image(self):
        """Clear the image being used and reset preview and data"""
        self.image_path = None
        self.image_data = None
        self.image_preview.config(image='', text="No image selected")

    def speak_response(self):
        """Speak the response text using Silero TTS"""
        text = self.text_output.get(1.0, tk.END).strip()
        if not text:
            return
        try:
            audio = self.tts_model.apply_tts(
                text=text,
                speaker=self.tts_speaker,
                sample_rate=self.tts_sample_rate
            )
            audio_np = np.array(audio)
            sd.play(audio_np, self.tts_sample_rate)
        except Exception as e:
            self.text_output.insert(tk.END, f"\nTTS error: {e}\n")

def main():
    """Main entry point for the application"""
    # Initialize Tk with specific options
    root_window = tk.Tk()
    
    # Set window title
    root_window.title("LLAVA Model Test")
    
    # Set window geometry
    root_window.geometry("600x800")
    
    # Make window resizable - explicitly set both dimensions
    root_window.resizable(width=True, height=True)
    
    # Set minimum window size
    root_window.minsize(400, 400)
    
    # Force update of window manager hints
    root_window.update_idletasks()
    
    # Create the app
    app = App(root_window)
    
    # Start the main loop
    try:
        root_window.mainloop()
    finally:
        app.stop_webcam_preview()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting on Ctrl-C (KeyboardInterrupt)")
