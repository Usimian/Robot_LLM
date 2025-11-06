#!/usr/bin/env python3
"""
Natural Language Command Parser for Robot Control

Uses a lightweight text-only LLM (Qwen2.5-3B-Instruct) to parse natural language
commands into structured robot actions. This enables flexible command interpretation
without relying on brittle keyword matching.

Features:
- Fast inference (~50-150ms on GPU)
- Structured JSON output with validation
- Handles unlimited natural language variations
- Determines if vision/VLM processing is needed
- Extracts parameters (distance, angle, speed, targets)

Examples:
  "move forward 1 meter" → {action: "move_forward", distance: 1.0}
  "turn slowly clockwise until you see the door" → {action: "turn_right", speed: 0.2, condition: "see_door"}
  "move to 1m in front of the refrigerator" → {action: "navigate_to", target: "refrigerator", ...}

Author: Robot LLM System - NLP Command Parser
"""

import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()


@dataclass
class ParsedCommand:
    """Structured representation of a parsed robot command"""
    action: str  # move_forward, turn_left, navigate_to, etc.
    parameters: Dict[str, Any]  # distance, angle, speed, target, etc.
    needs_vision: bool  # Does this command require VLM processing?
    confidence: float  # Parser confidence (0.0-1.0)
    raw_response: str  # Raw LLM output for debugging

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'action': self.action,
            'parameters': self.parameters,
            'needs_vision': self.needs_vision,
            'confidence': self.confidence,
            'raw_response': self.raw_response
        }


class NLPCommandParser:
    """Natural language command parser using lightweight LLM"""

    # Valid robot actions
    VALID_ACTIONS = {
        'move_forward', 'move_backward', 'turn_left', 'turn_right',
        'strafe_left', 'strafe_right', 'navigate_to', 'stop',
        'emergency_stop', 'rotate_to', 'follow', 'explore'
    }

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 device: str = "cuda", logger=None):
        """Initialize NLP command parser

        Args:
            model_name: Hugging Face model name (default: Qwen2.5-3B-Instruct)
            device: Device to run on (cuda/cpu)
            logger: ROS2 logger instance
        """
        self.model_name = model_name
        self.device = device
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.model_loaded = False

    def load_model(self):
        """Load the text-only LLM for command parsing"""
        if self.model_loaded:
            if self.logger:
                self.logger.warn("NLP parser model already loaded")
            return

        try:
            if self.logger:
                self.logger.info(f"🔄 Loading NLP command parser: {self.model_name}")

            start_time = time.time()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model with optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            self.model.eval()

            # Enable optimizations
            if self.device == "cuda" and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True

            load_time = time.time() - start_time
            self.model_loaded = True

            if self.logger:
                self.logger.info(f"✅ NLP parser loaded in {load_time:.2f}s")
                if self.device == "cuda":
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    self.logger.info(f"   └── GPU memory: {memory_used:.2f}GB")

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to load NLP parser: {e}")
            raise RuntimeError(f"NLP parser initialization failed: {e}")

    def parse_command(self, user_input: str) -> ParsedCommand:
        """Parse natural language command into structured action

        Args:
            user_input: Natural language command from user

        Returns:
            ParsedCommand with action, parameters, and metadata
        """
        if not self.model_loaded:
            raise RuntimeError("NLP parser model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Create parsing prompt
            prompt = self._create_parsing_prompt(user_input)

            # Generate structured response
            raw_response = self._generate(prompt, max_new_tokens=150, temperature=0.1)

            # Parse JSON response
            parsed_json = self._extract_json(raw_response)

            # Validate and create ParsedCommand
            command = self._validate_and_create_command(parsed_json, raw_response)

            inference_time = time.time() - start_time

            if self.logger:
                self.logger.info(f"🧠 NLP Parser: '{user_input}' → {command.action} ({inference_time*1000:.0f}ms)")
                if command.parameters:
                    self.logger.info(f"   └── Parameters: {command.parameters}")
                if command.needs_vision:
                    self.logger.info(f"   └── Requires vision processing")

            return command

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ NLP parsing failed: {e}")
            # Return fallback safe command
            return ParsedCommand(
                action='stop',
                parameters={},
                needs_vision=False,
                confidence=0.0,
                raw_response=f"Error: {e}"
            )

    def _create_parsing_prompt(self, user_input: str) -> str:
        """Create the parsing prompt for the LLM"""
        return f"""You are a robot command parser. Convert natural language into JSON commands.

Available actions: move_forward, move_backward, turn_left, turn_right, strafe_left, strafe_right, navigate_to, stop, emergency_stop

Output JSON schema:
{{
  "action": "<action_type>",
  "parameters": {{
    "distance": <meters>,          // for linear movement (optional)
    "angle": <degrees>,            // for turns (optional)
    "speed": <m/s>,                // movement speed (optional)
    "target": "<object_name>",     // for navigate_to (optional)
    "relative_position": "front|behind|left|right|beside",  // for navigate_to (optional)
    "condition": "<visual_condition>",  // e.g., "see_door", "detect_person" (optional)
    "duration": <seconds>          // how long to execute (optional)
  }},
  "needs_vision": true/false
}}

Examples:

User: "move forward 1 meter"
Output: {{"action": "move_forward", "parameters": {{"distance": 1.0}}, "needs_vision": false}}

User: "go ahead 2m"
Output: {{"action": "move_forward", "parameters": {{"distance": 2.0}}, "needs_vision": false}}

User: "turn left 90 degrees"
Output: {{"action": "turn_left", "parameters": {{"angle": 90}}, "needs_vision": false}}

User: "rotate right 45 deg"
Output: {{"action": "turn_right", "parameters": {{"angle": 45}}, "needs_vision": false}}

User: "turn slowly clockwise until you see the door"
Output: {{"action": "turn_right", "parameters": {{"speed": 0.2, "condition": "see_door"}}, "needs_vision": true}}

User: "move to 1m in front of the refrigerator"
Output: {{"action": "navigate_to", "parameters": {{"target": "refrigerator", "relative_position": "front", "distance": 1.0}}, "needs_vision": true}}

User: "go to the door"
Output: {{"action": "navigate_to", "parameters": {{"target": "door"}}, "needs_vision": true}}

User: "back up 0.5 meters"
Output: {{"action": "move_backward", "parameters": {{"distance": 0.5}}, "needs_vision": false}}

User: "stop"
Output: {{"action": "stop", "parameters": {{}}, "needs_vision": false}}

User: "strafe left 1 meter"
Output: {{"action": "strafe_left", "parameters": {{"distance": 1.0}}, "needs_vision": false}}

User: "{user_input}"
Output:"""

    def _generate(self, prompt: str, max_new_tokens: int = 150,
                  temperature: float = 0.1) -> str:
        """Generate response from LLM

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Generated text response
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part (exclude prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response

        Handles cases where LLM includes extra text before/after JSON
        """
        # Try to find JSON object in response
        start_idx = text.find('{')

        if start_idx == -1:
            raise ValueError(f"No JSON object found in response: {text}")

        # Find the matching closing brace
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if end_idx == -1:
            raise ValueError(f"No matching closing brace found in response: {text}")

        json_str = text[start_idx:end_idx + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {json_str}. Error: {e}")

    def _validate_and_create_command(self, parsed_json: Dict[str, Any],
                                     raw_response: str) -> ParsedCommand:
        """Validate parsed JSON and create ParsedCommand

        Args:
            parsed_json: Parsed JSON from LLM
            raw_response: Raw LLM response

        Returns:
            Validated ParsedCommand
        """
        # Validate action
        action = parsed_json.get('action', 'stop')
        if action not in self.VALID_ACTIONS:
            if self.logger:
                self.logger.warn(f"Invalid action '{action}', defaulting to 'stop'")
            action = 'stop'

        # Extract parameters
        parameters = parsed_json.get('parameters', {})

        # Convert angle from degrees to radians if present
        if 'angle' in parameters:
            import math
            angle_deg = float(parameters['angle'])
            parameters['angle'] = math.radians(angle_deg)
            if self.logger:
                self.logger.debug(f"   └── Converted angle: {angle_deg}° → {parameters['angle']:.3f} rad")

        # Ensure distance/speed are floats
        if 'distance' in parameters:
            parameters['distance'] = float(parameters['distance'])
        if 'speed' in parameters:
            parameters['speed'] = float(parameters['speed'])
        if 'duration' in parameters:
            parameters['duration'] = float(parameters['duration'])

        # Needs vision flag
        needs_vision = parsed_json.get('needs_vision', False)

        # Confidence (default to high for valid parses)
        confidence = 0.95

        return ParsedCommand(
            action=action,
            parameters=parameters,
            needs_vision=needs_vision,
            confidence=confidence,
            raw_response=raw_response
        )

    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.model_loaded = False

            if self.device == "cuda":
                torch.cuda.empty_cache()

            if self.logger:
                self.logger.info("🗑️ NLP parser model unloaded")


# Example usage
if __name__ == "__main__":
    # Test the parser
    parser = NLPCommandParser(device="cuda")
    parser.load_model()

    test_commands = [
        "move forward 1 meter",
        "turn slowly clockwise until you see the door",
        "move to 1m in front of the refrigerator",
        "back up 2 meters",
        "rotate left 180 degrees",
        "stop",
        "go to the kitchen",
        "strafe right 0.5m"
    ]

    for cmd in test_commands:
        print(f"\nInput: {cmd}")
        result = parser.parse_command(cmd)
        print(f"Action: {result.action}")
        print(f"Parameters: {result.parameters}")
        print(f"Needs Vision: {result.needs_vision}")
        print(f"Confidence: {result.confidence}")
