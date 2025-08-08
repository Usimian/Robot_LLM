# HTTP Dependencies Cleanup Report
==================================================

## Summary
- Total Python files: 397
- Deprecated files (can be removed): 6
- Files with HTTP dependencies: 213
- Clean ROS2 files: 178

## 🗑️ Deprecated Files (Safe to Remove)
These files have been replaced by ROS2 versions:

- ❌ robot_client_examples.py
- ❌ robot_vila_server.py
- ❌ unified_robot_controller.py
- ❌ unified_robot_client.py
- ❌ robot_gui.py
- ❌ vila_ros_node.py

## ⚠️ Files with HTTP Dependencies
These files still contain HTTP-related code:

### test_server_safety.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 7: `import requests`

**Pattern: `http://|https://`**
  - Line 12: `server_url = "http://localhost:5000"`

**Pattern: `localhost:\d+`**
  - Line 12: `server_url = "http://localhost:5000"`

**Pattern: `requests\.`**
  - Line 32: `response = requests.post(`
  - Line 64: `response = requests.post(`
  - Line 96: `response = requests.post(`
  - Line 126: `response = requests.post(`

### migrate_to_unified_system.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 12: `import requests`
  - Line 148: `python3 -c "import aiohttp, flask_socketio" 2>/dev/null`

**Pattern: `http://|https://`**
  - Line 122: `response = requests.get('http://localhost:5000/api/robots', timeout=5)`
  - Line 221: `- **Web Interface**: http://localhost:5000`
  - Line 266: `print("🌐 Web interface: http://localhost:5000")`

**Pattern: `localhost:\d+`**
  - Line 122: `response = requests.get('http://localhost:5000/api/robots', timeout=5)`
  - Line 221: `- **Web Interface**: http://localhost:5000`
  - Line 266: `print("🌐 Web interface: http://localhost:5000")`

**Pattern: `:\d+/`**
  - Line 122: `response = requests.get('http://localhost:5000/api/robots', timeout=5)`

**Pattern: `\.get\(.*http`**
  - Line 122: `response = requests.get('http://localhost:5000/api/robots', timeout=5)`

**Pattern: `requests\.`**
  - Line 122: `response = requests.get('http://localhost:5000/api/robots', timeout=5)`
  - Line 127: `except requests.RequestException as e:`

**Pattern: `websocket`**
  - Line 180: `- **Communication**: HTTP → WebSocket → TCP (multiple hops)`
  - Line 188: `- **Communication**: Direct HTTP/WebSocket (single hop)`

### test_robot_sensor_reader.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 7: `import requests`

**Pattern: `http://|https://`**
  - Line 16: `self.robot_url = f"http://{robot_ip}:{robot_port}"`

**Pattern: `requests\.`**
  - Line 23: `response = requests.get(self.sensor_url, timeout=3)`
  - Line 30: `except requests.exceptions.RequestException as e:`
  - Line 36: `response = requests.get(self.image_url, timeout=5)`
  - Line 51: `except requests.exceptions.RequestException as e:`

### cleanup_http_dependencies.py
**Pattern: `http://|https://`**
  - Line 25: `r'http://|https://',`

**Pattern: `websocket`**
  - Line 20: `r'import\s+(requests|flask|aiohttp|httpx|websockets|socketio)',`
  - Line 21: `r'from\s+(requests|flask|aiohttp|httpx|websockets|socketio)',`
  - Line 44: `# WebSocket patterns`
  - Line 46: `r'websocket',`
  - Line 212: `report.append("# websockets")`

**Pattern: `@.*\.event`**
  - Line 48: `r'@.*\.event',`

### ollama_cam.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 3: `import requests`

**Pattern: `http://|https://`**
  - Line 37: `self.api_url = "http://localhost:11434/api/generate"`
  - Line 38: `self.api_base = "http://localhost:11434"`

**Pattern: `localhost:\d+`**
  - Line 37: `self.api_url = "http://localhost:11434/api/generate"`
  - Line 38: `self.api_base = "http://localhost:11434"`

**Pattern: `:\d+/`**
  - Line 37: `self.api_url = "http://localhost:11434/api/generate"`

**Pattern: `requests\.`**
  - Line 166: `response = requests.get(f"{self.api_base}/api/tags", timeout=10)`
  - Line 315: `response = requests.post(self.api_url, json=payload, timeout=60)`
  - Line 325: `except requests.exceptions.Timeout:`
  - Line 328: `except requests.exceptions.ConnectionError:`

### test_sensors.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 7: `import requests`

**Pattern: `http://|https://`**
  - Line 32: `"http://localhost:5000/robots/yahboomcar_x3_01/sensors",`

**Pattern: `localhost:\d+`**
  - Line 32: `"http://localhost:5000/robots/yahboomcar_x3_01/sensors",`

**Pattern: `:\d+/`**
  - Line 32: `"http://localhost:5000/robots/yahboomcar_x3_01/sensors",`

**Pattern: `requests\.`**
  - Line 31: `response = requests.post(`
  - Line 43: `except requests.exceptions.RequestException as e:`

### test_robot_sensor_server.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 7: `from flask import Flask, jsonify`

**Pattern: `from\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 7: `from flask import Flask, jsonify`

**Pattern: `http://|https://`**
  - Line 85: `print("   GET http://localhost:8080/sensors - Get sensor data")`
  - Line 86: `print("   GET http://localhost:8080/image - Get camera image")`
  - Line 87: `print("   GET http://localhost:8080/health - Health check")`

**Pattern: `localhost:\d+`**
  - Line 85: `print("   GET http://localhost:8080/sensors - Get sensor data")`
  - Line 86: `print("   GET http://localhost:8080/image - Get camera image")`
  - Line 87: `print("   GET http://localhost:8080/health - Health check")`

**Pattern: `:\d+/`**
  - Line 85: `print("   GET http://localhost:8080/sensors - Get sensor data")`
  - Line 86: `print("   GET http://localhost:8080/image - Get camera image")`
  - Line 87: `print("   GET http://localhost:8080/health - Health check")`

**Pattern: `@app\.route`**
  - Line 14: `@app.route('/sensors', methods=['GET'])`
  - Line 33: `@app.route('/image', methods=['GET'])`
  - Line 71: `@app.route('/health', methods=['GET'])`

**Pattern: `Flask\(`**
  - Line 12: `app = Flask(__name__)`

**Pattern: `app\.run\(`**
  - Line 92: `app.run(host='0.0.0.0', port=8080, debug=False)`

**Pattern: `jsonify\(`**
  - Line 31: `return jsonify(sensor_data)`
  - Line 63: `return jsonify({`
  - Line 74: `return jsonify({`

### VILA/server.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 14: `import requests`

**Pattern: `requests\.`**
  - Line 62: `response = requests.get(video_url)`
  - Line 118: `response = requests.get(image_url)`

### VILA/llava/mm_utils.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# dynamic_preprocess and find_closest_aspect_ratio are referenced from https://github.com/OpenGVLab/InternVL`

### VILA/llava/media.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/conversation.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/constants.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/entry.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/utils/media.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 12: `import requests`

**Pattern: `http://|https://`**
  - Line 25: `if image.path.startswith("http://") or image.path.startswith("https://"):`

**Pattern: `requests\.`**
  - Line 26: `image = PIL.Image.open(requests.get(image.path, stream=True).raw)`

### VILA/llava/utils/merge_lora_weights_and_save_hf_model.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is modified from https://github.com/dvlab-research/LongLoRA`

### VILA/llava/utils/utils.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/utils/tokenizer.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/cli/upload2hf.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 106: `BASE_URL = "https://hf.co"`
  - Line 108: `BASE_URL = "https://hf.co/datasets"`

### VILA/llava/train/short_video_filter.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/train/train.py
**Pattern: `http://|https://`**
  - Line 1: `# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:`
  - Line 9: `#        http://www.apache.org/licenses/LICENSE-2.0`
  - Line 437: `Please install via `pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple --upgrade one-logger-utils`

### VILA/llava/train/transformer_normalize_monkey_patch.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/train/slurm_utils.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 25: `import requests`

**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/train/train_ln.py
**Pattern: `http://|https://`**
  - Line 1: `# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:`
  - Line 9: `#        http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/train/train_hybrid.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/train/args.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/train/train_mem_ln.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/train/train_llm_to_long.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is modified from https://github.com/dvlab-research/LongLoRA`

### VILA/llava/train/train_mem.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/train/utils.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/train/llava_trainer.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/train/sequence_parallel/all_to_all.py
**Pattern: `http://|https://`**
  - Line 6: `# This file is modified from https://github.com/feifeibear/long-context-attention`
  - Line 7: `# Implementation refers to USP Paper: https://arxiv.org/abs/2405.07719`
  - Line 73: `# https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single`
  - Line 123: `# https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single`
  - Line 200: `# https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single`
  - ... and 1 more matches

### VILA/llava/train/sequence_parallel/ulysses_attn.py
**Pattern: `http://|https://`**
  - Line 6: `# This file is modified from https://github.com/feifeibear/long-context-attention`
  - Line 7: `# Implementation refers to USP Paper: https://arxiv.org/abs/2405.07719`
  - Line 8: `# This file is also partly modified from https://github.com/microsoft/DeepSpeed`
  - Line 9: `# Implementation refers to Ulysses Paper: https://arxiv.org/abs/2309.14509`

### VILA/llava/train/sequence_parallel/globals.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# This file is modified from https://github.com/feifeibear/long-context-attention`
  - Line 18: `# Implementation refers to USP Paper: https://arxiv.org/abs/2405.07719`

### VILA/llava/train/sequence_parallel/input_utils.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/train/sequence_parallel/monkey_patch.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/train/sequence_parallel/hybrid_attn.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# This file is modified from https://github.com/feifeibear/long-context-attention`
  - Line 18: `# Implementation refers to USP Paper: https://arxiv.org/abs/2405.07719`

### VILA/llava/train/sequence_parallel/ring/zigzag_ring_flash_attn.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# Adopted from https://github.com/zhuzilin/ring-flash-attention.`
  - Line 18: `# Implementation refers to Ring Attention Paper: https://arxiv.org/abs/2310.01889`

### VILA/llava/train/sequence_parallel/ring/zigzag_ring_flash_attn_varlen.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# Adopted from https://github.com/zhuzilin/ring-flash-attention.`
  - Line 18: `# Implementation refers to Ring Attention Paper: https://arxiv.org/abs/2310.01889`

### VILA/llava/train/sequence_parallel/ring/ring_flash_attn.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# Adopted from https://github.com/zhuzilin/ring-flash-attention.`
  - Line 18: `# Implementation refers to Ring Attention Paper: https://arxiv.org/abs/2310.01889`

### VILA/llava/train/sequence_parallel/ring/triton_utils.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# Adopted from https://github.com/zhuzilin/ring-flash-attention.`
  - Line 18: `# Implementation refers to Ring Attention Paper: https://arxiv.org/abs/2310.01889`

### VILA/llava/train/sequence_parallel/ring/utils.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# Adopted from https://github.com/zhuzilin/ring-flash-attention.`
  - Line 18: `# Implementation refers to Ring Attention Paper: https://arxiv.org/abs/2310.01889`

### VILA/llava/train/sequence_parallel/ring/ring_flash_attn_varlen.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# Adopted from https://github.com/zhuzilin/ring-flash-attention.`
  - Line 18: `# Implementation refers to Ring Attention Paper: https://arxiv.org/abs/2310.01889`

### VILA/llava/train/sequence_parallel/ring/__init__.py
**Pattern: `http://|https://`**
  - Line 1: `# Adopted from https://github.com/zhuzilin/ring-flash-attention.`
  - Line 2: `# Implementation refers to Ring Attention Paper: https://arxiv.org/abs/2310.01889`

### VILA/llava/train/sequence_parallel/ring/stripe_flash_attn.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# Adopted from https://github.com/zhuzilin/ring-flash-attention.`
  - Line 18: `# Implementation refers to Striped Attention Paper: https://arxiv.org/abs/2311.09431`

### VILA/llava/trl/import_utils.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/trl/core.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 54: `filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)`
  - Line 58: `From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317`
  - Line 132: `See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591`
  - Line 190: `https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713`

### VILA/llava/trl/extras/best_of_n_sampler.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 58: `See `GenerationConfig` (https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig) for more details`

### VILA/llava/trl/extras/__init__.py
**Pattern: `http://|https://`**
  - Line 9: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/trl/extras/dataset_formatting.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/trl/trainer/iterative_sft_trainer.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/trl/trainer/ddpo_config.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 41: `"""Log with either 'wandb' or 'tensorboard', check  https://huggingface.co/docs/accelerate/usage_guides/tracking for more details"""`

### VILA/llava/trl/trainer/dpo_trainer.py
**Pattern: `http://|https://`**
  - Line 8: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 77: `The robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report that should be between 0 and 0.5.`
  - Line 79: `The type of DPO loss to use. Either `"sigmoid"` the default DPO loss,`"hinge"` loss from [SLiC](https://arxiv.org/abs/2305.10425) paper, `"ipo"` from [IPO](https://arxiv.org/abs/2310.12036) paper, or `"kto"` from the HALOs [report](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf).`
  - Line 376: `# see: https://github.com/huggingface/trl/pull/1255`
  - Line 421: `# Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473`
  - ... and 5 more matches

### VILA/llava/trl/trainer/base.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/trl/trainer/ppo_trainer.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 73: `This is a [TRL language model](https://github.com/huggingface/trl) that has been fine-tuned with reinforcement learning to`
  - Line 112: `https://github.com/openai/summarize-from-feedback`
  - Line 333: `# check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html`
  - Line 334: `# or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11`
  - ... and 4 more matches

### VILA/llava/trl/trainer/ddpo_trainer.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 57: `Note, this trainer is heavily inspired by the work here: https://github.com/kvablack/ddpo-pytorch`
  - Line 169: `# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices`

### VILA/llava/trl/trainer/ppo_config.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 45: `"""Log with either 'wandb' or 'tensorboard', check  https://huggingface.co/docs/accelerate/usage_guides/tracking for more details"""`

### VILA/llava/trl/trainer/reward_config.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 27: `[argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the`

### VILA/llava/trl/trainer/sft_trainer.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 102: `stack-llama example: https://github.com/huggingface/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53.`
  - Line 113: `fine-tuning. Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune`
  - Line 410: `# Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt`
  - Line 501: `Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914`

### VILA/llava/trl/trainer/utils.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 37: `https://arxiv.org/pdf/1909.08593.pdf`
  - Line 325: `# adapted from https://stackoverflow.com/questions/73256206`
  - Line 484: `https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75`
  - Line 524: `https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L57C1-L73C75`
  - ... and 2 more matches

### VILA/llava/trl/trainer/reward_trainer.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 51: `loss of the reward model as outlined in https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/.`
  - Line 181: `try:  # for bc before https://github.com/huggingface/transformers/pull/25435`

### VILA/llava/trl/trainer/model_config.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/trl/trainer/__init__.py
**Pattern: `http://|https://`**
  - Line 9: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/trl/environment/base_environment.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/trl/models/modeling_value_head.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 194: `Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)`

### VILA/llava/trl/models/modeling_sd_base.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 214: `CycleDiffusion. (https://arxiv.org/abs/2210.05559)`
  - Line 225: `# See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf`
  - Line 254: `# "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf`
  - Line 286: `# 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf`
  - ... and 11 more matches

### VILA/llava/trl/models/modeling_base.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/trl/models/utils.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 91: `# resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377`

### VILA/llava/trl/models/__init__.py
**Pattern: `http://|https://`**
  - Line 9: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/eval/model_vqa_videodemo.py
**Pattern: `http://|https://`**
  - Line 1: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/eval/m4c_evaluator.py
**Pattern: `http://|https://`**
  - Line 11: `https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897`
  - Line 299: `# pip install git+https://github.com/ronghanghu/coco-caption.git@python23`
  - Line 300: `# Original pycocoevalcap code is at https://github.com/tylin/coco-caption`
  - Line 308: `"pip install git+https://github.com/ronghanghu/coco-caption.git@python23"  # noqa`

### VILA/llava/eval/model_vqa_video.py
**Pattern: `http://|https://`**
  - Line 1: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/eval/vision_niah_vila/eval_vision_niah.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is modified from https://github.com/EvolvingLMMs-Lab/LongVA`

### VILA/llava/eval/vision_niah_vila/produce_haystack_embedding.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is adopted from https://github.com/EvolvingLMMs-Lab/LongVA`

### VILA/llava/eval/vision_niah_vila/produce_needle_embedding.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is adopted from https://github.com/EvolvingLMMs-Lab/LongVA`

### VILA/llava/eval/vision_niah_vila/zigzag_ring_attn/modeling_qwen2.py
**Pattern: `http://|https://`**
  - Line 12: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 116: `# See https://github.com/huggingface/transformers/pull/29285`
  - Line 325: `# flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.`
  - Line 598: `# Reference: https://github.com/pytorch/pytorch/issues/112577.`
  - Line 709: `This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.`
  - ... and 1 more matches

### VILA/llava/eval/vision_niah_vila/zigzag_ring_attn/prepare_inputs.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is adopted from https://github.com/EvolvingLMMs-Lab/LongVA`

### VILA/llava/eval/vision_niah_vila/zigzag_ring_attn/monkey_patch.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is adopted from https://github.com/EvolvingLMMs-Lab/LongVA`

### VILA/llava/eval/mmmu_utils/eval_utils.py
**Pattern: `http://|https://`**
  - Line 2: `# https://github.com/MMMU-Benchmark/MMMU`

### VILA/llava/eval/video/model_vqa_videodemo_benchmark.py
**Pattern: `http://|https://`**
  - Line 1: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/eval/video/utils.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 6: `import requests`

### VILA/llava/eval/video/eval_video_qa.py
**Pattern: `http://|https://`**
  - Line 1: `# This file is originated from: https://github.com/mbzuai-oryx/Video-ChatGPT`

### VILA/llava/eval/lmms/models/vila_internal.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 5: `import requests`

**Pattern: `requests\.`**
  - Line 100: `_unpatched_post = requests.post`
  - Line 108: `requests.post = _patched_post`

### VILA/llava/data/datasets_mixture.py
**Pattern: `http://|https://`**
  - Line 8: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/data/simple_vila_webdataset.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/data/utils.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/data/dataset.py
**Pattern: `http://|https://`**
  - Line 8: `#        http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/data/dataset_impl/coyo_recap.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/data/dataset_impl/hiertext.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# refernced from https://github.com/CVC-DAG/OCR_datasets/blob/master/src/datasets/ocr/hiertext.py`

### VILA/llava/data/dataset_impl/textocr.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/data/dataset_impl/general_img_text.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/data/dataset_impl/sam.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/data/dataset_impl/panda70m.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/llava_arch.py
**Pattern: `http://|https://`**
  - Line 7: `#        http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/apply_delta.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/model/make_delta.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/model/deprecate_consolidate.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/consolidate.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/configuration_llava.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/builder.py
**Pattern: `http://|https://`**
  - Line 1: `# This file is modified from https://github.com/haotian-liu/LLaVA/`
  - Line 8: `#        http://www.apache.org/licenses/LICENSE-2.0`
  - Line 61: `"There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."`

### VILA/llava/model/utils/utils.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/model/coat/optimizer/fp8_adamw.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 235: `# Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424`

### VILA/llava/model/coat/optimizer/kernels/setup.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/utils.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/_memory_io.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/mul_fwd.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/add_fwd.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/add_bwd.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/func_quantize.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/linear.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/gelu_bwd.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/func_rmsnorm.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/gelu_fwd.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/mul_bwd_legacy.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/_quantize_pertensor_transpose.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/_transpose.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/fp8linear.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/func_layernorm_noparam.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/silu_bwd_legacy.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/_quantize_pertensor.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/common.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/silu_bwd.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/_division_transpose.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/gelu_bwd_legacy.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/silu_fwd.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/mul_bwd_silu_fwd.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/_dequantize.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/mul_bwd.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/_division.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/real_quantization/_quantize.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/fake_quantization/FloatPointQuantizeTriton.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/fake_quantization/FloatPointQuantizeTorch.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/fake_quantization/quantize_function.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/fake_quantization/utils.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/coat/activation/models/coat_llama.py
**Pattern: `http://|https://`**
  - Line 12: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 901: `# flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.`
  - Line 921: `"make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"`
  - Line 1078: `# Reference: https://github.com/pytorch/pytorch/issues/112577.`
  - Line 1308: `"(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"`

### VILA/llava/model/coat/activation/models/coat_olmo.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 19: `[MosaiclML](https://github.com/mosaicml/examples.git) and`
  - Line 20: `[minGPT](https://github.com/karpathy/minGPT.git)`
  - Line 795: `# See https://github.com/pytorch/pytorch/issues/110966.`
  - Line 1844: `# For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222`

### VILA/llava/model/coat/activation/models/_fp8manager.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/liger/cross_entropy.py
**Pattern: `http://|https://`**
  - Line 46: `We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.`
  - Line 68: `# https://github.com/triton-lang/triton/issues/1058`
  - Line 90: `# Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867`
  - Line 100: `# See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310`
  - Line 162: `# https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34`
  - ... and 5 more matches

### VILA/llava/model/liger/utils.py
**Pattern: `http://|https://`**
  - Line 3: `See the original Unsloth repository at https://github.com/unslothai/unsloth.`
  - Line 6: `https://github.com/linkedin/Liger-Kernel/blob/7382a8761f9af679482b968f9348013d933947c7/src/liger_kernel/ops/utils.py#L23`
  - Line 8: `https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43`
  - Line 42: `# reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43`

### VILA/llava/model/multimodal_encoder/image_processor.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/multimodal_encoder/radio_torchhub_encoder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/multimodal_encoder/ps3_encoder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/multimodal_encoder/radio_encoder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/multimodal_encoder/siglip_encoder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/multimodal_encoder/intern_encoder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/multimodal_encoder/vision_encoder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/model/multimodal_encoder/visualize_features.py
**Pattern: `http://|https://`**
  - Line 44: `Computes the RankMe (http://arxiv.org/abs/2210.02885) and LiDAR (http://arxiv.org/abs/2312.04000)`

### VILA/llava/model/multimodal_encoder/clip_encoder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/model/multimodal_encoder/builder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/model/multimodal_encoder/siglip/modeling_siglip.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 1274: `>>> import requests`
  - Line 1403: `>>> import requests`
  - Line 1458: `>>> import requests`
  - Line 1597: `>>> import requests`

**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 70: `# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf`
  - Line 289: `https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174`
  - Line 301: `# see discussion at https://github.com/facebookresearch/dino/issues/8`
  - Line 456: `# flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.`
  - ... and 7 more matches

**Pattern: `requests\.`**
  - Line 1281: `>>> image = Image.open(requests.get(url, stream=True).raw)`
  - Line 1411: `>>> image = Image.open(requests.get(url, stream=True).raw)`
  - Line 1466: `>>> image = Image.open(requests.get(url, stream=True).raw)`
  - Line 1601: `>>> image = Image.open(requests.get(url, stream=True).raw)`

### VILA/llava/model/multimodal_encoder/intern/modeling_intern_vit.py
**Pattern: `http://|https://`**
  - Line 33: `DropBlock: A regularization method for convolutional networks (https://arxiv.org/abs/1810.12890)`
  - Line 35: `Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)`
  - Line 39: `- https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74`
  - Line 40: `- https://github.com/clovaai/assembled-cnn/blob/master/nets/blocks.py`
  - Line 82: `"""DropBlock. See https://arxiv.org/pdf/1810.12890.pdf`
  - ... and 3 more matches

### VILA/llava/model/multimodal_encoder/intern/flash_attention.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# https://github.com/Dao-AILab/flash-attention/blob/v0.2.8/flash_attn/flash_attention.py`

### VILA/llava/model/language_model/qllava_qllama.py
**Pattern: `http://|https://`**
  - Line 7: `#        http://www.apache.org/licenses/LICENSE-2.0`
  - Line 15: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/model/language_model/qllama.py
**Pattern: `http://|https://`**
  - Line 12: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 157: `# flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.`

### VILA/llava/model/language_model/llava_topdown_llama.py
**Pattern: `http://|https://`**
  - Line 7: `#        http://www.apache.org/licenses/LICENSE-2.0`
  - Line 15: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/model/language_model/realqmemllama.py
**Pattern: `http://|https://`**
  - Line 12: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 238: `"make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"`
  - Line 425: `# Reference: https://github.com/pytorch/pytorch/issues/112577.`

### VILA/llava/model/language_model/fp8activationqwen2.py
**Pattern: `http://|https://`**
  - Line 12: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 1070: `# flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.`
  - Line 1266: `# Reference: https://github.com/pytorch/pytorch/issues/112577.`
  - Line 1546: `"(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"`

### VILA/llava/model/language_model/fp8linearqwen2.py
**Pattern: `http://|https://`**
  - Line 12: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 203: `# flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.`

### VILA/llava/model/language_model/fp8activationresidualqwen2.py
**Pattern: `http://|https://`**
  - Line 12: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 991: `# flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.`
  - Line 1186: `# Reference: https://github.com/pytorch/pytorch/issues/112577.`
  - Line 1419: `"(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"`

### VILA/llava/model/language_model/qmemllama.py
**Pattern: `http://|https://`**
  - Line 12: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 242: `"make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"`
  - Line 429: `# Reference: https://github.com/pytorch/pytorch/issues/112577.`

### VILA/llava/model/language_model/llava_llama.py
**Pattern: `http://|https://`**
  - Line 7: `#        http://www.apache.org/licenses/LICENSE-2.0`
  - Line 15: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/model/language_model/builder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/multimodal_projector/base_projector.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/model/multimodal_projector/builder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/remote_code/mm_utils.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 17: `# dynamic_preprocess and find_closest_aspect_ratio are referenced from https://github.com/OpenGVLab/InternVL`

### VILA/llava/remote_code/tokenizer_utils.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/remote_code/media.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 10: `import requests`

**Pattern: `http://|https://`**
  - Line 47: `if image.path.startswith("http://") or image.path.startswith("https://"):`

**Pattern: `requests\.`**
  - Line 48: `image = PIL.Image.open(requests.get(image.path, stream=True).raw)`

### VILA/llava/remote_code/base_projector.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/remote_code/auto_processor.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 10: `import requests`
  - Line 65: `import requests`

**Pattern: `http://|https://`**
  - Line 42: `elif image.startswith("http://") or image.startswith("https://"):`

**Pattern: `requests\.`**
  - Line 43: `response = requests.get(image, stream=True)`
  - Line 71: `response = requests.get(url_or_fpath, stream=True)`

### VILA/llava/remote_code/conversation.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/remote_code/constants.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/remote_code/siglip_encoder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/remote_code/utils.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `# This file is modified from https://github.com/haotian-liu/LLaVA/`

### VILA/llava/remote_code/builder.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/llava/wids/wids_specs.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 93: `import requests`

**Pattern: `http://|https://`**
  - Line 120: `{"url": "http://example.com/file.tar", "nsamples": 1000},`
  - Line 125: `{"source_url": "http://example.com/dataset.json"},`
  - Line 127: `{"url": "http://example.com/file.tar", "nsamples": 1000},`

**Pattern: `requests\.`**
  - Line 95: `jsondata = requests.get(source).text`

### VILA/llava/wids/wids.py
**Pattern: `http://|https://`**
  - Line 262: `# NOTE(ligeng): https://stackoverflow.com/questions/11072705/twitter-trends-api-unicodedecodeerror-utf8-codec-cant-decode-byte-0x8b-in-po`
  - Line 598: `if url.startswith(("https://", "http://", "gs://", "/", "~")):`

### VILA/serving/query_nvila.py
**Pattern: `http://|https://`**
  - Line 7: `base_url="http://localhost:8000",`
  - Line 29: `video_url = "https://avtshare01.rz.tu-ilmenau.de/avt-vqdb-uhd-1/test_1/segments/bigbuck_bunny_8bit_200kbps_360p_60.0fps_hevc.mp4"`
  - Line 48: `#         "url": "https://blog.logomyway.com/wp-content/uploads/2022/01/NVIDIA-logo.jpg",`

**Pattern: `localhost:\d+`**
  - Line 7: `base_url="http://localhost:8000",`

### VILA/serving/server.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 14: `import requests`

**Pattern: `requests\.`**
  - Line 129: `response = requests.get(video_url)`
  - Line 150: `response = requests.get(image_url)`

### VILA/data_prepare/panda_split.py
**Pattern: `http://|https://`**
  - Line 7: `#      http://www.apache.org/licenses/LICENSE-2.0`

### VILA/data_prepare/mmc4/mmc4_merger.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/data_prepare/mmc4/mmc4_downloader.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 28: `import aiohttp`

**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

**Pattern: `aiohttp\.`**
  - Line 136: `conn = aiohttp.TCPConnector(ssl=ssl_context)`
  - Line 140: `async with aiohttp.ClientSession(connector=conn) as session:`

### VILA/data_prepare/mmc4/mmc4_filter_and_counter.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/data_prepare/sft/preprocess_flan.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/data_prepare/sft/merge_idefics2.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/data_prepare/sft/preprocess_m3it.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/data_prepare/sft/split_vflan.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/data_prepare/sft/preprocess_idefics2.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/data_prepare/sft/merge_llava_onevision.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/data_prepare/sft/merge_llava_onevision_eagle.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### VILA/data_prepare/coyo/coyo_downloader.py
**Pattern: `import\s+(requests|flask|aiohttp|httpx|websockets|socketio)`**
  - Line 26: `import aiohttp`

**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

**Pattern: `aiohttp\.`**
  - Line 119: `conn = aiohttp.TCPConnector(ssl=ssl_context)`
  - Line 123: `async with aiohttp.ClientSession(connector=conn) as session:`

### VILA/data_prepare/coyo/coyo_splitter.py
**Pattern: `http://|https://`**
  - Line 7: `#     http://www.apache.org/licenses/LICENSE-2.0`

### PS3/train/src/open_clip_train/main.py
**Pattern: `http://|https://`**
  - Line 51: `"""See http://www.codinghorror.com/blog/archives/001018.html"""`
  - Line 271: `# lock image tower as per LiT - https://arxiv.org/abs/2111.07991`

### PS3/train/src/open_clip_train/params.py
**Pattern: `http://|https://`**
  - Line 6: `# Params from paper (https://arxiv.org/pdf/2103.00020.pdf)`

### PS3/train/src/open_clip_train/data.py
**Pattern: `http://|https://`**
  - Line 590: `# hoping to resolve via https://github.com/webdataset/webdataset/issues/169`

### PS3/train/src/open_clip/zero_shot_metadata.py
**Pattern: `http://|https://`**
  - Line 87: `# from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb`

### PS3/train/src/open_clip/pos_embed.py
**Pattern: `http://|https://`**
  - Line 17: `# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py`
  - Line 18: `# MoCo v3: https://github.com/facebookresearch/moco-v3`
  - Line 73: `# DeiT: https://github.com/facebookresearch/deit`

### PS3/train/src/open_clip/loss.py
**Pattern: `http://|https://`**
  - Line 315: `""" Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343`

### PS3/train/src/open_clip/transformer.py
**Pattern: `http://|https://`**
  - Line 50: `https://arxiv.org/abs/2212.00794`
  - Line 411: `# TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372`
  - Line 501: `# TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372`
  - Line 1191: `# TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372`

### PS3/train/src/open_clip/hf_configs.py
**Pattern: `http://|https://`**
  - Line 3: `# https://huggingface.co/docs/transformers/model_doc/roberta#roberta`
  - Line 16: `# https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.XLMRobertaConfig`
  - Line 29: `# https://huggingface.co/docs/transformers/model_doc/mt5#mt5`
  - Line 33: `# https://github.com/google-research/text-to-text-transfer-transformer/issues/273`
  - Line 34: `# https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/t5/modeling_t5.py#L374`
  - ... and 2 more matches

### PS3/train/src/open_clip/pretrained.py
**Pattern: `http://|https://`**
  - Line 95: `url="https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",`
  - Line 100: `url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt",`
  - Line 105: `url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt",`
  - Line 113: `url="https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",`
  - Line 118: `url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt",`
  - ... and 25 more matches

### PS3/train/src/open_clip/coca_model.py
**Pattern: `http://|https://`**
  - Line 307: `# https://huggingface.co/docs/transformers/main/en/main_classes/text_generation`

### PS3/train/src/open_clip/timm_model.py
**Pattern: `http://|https://`**
  - Line 4: `Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.`
  - Line 124: `'Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`')`

### PS3/train/src/open_clip/model.py
**Pattern: `http://|https://`**
  - Line 3: `Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.`
  - Line 269: `# lock image tower as per LiT - https://arxiv.org/abs/2111.07991`
  - Line 476: `# lock image tower as per LiT - https://arxiv.org/abs/2111.07991`

### PS3/train/src/open_clip/hf_model.py
**Pattern: `http://|https://`**
  - Line 3: `Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.`

### PS3/train/src/open_clip/factory.py
**Pattern: `http://|https://`**
  - Line 207: `# If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712`

### PS3/train/src/open_clip/utils.py
**Pattern: `http://|https://`**
  - Line 25: `Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762`

### PS3/train/src/open_clip/tokenizer.py
**Pattern: `http://|https://`**
  - Line 3: `Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.`
  - Line 19: `# https://stackoverflow.com/q/62691279`
  - Line 112: `From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94`
  - Line 472: `"c4-en": "http://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model",`
  - Line 474: `"mc4": "http://storage.googleapis.com/t5-data/vocabs/mc4.250000.100extra/sentencepiece.model",`
  - ... and 1 more matches

### PS3/train/src/open_clip/openai.py
**Pattern: `http://|https://`**
  - Line 3: `Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.`

### PS3/train/src/open_clip/push_to_hf_hub.py
**Pattern: `http://|https://`**
  - Line 100: `# default CLIP tokenizers use https://huggingface.co/openai/clip-vit-large-patch14`

### PS3/ps3/modeling_ps3.py
**Pattern: `http://|https://`**
  - Line 8: `# http://www.apache.org/licenses/LICENSE-2.0`
  - Line 116: `raise ImportError("Please import the create_mlp_from_config function from https://github.com/NVlabs/RADIO/blob/main/radio/adaptor_mlp.py.")`
  - Line 927: `Adapted from timm ConvNeXt: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py`
  - Line 935: `Ref: https://github.com/huggingface/transformers/issues/29554`
  - Line 1005: `Adapted from timm PatchEmbed: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py`
  - ... and 7 more matches

### PS3/ps3/image_processing_ps3.py
**Pattern: `http://|https://`**
  - Line 8: `# http://www.apache.org/licenses/LICENSE-2.0`

### PS3/ps3/modeling_ps3_text.py
**Pattern: `http://|https://`**
  - Line 8: `# http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py`
  - Line 17: `Originally license: https://github.com/mlfoundations/open_clip/blob/main/LICENSE.`
  - Line 149: `https://arxiv.org/abs/2212.00794`
  - Line 408: `# TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372`

### PS3/ps3/configuration_ps3.py
**Pattern: `http://|https://`**
  - Line 8: `# http://www.apache.org/licenses/LICENSE-2.0`

### PS3/ps3/tokenization_ps3.py
**Pattern: `http://|https://`**
  - Line 8: `# http://www.apache.org/licenses/LICENSE-2.0`
  - Line 16: `Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py.`
  - Line 17: `Originally license: https://github.com/mlfoundations/open_clip/blob/main/LICENSE.`
  - Line 35: `# https://stackoverflow.com/q/62691279`
  - Line 87: `From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94`

## ✅ Clean ROS2 Files
These files are already ROS2-compatible:

- ✅ robot_gui_ros2.py
- ✅ launch_ros2_system.py
- ✅ ros2_command_gateway_validator.py
- ✅ robot_client_ros2.py
- ✅ vila_ros2_node.py
- ✅ robot_vila_server_ros2.py
- ✅ robot_launcher.py
- ✅ main_vila.py
- ✅ VILA/llava/__init__.py
- ✅ VILA/llava/utils/distributed.py
- ✅ VILA/llava/utils/logging.py
- ✅ VILA/llava/utils/io.py
- ✅ VILA/llava/utils/__init__.py
- ✅ VILA/llava/cli/run.py
- ✅ VILA/llava/cli/eval.py
- ✅ VILA/llava/cli/infer.py
- ✅ VILA/llava/train/__init__.py
- ✅ VILA/llava/train/callbacks/autoresume_callback.py
- ✅ VILA/llava/train/deepspeed_replace/runtime/zero/mics.py
- ✅ VILA/llava/train/sequence_parallel/__init__.py
- ✅ VILA/llava/trl/__init__.py
- ✅ VILA/llava/trl/environment/__init__.py
- ✅ VILA/llava/eval/pmcvqa.py
- ✅ VILA/llava/eval/pathvqa.py
- ✅ VILA/llava/eval/eval_model_loc.py
- ✅ VILA/llava/eval/aitz.py
- ✅ VILA/llava/eval/its.py
- ✅ VILA/llava/eval/eventbench.py
- ✅ VILA/llava/eval/textvqa.py
- ✅ VILA/llava/eval/widgetcap.py
- ✅ VILA/llava/eval/alfred.py
- ✅ VILA/llava/eval/mathvista.py
- ✅ VILA/llava/eval/refcoco.py
- ✅ VILA/llava/eval/model_refcoco.py
- ✅ VILA/llava/eval/egoschema.py
- ✅ VILA/llava/eval/domain.py
- ✅ VILA/llava/eval/tallyqa.py
- ✅ VILA/llava/eval/nuscenes.py
- ✅ VILA/llava/eval/cinepile.py
- ✅ VILA/llava/eval/eval_refcoco.py
- ✅ VILA/llava/eval/rtl.py
- ✅ VILA/llava/eval/rtl_nv_lita.py
- ✅ VILA/llava/eval/vnbench.py
- ✅ VILA/llava/eval/scicap.py
- ✅ VILA/llava/eval/__init__.py
- ✅ VILA/llava/eval/mathvista_utils/extract_answer.py
- ✅ VILA/llava/eval/mathvista_utils/calculate_score.py
- ✅ VILA/llava/eval/mathvista_utils/utilities.py
- ✅ VILA/llava/eval/mathvista_utils/prompts/ext_ans.py
- ✅ VILA/llava/eval/video/eval_benchmark_3_context.py
- ✅ VILA/llava/eval/video/eval_benchmark_4_temporal.py
- ✅ VILA/llava/eval/video/eval_benchmark_5_consistency.py
- ✅ VILA/llava/eval/video/eval_benchmark_2_detailed_orientation.py
- ✅ VILA/llava/eval/video/eval_benchmark_1_correctness.py
- ✅ VILA/llava/eval/finetuning/eval_aitz.py
- ✅ VILA/llava/eval/finetuning/eval_pathvqa.py
- ✅ VILA/llava/eval/finetuning/model_refcoco.py
- ✅ VILA/llava/eval/finetuning/eval_tallyqa.py
- ✅ VILA/llava/eval/finetuning/eval_multi_view.py
- ✅ VILA/llava/eval/finetuning/model_widgetcap.py
- ✅ VILA/llava/eval/finetuning/eval_scicap.py
- ✅ VILA/llava/eval/finetuning/eval_widgetcap.py
- ✅ VILA/llava/eval/finetuning/model_vqa_loader_multi.py
- ✅ VILA/llava/eval/finetuning/model_vqa_loader.py
- ✅ VILA/llava/eval/finetuning/eval_pmcvqa.py
- ✅ VILA/llava/eval/lmms/tasks/videomme.py
- ✅ VILA/llava/eval/lmms/tasks/__init__.py
- ✅ VILA/llava/eval/lmms/models/__init__.py
- ✅ VILA/llava/data/base.py
- ✅ VILA/llava/data/collate.py
- ✅ VILA/llava/data/__init__.py
- ✅ VILA/llava/data/builder.py
- ✅ VILA/llava/data/dataset_impl/lita.py
- ✅ VILA/llava/data/dataset_impl/eagle_video_wds.py
- ✅ VILA/llava/data/dataset_impl/llava.py
- ✅ VILA/llava/data/dataset_impl/llava_cot.py
- ✅ VILA/llava/data/dataset_impl/eagle_wds.py
- ✅ VILA/llava/data/dataset_impl/utils.py
- ✅ VILA/llava/data/dataset_impl/__init__.py
- ✅ VILA/llava/data/dataset_impl/dummy.py
- ✅ VILA/llava/data/dataset_impl/coyo_qa.py
- ✅ VILA/llava/model/FloatPointQuantizeTriton.py
- ✅ VILA/llava/model/loss.py
- ✅ VILA/llava/model/qfunction.py
- ✅ VILA/llava/model/FloatPointQuantizeTorch.py
- ✅ VILA/llava/model/qutils.py
- ✅ VILA/llava/model/qlinear_te.py
- ✅ VILA/llava/model/__init__.py
- ✅ VILA/llava/model/encoders/base.py
- ✅ VILA/llava/model/encoders/__init__.py
- ✅ VILA/llava/model/encoders/video/basic.py
- ✅ VILA/llava/model/encoders/video/tsp.py
- ✅ VILA/llava/model/encoders/video/__init__.py
- ✅ VILA/llava/model/encoders/image/basic.py
- ✅ VILA/llava/model/encoders/image/__init__.py
- ✅ VILA/llava/model/realquantize/division_transpose.py
- ✅ VILA/llava/model/realquantize/linear.py
- ✅ VILA/llava/model/realquantize/trans_grad_bias.py
- ✅ VILA/llava/model/realquantize/quantize_and_transpose.py
- ✅ VILA/llava/model/realquantize/division.py
- ✅ VILA/llava/model/realquantize/common.py
- ✅ VILA/llava/model/quantization/QLayerNorm.py
- ✅ VILA/llava/model/quantization/QFunction.py
- ✅ VILA/llava/model/quantization/FloatPointQuantizeTriton.py
- ✅ VILA/llava/model/quantization/FloatPointQuantizeTorch.py
- ✅ VILA/llava/model/quantization/QIdentity.py
- ✅ VILA/llava/model/quantization/Qconfig.py
- ✅ VILA/llava/model/quantization/QAdd.py
- ✅ VILA/llava/model/quantization/QAct.py
- ✅ VILA/llava/model/quantization/QGELU.py
- ✅ VILA/llava/model/quantization/QLinear.py
- ✅ VILA/llava/model/quantization/QMul.py
- ✅ VILA/llava/model/quantization/utils.py
- ✅ VILA/llava/model/quantization/__init__.py
- ✅ VILA/llava/model/utils/packing.py
- ✅ VILA/llava/model/utils/__init__.py
- ✅ VILA/llava/model/coat/fp8_trainer.py
- ✅ VILA/llava/model/coat/activation/__init__.py
- ✅ VILA/llava/model/coat/activation/real_quantization/__init__.py
- ✅ VILA/llava/model/coat/activation/models/_fp8_quantization_config.py
- ✅ VILA/llava/model/coat/activation/models/_fp8_weightcache.py
- ✅ VILA/llava/model/coat/activation/models/coat_llama_convert_from_hf.py
- ✅ VILA/llava/model/multimodal_encoder/siglip/__init__.py
- ✅ VILA/llava/model/multimodal_encoder/intern/configuration_intern_vit.py
- ✅ VILA/llava/model/language_model/fp8_qwen2_convert_from_hf.py
- ✅ VILA/llava/model/language_model/configuration_quantize.py
- ✅ VILA/llava/remote_code/distributed.py
- ✅ VILA/llava/remote_code/loss.py
- ✅ VILA/llava/remote_code/media_encoder.py
- ✅ VILA/llava/remote_code/modeling_vila.py
- ✅ VILA/llava/remote_code/model_utils_packing.py
- ✅ VILA/llava/remote_code/configuration_vila.py
- ✅ VILA/llava/wids/wids_tar.py
- ✅ VILA/llava/wids/wids_bench.py
- ✅ VILA/llava/wids/wids_lru.py
- ✅ VILA/llava/wids/wids_index.py
- ✅ VILA/llava/wids/wids_cleanup.py
- ✅ VILA/llava/wids/wids_dl.py
- ✅ VILA/llava/wids/__init__.py
- ✅ VILA/llava/wids/wids_dir.py
- ✅ VILA/llava/wids/wids_mmtar.py
- ✅ VILA/data_prepare/sft/SROIE.py
- ✅ VILA/data_prepare/sft/preprocess_kvqa.py
- ✅ VILA/data_prepare/sft/POIE.py
- ✅ VILA/data_prepare/sft/ART1_2.py
- ✅ VILA/data_prepare/sft/preprocess_docreason.py
- ✅ VILA/data_prepare/sft/ESTVQA.py
- ✅ VILA/data_prepare/sft/unichart_pretrain.py
- ✅ VILA/data_prepare/sft/preprocess_metamathqa.py
- ✅ VILA/data_prepare/sft/preprocess_viquae.py
- ✅ VILA/data_prepare/sft/LSVT.py
- ✅ VILA/data_prepare/sft/preprocess_cambrian.py
- ✅ VILA/data_prepare/sft/ReCTS.py
- ✅ VILA/data_prepare/sft/unichart_sft.py
- ✅ VILA/data_prepare/sft/preprocess_llava_onevision.py
- ✅ VILA/data_prepare/sft/preprocess_art_shangy.py
- ✅ VILA/data_prepare/sft/preprocess_cambrian_eagle.py
- ✅ VILA/data_prepare/sft/mtwi.py
- ✅ VILA/data_prepare/sft/preprocess_idefics2_eagle.py
- ✅ PS3/train/src/open_clip_train/zero_shot.py
- ✅ PS3/train/src/open_clip_train/distributed.py
- ✅ PS3/train/src/open_clip_train/train.py
- ✅ PS3/train/src/open_clip_train/file_utils.py
- ✅ PS3/train/src/open_clip_train/profiler.py
- ✅ PS3/train/src/open_clip_train/precision.py
- ✅ PS3/train/src/open_clip_train/scheduler.py
- ✅ PS3/train/src/open_clip_train/logger.py
- ✅ PS3/train/src/open_clip_train/__init__.py
- ✅ PS3/train/src/open_clip/modified_resnet.py
- ✅ PS3/train/src/open_clip/transform.py
- ✅ PS3/train/src/open_clip/zero_shot_classifier.py
- ✅ PS3/train/src/open_clip/constants.py
- ✅ PS3/train/src/open_clip/version.py
- ✅ PS3/train/src/open_clip/save_ps3_hf_ckpt.py
- ✅ PS3/train/src/open_clip/convert.py
- ✅ PS3/train/src/open_clip/__init__.py
- ✅ PS3/ps3/utils_radio_adapter_mlp.py
- ✅ PS3/ps3/__init__.py

## 🧹 Cleanup Recommendations

### 1. Remove Deprecated Files
```bash
# mv robot_client_examples.py deprecated/  # or rm robot_client_examples.py
# mv robot_vila_server.py deprecated/  # or rm robot_vila_server.py
# mv unified_robot_controller.py deprecated/  # or rm unified_robot_controller.py
# mv unified_robot_client.py deprecated/  # or rm unified_robot_client.py
# mv robot_gui.py deprecated/  # or rm robot_gui.py
# mv vila_ros_node.py deprecated/  # or rm vila_ros_node.py
```

### 2. Update Requirements
Remove HTTP dependencies from requirements.txt:
```
# Remove these packages:
# requests
# flask
# flask-socketio
# python-socketio
# aiohttp
# websockets
```

### 3. Use ROS2 Requirements
```bash
pip install -r requirements_ros2.txt
```
