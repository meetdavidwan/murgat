import os
import time
import yaml
import base64
import torch
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import json
import librosa
import numpy as np
import concurrent.futures

from google import genai 
from google.genai import types
from openai import OpenAI

# Transformers / Model Specific Imports
from transformers import (
    Qwen3OmniMoeProcessor, 
    AutoTokenizer, 
    AutoProcessor, 
    Gemma3ForConditionalGeneration,
    AutoModelForImageTextToText
)

try:
    from qwen_omni_utils import process_mm_info
    # vLLM and Omni imports
    from vllm import SamplingParams
    from vllm_omni.entrypoints.omni import Omni
    from vllm.multimodal.image import convert_image_mode
    from vllm.assets.video import video_to_ndarrays
except ImportError:
    process_mm_info = None 

try:
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
except ImportError:
    process_vision_info = None

# ==========================================
# Model Classes
# ==========================================

class Model:
    def __init__(self, model_name: str, thinking_level: Optional[str] = None):
        modelname2model = {
            "qwen3-omni-instruct": Qwen3OmniInstructModel,
            "qwen3-omni-thinking": Qwen3OmniThinkingModel,
            "qwen3-vl-instruct": Qwen3VLInstructModel,
            "qwen3-vl-thinking": Qwen3VLThinkingModel,
            "gemma-3-27b-it": Gemma37bItModel,
            "gemini-2.5-flash": Gemini25FlashModel,
            "gemini-3-flash": Gemini3FlashModel,
            "gemini-2.5-pro": Gemini25ProModel,
            "gemini-3-pro": Gemini3ProModel,
            "gpt-5": OpenAIModel,
            "molmo2-8b": Molmo2Model,
        }
        
        if model_name not in modelname2model:
            raise ValueError(f"Unknown model name: {model_name}.")
        
        if model_name in ["gemini-3-flash", "gemini-3-pro"]:
            self.model_instance = modelname2model[model_name](thinking_level=thinking_level)
        else:
            self.model_instance = modelname2model[model_name]()

    def generate_response(self, messages: List[Dict[str, Any]], num_samples: int = 1, fps: float = None):
        """
        Generate response with optional dynamic fps control.
        fps: If provided, attempts to sample video at this rate.
        """
        return self.model_instance.generate_response(messages=messages, num_samples=num_samples, fps=fps)

class Molmo2Model:
    def __init__(self, model_path="allenai/Molmo2-8B"):
        print(f"Loading {model_path} with Transformers...")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def generate_response(self, messages: List[Dict[str, Any]], num_samples: int = 1, fps: float = None):
        # 1. Format messages for Molmo processor
        # Molmo expects dicts like {"type": "video", "video": path_or_url}
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            new_content = []
            
            if isinstance(content, str):
                new_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        itype = item.get("type")
                        if itype == "text":
                            new_content.append(item)
                        elif itype == "image":
                            # Handle Image: Pass PIL object if local, or path string if preferred
                            path = item.get("image") or item.get("media_path")
                            if path:
                                # For Molmo2, passing the PIL image object is often safer for local files
                                if os.path.exists(path):
                                    new_content.append({"type": "image", "image": Image.open(path).convert("RGB")})
                                else:
                                    # Fallback to passing path string
                                    new_content.append({"type": "image", "image": path})
                        elif itype == "video":
                            # Handle Video: Pass path string (processor handles loading)
                            path = item.get("video") or item.get("media_path")
                            if path:
                                new_content.append({"type": "video", "video": path})
            
            formatted_messages.append({"role": role, "content": new_content})

        # 2. Process inputs
        inputs = self.processor.apply_chat_template(
            formatted_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Move inputs to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 3. Generate
        do_sample = True if num_samples > 1 else False
        temperature = 0.7 if num_samples > 1 else 1.0

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=2048,
                do_sample=do_sample,
                temperature=temperature,
                num_return_sequences=num_samples
            )

        # 4. Decode
        # We must slice off the input tokens to get only the new tokens
        input_len = inputs['input_ids'].size(1)
        decoded_texts = []
        
        for i in range(generated_ids.shape[0]):
            output_ids = generated_ids[i, input_len:]
            text = self.processor.tokenizer.decode(output_ids, skip_special_tokens=True)
            decoded_texts.append(text)

        if num_samples == 1:
            return decoded_texts[0]
        return decoded_texts

class Gemma37bItModel:
    def __init__(self, model_path="google/gemma-3-27b-it"):
        print(f"Loading {model_path} with Transformers...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()

    def _prepare_messages_with_images(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process message content to load images from disk."""
        processed_messages = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            new_content = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image":
                            # Load image
                            path = item.get("image", "")
                            if path and os.path.exists(path):
                                new_content.append({"type": "image", "image": Image.open(path).convert("RGB")})
                            else: new_content.append(item)
                        else: new_content.append(item)
            elif isinstance(content, str):
                new_content.append({"type": "text", "text": content})
            processed_messages.append({"role": role, "content": new_content})
        return processed_messages

    def generate_response(self, messages: List[Dict[str, Any]], num_samples: int = 1, fps: float = None):
        final_messages = self._prepare_messages_with_images(messages)
        inputs = self.processor.apply_chat_template(
            final_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        
        # Decide sampling strategy based on num_samples
        do_sample = True if num_samples > 1 else False
        temperature = 0.7 if num_samples > 1 else 0.0

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=2048, 
                do_sample=do_sample,
                temperature=temperature,
                num_return_sequences=num_samples
            )
        
        # Decode all sequences
        decoded_texts = []
        for seq in generation:
            # Slice prompt and decode
            response_tokens = seq[input_len:]
            decoded_texts.append(self.processor.decode(response_tokens, skip_special_tokens=True))

        if num_samples == 1:
            return decoded_texts[0]
        return decoded_texts


class Qwen3OmniInstructModel:
    def __init__(self, model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct", stage_configs_path="src/qwen3omniconfig.yaml", seed=42):
        self.model_path = model_path
        self.output_dir = "output_omni"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize the Omni engine
        self.omni_llm = Omni(
            model=model_path,
            stage_configs_path=stage_configs_path,
        )

        # --- Define Sampling Parameters for each stage ---
        # 1. Thinker: Generates the reasoning/text response
        self.sampling_params = SamplingParams(
            temperature=0.6, top_p=0.95, top_k=20, 
            max_tokens=16384, seed=1234
        )

        self.default_system = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
            "Group, capable of perceiving auditory and visual inputs, as well as "
            "generating text and speech."
        )

    def _load_media(self, media_path: str, media_type: str, **kwargs):
        """Helper to load media files into vLLM compatible formats."""
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"{media_type} file not found: {media_path}")

        if media_type == 'image':
            pil_image = Image.open(media_path)
            return convert_image_mode(pil_image, "RGB")
        
        elif media_type == 'video':
            num_frames = kwargs.get('num_frames', 16)
            return video_to_ndarrays(media_path, num_frames=num_frames)
        
        elif media_type == 'audio':
            sampling_rate = kwargs.get('sampling_rate', 16000)
            audio_signal, sr = librosa.load(media_path, sr=sampling_rate)
            return (audio_signal.astype(np.float32), sr)
            
        return None

    def _build_prompt(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Converts a list of messages into the Qwen3-Omni prompt format.
        Expected message format: 
        {'role': 'user', 'content': 'text...', 'image': 'path', 'video': 'path', 'audio': 'path'}
        """
        # Construct the raw prompt string with special tokens
        prompt_str = f"<|im_start|>system\n{self.default_system}<|im_end|>\n"
        
        multi_modal_data = {}
        limit_mm_per_prompt = {}
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            prompt_str += f"<|im_start|>{role}\n"
            
            # Handle Audio
            if 'audio' in msg:
                prompt_str += "<|audio_start|><|audio_pad|><|audio_end|>"
                multi_modal_data['audio'] = self._load_media(msg['audio'], 'audio')
                limit_mm_per_prompt['audio'] = limit_mm_per_prompt.get('audio', 0) + 1
            
            # Handle Image
            if 'image' in msg:
                prompt_str += "<|vision_start|><|image_pad|><|vision_end|>"
                multi_modal_data['image'] = self._load_media(msg['image'], 'image')
                limit_mm_per_prompt['image'] = limit_mm_per_prompt.get('image', 0) + 1

            # Handle Video
            if 'video' in msg:
                prompt_str += "<|vision_start|><|video_pad|><|vision_end|>"
                multi_modal_data['video'] = self._load_media(msg['video'], 'video')
                limit_mm_per_prompt['video'] = limit_mm_per_prompt.get('video', 0) + 1
            
            prompt_str += f"{content}<|im_end|>\n"

        prompt_str += "<|im_start|>assistant\n"

        return {
            "inputs": {
                "prompt": prompt_str,
                "multi_modal_data": multi_modal_data,
                "mm_processor_kwargs": {"use_audio_in_video": True} if 'video' in limit_mm_per_prompt else {}
            },
            "limit_mm_per_prompt": limit_mm_per_prompt
        }

    def generate_response(self, messages: List[Dict[str, Any]], num_samples: int = 1, fps: float = None):
        """
        Generates a text response using the simplified single-stage pipeline.
        """
        # 1. Prepare Inputs
        query_data = self._build_prompt(messages)
        inputs = query_data["inputs"]
        
        # Ensure only 'text' is requested in modalities
        inputs["modalities"] = ["text"]

        # 2. Run Generation
        omni_generator = self.omni_llm.generate([inputs], [self.sampling_params])

        final_text = ""
        for stage_outputs in omni_generator:
            if stage_outputs.final_output_type == "text":
                for output in stage_outputs.request_output:
                    final_text = output.outputs[0].text
        
        return final_text

class Qwen3OmniThinkingModel(Qwen3OmniInstructModel):
    def __init__(self, model_path="Qwen/Qwen3-Omni-30B-A3B-Thinking"):
        super().__init__(model_path=model_path)


class Qwen3VLInstructModel:
    def __init__(self, model_path="Qwen/Qwen3-VL-30B-A3B-Instruct"):
        os.environ['DECORD_EOF_RETRY_MAX'] = '20480'
        print(f"[{time.strftime('%X')}] Initializing Qwen3-VL: {model_path}...")
        
        # vLLM setup for VL models
        self.llm = LLM(
            model=model_path,
            mm_encoder_tp_mode="data",
            enable_expert_parallel=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=32768*4,
            # limit_mm_per_prompt={'image': 10, 'video': 2},
            enforce_eager=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print(f"[{time.strftime('%X')}] Qwen3-VL Loaded Successfully.")

    def generate_response(self, messages: List[Dict[str, Any]], num_samples: int = 1, fps: float = None):
        if process_vision_info is None:
            raise ImportError("Please install qwen-vl-utils: pip install qwen-vl-utils")

        # 1. Apply Chat Template
        prompt_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 2. Process Vision Info (Images/Videos)
        # VL models use patch_size for coordinate/pixel calculations
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        mm_data = {}
        if image_inputs is not None: mm_data['image'] = image_inputs
        if video_inputs is not None: mm_data['video'] = video_inputs

        # 3. Construct vLLM Input
        inputs = {
            'prompt': prompt_text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }

        # 4. Sampling
        sampling_params = SamplingParams(
            temperature=0.7 if num_samples > 1 else 0.0,
            top_p=0.95,
            max_tokens=16384,
            n=num_samples
        )

        outputs = self.llm.generate([inputs], sampling_params=sampling_params)
        generated_texts = [out.text for out in outputs[0].outputs]
        
        return generated_texts[0] if num_samples == 1 else generated_texts

class Qwen3VLThinkingModel(Qwen3VLInstructModel):
    def __init__(self, model_path="Qwen/Qwen3-VL-30B-A3B-Thinking"):
        super().__init__(model_path=model_path)

class Gemini25FlashModel:
    def __init__(self, model_path="gemini-2.5-flash"):
        if "GEMINI_API_KEY" not in os.environ: 
            raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client()
        self.model_path = model_path
        # Cache maps absolute local path -> uploaded file object
        self._file_cache = {} 

    def _upload_file_if_needed(self, file_path: str):
        if not os.path.exists(file_path): return None
        abs_path = os.path.abspath(file_path)
        
        # 1. Check if we already have it in cache and it is valid
        if abs_path in self._file_cache:
            try:
                cached_file = self._file_cache[abs_path]
                # verify it still exists in cloud
                if self.client.files.get(name=cached_file.name).state == "ACTIVE":
                    return cached_file
            except: 
                # If check fails, remove from cache and proceed to re-upload
                del self._file_cache[abs_path]
        
        # 2. Upload new file
        print(f"Uploading {os.path.basename(file_path)} to Gemini...")
        uploaded_file = self.client.files.upload(file=file_path)
        
        # 3. Wait for processing (videos need this)
        while uploaded_file.state != "ACTIVE":
            if uploaded_file.state == "FAILED": 
                raise ValueError(f"File upload failed for {file_path}")
            time.sleep(1)
            uploaded_file = self.client.files.get(name=uploaded_file.name)
        
        # 4. Update Cache
        self._file_cache[abs_path] = uploaded_file 
        return uploaded_file
    
    def _get_generation_config(self):
        return types.GenerateContentConfig(candidate_count=1)

    def _generate_single(self, contents):
        """Helper to generate exactly one sample."""
        try:
            config = self._get_generation_config()
            
            response = self.client.models.generate_content(
                model=self.model_path,
                contents=contents,
                config=config
            )
            if hasattr(response, 'text') and response.text:
                return response.text
            if hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            return ""
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_response(self, messages, num_samples: int = 1, fps: float = None):
        files_to_delete = []
        paths_used = []

        try:
            all_parts = [] 
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str) and content.strip(): 
                    all_parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text": 
                                all_parts.append(item.get("text"))
                            elif item.get("type") in ["video", "image", "audio"]:
                                path_key = "media_path" if "media_path" in item else item.get("type")
                                file_path = item.get(path_key)
                                if file_path:
                                    f = self._upload_file_if_needed(file_path)
                                    if f: 
                                        all_parts.append(f)
                                        if f not in files_to_delete:
                                            files_to_delete.append(f)
                                        paths_used.append(os.path.abspath(file_path))

            if not all_parts: return "Error: No content"

            if num_samples == 1:
                return self._generate_single(all_parts)
            
            print(f"  [System] Fan-out: Generating {num_samples} samples in parallel...")
            candidates = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as executor:
                futures = [executor.submit(self._generate_single, all_parts) for _ in range(num_samples)]
                for future in concurrent.futures.as_completed(futures):
                    candidates.append(future.result())
            return candidates

        finally:
            if files_to_delete:
                # print("Cleaning up Gemini cloud files...") 
                for f in files_to_delete:
                    try:
                        self.client.files.delete(name=f.name)
                    except Exception: pass
                for path in paths_used:
                    if path in self._file_cache:
                        del self._file_cache[path]


class Gemini25ProModel(Gemini25FlashModel):
    def __init__(self, model_path="gemini-2.5-pro"):
        super().__init__(model_path=model_path)
    
    # Signature update
    def generate_response(self, messages, num_samples: int = 1, fps: float = None):
        return super().generate_response(messages, num_samples, fps)

class Gemini3FlashModel(Gemini25FlashModel):
    def __init__(self, model_path="gemini-3-flash-preview", thinking_level: str = None):
        super().__init__(model_path=model_path)
        self.thinking_level = thinking_level

    def _get_generation_config(self):
        config = types.GenerateContentConfig(candidate_count=1)
        if self.thinking_level:
            # Apply thinking config if level is provided
            config.thinking_config = types.ThinkingConfig(thinking_level=self.thinking_level)
        return config
    
    # Signature update
    def generate_response(self, messages, num_samples: int = 1, fps: float = None):
        return super().generate_response(messages, num_samples, fps)


class Gemini3ProModel(Gemini25ProModel):
    def __init__(self, model_path="gemini-3-pro-preview", thinking_level: str = None):
        super().__init__(model_path=model_path)
        self.thinking_level = thinking_level

    def _get_generation_config(self):
        config = types.GenerateContentConfig(candidate_count=1)
        if self.thinking_level:
            config.thinking_config = types.ThinkingConfig(thinking_level=self.thinking_level)
        return config
    
    # Signature update
    def generate_response(self, messages, num_samples: int = 1, fps: float = None):
        return super().generate_response(messages, num_samples, fps)


def adapt_messages_to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert messages to OpenAI-compatible format."""
    openai_messages = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            openai_content = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text = item.get("text", "")
                        if text.strip(): openai_content.append({"type": "text", "text": text})
                    elif item_type == "image":
                        image_path = item.get("image", "")
                        if image_path and os.path.exists(image_path):
                            with open(image_path, "rb") as image_file:
                                image_data = image_file.read()
                                base64_image = base64.b64encode(image_data).decode('utf-8')
                                ext = Path(image_path).suffix.lower()
                                mime_type = "image/jpeg" # simplified for brevity
                                openai_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
                    elif item_type in ["video", "audio"]:
                        fpath = item.get(item_type, "")
                        openai_content.append({"type": "text", "text": f"[{item_type} file: {os.path.basename(fpath)}]"})
            if openai_content:
                openai_messages.append({"role": role, "content": openai_content})
    return openai_messages

class OpenAIModel:
    def __init__(self, model_path="gpt-5.2-chat"):
        endpoint = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
        self.client = OpenAI(
            base_url=endpoint,
        )
        self.model_path = model_path

    def generate_response(self, messages, num_samples: int = 1):
        openai_messages = adapt_messages_to_openai(messages)
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=openai_messages,
            n=num_samples # Efficient parallel generation
        )
        
        texts = [choice.message.content for choice in response.choices]
        
        if num_samples == 1:
            return texts[0]
        return texts