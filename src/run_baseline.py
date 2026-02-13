import os
import json
import sys
from datasets import load_from_disk
from tqdm import tqdm
from models import Model
from util import DATASET_CONFIGS, get_video_path

# --- Configuration ---
if len(sys.argv) < 3:
    print("Usage: python run_baseline.py <dataset_name> <model_name>")
    sys.exit(1)

data_name = sys.argv[1]
MODEL_NAME = sys.argv[2]

# Path Setup
if data_name not in DATASET_CONFIGS:
    raise ValueError(f"Unknown dataset: {data_name}. Available: {list(DATASET_CONFIGS.keys())}")

DATA = data_name
config = DATASET_CONFIGS[data_name]
DATA_DIR = config["hf_path"]
VIDEO_DIR = config["video_dir"]

REASONING_PROMPT_PATH = "prompts/base.txt"
model_name_str = MODEL_NAME.replace("-", "_")
OUTPUT_DIR = f"outputs/{DATA}_{model_name_str}_baseline.json"
MAX_NUM = 100

def get_unique_key(item):
    """Create unique identifier for tracking processed items."""
    return (item["video"], item["question"])

if __name__ == '__main__':
    dataset = load_from_disk(DATA_DIR)
    limit = min(MAX_NUM, len(dataset))
    
    with open(REASONING_PROMPT_PATH, "r") as f:
        reasoning_prompt = f.read()
    
    model = Model(MODEL_NAME)

    processed_map = {}
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

    if os.path.exists(OUTPUT_DIR):
        try:
            with open(OUTPUT_DIR, "r") as f:
                existing_data = json.load(f)
                
            # Create a lookup dictionary: Key -> Entry
            for entry in existing_data:
                if "video" in entry and "question" in entry:
                    key = get_unique_key(entry)
                    processed_map[key] = entry
            
            print(f"Found existing output with {len(processed_map)} valid entries.")
            
        except json.JSONDecodeError:
            print("Output file exists but is empty or corrupted. Starting from scratch.")
            processed_map = {}

    # This list will hold the final ordered data
    final_ordered_responses = []

    for i in tqdm(range(limit), total=limit, desc="Processing"):
        item = dataset[i]
        unique_key = get_unique_key(item)
        
        # --- CHECK: If item is already processed, use the existing result ---
        if unique_key in processed_map:
            final_ordered_responses.append(processed_map[unique_key])
            continue
            
        entry = dict(item)
        question = item["question"]
        original_options = item["options"]
        video_filename = item["video"]
        
        # Format inputs
        video_path = get_video_path(DATA, video_filename)
        if not video_path:
            print(f"Warning: Could not find video path for {video_filename}")
            continue
            
        formatted_options = "\n".join([f"{chr(65 + j)}: {opt}" for j, opt in enumerate(original_options)])
        
        input_dict = {
            "question": question,
            "options": formatted_options,
        }
        
        prompt = reasoning_prompt.format_map(input_dict)

        # Prepare Messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Generate and Save
        try:
            response = model.generate_response(messages)
            
            entry["model_generation"] = response 
            
            # Add to our ordered list and the map (to prevent duplicate processing if logic changes)
            final_ordered_responses.append(entry)
            processed_map[unique_key] = entry

            with open(OUTPUT_DIR, "w") as f:
                json.dump(final_ordered_responses, f, indent=4)
                
        except Exception as e:
            print(f"\nError processing index {i}: {e}")
    with open(OUTPUT_DIR, "w") as f:
        json.dump(final_ordered_responses, f, indent=4)

    # Final check to print status
    print(f"Finished. Total items in output: {len(final_ordered_responses)}/{limit}")