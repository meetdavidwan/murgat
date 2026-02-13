import os
import json
import sys
from datasets import load_from_disk
from tqdm import tqdm
from models import Model
from util import DATASET_CONFIGS, get_video_path
from program_utils import parse_program_output, execute_program, generate_program_plan

# --- Configuration ---
if len(sys.argv) < 3:
    print("Usage: python run_generation_program.py <dataset_name> <model_name>")
    sys.exit(1)

data_name = sys.argv[1]
MODEL_NAME = sys.argv[2]

if data_name not in DATASET_CONFIGS:
    raise ValueError(f"Unknown dataset: {data_name}. Available: {list(DATASET_CONFIGS.keys())}")

DATA = data_name
config = DATASET_CONFIGS[data_name]
DATA_DIR = config["hf_path"]

# Path to the new GenerationProgram prompt
PROMPT_PATH = "prompts/video_generation_program.txt" 
model_name_str = MODEL_NAME.replace("-", "_")
OUTPUT_DIR = f"outputs/{DATA}_{model_name_str}_program_with_feedback.json"
TEMP_CLIP_DIR = "temp_clips" # Directory for ffmpeg cuts

MAX_NUM = 1

if __name__ == '__main__':
    # 1. Setup Resources
    dataset = load_from_disk(DATA_DIR)
    limit = min(MAX_NUM, len(dataset))
    
    with open(PROMPT_PATH, "r") as f:
        program_prompt_template = f.read()
    
    model = Model(MODEL_NAME)
    
    # Clean up temp dir if exists
    if not os.path.exists(TEMP_CLIP_DIR):
        os.makedirs(TEMP_CLIP_DIR)

    # 2. Resume Logic
    responses = []
    start_index = 0
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    if os.path.exists(OUTPUT_DIR):
        try:
            with open(OUTPUT_DIR, "r") as f:
                responses = json.load(f)
                start_index = len(responses)
                print(f"Resuming from {start_index}...")
        except:
            responses = []

    # 3. Processing Loop
    for i in tqdm(range(start_index, limit), initial=start_index, total=limit):
        item = dataset[i]
        entry = dict(item)
        
        question = item["question"]
        original_options = item["options"]
        video_filename = item["video"]
        video_path = get_video_path(DATA, video_filename)
        
        if not video_path or not os.path.exists(video_path):
            print(f"Skipping {video_filename}: Not found")
            entry["program_error"] = "Video not found"
            responses.append(entry)
            continue

        formatted_options = "\n".join([f"{chr(65 + j)}: {opt}" for j, opt in enumerate(original_options)])
        
        # --- PHASE 1: Generate Program ---
        try:
            plan_text = generate_program_plan(
                model=model,
                video_path=video_path,
                question=question,
                formatted_options=formatted_options,
                prompt_template=program_prompt_template
            )
            
            entry["generated_plan"] = plan_text
            print(f"\n[Generated Plan]:\n{plan_text}")
            
            # --- PHASE 2: Parse Program ---
            parsed_calls = parse_program_output(plan_text)
            entry["parsed_program"] = [p["raw"] for p in parsed_calls]
            
            if not parsed_calls:
                print("Error: Could not parse program.")
                entry["final_answer"] = "Error: Could not parse program."
            else:
                # --- PHASE 3: Execute Program ---
                execution_trace = execute_program(
                    model, 
                    parsed_calls, 
                    video_path,
                    question,
                    use_feedback=True,
                )
                
                entry["execution_trace"] = execution_trace
                entry["model_generation"] = " ".join([ans["final_output"] for ans in execution_trace])
            
            responses.append(entry)
            
            with open(OUTPUT_DIR, "w") as f:
                json.dump(responses, f, indent=4)
                
        except Exception as e:
            print(f"Error at index {i}: {e}")
            entry["error"] = str(e)
            responses.append(entry)
            with open(OUTPUT_DIR, "w") as f:
                json.dump(responses, f, indent=4)