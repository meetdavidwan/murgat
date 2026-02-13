import os
import json
import argparse
import re
import copy
import shutil
import subprocess
from tqdm import tqdm
from models import Model
import util

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_DECOMP = "gemini-3-flash"
MODEL_VERIFICATION = "gemini-3-pro"
MODEL_ENTAILMENT = "gemini-2.5-flash"

TEMP_SNIPPET_DIR = "./temp_snippets"

# --- VIDEO PATHS (Configure before running) ---
VIDEO_DIR_VIDEOMMMU = ""
VIDEO_DIR_WORLDSENSE = ""

PROMPT_FILES = {
    "decontextualization": "prompts/decontextualization.txt",
    "decomposition": "prompts/atomic_decomposition_sentence_level.txt",
    "verification_worthiness": "prompts/coverage_prompt_json_sentence.txt",
    "entailment": "prompts/entailment_prompt_simple.txt"
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_prompt_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, 'r') as f:
        return f.read()

def format_sentence_prompt(template, target_sentence, full_context=""):
    formatted = template
    for tag in ["{sent}", "{sentence}", "{target_sentence}", "{text}"]:
        if tag in formatted:
            formatted = formatted.replace(tag, target_sentence)
    for tag in ["{context}", "{full_paragraph}", "{full_text}"]:
        if tag in formatted:
            formatted = formatted.replace(tag, full_context)
    return formatted

def extract_prediction_json(raw_output):
    if not raw_output: return None
    try:
        clean_text = re.sub(r"```json\s*|\s*```", "", raw_output).strip()
        data = json.loads(clean_text)
        ans = data.get("label") or data.get("answer")
        if isinstance(ans, bool): return ans
        if isinstance(ans, str): return ans.lower() == "yes"
        return None
    except:
        clean = util.clean_model_output(raw_output).strip().lower()
        if clean == "yes": return True
        if clean == "no": return False
        return None

def extract_prediction_simple(raw_output):
    if not raw_output: return None
    clean = util.clean_model_output(raw_output).strip().lower()
    if clean == "yes": return True
    if clean == "no": return False
    if clean.endswith("yes") or "answer: yes" in clean: return True
    if clean.endswith("no") or "answer: no" in clean: return False
    return None

def construct_multimodal_message(prompt_text, source_paths):
    content = [{"type": "text", "text": prompt_text}]
    if isinstance(source_paths, str):
        source_paths = [source_paths]
    for path in source_paths:
        if os.path.exists(path):
            content.append({"type": "video", "media_path": path})
        else:
            print(f"Warning: Media path not found: {path}")
    return [{"role": "user", "content": content}]

def calculate_harmonic_mean(p, r):
    if (p + r) == 0: return 0.0
    return 2 * (p * r) / (p + r)

def determine_extraction_mode(modality, start, end):
    modality = modality.lower()
    if "audio" in modality:
        return "audio_only", "mp3"
    if start == end:
        return "image", "jpg"
    else:
        return "video_only", "mp4"

def extract_media_segment(source_path, output_path, start_time, end_time, mode):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True

    cmd = ['ffmpeg', '-y', '-ss', str(start_time)]
    cmd.extend(['-i', source_path])

    duration = end_time - start_time
    if duration <= 0: duration = 0.1 
    
    if mode == 'image':
        cmd.extend(['-vframes', '1', '-q:v', '2', output_path])
    elif mode == 'audio_only':
        cmd.extend(['-t', str(duration), '-vn', '-acodec', 'libmp3lame', '-q:a', '2', output_path])
    elif mode == 'video_only':
        cmd.extend(['-t', str(duration), '-c:v', 'libx264', '-an', output_path])
    else:
        return False

    try:
        # CHANGE 1: Capture the output (stderr) instead of sending it to NULL
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        # CHANGE 2: Decode and print the actual error message from FFmpeg
        error_message = e.stderr.decode('utf-8') if e.stderr else "No error message captured"
        print(f"  [FFMPEG Error] Command failed.")
        print(f"  [FFMPEG Output] {error_message.strip()}") # Prints the actual technical reason
        return False
    except Exception as e:
        print(f"  [General Error] {e}")
        return False

# ==========================================
# PIPELINE CLASS
# ==========================================

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.model_decomp = Model(MODEL_DECOMP)
        self.model_verification = Model(MODEL_VERIFICATION)
        self.model_entailment = Model(MODEL_ENTAILMENT)
        self.prompts = {k: load_prompt_text(v) for k, v in PROMPT_FILES.items()}

        self.temp_dir = f"./temp_snippets_{os.getpid()}"
        os.makedirs(self.temp_dir, exist_ok=True)

    def resolve_video_path(self, video_id):
        """
        Checks hardcoded paths for the video file. Raises error if not found.
        """
        if not video_id:
            raise ValueError("Video ID is missing.")

        # Ensure extension
        filename = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"

        # Path 1: VideoMMMU
        path1 = os.path.join(VIDEO_DIR_VIDEOMMMU, filename)
        if os.path.exists(path1):
            return path1

        # Path 2: WorldSense
        path2 = os.path.join(VIDEO_DIR_WORLDSENSE, filename)
        if os.path.exists(path2):
            return path2

        # If both fail
        raise FileNotFoundError(
            f"Video '{filename}' not found in:\n"
            f"  1. {VIDEO_DIR_VIDEOMMMU}\n"
            f"  2. {VIDEO_DIR_WORLDSENSE}"
        )

    def process_entry(self, entry, parse_reasoning=False):
        original_text = entry.get("model_generation") or entry.get("original_full_generation", "")
        if not original_text: return entry

        if parse_reasoning:
            original_text = util.extract_reasoning(util.clean_model_output(original_text))
        
        entry_id = entry.get('id', 'unknown')
        print(f"\n--- Processing Entry {entry_id} ---")

        # 1. Resolve Video Path (New Logic)
        try:
            video_source_path = self.resolve_video_path(entry.get("video", ""))
        except FileNotFoundError as e:
            # We raise this here so the main loop catches it and skips the entry
            raise e

        # 2. Decontextualization
        if "decontextualized_text" not in entry or not entry["decontextualized_text"]:
            if self.prompts["decontextualization"]:
                prompt = self.prompts["decontextualization"].replace("{{INPUT_TEXT}}", original_text)
                try:
                    raw_decontext = self.model_decomp.generate_response([{"role": "user", "content": prompt}])
                    entry["decontextualized_text"] = util.clean_model_output(raw_decontext)
                except Exception:
                    entry["decontextualized_text"] = original_text
            else:
                 entry["decontextualized_text"] = original_text

        text_to_process = entry["decontextualized_text"]
        if "evaluations" not in entry: entry["evaluations"] = {}
        
        sentences = util.split_text_into_sentences(text_to_process)

        for sent in sentences:
            if sent not in entry["evaluations"]:
                entry["evaluations"][sent] = {}
            sent_data = entry["evaluations"][sent]

            # 3. Verification Worthiness
            if "verification_worthy" not in sent_data:
                prompt = format_sentence_prompt(self.prompts["verification_worthiness"], sent, text_to_process)
                try:
                    raw_vw = self.model_verification.generate_response([{"role": "user", "content": prompt}])
                    pred_vw = extract_prediction_json(raw_vw)
                    sent_data["verification_worthy"] = {"prediction": pred_vw, "prediction_raw": raw_vw}
                except Exception as e:
                    sent_data["verification_worthy"] = {"prediction": False, "error": str(e)}

            if not sent_data.get("verification_worthy", {}).get("prediction", False):
                continue

            # 4. Atomic Fact Decomposition
            if "atomic_facts" not in sent_data:
                prompt = format_sentence_prompt(self.prompts["decomposition"], sent, text_to_process)
                try:
                    raw_decomp = self.model_decomp.generate_response([{"role": "user", "content": prompt}])
                    facts_list = util.parse_llm_list(raw_decomp)
                    sent_data["atomic_facts"] = [{"fact": f} for f in facts_list]
                except Exception:
                    sent_data["atomic_facts"] = []

            # 5. Citation-based Fact Entailment
            for fact_obj in sent_data.get("atomic_facts", []):
                if "scores" in fact_obj: continue

                raw_fact_text = fact_obj["fact"]
                clean_fact_text = re.sub(r'\s*\([^)]+\)', '', raw_fact_text).strip()
                
                citations = util.extract_citations_from_sentence(raw_fact_text)
                
                if not citations:
                    # fact_obj["is_verification_worthy"] = False
                    fact_obj["scores"] = {"precision": 0.0, "recall": 0.0, "citation_judgments": {}}
                    continue
                
                citation_judgments = {}
                valid_snippet_paths = []
                yes_count = 0
                extraction_error = False # ### CHANGE: Track if ffmpeg fails
                
                # A. Precision Loop
                for cit in citations:
                    start, end = cit['start_time'], cit['end_time']
                    modality = cit['modality']
                    raw_citation_str = cit['citation_segment']
                    
                    if start == -1:
                        citation_judgments[raw_citation_str] = False
                        continue

                    extract_mode, ext = determine_extraction_mode(modality, start, end)
                    
                    snippet_filename = f"{entry_id}_{start:.2f}_{end:.2f}.{ext}"
                    
                    snippet_path = os.path.join(self.temp_dir, snippet_filename)
                    
                    success = extract_media_segment(video_source_path, snippet_path, start, end, extract_mode)
                    
                    if not success:
                        # ### CHANGE: If extraction fails, mark the error and stop processing this fact
                        print(f"    [Warn] Extraction failed for {start}-{end}. Marking as None.")
                        extraction_error = True
                        citation_judgments[raw_citation_str] = None 
                        continue
                        
                    prompt = self.prompts["entailment"].format(fact=clean_fact_text)
                    messages = construct_multimodal_message(prompt, [snippet_path])
                    
                    try:
                        raw_ent = self.model_entailment.generate_response(messages)
                        is_entailed = extract_prediction_simple(raw_ent)
                        citation_judgments[raw_citation_str] = is_entailed
                        print(f"    [Cit] {is_entailed} | {raw_citation_str} ({extract_mode})")
                        
                        if is_entailed:
                            yes_count += 1
                        
                        valid_snippet_paths.append(snippet_path)
                    except Exception as e:
                        print(f"    [Err] {e}")
                        citation_judgments[raw_citation_str] = False

                # ### CHANGE: Logic to return None if extraction failed
                if extraction_error:
                    precision = None
                    recall = None
                    print(f"    [Recall] None (Extraction Failed)")
                else:
                    precision = (yes_count / len(citations)) if citations else 0.0
                    
                    # B. Recall Calculation
                    recall = 0.0
                    if len(citations) == 1:
                        recall = precision
                        print(f"    [Recall] {recall} (Optimized)")
                    elif valid_snippet_paths:
                        prompt = self.prompts["entailment"].format(fact=clean_fact_text)
                        messages = construct_multimodal_message(prompt, valid_snippet_paths)
                        try:
                            raw_recall_ent = self.model_entailment.generate_response(messages)
                            is_recall_entailed = extract_prediction_simple(raw_recall_ent)
                            recall = 1.0 if is_recall_entailed else 0.0
                            print(f"    [Recall] {recall}")
                        except Exception:
                            recall = 0.0

                fact_obj["scores"] = {
                    "precision": precision,
                    "recall": recall,
                    "citation_judgments": citation_judgments
                }

        return self.calculate_entry_scores(entry)

    def calculate_entry_scores(self, entry):
        evals = entry.get("evaluations", {})
        vw_sentences_count = 0
        cited_vw_sentences_count = 0
        pool_precision = []
        pool_recall = []

        for sent_text, sent_data in evals.items():
            if sent_data.get("verification_worthy", {}).get("prediction", False):
                vw_sentences_count += 1
                facts = sent_data.get("atomic_facts", [])
                
                sent_has_valid_citation = False
                
                for fact_obj in facts:
                    scores = fact_obj.get("scores")
                    if scores and scores.get("citation_judgments"):
                        sent_has_valid_citation = True
                        
                        # ### CHANGE: Only add to pool if value is not None
                        p_val = scores.get("precision")
                        r_val = scores.get("recall")
                        
                        if p_val is not None:
                            pool_precision.append(p_val)
                        if r_val is not None:
                            pool_recall.append(r_val)
                
                if sent_has_valid_citation:
                    cited_vw_sentences_count += 1

        coverage = cited_vw_sentences_count / vw_sentences_count if vw_sentences_count > 0 else 0.0
        precision = sum(pool_precision) / len(pool_precision) if pool_precision else 0.0
        recall = sum(pool_recall) / len(pool_recall) if pool_recall else 0.0
        f1 = calculate_harmonic_mean(precision, recall)
        
        entry["response_score"] = {
            "coverage": coverage,
            "attribution_precision": precision,
            "attribution_recall": recall,
            "citation_f1": f1,
            "final_score": coverage * f1
        }
        return entry

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_num", type=int, default=None)
    parser.add_argument("--parse_reasoning", action="store_true", default=False)
    args = parser.parse_args()

    data = []
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f:
                data = json.load(f)
        except:
            pass
    if not data:
        with open(args.input_file, 'r') as f:
            data = json.load(f)

    if args.max_num:
        data = data[:args.max_num]

    print(f"Processing {len(data)} items...")
    
    pipeline = Pipeline(args)

    for i in tqdm(range(len(data))):
        item = data[i]
        if "response_score" in item: continue
        try:
            processed_item = pipeline.process_entry(copy.deepcopy(item), parse_reasoning=args.parse_reasoning)
            data[i] = processed_item
            with open(args.output_file, 'w') as f:
                json.dump(data, f, indent=4)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"\n[Error] Item {i} ({item.get('id')}) failed: {e}")
            continue
    
    if os.path.exists(pipeline.temp_dir):
        shutil.rmtree(pipeline.temp_dir)

if __name__ == "__main__":
    main()