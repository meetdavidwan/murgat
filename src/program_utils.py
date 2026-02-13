import ast
import os
import subprocess
import re
import json
import traceback
import time
import functools
from typing import List, Dict, Any, Tuple, Optional, Union
from util import parse_timestamp, clean_model_output

# --- CONSTANTS ---
NUM_SAMPLES = 5

# --- LOAD PROMPTS ---
def load_prompt(filename):
    try:
        with open(os.path.join("prompts", filename), "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

PROMPT_MULTIMODAL_ENTAILMENT = load_prompt("multimodal_entailment.txt")
PROMPT_TEXT_ENTAILMENT = load_prompt("text_entailment.txt")
PROMPT_DESCRIBE_VISION = load_prompt("describe_vision.txt")
PROMPT_DESCRIBE_AUDIO = load_prompt("describe_audio.txt")
PROMPT_SYNTHESIZE = load_prompt("synthesize.txt")
PROMPT_FIND_TIMESTAMP = load_prompt("find_timestamp.txt")

# ==========================================
# 1. DECORATORS & GENERATION WRAPPERS
# ==========================================

def retry_with_backoff(retries=3, backoff_in_seconds=5):
    """Decorator to retry functions (like API calls) on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    msg = str(e).lower()
                    if x == retries:
                        raise e
                    if '503' in msg or 'overloaded' in msg or 'rate limit' in msg:
                        sleep_time = (backoff_in_seconds * 2 ** x) 
                        print(f"  [API Warn] Model overloaded. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                        x += 1
                    else:
                        raise e
        return wrapper
    return decorator

def safe_generate(model, messages, num_samples=1):
    """Wrapper for model generation that applies retry logic."""
    @retry_with_backoff(retries=3, backoff_in_seconds=5)
    def _call():
        return model.generate_response(messages, num_samples=num_samples)
    return _call()

# ==========================================
# 2. HELPERS (Time, Parsing, Video)
# ==========================================

def parse_time_range(range_str: Union[str, List[str]]) -> Tuple[float, float]:
    if isinstance(range_str, list):
        range_str = range_str[0]
    range_str = str(range_str).strip()
    try:
        if '-' in range_str:
            start_str, end_str = range_str.split('-')
            return parse_timestamp(start_str), parse_timestamp(end_str)
        else:
            t = parse_timestamp(range_str)
            return t, t + 2.0
    except Exception:
        return 0.0, 2.0

def merge_overlapping_intervals(timestamps: List[str], max_gap: float = 2.0, limit: int = 10) -> List[str]:
    if not timestamps: return []
    
    ranges = []
    for t in timestamps:
        start = parse_timestamp(t.split('-')[0])
        end = parse_timestamp(t.split('-')[-1]) if '-' in t else start + 2.0
        if start != -1 and end != -1:
            ranges.append((start, end))
            
    if not ranges: return []
    ranges.sort(key=lambda x: x[0])

    merged = []
    if ranges:
        curr_start, curr_end = ranges[0]
        for next_start, next_end in ranges[1:]:
            if next_start <= curr_end + max_gap:
                curr_end = max(curr_end, next_end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))

    if len(merged) > limit:
        merged = merged[:limit]

    def fmt(val):
        m, s = divmod(val, 60)
        return f"{int(m):02d}:{int(s):02d}"

    return [f"{fmt(s)}-{fmt(e)}" for s, e in merged]

def cut_video_segments(input_path: str, timestamp_list: Union[List[str], str], output_dir: str = "temp_clips") -> List[str]:
    if isinstance(timestamp_list, str): timestamp_list = [timestamp_list]
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    clip_paths = []
    base_name = os.path.basename(input_path)
    # Filter out empty or invalid timestamps before processing
    unique_timestamps = sorted(list(set([t for t in timestamp_list if t])))

    for i, ts in enumerate(unique_timestamps):
        start, end = parse_time_range(ts)
        duration = end - start
        if duration <= 0: continue
        
        start_str = f"{start:.2f}".replace('.', '-')
        end_str = f"{end:.2f}".replace('.', '-')
        clip_name = f"{base_name}_clip{i}_{start_str}_{end_str}.mp4"
        clip_path = os.path.join(output_dir, clip_name)
        
        # Only run FFMPEG if file doesn't exist
        if not os.path.exists(clip_path):
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-i", input_path, "-t", str(duration),
                "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
                "-loglevel", "error", clip_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Check if file was actually created and has size (ffmpeg might exit 0 but write nothing if out of bounds)
                if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
                     raise Exception("Empty output file")
            except Exception as e:
                print(f"  [FFMPEG Error] Could not extract {ts} from {base_name}. Reason: {e}")
                continue # Skip adding this path
        
        clip_paths.append(clip_path)
        
    return clip_paths

def construct_multimodal_message(video_paths: List[str], text_prompt: str) -> List[Dict]:
    content = []
    if video_paths:
        for v_path in video_paths:
            content.append({"type": "video", "video": v_path})
    content.append({"type": "text", "text": text_prompt})
    return [{"role": "user", "content": content}]

def parse_json_output(text: str) -> Any:
    text = clean_model_output(text)
    try:
        match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if match: return json.loads(match.group(0))
        return json.loads(text) 
    except: return None

# ==========================================
# 3. CITATION RESTORATION
# ==========================================

def seconds_to_str(val):
    try:
        if isinstance(val, str) and ":" in val: return val
        seconds = float(val)
        mm = int(seconds // 60)
        ss = int(seconds % 60)
        return f"{mm:02d}:{ss:02d}"
    except:
        return str(val)

def restore_citations(text: str, evidence_list: List[Dict]) -> str:
    """
    Replaces [1] with (visual, 00:01). Includes explicit space padding to avoid stuck text.
    """
    def get_citation_parts(idx):
        if not (0 <= idx < len(evidence_list)): return []
        item = evidence_list[idx]
        
        raw_mod = str(item.get('modality', 'visual')).lower().strip()
        if 'audio' in raw_mod and 'visual' in raw_mod: modality = 'visual' 
        elif 'audio' in raw_mod: modality = 'audio'
        else: modality = 'visual'

        timestamps = item.get('timestamps', [])
        if not isinstance(timestamps, list): timestamps = [timestamps]
        
        parts = []
        for t in timestamps:
            parts.append(f"{modality}, {seconds_to_str(t)}")
        if not parts: parts.append(f"{modality}, unknown time")
        return parts

    def replace_match(match):
        content = match.group(1) 
        all_numbers = re.findall(r'\d+', content)
        indices = set(int(n) for n in all_numbers)
        
        sorted_indices = sorted(list(indices))
        all_citation_strings = []
        
        for i in sorted_indices:
            idx = i - 1 
            cit_parts = get_citation_parts(idx)
            all_citation_strings.extend(cit_parts)
        
        if not all_citation_strings: return match.group(0)
        
        combined = "; ".join(all_citation_strings)
        return f" ({combined})"

    pattern = r'((?:\[[\d\s,-]+\]\s*)+)' 
    return re.sub(pattern, replace_match, text)

# ==========================================
# 4. BACKEND LOGIC
# ==========================================

def check_multimodal_entailment(model: Any, paths: List[str], target_fact: str) -> Tuple[bool, str]:
    prompt = PROMPT_MULTIMODAL_ENTAILMENT.format(target_fact=target_fact)
    messages = construct_multimodal_message(paths, prompt)
    
    # Store RAW response
    raw_response = safe_generate(model, messages, num_samples=1)
    if isinstance(raw_response, list): raw_response = raw_response[0]
    
    # Use CLEAN response for logic
    cleaned_response = clean_model_output(str(raw_response))
    
    # parse_json_output now handles cleaning too, but we cleaned above to be safe for non-json fallback checks
    data = parse_json_output(cleaned_response)
    
    if data and isinstance(data, dict):
        ans = str(data.get("answer", "")).strip().lower()
        return (ans == "yes"), data.get("reasoning", "")
        
    # Return raw_response in string form so logs see the think process if JSON failed
    return False, str(raw_response)

def check_text_entailment(model: Any, evidence_text: str, claim: str) -> Tuple[bool, str]:
    prompt = PROMPT_TEXT_ENTAILMENT.format(evidence=evidence_text, claim=claim)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    raw_response = safe_generate(model, messages, num_samples=1)
    if isinstance(raw_response, list): raw_response = raw_response[0]
    
    cleaned_response = clean_model_output(str(raw_response))
    data = parse_json_output(cleaned_response)
    
    if data and isinstance(data, dict):
        ans = str(data.get("answer", "")).strip().lower()
        return (ans == "yes"), data.get("reasoning", "")
        
    return False, str(raw_response)

def run_find_timestamp(model: Any, video_path: str, query: str, modality: str, use_verification: bool) -> Tuple[List[str], Dict]:
    print(f"  [Find] '{query}' ({modality})")
    prompt = PROMPT_FIND_TIMESTAMP.format(query=query)
    messages = construct_multimodal_message([video_path], prompt)
    
    n_samples = NUM_SAMPLES if use_verification else 1
    raw_responses = safe_generate(model, messages, num_samples=n_samples)
    if isinstance(raw_responses, str): raw_responses = [raw_responses]

    samples_parsed = []
    for resp in raw_responses:
        # Pass raw response to parser; parser will clean it internally via parse_json_output modification
        parsed = parse_json_output(resp)
        found_here = []
        if isinstance(parsed, list): found_here = parsed
        elif isinstance(parsed, dict):
            for key in ["timestamps", "events", "intervals", "segments"]:
                if key in parsed and isinstance(parsed[key], list):
                    found_here = parsed[key]; break
        samples_parsed.append([str(ts).strip() for ts in found_here if ts])

    if not use_verification:
        for s in samples_parsed: 
            if s: return s, {"raw_responses": raw_responses}
        return [], {"raw_responses": raw_responses}

    unique_ts_pool = set(ts for sample in samples_parsed for ts in sample)
    valid_ts_pool = set()
    rejected_log = []

    print(f"    [Verify] Checking {len(unique_ts_pool)} unique candidates...")
    for ts in unique_ts_pool:
        paths = cut_video_segments(video_path, [ts], "temp_clips/verify")
        
        if not paths: 
            rejected_log.append({"timestamp": ts, "reason": "FFMPEG extraction failed", "status": "error"})
            print(f"    [Error] {ts} - Extraction failed")
            continue
        
        is_true, reason = check_multimodal_entailment(model, paths, query)
        if is_true:
            valid_ts_pool.add(ts)
            print(f"    [Confirmed] {ts}")
        else:
            rejected_log.append({"timestamp": ts, "reason": reason, "status": "rejected"})
            print(f"    [Rejected] {ts}")

    best_sample_idx = -1
    max_valid_count = -1
    
    for i, sample in enumerate(samples_parsed):
        valid_count = sum(1 for ts in sample if ts in valid_ts_pool)
        if valid_count > max_valid_count:
            max_valid_count = valid_count
            best_sample_idx = i

    if max_valid_count > 0:
        winning_sample = samples_parsed[best_sample_idx]
        final_output = [ts for ts in winning_sample if ts in valid_ts_pool]
        return final_output, {"raw_responses": raw_responses, "rejected": rejected_log, "confirmed": final_output, "fallback_used": False}
    else:
        print("    [Fallback] All timestamps rejected or errored. Reverting to raw output.")
        fallback_output = []
        for sample in samples_parsed:
            if sample: fallback_output = sample; break
        return fallback_output, {"raw_responses": raw_responses, "rejected": rejected_log, "confirmed": [], "fallback_used": True}

def run_describe(model: Any, video_path: str, timestamps: List[str], instruction: str, modality: str, use_verification: bool) -> Tuple[str, List[Dict]]:
    clip_paths = cut_video_segments(video_path, timestamps)

    if not clip_paths:
        print(f"  [Warn] Video extraction failed for {timestamps}. Returning None.")
        return "Video content extraction failed.", [{"candidate": "Error", "score": None, "reasoning": "FFMPEG Error"}]

    prompt_tmpl = PROMPT_DESCRIBE_AUDIO if "audio" in modality.lower() else PROMPT_DESCRIBE_VISION
    prompt = prompt_tmpl.format(instruction=instruction)
    messages = construct_multimodal_message(clip_paths, prompt)
    
    if not use_verification:
        raw_resp = safe_generate(model, messages, num_samples=1)
        if isinstance(raw_resp, list): raw_resp = raw_resp[0]
        
        # CLEANING ADDED
        clean_resp = clean_model_output(raw_resp)
        return clean_resp, [{"candidate": clean_resp, "raw_candidate": raw_resp, "score": 1}]

    candidates_raw = safe_generate(model, messages, num_samples=NUM_SAMPLES)
    if isinstance(candidates_raw, str): candidates_raw = [candidates_raw]

    scored_candidates = []
    for cand_raw in candidates_raw:
        # CLEANING ADDED for verification
        cand_clean = clean_model_output(cand_raw)
        
        # Verify the CLEAN version, as entailment checker expects standard text
        is_true, reason = check_multimodal_entailment(model, clip_paths, cand_clean)
        score = 1 if is_true else 0
        scored_candidates.append({"candidate": cand_clean, "raw_candidate": cand_raw, "score": score, "reasoning": reason})
    
    scored_candidates.sort(key=lambda x: (x['score'] is not None, x['score']), reverse=True)
    best_answer = scored_candidates[0]["candidate"] if scored_candidates else "Unsure."
    return best_answer, scored_candidates

def run_synthesize(
    model: Any, 
    evidence_list: List[Dict], 
    instruction: str, 
    options: str, 
    use_verification: bool
) -> Tuple[str, List[Dict]]:
    # 1. Format Evidence
    evidence_str = ""
    for idx, item in enumerate(evidence_list):
        content = item.get('output', str(item))
        evidence_str += f"[{idx+1}] {content}\n"
        
    prompt = PROMPT_SYNTHESIZE.format(evidence=evidence_str, instruction=instruction)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    # 2. Fast Path (No Verification)
    if not use_verification:
        raw_answer = safe_generate(model, messages, num_samples=1)
        if isinstance(raw_answer, list): raw_answer = raw_answer[0]
        
        # CLEANING ADDED: Must clean before restoring citations
        clean_answer = clean_model_output(raw_answer)
        final_answer = restore_citations(clean_answer, evidence_list)
        
        return final_answer, [{
            "raw_candidate": raw_answer, 
            "final_candidate": final_answer, 
            "score": 1.0, 
            "reasoning": "No verification requested."
        }]

    # 3. Verification Path
    print(f"  [Synthesize] Generating {NUM_SAMPLES} candidates...")
    candidates_raw = safe_generate(model, messages, num_samples=NUM_SAMPLES)
    if isinstance(candidates_raw, str): candidates_raw = [candidates_raw]
    
    scored_candidates = []

    for cand_raw in candidates_raw:
        # CLEANING ADDED
        cand_clean = clean_model_output(cand_raw)
        
        # Check entailment against the aggregated evidence text
        is_supported, reason = check_text_entailment(model, evidence_str, cand_clean)
        score = 1.0 if is_supported else 0.0
        final_form = restore_citations(cand_clean, evidence_list)
        
        scored_candidates.append({
            "raw_candidate": cand_raw, 
            "final_candidate": final_form, 
            "score": score, 
            "reasoning": reason
        })

    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    if scored_candidates:
        best_candidate_obj = scored_candidates[0]
        best_final_answer = best_candidate_obj["final_candidate"]
        if best_candidate_obj['score'] == 0:
            print("    [Warning] All synthesis candidates failed verification.")
        else:
            print("    [Confirmed] Selected best candidate.")
    else:
        best_final_answer = "Error: No candidates generated."

    return best_final_answer, scored_candidates

def generate_program_plan(model: Any, video_path: str, question: str, formatted_options: str, prompt_template: str) -> str:
    # NOTE: We return the RAW response here. This ensures the <think> blocks are saved 
    # to the output log in main.py. Cleaning happens in execute_narrative_program.
    try: prompt = prompt_template.format(question=question, options=formatted_options)
    except KeyError: prompt = f"{prompt_template}\n\nQuestion: {question}\nOptions:\n{formatted_options}"
    messages = construct_multimodal_message([video_path], prompt)
    response = safe_generate(model, messages, num_samples=1)
    if isinstance(response, list): return response[0]
    return response

def parse_narrative_plan(text: str) -> List[Dict]:
    # Text input here is assumed to be cleaned already by the caller
    lines = text.split('\n'); calls = []
    for line in lines:
        line = line.strip()
        if not line.startswith("-"): continue
        code = line.lstrip("- ").strip()
        try:
            tree = ast.parse(code, mode='eval')
            if isinstance(tree.body, ast.Call):
                calls.append({
                    "func": tree.body.func.id,
                    "args": [recursive_parse_arg(a) for a in tree.body.args],
                    "kwargs": {k.arg: recursive_parse_arg(k.value) for k in tree.body.keywords},
                    "raw": code
                })
        except: pass
    return calls

def recursive_parse_arg(node):
    if isinstance(node, ast.Call):
        args = [recursive_parse_arg(a) for a in node.args]
        kwargs = {k.arg: recursive_parse_arg(k.value) for k in node.keywords}
        return {"type": "call", "func": node.func.id, "args": args, "kwargs": kwargs}
    if isinstance(node, ast.List): return [recursive_parse_arg(e) for e in node.elts]
    if isinstance(node, ast.Constant): return node.value
    return str(node)

def execute_narrative_program(model, plan_text, video_path, options_str, use_feedback=True) -> Tuple[str, List[Dict]]:
    # CLEANING ADDED: Strip <think> blocks before parsing the plan
    cleaned_plan_text = clean_model_output(plan_text)
    
    parsed = parse_narrative_plan(cleaned_plan_text) 
    
    # 2. Initialize the lists used inside the loop
    trace = []
    evidence_list = []
    
    for call in parsed:
        func = call['func']; args = call['args']; kwargs = call['kwargs']
        instruction = kwargs.get('instruction', args[0] if args and func == 'synthesize' else args[-1] if args and func == 'describe' else "Process")
        modality = kwargs.get('modality', 'visual')

        if func == 'synthesize':
            final_ans, debug_log = run_synthesize(model, evidence_list, instruction, options_str, use_feedback)
            trace.append({"step": "synthesize", "evidence_used": evidence_list, "instruction": instruction, "output": final_ans, "verification_log": debug_log})
            return final_ans, trace
        
        elif func == 'describe':
            if not args: continue
            ts_arg = args[0]
            timestamps = []; find_debug_info = {}
            
            # --- NESTED CALL HANDLER (e.g. find_events inside describe) ---
            if isinstance(ts_arg, dict) and ts_arg.get('type') == 'call':
                nested_func = ts_arg.get('func')
                nested_args = ts_arg.get('args', [])
                nested_kwargs = ts_arg.get('kwargs', {})
                
                # Check kwargs first for 'query', then positional args
                q = nested_kwargs.get('query', nested_args[0] if nested_args else "unknown event")
                # Check kwargs for 'modality', then positional
                m = nested_kwargs.get('modality', nested_args[1] if len(nested_args) > 1 else 'visual')
                
                timestamps, find_debug_info = run_find_timestamp(model, video_path, q, m, use_feedback)
                trace.append({"step": "find_timestamp", "query": q, "result": timestamps, "verification_log": find_debug_info})
            
            elif isinstance(ts_arg, list): timestamps = ts_arg
            else: timestamps = [ts_arg]
            
            out, describe_debug_info = run_describe(model, video_path, timestamps, instruction, modality, use_feedback)
            evidence_list.append({"instruction": instruction, "output": out, "timestamps": timestamps, "modality": modality})
            trace.append({"step": "describe", "timestamps": timestamps, "instruction": instruction, "output": out, "verification_log": describe_debug_info})
            print(f"  [Obs] {out}...")

    return "Error: No synthesis reached.", trace

# ==========================================
# 6. EXECUTION: LOGIC
# ==========================================

class VideoWrapper:
    """
    Shim object passed to the generated logic code.
    Records execution trace in a format identical to the Narrative parser.
    """
    def __init__(self, model, video_path, use_feedback):
        self.model = model
        self.video_path = video_path
        self.use_feedback = use_feedback
        self.trace = []          # Logs steps (Find, Describe)
        self.evidence_list = []  # Accumulates evidence for Synthesis

    def find(self, event_description: str, modality: str = "visual") -> List[str]:
        # Normalize modality
        if modality not in ["visual", "audio"]: 
            modality = "audio" if "audio" in modality else "visual"
            
        raw_timestamps, debug_info = run_find_timestamp(
            self.model, self.video_path, event_description, modality, self.use_feedback
        )
        
        merged_timestamps = merge_overlapping_intervals(raw_timestamps, limit=10)
        
        # Log to trace matching Narrative format
        self.trace.append({
            "step": "find_timestamp", # Standardized key
            "query": event_description,
            "modality": modality,
            "result": merged_timestamps,
            "verification_log": debug_info # Contains raw candidates/scores
        })
        return merged_timestamps

    def query(self, time: Union[str, List[str]], question: str, modality: str = "visual") -> str:
        # Normalize inputs
        if modality not in ["visual", "audio"]: 
            modality = "audio" if "audio" in modality else "visual"
        timestamps_str = [time] if isinstance(time, str) else time
        
        output, debug_info = run_describe(
            self.model, self.video_path, timestamps_str, question, modality, self.use_feedback
        )
        
        # Add to evidence list for final synthesis
        self.evidence_list.append({
            "instruction": question, 
            "output": output, 
            "timestamps": timestamps_str, 
            "modality": modality
        })
        
        # Log to trace
        self.trace.append({
            "step": "describe", # Standardized key
            "timestamps": timestamps_str,
            "instruction": question,
            "modality": modality,
            "output": output,
            "verification_log": debug_info # Contains candidates/scores
        })
        return output

def execute_logic_program(
    model: Any, 
    code_text: str, 
    video_path: str, 
    options_list: List[str], 
    use_feedback: bool = True
) -> Tuple[str, List[Dict]]:
    
    # CLEANING ADDED: Remove <think> blocks before extracting Python code
    clean_code_text = clean_model_output(code_text)
    
    # Extract Code Block from cleaned text
    code_match = re.search(r"def execute_command.*", clean_code_text, re.DOTALL)
    if not code_match: 
        return "Error: No execute_command function found.", []
    
    clean_code = code_match.group(0)
    if "```" in clean_code: 
        clean_code = clean_code.split("```")[0]

    video_shim = VideoWrapper(model, video_path, use_feedback)
    execution_state = {"final_instruction": None, "final_options": None}

    def answer_shim(question, evidence=None, options=None):
        execution_state["final_instruction"] = question
        execution_state["final_options"] = options
        return "Process Completed. Ready for Synthesis."

    execution_env = {
        "Video": VideoWrapper,
        "answer_question": answer_shim,
        "List": List, "Union": Union, "Optional": Optional, "Dict": Dict,
        "print": print, 
        "__builtins__": __builtins__
    }

    try:
        exec(clean_code, execution_env)
        
        if "execute_command" in execution_env:
            execution_env["execute_command"](video_shim, options_list)
        else:
            return "Error: execute_command not defined in generated code.", video_shim.trace
            
        final_instruction = execution_state.get("final_instruction")
        
        if final_instruction:
            evidence_to_use = video_shim.evidence_list
            if not evidence_to_use:
                evidence_to_use = [{
                    "instruction": "Logic Execution Trace", 
                    "output": "The program executed but returned no specific visual observations. Check logic.", 
                    "timestamps": []
                }]

            options_str = "\n".join(options_list)
            if execution_state["final_options"]:
                 options_str = str(execution_state["final_options"])

            final_ans, synthesis_candidates = run_synthesize(
                model, 
                evidence_to_use, 
                final_instruction, 
                options_str, 
                use_feedback
            )
            
            video_shim.trace.append({
                "step": "synthesize",
                "instruction": final_instruction,
                "evidence_used": [e['output'] for e in evidence_to_use], 
                "output": final_ans,
                "verification_log": synthesis_candidates 
            })
            
            return final_ans, video_shim.trace
        
        return "Error: Logic program finished but did not call answer_question().", video_shim.trace

    except Exception as e:
        traceback.print_exc()
        video_shim.trace.append({
            "step": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })
        return f"Runtime Error: {str(e)}", video_shim.trace