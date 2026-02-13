"""
Utility functions for multimodal grounding annotation pipeline.

This module contains shared functions used across multiple scripts:
- Dataset configurations
- Text processing (reasoning/answer extraction, cleaning)
- Citation extraction and parsing
- Sentence splitting
- File loading utilities
- Timestamp parsing
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# DATASET CONFIGURATIONS
# ==========================================

DATASET_CONFIGS = {
    "videommmu": {
        "video_dir": "<VIDEO_DIR_VIDEOMMMU>",
        "hf_path": "data/VideoMMMU_sample",
    },
    "worldsense": {
        "video_dir": "<>VIDEO_DIR_WORLDSENSE>",
        "hf_path": "data/WorldSense_sample",
    },
}


# ==========================================
# TEXT PROCESSING FUNCTIONS
# ==========================================

# Regex pattern for thinking blocks
RE_THINKING_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)


def clean_model_output(text: str) -> str:
    """
    Removes <think>...</think> blocks and whitespace from model output.
    
    Args:
        text: Raw model output text
        
    Returns:
        Cleaned text with thinking blocks removed
    """
    if not text:
        return ""
    cleaned_text = RE_THINKING_BLOCK.sub('', text)
    return cleaned_text.strip()


def extract_reasoning(text: str) -> str:
    """Extracts text between 'Reasoning:' and 'Answer:' markers."""
    if not text:
        return ""

    start_match = re.search(r"(?:^|\n|\*)\s*Reasoning\s*[:*]*\s*", text, re.IGNORECASE)
    if start_match is None:
        return text
    
    content = text[start_match.end():]

    end_pattern = r"(?:\n|^)\s*Answer\s*[:\s]"
    end_match = re.search(end_pattern, content, re.IGNORECASE)
    
    if end_match is None:
        return text
    
    if end_match:
        content = content[:end_match.start()]

    return content.strip()


def extract_answer(text: str) -> str:
    """
    Extracts answer from model text. 
    Prioritizes LaTeX \\boxed{...} content, then falls back to 'Answer:' markers.
    
    Args:
        text: Full model response text
        
    Returns:
        Extracted answer text
    """
    
    # 1. Strategy: Look for LaTeX \boxed{...}
    # This regex looks for \boxed{ followed by distinct content.
    # We use findall because models often output the box multiple times or 
    # put intermediate steps in boxes. We usually want the LAST one.
    boxed_pattern = r"\\boxed\s*\{(.*?)\}"
    boxed_matches = re.findall(boxed_pattern, text)
    
    if boxed_matches:
        # Return the content of the very last \boxed block
        return boxed_matches[-1].strip()

    # 2. Strategy: Original 'Answer:' marker logic
    pattern = r"Answer:\s*(.*?)(?:\n|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 3. Fallback: try to find answer at the end
    if "Answer:" in text or "answer:" in text:
        parts = re.split(r"Answer:\s*", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[-1].strip()
    
    return ""


# ==========================================
# CITATION EXTRACTION FUNCTIONS
# ==========================================

def parse_timestamp_str(time_str: str) -> float:
    """Helper to convert 'MM:SS' string to seconds (float)."""
    try:
        parts = time_str.strip().split(':')
        if len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        return -1.0
    except ValueError:
        return -1.0

def extract_citations_from_sentence(sentence: str) -> List[Dict[str, Any]]:
    """
    Extracts citation patterns from a sentence, handling multiple citations
    separated by semicolons and filtering out invalid formats.
    
    Valid formats: 
    - (visual, 0:20)
    - (audio, 1:20-1:30)
    - (visual, 0:02-0:04; visual, 0:16-0:19)
    """
    citations = []
    parenthetical_matches = re.finditer(r'\((.*?)\)', sentence)
    
    for match in parenthetical_matches:
        full_content = match.group(1)
        raw_full = match.group(0)
        segments = full_content.split(';')
        
        for segment in segments:
            segment = segment.strip()
            citation_pattern = r'^\s*([a-zA-Z]+)\s*,\s*(\d+:\d+(?:-\d+:\d+)?)\s*$'
            seg_match = re.match(citation_pattern, segment)
            
            if seg_match:
                modality = seg_match.group(1).lower()
                time_part = seg_match.group(2)
                
                if '-' in time_part:
                    start_str, end_str = time_part.split('-', 1)
                    start_time = parse_timestamp_str(start_str)
                    end_time = parse_timestamp_str(end_str)
                else:
                    start_time = parse_timestamp_str(time_part)
                    end_time = start_time
                
                citations.append({
                    'raw': raw_full,
                    'citation_segment': f"({segment})",
                    'modality': modality,
                    'start_time': start_time,
                    'end_time': end_time
                })

    return citations


# ==========================================
# SENTENCE PROCESSING FUNCTIONS
# ==========================================

def split_text_into_sentences(text: str) -> List[str]:
    """Split text into sentences, handling citations and ellipses."""
    if not text:
        return []
    
    sentences = []
    current_pos = 0
    
    splitter_pattern = re.compile(r'(?P<cit>\([^)]+\))|(?P<ellipsis>\.{2,})|(?P<punct>[.!?]+)')
    pending_punct_split = False
    
    for match in splitter_pattern.finditer(text):
        if match.group('ellipsis'):
            pending_punct_split = False

        elif match.group('punct'):
            remaining_text = text[match.end():]
            
            if re.match(r'^[\s\'"*_]*\(', remaining_text):
                pending_punct_split = True
            else:
                sentences.append(text[current_pos:match.end()].strip())
                current_pos = match.end()
                pending_punct_split = False
                
        elif match.group('cit'):
            if pending_punct_split:
                sentences.append(text[current_pos:match.end()].strip())
                current_pos = match.end()
                pending_punct_split = False
    
    if current_pos < len(text):
        rem = text[current_pos:].strip()
        if rem:
            sentences.append(rem)
    
    return sentences


def parse_llm_list(text: str) -> List[str]:
    """Extract bulleted list items (lines starting with '-')."""
    if not text:
        return []
    lines = text.split('\n')
    return [line.strip() for line in lines if line.strip().startswith("-")]


# ==========================================
# FILE AND DATA LOADING FUNCTIONS
# ==========================================

def load_prompt(path: str) -> str:
    """
    Loads a prompt template from a file.
    
    Args:
        path: Path to the prompt file
        
    Returns:
        Prompt text as string, empty string if file not found
    """
    if not os.path.exists(path):
        logger.warning(f"Prompt file not found: {path}")
        return ""
    with open(path, "r", encoding='utf-8') as f:
        return f.read()


def load_metadata(ds_name: str) -> Dict[str, Dict[str, str]]:
    """
    Loads metadata for a dataset from HuggingFace disk.
    
    Args:
        ds_name: Dataset name (must be in DATASET_CONFIGS)
        
    Returns:
        Dictionary mapping video IDs to metadata (question, path)
    """
    config = DATASET_CONFIGS.get(ds_name)
    if not config:
        logger.error(f"Unknown dataset: {ds_name}")
        return {}
    
    if load_from_disk is None:
        logger.error("datasets library not available")
        return {}
    
    try:
        ds = load_from_disk(config["hf_path"])
        return {
            row["video"]: {
                "question": row.get('question', ''),
                "path": os.path.join(config["video_dir"], f"{row['video']}.mp4")
            } for row in ds
        }
    except Exception as e:
        logger.error(f"Error loading metadata for {ds_name}: {e}")
        return {}


def get_video_path(ds_name: str, video_id: str) -> Optional[str]:
    """
    Gets the full path to a video file for a given dataset and video ID.
    
    Args:
        ds_name: Dataset name
        video_id: Video identifier
        
    Returns:
        Full path to video file, or None if dataset not found
    """
    config = DATASET_CONFIGS.get(ds_name)
    if not config:
        return None
    return os.path.join(config["video_dir"], f"{video_id}.mp4")


# ==========================================
# TIMESTAMP PARSING FUNCTIONS
# ==========================================

def parse_timestamp(time_str: Any) -> float:
    """
    Parses a timestamp string into seconds (float).
    
    Supports formats:
    - Single number: "5" -> 5.0
    - MM:SS: "1:30" -> 90.0
    - HH:MM:SS: "1:05:30" -> 3930.0
    
    Args:
        time_str: Timestamp string or number
        
    Returns:
        Time in seconds, or -1.0 if parsing fails
    """
    try:
        if isinstance(time_str, (int, float)):
            return float(time_str)
        
        parts = [float(p) for p in str(time_str).strip().split(':')]
        
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        else:  # len(parts) == 3
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        return -1.0


# ==========================================
# FILE UTILITY FUNCTIONS
# ==========================================

def is_valid_file(filepath: str) -> bool:
    """
    Returns True if file exists and has content.
    
    Args:
        filepath: Path to file to check
        
    Returns:
        True if file exists and has size > 0
    """
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0


def remove_thinking_from_text(text: str) -> str:
    """
    Removes thinking blocks from text (alternative to clean_model_output).
    Keeps everything after </think> tag.
    
    Args:
        text: Text that may contain thinking blocks
        
    Returns:
        Text with thinking blocks removed
    """
    if not text:
        return ""
    
    if "</think>" in text:
        idx = text.index("</think>")
        return text[idx + len("</think>"):].strip()
    
    return text.strip()
