# Multimodal Fact-Level Attribution for Verifiable Reasoning

**Authors:** David Wan, Han Wang, Ziyang Wang, Elias Stengel-Eskin, Hyunji Lee, Mohit Bansal

This repository contains the code and data for the paper **"Multimodal Fact-Level Attribution for Verifiable Reasoning"**.

![Image](https://github.com/user-attachments/assets/f0554a5a-63eb-4bdb-8601-6ce52bfa3e9c)

## Installation

### Prerequisites

* **Python:** 3.12+
* **Package Manager:** `uv` (recommended)
* **System Dependencies:** CUDA, FFmpeg (for video processing)

### 1. General Setup

For most models (excluding Qwen-VL and Qwen-Omni), you can simply install the dependencies from the requirements file:

```bash
uv pip install -r requirements.txt

```

### 2. Model-Specific Setup (Qwen)

Due to conflicting dependencies, we recommend maintaining **separate virtual environments** for Qwen-VL and Qwen-Omni.

**For Qwen-VL:**
Install `vllm` via the official package:

```bash
uv pip install vllm --torch-backend=auto

```

**For Qwen-Omni:**
Install the specific version of `vllm` and the `vllm-omni` fork:

```bash
uv pip install vllm==0.15.0 --torch-backend=auto
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .

```

### 3. Configuration

**API Keys**
If you are using cloud-based models (e.g., Gemini), set the necessary environment variables:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
# Add other keys as needed (e.g., OPENAI_API_KEY)
```

**Dataset Paths**
Update `src/util.py` to point to your local dataset directories:

```python
DATASET_CONFIGS = {
    "videommmu": {
        "video_dir": "/path/to/VideoMMMU/videos/",
        "hf_path": "/path/to/data/VideoMMMU_sample",
    },
    # Add other datasets here
}
```

## Project Structure

```text
.
├── README.md                          # This file
├── src/                               # Core source code
│   ├── models.py                      # Multimodal model implementations
│   ├── util.py                        # Shared utilities (text processing, data loading)
│   ├── run_baseline.py                # Generate baseline responses
│   ├── run_baseline_with_citation.py  # Generate responses with citations
│   ├── run_metric.py                  # Evaluation pipeline
│   ├── run_generation_program.py      # Generation with iterative feedback
│   └── generation_program_util.py     # Utilities for the generation program
│
├── prompts/                           # Prompt templates
│   ├── base.txt                       # Base reasoning prompt
│   ├── base_with_citations.txt        # Reasoning with citation extraction
│   ├── decontextualization.txt        # Remove document context
│   ├── atomic_decomposition.txt       # Break into atomic facts
│   ├── coverage_prompt.txt            # Fact verification
│   ├── entailment_prompt.txt          # Entailment checking
│   └── ...                            # Additional task-specific prompts
│
└── requirements.txt                   # Python dependencies

```

## Data

Datasets and model generations are available via Google Drive: **[[Link to Drive]](https://drive.google.com/drive/folders/1j5yOT2rVLxT8-DoTrNs7YaubWiw56EEO?usp=sharing)**

## Usage

### 1. Generate Baseline Responses

Generate model responses without citations.

```bash
python src/run_baseline.py <dataset_name> <model_name>

```

* **Example:** `python src/run_baseline.py videommmu gemini-2.5-flash`
* **Arguments:**
* `dataset_name`: Must be defined in `DATASET_CONFIGS`.
* `model_name`: Model identifier from the supported models list.



### 2. Generate Responses with Citations

Generate responses using the `base_with_citations.txt` prompt to encourage explicit citations.

```bash
python src/run_baseline_with_citation.py <dataset_name> <model_name>

```

### 3. Run Evaluation Pipeline

This pipeline decontextualizes responses, decomposes them into atomic facts, extracts citations, and computes verification scores.

```bash
python src/run_metric.py --input-file <path-to-generations.json> --output-dir <output-directory>

```

### 4. Program-Aided Generation

Generate responses with iterative refinement and feedback.

```bash
python src/run_generation_program.py <dataset_name> <model_name>

```

### Advanced Configuration

You can tune performance using the following environment variables:

```bash
export OPENAI_API_BASE="your-openai-endpoint"          # For custom deployments
export VLLM_USE_V1=0                                   # Toggle VLLM versions
export DECORD_EOF_RETRY_MAX=20480                      # Video processing stability
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" # Memory management

```

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{wan2026multimodalfactlevelattributionverifiable,
      title={Multimodal Fact-Level Attribution for Verifiable Reasoning}, 
      author={David Wan and Han Wang and Ziyang Wang and Elias Stengel-Eskin and Hyunji Lee and Mohit Bansal},
      year={2026},
      eprint={2602.11509},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.11509}, 
}
```
