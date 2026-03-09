# AQUA Open-Source Repository Guide

This document summarizes the repository layout for the ICLR 2026 paper:
"Safeguarding Multimodal Knowledge Copyright in the RAG-as-a-Service Environment".

## 1) Core Runtime Structure

The pipeline centers on `multimodalrag.py` and experiment scripts under `experiments/`.

```text
AQUA/
  multimodalrag.py                  # Main multimodal RAG + watermark pipeline
  experiments/
    effectiveness/                  # p-value, rank, FPR/TPR, CGSR
    robustness/                     # attacked watermark robustness
    harmlessness/                   # normal query behavior
    stealthiness/                   # retrieval ratio + PCA analysis
  utils/
    indexing_faiss.py               # Build FAISS index files
  datasets/                         # Dataset metadata + sample probe/watermark files
  prompts/                          # Prompt templates
  qwenvl/                           # Qwen-VL helper wrapper
  Qwen_VL_Chat/                     # Legacy Qwen-VL-Chat model wrapper
  llava/                            # Vendored LLaVA utilities (optional for some workflows)
  vcd_utils/                        # LLaVA/VCD helper utilities
  environment.yml                   # Environment definition
  README.md
  LICENSE
```

## 2) Keep vs Remove for Open Source

### Keep (required)

- `multimodalrag.py`
- `experiments/`
- `utils/indexing_faiss.py`
- `qwenvl/`
- `prompts/`
- `datasets/` lightweight metadata and demo probe/watermark files only
- `environment.yml`
- `README.md`
- `LICENSE`

### Keep (optional, but useful)

- `Qwen_VL_Chat/` (needed when using `Qwen-VL-Chat` generator mode)
- `llava/` and `vcd_utils/` (needed only for local/legacy LLaVA utility paths)

### Remove or keep out of Git (recommended)

- `models/` (local model checkpoints, very large)
- `results/` (generated outputs, reproducible artifacts)
- full raw images and large FAISS binaries under `datasets/**` (publish download script instead)
- `TinyLLaVABench/` (external baseline repo clone)
- `WAVES/` (external baseline repo clone)
- `mmjbench/` (external benchmark repo clone)
- `rebuttal/` (paper rebuttal artifacts)
- `test/` (local debugging scripts/notebooks)
- `iclr2026_conference.pdf` (large paper file; prefer arXiv/ICLR link in README)
- `__pycache__/`, local IDE folders, temporary files

## 3) Suggested Release Policy

- Publish only source code + lightweight samples in Git.
- Move all heavy assets (models, full datasets, generated results) to external storage.
- Provide dataset/model download instructions in README instead of uploading binaries.
- Keep experiment entrypoints runnable with relative paths.

## 4) Pre-Release Check Commands

```bash
# inspect tracked files that will be pushed
git ls-files

# inspect untracked files that may be added by mistake
git status --short

# inspect largest local directories
du -sh * | sort -h
```

