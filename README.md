# AQUA: Safeguarding Multimodal Knowledge Copyright in the RAG-as-a-Service Environment

[![Paper](https://img.shields.io/badge/Paper-ICLR%202026-blue)](https://arxiv.org/abs/2506.10030)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official code repository for the ICLR 2026 paper:
**Safeguarding Multimodal Knowledge Copyright in the RAG-as-a-Service Environment**.

AQUA is a watermarking framework for text-to-text multimodal RAG systems. It injects image-side watermark signals and verifies ownership through probe-query responses.

## 1. Repository Layout

```text
AQUA/
  multimodalrag.py                 # Core multimodal RAG + watermark pipeline
  experiments/
    effectiveness/                 # p-value, rank, CGSR, FPR/TPR scripts
    harmlessness/                  # normal-query and relevant-query evaluation
    robustness/                    # attacked watermark robustness evaluation
    stealthiness/                  # retrieval-ratio and PCA analysis
  utils/indexing_faiss.py          # FAISS index building scripts
  datasets/                        # sample metadata, probe-query examples, watermark examples
  Qwen_VL_Chat/                    # legacy Qwen-VL-Chat wrapper
  qwenvl/                          # Qwen helper script
  prompts/
  environment.yml
```

## 2. Environment Setup

```bash
git clone https://github.com/tychenn/AQUA.git
cd AQUA
conda env create -f environment.yml
conda activate AQUA
```

## 3. Required External Assets

This repository intentionally excludes large assets (raw datasets and model checkpoints).

### 3.1 Datasets

Download and place raw images to:

- `datasets/MMQA/images`
- `datasets/WebQA/images`

### 3.2 Models

`multimodalrag.py` loads models from local paths under `models/`.
At minimum, ensure the model folders used by your chosen config exist.

Common examples:

- Retriever (default): `models/clip-vit-large-patch14-336`
- Retriever (optional): `models/siglip-so400m-patch14-384`
- Generator (default): `models/Qwen2.5-VL-7B-Instruct`

If your local model directory names differ, align them with the names above or update the paths in `multimodalrag.py`.

### 3.3 FAISS Index and Mapping

By default, the pipeline expects:

- `datasets/MMQA/faiss_index/MMQA_all_hf_clip.index`
- `datasets/MMQA/jsons/WatermarkMMRAG/MMQA_all_index_to_image_id.json`
- `datasets/WebQA/faiss_index/WebQA_hf_clip_100%.index`
- `datasets/WebQA/jsons/WebQA_all_index_to_image_id.json`

You can also provide custom paths at runtime:

- `--index_path`
- `--index_mapping_path`

## 4. Watermark and Probe-Query Data Format

Place generated watermark images and probe-query JSON files as follows.

| Usage | Path pattern |
|---|---|
| Acronym watermark images | `datasets/watermark_images/acronym/*.png` |
| Spatial watermark images | `datasets/watermark_images/spatial/*.png` |
| Naive watermark images | `datasets/watermark_images/naive/*.(png/jpg/jpeg)` |
| Opt watermark images | `datasets/watermark_images/opt/*.(png/jpg/jpeg)` |
| Acronym probe queries | `datasets/probe_query/acronym/*.json` |
| Spatial probe queries | `datasets/probe_query/spatial/*.json` |
| Opt probe queries for LLaVA/TinyLLaVA | `datasets/probe_query/opt/llava/*.json` |
| Opt probe queries for Qwen-VL-Chat | `datasets/probe_query/opt/qwen/*.json` |
| Opt probe queries for InternVL | `datasets/probe_query/opt/intern/*.json` |
| Opt probe queries for Qwen2.5/Qwen3 | `datasets/probe_query/opt/qwen25/*.json` |

Probe-query JSON format:

```json
[
  {
    "watermark_path": "datasets/watermark_images/acronym/BJT.png",
    "gt": "Bai Jing Ting",
    "probe_query": "Who is BJT? Answer the name related to BJT."
  }
]
```

Notes:

1. `watermark_path` must be a valid **relative path** to the watermark image.
2. Keep `watermark_type`, watermark image directory, and probe-query directory consistent.
3. For `watermark_type=opt`, image retrieval uses `datasets/watermark_images/opt/`.
4. If your old data is under `datasets/probe_query/opt/qwenvl/`, copy or rename it to `datasets/probe_query/opt/qwen/` for current scripts.

## 5. Build FAISS Index

```bash
# WebQA ratio indices (default script behavior)
python -m utils.indexing_faiss --datasets WebQA --clip_type hf_clip

# MMQA ratio indices
python -m utils.indexing_faiss --datasets MMQA_ratio --clip_type hf_clip
```

## 6. Run Core Pipeline Smoke Test

```bash
python multimodalrag.py \
  --dataset MMQA \
  --retriever_type clip \
  --generator_type Qwen2.5-VL-7B-Instruct \
  --watermark_type acronym \
  --clip_topk 5 \
  --retriever_device cuda:0 \
  --generator_device cuda:0
```

## 7. Run Experiments

### 7.1 Effectiveness

```bash
# Rank
python -m experiments.effectiveness.rank \
  --dataset MMQA \
  --retriever_type clip \
  --generator_type Qwen2.5-VL-7B-Instruct \
  --watermark_type acronym

# p-value
python -m experiments.effectiveness.pvalue \
  --dataset MMQA \
  --retriever_type clip \
  --generator_type Qwen2.5-VL-7B-Instruct \
  --watermark_type acronym

# CGSR
python -m experiments.effectiveness.CGSR \
  --dataset MMQA \
  --retriever_type clip \
  --generator_type Qwen2.5-VL-7B-Instruct \
  --watermark_type acronym
```

### 7.2 Harmlessness

```bash
python -m experiments.harmlessness.normal_query \
  --dataset MMQA \
  --retriever_type clip \
  --generator_type Qwen2.5-VL-7B-Instruct \
  --watermark_type acronym
```

### 7.3 Robustness

```bash
python -m experiments.robustness.table \
  --dataset MMQA \
  --retriever_type clip \
  --generator_type Qwen2.5-VL-7B-Instruct \
  --watermark_type acronym_all
```

### 7.4 Stealthiness

```bash
python -m experiments.stealthiness.calculate_retrieval_ratio \
  --dataset WebQA \
  --retriever_type clip \
  --generator_type None
```

## 8. Common Issues

- **FileNotFoundError for index/mapping**: verify FAISS files exist, or pass `--index_path` and `--index_mapping_path`.
- **Image file not found for ID**: check image roots (`datasets/MMQA/images`, `datasets/WebQA/images`) and watermark directories.
- **Model loading failure**: confirm local model directory names match those referenced in `multimodalrag.py`.

## 9. License

This project is released under the MIT License. See [LICENSE](LICENSE).

## 10. Citation

```bibtex
@misc{chen2025safeguardingmultimodalknowledgecopyright,
  title={Safeguarding Multimodal Knowledge Copyright in the RAG-as-a-Service Environment},
  author={Tianyu Chen and Jian Lou and Wenjie Wang},
  year={2025},
  eprint={2506.10030},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2506.10030}
}
```
