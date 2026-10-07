# Data preparation

Run commands from the AQUA repository root. Images, query records, and generated indices use the following layout:

```text
datasets/
├── MMQA/
│   ├── images/
│   ├── jsons/MMQA_all_image.json
│   ├── jsons/WatermarkMMRAG/MMQA_all_index_to_image_id.json
│   └── faiss_index/MMQA_all_hf_clip.index
├── WebQA/
│   ├── images/
│   ├── jsons/WebQA_train_val.json
│   ├── jsons/WebQA_all_index_to_image_id.json
│   └── faiss_index/WebQA_hf_clip_100%.index
├── watermark_images/
│   ├── acronym/
│   ├── spatial/
│   ├── naive/
│   └── opt/
├── probe_query/
│   ├── acronym/
│   ├── spatial/
│   ├── naive/
│   └── opt/{llava,qwen,intern,qwen25}/
└── special_query_attack/
    └── acronym_all/
```

## Dataset sources

| Dataset | Download and format reference | AQUA image directory |
| --- | --- | --- |
| MultiModalQA | [Official dataset repository](https://github.com/allenai/multimodalqa) | `datasets/MMQA/images/` |
| WebQA | [Official download instructions](https://github.com/WebQnA/WebQA#download-data) | `datasets/WebQA/images/` |

For MMQA, use the image metadata's `id` as each image's filename stem, retaining its extension. The upstream `path` field identifies the source image. Save image-question records as a JSON list at `datasets/MMQA/jsons/MMQA_all_image.json`, with their `question`, `answers`, and `metadata.image_doc_ids` fields.

For WebQA, use the upstream image-reading [notebook](https://github.com/WebQnA/WebQA/blob/main/demo/Take_a_look_WebQA.ipynb) to read `imgs.tsv` and `imgs.lineidx`, and save images as `<image_id>.png` or `<image_id>.jpg`. Place `WebQA_train_val.json` under `datasets/WebQA/jsons/`.

Both datasets use `--normal_queries_path` to select a different question file. AQUA accepts a JSON list of question records. For WebQA it also accepts a dictionary keyed by question ID and normalizes `Q`, `A`, and `img_posFacts` to `question`, `answers`, and `metadata.image_doc_ids`.

### Normal query format

```json
[
  {
    "qid": "question-1",
    "question": "What object is on the table?",
    "answers": ["a cup"],
    "metadata": {"image_doc_ids": ["image-1"]}
  }
]
```

Use image IDs matching the filenames in the image directory. Answer accuracy accepts answer strings or objects containing an `answer` field.

## Watermarks and probe queries

Prepare watermark images using the acronym or spatial construction in the [AQUA paper](https://arxiv.org/abs/2506.10030), and store each image with its probe queries.

| Watermark type | Image directory | Probe-query directory |
| --- | --- | --- |
| Acronym | `datasets/watermark_images/acronym/` | `datasets/probe_query/acronym/` |
| Spatial | `datasets/watermark_images/spatial/` | `datasets/probe_query/spatial/` |
| Naive | `datasets/watermark_images/naive/` | `datasets/probe_query/naive/` |
| Optimized | `datasets/watermark_images/opt/` | `datasets/probe_query/opt/<family>/` |

For optimized probes, `<family>` is `llava` for LLaVA/TinyLLaVA, `qwen` for Qwen-VL-Chat, `intern` for InternVL, and `qwen25` for Qwen2.5-VL/Qwen3-VL. Normal-query injection also accepts images under `opt/<family>/`.

Each probe-query JSON file contains a list:

```json
[
  {
    "watermark_path": "datasets/watermark_images/acronym/BJT.png",
    "gt": "Bai Jing Ting",
    "probe_query": "Who is BJT? Answer the name related to BJT."
  }
]
```

`watermark_path` identifies the image to inject, `gt` is the expected response text, and `probe_query` is the verification question. Paths resolve from the repository root; both relative and absolute image paths are accepted. Keep `--watermark_type` consistent with the selected directories.

## Attack queries

The robustness script reads JSON lists from `datasets/special_query_attack/<attack>/`. Attack names combine `acronym`, `spatial`, or `opt` with `rescale`, `rotate`, `gaussian`, or `all`. The acronym rescale and rotate directories are named `acronym_rescale_1_5` and `acronym_rotate_45`.

```json
[
  {
    "watermark_path": "datasets/watermark_images_attackd/acronym_all/BJT.png",
    "gt": "Bai Jing Ting",
    "special_query": "Who is BJT? Answer the name related to BJT."
  }
]
```

`watermark_path` points to the transformed image. Pass `--json_dir path/to/queries` to use another directory of attack-query files.

## Index construction

```bash
python -m utils.indexing_faiss --datasets MMQA_ratio --clip_type hf_clip
python -m utils.indexing_faiss --datasets WebQA --clip_type hf_clip
```

The builder creates embeddings, indices for 20%, 40%, 60%, 80%, and 100% of the images, and an image-ID mapping for each ratio. The 100% MMQA index is also saved as `MMQA_all_hf_clip.index` for the core pipeline.

When selecting a ratio index, pass its matching mapping through `--index_mapping_path`. The index and query encoder should use the same model weights.
