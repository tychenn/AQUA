import argparse
import json
import os
import random
from pathlib import Path

from tqdm import tqdm

from experiments.retrieval_data import (
    IMAGE_SUFFIXES,
    load_query_records,
    normalized_image_path,
    optimization_model_family,
)
from utils.index_metadata import clone_image_database

seed_value = 42
random.seed(seed_value)


def _load_watermark_candidates(args):
    explicit_path = getattr(args, "special_queries_file_path", None)
    if explicit_path:
        json_paths = [Path(explicit_path)]
    else:
        directory = Path("datasets/probe_query") / args.watermark_type
        if args.watermark_type == "opt":
            directory /= optimization_model_family(args.generator_type)
        json_paths = sorted(directory.glob("*.json"))
    paths = []
    for json_path in json_paths:
        with json_path.open(encoding="utf-8") as stream:
            records = json.load(stream)
        if not isinstance(records, list):
            raise ValueError(f"{json_path}: expected a list of probe-query records.")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("watermark_path"), str):
                raise ValueError(f"{json_path}: every probe-query record needs a watermark_path string.")
            paths.append(normalized_image_path(record["watermark_path"]))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("No watermark paths found. Provide probe-query JSON files or --special_queries_file_path.")
    return paths


def _baseline_image_paths(dataset, normal_queries):
    dataset_dir = "WebQA" if dataset.upper() == "WEBQA" else "MMQA"
    root = Path("datasets") / dataset_dir / "images"
    images = {
        path.stem: normalized_image_path(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    ids = {
        str(image_id)
        for query in normal_queries
        for image_id in query.get("metadata", {}).get("image_doc_ids", [])
    }
    return {image_id: images[image_id] for image_id in sorted(ids) if image_id in images}


def retrieval_ratio_along_watermark_num(watermarkedmmrag):
    """Query hit rates as images are injected into independent database copies.

    Watermark modes count queries retrieving any of the images injected for the
    current count. Baseline retains ground-truth image retrieval as its metric.
    """
    args = watermarkedmmrag.args
    if args.watermark_type not in {"acronym", "spatial", "opt", "baseline"}:
        raise ValueError(f"Unsupported watermark type: {args.watermark_type}")
    inject_num_list = getattr(args, "inject_num_list", [1, 50, 100, 1000, 10000])
    if not inject_num_list or any(count <= 0 for count in inject_num_list):
        raise ValueError("Injection counts must be positive integers.")
    normal_queries = load_query_records(args.dataset, getattr(args, "normal_queries_path", None))
    baseline_paths = {}
    if args.watermark_type == "baseline":
        baseline_paths = _baseline_image_paths(args.dataset, normal_queries)
        candidates = list(baseline_paths.values())
    else:
        candidates = _load_watermark_candidates(args)
    if not candidates:
        raise ValueError("No candidate images found for injection.")

    ratios = []
    for inject_num in inject_num_list:
        # Repeat the candidate pool as needed, including small user-provided sets.
        pool = candidates * ((inject_num + len(candidates) - 1) // len(candidates))
        selected_paths = random.sample(pool, inject_num)
        tmp_database = clone_image_database(watermarkedmmrag.images_database)
        for path in selected_paths:
            watermarkedmmrag.add_watermark_to_image_database(tmp_database, str(path))
        injected_paths = set(selected_paths)
        retrieved_num = 0
        for query in tqdm(normal_queries, "normal queries"):
            image_paths, _ = watermarkedmmrag.retriever(tmp_database, query["question"])
            if args.watermark_type == "baseline":
                expected_paths = {
                    baseline_paths[str(image_id)]
                    for image_id in query.get("metadata", {}).get("image_doc_ids", [])
                    if str(image_id) in baseline_paths
                }
            else:
                expected_paths = injected_paths
            if any(normalized_image_path(path) in expected_paths for path in image_paths):
                retrieved_num += 1
        ratio = retrieved_num / len(normal_queries)
        ratios.append(ratio)
        print(f"Injected images: {inject_num}; query hit rate: {ratio}")
    return ratios


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_type", type=str, default="clip", choices=["clip","siglip-so400m-patch14-384"])
    parser.add_argument("--clip_topk", type=int, default=3)
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--index_mapping_path", type=str, default=None)
    parser.add_argument("--special_queries_file_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--experiment_time", type=int, default=10)
    parser.add_argument("--watermark_num", type=str, default="single", choices=["no", "single", "all"])

    parser.add_argument("--dataset", type=str, default="WebQA", choices=["MMQA","WebQA"])
    parser.add_argument("--max_memory_cuda0", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda1", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda2", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda3", type=str, default="45GB")
    parser.add_argument("--retriever_device", type=str, default="cuda:0")
    parser.add_argument("--generator_device", type=str, default="cuda:0")
    parser.add_argument("--generator_type", type=str, default="None", choices=["LLaVA",
                                                                                    "TinyLLaVA-3.1B",
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "Qwen3-VL-32B-Instruct",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B",
                                                                                    "None"])
    parser.add_argument("--watermark_type", type=str, default="acronym", choices=["acronym", "spatial", "opt", "baseline"])
    parser.add_argument("--normal_queries_path", type=str, default=None)
    parser.add_argument("--inject_num_list", type=int, nargs="+", default=[1, 50, 100, 1000, 10000])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    from multimodalrag import MultimodalRAG

    os.makedirs(args.save_dir, exist_ok=True)

    watermarkedmmrag=MultimodalRAG(args)
    #r=retrieve_rank(watermarkedmmrag)
    r=retrieval_ratio_along_watermark_num(watermarkedmmrag)
    #r=CGSR_along_watermark_num(watermarkedmmrag)

