from tqdm import tqdm
from pathlib import Path
from utils.index_metadata import clone_image_database
import json
import argparse
from experiments.effectiveness.pvalue import probe_directory

def calculate_rank(watermarkedmmrag):

    directory_path = probe_directory(watermarkedmmrag.args)
    if not directory_path.is_dir():
        raise FileNotFoundError(f"Probe-query directory does not exist: {directory_path}")
    watermark_rank_sum=0
    all_query_times=0
    for jsonname in tqdm(sorted(directory_path.glob("*.json")), desc="Processing JSON files"):
        json_path = jsonname
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for item in tqdm(json_data, desc=f"Processing queries from {jsonname}", leave=False):
            all_query_times+=1
            tmp_database = clone_image_database(watermarkedmmrag.images_database)
            watermarkedmmrag.add_watermark_to_image_database(tmp_database, item["watermark_path"])
            image_paths,similarity_json=watermarkedmmrag.retriever(tmp_database,item["probe_query"])
            normalized_paths = [Path(path).resolve() for path in image_paths]
            watermark_path = Path(item["watermark_path"]).resolve()
            if watermark_path in normalized_paths:
                watermark_rank=normalized_paths.index(watermark_path)+1
                watermark_rank_sum+=watermark_rank
            else:
                watermark_rank_sum+=watermarkedmmrag.args.clip_topk
    return watermark_rank_sum / all_query_times if all_query_times else float("nan")

if __name__ == "__main__":
    from multimodalrag import MultimodalRAG, AVAILABLE_DATASETS

    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_topk", type=int, default=5)
    parser.add_argument("--experiment_time", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="MMQA", choices=AVAILABLE_DATASETS)
    parser.add_argument("--retriever_type",type=str,default='clip',choices=['clip','clip_finetune','siglip-so400m-patch14-384'])
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--index_mapping_path", type=str, default=None)
    parser.add_argument("--max_memory_cuda0", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda1", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda2", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda3", type=str, default="45GB")
    parser.add_argument("--retriever_device", type=str, default="cuda:2")
    parser.add_argument("--generator_device", type=str, default="cuda:2")
    parser.add_argument("--generator_type", type=str, default="Qwen2.5-VL-7B-Instruct", choices=["LLaVA",
                                                                                    "TinyLLaVA-3.1B",
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "Qwen3-VL-32B-Instruct",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B",
                                                                                    "None"])
    parser.add_argument("--watermark_type", type=str, default="spatial", choices=["acronym", "acronym_stealthy", "spatial", "opt", "naive"])
    args = parser.parse_args()
    watermarkedmmrag=MultimodalRAG(args)
    rank=calculate_rank(watermarkedmmrag)#generator_type=="None"
    print("rank=", rank)

