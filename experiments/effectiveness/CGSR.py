from tqdm import tqdm
from pathlib import Path
from utils.index_metadata import clone_image_database
import json
import argparse
from datetime import datetime, timezone
from experiments.effectiveness.pvalue import contains_ignoring_case_punctuation_space, probe_directory

def calculate_CGSR(watermarkedmmrag):
    directory_path = probe_directory(watermarkedmmrag.args)
    if not directory_path.is_dir():
        raise FileNotFoundError(f"Probe-query directory does not exist: {directory_path}")
    retrieved_num=0
    retrieved_generated_num=0
    all_query_times=0
    for i in range(watermarkedmmrag.args.experiment_time):
        for jsonname in tqdm(sorted(directory_path.glob("*.json")), desc=f"Experiment-{i}, probe querying:"):
            json_path = jsonname
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            for item in json_data:
                all_query_times+=1
                tmp_database = clone_image_database(watermarkedmmrag.images_database)
                watermarkedmmrag.add_watermark_to_image_database(tmp_database, item["watermark_path"])
                image_paths,similarity_json=watermarkedmmrag.retriever(tmp_database,item["probe_query"])
                normalized_paths = [Path(path).resolve() for path in image_paths]
                watermark_path = Path(item["watermark_path"]).resolve()
                if watermark_path in normalized_paths:
                    retrieved_num+=1
                    output=watermarkedmmrag.generator(image_paths,item["probe_query"])
                    if contains_ignoring_case_punctuation_space(output,item["gt"]):
                        retrieved_generated_num+=1
    cgsr = retrieved_generated_num / retrieved_num if retrieved_num else float("nan")
    print("retrieved_generated_num:",retrieved_generated_num)
    print("retrieved_num:",retrieved_num)
    print("CGSR:",cgsr)

    result_path = Path(watermarkedmmrag.args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": watermarkedmmrag.args.dataset,
        "generator_type": watermarkedmmrag.args.generator_type,
        "watermark_type": watermarkedmmrag.args.watermark_type,
        "experiment_time": watermarkedmmrag.args.experiment_time,
        "retrieved_generated_num": retrieved_generated_num,
        "retrieved_num": retrieved_num,
        "query_count": all_query_times,
        "CGSR": cgsr if retrieved_num else None,
        "status": "ok" if retrieved_num else "unavailable: no retrieved watermark"
    }
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_record, ensure_ascii=False) + "\n")

    return cgsr

if __name__ == "__main__":
    from multimodalrag import MultimodalRAG, AVAILABLE_DATASETS

    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_topk", type=int, default=5)
    parser.add_argument("--experiment_time", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="MMQA", choices=AVAILABLE_DATASETS)
    parser.add_argument("--retriever_type",type=str,default='clip',choices=['clip','siglip-so400m-patch14-384'])
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--index_mapping_path", type=str, default=None)
    parser.add_argument("--max_memory_cuda0", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda1", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda2", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda3", type=str, default="45GB")
    parser.add_argument("--retriever_device", type=str, default="cuda:1")
    parser.add_argument("--generator_device", type=str, default="cuda:0")
    parser.add_argument("--generator_type", type=str, default="Qwen3-VL-32B-Instruct", choices=["LLaVA",
                                                                                    "LLaVA1_5",
                                                                                    "TinyLLaVA-3.1B",
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "qwen2.5-vl-finetune",
                                                                                    "Qwen3-VL-2B-Instruct",
                                                                                    "Qwen3-VL-32B-Instruct",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B",
                                                                                    "InternVL3_5-38B",
                                                                                    "None"])
    parser.add_argument("--watermark_type", type=str, default="acronym", choices=["acronym", "acronym_stealthy", "spatial", "opt", "naive"])
    parser.add_argument("--result_file", type=str, default="results/cgsr.log")
    args = parser.parse_args()
    watermarkedmmrag=MultimodalRAG(args)

    CGSR=calculate_CGSR(watermarkedmmrag)
