from tqdm import tqdm
import os
import copy
import json
import argparse
from multimodalrag import MultimodalRAG, AVAILABLE_DATASETS

def calculate_rank(watermarkedmmrag):
    
    if watermarkedmmrag.args.watermark_type=='acronym':
        directory_path="datasets/probe_query/acronym"
    elif watermarkedmmrag.args.watermark_type=='acronym_stealthy':
        directory_path="datasets/probe_query/acronym_stealthy"
    elif watermarkedmmrag.args.watermark_type=='spatial':
        directory_path="datasets/probe_query/spatial"
    elif watermarkedmmrag.args.watermark_type=='opt':
        if watermarkedmmrag.args.generator_type in ["LLaVA", "TinyLLaVA-3.1B"]:
            directory_path="datasets/probe_query/opt/llava"
        elif watermarkedmmrag.args.generator_type=="Qwen-VL-Chat":
            directory_path="datasets/probe_query/opt/qwen" 
        elif watermarkedmmrag.args.generator_type=="InternVL3-2B":
            directory_path="datasets/probe_query/opt/intern"
        elif watermarkedmmrag.args.generator_type in ["Qwen2.5-VL-7B-Instruct","Qwen3-VL-32B-Instruct"]:
            directory_path="datasets/probe_query/opt/qwen25"  
    elif watermarkedmmrag.args.watermark_type=='naive':
        directory_path="datasets/probe_query/naive"
    else:
        print("error")
    watermark_rank_sum=0
    all_query_times=0
    for jsonname in tqdm(os.listdir(directory_path), desc="Processing JSON files"):
        json_path = os.path.join(directory_path, jsonname)
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        tmp_database=copy.deepcopy(watermarkedmmrag.images_database)
        watermarkedmmrag.add_watermark_to_image_database(tmp_database,json_data[0]["watermark_path"])
        for item in tqdm(json_data, desc=f"Processing queries from {jsonname}", leave=False): 
            all_query_times+=1
            image_paths,similarity_json=watermarkedmmrag.retriever(tmp_database,item["probe_query"])
            image_paths=[str(path) for path in image_paths]
            if item["watermark_path"] in image_paths:
                watermark_rank=image_paths.index(item["watermark_path"])+1
                watermark_rank_sum+=watermark_rank
            else:
                watermark_rank_sum+=watermarkedmmrag.args.clip_topk
    return float(watermark_rank_sum/all_query_times)

if __name__ == "__main__":
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
    
