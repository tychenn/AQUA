from tqdm import tqdm
import os
import copy
import json
import argparse
from datetime import datetime
from multimodalrag import MultimodalRAG, AVAILABLE_DATASETS
from experiments.effectiveness.pvalue import contains_ignoring_case_punctuation_space

def calculate_CGSR(watermarkedmmrag):
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
        elif watermarkedmmrag.args.generator_type in ["InternVL3-2B","InternVL3-8B","InternVL3_5-38B"]:
            directory_path="datasets/probe_query/opt/intern"
        elif watermarkedmmrag.args.generator_type in ["Qwen2.5-VL-7B-Instruct","Qwen3-VL-32B-Instruct","qwen2.5-vl-finetune"]:
            directory_path="datasets/probe_query/opt/qwen25"  
    elif watermarkedmmrag.args.watermark_type=='naive':
        directory_path="datasets/probe_query/naive"
    else:
        print("error")
    retrieved_num=0
    retrieved_generated_num=0
    all_query_times=0
    for i in range(watermarkedmmrag.args.experiment_time):
        for jsonname in tqdm(os.listdir(directory_path), desc=f"Experiment-{i}, probe querying:"):
            json_path = os.path.join(directory_path, jsonname)
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            tmp_database=copy.deepcopy(watermarkedmmrag.images_database)
            watermarkedmmrag.add_watermark_to_image_database(tmp_database,json_data[0]["watermark_path"])
            for item in json_data:
                all_query_times+=1
                image_paths,similarity_json=watermarkedmmrag.retriever(tmp_database,item["probe_query"])
                image_paths=[str(path) for path in image_paths]
                if item["watermark_path"] in image_paths:
                    retrieved_num+=1
                    output=watermarkedmmrag.generator(image_paths,item["probe_query"])
                    if contains_ignoring_case_punctuation_space(output,item["gt"]):
                        retrieved_generated_num+=1
    cgsr=float(retrieved_generated_num/retrieved_num)
    print("retrieved_generated_num:",retrieved_generated_num)
    print("retrieved_num:",retrieved_num)
    print("CGSR:",cgsr)

    result_path = watermarkedmmrag.args.result_file
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    result_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": watermarkedmmrag.args.dataset,
        "generator_type": watermarkedmmrag.args.generator_type,
        "watermark_type": watermarkedmmrag.args.watermark_type,
        "experiment_time": watermarkedmmrag.args.experiment_time,
        "retrieved_generated_num": retrieved_generated_num,
        "retrieved_num": retrieved_num,
        "CGSR": cgsr
    }
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_record, ensure_ascii=False) + "\n")

    return cgsr

if __name__ == "__main__":
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
