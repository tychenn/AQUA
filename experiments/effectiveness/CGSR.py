from tqdm import tqdm
import os
import copy
import json
from experiments.effectiveness.pvalue import contains_ignoring_case_punctuation_space
def calculate_CGSR(watermarkedmmrag):
    if watermarkedmmrag.args.watermark_type=='acronym':
        directory_path="datasets/probe_query/acronym"
    elif watermarkedmmrag.args.watermark_type=='spatial':
        directory_path="datasets/probe_query/spatial"
    elif watermarkedmmrag.args.watermark_type=='opt':
        if watermarkedmmrag.args.generator_type=="LLaVA":
            directory_path="datasets/probe_query/opt/llava"
        elif watermarkedmmrag.args.generator_type=="Qwen-VL-Chat":
            directory_path="datasets/probe_query/opt/qwen" 
        elif watermarkedmmrag.args.generator_type=="InternVL3-2B":
            directory_path="datasets/probe_query/opt/intern"
        elif watermarkedmmrag.args.generator_type=="Qwen2.5-VL-7B-Instruct":
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
    print("retrieved_generated_num:",retrieved_generated_num)
    print("retrieved_num:",retrieved_num)
    return float(retrieved_generated_num/retrieved_num)