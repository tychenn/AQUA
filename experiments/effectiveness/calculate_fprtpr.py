
import os
import numpy as np
import json
import torch
import copy
import argparse
import random
from multimodalrag import MultimodalRAG
from experiments.effectiveness.pvalue import contains_ignoring_case_punctuation_space
from tqdm import tqdm 
seed_value = 42 

random.seed(seed_value) 
np.random.seed(seed_value)
torch.manual_seed(seed_value) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value) 
    torch.cuda.manual_seed_all(seed_value)
    
def calculate_fpr(watermarkedmmrag):
    if watermarkedmmrag.args.watermark_type=='acronym':
        print("Experiment of acronym fpr")
        directory_path="datasets/probe_query/acronym"
    elif watermarkedmmrag.args.watermark_type=='spatial':
        print("Experiment of spatial fpr")
        directory_path="datasets/probe_query/spatial"
    elif watermarkedmmrag.args.watermark_type=='opt':
        print("Experiment of opt llava fpr")
        directory_path="datasets/probe_query/opt/llava"
    elif watermarkedmmrag.args.watermark_type=='naive':
        print("Experiment of naive fpr")
        directory_path="datasets/probe_query/naive"
    inject_num_list=[1,50,100,500,1000]
    FPRs=[]
    all_querys=[]
    for i in range(len(inject_num_list)):
        print(f"This is FPR experiment-{i}")
        FPR=0
        all_query_num=0
        probe_query_list=[]
        for jsonname in os.listdir(directory_path):
            json_path = os.path.join(directory_path, jsonname)
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            for item in json_data:
                probe_query_list.append(item)
        for item in tqdm(probe_query_list,"Generating the answer"):
            all_query_num+=1
            image_paths,_=watermarkedmmrag.retriever(watermarkedmmrag.images_database,item["probe_query"])
            output=watermarkedmmrag.generator(image_paths,item["probe_query"])
            if contains_ignoring_case_punctuation_space(output,item["gt"]):
                FPR+=1
        FPRs.append(FPR)
        all_querys.append(all_query_num)
        print(f"FPR of this experiment:{FPR}")
    for i in range(len(FPRs)):
        print(f"{i}-th FPR:{FPRs[i]}")
        print(f"{i}-th all query num is:{all_querys[i]}")
    print(f"FPRs:{FPRs}")
    
def calculate_tpr(watermarkedmmrag):
    if watermarkedmmrag.args.watermark_type=='acronym':
        print("Experiment of acronym fpr")
        directory_path="datasets/probe_query/acronym"
    elif watermarkedmmrag.args.watermark_type=='spatial':
        print("Experiment of spatial fpr")
        directory_path="datasets/probe_query/spatial"
    elif watermarkedmmrag.args.watermark_type=='opt':
        print("Experiment of opt llava fpr")
        directory_path="datasets/probe_query/opt/llava"
    elif watermarkedmmrag.args.watermark_type=='naive':
        print("Experiment of naive fpr")
        directory_path="datasets/probe_query/naive"
    inject_num_list=[1,50,100,500,1000]
    TPRs=[]
    all_querys=[]
    items=[]
    for jsonname in os.listdir(directory_path):
        json_path = os.path.join(directory_path, jsonname)
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for item in json_data:
            items.append(item)
    for i,inject_num in enumerate(inject_num_list):
        TPR=0
        all_query_num=0
        for item in tqdm(items,"special qeury们"):
            all_query_num+=1
            tmp_database=copy.deepcopy(watermarkedmmrag.images_database)
            for _ in range(inject_num):
                watermarkedmmrag.add_watermark_to_image_database(tmp_database,item['watermark_path'])
            image_paths,_=watermarkedmmrag.retriever(tmp_database,item["probe_query"])
            output=watermarkedmmrag.generator(image_paths,item["probe_query"])
            if contains_ignoring_case_punctuation_space(output,item["gt"]):
                TPR+=1     
        TPRs.append(TPR)
        all_querys.append(all_query_num)
        print(f"TPR of this experiment:{TPR}")
    for i in range(len(TPRs)):
        print(f"{i}-th FPR:{TPRs[i]}")
        print(f"{i}:{all_querys[i]}")
    print(f"TPRs:{TPRs}")
    

                    
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_type", type=str, default="clip", choices=["clip"])
    parser.add_argument("--clip_topk", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--experiment_time", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="MMQA", choices=["MMQA","WebQA"])
    parser.add_argument("--max_memory_cuda0", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda1", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda2", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda3", type=str, default="45GB")
    parser.add_argument("--retriever_device", type=str, default="cuda:3")
    parser.add_argument("--generator_device", type=str, default="cuda:3")
    parser.add_argument("--generator_type", type=str, default="LLaVA", choices=["LLaVA", 
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B"])
    parser.add_argument("--watermark_type", type=str, default="opt", choices=["spatial", "acronym", "opt","naive"])
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    watermarkedmmrag=MultimodalRAG(args)
    calculate_tpr(watermarkedmmrag)
    calculate_fpr(watermarkedmmrag)