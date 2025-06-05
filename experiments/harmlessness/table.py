
import os
import numpy as np
import re
import glob
import json
import torch
import copy
import argparse
from multimodalrag import MultimodalRAG
from tqdm import tqdm 
import random
from experiments.harmlessness.simscore import calculate_simscore
seed_value = 42 

random.seed(seed_value) 
np.random.seed(seed_value) 
torch.manual_seed(seed_value) 

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value) 
    torch.cuda.manual_seed_all(seed_value) 
    
def retrieve_rank(watermarkedmmrag):
    if watermarkedmmrag.args.relevant_query_type=='acronym_replace':
        json_path="datasets/relavent_query/relevant_query_acronym_replace.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    elif watermarkedmmrag.args.relevant_query_type=='acronym_no_instruction':
        json_path="datasets/relavent_query/relevant_query_acronym_no_instruction.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    elif watermarkedmmrag.args.relevant_query_type=='spatial':
        json_dir="datasets/relavent_query/diffusion"
        json_pattern = os.path.join(json_dir, '*.json')
        json_files = glob.glob(json_pattern)
        json_data=[]
        for file in json_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    json_data.append(item)
    elif watermarkedmmrag.args.relevant_query_type=='adv':
        json_dir="datasets/relavent_query/adv"
        json_pattern = os.path.join(json_dir, '*.json')
        json_files = glob.glob(json_pattern)
        json_data=[]
        for file in json_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    json_data.append(item)
    watermark_rank_sum=0
    all_query_times=0
    filename = "experiments/harmlessness/output.txt"
    with open(filename, 'w', encoding='utf-8') as file_object:
        for item in tqdm(json_data,""):
            all_query_times+=1
            tmp_database=copy.deepcopy(watermarkedmmrag.images_database)
            watermarkedmmrag.add_watermark_to_image_database(tmp_database,item["watermark_path"])
            image_paths,similarity_json=watermarkedmmrag.retriever(tmp_database,item["probe_query"])
            absolute_str_image_paths = [str(p.resolve()) for p in image_paths]
            if item["watermark_path"] in absolute_str_image_paths:
                watermark_rank=absolute_str_image_paths.index(item["watermark_path"])+1
                watermark_rank_sum+=watermark_rank
                file_object.write(str(watermark_rank) + '\n')
                print("watermark_rank:",watermark_rank)
            else:
                watermark_rank_sum+=watermarkedmmrag.args.clip_topk
    return float(watermark_rank_sum/all_query_times)



def sim_score(watermarkedmmrag:MultimodalRAG):
    
    if watermarkedmmrag.args.dataset=='MMQA': 
        data:list[dict[str,str]]=[]
        if watermarkedmmrag.args.relevant_query_type=='acronym_replace': 
            relevant_query_json="datasets/relavent_query/relevant_query_acronym_replace.json"
            with open(relevant_query_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif watermarkedmmrag.args.relevant_query_type=='acronym_no_instruction': 
            relevant_query_json="datasets/relavent_query/relevant_query_acronym_no_instruction.json"
            with open(relevant_query_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif watermarkedmmrag.args.relevant_query_type=='spatial': 
            json_dir="datasets/relavent_query/spatial"
            json_pattern = os.path.join(json_dir, '*.json')
            json_files = glob.glob(json_pattern)
            for file in json_files:
                with open(file, 'r', encoding='utf-8') as f:
                    tmpdata = json.load(f)
                    for item in tmpdata:
                        data.append(item) 
        else:
            pass
            
        scores=[]
        for item in tqdm(data,"Relevant query ing:"):
            
            question=item["probe_query"]
            
            tmp_database=copy.deepcopy(watermarkedmmrag.images_database)
            clean_image_paths,_=watermarkedmmrag.retriever(tmp_database,question)
            clean_answer=watermarkedmmrag.generator(clean_image_paths,question)
            
            watermarkedmmrag.add_watermark_to_image_database(tmp_database,item["watermark_path"])
            watermark_image_paths,_=watermarkedmmrag.retriever(tmp_database,question)
            watermark_answer=watermarkedmmrag.generator(watermark_image_paths,question)
        
            score=calculate_simscore(clean_answer,watermark_answer)
            
            pattern = r'\b(?:[1-9]?\d|100)\b'
            found_numbers_str = re.findall(pattern, score)
            for num_str in found_numbers_str:
                num_int = int(num_str)
                scores.append(num_int)

        total_sum = sum(scores)  
        count = len(scores)      
        average = total_sum / count   
        return average
                    
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_type", type=str, default="clip", choices=["clip", "openclip"])
    parser.add_argument("--clip_topk", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="WebQA", choices=["MMQA","WebQA"])
    parser.add_argument("--max_memory_cuda0", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda1", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda2", type=str, default="46GB")
    parser.add_argument("--max_memory_cuda3", type=str, default="45GB")
    parser.add_argument("--retriever_device", type=str, default="cuda:3")
    parser.add_argument("--generator_device", type=str, default="cuda:3")
    parser.add_argument("--generator_type", type=str, default="Qwen2.5-VL-7B-Instruct", choices=["LLaVA", 
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B"])
    parser.add_argument("--relevant_query_type", type=str, default="acronym_replace", choices=["acronym_replace","acronym_no_instruction", "spatial", "adv"])
    args = parser.parse_args()

    watermarkedmmrag=MultimodalRAG(args)
    r=retrieve_rank(watermarkedmmrag)
    #r=CGSR(watermarkedmmrag)
    #r=sim_score(watermarkedmmrag)
    