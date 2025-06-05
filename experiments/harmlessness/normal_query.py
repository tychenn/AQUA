import argparse
import json
import os
from tqdm import tqdm
from pathlib import Path
import torch
from multimodalrag import MultimodalRAG
from experiments.effectiveness.pvalue import contains_ignoring_case_punctuation_space

def add_watermarks(mmRAG):
    if mmRAG.args.watermark_type=='acronym':
        watermarks_dir="datasets/watermark_images/acronym"
    elif mmRAG.args.watermark_type=='spatial':
        watermarks_dir="datasets/watermark_images/spatial"
    elif mmRAG.args.watermark_type=='opt':
        if mmRAG.args.generator_type=="LLaVA":
            watermarks_dir="datasets/watermark_images/opt/llava"
        elif mmRAG.args.generator_type=="Qwen-VL-Chat":
            watermarks_dir="datasets/watermark_images/opt/qwen" 
        elif mmRAG.args.generator_type=="InternVL3-2B":
            watermarks_dir="datasets/watermark_images/opt/intern"
        elif mmRAG.args.generator_type=="Qwen2.5-VL-7B-Instruct":
            watermarks_dir="datasets/watermark_images/opt/qwen25"  
    elif mmRAG.args.watermark_type=='naive':
        watermarks_dir="datasets/watermark_images/naive"
    else:
        print("error")
    watermarks_dir="datasets/watermark_images/acronym"# You need to complement the path
    watermark_paths=os.listdir(watermarks_dir)
    for watermark_file_name in watermark_paths:
        watermark_path = os.path.join(watermarks_dir, watermark_file_name)
        mmRAG.add_watermark_to_image_database(mmRAG.images_database,watermark_path)
        
        
def cal_retrieved_watermark_ratio(mmRAG):
    
    json_image_all_file_path='./datasets/MMQA/jsons/MMQA_all_image.json'
    with open(json_image_all_file_path, 'r', encoding='utf-8') as f:
        jsons=json.load(f)
    
    watermarks_dir=Path("datasets/watermark_images")
    retrieved_watermark_num=0
    for item in tqdm(jsons, desc="Processing questions"):
        question=item["question"]    
        with torch.no_grad():
            image_paths,_=mmRAG.retriever(mmRAG.images_database,question)
        
        for image_path in image_paths:
            image_name=image_path.name
            test_path=watermarks_dir/image_name
            if test_path.exists():
                retrieved_watermark_num+=1
        
    print(":",retrieved_watermark_num/len(jsons))
    output_filepath=f"results/MMQA/normal_query/{mmRAG.args.generator_type}_result.txt"
    with open(output_filepath,'w') as f:
        f.write(f"{retrieved_watermark_num/len(jsons)}")
        
        
def cal_wrong_answer_ratio(mmRAG):
    json_image_all_file_path='./datasets/MMQA/jsons/MMQA_all_image.json'
    with open(json_image_all_file_path, 'r', encoding='utf-8') as f:
        jsons=json.load(f)
    
    wrong_answer_num=0
    for item in tqdm(jsons, desc="Testing wrong answer:"):
        question=item["question"]    
        with torch.no_grad():
            image_paths,_=mmRAG.retriever(mmRAG.images_database,question)
            output=mmRAG.generator(image_paths, question=question)
        gt_str="Chen Tian Yu"
        if contains_ignoring_case_punctuation_space(output,gt_str):
            wrong_answer_num+=1
        
    print("",wrong_answer_num/len(jsons))
    output_filepath=f"results/MMQA/normal_query/{mmRAG.args.generator_type}_result.txt"
    with open(output_filepath,'w') as f:
        f.write(f"{wrong_answer_num/len(jsons)}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMQA", help=["MMQA","WEBQA"])
    parser.add_argument("--retriever_type", type=str, default="clip", help=["clip"])
    parser.add_argument("--clip_topk", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--retriever_device", type=str, default="cuda:2")
    parser.add_argument("--generator_device", type=str, default="cuda:2")
    parser.add_argument("--generator_type", type=str, default="LLaVA", choices=["LLaVA", 
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B"])
    parser.add_argument("--experiment_time", type=int, default=1)
    parser.add_argument("--watermark_type", type=str, default="acronym", choices=["acronym", "spatial", "opt", "naive"])
    args = parser.parse_args()
    watermarked_mmRAG=MultimodalRAG(args)
    print("",watermarked_mmRAG.images_database.ntotal)
    
    add_watermarks(watermarked_mmRAG)
    print("",watermarked_mmRAG.images_database.ntotal)
    cal_retrieved_watermark_ratio(watermarked_mmRAG)
    #cal_wrong_answer_ratio(watermarked_mmRAG)