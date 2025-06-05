
import os
import numpy as np
import re
import json
from pathlib import Path
import torch
import copy
import argparse
from multimodalrag import MultimodalRAG
from experiments.effectiveness.pvalue import contains_ignoring_case_punctuation_space
from tqdm import tqdm 
import random
seed_value = 42 

random.seed(seed_value) 
np.random.seed(seed_value) 
torch.manual_seed(seed_value) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value) 
    torch.cuda.manual_seed_all(seed_value) 

def retrieval_ratio_along_watermark_num(watermarkedmmrag):
    retrieval_ratio_opt_list=[]
    inject_num_list=[1,50,100,1000,10000]
    if watermarkedmmrag.args.watermark_type=="acronym":
        retrieval_ratio_ocr_list=[]
        for i,inject_num in enumerate(inject_num_list):
            tmp_list=[]
            directory_path="datasets/special_query/acronym"
            for jsonname in os.listdir(directory_path):
                json_path = os.path.join(directory_path, jsonname)
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                for item in json_data:
                    tmp_list.append(item["watermark_path"])
            if i==0:
                tmplist2=tmp_list
            elif i==1:
                tmplist2=tmp_list
            elif i==2:
                tmplist2=tmp_list*2
            elif i==3:
                tmplist2=tmp_list*20
            elif i==4:
                tmplist2=tmp_list*200
            ocr_image_list=random.sample(tmplist2, inject_num)
            tmp_database=copy.deepcopy(watermarkedmmrag.images_database)
            print("",tmp_database.ntotal)
            for i in range(inject_num):
                watermarkedmmrag.add_watermark_to_image_database(tmp_database,ocr_image_list[i])
            print("",tmp_database.ntotal)
            if watermarkedmmrag.args.dataset=="WebQA":
                normal_query_json_path="/home/cty/WatermarkmmRAG/datasets/WebQA/jsons/WebQA_all_index_to_image_id.json"
            else:
                normal_query_json_path="datasets/MMQA/jsons/MMQA_all_image.json"
            with open(normal_query_json_path, 'rb') as f:
                normal_query_json = json.load(f)
            all_query_num=0 
            retrieved_num=0
            for item in tqdm(normal_query_json,"normal query中"): 
                image_paths,similarity_json=watermarkedmmrag.retriever(tmp_database,item["question"])
                all_query_num+=1
                image_ids = [p.stem for p in image_paths]
                for i,image_id in enumerate(image_ids):
                    if len(image_id)<=5:
                        retrieved_num+=+1
                        break
            
            print("retrieved_num:",retrieved_num)
            print("all_query_num:",all_query_num)
            retrieval_ratio_ocr_list.append(float(retrieved_num/all_query_num))
        print("",watermarkedmmrag.images_database.ntotal)
        print("",tmp_database.ntotal)
        decimal_places = 30
        for i, ratio in enumerate(retrieval_ratio_opt_list):
            print(f"{i}{ratio:.{decimal_places}f}")
        print(":",retrieval_ratio_ocr_list)
        print(f"{watermarkedmmrag.args.watermark_type}")
    elif watermarkedmmrag.args.watermark_type=="spatial":
        retrieval_ratio_pos_list=[]
        for i,inject_num in enumerate(inject_num_list):
            tmp_list=[]
            directory_path="datasets/special_query/diffusion"
            for jsonname in os.listdir(directory_path):
                json_path = os.path.join(directory_path, jsonname)
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                for item in json_data:
                    tmp_list.append(item["watermark_path"])
            if i==0:
                tmplist2=tmp_list
            elif i==1:
                tmplist2=tmp_list
            elif i==2:
                tmplist2=tmp_list*2
            elif i==3:
                tmplist2=tmp_list*20
            elif i==4:
                tmplist2=tmp_list*200
            ocr_image_list=random.sample(tmplist2, inject_num)
            tmp_database=copy.deepcopy(watermarkedmmrag.images_database)
            print("",tmp_database.ntotal)
            for i in range(inject_num):
                watermarkedmmrag.add_watermark_to_image_database(tmp_database,ocr_image_list[i])
            print("",tmp_database.ntotal)
            normal_query_json_path="datasets/MMQA/jsons/MMQA_all_image.json"
            with open(normal_query_json_path, 'rb') as f:
                normal_query_json = json.load(f)
            all_query_num=0 
            retrieved_num=0
            for item in tqdm(normal_query_json,"normal"): 
                
                image_paths,similarity_json=watermarkedmmrag.retriever(tmp_database,item["question"])
                all_query_num+=1
                absolute_str_image_paths = [str(p.resolve()) for p in image_paths]
                image_ids = [p.stem for p in image_paths]
                for i,image_id in enumerate(image_ids):
                    if len(image_id)<=5:
                        retrieved_num+=+1
                        break
            
            print("retrieved_num:",retrieved_num)
            print("all_query_num:",all_query_num)
            retrieval_ratio_pos_list.append(float(retrieved_num/all_query_num)   )
        print("",watermarkedmmrag.images_database.ntotal)
        print("",tmp_database.ntotal)
        decimal_places = 30
        for i, ratio in enumerate(retrieval_ratio_opt_list):
            print(f"{i}{ratio:.{decimal_places}f}")
        print(":",retrieval_ratio_pos_list)
        print(f"{watermarkedmmrag.args.watermark_type}")
    elif watermarkedmmrag.args.watermark_type=="opt":
        print("")
        for i,inject_num in enumerate(inject_num_list):
            tmp_list=[]
            directory_path="datasets/special_query/optimization/llava"
            for jsonname in os.listdir(directory_path):
                json_path = os.path.join(directory_path, jsonname)
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                for item in json_data:
                    tmp_list.append(item["watermark_path"])
            if i==0:
                tmplist2=tmp_list
            elif i==1:
                tmplist2=tmp_list*8
            elif i==2:
                tmplist2=tmp_list*15
            elif i==3:
                tmplist2=tmp_list*150
            elif i==4:
                tmplist2=tmp_list*1500
            ocr_image_list=random.sample(tmplist2, inject_num)
            tmp_database=copy.deepcopy(watermarkedmmrag.images_database)
            print("",tmp_database.ntotal)
            for i in range(inject_num):
                watermarkedmmrag.add_watermark_to_image_database(tmp_database,ocr_image_list[i])
            print("",tmp_database.ntotal)
            normal_query_json_path="datasets/MMQA/jsons/MMQA_all_image.json"
            with open(normal_query_json_path, 'rb') as f:
                normal_query_json = json.load(f)
            all_query_num=0 
            retrieved_num=0
            for item in tqdm(normal_query_json,"normal "): 
                
                image_paths,similarity_json=watermarkedmmrag.retriever(tmp_database,item["question"])
                all_query_num+=1
                absolute_str_image_paths = [str(p.resolve()) for p in image_paths]
                image_ids = [p.stem for p in image_paths]
                for i,image_id in enumerate(image_ids):
                    if len(image_id)<=5:
                        retrieved_num+=+1
                        break
            
            print("retrieved_num:",retrieved_num)
            print("all_query_num:",all_query_num)
            retrieval_ratio_opt_list.append(float(retrieved_num/all_query_num)   )
        print("",watermarkedmmrag.images_database.ntotal)
        print("",tmp_database.ntotal)
        decimal_places = 30
        for i, ratio in enumerate(retrieval_ratio_opt_list):
            print(f"{i}{ratio:.{decimal_places}f}")
        print(":",retrieval_ratio_opt_list)
        print(f"{watermarkedmmrag.args.watermark_type}")
    elif watermarkedmmrag.args.watermark_type=="baseline":
        print("")
        for i,inject_num in enumerate(inject_num_list):
            tmp_list=[]
            with open("/home/cty/WatermarkmmRAG/datasets/MMQA/jsons/MMQA_all_image.json", 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for item in json_data:
                    for itemm in item["metadata"]["image_doc_ids"]:
                        tmp_list.append(itemm)
            image_paths = []
            for i, image_id in enumerate(tmp_list):
                image_path = None
                base_paths=[
                    Path("datasets/MMQA/images"),
                ]
                for base_path in base_paths:
                    for ext in ['.jpg','.JPG','.Jpg','.jpeg','.JPEG', '.png', '.PNG','.gif','.tif','.tiff']:
                        temp_path = base_path/f"{image_id}{ext}"
                        if temp_path.exists():
                            image_path = temp_path
                            break  
                if image_path: 
                    image_paths.append(image_path)

            ocr_image_list=random.sample(image_paths, inject_num)
            
            tmp_database=copy.deepcopy(watermarkedmmrag.images_database)
            print("",tmp_database.ntotal)
            for i in range(inject_num):
                watermarkedmmrag.add_watermark_to_image_database(tmp_database,ocr_image_list[i])
            print("",tmp_database.ntotal)
            normal_query_json_path="datasets/MMQA/jsons/MMQA_all_image.json"
            with open(normal_query_json_path, 'rb') as f:
                normal_query_json = json.load(f)
            all_query_num=0 
            retrieved_num=0
            for item in tqdm(normal_query_json,"normal query中"): 
                
                image_paths,similarity_json=watermarkedmmrag.retriever(tmp_database,item["question"])
                all_query_num+=1
                absolute_str_image_paths = [str(p.resolve()) for p in image_paths]
                image_ids = [p.stem for p in image_paths]
                for i,image_id in enumerate(image_ids):
                    if image_id in item["metadata"]["image_doc_ids"]:
                        retrieved_num+=+1
                        break
            
            print("retrieved_num:",retrieved_num)
            print("all_query_num:",all_query_num)
            retrieval_ratio_opt_list.append(float(retrieved_num/all_query_num))
        print("",watermarkedmmrag.images_database.ntotal)
        print("",tmp_database.ntotal)
        decimal_places = 30
        for i, ratio in enumerate(retrieval_ratio_opt_list):
            print(f"{i}{ratio:.{decimal_places}f}")
        print(":",retrieval_ratio_opt_list)
        print(f"{watermarkedmmrag.args.watermark_type}")

                    
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_type", type=str, default="clip", help=["clip"])
    parser.add_argument("--clip_topk", type=int, default=3)
    parser.add_argument("--special_queries_file_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--experiment_time", type=int, default=10)
    parser.add_argument("--watermark_num", type=str, default="single", help=["no", "single", "all"])
    
    parser.add_argument("--dataset", type=str, default="WebQA", help=["MMQA","WebQA"])
    parser.add_argument("--max_memory_cuda0", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda1", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda2", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda3", type=str, default="45GB")
    parser.add_argument("--retriever_device", type=str, default="cuda:0")
    parser.add_argument("--generator_device", type=str, default="cuda:0")
    parser.add_argument("--generator_type", type=str, default="None", help=["LLaVA", 
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B",
                                                                                    "None"])
    parser.add_argument("--watermark_type", type=str, default="ocr", help=["ocr", "pos", "opt","baseline"])
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    watermarkedmmrag=MultimodalRAG(args)
    #r=retrieve_rank(watermarkedmmrag)
    r=retrieval_ratio_along_watermark_num(watermarkedmmrag)
    #r=CGSR_along_watermark_num(watermarkedmmrag)
    