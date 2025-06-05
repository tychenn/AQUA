
import os
import gc
import numpy as np
import json
import copy
import argparse
from scipy.stats import ttest_ind
from multimodalrag import MultimodalRAG
from experiments.effectiveness.pvalue import contains_ignoring_case_punctuation_space,cal_WSR
from tqdm import tqdm 
def rank(watermarkedmmrag):
    
    if watermarkedmmrag.args.watermark_type=='acronym_rescale':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/acronym_rescale_1_5"
    elif watermarkedmmrag.args.watermark_type=='acronym_rotate':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/acronym_rotate_45"
    elif watermarkedmmrag.args.watermark_type=='acronym_gaussian':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/acronym_gaussian"
    elif watermarkedmmrag.args.watermark_type=='acronym_all':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/acronym_all"
    elif watermarkedmmrag.args.watermark_type=='spatial_rescale':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/spatial_rescale"
    elif watermarkedmmrag.args.watermark_type=='spatial_rotate':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/spatial_rotate"
    elif watermarkedmmrag.args.watermark_type=='spatial_gaussian':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/spatial_gaussian"
    elif watermarkedmmrag.args.watermark_type=='spatial_all':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/spatial_all"
    elif watermarkedmmrag.args.watermark_type=='opt_rescale':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/opt_rescale"
    elif watermarkedmmrag.args.watermark_type=='opt_rotate':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/opt_rotate"
    elif watermarkedmmrag.args.watermark_type=='opt_gaussian':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/opt_gaussian"
    elif watermarkedmmrag.args.watermark_type=='opt_all':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/opt_all"
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
            image_paths,similarity_json=watermarkedmmrag.retriever(tmp_database,item["special_query"])
            image_ids = [p.stem for p in image_paths]
            flag=0
            for i,image_id in enumerate(image_ids):
                if len(image_id)<=16:
                    watermark_rank_sum+=i+1
                    flag=1
                    break
            if flag==0:  
                watermark_rank_sum+=watermarkedmmrag.args.clip_topk
    return float(watermark_rank_sum/all_query_times)

def CGSR(watermarkedmmrag):
    if watermarkedmmrag.args.watermark_type=='acronym_rescale':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/acronym_rescale_1_5"
        image_path_dir="/home/cty/WatermarkmmRAG/datasets/watermark_images_attackd/acronym_rescale_1_5"
    elif watermarkedmmrag.args.watermark_type=='acronym_rotate':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/acronym_rotate_45"
        image_path_dir="/home/cty/WatermarkmmRAG/datasets/watermark_images_attackd/acronym_rotate_45"
    elif watermarkedmmrag.args.watermark_type=='acronym_gaussian':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/acronym_gaussian"
        image_path_dir="/home/cty/WatermarkmmRAG/datasets/watermark_images_attackd/acronym_gaussian"
    elif watermarkedmmrag.args.watermark_type=='acronym_all':
        directory_path="/home/cty/WatermarkmmRAG/datasets/special_query_attack/acronym_all"
        image_path_dir="/home/cty/WatermarkmmRAG/datasets/watermark_images_attackd/acronym_all"
    retrieved_num=0
    retrieved_generated_num=0
    all_query_times=0
    for jsonname in tqdm(os.listdir(directory_path), desc="Processing JSON files"):
        json_path = os.path.join(directory_path, jsonname)
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        tmp_database=copy.deepcopy(watermarkedmmrag.images_database)
        watermarkedmmrag.add_watermark_to_image_database(tmp_database,json_data[0]["watermark_path"])
        for item in json_data:
            all_query_times+=1
            image_paths,similarity_json=watermarkedmmrag.retriever(tmp_database,item["special_query"])
            image_ids = [p.stem for p in image_paths]
            absolute_str_image_paths=[f"{image_path_dir}/{id}.png" for id in image_ids]
            if item["watermark_path"] in absolute_str_image_paths:
                retrieved_num+=1
                output=watermarkedmmrag.generator(image_paths,item["special_query"])
                if contains_ignoring_case_punctuation_space(output,item["gt"]):
                    retrieved_generated_num+=1
    print("retrieved_generated_num:",retrieved_generated_num)
    print("retrieved_num:",retrieved_num)
    print("CGSR",)
    return float(retrieved_generated_num/retrieved_num)

def pvalue(watermarked_mmRAG):
    os.makedirs(args.save_dir, exist_ok=True)

    json_directory = watermarked_mmRAG.args.json_dir
    all_wsr_no = []
    all_wsr_single = []

    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

    wsr_details_file_path = watermarked_mmRAG.args.wsr_details_output_file
    wsr_details_dir = os.path.dirname(wsr_details_file_path)
    if wsr_details_dir:
        os.makedirs(wsr_details_dir, exist_ok=True)


    
    wsr_details_file = open(wsr_details_file_path, 'w', encoding='utf-8')
    for json_filename in tqdm(json_files, desc=""):
        current_json_path = os.path.join(json_directory, json_filename)

        watermarked_mmRAG.args.special_queries_file_path = current_json_path

        current_file_wsr_results = {}
        
        for w_type in ["no", "single"]:
            watermarked_mmRAG.args.watermark_num = w_type
            
            wsr_list = cal_WSR(watermarked_mmRAG)
            
            current_file_wsr_results[w_type] = wsr_list
            if w_type == "no":
                all_wsr_no.extend(wsr_list)
            elif w_type == "single":
                all_wsr_single.extend(wsr_list)
            gc.collect()
            import torch
            torch.cuda.empty_cache() 
        if wsr_details_file and current_file_wsr_results: 
            wsr_details_file.write(f"--- File: {json_filename} ---\n")
            wsr_details_file.write(f"no_watermark_WSR: {current_file_wsr_results.get('no', [])}\n")
            wsr_details_file.write(f"single_watermark_WSR: {current_file_wsr_results.get('single', [])}\n\n")
            wsr_details_file.flush()


    

    if wsr_details_file:
        wsr_details_file.close()
        

    overall_p_value = "N/A"
    if len(all_wsr_no) > 1 and len(all_wsr_single) > 1: 
       
        t_stat, p_value = ttest_ind(all_wsr_no, all_wsr_single, equal_var=False, nan_policy='omit',alternative='two-sided')
        overall_p_value = p_value
            
    output_file_path = args.pvalue_output_file
    output_dir = os.path.dirname(output_file_path)

    if output_dir: 
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f"Overall p-value (Welch's t-test comparing 'no' vs 'single' watermark WSRs across all JSON files):\n")
        f_out.write(f"{overall_p_value}\n")
        f_out.write(f"\n--- Aggregated Data Summary ---\n")
        f_out.write(f"Total 'no' watermark WSR data points: {len(all_wsr_no)}\n")
        f_out.write(f"Total 'single' watermark WSR data points: {len(all_wsr_single)}\n")
        if isinstance(overall_p_value, (float, np.number)):
            mean_no = np.mean(all_wsr_no) if len(all_wsr_no) > 0 else 'N/A'
            std_no = np.std(all_wsr_no) if len(all_wsr_no) > 0 else 'N/A'
            mean_single = np.mean(all_wsr_single) if len(all_wsr_single) > 0 else 'N/A'
            std_single = np.std(all_wsr_single) if len(all_wsr_single) > 0 else 'N/A'
            f_out.write(f"Mean WSR (no watermark): {mean_no}\n")
            f_out.write(f"Std Dev WSR (no watermark): {std_no}\n")
            f_out.write(f"Mean WSR (single watermark): {mean_single}\n")
            f_out.write(f"Std Dev WSR (single watermark): {std_single}\n")
       
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_type", type=str, default="clip", help=["clip", "openclip"])
    parser.add_argument("--reranker_type", type=str, default="LLaVA", help=["LLaVA", "qwen"])
    parser.add_argument("--clip_topk", type=int, default=5)
    parser.add_argument("--special_queries_file_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--experiment_time", type=int, default=3)
    parser.add_argument("--watermark_num", type=str, default="single", help=["no", "single", "all"])
    parser.add_argument("--dataset", type=str, default="MMQA", help=["MMQA","WebQA"])
    parser.add_argument("--max_memory_cuda0", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda1", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda2", type=str, default="46GB")
    parser.add_argument("--max_memory_cuda3", type=str, default="45GB")
    parser.add_argument("--retriever_device", type=str, default="cuda:3")
    parser.add_argument("--generator_device", type=str, default="cuda:3")
    parser.add_argument("--generator_type", type=str, default="LLaVA", help=["LLaVA", 
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B"])
    parser.add_argument("--watermark_type", type=str, default="opt_rescale", help=["acronym_rescale", "acronym_rotate", "acronym_gaussian","acronym_all",
                                                                                  "spatial_rescale","spatial_rotate","spatial_gaussian","spatial_all",
                                                                                  "opt_rescale","opt_rotate","opt_gaussian","opt_all"])
    exp_name="spatial_all"
    parser.add_argument("--json_dir", type=str, default=f"/home/cty/WatermarkmmRAG/datasets/special_query_attack/{exp_name}", help="")
    parser.add_argument("--wsr_details_output_file", type=str, default=f"results/MMQA/robustness/{exp_name}_wsr_details.txt", help="")
    parser.add_argument("--pvalue_output_file", type=str, default=f"results/MMQA/robustness/{exp_name}.txt", help="")

    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    watermarkedmmrag=MultimodalRAG(args)
    #r=rank(watermarkedmmrag)
    #r=CGSR(watermarkedmmrag)
    r=pvalue(watermarkedmmrag)
    