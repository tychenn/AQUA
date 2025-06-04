
import os
import numpy as np
import re
import json
import copy
import scipy.stats as stats
from scipy.stats import ttest_ind
from tqdm import tqdm 

def calculate_pvalue(MultimodalRAG):
    #save_dir
    wsr_details_file_path = 'results/effectiveness/pvalue/wsr_details'
    wsr_details_dir = os.path.dirname(wsr_details_file_path)
    if wsr_details_dir:
        os.makedirs(wsr_details_dir, exist_ok=True)
    wsr_details_file = open(wsr_details_file_path, 'w', encoding='utf-8')
    
    if MultimodalRAG.args.watermark_type=='acronym':
        directory_path="datasets/probe_query/acronym"
    elif MultimodalRAG.args.watermark_type=='spatial':
        directory_path="datasets/probe_query/spatial"
    elif MultimodalRAG.args.watermark_type=='opt':
        if MultimodalRAG.args.generator_type=="LLaVA":
            directory_path="datasets/probe_query/opt/llava"
        elif MultimodalRAG.args.generator_type=="Qwen-VL-Chat":
            directory_path="datasets/probe_query/opt/qwen" 
        elif MultimodalRAG.args.generator_type=="InternVL3-2B":
            directory_path="datasets/probe_query/opt/intern"
        elif MultimodalRAG.args.generator_type=="Qwen2.5-VL-7B-Instruct":
            directory_path="datasets/probe_query/opt/qwen25"  
    elif MultimodalRAG.args.watermark_type=='naive':
        directory_path="datasets/probe_query/naive"
    else:
        print("error")

    clean_list = []
    watermark_list = []

    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    for i in range(MultimodalRAG.args.experiment_time):
        for json_filename in tqdm(json_files, desc=f"Experiment-{i} Calculate pvalue:"):
            clean_success_num=0
            watermark_success_num=0
            current_json_path = os.path.join(directory_path, json_filename)
            current_file_wsr_results = {}
            with open (current_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    tmpdatabase=copy.deepcopy(MultimodalRAG.images_database)
                    
                    clean_image_paths,_=MultimodalRAG.retriever(MultimodalRAG.images_database,item["probe_query"])
                    clean_output=MultimodalRAG.generator(clean_image_paths,item["probe_query"])
                    
                    MultimodalRAG.add_watermark_to_image_database(tmpdatabase,item["watermark_path"])
                    watermark_image_paths,_=MultimodalRAG.retriever(tmpdatabase,item["probe_query"])
                    watermark_output=MultimodalRAG.generator(watermark_image_paths,item["probe_query"])
                    
                    if contains_ignoring_case_punctuation_space(clean_output,item["gt"]):
                        clean_success_num+=1
                    if contains_ignoring_case_punctuation_space(watermark_output,item["gt"]):
                        watermark_success_num+=1
                clean_list.append(float(clean_success_num/len(data)))
                watermark_list.append(float(watermark_success_num/len(data)))
                
                if wsr_details_file and current_file_wsr_results: 
                    wsr_details_file.write(f"--- File: {json_filename} ---\n")
                    wsr_details_file.write(f"clean_WSR: {current_file_wsr_results.get('no', [])}\n")
                    wsr_details_file.write(f"watermarked_WSR: {current_file_wsr_results.get('single', [])}\n\n")
                    wsr_details_file.flush()
    s1 = np.array(clean_list)
    s2 = np.array(watermark_list)
    t_statistic, p_value = stats.ttest_ind(s1, s2, equal_var=False, nan_policy='omit')
    print(f"t_statistic, p_value:{t_statistic, p_value}")
    
    with open(f'results/effectiveness/pvalue/{MultimodalRAG.args.generator_type}_{MultimodalRAG.args.watermark_type}', 'w', encoding='utf-8') as f_out:
        f_out.write(f"Overall p-value (Welch's t-test comparing 'no' vs 'single' watermark WSRs across all JSON files):\n")
        f_out.write(f"{p_value}\n")
        f_out.write(f"\n--- Aggregated Data Summary ---\n")
        f_out.write(f"Total 'no' watermark WSR data points: {len(clean_list)}\n")
        f_out.write(f"Total 'single' watermark WSR data points: {len(watermark_list)}\n")
        if isinstance(p_value, (float, np.number)): 
            mean_no = np.mean(clean_list) if len(clean_list) > 0 else 'N/A'
            std_no = np.std(clean_list) if len(clean_list) > 0 else 'N/A'
            mean_single = np.mean(watermark_list) if len(watermark_list) > 0 else 'N/A'
            std_single = np.std(watermark_list) if len(watermark_list) > 0 else 'N/A'
            f_out.write(f"Mean WSR (no watermark): {mean_no}\n")
            f_out.write(f"Std Dev WSR (no watermark): {std_no}\n")
            f_out.write(f"Mean WSR (single watermark): {mean_single}\n")
            f_out.write(f"Std Dev WSR (single watermark): {std_single}\n")
    return t_statistic, p_value           

def contains_ignoring_case_punctuation_space(response_str, gt_str):
    def preprocess_string(s):
        if isinstance(s,list):
            s=s[0]
        s = s.lower()
        s = re.sub(r'[^\w\s]', '', s)  
        s = ''.join(s.split())  
        return s

    processed_response_str = preprocess_string(response_str)
    processed_gt_str = preprocess_string(gt_str)

    return processed_gt_str in processed_response_str


    