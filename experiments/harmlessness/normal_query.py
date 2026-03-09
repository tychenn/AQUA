import argparse
import json
import os
from tqdm import tqdm
from pathlib import Path
import torch
from multimodalrag import MultimodalRAG
from experiments.effectiveness.pvalue import contains_ignoring_case_punctuation_space

NORMAL_QUERY_JSON_PATHS = {
    "MMQA": Path("datasets/MMQA/jsons/MMQA_all_image.json"),
}


def load_normal_queries(dataset, max_examples=None):
    dataset_key = dataset.upper()
    json_path = NORMAL_QUERY_JSON_PATHS.get(dataset_key)
    if json_path is None:
        raise ValueError(f"Normal queries are not configured for dataset '{dataset}'.")
    if not json_path.exists():
        raise FileNotFoundError(f"Normal query file not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    if max_examples is not None and max_examples > 0:
        queries = queries[:max_examples]
    print(f"Loaded {len(queries)} normal queries from {json_path}.")
    return queries


def get_normal_query_output_dir(args):
    output_dir = Path(args.save_dir) / args.dataset / "normal_query"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _extract_answer_texts(example):
    answer_texts = []
    for ans in example.get("answers", []):
        candidate = ans.get("answer") if isinstance(ans, dict) else ans
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate:
                answer_texts.append(candidate)
        elif isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, str):
                    item = item.strip()
                    if item:
                        answer_texts.append(item)
    return answer_texts

def add_watermarks(mmRAG):
    if mmRAG.args.watermark_type=='acronym':
        watermarks_dir="datasets/watermark_images/acronym"
    elif mmRAG.args.watermark_type=='spatial':
        watermarks_dir="datasets/watermark_images/spatial"
    elif mmRAG.args.watermark_type=='opt':
        if mmRAG.args.generator_type in ["LLaVA", "TinyLLaVA-3.1B"]:
            watermarks_dir="datasets/watermark_images/opt/llava"
        elif mmRAG.args.generator_type=="Qwen-VL-Chat":
            watermarks_dir="datasets/watermark_images/opt/qwen" 
        elif mmRAG.args.generator_type=="InternVL3-2B":
            watermarks_dir="datasets/watermark_images/opt/intern"
        elif mmRAG.args.generator_type in ["Qwen2.5-VL-7B-Instruct","Qwen3-VL-32B-Instruct","qwen2.5-vl-finetune"]:
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
        
        
def cal_retrieved_watermark_ratio(mmRAG, normal_queries):
    if not normal_queries:
        print("No normal queries available for watermark retrieval calculation.")
        return 0.0
    watermarks_dir=Path("datasets/watermark_images")
    retrieved_watermark_num=0
    processed_queries=0
    for item in tqdm(normal_queries, desc="Processing questions"):
        question=item.get("question")
        if not question:
            continue
        processed_queries+=1
        with torch.no_grad():
            image_paths,_=mmRAG.retriever(mmRAG.images_database,question)
        
        for image_path in image_paths:
            image_name=image_path.name
            test_path=watermarks_dir/image_name
            if test_path.exists():
                retrieved_watermark_num+=1
        
    ratio=retrieved_watermark_num/processed_queries if processed_queries else 0.0
    print("Retrieved watermark ratio:",ratio)
    output_dir=get_normal_query_output_dir(mmRAG.args)
    output_filepath=output_dir/f"{mmRAG.args.generator_type}_result.txt"
    with open(output_filepath,'w') as f:
        f.write(f"{ratio}")
    return ratio


def evaluate_normal_query_accuracy(mmRAG, normal_queries):
    if not normal_queries:
        print("No normal queries available for accuracy evaluation.")
        return 0.0
    correct_answers=0
    evaluated_queries=0
    skipped_queries=0
    failed_queries=0
    for item in tqdm(normal_queries, desc="Evaluating normal queries"):
        question=item.get("question")
        answer_texts=_extract_answer_texts(item)
        if not question or not answer_texts:
            skipped_queries+=1
            continue
        try:
            with torch.no_grad():
                image_paths,_=mmRAG.retriever(mmRAG.images_database,question)
                output=mmRAG.generator(image_paths, question=question)
        except Exception as exc:
            failed_queries+=1
            tqdm.write(f"Failed to answer qid {item.get('qid','unknown')}: {exc}")
            continue
        evaluated_queries+=1
        if any(contains_ignoring_case_punctuation_space(output,gt) for gt in answer_texts):
            correct_answers+=1
    accuracy=correct_answers/evaluated_queries if evaluated_queries else 0.0
    print("Normal query accuracy:",accuracy)
    print(f"Evaluated {evaluated_queries} queries (skipped={skipped_queries}, failed={failed_queries}).")
    output_dir=get_normal_query_output_dir(mmRAG.args)
    output_filepath=output_dir/f"{mmRAG.args.generator_type}_accuracy.json"
    metrics={
        "accuracy":accuracy,
        "correct":correct_answers,
        "evaluated":evaluated_queries,
        "skipped_missing_answer":skipped_queries,
        "failed_generation":failed_queries,
        "total_loaded_queries":len(normal_queries),
    }
    with open(output_filepath,'w',encoding='utf-8') as f:
        json.dump(metrics,f,ensure_ascii=False,indent=2)
    return accuracy
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMQA", choices=["MMQA","WEBQA"])
    parser.add_argument("--retriever_type", type=str, default="clip", choices=["clip","siglip-so400m-patch14-384"])
    parser.add_argument("--clip_topk", type=int, default=5)
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--index_mapping_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--retriever_device", type=str, default="cuda:2")
    parser.add_argument("--generator_device", type=str, default="cuda:2")
    parser.add_argument("--generator_type", type=str, default="LLaVA", choices=["LLaVA", 
                                                                                    "TinyLLaVA-3.1B",
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "Qwen3-VL-32B-Instruct",
                                                                                    "qwen2.5-vl-finetune",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B"])
    parser.add_argument("--experiment_time", type=int, default=1)
    parser.add_argument("--watermark_type", type=str, default="acronym", choices=["acronym", "spatial", "opt", "naive"])
    parser.add_argument("--max_normal_queries", type=int, default=None)
    args = parser.parse_args()
    watermarked_mmRAG=MultimodalRAG(args)
    print("",watermarked_mmRAG.images_database.ntotal)
    
    add_watermarks(watermarked_mmRAG)
    print("",watermarked_mmRAG.images_database.ntotal)
    normal_queries=load_normal_queries(args.dataset, args.max_normal_queries)
    cal_retrieved_watermark_ratio(watermarked_mmRAG, normal_queries)
    evaluate_normal_query_accuracy(watermarked_mmRAG, normal_queries)
