
import argparse
from multimodalrag import MultimodalRAG
from experiments.effectiveness.pvalue import calculate_pvalue
from experiments.effectiveness.rank import calculate_rank
from experiments.effectiveness.CGSR import calculate_CGSR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_topk", type=int, default=5)
    parser.add_argument("--experiment_time", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="WebQA", choices=["MMQA","WebQA"])
    parser.add_argument("--retriever_type",type=str,default='clip',choices=['clip'])
    parser.add_argument("--max_memory_cuda0", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda1", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda2", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda3", type=str, default="45GB")
    parser.add_argument("--retriever_device", type=str, default="cuda:2")
    parser.add_argument("--generator_device", type=str, default="cuda:2")
    parser.add_argument("--generator_type", type=str, default="Qwen2.5-VL-7B-Instruct", choices=["LLaVA", 
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B",
                                                                                    "None"])
    parser.add_argument("--watermark_type", type=str, default="acronym", choices=["acronym", "spatial", "opt", "naive"])
    args = parser.parse_args()
    watermarkedmmrag=MultimodalRAG(args)
    #rank=calculate_rank(watermarkedmmrag)#generator_type=="None"
    #CGSR=calculate_CGSR(watermarkedmmrag)
    #t_statistic, p_value=calculate_pvalue(watermarkedmmrag)