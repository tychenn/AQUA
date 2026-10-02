"""Evaluate attacked watermarks using their recorded paths and paired probes."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm
from utils.index_metadata import clone_image_database
from experiments.effectiveness.pvalue import (
    calculate_pvalue,
    contains_ignoring_case_punctuation_space,
)


ATTACK_DIRECTORIES = {
    f"{kind}_{attack}": f"{kind}_{attack}"
    for kind in ("acronym", "spatial", "opt")
    for attack in ("rescale", "rotate", "gaussian", "all")
}
ATTACK_DIRECTORIES.update({
    "acronym_rescale": "acronym_rescale_1_5",
    "acronym_rotate": "acronym_rotate_45",
})


def attack_directory(args):
    if args.watermark_type not in ATTACK_DIRECTORIES:
        raise ValueError(f"Unsupported attack watermark type: {args.watermark_type!r}")
    override = getattr(args, "json_dir", None)
    directory = Path(override) if override else (
        Path("datasets/special_query_attack") / ATTACK_DIRECTORIES[args.watermark_type]
    )
    if not directory.is_dir():
        raise FileNotFoundError(f"Attack-query directory does not exist: {directory}")
    return directory


def attack_records(args):
    for json_path in tqdm(sorted(attack_directory(args).glob("*.json")), desc="Processing JSON files"):
        with json_path.open(encoding="utf-8") as source:
            records = json.load(source)
        if not isinstance(records, list):
            raise ValueError(f"Expected a list of query records in {json_path}")
        yield from records


def rank(rag):
    watermark_rank_sum = query_count = 0
    for item in attack_records(rag.args):
        database = clone_image_database(rag.images_database)
        rag.add_watermark_to_image_database(database, item["watermark_path"])
        image_paths, _ = rag.retriever(database, item["special_query"])
        target = Path(item["watermark_path"]).resolve()
        position = next((i + 1 for i, path in enumerate(image_paths)
                         if Path(path).resolve() == target), rag.args.clip_topk)
        watermark_rank_sum += position
        query_count += 1
    return watermark_rank_sum / query_count if query_count else float("nan")


def CGSR(rag):
    retrieved = generated = 0
    for item in attack_records(rag.args):
        database = clone_image_database(rag.images_database)
        rag.add_watermark_to_image_database(database, item["watermark_path"])
        image_paths, _ = rag.retriever(database, item["special_query"])
        target = Path(item["watermark_path"]).resolve()
        if target in {Path(path).resolve() for path in image_paths}:
            retrieved += 1
            output = rag.generator(image_paths, item["special_query"])
            generated += contains_ignoring_case_punctuation_space(output, item["gt"])
    print("retrieved_generated_num:", generated)
    print("retrieved_num:", retrieved)
    return generated / retrieved if retrieved else float("nan")


def pvalue(rag):
    args = rag.args
    result_dir = Path(getattr(args, "save_dir", "results")) / args.dataset / "robustness"
    output_file = getattr(args, "pvalue_output_file", None) or result_dir / f"{args.watermark_type}.txt"
    details_file = getattr(args, "wsr_details_output_file", None) or result_dir / f"{args.watermark_type}_wsr_details.jsonl"
    return calculate_pvalue(
        rag, query_times=getattr(args, "query_times", None),
        directory_path=attack_directory(args), query_key="special_query",
        output_file=output_file, details_file=details_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_type", type=str, default="clip", choices=["clip", "siglip-so400m-patch14-384"])
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--index_mapping_path", type=str, default=None)
    parser.add_argument("--reranker_type", type=str, default="LLaVA", choices=["LLaVA", "qwen"])
    parser.add_argument("--clip_topk", type=int, default=5)
    parser.add_argument("--special_queries_file_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--experiment_time", type=int, default=3)
    parser.add_argument("--watermark_num", type=str, default="single", choices=["no", "single", "all"])
    parser.add_argument("--dataset", type=str, default="MMQA", choices=["MMQA","WebQA"])
    parser.add_argument("--max_memory_cuda0", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda1", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda2", type=str, default="46GB")
    parser.add_argument("--max_memory_cuda3", type=str, default="45GB")
    parser.add_argument("--retriever_device", type=str, default="cuda:3")
    parser.add_argument("--generator_device", type=str, default="cuda:3")
    parser.add_argument("--generator_type", type=str, default="LLaVA", choices=["LLaVA",
                                                                                    "TinyLLaVA-3.1B",
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "Qwen3-VL-32B-Instruct",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B"])
    parser.add_argument("--watermark_type", type=str, default="opt_rescale", choices=sorted(ATTACK_DIRECTORIES))
    parser.add_argument("--json_dir", type=str, default=None, help="Override the selected attack's query directory")
    parser.add_argument("--wsr_details_output_file", type=str, default=None)
    parser.add_argument("--pvalue_output_file", type=str, default=None)
    parser.add_argument("--query_times", type=int, default=None, help="Maximum paired probes; default evaluates all repetitions")

    args = parser.parse_args()

    from multimodalrag import MultimodalRAG

    watermarkedmmrag=MultimodalRAG(args)
    result = pvalue(watermarkedmmrag)
    print("t_statistic, p_value:", result)

