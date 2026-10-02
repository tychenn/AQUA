"""Compare clean and watermarked responses with an exact paired-query budget."""

import argparse
import json
from pathlib import Path
import re
import warnings

import numpy as np
from scipy import stats
from tqdm import tqdm
from utils.index_metadata import clone_image_database


def probe_directory(args):
    watermark_type = args.watermark_type
    if watermark_type in {"acronym", "acronym_stealthy", "spatial", "naive"}:
        return Path("datasets/probe_query") / watermark_type
    if watermark_type == "opt":
        generator = args.generator_type
        if generator in {"LLaVA", "LLaVA1_5", "TinyLLaVA-3.1B"}:
            family = "llava"
        elif generator == "Qwen-VL-Chat":
            family = "qwen"
        elif generator.startswith("InternVL"):
            family = "intern"
        elif generator.startswith(("Qwen2.5-VL", "Qwen3-VL")) or generator == "qwen2.5-vl-finetune":
            family = "qwen25"
        else:
            raise ValueError(f"No optimized probe queries configured for {generator!r}")
        return Path("datasets/probe_query/opt") / family
    raise ValueError(f"Unsupported watermark type: {watermark_type!r}")


def compare_wsr(clean, watermarked):
    """Return independent test statuses so an unavailable test cannot stop saving."""
    clean = np.asarray(clean, dtype=float)
    watermarked = np.asarray(watermarked, dtype=float)

    def result(status, statistic=float("nan"), pvalue=float("nan")):
        return {"status": status, "statistic": float(statistic), "pvalue": float(pvalue)}

    if clean.size == 0 or watermarked.size == 0:
        return {name: result("unavailable: empty samples") for name in ("welch", "wilcoxon")}
    if clean.size < 2 or watermarked.size < 2:
        return {name: result("unavailable: fewer than two samples") for name in ("welch", "wilcoxon")}

    comparisons = {}
    if np.ptp(clean) == 0 and np.ptp(watermarked) == 0:
        if clean[0] == watermarked[0]:
            comparisons["welch"] = result("identical constant samples", 0, 1)
        else:
            comparisons["welch"] = result("unavailable: both samples have zero variance")
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            test = stats.ttest_ind(clean, watermarked, equal_var=False)
        status = "ok" if np.isfinite(test.pvalue) else "unavailable: non-finite result"
        if caught:
            status += "; " + "; ".join(str(w.message) for w in caught)
        comparisons["welch"] = result(status, test.statistic, test.pvalue)

    if clean.shape != watermarked.shape:
        comparisons["wilcoxon"] = result("unavailable: samples are not paired")
    elif np.array_equal(clean, watermarked):
        comparisons["wilcoxon"] = result("all paired differences are zero", 0, 1)
    else:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                test = stats.wilcoxon(clean, watermarked)
            status = "ok" if np.isfinite(test.pvalue) else "unavailable: non-finite result"
            if caught:
                status += "; " + "; ".join(str(w.message) for w in caught)
            comparisons["wilcoxon"] = result(status, test.statistic, test.pvalue)
        except ValueError as exc:
            comparisons["wilcoxon"] = result(f"unavailable: {exc}")
    return comparisons


def calculate_pvalue(rag, query_times=500, *, directory_path=None,
                     query_key="probe_query", output_file=None, details_file=None):
    """Evaluate at most query_times paired probes across all repetitions.

    Each probe executes one clean and one watermarked retrieval/generation.
    A WSR observation is a processed JSON batch, including a partial final batch.
    Passing query_times=None evaluates every configured repetition.
    """
    if query_times is not None and query_times < 0:
        raise ValueError("query_times must be nonnegative or None")
    if rag.args.experiment_time < 0:
        raise ValueError("experiment_time must be nonnegative")
    directory = Path(directory_path) if directory_path is not None else probe_directory(rag.args)
    if not directory.is_dir():
        raise FileNotFoundError(f"Probe-query directory does not exist: {directory}")
    json_files = sorted(directory.glob("*.json"))
    output_path = Path(output_file) if output_file is not None else Path(
        f"results/effectiveness/pvalue/{rag.args.generator_type}_{rag.args.watermark_type}"
    )
    details_path = Path(details_file) if details_file is not None else output_path.with_name(
        output_path.name + "_wsr_details.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    clean_list, watermark_list = [], []
    query_count = 0
    repetitions_with_queries = set()
    with details_path.open("w", encoding="utf-8") as details:
        for repetition in range(rag.args.experiment_time):
            if query_times is not None and query_count >= query_times:
                break
            for json_path in tqdm(json_files, desc=f"Experiment-{repetition} Calculate pvalue"):
                if query_times is not None and query_count >= query_times:
                    break
                with json_path.open(encoding="utf-8") as source:
                    data = json.load(source)
                if not isinstance(data, list):
                    raise ValueError(f"Expected a list of probe records in {json_path}")
                processed = clean_success = watermark_success = 0
                for item in data:
                    if query_times is not None and query_count >= query_times:
                        break
                    query = item[query_key]
                    clean_paths, _ = rag.retriever(rag.images_database, query)
                    clean_output = rag.generator(clean_paths, query)
                    database = clone_image_database(rag.images_database)
                    rag.add_watermark_to_image_database(database, item["watermark_path"])
                    watermark_paths, _ = rag.retriever(database, query)
                    watermark_output = rag.generator(watermark_paths, query)
                    clean_success += contains_ignoring_case_punctuation_space(clean_output, item["gt"])
                    watermark_success += contains_ignoring_case_punctuation_space(watermark_output, item["gt"])
                    processed += 1
                    query_count += 1
                if not processed:
                    continue
                repetitions_with_queries.add(repetition)
                clean_wsr = clean_success / processed
                watermark_wsr = watermark_success / processed
                clean_list.append(clean_wsr)
                watermark_list.append(watermark_wsr)
                details.write(json.dumps({
                    "file": json_path.name, "repetition": repetition + 1,
                    "queries": processed, "available_queries": len(data),
                    "clean_successes": clean_success, "watermarked_successes": watermark_success,
                    "clean_WSR": clean_wsr, "watermarked_WSR": watermark_wsr,
                }) + "\n")
                details.flush()

    comparisons = compare_wsr(clean_list, watermark_list)
    welch = comparisons["welch"]
    with output_path.open("w", encoding="utf-8") as result_file:
        result_file.write("Overall p-value (Welch's t-test comparing clean and single-watermark WSRs):\n")
        result_file.write(f"{welch['pvalue'] if np.isfinite(welch['pvalue']) else 'N/A'}\n")
        result_file.write(f"Welch status: {welch['status']}\n")
        wilcoxon = comparisons["wilcoxon"]
        result_file.write(f"Wilcoxon p-value: {wilcoxon['pvalue'] if np.isfinite(wilcoxon['pvalue']) else 'N/A'}\n")
        result_file.write(f"Wilcoxon status: {wilcoxon['status']}\n")
        result_file.write(f"Query budget (paired probes): {query_times if query_times is not None else 'unlimited'}\n")
        result_file.write(f"Actual paired queries: {query_count}\n")
        result_file.write(f"Retrieval calls: {2 * query_count}\nGenerator calls: {2 * query_count}\n")
        result_file.write(f"Requested repetitions: {rag.args.experiment_time}\n")
        result_file.write(f"Repetitions with observations: {len(repetitions_with_queries)}\n")
        for label, values in (("no", clean_list), ("single", watermark_list)):
            result_file.write(f"Total '{label}' watermark WSR data points: {len(values)}\n")
            result_file.write(f"Mean WSR ({label} watermark): {np.mean(values) if values else 'N/A'}\n")
            result_file.write(f"Std Dev WSR ({label} watermark): {np.std(values) if values else 'N/A'}\n")
    print(f"Paired queries: {query_count}; WSR samples per condition: {len(clean_list)}")
    print(f"Welch: {welch}; Wilcoxon: {comparisons['wilcoxon']}")
    return welch["statistic"], welch["pvalue"]


def contains_ignoring_case_punctuation_space(response_str, gt_str):
    def preprocess_string(value):
        if isinstance(value, list):
            value = value[0] if value else ""
        value = re.sub(r"[^\w\s]", "", value.lower())
        return "".join(value.split())

    response = preprocess_string(response_str)
    target = preprocess_string(gt_str)
    return bool(target) and target in response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_topk", type=int, default=5)
    parser.add_argument("--experiment_time", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="MMQA", choices=["MMQA","WebQA"])
    parser.add_argument("--retriever_type",type=str,default='clip',choices=['clip','siglip-so400m-patch14-384'])
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--index_mapping_path", type=str, default=None)
    parser.add_argument("--max_memory_cuda0", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda1", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda2", type=str, default="45GB")
    parser.add_argument("--max_memory_cuda3", type=str, default="45GB")
    parser.add_argument("--retriever_device", type=str, default="cuda:2")
    parser.add_argument("--generator_device", type=str, default="cuda:2")
    parser.add_argument("--generator_type", type=str, default="Qwen2.5-VL-7B-Instruct", choices=["LLaVA",
                                                                                    "TinyLLaVA-3.1B",
                                                                                    "Qwen-VL-Chat",
                                                                                    "Qwen2.5-VL-7B-Instruct",
                                                                                    "Qwen2.5-VL-32B-Instruct(8bit)",
                                                                                    "Qwen2.5-VL-32B-Instruct",
                                                                                    "Qwen3-VL-32B-Instruct",
                                                                                    "InternVL3-2B",
                                                                                    "InternVL3-8B",
                                                                                    "None"])
    parser.add_argument("--watermark_type", type=str, default="acronym", choices=["acronym", "acronym_stealthy", "spatial", "opt", "naive"])
    parser.add_argument("--query_times", type=int, default=500, help="Maximum paired probes across all repetitions")
    args = parser.parse_args()
    from multimodalrag import MultimodalRAG
    watermarkedmmrag=MultimodalRAG(args)
    t_statistic, p_value = calculate_pvalue(watermarkedmmrag, query_times=args.query_times)
    print("t_statistic, p_value:", t_statistic, p_value)
