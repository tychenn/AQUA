"""Shared input and path handling for retrieval experiments."""

import json
from pathlib import Path


NORMAL_QUERY_JSON_PATHS = {
    "MMQA": Path("datasets/MMQA/jsons/MMQA_all_image.json"),
    "WEBQA": Path("datasets/WebQA/jsons/WebQA_train_val.json"),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".bmp", ".webp"}


def normalized_image_path(path):
    return Path(path).expanduser().resolve()


def optimization_model_family(generator_type):
    if generator_type in {"LLaVA", "TinyLLaVA-3.1B", "None", None}:
        return "llava"
    if generator_type == "Qwen-VL-Chat":
        return "qwen"
    if generator_type.startswith("InternVL"):
        return "intern"
    if generator_type.startswith(("Qwen2.5-VL", "Qwen3-VL", "qwen2.5-vl")):
        return "qwen25"
    raise ValueError(f"No optimization watermark directory for '{generator_type}'.")


def watermark_image_directory(args):
    root = Path("datasets/watermark_images")
    if args.watermark_type in {"acronym", "spatial", "naive"}:
        return root / args.watermark_type
    if args.watermark_type == "opt":
        family_dir = root / "opt" / optimization_model_family(args.generator_type)
        # Both the documented flat layout and model-specific directories are supported.
        return family_dir if family_dir.is_dir() else root / "opt"
    raise ValueError(f"Unsupported watermark type: {args.watermark_type}")


def load_query_records(dataset, json_path=None, max_examples=None):
    """Load MMQA records or WebQA records keyed by question ID.

    WebQA's Q, A and img_posFacts fields are normalized to the MMQA-style
    question, answers and metadata.image_doc_ids fields used by experiments.
    """
    dataset_key = dataset.upper()
    if dataset_key not in NORMAL_QUERY_JSON_PATHS:
        raise ValueError(f"Normal queries are not configured for dataset '{dataset}'.")
    path = Path(json_path) if json_path else NORMAL_QUERY_JSON_PATHS[dataset_key]
    if not path.is_file():
        raise FileNotFoundError(
            f"Normal query file not found: {path}. "
            "Provide question records with --normal_queries_path."
        )
    with path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if isinstance(records, dict) and dataset_key == "WEBQA":
        items = records.items()
    elif isinstance(records, list):
        items = enumerate(records)
    else:
        raise ValueError(f"{path}: expected a list of question records or a WebQA question-ID dictionary.")

    queries = []
    for key, record in items:
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {key!r} must be a question object, not an image-index mapping.")
        question = record.get("question", record.get("Q"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}: record {key!r} requires a nonempty 'question' or 'Q' string.")
        query = dict(record, question=question)
        query.setdefault("qid", key)
        if dataset_key == "WEBQA":
            answers = record.get("answers", record.get("A", []))
            query["answers"] = [answers] if isinstance(answers, str) else answers
            if "metadata" not in query and "img_posFacts" in record:
                facts = record["img_posFacts"]
                if not isinstance(facts, list) or any(
                    not isinstance(fact, dict) or "image_id" not in fact for fact in facts
                ):
                    raise ValueError(f"{path}: record {key!r} has invalid img_posFacts.")
                query["metadata"] = {"image_doc_ids": [str(fact["image_id"]) for fact in facts]}
        queries.append(query)
    if not queries:
        raise ValueError(f"{path}: no normal question records found.")
    if max_examples is not None and max_examples > 0:
        queries = queries[:max_examples]
    print(f"Loaded {len(queries)} normal queries from {path}.")
    return queries
