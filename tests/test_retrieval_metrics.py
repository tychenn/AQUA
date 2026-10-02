import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.harmlessness import normal_query, table
from experiments.retrieval_data import load_query_records, watermark_image_directory
from experiments.stealthiness import calculate_retrieval_ratio as stealth


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def create_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


class FakeRAG:
    def __init__(self, args, responses):
        self.args = args
        self.images_database = SimpleNamespace(paths=[])
        self.responses = responses
        self.injections = []
        self.questions = []

    def add_watermark_to_image_database(self, database, path):
        assert Path(path).is_file()
        database.paths.append(Path(path).resolve())
        self.injections.append(Path(path).resolve())

    def retriever(self, database, question):
        self.questions.append(question)
        return self.responses[question], {}


def test_normal_queries_count_actual_spatial_paths_once_per_query(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spatial = Path("datasets/watermark_images/spatial")
    watermark = create_image(spatial / "LONGNAME.png")
    other = create_image(spatial / "other.png")
    create_image(spatial / "notes.txt")
    (spatial / "directory.png").mkdir()
    create_image(Path("datasets/watermark_images/acronym/WRONG.png"))
    ordinary = create_image(Path("datasets/MMQA/images/LONGNAME.png"))
    args = SimpleNamespace(watermark_type="spatial", generator_type="LLaVA", dataset="MMQA", save_dir="results")
    rag = FakeRAG(args, {"hit": [str(watermark.resolve()), other], "miss": [ordinary]})

    injected = normal_query.add_watermarks(rag)
    ratio = normal_query.cal_retrieved_watermark_ratio(
        rag, [{"question": "hit"}, {"question": "miss"}, {"question": ""}]
    )

    assert injected == {watermark.resolve(), other.resolve()}
    assert rag.injections == sorted(injected)
    assert ratio == 0.5
    assert Path("results/MMQA/normal_query/LLaVA_result.txt").read_text() == "0.5"


@pytest.mark.parametrize("generator,family", [
    ("LLaVA", "llava"), ("TinyLLaVA-3.1B", "llava"), ("Qwen-VL-Chat", "qwen"),
    ("InternVL3-8B", "intern"), ("Qwen2.5-VL-32B-Instruct(8bit)", "qwen25"),
    ("Qwen3-VL-32B-Instruct", "qwen25"), ("qwen2.5-vl-finetune", "qwen25"),
])
def test_opt_watermark_directory_follows_generator(tmp_path, monkeypatch, generator, family):
    monkeypatch.chdir(tmp_path)
    expected = Path("datasets/watermark_images/opt") / family
    expected.mkdir(parents=True)
    args = SimpleNamespace(watermark_type="opt", generator_type=generator)
    assert watermark_image_directory(args) == expected


def test_webqa_native_questions_are_normalized(tmp_path):
    path = write_json(tmp_path / "questions.json", {
        "query-1": {"Q": "What is visible?", "A": ["cat"], "img_posFacts": [{"image_id": 42}]}
    })
    queries = load_query_records("WebQA", path)
    assert queries[0]["question"] == "What is visible?"
    assert queries[0]["answers"] == ["cat"]
    assert queries[0]["metadata"]["image_doc_ids"] == ["42"]
    assert queries[0]["qid"] == "query-1"


@pytest.mark.parametrize("data", [{"0": "30240915"}, [{"Q": 12}], [{"question": ""}], []])
def test_invalid_question_files_fail_explicitly(tmp_path, data):
    path = write_json(tmp_path / "questions.json", data)
    with pytest.raises(ValueError):
        load_query_records("WebQA", path)


def test_relative_watermark_ranks_first_and_string_paths_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("experiments/harmlessness").mkdir(parents=True)
    watermark = create_image(Path("datasets/watermark_images/acronym/LONGNAME.png"))
    ordinary = create_image(Path("datasets/MMQA/images/12345.jpg"))
    write_json(Path("datasets/relavent_query/relevant_query_acronym_replace.json"), [
        {"watermark_path": str(watermark), "probe_query": "hit"},
        {"watermark_path": str(watermark), "probe_query": "miss"},
    ])
    rag = FakeRAG(SimpleNamespace(relevant_query_type="acronym_replace", clip_topk=5), {
        "hit": [str(watermark.resolve()), ordinary], "miss": [ordinary]
    })
    assert table.retrieve_rank(rag) == 3.0  # rank 1 and the retained miss score 5
    assert rag.images_database.paths == []


@pytest.mark.parametrize("watermark_type", ["acronym", "spatial", "opt"])
@pytest.mark.parametrize("dataset", ["MMQA", "WebQA"])
def test_stealthiness_uses_injected_paths_and_dataset_questions(tmp_path, monkeypatch, watermark_type, dataset):
    monkeypatch.chdir(tmp_path)
    watermark = create_image(Path("datasets/watermark_images") / watermark_type / "LONGNAME.png")
    ordinary = create_image(Path("datasets") / dataset / "images/12345.jpg")
    probe_dir = Path("datasets/probe_query") / watermark_type
    if watermark_type == "opt":
        probe_dir /= "llava"
    write_json(probe_dir / "queries.json", [{"watermark_path": str(watermark)}])
    if dataset == "WebQA":
        write_json(Path("datasets/WebQA/jsons/WebQA_train_val.json"), {
            "q1": {"Q": "hit"}, "q2": {"Q": "miss"}
        })
    else:
        write_json(Path("datasets/MMQA/jsons/MMQA_all_image.json"), [
            {"question": "hit"}, {"question": "miss"}
        ])
    # A legitimate index mapping must never be read as question records.
    write_json(Path("datasets/WebQA/jsons/WebQA_all_index_to_image_id.json"), {"0": "12345"})
    args = SimpleNamespace(watermark_type=watermark_type, generator_type="None", dataset=dataset, inject_num_list=[1, 3])
    rag = FakeRAG(args, {"hit": [str(watermark.resolve())], "miss": [ordinary]})
    assert stealth.retrieval_ratio_along_watermark_num(rag) == [0.5, 0.5]
    assert rag.questions == ["hit", "miss", "hit", "miss"]
    assert len(rag.injections) == 4
    assert rag.images_database.paths == []


def test_baseline_preserves_ground_truth_retrieval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = create_image(Path("datasets/WebQA/images/101.jpg"))
    second = create_image(Path("datasets/WebQA/images/202.jpg"))
    query_path = write_json(Path("questions.json"), {
        "q1": {"Q": "first", "img_posFacts": [{"image_id": 101}]},
        "q2": {"Q": "second", "img_posFacts": [{"image_id": 202}]},
    })
    args = SimpleNamespace(watermark_type="baseline", dataset="WebQA", inject_num_list=[1], normal_queries_path=query_path)
    rag = FakeRAG(args, {"first": [first], "second": [second]})
    assert stealth.retrieval_ratio_along_watermark_num(rag) == [1.0]


def test_stealthiness_parser_uses_supported_default_and_rejects_ocr():
    parser = stealth.build_parser()
    assert parser.parse_args([]).watermark_type == "acronym"
    assert "--normal_queries_path" in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--watermark_type", "ocr"])
