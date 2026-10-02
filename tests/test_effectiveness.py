"""Path and empty-result regressions for effectiveness rank and CGSR."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.effectiveness.CGSR import calculate_CGSR
from experiments.effectiveness.rank import calculate_rank


class FakeRAG:
    def __init__(self, records, *, miss=False, repetitions=1):
        self.args = SimpleNamespace(
            watermark_type="acronym", generator_type="Qwen2.5-VL-7B-Instruct",
            dataset="MMQA", clip_topk=5, experiment_time=repetitions,
            result_file="cgsr.jsonl",
        )
        self.images_database = SimpleNamespace(watermarks=[])
        self.records = {record["probe_query"]: record for record in records}
        self.calls = []
        self.generations = []
        self.miss = miss

    def add_watermark_to_image_database(self, database, path):
        database.watermarks.append(path)

    def retriever(self, database, query):
        self.calls.append((query, list(database.watermarks)))
        if self.miss:
            return [Path("ordinary/12345.png")], {}
        return [Path(database.watermarks[-1]).resolve()], {}

    def generator(self, image_paths, query):
        self.generations.append(query)
        record = self.records[query]
        return record["gt"] if Path(record["watermark_path"]).resolve() in image_paths else "incorrect"


def records_at(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "probes.json").write_text(json.dumps(records))
    (directory / "empty.json").write_text("[]")
    (directory / "notes.txt").write_text("not a probe JSON")


@pytest.fixture
def probes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = [
        {"watermark_path": "watermarks/first.png", "probe_query": "first query", "gt": "first target"},
        {"watermark_path": "watermarks/second.png", "probe_query": "second query", "gt": "second target"},
    ]
    records_at(Path("datasets/probe_query/acronym"), records)
    return records


def test_rank_normalizes_paths_and_injects_each_records_target(probes):
    rag = FakeRAG(probes)
    assert calculate_rank(rag) == 1
    assert rag.calls == [(record["probe_query"], [record["watermark_path"]]) for record in probes]
    assert rag.images_database.watermarks == []


def test_cgsr_normalizes_paths_and_handles_filename_only_log(probes):
    rag = FakeRAG(probes, repetitions=2)
    assert calculate_CGSR(rag) == 1
    expected_calls = [(record["probe_query"], [record["watermark_path"]]) for record in probes]
    assert rag.calls == expected_calls * 2
    assert len(rag.generations) == 4
    record = json.loads(Path("cgsr.jsonl").read_text())
    assert record["retrieved_num"] == record["retrieved_generated_num"] == record["query_count"] == 4
    assert record["CGSR"] == 1 and record["status"] == "ok"
    assert rag.images_database.watermarks == []


def test_missed_watermark_uses_rank_fallback_and_unavailable_cgsr(probes):
    rag = FakeRAG(probes, miss=True)
    assert calculate_rank(rag) == rag.args.clip_topk
    assert math.isnan(calculate_CGSR(rag))
    assert not rag.generations
    record = json.loads(Path("cgsr.jsonl").read_text())
    assert record["CGSR"] is None
    assert record["retrieved_num"] == 0 and record["query_count"] == 2
    assert "unavailable" in record["status"]


def test_empty_probe_lists_are_valid_and_log_nested_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records_at(Path("datasets/probe_query/acronym"), [])
    rag = FakeRAG([])
    rag.args.result_file = "nested/results/cgsr.jsonl"
    assert math.isnan(calculate_rank(rag))
    assert math.isnan(calculate_CGSR(rag))
    assert rag.calls == []
    record = json.loads(Path(rag.args.result_file).read_text())
    assert record["CGSR"] is None and record["query_count"] == 0


@pytest.mark.parametrize("function", [calculate_rank, calculate_CGSR])
def test_optimized_internvl_family_uses_supported_directory(tmp_path, monkeypatch, function):
    monkeypatch.chdir(tmp_path)
    records = [{"watermark_path": "wm/intern.png", "probe_query": "query", "gt": "target"}]
    records_at(Path("datasets/probe_query/opt/intern"), records)
    rag = FakeRAG(records)
    rag.args.watermark_type = "opt"
    rag.args.generator_type = "InternVL3-8B"
    assert function(rag) == 1
