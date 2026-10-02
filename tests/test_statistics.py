"""CPU regression checks for exact query budgets and robustness entry points."""

import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.effectiveness import pvalue as effectiveness
from experiments.robustness import table as robustness


class FakeRAG:
    def __init__(self, *, watermark_type="acronym", repetitions=1, identical=False, miss=False):
        self.args = SimpleNamespace(
            watermark_type=watermark_type, generator_type="Qwen2.5-VL-7B-Instruct",
            experiment_time=repetitions, dataset="MMQA", clip_topk=5,
        )
        self.images_database = SimpleNamespace(watermarks=[])
        self.retrievals = []
        self.generations = []
        self.identical = identical
        self.miss = miss

    def add_watermark_to_image_database(self, database, path):
        database.watermarks.append(path)

    def retriever(self, database, query):
        self.retrievals.append((query, list(database.watermarks)))
        paths = [Path("ordinary/12345.jpg")]
        if database.watermarks and not self.miss:
            paths.append(Path(database.watermarks[-1]).resolve())
        return paths, {}

    def generator(self, paths, query):
        self.generations.append(query)
        return "The Target!" if len(paths) > 1 and not self.identical else "other response"


def write_probes(directory, *, files=2, items=20, query_key="probe_query"):
    directory.mkdir(parents=True, exist_ok=True)
    for file_index in range(files):
        records = [
            {"watermark_path": f"watermarks/LONG_WATERMARK_NAME_{file_index}_{i}.png",
             "gt": "the target", query_key: f"query-{file_index}-{i}"}
            for i in range(items)
        ]
        (directory / f"{file_index:03}.json").write_text(json.dumps(records))
    return directory


def result_paths(rag):
    output = Path("results/effectiveness/pvalue") / f"{rag.args.generator_type}_{rag.args.watermark_type}"
    return output, output.with_name(output.name + "_wsr_details.jsonl")


@pytest.mark.parametrize("budget, expected_batches", [(20, [20]), (25, [20, 5]), (500, [20] * 25)])
def test_budget_retains_exact_and_partial_batches(tmp_path, monkeypatch, budget, expected_batches):
    monkeypatch.chdir(tmp_path)
    directory = write_probes(Path("probes"), files=30)
    rag = FakeRAG()
    effectiveness.calculate_pvalue(rag, budget, directory_path=directory)
    output, details = result_paths(rag)
    records = [json.loads(line) for line in details.read_text().splitlines()]
    assert [r["queries"] for r in records] == expected_batches
    assert all(r["clean_WSR"] == 0 and r["watermarked_WSR"] == 1 for r in records)
    assert len(rag.retrievals) == len(rag.generations) == 2 * budget
    assert rag.images_database.watermarks == []
    assert f"Actual paired queries: {budget}" in output.read_text()
    assert f"Total 'no' watermark WSR data points: {len(expected_batches)}" in output.read_text()


def test_budget_applies_across_repetitions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rag = FakeRAG(repetitions=3)
    directory = write_probes(Path("probes"), files=1, items=3)
    effectiveness.calculate_pvalue(rag, 8, directory_path=directory)
    output, details = result_paths(rag)
    records = [json.loads(line) for line in details.read_text().splitlines()]
    assert [r["queries"] for r in records] == [3, 3, 2]
    assert [r["repetition"] for r in records] == [1, 2, 3]
    assert "Repetitions with observations: 3" in output.read_text()
    assert "Requested repetitions: 3" in output.read_text()


@pytest.mark.parametrize("files, items, budget", [(0, 0, 50), (1, 0, 50), (1, 1, 50), (1, 20, 0)])
def test_empty_or_insufficient_samples_save_results(tmp_path, monkeypatch, files, items, budget):
    monkeypatch.chdir(tmp_path)
    rag = FakeRAG()
    directory = write_probes(Path("probes"), files=files, items=items)
    statistic, pvalue = effectiveness.calculate_pvalue(rag, budget, directory_path=directory)
    assert math.isnan(statistic) and math.isnan(pvalue)
    output, details = result_paths(rag)
    assert "unavailable:" in output.read_text()
    assert details.exists()


def test_identical_wsr_is_saved_without_wilcoxon_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rag = FakeRAG(identical=True)
    directory = write_probes(Path("probes"), files=2, items=1)
    statistic, pvalue = effectiveness.calculate_pvalue(rag, 2, directory_path=directory)
    assert (statistic, pvalue) == (0, 1)
    assert "all paired differences are zero" in result_paths(rag)[0].read_text()


def test_optional_wilcoxon_failure_preserves_welch(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("unsupported difference pattern")
    monkeypatch.setattr(effectiveness.stats, "wilcoxon", fail)
    results = effectiveness.compare_wsr([0, .2, .3], [.5, .6, .9])
    assert math.isfinite(results["welch"]["pvalue"])
    assert "unsupported difference pattern" in results["wilcoxon"]["status"]


@pytest.mark.parametrize("watermark_type, subdirectory", robustness.ATTACK_DIRECTORIES.items())
def test_robustness_types_use_matching_directory_and_actual_paths(tmp_path, monkeypatch, watermark_type, subdirectory):
    monkeypatch.chdir(tmp_path)
    directory = write_probes(Path("datasets/special_query_attack") / subdirectory,
                             files=1, items=2, query_key="special_query")
    # Non-query artifacts must be ignored.
    (directory / "notes.txt").write_text("metadata")
    rag = FakeRAG(watermark_type=watermark_type, repetitions=2)
    assert robustness.attack_directory(rag.args) == directory
    assert robustness.rank(rag) == 2
    assert robustness.CGSR(rag) == 1
    statistic, pvalue = robustness.pvalue(rag)
    assert math.isnan(statistic) and math.isnan(pvalue)  # Two unequal constant WSR groups.
    output = Path("results/MMQA/robustness") / f"{watermark_type}.txt"
    assert "Actual paired queries: 4" in output.read_text()
    assert "Total 'single' watermark WSR data points: 2" in output.read_text()
    # Each item has its own watermark, even when multiple items share a JSON file.
    assert any("LONG_WATERMARK_NAME_0_1.png" in paths[0]
               for _, paths in rag.retrievals if paths)


def test_robustness_explicit_paths_and_query_budget(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = write_probes(Path("custom"), files=2, items=3, query_key="special_query")
    rag = FakeRAG(watermark_type="spatial_all")
    rag.args.json_dir = str(directory)
    rag.args.pvalue_output_file = "custom_results/out.txt"
    rag.args.wsr_details_output_file = "custom_results/details.jsonl"
    rag.args.query_times = 4
    robustness.pvalue(rag)
    assert "Actual paired queries: 4" in Path(rag.args.pvalue_output_file).read_text()
    records = [json.loads(line) for line in Path(rag.args.wsr_details_output_file).read_text().splitlines()]
    assert [record["queries"] for record in records] == [3, 1]


def test_robustness_no_hits_and_empty_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = write_probes(Path("custom"), files=1, items=1, query_key="special_query")
    rag = FakeRAG(watermark_type="opt_all", miss=True)
    rag.args.json_dir = str(directory)
    assert robustness.rank(rag) == rag.args.clip_topk
    assert math.isnan(robustness.CGSR(rag))
    (directory / "000.json").write_text("[]")
    assert math.isnan(robustness.rank(rag))
    assert math.isnan(robustness.CGSR(rag))


def test_text_matching_handles_empty_responses_and_targets():
    assert effectiveness.contains_ignoring_case_punctuation_space(["The, Target!"], "the target")
    assert not effectiveness.contains_ignoring_case_punctuation_space([], "target")
    assert not effectiveness.contains_ignoring_case_punctuation_space("anything", "!!!")


@pytest.mark.parametrize("module", ["experiments.effectiveness.pvalue", "experiments.robustness.table"])
def test_metric_cli_help_without_model_dependencies(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--query_times" in result.stdout
