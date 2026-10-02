"""CPU regressions for index artifacts and first-run text PCA."""

import ast
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import pickle
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def temporary_workdir():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        try:
            yield Path(directory)
        finally:
            os.chdir(previous)


class FakeIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.ntotal = 0

    def add(self, vectors):
        self.vectors = vectors.copy()
        self.ntotal = len(vectors)


def load_indexing():
    faiss = types.ModuleType("faiss")
    faiss.IndexFlatIP = FakeIndex
    faiss.written = {}

    def write_index(index, filename):
        # Opening this file also verifies that the production code made its parent.
        with open(filename, "wb") as handle:
            pickle.dump(index.vectors, handle)
        faiss.written[str(filename)] = index

    faiss.write_index = write_index
    transformers = types.ModuleType("transformers")
    transformers.CLIPModel = object
    transformers.CLIPProcessor = object
    spec = importlib.util.spec_from_file_location("aqua_test_indexing", ROOT / "utils/indexing_faiss.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"faiss": faiss, "transformers": transformers}):
        spec.loader.exec_module(module)
    module.device = "cpu"
    return module, faiss


def default_index_paths(dataset):
    tree = ast.parse((ROOT / "multimodalrag.py").read_text())
    registry = next(node.value for node in tree.body
                    if isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "DEFAULT_INDEX_REGISTRY")
    return eval(compile(ast.Expression(registry), "multimodalrag.py", "eval"), {"Path": Path})[dataset]["clip"]


class IndexingTests(unittest.TestCase):
    def test_fresh_image_build_creates_default_index_and_mapping(self):
        for dataset in ("MMQA", "WebQA"):
            with self.subTest(dataset=dataset), temporary_workdir():
                module, faiss = load_indexing()
                image_dir = Path("datasets") / dataset / "images"
                image_dir.mkdir(parents=True)
                for i in range(5):
                    Image.new("RGB", (7, 1) if i == 0 else (7 + i, 8)).save(image_dir / f"image{i}.png")

                seen_sizes = []

                def preprocess(images, return_tensors):
                    seen_sizes.append(images.size)
                    batch = types.SimpleNamespace(pixel_values=torch.tensor([[*images.size, 1]], dtype=torch.float32))
                    batch.to = lambda device: batch
                    return batch

                model = types.SimpleNamespace(get_image_features=lambda pixels: pixels)
                with patch.object(module, "load_clip", return_value=(model, preprocess, None)) as load_model:
                    module.build_ratio_indices(dataset)
                    load_model.assert_called_once()
                    self.assertEqual(load_model.call_args.args[0].clip_type, "hf_clip")

                self.assertIn((7, 10), seen_sizes)
                defaults = default_index_paths(dataset)
                self.assertTrue(defaults["index"].is_file())
                mapping = json.loads(defaults["mapping"].read_text())
                full_index = faiss.written[str(defaults["index"])]
                self.assertEqual(full_index.ntotal, 5)
                self.assertEqual(set(mapping.values()), {f"image{i}" for i in range(5)})
                np.testing.assert_allclose(np.linalg.norm(full_index.vectors, axis=1), 1, rtol=1e-6)
                prefix = "MMQA_ratio" if dataset == "MMQA" else "WebQA"
                for percent, expected_count in ((20, 1), (40, 2), (60, 3), (80, 4), (100, 5)):
                    ratio_index_path = Path("datasets") / dataset / "faiss_index" / f"{prefix}_hf_clip_{percent}%.index"
                    ratio_mapping_path = Path("datasets") / dataset / "jsons" / f"{dataset}_all_index_to_image_id_{percent}%.json"
                    ratio_mapping = json.loads(ratio_mapping_path.read_text())
                    self.assertEqual(faiss.written[str(ratio_index_path)].ntotal, expected_count)
                    self.assertEqual(ratio_mapping, {str(i): mapping[str(i)] for i in range(expected_count)})

                with patch.object(module, "load_clip", side_effect=AssertionError("Cached indices must not load a model")):
                    module.build_ratio_indices(dataset)

    def test_missing_mapping_rebuilds_mmqa_cache(self):
        module, _ = load_indexing()
        with temporary_workdir():
            embedding_path = Path("datasets/MMQA/faiss_index/embeddings.pkl")
            embedding_path.parent.mkdir(parents=True)
            embedding_path.write_bytes(pickle.dumps([np.array([[1, 0]], dtype=np.float32)]))

            def rebuild(clip_type):
                mapping_path = default_index_paths("MMQA")["mapping"]
                mapping_path.parent.mkdir(parents=True)
                mapping_path.write_text(json.dumps({"0": "image0"}))

            with patch.object(module, "build_MMQA_embeddings", side_effect=rebuild) as builder:
                module.build_ratio_indices("MMQA")
                builder.assert_called_once_with(clip_type="hf_clip")

    def test_invalid_ratios_and_empty_embeddings_fail_clearly(self):
        module, _ = load_indexing()
        for ratio in (None, 0, -0.1, 1.1, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, "ratio must"):
                module.ratio_embeddings_to_faiss(ratio=ratio)
        with temporary_workdir():
            Path("empty.pkl").write_bytes(pickle.dumps([]))
            with self.assertRaisesRegex(ValueError, "empty embedding cache"):
                module.ratio_embeddings_to_faiss("empty.pkl", ratio=1)


class TextPCATests(unittest.TestCase):
    def test_first_run_uses_cpu_and_accepts_missing_optional_sources(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        devices = []

        class Batch(dict):
            def to(self, device):
                devices.append(device)
                return self

        class Tokenizer:
            @classmethod
            def from_pretrained(cls, path):
                return cls()

            def __call__(self, texts, **kwargs):
                return Batch(input_ids=torch.tensor([int(texts[0].split()[-1])]))

        class TextModel:
            @classmethod
            def from_pretrained(cls, path):
                return cls()

            def to(self, device):
                devices.append(device)
                return self

            def eval(self):
                return self

            def __call__(self, input_ids):
                value = input_ids.item() + 1
                embedding = torch.tensor([[value, value ** 2, value ** 3, 1]], dtype=torch.float32)
                return types.SimpleNamespace(text_embeds=embedding)

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = Tokenizer
        transformers.CLIPTextModelWithProjection = TextModel
        with temporary_workdir():
            queries = Path("datasets/MMQA/jsons/MMQA_all_image.json")
            queries.parent.mkdir(parents=True)
            queries.write_text(json.dumps([{"question": f"Question {i}"} for i in range(5)]))
            try:
                with patch.dict(sys.modules, {"transformers": transformers}), patch.object(torch.cuda, "is_available", return_value=False):
                    runpy.run_path(str(ROOT / "experiments/stealthiness/PCA_text.py"), run_name="__main__")
                self.assertTrue(Path("experiments/stealthiness/PCA_MMQA_text.png").is_file())
                self.assertTrue(Path("datasets/MMQA/embeddings_cache/normal_query_embeddings.pkl").is_file())
                self.assertTrue(devices)
                self.assertEqual(set(devices), {"cpu"})
            finally:
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
