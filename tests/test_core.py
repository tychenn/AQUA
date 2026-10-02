"""Exercise pipeline branches with real CPU tensors and temporary images."""

import ast
import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from utils.index_metadata import clone_image_database

ROOT = Path(__file__).resolve().parents[1]


def load_core_class():
    """Load the actual class without requiring optional model packages."""
    tree = ast.parse((ROOT / "multimodalrag.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MultimodalRAG")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    namespace = dict(os=os, json=json, gc=gc, Path=Path, np=np, torch=torch, Image=Image,
                     clone_image_database=clone_image_database, tqdm=lambda values: values,
                     timestamp_str="test-run", DATASET_IMAGE_ROOTS={"MMQA": [Path("images")], "WebQA": [Path("images")]})
    exec(compile(module, str(ROOT / "multimodalrag.py"), "exec"), namespace)
    return namespace["MultimodalRAG"]


class FakeIndex:
    def __init__(self):
        self.vectors = np.array([[0., 1.]], dtype="float32")

    @property
    def ntotal(self):
        return len(self.vectors)

    def __deepcopy__(self, memo):
        # Match FAISS serialization, which drops Python attributes.
        copied = FakeIndex()
        copied.vectors = self.vectors.copy()
        return copied

    def add(self, vectors):
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query, k):
        scores = (self.vectors @ query.T).ravel()
        ids = np.argsort(-scores)[:k]
        return (np.array([list(scores[ids]) + [-np.inf] * (k - len(ids))]),
                np.array([list(ids) + [-1] * (k - len(ids))]))


class Batch(dict):
    def to(self, device):
        self.device = device
        return self


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.RAG = load_core_class()

    def setUp(self):
        self.previous = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        os.chdir(self.temporary.name)
        Path("images").mkdir()
        Image.new("RGB", (4, 4)).save("images/same.png")
        Path("attacked").mkdir()
        Image.new("RGB", (4, 4)).save("attacked/same.png")
        Image.new("RGB", (4, 4)).save("attacked/long_watermark_name.png")
        self.rag = self.RAG.__new__(self.RAG)
        self.rag.args = SimpleNamespace(dataset="MMQA", retriever_type="clip", watermark_type="acronym_all",
                                        clip_topk=5, generator_type="test", watermark_num="no",
                                        special_queries_file_path="queries.json")
        self.rag.device_map = {"retriever": "cpu", "generator": "cpu"}
        self.rag.images_database = FakeIndex()
        self.rag.images_database_index_to_image_id = {"0": "same"}
        self.rag.retriever_vision_processor = lambda **kwargs: Batch()
        self.rag.retriever_vision_model = lambda **kwargs: SimpleNamespace(image_embeds=torch.tensor([[1., 0.]]))
        self.rag.retriever_tokenizer = lambda *args, **kwargs: Batch()
        self.rag.retriever_text_model = lambda **kwargs: SimpleNamespace(text_embeds=torch.tensor([[1., 0.]]))

    def tearDown(self):
        os.chdir(self.previous)
        self.temporary.cleanup()

    def test_injected_paths_survive_copy_without_overwriting_base_or_other_index(self):
        first = clone_image_database(self.rag.images_database)
        second = clone_image_database(self.rag.images_database)
        self.rag.add_watermark_to_image_database(first, "attacked/same.png")
        self.rag.add_watermark_to_image_database(second, "attacked/long_watermark_name.png")
        for index, expected in ((first, "same.png"), (second, "long_watermark_name.png"),
                                (clone_image_database(first), "same.png")):
            paths, scores = self.rag.retriever(index, "query")
            self.assertEqual(paths[0], Path("attacked", expected).resolve())
            self.assertEqual(paths[1].resolve(), Path("images/same.png").resolve())
            self.assertEqual(len(paths), 2)  # padded -1 hits are ignored
            self.assertEqual(len(scores), 2)
        self.assertEqual(self.rag.images_database_index_to_image_id, {"0": "same"})
        self.assertEqual(self.rag.images_database.ntotal, 1)
        first._aqua_image_paths.clear()
        self.assertEqual(len(second._aqua_image_paths), 1)

    def test_pipeline_modes_save_images_and_use_each_queries_watermark(self):
        records = [{"special_query": f"question {i}", "watermark_path": path}
                   for i, path in enumerate(["attacked/same.png", "attacked/long_watermark_name.png"])]
        Path("queries.json").write_text(json.dumps(records))
        seen = []
        self.rag.generator = lambda image_paths, question: seen.append(image_paths) or "answer"
        self.rag.cal_retriever_relevance = lambda *args: 0.75
        for method_name in ("run_mmqa", "run_webqa"):
            for mode in ("no", "single", "all"):
                with self.subTest(method=method_name, mode=mode):
                    seen.clear()
                    results = getattr(self.rag, method_name)(is_images=True, watermark_num=mode)
                    key = f"yes_images_{mode}_watermark_response"
                    self.assertEqual([row[key] for row in results], ["answer", "answer"])
                    if mode == "single":
                        self.assertEqual([paths[0].name for paths in seen], ["same.png", "long_watermark_name.png"])
                    if mode == "all":
                        self.assertTrue(all(len(paths) == 3 for paths in seen))
                    images = list(Path("results").glob(f"**/yes_images_{mode}_watermark/images/*"))
                    self.assertTrue(images)
                    self.assertTrue(all(Image.open(path).size == (4, 4) for path in images))
        self.assertEqual(self.rag.images_database.ntotal, 1)

    def test_clean_and_text_modes_do_not_open_missing_watermarks(self):
        Path("queries.json").write_text(json.dumps([{"probe_query": "query", "watermark_path": "missing.png"}]))
        self.rag.generator = lambda image_paths, question: "answer"
        for method_name in ("run_mmqa", "run_webqa"):
            for is_images in (False, True):
                results = getattr(self.rag, method_name)(is_images=is_images, is_write_file=False, watermark_num="no")
                self.assertEqual(len(results), 1)
        self.assertFalse(Path("results").exists())

    def test_empty_queries_return_and_save_empty_list(self):
        Path("queries.json").write_text("[]")
        for method_name in ("run_mmqa", "run_webqa"):
            self.assertEqual(getattr(self.rag, method_name)(is_images=True, watermark_num="single"), [])

    def test_llava_text_inputs_use_model_device(self):
        self.rag.args.generator_type = "LLaVA"
        batch = Batch(input_ids=torch.tensor([[1, 2]]))
        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                return "prompt"
            def __call__(self, **kwargs):
                return batch
            def decode(self, *args, **kwargs):
                return "answer"
        self.rag.generator_processor = Processor()
        self.rag.generator_model = SimpleNamespace(device="cpu", generate=lambda **kwargs: torch.tensor([[1, 2, 3]]))
        self.assertEqual(self.rag.generator(None, "query"), "answer")
        self.assertEqual(batch.device, "cpu")

    def test_retrieval_only_generator_does_not_import_optional_models(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "transformers" or name.startswith("Qwen_VL_Chat"):
                raise ImportError("optional model unavailable")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            self.assertEqual(self.rag.load_generator("None"), (0, 0))
            with self.assertRaisesRegex(ImportError, "Qwen3-VL requires"):
                self.rag.load_generator("Qwen3-VL-32B-Instruct")


if __name__ == "__main__":
    unittest.main()
