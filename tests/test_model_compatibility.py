"""Optional integration regressions using installed Transformers and FAISS on CPU."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from utils.index_metadata import clone_image_database

HAS_MODELS = all(importlib.util.find_spec(name) is not None for name in ("transformers", "faiss", "qwen_vl_utils"))


@unittest.skipUnless(HAS_MODELS, "requires the AQUA model dependencies")
class ModelCompatibilityTests(unittest.TestCase):
    def test_legacy_qwen_generates_with_tuple_cache(self):
        from Qwen_VL_Chat.configuration_qwen import QWenConfig
        from Qwen_VL_Chat.modeling_qwen import QWenLMHeadModel

        config = QWenConfig(
            vocab_size=1104, hidden_size=128, num_hidden_layers=1,
            num_attention_heads=2, kv_channels=64, intermediate_size=256,
            seq_length=32, max_position_embeddings=32, fp32=True,
            use_flash_attn=False, use_dynamic_ntk=False, use_logn_attn=False,
            pad_token_id=0, eos_token_id=None,
            visual=dict(image_size=28, patch_size=14, width=128, layers=1,
                        heads=2, mlp_ratio=2, n_queries=4, output_dim=128,
                        image_start_id=1000),
        )
        model = QWenLMHeadModel(config).eval()
        with torch.no_grad():
            result = model.generate(torch.tensor([[1, 2]]), max_new_tokens=2, do_sample=False)
        self.assertEqual(tuple(result.shape), (1, 4))
        # Four image slots between the image-start and image-end tokens.
        image_input = torch.tensor([[1, 1000, 3, 1002, 1002, 1002, 1001, 2]])
        with patch.object(model.transformer.visual, "encode", return_value=torch.zeros(1, 4, 128)) as encode:
            with torch.no_grad():
                result = model.generate(image_input, max_new_tokens=2, do_sample=False)
        self.assertEqual(tuple(result.shape), (1, 10))
        encode.assert_called_once()

    def test_native_faiss_clone_retains_injected_paths(self):
        import faiss
        from multimodalrag import MultimodalRAG

        class Batch(dict):
            def to(self, device):
                return self

        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.args = SimpleNamespace(retriever_type="clip", dataset="MMQA", clip_topk=5)
        rag.device_map = {"retriever": "cpu"}
        rag.images_database = faiss.IndexFlatIP(2)
        rag.images_database_index_to_image_id = {}
        rag.retriever_vision_processor = lambda **kwargs: Batch()
        rag.retriever_vision_model = lambda **kwargs: SimpleNamespace(image_embeds=torch.tensor([[1., 0.]]))
        rag.retriever_tokenizer = lambda *args, **kwargs: Batch()
        rag.retriever_text_model = lambda **kwargs: SimpleNamespace(text_embeds=torch.tensor([[1., 0.]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long_watermark.png"
            Image.new("RGB", (4, 4)).save(path)
            first = clone_image_database(rag.images_database)
            rag.add_watermark_to_image_database(first, path)
            copied = clone_image_database(first)
            first._aqua_image_paths.clear()
            paths, scores = rag.retriever(copied, "query")
            self.assertEqual(paths, [path.resolve()])
            np.testing.assert_allclose(list(scores.values()), [1.])
            self.assertEqual(rag.images_database.ntotal, 0)


if __name__ == "__main__":
    unittest.main()
