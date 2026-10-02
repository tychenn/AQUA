import os
import json
import faiss
import gc
import torch
import csv
import argparse
from argparse import Namespace
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import (
    AutoProcessor, AutoModel,AutoModelForZeroShotImageClassification, AutoTokenizer, BitsAndBytesConfig,
    CLIPTextModelWithProjection, CLIPVisionModelWithProjection,CLIPProcessor, CLIPModel,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from utils.index_metadata import clone_image_database
import datetime



now = datetime.datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H:%M:%S")

DEFAULT_INDEX_REGISTRY: dict[str, dict[str, dict[str, Path]]] = {
    "MMQA": {
        "clip": {
            "index": Path("datasets/MMQA/faiss_index/MMQA_all_hf_clip.index"),
            "mapping": Path("datasets/MMQA/jsons/WatermarkMMRAG/MMQA_all_index_to_image_id.json"),
        },
        "clip_finetune": {
            "index": Path("datasets/MMQA/faiss_index/MMQA_all_hf_clip.index"),
            "mapping": Path("datasets/MMQA/jsons/WatermarkMMRAG/MMQA_all_index_to_image_id.json"),
        },
        "siglip-so400m-patch14-384": {
            "index": Path("datasets/MMQA/faiss_index/MMQA_all_siglip.index"),
            "mapping": Path("datasets/MMQA/jsons/WatermarkMMRAG/MMQA_all_index_to_image_id_siglip.json"),
        },
    },
    "WebQA": {
        "clip": {
            "index": Path("datasets/WebQA/faiss_index/WebQA_hf_clip_100%.index"),
            "mapping": Path("datasets/WebQA/jsons/WebQA_all_index_to_image_id.json"),
        },
        "clip_finetune": {
            "index": Path("datasets/WebQA/faiss_index/WebQA_hf_clip_100%.index"),
            "mapping": Path("datasets/WebQA/jsons/WebQA_all_index_to_image_id.json"),
        },
        "siglip-so400m-patch14-384": {
            "index": Path("datasets/WebQA/faiss_index/WebQA_all_siglip.index"),
            "mapping": Path("datasets/WebQA/jsons/WebQA_all_index_to_image_id_siglip.json"),
        },
    },
}

WEBQA_SAMPLE_FOLDERS: dict[str, str] = {
    "WebQA10k": "webqa_10000images",
    "WebQA20k": "webqa_20000images",
    "WebQA30k": "webqa_30000images",
    "WebQA40k": "webqa_40000images",
    "WebQA50k": "webqa_50000images",
}

MMQA_WEBQA_COMBINED_FOLDERS: dict[str, str] = {
    "MMQAWebQA10k": "webqa_10000images",
    "MMQAWebQA20k": "webqa_20000images",
    "MMQAWebQA30k": "webqa_30000images",
    "MMQAWebQA40k": "webqa_40000images",
    "MMQAWebQA50k": "webqa_50000images",
}

for dataset_name, folder in WEBQA_SAMPLE_FOLDERS.items():
    root = Path("datasets/MMQA/webqa_samples") / folder
    clip_index = root / "faiss_index" / f"{dataset_name}_clip.index"
    mapping_path = root / "index_to_image_id.json"
    DEFAULT_INDEX_REGISTRY[dataset_name] = {
        "clip": {
            "index": clip_index,
            "mapping": mapping_path,
        },
        "clip_finetune": {
            "index": clip_index,
            "mapping": mapping_path,
        },
    }

for dataset_name, folder in MMQA_WEBQA_COMBINED_FOLDERS.items():
    root = Path("datasets/MMQA/webqa_samples") / folder
    clip_index = root / "faiss_index" / f"{dataset_name}_clip.index"
    mapping_path = root / f"{dataset_name}_index_to_image_id.json"
    DEFAULT_INDEX_REGISTRY[dataset_name] = {
        "clip": {
            "index": clip_index,
            "mapping": mapping_path,
        },
        "clip_finetune": {
            "index": clip_index,
            "mapping": mapping_path,
        },
    }

DATASET_IMAGE_ROOTS: dict[str, list[Path]] = {
    "MMQA": [Path("datasets/MMQA/images")],
    "WebQA": [Path("datasets/WebQA/images")],
}
for dataset_name, folder in WEBQA_SAMPLE_FOLDERS.items():
    DATASET_IMAGE_ROOTS[dataset_name] = [
        Path("datasets/MMQA/webqa_samples") / folder / "images"
    ]
for dataset_name, folder in MMQA_WEBQA_COMBINED_FOLDERS.items():
    DATASET_IMAGE_ROOTS[dataset_name] = [
        Path("datasets/MMQA/images"),
        Path("datasets/MMQA/webqa_samples") / folder / "images",
    ]

AVAILABLE_DATASETS: tuple[str, ...] = tuple(DEFAULT_INDEX_REGISTRY.keys())

class MultimodalRAGArguments:
    retriever_device: str
    generator_device: str
    retriever_type:str
    generator_type:str
    dataset:str
    clip_topk:int
    watermark_type:str
    index_path:str|None
    index_mapping_path:str|None
    
class MultimodalRAG:
    args: MultimodalRAGArguments|Namespace
    device_map: dict[str,str]
    
    def __init__(self, args:MultimodalRAGArguments|Namespace):
        print('\n\n\n')
        self.args = args
        if not hasattr(self.args, "index_path"):
            self.args.index_path = None
        if not hasattr(self.args, "index_mapping_path"):
            self.args.index_mapping_path = None
        self.device_map={
            "retriever":args.retriever_device,
            "generator": args.generator_device,
        }
        
        # Dataset
        self.images_database,\
        self.images_database_index_to_image_id= self.load_index()
        images_num=self.images_database.ntotal
        
        
        #retriever
        self.retriever_model, \
        self.retriever_text_model, \
        self.retriever_tokenizer, \
        self.retriever_vision_model, \
        self.retriever_vision_processor = self.load_retriever()
        
        #generator
        self.generator_model, self.generator_processor = self.load_generator(args.generator_type)

    def _resolve_index_paths(self) -> tuple[Path, Path]:
        custom_index_path = getattr(self.args, "index_path", None)
        custom_mapping_path = getattr(self.args, "index_mapping_path", None)
        if custom_index_path is not None:
            if custom_mapping_path is None:
                raise ValueError("index_mapping_path must be provided when index_path is specified.")
            return Path(custom_index_path), Path(custom_mapping_path)

        dataset_registry = DEFAULT_INDEX_REGISTRY.get(self.args.dataset)
        if dataset_registry is None:
            raise ValueError(f"No FAISS index registry configured for dataset: {self.args.dataset}")
        retriever_registry = dataset_registry.get(self.args.retriever_type)
        if retriever_registry is None:
            available = ", ".join(sorted(dataset_registry.keys()))
            raise ValueError(
                "No default FAISS index configured for "
                f"dataset {self.args.dataset} with retriever {self.args.retriever_type}. "
                f"Known retriever options for this dataset: {available}. "
                "Please pass --index_path/--index_mapping_path to use a custom index."
            )
        return retriever_registry["index"], retriever_registry["mapping"]

    def load_retriever(self)->tuple[
        AutoModelForZeroShotImageClassification,
        CLIPTextModelWithProjection,
        AutoTokenizer,
        CLIPVisionModelWithProjection,
        AutoProcessor
    ]:
        model = None
        text_model = None
        tokenizer = None
        vision_model = None
        vision_processor = None
        retriever_type = self.args.retriever_type
        clip_like_models = {
            "clip": "models/clip-vit-large-patch14-336",
            "clip_finetune": "models/clip-vit-large-patch14-336-finetune",
        }
        if retriever_type in clip_like_models:
            model_path = clip_like_models[retriever_type]
            model = AutoModelForZeroShotImageClassification.from_pretrained(  # type: ignore
                model_path
            ).to(self.device_map["retriever"])
            text_model = CLIPTextModelWithProjection.from_pretrained(
                model_path
            ).to(self.device_map["retriever"])  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore
            vision_model = CLIPVisionModelWithProjection.from_pretrained(
                model_path
            ).to(self.device_map["retriever"])  # type: ignore
            vision_processor = AutoProcessor.from_pretrained(
                model_path, use_fast=True
            )  # type: ignore
        elif retriever_type == "siglip-so400m-patch14-384":
            local_model_dir = Path("models/siglip-so400m-patch14-384")
            model_name = (
                str(local_model_dir)
                if local_model_dir.exists()
                else "google/siglip-so400m-patch14-384"
            )
            model = AutoModel.from_pretrained(model_name).to(self.device_map["retriever"])  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # type: ignore
            vision_processor = AutoProcessor.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")

        for sub_model in (model, text_model, vision_model):
            if sub_model is not None:
                sub_model.eval()
        return model, text_model, tokenizer, vision_model, vision_processor
    
    def load_generator(self, mllm_type):
        if mllm_type.startswith("Qwen3-VL-"):
            try:
                from transformers import Qwen3VLForConditionalGeneration
            except ImportError as exc:
                raise ImportError(
                    "Qwen3-VL requires a Transformers version that includes "
                    "Qwen3VLForConditionalGeneration (for example 4.57.1)."
                ) from exc
        if mllm_type=="None":
            mllm=0
            processor=0
        elif mllm_type == "LLaVA":
            model_name = "models/llava-v1.6-mistral-7b-hf"
            processor = LlavaNextProcessor.from_pretrained(model_name)
            # with init_empty_weights():
            #     mllm=
            mllm = LlavaNextForConditionalGeneration.from_pretrained(
                model_name, 
                device_map=self.device_map["generator"],
                #max_memory=self.max_memory,
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True
            )
            mllm.eval()
        elif mllm_type == "LLaVA1_5":
            model_name = "models/llava-1.5-7b-hf"
            mllm = LlavaForConditionalGeneration.from_pretrained(model_name, device_map=self.device_map["generator"],
                                                                  torch_dtype=torch.float16)
            processor = AutoProcessor.from_pretrained(model_name)
            # with init_empty_weights():
            #     mllm=
            
            mllm.eval()
        elif mllm_type == "Qwen-VL-Chat":
            from Qwen_VL_Chat.modeling_qwen import QWenLMHeadModel

            model_name = "models/Qwen-VL-Chat"   # Qwen-VL-Chat model
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)  # Loads both vision and text processor
            mllm = QWenLMHeadModel.from_pretrained(
                model_name, 
                device_map=self.device_map["generator"],
                trust_remote_code=True, 
                torch_dtype=torch.float16)
            mllm.eval()
        
        elif mllm_type=="Qwen2.5-VL-7B-Instruct":
            model_name="models/Qwen2.5-VL-7B-Instruct"
            mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map=self.device_map["generator"],
                torch_dtype=torch.bfloat16
            )
            processor = AutoProcessor.from_pretrained(model_name,use_fast=True)
            mllm.eval()
        elif mllm_type=="qwen2.5-vl-finetune":
            model_name="models/qwen2.5-vl-finetune"
            mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map=self.device_map["generator"],
                torch_dtype=torch.bfloat16,
            )
            processor = AutoProcessor.from_pretrained(model_name)
            mllm.eval()
        
        elif mllm_type=="Qwen2.5-VL-32B-Instruct(8bit)":
            model_name="models/Qwen2.5-VL-32B-Instruct"
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                # bnb_4bit_compute_dtype=torch.float16, # 移除 4bit 相关的参数
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_use_double_quant=True,
            )
            mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, 
                device_map=self.device_map["generator"],
                quantization_config=quantization_config,
                
                #max_memory=self.max_memory,
                #torch_dtype=torch.bfloat16,
            )   
            processor = AutoProcessor.from_pretrained(model_name)
            mllm.eval()
        elif mllm_type=="Qwen2.5-VL-32B-Instruct":
            model_name="models/Qwen2.5-VL-32B-Instruct"
            device_map_value = "auto" if self.device_map["generator"] == "auto" else self.device_map["generator"]
            mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map=device_map_value,
                torch_dtype=torch.bfloat16,
            )
            processor = AutoProcessor.from_pretrained(model_name)
            mllm.eval()
        elif mllm_type=="Qwen3-VL-2B-Instruct":
            local_model_dir = Path("models/Qwen3-VL-2B-Instruct")
            model_name = str(local_model_dir) if local_model_dir.exists() else "Qwen/Qwen3-VL-2B-Instruct"
            mllm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map=self.device_map["generator"],
                torch_dtype="auto",
            )
            processor = AutoProcessor.from_pretrained(model_name)
            mllm.eval()
        elif mllm_type=="Qwen3-VL-32B-Instruct":
            local_model_dir = Path("models/Qwen3-VL-32B-Instruct")
            model_name = str(local_model_dir) if local_model_dir.exists() else "Qwen/Qwen3-VL-32B-Instruct"
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            mllm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map=self.device_map["generator"],
                quantization_config=quantization_config,
            )
            processor = AutoProcessor.from_pretrained(model_name)
            mllm.eval()
        elif mllm_type=="InternVL3-2B":
            model_name="models/InternVL3-2B"
            processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            mllm=AutoModel.from_pretrained(
                model_name,
                device_map=self.device_map["generator"],
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                #load_in_8bit=True,
            )
            mllm.eval()
        elif mllm_type=="InternVL3-8B":
            model_name="models/InternVL3-8B"
            processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            mllm=AutoModel.from_pretrained(
                model_name,
                device_map=self.device_map["generator"],
                #torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
            )
            mllm.eval()
        elif mllm_type=="InternVL3_5-38B":
            model_name="models/InternVL3_5-38B"
            processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            device_map_value = "auto" if self.device_map["generator"] == "auto" else self.device_map["generator"]
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            mllm=AutoModel.from_pretrained(
                model_name,
                device_map=device_map_value,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                use_flash_attn=True,
            )
            mllm.eval()
        elif mllm_type=="TinyLLaVA-3.1B":
            try:
                from tinyllava.model import load_pretrained_model
            except ImportError as exc:
                raise ImportError(
                    "tinyllava is required for TinyLLaVA-3.1B. "
                    "Please install TinyLLaVABench (pip install -e .) before selecting this generator."
                ) from exc
            local_model_dir = Path("models/TinyLLaVA-3.1B")
            model_path = str(local_model_dir) if local_model_dir.exists() else "bczhou/TinyLLaVA-3.1B"
            model, tokenizer, image_processor, context_len = load_pretrained_model(
                model_name_or_path=model_path,
            )
            if self.device_map["generator"] != "auto":
                model = model.to(self.device_map["generator"])
            processor = {
                "tokenizer": tokenizer,
                "image_processor": image_processor,
                "context_len": context_len,
                "conv_mode": "phi",
                "model_path": model_path,
            }
            mllm = model
            mllm.eval()
        
        else:
            raise ValueError(f"Unsupported generator type: {mllm_type}")
        return mllm, processor
    
    def load_index(self)->tuple[faiss.Index,dict[str,str]]:
        index_path, mapping_path = self._resolve_index_paths()
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. Provide a valid path via --index_path."
            )
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Index-to-image-id mapping not found at {mapping_path}. Provide a valid path via --index_mapping_path."
            )

        index = faiss.read_index(str(index_path))
        with open(mapping_path, "r", encoding="utf-8") as f:
            index_to_image_id = json.load(f)

        print(f"Loaded {self.args.dataset} FAISS index from {index_path}.")
        return index,index_to_image_id

    def add_watermark_to_image_database(self, images_database,watermark_path):
        
        assert os.path.exists(watermark_path), f"Image path {watermark_path} does not exist."

        try:
            watermark = Image.open(watermark_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Failed to open watermark image {watermark_path}: {exc}") from exc
        if self.args.retriever_type in {"clip", "clip_finetune"}:
            inputs = self.retriever_vision_processor(images=watermark, return_tensors="pt").to(self.device_map["retriever"])
            outputs = self.retriever_vision_model(**inputs)
            image_embeds = outputs.image_embeds
        elif self.args.retriever_type == "siglip-so400m-patch14-384":
            processor_inputs = self.retriever_vision_processor(images=watermark, return_tensors="pt")
            pixel_values = processor_inputs["pixel_values"].to(self.device_map["retriever"])
            with torch.no_grad():
                image_embeds = self.retriever_model.get_image_features(pixel_values=pixel_values)
        else:
            raise ValueError(f"Unsupported retriever type: {self.args.retriever_type}")
        normalized_embedding = image_embeds / image_embeds.norm(
                dim=-1, keepdim=True
        )
        normalized_embedding = normalized_embedding.cpu().detach().numpy().astype("float32")

        images_database.add(normalized_embedding)
        
        if not hasattr(images_database, "_aqua_image_paths"):
            images_database._aqua_image_paths = {}
        images_database._aqua_image_paths[images_database.ntotal - 1] = Path(watermark_path).resolve()
        
    def retriever(self, images_database,question):
        
        # text->embedding
        text_inputs = None
        text_outputs = None
        with torch.no_grad():
            if self.args.retriever_type in {"clip", "clip_finetune"}:
                text_inputs = self.retriever_tokenizer([question], return_tensors="pt").to(self.device_map["retriever"]) #type:ignore
                text_outputs = self.retriever_text_model(**text_inputs)
                text_embeds = text_outputs.text_embeds
            elif self.args.retriever_type == "siglip-so400m-patch14-384":
                text_inputs = self.retriever_tokenizer([question], return_tensors="pt", padding="max_length", truncation=True).to(self.device_map["retriever"]) #type:ignore
                text_embeds = self.retriever_model.get_text_features(**text_inputs)
            else:
                raise ValueError(f"Unsupported retriever type: {self.args.retriever_type}")
        # normalization
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)#type:ignore
        text_embeddings = text_embeds.cpu().detach().numpy().astype("float32")
        # search
        similarity_scores_list, indices_list = images_database.search(text_embeddings, self.args.clip_topk)
        similarity_json = {}
        image_paths = []
        injected_paths = getattr(images_database, "_aqua_image_paths", {})
        for score, index_id in zip(similarity_scores_list[0], indices_list[0]):
            # FAISS pads short result sets with -1.
            if index_id < 0:
                continue
            if int(index_id) in injected_paths:
                image_path = injected_paths[int(index_id)]
                image_id = image_path.stem
            else:
                image_id = str(self.images_database_index_to_image_id[str(index_id)])
                dataset_bases = DATASET_IMAGE_ROOTS.get(self.args.dataset)
                if not dataset_bases:
                    raise ValueError(f"Unknown dataset '{self.args.dataset}'.")
                image_path = next((
                    base / f"{image_id}{ext}"
                    for base in dataset_bases
                    for ext in ['.jpg', '.JPG', '.Jpg', '.jpeg', '.JPEG', '.png', '.PNG', '.gif', '.tif', '.tiff']
                    if (base / f"{image_id}{ext}").is_file()
                ), None)
            if image_path is None or not image_path.is_file():
                raise FileNotFoundError(f"Image file not found for ID: {image_id}")
            image_paths.append(image_path)
            score_key = image_id if image_id not in similarity_json else str(image_path.resolve())
            similarity_json[score_key] = float(score)
        
        
        if text_inputs is not None:
            del text_inputs
        if text_outputs is not None:
            del text_outputs
        del text_embeds 
        del text_embeddings

        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        return image_paths,similarity_json

    def generator(self, image_paths=None, question=None)->str:  
        
        if self.args.generator_type in [ "LLaVA" , "LLaVA1_5"]:
            
            if image_paths is None:
                question=(
                    f"Respond to the question: {question}\n"
                    f"Answer the question using phrase."
                )
                conversation=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}],
                    },
                ]
                prompt = self.generator_processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.generator_processor(text=prompt, return_tensors="pt").to(self.generator_model.device)
            
            else:
                images = [Image.open(image_path) for image_path in image_paths]
                question = ( 
                    f"{question}"
                ) 
                conversation = [
                    {
                        "role": "user",
                        "content": 
                            [ {"type": "image"} for _ in range(len(images)) ] + 
                            [{"type": "text", "text": question}],
                    },
                ]
                prompt = self.generator_processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.generator_processor(images=images, text=prompt, return_tensors="pt").to(self.generator_model.device)
            
            
            #output = self.generator_model.generate(**inputs, max_new_tokens=300,num_beams=3,do_sample=True)
            output = self.generator_model.generate(**inputs, max_new_tokens=300,do_sample=True,num_beams=2, temperature=1.0)

            text_outputs = []
            for j, cur_input_tokens in enumerate(inputs['input_ids']):
                prompt_len = len(cur_input_tokens)
                cur_output = output[j][prompt_len:]
                text_output = self.generator_processor.decode(cur_output, skip_special_tokens=True)
                text_outputs.append(text_output)

            return text_outputs[0]
        
        elif self.args.generator_type == "Qwen-VL-Chat":
            from qwenvl.run_qwenvl import qwen_chat

            if image_paths is None:
                question = ( 
                    f"{question}\n"
                )
                mllm_tokenizer = AutoTokenizer.from_pretrained("models/Qwen-VL-Chat", trust_remote_code=True)
                output = qwen_chat(image_paths, question, self.generator_model, mllm_tokenizer)
            else:
                question = ( 
                    f"{question}"
                )
                mllm_tokenizer = AutoTokenizer.from_pretrained("models/Qwen-VL-Chat", trust_remote_code=True)
                tmp_list=[]
                for item in image_paths:
                    tmp_list.append(str(item))
                image_paths=tmp_list
                output = qwen_chat(image_paths, question, self.generator_model, mllm_tokenizer)
            return output
        
        elif self.args.generator_type in["Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct(8bit)","Qwen2.5-VL-32B-Instruct","qwen2.5-vl-finetune"]:
            if image_paths is None:
                messages = [
                    {
                        "role": "user",
                        "content": 
                            [{"type": "text", "text": question},]
                    }
                ]
                text = self.generator_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                first_param_device = next(self.generator_model.parameters()).device
                inputs = self.generator_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(first_param_device)
                #param_device = next(self.generator_model.parameters()).device
                #inputs = {k: v.to(param_device) for k, v in inputs.items()}
                generated_ids = self.generator_model.generate(**inputs, max_new_tokens=128,do_sample=False,num_beams=1,temperature=1.0)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.generator_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            else:
                messages = [
                    {
                        "role": "user",
                        "content": 
                            [{"type": "image", "image": str(image_path)} for image_path in image_paths]+
                            [{"type": "text", "text": question},]
                    }
                ]
                text = self.generator_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                first_param_device = next(self.generator_model.parameters()).device
                inputs = self.generator_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(first_param_device)
                with torch.no_grad():
                    generated_ids = self.generator_model.generate(**inputs, max_new_tokens=128,do_sample=True,num_beams=1,temperature=1.2)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = self.generator_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                del inputs
                del generated_ids
                del generated_ids_trimmed
                del image_inputs 
                del video_inputs 

                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # print(f"Generator: After cleanup - Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB") # Debug
            return output_text
        elif self.args.generator_type in ["Qwen3-VL-2B-Instruct", "Qwen3-VL-32B-Instruct"]:
            content: list[dict[str, str]] = []
            if image_paths is not None:
                content.extend([{"type": "image", "image": str(image_path)} for image_path in image_paths])
            content.append({"type": "text", "text": question})
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            inputs = self.generator_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            first_param_device = next(self.generator_model.parameters()).device
            inputs = inputs.to(first_param_device)
            with torch.no_grad():
                generated_ids = self.generator_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.generator_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            del inputs
            del generated_ids
            del generated_ids_trimmed
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return output_text
        elif self.args.generator_type in ["InternVL3-2B","InternVL3-8B","InternVL3_5-38B"]:
            assert image_paths is not None
            def build_transform(input_size):
                import torchvision.transforms as T
                IMAGENET_MEAN = (0.485, 0.456, 0.406)
                IMAGENET_STD = (0.229, 0.224, 0.225)    
                MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
                from torchvision.transforms.functional import InterpolationMode
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD)
                ])
                return transform
            def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
                best_ratio_diff = float('inf')
                best_ratio = (1, 1)
                area = width * height
                for ratio in target_ratios:
                    target_aspect_ratio = ratio[0] / ratio[1]
                    ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                    if ratio_diff < best_ratio_diff:
                        best_ratio_diff = ratio_diff
                        best_ratio = ratio
                    elif ratio_diff == best_ratio_diff:
                        if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                            best_ratio = ratio
                return best_ratio
            def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
                orig_width, orig_height = image.size
                aspect_ratio = orig_width / orig_height

                # calculate the existing image aspect ratio
                target_ratios = set(
                    (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                    i * j <= max_num and i * j >= min_num)
                target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

                # find the closest aspect ratio to the target
                target_aspect_ratio = find_closest_aspect_ratio(
                    aspect_ratio, target_ratios, orig_width, orig_height, image_size)

                # calculate the target width and height
                target_width = image_size * target_aspect_ratio[0]
                target_height = image_size * target_aspect_ratio[1]
                blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

                # resize the image
                resized_img = image.resize((target_width, target_height))
                processed_images = []
                for i in range(blocks):
                    box = (
                        (i % (target_width // image_size)) * image_size,
                        (i // (target_width // image_size)) * image_size,
                        ((i % (target_width // image_size)) + 1) * image_size,
                        ((i // (target_width // image_size)) + 1) * image_size
                    )
                    # split the image
                    split_img = resized_img.crop(box)
                    processed_images.append(split_img)
                assert len(processed_images) == blocks
                if use_thumbnail and len(processed_images) != 1:
                    thumbnail_img = image.resize((image_size, image_size))
                    processed_images.append(thumbnail_img)
                return processed_images
            def load_image(image_file, input_size=448, max_num=12):
                image = Image.open(image_file).convert('RGB')
                transform = build_transform(input_size=input_size)
                images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
                pixel_values = [transform(image) for image in images]
                pixel_values = torch.stack(pixel_values)
                return pixel_values
            patch_embedding = getattr(
                getattr(
                    getattr(self.generator_model, "vision_model", None),
                    "embeddings",
                    None,
                ),
                "patch_embedding",
                None,
            )
            patch_weight = getattr(patch_embedding, "weight", None)
            if patch_weight is not None:
                vision_dtype = patch_weight.dtype
                vision_device = patch_weight.device
            else:
                first_param = next(self.generator_model.parameters(), None)
                vision_dtype = getattr(first_param, "dtype", torch.bfloat16)
                vision_device = getattr(first_param, "device", self.device_map["generator"])
            pixel_values=[]
            for image_path in image_paths:
                image_tensor = load_image(image_path)
                image_tensor = image_tensor.to(device=vision_device, dtype=vision_dtype)
                pixel_values.append(image_tensor)
            pixel_value=torch.cat(pixel_values,dim=0)
            question = f'<image>\n{question}'
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            response, history = self.generator_model.chat(self.generator_processor, pixel_value, question, generation_config,
                                        history=None, return_history=True)
            return response
        elif self.args.generator_type == "TinyLLaVA-3.1B":
            try:
                from tinyllava.data.image_preprocess import ImagePreprocess
                from tinyllava.data.text_preprocess import TextPreprocess
                from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN
                from tinyllava.utils.eval_utils import KeywordsStoppingCriteria
                from tinyllava.utils.message import Message
            except ImportError as exc:
                raise ImportError(
                    "tinyllava is required for TinyLLaVA-3.1B. "
                    "Please install TinyLLaVABench (pip install -e .) before selecting this generator."
                ) from exc
            tokenizer = self.generator_processor["tokenizer"]
            base_image_processor = self.generator_processor["image_processor"]
            conv_mode = self.generator_processor.get("conv_mode", "phi")
            text_processor = TextPreprocess(tokenizer, conv_mode)
            image_processor = ImagePreprocess(base_image_processor, self.generator_model.config)
            question = question or ""
            if image_paths:
                prefix = "\n".join(DEFAULT_IMAGE_TOKEN for _ in image_paths)
                question = prefix + ("\n" if question else "") + question
            msg = Message()
            msg.add_message(question)
            processed_text = text_processor(msg.messages, mode="eval")
            input_ids = processed_text["input_ids"].unsqueeze(0).to(self.generator_model.device)
            separator = text_processor.template.separator.apply()
            if isinstance(separator, (tuple, list)):
                stop_str = separator[1]
            else:
                stop_str = separator
            stop_words = [stop_str] if stop_str else []
            stopping_criteria = (
                KeywordsStoppingCriteria(stop_words, tokenizer, input_ids) if stop_words else None
            )
            images_tensor = None
            if image_paths:
                processed_images = []
                for image_path in image_paths:
                    img = Image.open(image_path).convert("RGB")
                    processed_images.append(image_processor(img))
                if processed_images:
                    same_shape = all(
                        tensor.shape == processed_images[0].shape for tensor in processed_images
                    )
                    images_tensor = (
                        torch.stack(processed_images, dim=0) if same_shape else processed_images
                    )
            if isinstance(images_tensor, list):
                images_tensor = [
                    tensor.to(self.generator_model.device, dtype=torch.float16)
                    for tensor in images_tensor
                ]
            elif images_tensor is not None:
                images_tensor = images_tensor.to(self.generator_model.device, dtype=torch.float16)
            stopping_list = [stopping_criteria] if stopping_criteria is not None else None
            with torch.inference_mode():
                output_ids = self.generator_model.generate(
                    input_ids,
                    images=images_tensor,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0,
                    top_p=None,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_list,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if stop_str and outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)].strip()
            return outputs

        else:
            #todo raise
            return f"{self.args.generator_type} does not support."
    def cal_retriever_relevance(self, watermark_path, special_query):
        
        assert os.path.exists(watermark_path), f"Image path {watermark_path} does not exist."
        watermark = Image.open(watermark_path).convert("RGB")
        if self.args.retriever_type in {"clip", "clip_finetune"}:
            inputs = self.retriever_vision_processor(images=watermark, return_tensors="pt").to(self.device_map["retriever"])
            outputs = self.retriever_vision_model(**inputs)
            image_embeds = outputs.image_embeds
        elif self.args.retriever_type == "siglip-so400m-patch14-384":
            processor_inputs = self.retriever_vision_processor(images=watermark, return_tensors="pt")
            pixel_values = processor_inputs["pixel_values"].to(self.device_map["retriever"])
            with torch.no_grad():
                image_embeds = self.retriever_model.get_image_features(pixel_values=pixel_values)
        else:
            raise ValueError(f"Unsupported retriever type: {self.args.retriever_type}")
        normalized_embedding = image_embeds / image_embeds.norm(
                dim=-1, keepdim=True
        )
        normalized_watermark_embedding = normalized_embedding.cpu().detach().numpy().astype("float32")

        if self.args.retriever_type in {"clip", "clip_finetune"}:
            inputs = self.retriever_tokenizer([special_query], return_tensors="pt").to(self.device_map["retriever"])
            outputs = self.retriever_text_model(**inputs)
            text_embeds = outputs.text_embeds
        elif self.args.retriever_type == "siglip-so400m-patch14-384":
            inputs = self.retriever_tokenizer([special_query], return_tensors="pt", padding="max_length", truncation=True).to(self.device_map["retriever"])
            text_embeds = self.retriever_model.get_text_features(**inputs)
        else:
            raise ValueError(f"Unsupported retriever type: {self.args.retriever_type}")

        normalized_text_embeds = text_embeds/text_embeds.norm(dim=-1, keepdim=True)
        normalized_text_embeddings = normalized_text_embeds.cpu().detach().numpy().astype("float32")
        
        cosine_similarity = np.dot(normalized_watermark_embedding, normalized_text_embeddings.T)[0][0] 
        return cosine_similarity
         
    def cal_relevance_generator(self, image_path, query):
        if self.args.reranker_type == "LLaVA":
            image = Image.open(image_path)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query},
                    ],
                },
            ]

            prompt = self.reranker_processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.reranker_processor(image, prompt, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                generation_output = self.reranker_model.forward(
                    **inputs,
                )
                logits = generation_output['logits'][0, -1, :]

            yes_id = self.reranker_processor.tokenizer.encode("Yes")[-1]
            no_id = self.reranker_processor.tokenizer.encode("No")[-1]

            probs = (torch.nn.functional.softmax(torch.tensor([logits[yes_id], logits[no_id],]), dim=0,))
            probs = probs.float().cpu().detach().numpy()
            probs = probs[0]
        
        elif self.args.reranker_type == "qwen":
            from qwenvl.run_qwenvl import qwen_eval_relevance

            probs = qwen_eval_relevance(image_path, query, self.reranker_model, self.reranker_processor)
        return probs
    
    def run_mmqa(self, is_images=False, is_write_file=True, watermark_num=None):
        return self._run_queries("MMQA", is_images, is_write_file, watermark_num)

    def run_webqa(self, is_images=False, is_write_file=True, watermark_num=None):
        return self._run_queries("WebQA", is_images, is_write_file, watermark_num)

    def _run_queries(self, dataset, is_images, is_write_file, watermark_num=None):
        mode = watermark_num if watermark_num is not None else getattr(self.args, "watermark_num", "no")
        if mode not in {"no", "single", "all"}:
            raise ValueError(f"Unsupported watermark_num: {mode}")
        with open(self.args.special_queries_file_path, encoding="utf-8") as handle:
            queries = json.load(handle)
        if not isinstance(queries, list):
            raise ValueError("Probe-query data must be a list of records.")

        database = self.images_database
        if is_images and mode == "all":
            database = clone_image_database(self.images_database)
            paths = dict.fromkeys(str(Path(item["watermark_path"]).resolve()) for item in queries)
            for path in paths:
                self.add_watermark_to_image_database(database, path)

        save_dir = Path("results") / dataset / self.args.generator_type / timestamp_str
        if is_write_file:
            save_dir.mkdir(parents=True, exist_ok=True)
        response_keys = [
            "no_images_no_watermark_response",
            "yes_images_no_watermark_response",
            "yes_images_single_watermark_response",
            "yes_images_all_watermark_response",
        ]
        response_key = f"yes_images_{mode}_watermark_response" if is_images else response_keys[0]
        results = []
        for query_index, item in enumerate(tqdm(queries)):
            question = item.get("special_query", item.get("probe_query"))
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Query record {query_index} requires special_query or probe_query.")
            image_paths = None
            similarity_json = {}
            if is_images:
                if mode == "single":
                    database = clone_image_database(self.images_database)
                    self.add_watermark_to_image_database(database, item["watermark_path"])
                image_paths, similarity_json = self.retriever(database, question)
                if mode == "all":
                    similarity_json = dict(similarity_json)
                    similarity_json["watermark"] = float(self.cal_retriever_relevance(item["watermark_path"], question))
            with torch.no_grad():
                output = self.generator(image_paths=image_paths, question=question)
            result = dict.fromkeys(response_keys)
            result["special_query"] = question
            result[response_key] = output
            results.append(result)

            if is_write_file and is_images:
                query_name = question.replace("\n", " ").replace("/", "_")[:120]
                query_dir = save_dir / f"{query_index:04d}_{query_name}" / response_key.removesuffix("_response")
                images_dir = query_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                for image_index, image_path in enumerate(image_paths):
                    with Image.open(image_path) as img:
                        if img.mode == "RGBA":
                            img = img.convert("RGB")
                        img.save(images_dir / f"{image_index}_{Path(image_path).name}")
                with open(query_dir / "similarity_scores.json", "w", encoding="utf-8") as handle:
                    json.dump(similarity_json, handle, indent=4)
        if is_write_file:
            with open(save_dir / "results.json", "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=4, ensure_ascii=False)
        return results


def run_pipeline_logger(args):
    if args.dataset=='MMQA_sample':
        results=None
        #ok no_images_no_watermark_response
        texts_RAG=MultimodalRAG(args)
        no_images_no_watermark_response=texts_RAG.run_mmqa(is_images=False,watermark_num='no')
        results=no_images_no_watermark_response
        
        #ok yes_images_no_watermark_response
        images_RAG=texts_RAG
        yes_images_no_watermark_response=images_RAG.run_mmqa(is_images=True,watermark_num='no')
        
        tmp_lookup={item['special_query']:item for item in yes_images_no_watermark_response}
        for dict_item in results:
            tmp_dict=tmp_lookup.get(dict_item['special_query'])
            if tmp_dict:
                dict_item.update(yes_images_no_watermark_response=tmp_dict['yes_images_no_watermark_response'])
        
        #ok yes_images_single_watermark_response
        watermarked_RAG=images_RAG
        yes_images_single_watermark_response=watermarked_RAG.run_mmqa(is_images=True,watermark_num='single')
        
        tmp_lookup={item['special_query']:item for item in yes_images_single_watermark_response}
        for dict_item in results:
            tmp_dict=tmp_lookup.get(dict_item['special_query'])
            if tmp_dict:
                dict_item.update(yes_images_single_watermark_response=tmp_dict['yes_images_single_watermark_response'])
        
        #ok yes_images_all_watermark_response
        yes_images_all_watermark_response=watermarked_RAG.run_mmqa(is_images=True,watermark_num='all')
        
        tmp_lookup={item['special_query']:item for item in yes_images_all_watermark_response}
        for dict_item in results:
            tmp_dict=tmp_lookup.get(dict_item['special_query'])
            if tmp_dict:
                dict_item.update(yes_images_all_watermark_response=tmp_dict['yes_images_all_watermark_response'])
        
        fieldnames=results[0].keys()
        csv_filepath=f"results/MMQA/{timestamp_str}results_sample.csv"
        with open(csv_filepath,'w',newline='',encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()  
            writer.writerows(results)

    elif args.dataset=='MMQA_all':
        results=None
        #ok no_images_no_watermark_response
        texts_RAG=MultimodalRAG(args)
        no_images_no_watermark_response=texts_RAG.run_mmqa(is_images=False,watermark_num='no')
        results=no_images_no_watermark_response
        
        #ok yes_images_no_watermark_response
        images_RAG=texts_RAG
        yes_images_no_watermark_response=images_RAG.run_mmqa(is_images=True,watermark_num='no')
        
        tmp_lookup={item['special_query']:item for item in yes_images_no_watermark_response}
        for dict_item in results:
            tmp_dict=tmp_lookup.get(dict_item['special_query'])
            if tmp_dict:
                dict_item.update(yes_images_no_watermark_response=tmp_dict['yes_images_no_watermark_response'])
        
        #ok yes_images_single_watermark_response
        watermarked_RAG=images_RAG
        yes_images_single_watermark_response=watermarked_RAG.run_mmqa(is_images=True,watermark_num='single')
        
        tmp_lookup={item['special_query']:item for item in yes_images_single_watermark_response}
        for dict_item in results:
            tmp_dict=tmp_lookup.get(dict_item['special_query'])
            if tmp_dict:
                dict_item.update(yes_images_single_watermark_response=tmp_dict['yes_images_single_watermark_response'])
        
        #ok yes_images_all_watermark_response
        yes_images_all_watermark_response=watermarked_RAG.run_mmqa(is_images=True, watermark_num='all')
        
        tmp_lookup={item['special_query']:item for item in yes_images_all_watermark_response}
        for dict_item in results:
            tmp_dict=tmp_lookup.get(dict_item['special_query'])
            if tmp_dict:
                dict_item.update(yes_images_all_watermark_response=tmp_dict['yes_images_all_watermark_response'])
        
        #ok 
        fieldnames=results[0].keys()
        csv_filepath=f"results/MMQA/{args.generator_type}/{timestamp_str}/results_all.csv"
        with open(csv_filepath,'w',newline='',encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()  
            writer.writerows(results)
    

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMQA", choices=["MMQA","WebQA"])
    parser.add_argument("--retriever_type", type=str, default="clip", choices=["clip", "siglip-so400m-patch14-384", "openclip"])
    parser.add_argument("--generator_type", type=str, default="Qwen2.5-VL-7B-Instruct", choices=[
        "LLaVA",
        "TinyLLaVA-3.1B",
        "Qwen-VL-Chat",
        "Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-32B-Instruct",
        "qwen2.5-vl-finetune",
        "Qwen3-VL-32B-Instruct"
    ])
    parser.add_argument("--clip_topk", type=int, default=5)
    parser.add_argument("--retriever_device", type=str, default="cuda:0")
    parser.add_argument("--generator_device", type=str, default="cuda:0")
    parser.add_argument("--watermark_type", type=str, default="acronym")
    parser.add_argument("--watermark_num", type=str, default="no")
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--index_mapping_path", type=str, default=None)

    args = parser.parse_args()

    # Simple test
    print("Initializing MultimodalRAG...")
    rag = MultimodalRAG(args)

    test_query = "What is shown in the image?"
    print(f"\nTest query: {test_query}")

    print("Retrieving images...")
    image_paths, similarity_scores = rag.retriever(rag.images_database, test_query)
    print(f"Retrieved {len(image_paths)} images")

    print("Generating response...")
    response = rag.generator(image_paths, test_query)
    print(f"Response: {response}")
    print("\nTest completed successfully!")
