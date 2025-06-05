import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import open_clip
import json
import faiss
import gc
import torch
import copy
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
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.run_qwenvl import qwen_chat, qwen_eval_relevance
from qwen_vl_utils import process_vision_info
import datetime
import sys
sys.path.insert(0, os.path.abspath("./Qwen-VL-Chat"))
print(sys.path) 
from Qwen_VL_Chat.modeling_qwen import QWenLMHeadModel



now = datetime.datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H:%M:%S")

class MultimodalRAGArguments:
    retriever_device: str
    generator_device: str
    retriever_type:str
    generator_type:str
    dataset:str
    clip_topk:int
    watermark_type:str
    
class MultimodalRAG:
    args: MultimodalRAGArguments|Namespace
    device_map: dict[str,str]
    
    def __init__(self, args:MultimodalRAGArguments|Namespace):
        print('\n\n\n')
        self.args = args
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
        
    def load_retriever(self)->tuple[
        AutoModelForZeroShotImageClassification,
        CLIPTextModelWithProjection,
        AutoTokenizer,
        CLIPVisionModelWithProjection,
        AutoProcessor
    ]:
        if self.args.retriever_type == "clip":
            model = AutoModelForZeroShotImageClassification.from_pretrained( # type: ignore
                "models/clip-vit-large-patch14-336"
            ).to(self.device_map["retriever"])
            text_model = CLIPTextModelWithProjection.from_pretrained("models/clip-vit-large-patch14-336").to(self.device_map["retriever"]) # type: ignore
            tokenizer = AutoTokenizer.from_pretrained("models/clip-vit-large-patch14-336") #type: ignore
            vision_model = CLIPVisionModelWithProjection.from_pretrained("models/clip-vit-large-patch14-336").to(self.device_map["retriever"]) #type:ignore
            vision_processor = AutoProcessor.from_pretrained("models/clip-vit-large-patch14-336",use_fast=True) #type:ignore
        else:
            pass#todo
        model.eval() #type:ignore
        text_model.eval()
        vision_model.eval()
        return model, text_model, tokenizer, vision_model, vision_processor
    
    def load_generator(self, mllm_type):
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
        
        elif mllm_type == "Qwen-VL-Chat":
            model_name = "models/Qwen-VL-Chat"   # Qwen-VL-Chat model
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)  # Loads both vision and text processor
            mllm = QWenLMHeadModel.from_pretrained(
                model_name, 
                device_map=self.device_map["generator"],
                trust_remote_code=False, 
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
            mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, 
                device_map="auto",
                torch_dtype=torch.bfloat16,
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
        
        return mllm, processor
    
    def load_index(self,index_file_path="datasets/MMQA/faiss_index/MMQA_all_hf_clip.index")->tuple[faiss.Index,dict[str,str]]:
        if self.args.dataset=="MMQA":
            index=faiss.read_index(index_file_path)
            with open(f"./datasets/MMQA/jsons/WatermarkMMRAG/MMQA_all_index_to_image_id.json", "r") as f:
                index_to_image_id = json.load(f)
        elif self.args.dataset=="WebQA":
            index_file_path="datasets/WebQA/faiss_index/WebQA_hf_clip_100%.index"
            index=faiss.read_index(index_file_path)
            with open(f"./datasets/WebQA/jsons/WebQA_all_index_to_image_id.json", "r") as f:
                index_to_image_id = json.load(f)
            print("Successful load WebQA as database.")
        return index,index_to_image_id

    def add_watermark_to_image_database(self, images_database,watermark_path):
        
        assert os.path.exists(watermark_path), f"Image path {watermark_path} does not exist."

        watermark = Image.open(watermark_path)
        inputs = self.retriever_vision_processor(images=watermark, return_tensors="pt").to(self.device_map["retriever"])
        outputs = self.retriever_vision_model(**inputs)
        image_embeds = outputs.image_embeds
        normalized_embedding = image_embeds / image_embeds.norm(
                dim=-1, keepdim=True
        )
        normalized_embedding = normalized_embedding.cpu().detach().numpy().astype("float32")

        images_database.add(normalized_embedding)
        
        watermark_index=str(images_database.ntotal-1)
        watermark_filename=os.path.basename(watermark_path)
        watermark_filename_without_ext,watermark_filename_ext=os.path.splitext(watermark_filename)
        self.images_database_index_to_image_id[watermark_index]=watermark_filename_without_ext
        #print("added watermark!")
        
    def retriever(self, images_database,question):
        
        # text->embedding
        with torch.no_grad():
            if self.args.retriever_type == "clip":
                inputs = self.retriever_tokenizer([question], return_tensors="pt").to(self.device_map["retriever"]) #type:ignore
                outputs = self.retriever_text_model(**inputs)
                text_embeds = outputs.text_embeds
            else:
                pass
        # normalization
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)#type:ignore
        text_embeddings = text_embeds.cpu().detach().numpy().astype("float32")
        # search
        similarity_scores_list, indices_list = images_database.search(text_embeddings, self.args.clip_topk)
        # store image names
        retrieved_image_names = []
        for d, j in zip(similarity_scores_list[0], indices_list[0]):
            image_id = self.images_database_index_to_image_id[str(j)]
            retrieved_image_names.append(image_id)
        
        similarity_json = {}
        image_paths = []
        for i, image_id in enumerate(retrieved_image_names):
            image_path = None
            base_paths=[]
            #original image base path
            if self.args.dataset=='MMQA':
                base_paths.append(Path("datasets/MMQA/images"))
            elif self.args.dataset=='WebQA':
                base_paths.append(Path("datasets/WebQA/images"))
            #watermark image base path
            if self.args.watermark_type=='acronym':
                base_paths.append(Path("datasets/watermark_images/acronym"))
            elif self.args.watermark_type=='spatial':
                base_paths.append(Path("datasets/watermark_images/spatial"))
            elif self.args.watermark_type=='opt':
                base_paths.append(Path("datasets/watermark_images/opt"))
            elif self.args.watermark_type=='naive':
                base_paths.append(Path("datasets/watermark_images/naive"))
            # all possible ext
            for base_path in base_paths:
                for ext in ['.jpg','.JPG','.Jpg','.jpeg','.JPEG', '.png', '.PNG','.gif','.tif','.tiff']:
                    temp_path = base_path/f"{image_id}{ext}"
                    if temp_path.exists():
                        image_path = temp_path
                        break  
            if image_path: 
                image_paths.append(image_path)
            else:
                raise FileNotFoundError(f"Image file not found for ID: {image_id}") 
        
            similarity_json[image_id] = float(similarity_scores_list[0][i]) 
        
        
        del inputs
        del outputs
        del text_embeds 
        del text_embeddings

        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        return image_paths,similarity_json

    def generator(self, image_paths=None, question=None)->str:  
        
        if self.args.generator_type == "LLaVA":
            
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
                inputs = self.generator_processor(text=prompt, return_tensors="pt").to(self.main_device)
            
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
                inputs = self.generator_processor(images=images, text=prompt, return_tensors="pt").to(self.device_map["generator"])
            
            
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
            if image_paths is None:
                question = ( 
                    f"{question}\n"
                )
                mllm_tokenizer = AutoTokenizer.from_pretrained("/home/cty/WatermarkmmRAG/models/Qwen-VL-Chat", trust_remote_code=True)
                output = qwen_chat(image_paths, question, self.generator_model, mllm_tokenizer)
            else:
                question = ( 
                    f"{question}"
                )
                mllm_tokenizer = AutoTokenizer.from_pretrained("/home/cty/WatermarkmmRAG/models/Qwen-VL-Chat", trust_remote_code=True)
                tmp_list=[]
                for item in image_paths:
                    tmp_list.append(str(item))
                image_paths=tmp_list
                output = qwen_chat(image_paths, question, self.generator_model, mllm_tokenizer)
            return output
        
        elif self.args.generator_type in["Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct(8bit)","Qwen2.5-VL-32B-Instruct"]:
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
    
        elif self.args.generator_type in ["InternVL3-2B","InternVL3-8B"]:
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
            pixel_values=[]
            for image_path in image_paths:
                pixel_values.append(load_image(image_path).to(torch.bfloat16).to(self.device_map["generator"]))
            pixel_value=torch.cat(pixel_values,dim=0)
            question = f'<image>\n{question}'
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            response, history = self.generator_model.chat(self.generator_processor, pixel_value, question, generation_config,
                                        history=None, return_history=True)
            return response

        else:
            #todo raise
            return f"{self.args.generator_type} does not support."
    def cal_retriever_relevance(self, watermark_path, special_query):
        
        assert os.path.exists(watermark_path), f"Image path {watermark_path} does not exist."
        watermark = Image.open(watermark_path)
        inputs = self.retriever_vision_processor(images=watermark, return_tensors="pt").to(self.device_map["retriever"])
        outputs = self.retriever_vision_model(**inputs)
        image_embeds = outputs.image_embeds
        normalized_embedding = image_embeds / image_embeds.norm(
                dim=-1, keepdim=True
        )
        normalized_watermark_embedding = normalized_embedding.cpu().detach().numpy().astype("float32")

        if self.args.retriever_type == "clip":
            inputs = self.retriever_tokenizer([special_query], return_tensors="pt").to(self.device_map["retriever"])
            outputs = self.retriever_text_model(**inputs)
            text_embeds = outputs.text_embeds
        else:
            inputs = self.retriever_tokenizer([special_query]).to(self.device)
            text_embeds = self.retriever_text_model.encode_text(inputs)

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
            probs = qwen_eval_relevance(image_path, query, self.reranker_model, self.reranker_processor)
        return probs
    
    def run_mmqa(self, is_images=False,is_write_file=True):
        data_key=[
            "special_query", 
            "no_images_no_watermark_response",
            "yes_images_no_watermark_response",
            "yes_images_single_watermark_response",
            "yes_images_all_watermark_response",
        ]
        results=[]
        
        with open(self.args.special_queries_file_path, 'r', encoding='utf-8') as f:
            special_query_watermark_path_s = json.load(f)
            
        watermarked_images_database=copy.deepcopy(self.images_database)
        for item in special_query_watermark_path_s:
            self.add_watermark_to_image_database(
                images_database=watermarked_images_database,
                watermark_path=item["watermark_path"]
            )
        single_watermark_images_database=copy.deepcopy(self.images_database)  
        item=special_query_watermark_path_s[0]
        self.add_watermark_to_image_database(
            images_database=single_watermark_images_database,
            watermark_path=item["watermark_path"]
        )
        
        save_dir=Path("/home/cty/WatermarkmmRAG/results/MMQA")/self.args.generator_type/timestamp_str
        os.makedirs(save_dir,exist_ok=True) 
        
        with open(self.args.special_queries_file_path, "r", encoding="utf-8") as f:
            special_queries_watermarks = json.load(f)
        for index, item in enumerate(tqdm(special_queries_watermarks)):
            special_query=item["special_query"]
            special_query_no_newline=special_query.replace('\n',' ')
            watermark_path=item["watermark_path"]
            
            special_query_save_dir=Path("/home/cty/WatermarkmmRAG/results/MMQA")/self.args.generator_type/timestamp_str/special_query_no_newline
            data={key:None for key in data_key}
            
            #ok no_images_no_watermark_response
            if is_images==False:
                data['special_query']=special_query
                output = self.generator(image_paths=None, question=special_query)
                data['no_images_no_watermark_response']=output
                results.append(data)
            
            #ok yes_images_no_watermark_response
            elif is_images==True and self.args.watermark_num=="no":
                
                data['special_query']=special_query
                
                
                image_paths,similarity_json= self.retriever(self.images_database,special_query)
            
                
                if is_write_file:
                    special_query_save_dir=special_query_save_dir/"yes_images_no_watermark"
                    os.makedirs(special_query_save_dir,exist_ok=True) 
                    images_save_dir=special_query_save_dir/"images"
                    os.makedirs(images_save_dir, exist_ok=True)
                    similarity_json_save_dir=special_query_save_dir
                    
                    for image_path in image_paths:
                        img = Image.open(image_path)

                        if img.mode=='RGBA':
                            img=img.convert('RGB')
                            
                        img.save(images_save_dir/f"{image_id}{image_path.suffix}") 
                    
                    similarity_score_file = similarity_json_save_dir/"similarity_scores.json"
                    with open(similarity_score_file, 'w') as f:
                        json.dump(similarity_json, f, indent=4)  
                
                with torch.no_grad():
                    output = self.generator(image_paths=image_paths, question=special_query)
                data['yes_images_no_watermark_response']=output
                results.append(data)
                
            #ok yes_images_single_watermark_response
            elif is_images==True and self.args.watermark_num=="single":
                 
                data['special_query']=special_query
                
                image_paths, similarity_json = self.retriever(single_watermark_images_database,special_query)                
                output = self.generator(image_paths=image_paths, question=special_query)
                
                data['yes_images_single_watermark_response']=output
                results.append(data)
                
                if is_write_file:
                    special_query_save_dir=special_query_save_dir/"yes_images_single_watermark"
                    os.makedirs(special_query_save_dir,exist_ok=True) 
                    images_save_dir=special_query_save_dir/"images"
                    os.makedirs(images_save_dir, exist_ok=True)
                    similarity_json_save_dir=special_query_save_dir
                    similarity_score_file = similarity_json_save_dir/ "similarity_scores.json"
                    with open(similarity_score_file, 'w') as f:
                        json.dump(similarity_json, f, indent=4)  
                    
                del output, image_paths,similarity_json
            
            #ok yes_images_all_watermark_response
            elif is_images==True and self.args.watermark_num=="all":
                
                data['special_query']=special_query
                
                similarity_scores, indices_list = self.retriever(watermarked_images_database,special_query)
                
                retrieved_image_names = []
                for d, j in zip(similarity_scores[0], indices_list[0]):
                    image_id = self.images_database_index_to_image_id[str(j)]
                    retrieved_image_names.append(image_id)
                
                if is_write_file:
                    special_query_save_dir=special_query_save_dir/"yes_images_all_watermark"
                    os.makedirs(special_query_save_dir,exist_ok=True) 
                    images_save_dir=special_query_save_dir/"images"
                    os.makedirs(images_save_dir, exist_ok=True)
                    similarity_json_save_dir=special_query_save_dir
                    similarity_score_file = similarity_json_save_dir/ "similarity_scores.json"
                    
                similarity_data = {}
                image_paths = []
                for i, image_id in enumerate(retrieved_image_names):
                    image_path = None
                    base_paths=[
                        "datasets/MMQA/images",
                        "datasets/watermark_images"
                    ]
                    for base_path in base_paths:
                        for ext in ['.jpg','.jpeg','.JPG', '.png', '.PNG']:
                            temp_path = base_path/f"{image_id}{ext}"
                            if temp_path.exists():
                                image_path = temp_path
                                break  
                    if image_path: 
                        image_paths.append(image_path)
                        
                        
                        if is_write_file:
                            img = Image.open(image_path)
                            
                            if img.mode=='RGBA':
                                img=img.convert('RGB')
                                
                            img.save(images_save_dir/f"{image_id}{image_path.suffix}") 
                        
                    else:
                        raise FileNotFoundError(f"Image file not found for ID: {image_id}")
                    similarity_data[image_id] = float(similarity_scores[0][i]) 
                tmp_score=self.cal_retriever_relevance(watermark_path,special_query)
                similarity_data["watermark"]=float(tmp_score)
                
                
                if is_write_file:
                    with open(similarity_score_file, 'w') as f:
                        json.dump(similarity_data, f, indent=4)  
                
                output = self.generator(image_paths=image_paths, question=special_query)
                data['yes_images_all_watermark_response']=output
                results.append(data)   
        if is_write_file:
            with open(save_dir/"results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        return results
    
    def run_webqa(self, is_images=False,is_write_file=True):
        data_key=[
            "special_query", 
            "no_images_no_watermark_response",
            "yes_images_no_watermark_response",
            "yes_images_single_watermark_response",
            "yes_images_all_watermark_response",
        ]
        results=[]
        
        with open(self.args.special_queries_file_path, 'r', encoding='utf-8') as f:
            special_query_watermark_path_s = json.load(f)
            
        watermarked_images_database=copy.deepcopy(self.images_database)
        for item in special_query_watermark_path_s:
            self.add_watermark_to_image_database(
                images_database=watermarked_images_database,
                watermark_path=item["watermark_path"]
            )
        single_watermark_images_database=copy.deepcopy(self.images_database)  
        item=special_query_watermark_path_s[0]
        self.add_watermark_to_image_database(
            images_database=single_watermark_images_database,
            watermark_path=item["watermark_path"]
        )
        if is_write_file:
            save_dir=Path("/home/cty/WatermarkmmRAG/results/WebQA")/self.args.generator_type/timestamp_str
            os.makedirs(save_dir,exist_ok=True) 
        
        with open(self.args.special_queries_file_path, "r", encoding="utf-8") as f:
            special_queries_watermarks = json.load(f)
        for index, item in enumerate(tqdm(special_queries_watermarks)):
            special_query=item["special_query"]
            special_query_no_newline=special_query.replace('\n',' ')
            watermark_path=item["watermark_path"]
            
            if is_write_file:
                special_query_save_dir=Path("/home/cty/WatermarkmmRAG/results/WebQA")/self.args.generator_type/timestamp_str/special_query_no_newline
            data={key:None for key in data_key}
            
            
            
            #ok no_images_no_watermark_response
            if is_images==False:
                data['special_query']=special_query
                output = self.generator(image_paths=None, question=special_query)
                data['no_images_no_watermark_response']=output
                results.append(data)
            
            #ok yes_images_no_watermark_response
            elif is_images==True and self.args.watermark_num=="no":
                
                data['special_query']=special_query
                
                #ok检索
                image_paths,similarity_json= self.retriever(self.images_database,special_query)
            
                
                if is_write_file:
                    special_query_save_dir=special_query_save_dir/"yes_images_no_watermark"
                    os.makedirs(special_query_save_dir,exist_ok=True) 
                    images_save_dir=special_query_save_dir/"images"
                    os.makedirs(images_save_dir, exist_ok=True)
                    similarity_json_save_dir=special_query_save_dir
                    
                    for image_path in image_paths:
                        img = Image.open(image_path)

                        if img.mode=='RGBA':
                            img=img.convert('RGB')
                        img.save(images_save_dir/f"{image_id}{image_path.suffix}") 
                    
                    similarity_score_file = similarity_json_save_dir/"similarity_scores.json"
                    with open(similarity_score_file, 'w') as f:
                        json.dump(similarity_json, f, indent=4)  
                
                with torch.no_grad():
                    output = self.generator(image_paths=image_paths, question=special_query)
                data['yes_images_no_watermark_response']=output
                results.append(data)
                
            #ok yes_images_single_watermark_response
            elif is_images==True and self.args.watermark_num=="single":
                 
                data['special_query']=special_query
                
                image_paths, similarity_json = self.retriever(single_watermark_images_database,special_query)
                output = self.generator(image_paths=image_paths, question=special_query)
                
                data['yes_images_single_watermark_response']=output
                results.append(data)
                
                if is_write_file:
                    special_query_save_dir=special_query_save_dir/"yes_images_single_watermark"
                    os.makedirs(special_query_save_dir,exist_ok=True) 
                    images_save_dir=special_query_save_dir/"images"
                    os.makedirs(images_save_dir, exist_ok=True)
                    similarity_json_save_dir=special_query_save_dir
                    similarity_score_file = similarity_json_save_dir/ "similarity_scores.json"
                    with open(similarity_score_file, 'w') as f:
                        json.dump(similarity_json, f, indent=4)  
                    
                del output, image_paths,similarity_json
            #ok yes_images_all_watermark_response
            elif is_images==True and self.args.watermark_num=="all":
                
                data['special_query']=special_query
                
                similarity_scores, indices_list = self.retriever(watermarked_images_database,special_query)
                
                retrieved_image_names = []
                for d, j in zip(similarity_scores[0], indices_list[0]):
                    image_id = self.images_database_index_to_image_id[str(j)]
                    retrieved_image_names.append(image_id)
                
                if is_write_file:
                    special_query_save_dir=special_query_save_dir/"yes_images_all_watermark"
                    os.makedirs(special_query_save_dir,exist_ok=True) 
                    images_save_dir=special_query_save_dir/"images"
                    os.makedirs(images_save_dir, exist_ok=True)
                    similarity_json_save_dir=special_query_save_dir
                    similarity_score_file = similarity_json_save_dir/ "similarity_scores.json"
                    
                similarity_data = {}
                image_paths = []
                for i, image_id in enumerate(retrieved_image_names):
                    image_path = None
                    base_paths=[
                        "datasets/MMQA/images",
                        "datasets/watermark_images"
                    ]
                    for base_path in base_paths:
                        for ext in ['.jpg','.jpeg','.JPG', '.png', '.PNG']:
                            temp_path = base_path/f"{image_id}{ext}"
                            if temp_path.exists():
                                image_path = temp_path
                                break  
                    if image_path: 
                        image_paths.append(image_path)
                        
                        if is_write_file:
                            img = Image.open(image_path)
                            
                            if img.mode=='RGBA':
                                img=img.convert('RGB')
                                
                            img.save(images_save_dir/f"{image_id}{image_path.suffix}") 
                        
                    else:
                        raise FileNotFoundError(f"Image file not found for ID: {image_id}") 
                    similarity_data[image_id] = float(similarity_scores[0][i]) 
                tmp_score=self.cal_retriever_relevance(watermark_path,special_query)
                similarity_data["watermark"]=float(tmp_score)
                
                
                if is_write_file:
                    with open(similarity_score_file, 'w') as f:
                        json.dump(similarity_data, f, indent=4)  
                
                output = self.generator(image_paths=image_paths, question=special_query)
                data['yes_images_all_watermark_response']=output
                results.append(data)   
        if is_write_file:
            with open(save_dir/"results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
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
        csv_filepath=f"/home/cty/WatermarkmmRAG/results/MMQA/{timestamp_str}results_sample.csv"
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
        csv_filepath=f"/home/cty/WatermarkmmRAG/results/MMQA/{args.generator_type}/{timestamp_str}/results_all.csv"
        with open(csv_filepath,'w',newline='',encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()  
            writer.writerows(results)
    

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMQA_ratio", choices=["MMQA","WEBQA"])
    parser.add_argument("--retriever_type", type=str, default="clip", choices=["clip", "openclip"])
    parser.add_argument("--generator_type", type=str, default="LLaVA", choices=["LLaVA", "Qwen-VL-Chat","Qwen2.5-VL-7B-Instruct","Qwen2.5-VL-32B-Instruct"])
    parser.add_argument("--clip_topk", type=int, default=5)
    parser.add_argument("--special_queries_file_path", type=str, default="/home/cty/WatermarkmmRAG/datasets/special_queries_instruction.json")
    parser.add_argument("--save_dir", type=str, default="results")
    
    args = parser.parse_args()
