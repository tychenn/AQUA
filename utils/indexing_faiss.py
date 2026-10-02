import faiss
import numpy as np
from PIL import Image,ImageFile
import os
import torch
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse
from transformers import CLIPProcessor, CLIPModel
import struct
import logging
ImageFile.MAX_DECOMPRESSED_DATA = 1024 * 1024 * 1024 # 1GB
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------- build_index -------------
def build_faiss_webqa(val_dataset, device, model, clip_type="clip", preprocess=None):
    embeddings = []
    index_to_image_id = {}
    count = 0
    for i in tqdm(val_dataset):
        datum = val_dataset[i]
        pos_imgs = datum["img_posFacts"]

        for j in range(len(pos_imgs)):
            image_id = pos_imgs[j]["image_id"]
            if image_id in index_to_image_id.values():
                continue
            # image_path = "../finetune/tasks/train_img/" + str(image_id) + ".png"
            image_path = "./finetune/tasks/WebQA_imgs/test/" + str(image_id) + ".png"
            if not os.path.exists(image_path):
                image_path = "./finetune/tasks/WebQA_imgs/train/" + str(image_id) + ".png"
                if not os.path.exists(image_path):
                    image_path =  "./finetune/tasks/WebQA_imgs/val/" + str(image_id) + ".png"
            assert os.path.exists(image_path)
                        
            with torch.no_grad():
                if clip_type == "clip":
                    image = preprocess(Image.open(image_path)).to(device)
                    image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
                elif clip_type == "openclip":
                    image = preprocess(Image.open(image_path).convert("RGB")).to(device)
                    image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
                elif "bge" in clip_type:
                    image_embeddings = model.encode(image=image_path)
                else:
                    pixel_values = preprocess(
                        images=Image.open(image_path).convert("RGB"),
                        return_tensors="pt",
                    ).pixel_values
                    pixel_values = pixel_values.to(torch.bfloat16).to(device)
                    image_embeddings = model.encode_image(
                        pixel_values, mode=clip_type
                    ).to(torch.float)

            combined_embedding = image_embeddings
            normalized_embedding = combined_embedding / combined_embedding.norm(
                dim=-1, keepdim=True
            )
            embeddings.append(normalized_embedding.cpu().numpy())

            index_to_image_id[count] = image_id
            count += 1

    embeddings = np.vstack(embeddings).astype("float32")

    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, index_to_image_id

def load_clip(args):
    clip_type = args.clip_type
    if clip_type == "clip":
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        tokenizer = clip.tokenize
    elif clip_type == "openclip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model.to(device)
    elif "bge" in clip_type:
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained(clip_type)
        model = AutoModel.from_pretrained(clip_type)
        model.to(device)
        preprocess = processor
        tokenizer = processor
    elif clip_type == "hf_clip":
        model = CLIPModel.from_pretrained("models/clip-vit-large-patch14-336")
        processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14-336")
        model.to(device)
        preprocess = processor
        tokenizer = processor
    else:
        raise ValueError("clip_type not supported")
    model.eval()
    return model, preprocess, tokenizer

def build_MMQA_embeddings(clip_type='hf_clip'):
    model, preprocess, tokenizer = load_clip(argparse.Namespace(clip_type=clip_type))
    images_dir="datasets/MMQA/images"
    embeddings = []
    index_to_image_id = {}
    count = 0
    for filename in tqdm(os.listdir(images_dir), desc="Processing images"):
        image_path=os.path.join(images_dir,filename)
        image_id = Path(filename).stem
        img = None 
        img = Image.open(image_path)

        with torch.no_grad():
            if clip_type == "clip":
                image = preprocess(img).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif clip_type == "openclip":
                image = preprocess(img.convert("RGB")).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif "bge" in clip_type:
                image_embeddings = model.encode(image=image_path)
            elif clip_type == "hf_clip":
                image = img.convert("RGB")
                if image.size[0]==1:
                    image = image.resize((10, image.size[1]))
                elif image.size[1]==1:
                    image = image.resize((image.size[0],10))
                inputs = preprocess(
                    images=image,
                    return_tensors="pt"
                ).to(device)
                image_embeddings = model.get_image_features(inputs.pixel_values)
            else:
                pixel_values = preprocess(
                    images=img.convert("RGB"),
                    return_tensors="pt",
                ).pixel_values
                pixel_values = pixel_values.to(torch.bfloat16).to(device)
                image_embeddings = model.encode_image(pixel_values, mode=clip_type).to(
                    torch.float
                )

        combined_embedding = image_embeddings
        normalized_embedding = combined_embedding / combined_embedding.norm(
            dim=-1, keepdim=True
        )
        embeddings.append(normalized_embedding.cpu().numpy())

        index_to_image_id[count] = image_id
        count += 1


    embeddings_path="datasets/MMQA/faiss_index/embeddings.pkl"
    Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    json_filepath = "datasets/MMQA/jsons/WatermarkMMRAG/MMQA_all_index_to_image_id.json"
    Path(json_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(json_filepath,"w") as f:
        json.dump(index_to_image_id,f,indent=4)
    return index_to_image_id

def build_WebQA_embeddings(clip_type='hf_clip'):
    
        
    model, preprocess, tokenizer = load_clip(argparse.Namespace(clip_type=clip_type))
    images_dir="datasets/WebQA/images"
    embeddings = []
    index_to_image_id = {}
    count = 0
    for filename in tqdm(os.listdir(images_dir), desc="Processing images"):
        image_path=os.path.join(images_dir,filename)
        image_id = Path(filename).stem
        img = None 
        img = Image.open(image_path)

        with torch.no_grad():
            if clip_type == "clip":
                image = preprocess(img).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif clip_type == "openclip":
                image = preprocess(img.convert("RGB")).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif "bge" in clip_type:
                image_embeddings = model.encode(image=image_path)
            elif clip_type == "hf_clip":
                image = img.convert("RGB")
                if image.size[0]==1:
                    image = image.resize((10, image.size[1]))
                elif image.size[1]==1:
                    image = image.resize((image.size[0],10))
                inputs = preprocess(
                    images=image,
                    return_tensors="pt"
                ).to(device)
                image_embeddings = model.get_image_features(inputs.pixel_values)
            else:
                pixel_values = preprocess(
                    images=img.convert("RGB"),
                    return_tensors="pt",
                ).pixel_values
                pixel_values = pixel_values.to(torch.bfloat16).to(device)
                image_embeddings = model.encode_image(pixel_values, mode=clip_type).to(
                    torch.float
                )

        combined_embedding = image_embeddings
        normalized_embedding = combined_embedding / combined_embedding.norm(
            dim=-1, keepdim=True
        )
        embeddings.append(normalized_embedding.cpu().numpy())

        index_to_image_id[count] = image_id
        count += 1

       
    embeddings_path="datasets/WebQA/faiss_index/embeddings.pkl"
    Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    json_filepath = "datasets/WebQA/jsons/WebQA_all_index_to_image_id.json"
    Path(json_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(json_filepath,"w") as f:
        json.dump(index_to_image_id,f,indent=4)
    return index_to_image_id

def ratio_embeddings_to_faiss(
    embeddings_path="datasets/MMQA/faiss_index/embeddings.pkl",
    ratio=None
):
    if ratio is None or not 0 < ratio <= 1:
        raise ValueError("ratio must be greater than 0 and at most 1")
    with open(embeddings_path, 'rb') as f:
        embeddings=pickle.load(f)
    total_images_num=len(embeddings)
    if total_images_num == 0:
        raise ValueError("Cannot build a FAISS index from an empty embedding cache")
    
    sample_count = max(1, int(ratio * total_images_num))
    embeddings = np.vstack(embeddings[:sample_count]).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index
    
   
    

def build_faiss_mmqa(
    val_dataset, metadata, device, model, clip_type="clip", preprocess=None
):

    embeddings = []
    index_to_image_id = {}
    count = 0
    for datum in tqdm(val_dataset):
        supporting_image_datastructure = datum["supporting_context"][0]
        image_id = supporting_image_datastructure["doc_id"]
        if image_id in index_to_image_id.values():
            continue
        image_path = "datasets/MMQA/images/" + metadata[image_id]["path"]
        with torch.no_grad():
            if clip_type == "clip":
                image = preprocess(Image.open(image_path)).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif clip_type == "openclip":
                image = preprocess(Image.open(image_path).convert("RGB")).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif "bge" in clip_type:
                image_embeddings = model.encode(image=image_path)
            elif clip_type == "hf_clip":
                image = Image.open(image_path).convert("RGB")
                inputs = preprocess(
                    images=image, 
                    return_tensors="pt"
                ).to(device)
                image_embeddings = model.get_image_features(inputs.pixel_values)
            else:
                pixel_values = preprocess(
                    images=Image.open(image_path).convert("RGB"),
                    return_tensors="pt",
                ).pixel_values
                pixel_values = pixel_values.to(torch.bfloat16).to(device)
                image_embeddings = model.encode_image(pixel_values, mode=clip_type).to(
                    torch.float
                )

        combined_embedding = image_embeddings
        normalized_embedding = combined_embedding / combined_embedding.norm(
            dim=-1, keepdim=True
        )
        embeddings.append(normalized_embedding.cpu().numpy())

        index_to_image_id[count] = image_id
        count += 1

    embeddings = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, index_to_image_id


def build_ratio_indices(dataset, clip_type="hf_clip"):
    root = Path("datasets") / dataset
    embeddings_path = root / "faiss_index" / "embeddings.pkl"
    mapping_path = root / "jsons" / f"{dataset}_all_index_to_image_id.json"
    if dataset == "MMQA":
        mapping_path = root / "jsons" / "WatermarkMMRAG" / mapping_path.name
        build_embeddings = build_MMQA_embeddings
        filename_prefix = "MMQA_ratio"
    elif dataset == "WebQA":
        build_embeddings = build_WebQA_embeddings
        filename_prefix = "WebQA"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if not (embeddings_path.exists() and mapping_path.exists()):
        build_embeddings(clip_type=clip_type)
    with mapping_path.open() as f:
        index_to_image_id = json.load(f)

    index_dir = root / "faiss_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    for ratio in (0.2, 0.4, 0.6, 0.8, 1.0):
        print(f"Start to save {ratio=}")
        index = ratio_embeddings_to_faiss(embeddings_path=embeddings_path, ratio=ratio)
        ratio_mapping = {str(i): index_to_image_id[str(i)] for i in range(index.ntotal)}
        faiss.write_index(index, str(index_dir / f"{filename_prefix}_{clip_type}_{ratio:.0%}.index"))
        ratio_mapping_path = root / "jsons" / f"{dataset}_all_index_to_image_id_{ratio:.0%}.json"
        ratio_mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with ratio_mapping_path.open("w") as f:
            json.dump(ratio_mapping, f, indent=4)
        if dataset == "MMQA" and ratio == 1.0:
            faiss.write_index(index, str(index_dir / f"MMQA_all_{clip_type}.index"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--datasets", type=str, default="WebQA", choices=["WebQA", "MMQA", "MMQA_ratio"])
    parser.add_argument("--clip_type", type=str, default="hf_clip")
    args = parser.parse_args()

    if args.datasets in ("WebQA", "MMQA_ratio"):
        dataset = "MMQA" if args.datasets == "MMQA_ratio" else args.datasets
        build_ratio_indices(dataset, clip_type=args.clip_type)
    #ok deprecated
    elif args.datasets == "MMQA":
        model, preprocess, tokenizer = load_clip(args)

        with open("datasets/MMQA/jsons/MM_PoisonRAG/MMQA_test_image.json", "r") as f:
            val_dataset = json.load(f)
        with open("datasets/MMQA/jsons/MM_PoisonRAG/MMQA_image_metadata.json", "r") as f:
            metadata = json.load(f)


        index, index_to_image_id = build_faiss_mmqa(
            val_dataset,
            metadata,
            device,
            model,
            clip_type=args.clip_type,
            preprocess=preprocess,
        )
