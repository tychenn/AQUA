import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import random
import os 
plt.style.use('seaborn-v0_8-whitegrid')

seed_value = 42 

random.seed(seed_value) 
np.random.seed(seed_value) 
torch.manual_seed(seed_value) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value) 
    torch.cuda.manual_seed_all(seed_value) 
   

colors = plt.cm.tab10.colors
query_sources = [
    {'type': 'normal', 'label': 'Normal Query', 'color': colors[0],
     'json_path': 'datasets/MMQA/jsons/MMQA_all_image.json',
     'cache_path': 'datasets/MMQA/embeddings_cache/normal_query_embeddings.pkl',
     'query_field': 'question'}, 
    {'type': 'special_1', 'label': 'AQUA_acronym', 'color': colors[1],
     'json_path': 'datasets/special_query/CTYs/special_queries_CTY.json', 
     'cache_path': 'datasets/MMQA/embeddings_cache/special_query_embeddings_type1.pkl',
     'query_field': 'special_query'}, 
    {'type': 'special_2', 'label': 'AQUA_spatial', 'color': colors[3],
     'json_path': 'datasets/special_query/diffusion/banana.json', 
     'cache_path': 'datasets/MMQA/embeddings_cache/special_query_embeddings_type2.pkl',
     'query_field': 'special_query'}, 
    
]

cache_dir = "datasets/MMQA/embeddings_cache"
os.makedirs(cache_dir, exist_ok=True) 
all_embeddings_lists = {} 
all_cache_paths = [source['cache_path'] for source in query_sources]
all_caches_exist = all(os.path.exists(p) for p in all_cache_paths)

if all_caches_exist:
    for source in query_sources:
        cache_path = source['cache_path']
        with open(cache_path, 'rb') as f:
            all_embeddings_lists[source['type']] = pickle.load(f)

else: 
   
    tokenizer = AutoTokenizer.from_pretrained("models/clip-vit-large-patch14-336")
    text_model = CLIPTextModelWithProjection.from_pretrained("models/clip-vit-large-patch14-336").to(device)
    text_model.eval() 

    for source in query_sources:
        source_type = source['type']
        json_path = source['json_path']
        cache_path = source['cache_path']
        query_field = source['query_field']
        label = source['label']

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                query_jsons = json.load(f)
        except FileNotFoundError:
            all_embeddings_lists[source_type] = [] 
            continue
        except json.JSONDecodeError:
            
            all_embeddings_lists[source_type] = []
            continue

        current_embeddings_list=[]
        for item in tqdm(query_jsons):
            if query_field not in item:
                
                continue
            query_text = item[query_field]
            if not query_text or not isinstance(query_text, str): 
                 
                 continue

            inputs = tokenizer([query_text], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device) 
            with torch.no_grad():
                outputs = text_model(**inputs)
                
                text_embeds = outputs.text_embeds
                normalized_embedding = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                normalized_embedding = normalized_embedding.cpu().detach().numpy().astype("float32")
                current_embeddings_list.append(normalized_embedding) 

        all_embeddings_lists[source_type] = current_embeddings_list

        with open(cache_path, 'wb') as f:
            pickle.dump(current_embeddings_list, f)

    
all_embeddings_np = {} 
valid_sources_for_pca = [] 
for source in query_sources:
    source_type = source['type']
    label = source['label']
    embeddings_list = all_embeddings_lists.get(source_type) 

   
    embeddings_np = np.concatenate(embeddings_list, axis=0)

    all_embeddings_np[source_type] = embeddings_np
    valid_sources_for_pca.append(source) 


normal_query_type = query_sources[0]['type']
if normal_query_type not in all_embeddings_np:
    exit()

normal_embeddings_np = all_embeddings_np[normal_query_type]

n_samples_to_sample = 240 
if normal_embeddings_np.shape[0] < n_samples_to_sample:
    
    sampled_normal_embeddings = normal_embeddings_np
else:
    indices = np.random.choice(normal_embeddings_np.shape[0], size=n_samples_to_sample, replace=False) 
    sampled_normal_embeddings = normal_embeddings_np[indices]


n_components = 3 
pca = PCA(n_components=n_components)

pca.fit(sampled_normal_embeddings)

reduced_embeddings_dict = {} 
for source in valid_sources_for_pca: 
    source_type = source['type']
    label = source['label']
    embeddings_to_transform = all_embeddings_np[source_type]

    if source_type == normal_query_type:
        embeddings_to_transform = sampled_normal_embeddings 

    reduced_embeddings = pca.transform(embeddings_to_transform)
    reduced_embeddings_dict[source_type] = reduced_embeddings
target_elev = 20  
target_azim = 49  

fig = plt.figure(figsize=(8, 8)) 
ax = fig.add_subplot(111, projection='3d') 

all_x_coords = []
all_y_coords = []
all_z_coords = []

for source_type_key in reduced_embeddings_dict:
    data = reduced_embeddings_dict[source_type_key]
    if data.shape[0] > 0: 
        all_x_coords.extend(data[:, 0])
        all_y_coords.extend(data[:, 1])
        all_z_coords.extend(data[:, 2])
for source in valid_sources_for_pca: 
    source_type = source['type']
    label = source['label']
    color = source['color']
    reduced_data = reduced_embeddings_dict[source_type]

    if source_type == normal_query_type:
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                   color=color,
                   label=label,
                   alpha=0.3, 
                   s=100)      
    else:
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                   color=color,
                   s=450, 
                   label=label,
                   marker='*', 
                   edgecolors='k', 
                   linewidths=0.5) 

if all_x_coords and all_y_coords and all_z_coords: 
    x_min, x_max = np.min(all_x_coords), np.max(all_x_coords)
    y_min, y_max = np.min(all_y_coords), np.max(all_y_coords)
    z_min, z_max = np.min(all_z_coords), np.max(all_z_coords)

    x_padding = (x_max - x_min) * 0.05 if (x_max - x_min) > 0 else 0.1 
    y_padding = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.1
    z_padding = (z_max - z_min) * 0.05 if (z_max - z_min) > 0 else 0.1
    
    if x_padding == 0: x_padding = np.abs(x_min * 0.1) if x_min != 0 else 0.1
    if y_padding == 0: y_padding = np.abs(y_min * 0.1) if y_min != 0 else 0.1
    if z_padding == 0: z_padding = np.abs(z_min * 0.1) if z_min != 0 else 0.1


    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_zlim(z_min - z_padding, z_max + z_padding)
# 
ax.set_xlabel('Principal Component 1',fontsize=20,labelpad=-10) 
ax.set_ylabel('Principal Component 2',fontsize=20,labelpad=-10) 
ax.set_zlabel('Principal Component 3',fontsize=20,labelpad=-8) 

ax.view_init(elev=target_elev, azim=target_azim)

ax.legend(loc=(0.53,0.65),fontsize=23) 


plt.grid(True)

output_dir = '/home/cty/WatermarkmmRAG/experiments/stealthiness'
os.makedirs(output_dir, exist_ok=True) 

save_filename = f'PCA_MMQA_text.png' 
save_path = os.path.join(output_dir, save_filename)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.savefig(save_path, bbox_inches='tight') 