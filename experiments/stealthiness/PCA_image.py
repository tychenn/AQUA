import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import os
import random
import torch
from PIL import Image 
from transformers import AutoProcessor, CLIPVisionModelWithProjection 
plt.style.use('seaborn-v0_8-whitegrid') 
seed_value = 42 

random.seed(seed_value) 
np.random.seed(seed_value) 
torch.manual_seed(seed_value) 

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value) 
    torch.cuda.manual_seed_all(seed_value) 
def add_watermark_embeddings(watermark_images_dir):
    watermark_embeddings = [] 

    retriever_vision_processor = AutoProcessor.from_pretrained("models/clip-vit-large-patch14-336") 
    retriever_vision_model = CLIPVisionModelWithProjection.from_pretrained("models/clip-vit-large-patch14-336")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    retriever_vision_model.to(device) 
    retriever_vision_model.eval() 

    image_files = [f for f in os.listdir(watermark_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    


    for filename in image_files:
        image_path = os.path.join(watermark_images_dir, filename) 
        watermark = Image.open(image_path).convert("RGB") 
        inputs = retriever_vision_processor(images=watermark, return_tensors="pt").to(device) 
        with torch.no_grad(): 
            outputs = retriever_vision_model(**inputs)
            image_embeds = outputs.image_embeds 
            normalized_embedding = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            normalized_embedding = normalized_embedding.cpu().detach().numpy().astype("float32")
            watermark_embeddings.append(normalized_embedding) 
    return watermark_embeddings

colors = plt.cm.tab10.colors 
watermark_sources = [
    {'path': 'datasets/watermark_for_PCA/CTYs', 'label': 'AQUA_acronym', 'color': colors[1]},
    {'path': 'datasets/watermark_for_PCA/positions', 'label': 'AQUA_spatial', 'color': colors[3]},]

embeddings_path="/home/cty/WatermarkmmRAG/datasets/MMQA/faiss_index/embeddings.pkl"
with open(embeddings_path, 'rb') as f:
    embeddings=pickle.load(f)
embeddings=np.array(embeddings)
if embeddings.ndim == 3 and embeddings.shape[1] == 1:
    embeddings = np.squeeze(embeddings, axis=1)

n_samples_to_sample = 287 
if embeddings.shape[0] < n_samples_to_sample:
    sampled_embeddings = embeddings
else:
    indices = np.random.choice(embeddings.shape[0], size=n_samples_to_sample, replace=False)
    sampled_embeddings = embeddings[indices]

all_watermark_embeddings_raw = []
for source in watermark_sources:
    embeddings_list = add_watermark_embeddings(source['path'])
    all_watermark_embeddings_raw.append(embeddings_list)

all_watermark_embeddings_processed = []
valid_watermark_data = [] 

for i, embeddings_list in enumerate(all_watermark_embeddings_raw):
    source_info = watermark_sources[i]

    watermark_embeddings_np = np.array(embeddings_list)
    if watermark_embeddings_np.ndim == 3 and watermark_embeddings_np.shape[1] == 1:
        watermark_embeddings_np = np.squeeze(watermark_embeddings_np, axis=1)


    if watermark_embeddings_np.shape[1] != sampled_embeddings.shape[1]:
        continue

    all_watermark_embeddings_processed.append(watermark_embeddings_np)
    valid_watermark_data.append({
        'embeddings': watermark_embeddings_np,
        'label': source_info['label'],
        'color': source_info['color']
    })

n_components = 3 
pca = PCA(n_components=n_components)

pca.fit(sampled_embeddings)

reduced_embeddings = pca.transform(sampled_embeddings)

all_reduced_watermark_embeddings = []
for data in valid_watermark_data:
    reduced_wm_embeddings = pca.transform(data['embeddings'])
    all_reduced_watermark_embeddings.append(reduced_wm_embeddings)


target_elev = 20 
target_azim = 44  

fig = plt.figure(figsize=(8,8)) 
ax = fig.add_subplot(111, projection='3d') 
all_points_for_limits = []
if reduced_embeddings.shape[0] > 0:
    all_points_for_limits.append(reduced_embeddings)

for reduced_wm in all_reduced_watermark_embeddings:
    if reduced_wm.shape[0] > 0:
        all_points_for_limits.append(reduced_wm)
ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], label='MMQA Image', alpha=0.3, s=100) 

for i, reduced_wm in enumerate(all_reduced_watermark_embeddings):
    data_info = valid_watermark_data[i]
    ax.scatter(reduced_wm[:, 0], reduced_wm[:, 1], reduced_wm[:, 2],
               color=data_info['color'],
               s=470, 
               label=data_info['label'],
               marker='*',
               edgecolors='k',
            linewidths=0.5) 
all_points_np = np.concatenate(all_points_for_limits, axis=0)
x_min, x_max = np.min(all_points_np[:, 0]), np.max(all_points_np[:, 0])
y_min, y_max = np.min(all_points_np[:, 1]), np.max(all_points_np[:, 1])
z_min, z_max = np.min(all_points_np[:, 2]), np.max(all_points_np[:, 2])

x_padding = (x_max - x_min) * 0.05 
y_padding = (y_max - y_min) * 0.05
z_padding = (z_max - z_min) * 0.05

ax.set_xlim(x_min - x_padding, x_max + x_padding)
ax.set_ylim(y_min - y_padding, y_max + y_padding)
ax.set_zlim(z_min - z_padding, z_max + z_padding)
axis_label_pad=-10
ax.set_xlabel('Principal Component 1',fontsize=20,labelpad=axis_label_pad) 
ax.set_ylabel('Principal Component 2',fontsize=20,labelpad=-12) 
ax.set_zlabel('Principal Component 3',fontsize=20,labelpad=-6) 


ax.view_init(elev=target_elev, azim=target_azim)

ax.legend(loc=(0.555,0.62),fontsize=23) 

plt.grid(True)

output_dir = '/home/cty/WatermarkmmRAG/experiments/stealthiness'
os.makedirs(output_dir, exist_ok=True)

save_filename = f'PCA_MMQA_image.png'
save_path = os.path.join(output_dir, save_filename)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.savefig(save_path) 


