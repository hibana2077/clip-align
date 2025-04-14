# import torch
# import torch.nn as nn
# from torchvision import datasets
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from torch.utils.data import DataLoader, Dataset
# from transformers import CLIPProcessor, CLIPModel
# import timm
# import numpy as np
# from PIL import Image

# from datasets import load_dataset

# # # ds = load_dataset("pixparse/cc3m-wds")
# # ds = load_dataset("julianmoraes/doodles-captions-manual", split="train")
# ds = load_dataset("eltorio/ROCOv2-radiology", split="test") # train test val
# print(ds)
# print(ds.to_dict().keys())
# print(len(ds.to_dict()['image']))
# print(len(ds.to_dict()['caption']))
# print(type(ds.to_dict()['image'][0]))
# print(type(ds.to_dict()['image'][0]['path']))
# print(type(ds.to_dict()['image'][0]['bytes']))
# print(ds.to_dict()['image'][0].keys())
# print(ds.to_dict()['caption'][0])

# import torch
# from PIL import Image
# import open_clip

# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
# tokenizer = open_clip.get_tokenizer('ViT-B-32')

# image = preprocess(Image.open("../doc/test_clip.jpg")).unsqueeze(0)
# text = tokenizer(["a man", "a woman", "a cat"])

# with torch.no_grad(), torch.autocast("cuda"):
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 假設有5筆樣本，每筆特徵為4維
np.random.seed(0)
CVOUT = np.random.randn(5, 4)
CIB = np.random.randn(5, 4)

# 計算 cosine similarity 矩陣
def cosine_similarity_matrix(A, B):
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return np.dot(A_norm, B_norm.T)

sim_cvout = cosine_similarity_matrix(CVOUT, CIB)
sim_cib = cosine_similarity_matrix(CIB, CIB)

print(f"Shape of CVOUT: {CVOUT.shape}")
print(f"Shape of CIB: {CIB.shape}")
print(f"Shape of sim_cvout: {sim_cvout.shape}")
print(f"Shape of sim_cib: {sim_cib.shape}")

# 繪製 heatmap
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(sim_cvout, ax=axes[0], cmap="viridis", annot=True, cbar=False)
axes[0].set_title("CVOUT @ CIB.T")
axes[0].set_xlabel("CIB Index")
axes[0].set_ylabel("CVOUT Index")

sns.heatmap(sim_cib, ax=axes[1], cmap="viridis", annot=True, cbar=False)
axes[1].set_title("CIB @ CIB.T")
axes[1].set_xlabel("CIB Index")
axes[1].set_ylabel("CIB Index")

plt.tight_layout()
plt.show()