import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
import timm
import numpy as np
from PIL import Image
# from clip_align.flickr30k import FlickrDataset

# dataset = FlickrDataset()
# dataset.download_and_prepare()
# dataset = dataset.as_dataset()

# # print(len(dataset))
# print(dataset.keys())
# print(len(dataset['test']))
# print(dataset['test'][0])
# from clip_align.dataset import EmbeddingDataset
# dataset = EmbeddingDataset("flickr30k", "resnet50")
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# print(f"ResNet Model Embedding Size: {dataset.img_model_embedding_size}")
# print(f"CLIP Model Embedding Size: {dataset.clip_model_embedding_size}")
    
# for clip_embedding, resnet_embedding, label in dataloader:
#     print(f"CLIP Embedding Shape: {clip_embedding.shape}")
#     print(f"ResNet Embedding Shape: {resnet_embedding.shape}")
#     print(f"Label: {label.shape}")
#     break

caption = [ "Two young guys with shaggy hair look at their hands while hanging out in the yard.", "Two young, White males are outside near many bushes.", "Two men in green shirts are standing in a yard.", "A man in a blue shirt standing in a garden.", "Two friends enjoy time spent together." ]
minidx = caption.index(min(caption, key=len))
print(minidx)
print(caption[minidx])