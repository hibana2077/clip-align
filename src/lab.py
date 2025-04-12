# import torch
# import torch.nn as nn
# from torchvision import datasets
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from torch.utils.data import DataLoader, Dataset
# from transformers import CLIPProcessor, CLIPModel
# import timm
# import numpy as np
# from PIL import Image

from datasets import load_dataset

# # ds = load_dataset("pixparse/cc3m-wds")
# ds = load_dataset("julianmoraes/doodles-captions-manual", split="train")
ds = load_dataset("eltorio/ROCOv2-radiology", split="test") # train test val
print(ds)
print(ds.to_dict().keys())
print(len(ds.to_dict()['image']))
print(len(ds.to_dict()['caption']))
print(type(ds.to_dict()['image'][0]))
print(type(ds.to_dict()['image'][0]['path']))
print(type(ds.to_dict()['image'][0]['bytes']))
print(ds.to_dict()['image'][0].keys())
print(ds.to_dict()['caption'][0])

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