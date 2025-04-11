import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
import timm
import numpy as np
from PIL import Image

from datasets import load_dataset

# ds = load_dataset("pixparse/cc3m-wds")
# ds = load_dataset("julianmoraes/doodles-captions-manual", split="train")
ds = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval", split="test")
print(ds)
print(ds.to_dict().keys())
print(len(ds.to_dict()['image']))
print(len(ds.to_dict()['caption']))
print(type(ds.to_dict()['image'][0]))
print(type(ds.to_dict()['image'][0]['path']))
print(type(ds.to_dict()['image'][0]['bytes']))
print(ds.to_dict()['image'][0].keys())
print(ds.to_dict()['caption'][0][0])