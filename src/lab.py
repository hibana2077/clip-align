import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
import timm
import numpy as np
from PIL import Image

img_model = timm.create_model("vit_xsmall_patch16_clip_224", pretrained=True, num_classes=0)
print(img_model)
test = torch.randn(1, 3, 224, 224)
out = img_model(test)
print(out.shape)