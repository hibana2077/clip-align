import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
import timm
import numpy as np
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"

model.eval()
model.to(device)

test_img_tensor = torch.randn(600, 3, 224, 224).to(device)
# test_text_tensor = torch.randn(600, 5, 77).to(device)
test_text_tensor = torch.randint(0, 1000, (600, 77)).to(device)  # Dummy text tensor
with torch.no_grad():
    # Get image features
    clip_image_embedding = model.get_image_features(test_img_tensor)
    # Get text features
    clip_text_embedding = model.get_text_features(test_text_tensor)

print(clip_image_embedding.shape)  # Should be (600, 512)
print(clip_text_embedding.shape)  # Should be (600, 512)