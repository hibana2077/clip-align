import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
import timm
from PIL import Image

DATASET_DICT = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
}

def default_transforms():
    img_transform = Compose([
        Resize(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img_transform

class EmbeddingDataset(Dataset):
    def __init__(self, dataset, img_model, img_model_transform=default_transforms(), device=None):
        # Automatically select CUDA if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = DATASET_DICT[dataset](root="./data", train=True, download=True)
        self.dataset = [(img, label) for img, label in zip(self.dataset.data, self.dataset.targets)]
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.resnet_features = timm.create_model(img_model, pretrained=True)
        self.resnet_features = nn.Sequential(*list(self.resnet_features.children())[:-1])
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.img_model_transform = img_model_transform
        self.device = device
        self.clip_model.to(self.device)
        self.resnet_features.to(self.device)
        self.clip_model.eval()
        self.resnet_features.eval()
        # Test a dummy input on the chosen device to ensure models are working with CUDA if selected
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.img_model_embedding_size = self.resnet_features(dummy_input).shape[1]
        self.clip_model_embedding_size = self.clip_model.get_image_features(dummy_input).shape[1]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = Image.fromarray(img)
        
        # Get CLIP image embedding
        with torch.no_grad():
            clip_inputs = self.clip_processor(images=img, return_tensors="pt")
            # Move inputs to the device
            for key in clip_inputs:
                clip_inputs[key] = clip_inputs[key].to(self.device)
            clip_embedding = self.clip_model.get_image_features(**clip_inputs).squeeze()
        
        # Get ResNet image embedding
        resnet_img = self.img_model_transform(img).unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            resnet_embedding = self.resnet_features(resnet_img).squeeze()

        return clip_embedding, resnet_embedding, label

# test
if __name__ == "__main__":
    dataset = EmbeddingDataset("cifar10", "resnet50")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"CLIP Model Embedding Size: {dataset.clip_model_embedding_size}")
    print(f"ResNet Model Embedding Size: {dataset.img_model_embedding_size}")
    
    for clip_embedding, resnet_embedding, label in dataloader:
        print(f"CLIP Embedding Shape: {clip_embedding.shape}")
        print(f"ResNet Embedding Shape: {resnet_embedding.shape}")
        print(f"Label: {label}")
        break