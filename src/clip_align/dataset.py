import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
import timm
import numpy as np
from PIL import Image

DATASET_DICT = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "flickr30k": datasets.Flickr30k,  # 添加Flickr30k到數據集字典
}

def default_transforms():
    img_transform = Compose([
        Resize(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img_transform

class EmbeddingDataset(Dataset):
    def __init__(self, dataset, img_model, ann_file=None, img_model_transform=default_transforms(), device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 根據數據集類型初始化
        if dataset in ['cifar10', 'cifar100']:
            self.dataset_name = dataset
            self.dataset = DATASET_DICT[dataset](root="./data", train=True, download=True)
            # 將CIFAR數據轉換為(img, label)列表
            self.dataset = list(zip(self.dataset.data, self.dataset.targets))
        elif dataset == 'flickr30k':
            self.dataset_name = dataset
            if ann_file is None:
                raise ValueError("ann_file must be provided for Flickr30k dataset")
            # 加載Flickr30k數據集，返回圖像路徑和標注
            self.dataset = DATASET_DICT[dataset](root="./data/flickr30k", ann_file=ann_file)
            # 將標注轉換為0（或根據需求調整）
            self.dataset = [(img, 0) for img, _ in self.dataset]
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        # 初始化模型和轉換
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.resnet_features = timm.create_model(img_model, pretrained=True)
        self.resnet_features = torch.nn.Sequential(*list(self.resnet_features.children())[:-1]).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.img_model_transform = img_model_transform
        self.device = device
        
        # 設置模型為評估模式並測試虛擬輸入
        self.clip_model.eval()
        self.resnet_features.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        self.img_model_embedding_size = self.resnet_features(dummy_input).shape[1]
        self.clip_model_embedding_size = self.clip_model.get_image_features(dummy_input).shape[1]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_data, label = self.dataset[index]
        
        # 處理不同數據集的圖像類型
        if isinstance(img_data, np.ndarray):  # CIFAR圖像數組
            img = Image.fromarray(img_data)
        elif isinstance(img_data, Image.Image):  # Flickr30k已加載的PIL圖像
            img = img_data
        else:
            raise TypeError("Unsupported image type")
        
        # 獲取CLIP嵌入
        with torch.no_grad():
            clip_inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
            clip_embedding = self.clip_model.get_image_features(**clip_inputs).squeeze()
            # if self.dataset_name == 'flickr30k':
            #     # get text features for Flickr30k
        
        # 獲取ResNet嵌入
        resnet_img = self.img_model_transform(img).unsqueeze(0).to(self.device)
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