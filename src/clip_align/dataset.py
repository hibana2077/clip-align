from torchvision import transforms
from PIL import Image
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from transformers import CLIPModel, CLIPProcessor
import datasets  # HuggingFace datasets
from .flickr30k import FlickrDataset
import numpy as np

DATASET_DICT = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    # "flickr30k": FlickrDataset,  # 替換為您的自定義數據集
}

def default_transforms():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class EmbeddingDataset(Dataset):
    def __init__(self, dataset, img_model, ann_file=None, img_model_transform=None, device=None, **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 初始化數據集
        if dataset == "flickr30k":
            self.dataset_name = dataset
            # 使用HuggingFace的load_dataset加載自定義數據集
            self.dataset = FlickrDataset()
            self.dataset.download_and_prepare()  # 需要手動下載和準備數據
            self.dataset = self.dataset.as_dataset()  # 轉換為可索引格式
        else:
            # 處理CIFAR等其他數據集
            self.dataset_name = dataset
            self.dataset = DATASET_DICT[dataset](root="./data", train=True, download=True)
            self.dataset = list(zip(self.dataset.data, self.dataset.targets))
        
        # 初始化模型
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.img_features = timm.create_model(img_model, pretrained=True).to(device)
        self.img_features = torch.nn.Sequential(*list(self.img_features.children())[:-1])
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 設置轉換
        self.img_model_transform = img_model_transform or default_transforms()
        
        # 評估模式
        self.clip_model.eval()
        self.img_features.eval()

        # 獲取嵌入大小
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        self.img_model_embedding_size = self.img_features(dummy_input).shape[1]
        self.clip_model_embedding_size = self.clip_model.get_image_features(dummy_input).shape[1]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # 根據數據集類型獲取數據
        if self.dataset_name == "flickr30k":
            # 處理HuggingFace數據集格式（您的F30kDataset）
            item = self.dataset['test'][index]
            img = item["image"]  # PIL
            # label = item["caption"][0]
            label = item["caption"][item["caption"].index(min(item["caption"], key=len))] # 獲取最短的caption
        else:
            # 處理CIFAR等其他數據集
            img, label = self.dataset[index]
            img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        
        # 獲取CLIP嵌入
        with torch.no_grad():
            clip_inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
            clip_embedding = self.clip_model.get_image_features(**clip_inputs).squeeze()
            if self.dataset_name == "flickr30k":
                # get text embedding
                text_inputs = self.clip_processor(text=label, return_tensors="pt").to(self.device)
                label = self.clip_model.get_text_features(**text_inputs).squeeze()

        # 獲取ResNet嵌入
        resnet_img = self.img_model_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_embedding = self.img_features(resnet_img).squeeze()

        return clip_embedding, img_embedding, label

# test
if __name__ == "__main__":
    # dataset = EmbeddingDataset("cifar10", "resnet50")
    dataset = EmbeddingDataset("flickr30k", "resnet50")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"CLIP Model Embedding Size: {dataset.clip_model_embedding_size}")
    print(f"ResNet Model Embedding Size: {dataset.img_model_embedding_size}")
    
    for clip_embedding, img_embedding, label in dataloader:
        print(f"CLIP Embedding Shape: {clip_embedding.shape}")
        print(f"ResNet Embedding Shape: {img_embedding.shape}")
        print(f"Label: {label}")
        break