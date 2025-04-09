from torchvision import transforms
from PIL import Image
import torch
import timm
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from transformers import CLIPModel, CLIPProcessor
import datasets  # HuggingFace datasets
from .flickr30k import FlickrDataset
from .mscoco import MSCOCODataset
import numpy as np

DATASET_DICT = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    # "flickr30k": FlickrDataset,  # 替換為您的自定義數據集
}

class EmbeddingDataset(Dataset):
    def __init__(self, dataset, img_model, ann_file=None, img_model_transform=None, device=None, batch_size=512, **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size

        # 初始化數據集
        if dataset == "flickr30k":
            self.dataset_name = dataset
            # 使用HuggingFace的load_dataset加載自定義數據集
            self.dataset = FlickrDataset()
            self.dataset.download_and_prepare()  # 需要手動下載和準備數據
            self.dataset = self.dataset.as_dataset()  # 轉換為可索引格式
            self.dataset = self.dataset['test'][:30000]
        elif dataset == "mscoco":
            self.dataset_name = dataset
            self.dataset = MSCOCODataset(
                parquet_file_path="./data/mscoco_test2017.parquet",
                index_url="https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/resolve/main/mscoco.parquet?download=true",
                split="train2017",
                download=True,
                cache_dir="./data/mscoco_cache",
                sample=50000
            )
            self.dataset.download_all(num_workers=8)
        else:
            # 處理CIFAR等其他數據集
            self.dataset_name = dataset
            self.dataset = DATASET_DICT[dataset](root="./data", train=True, download=True)
            self.dataset = list(zip(self.dataset.data, self.dataset.targets))
        
        # 初始化模型
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.img_features = timm.create_model(img_model, pretrained=True, num_classes=0).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 設置轉換
        data_config = timm.data.resolve_model_data_config(self.img_features)
        self.img_model_transform = img_model_transform or timm.data.create_transform(**data_config)
        
        # 評估模式
        self.clip_model.eval()
        self.img_features.eval()

        # 獲取嵌入大小
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        self.img_model_embedding_size = self.img_features(self.img_model_transform(dummy_input.squeeze(0)).unsqueeze(0)).shape[1]
        self.clip_model_embedding_size = self.clip_model.get_image_features(dummy_input).shape[1]
        
        # 預處理數據
        self.preprocess_dataset()

    def preprocess_dataset(self):
        """預處理整個數據集並將嵌入存儲在內存中"""
        print(f"預處理數據集 {self.dataset_name}...")
        self.clip_embeddings = []
        self.img_embeddings = []
        self.text_embeddings = []
        
        # 計算批次數
        total_samples = len(self.dataset['image']) if self.dataset_name == "flickr30k" else len(self.dataset)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="預處理批次"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            
            batch_imgs = []
            batch_texts = []
            
            # 收集當前批次的數據
            for idx in range(start_idx, end_idx):
                if self.dataset_name == "flickr30k":
                    # item = self.dataset[idx]
                    # img = item["image"]  # PIL
                    # text = item["caption"][item["caption"].index(min(item["caption"], key=len))]  # 獲取最短的caption
                    img = self.dataset['image'][idx]  # PIL
                    text = self.dataset['caption'][idx][0]
                elif self.dataset_name == "mscoco":
                    img, text = self.dataset[idx]  # PIL, str
                else:
                    img, label = self.dataset[idx]
                    img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
                    text = str(label)  # 將標籤轉換為字符串
                
                batch_imgs.append(img)
                batch_texts.append(text)
            
            # 批次處理圖像嵌入
            batch_clip_embeddings, batch_img_embeddings, batch_text_embeddings = self.process_batch(batch_imgs, batch_texts)
            
            # 擴展嵌入列表
            self.clip_embeddings.extend(batch_clip_embeddings)
            self.img_embeddings.extend(batch_img_embeddings)
            self.text_embeddings.extend(batch_text_embeddings)
        
        # 轉換為張量
        self.clip_embeddings = torch.stack(self.clip_embeddings)
        self.img_embeddings = torch.stack(self.img_embeddings)
        self.text_embeddings = torch.stack(self.text_embeddings)
        
        print(f"預處理完成。數據集大小: {len(self.clip_embeddings)}")

    def process_batch(self, batch_imgs, batch_texts):
        """批次處理圖像和文本以獲取嵌入"""
        # 處理CLIP圖像嵌入
        with torch.no_grad():
            clip_inputs = self.clip_processor(images=batch_imgs, return_tensors="pt", padding=True).to(self.device)
            clip_embeddings = self.clip_model.get_image_features(**clip_inputs)
            
            # 處理文本嵌入（如果適用）
            if self.dataset_name in ["flickr30k", "mscoco"]:
                text_inputs = self.clip_processor(
                    text=batch_texts, 
                    return_tensors="pt", 
                    max_length=77, 
                    padding='max_length', 
                    truncation=True
                ).to(self.device)
                text_embeddings = self.clip_model.get_text_features(**text_inputs)
            else:
                # 對於其他數據集，只需使用標籤
                text_embeddings = torch.tensor([int(t) for t in batch_texts]).to(self.device)
        
        # 處理模型特定圖像嵌入
        batch_transformed_imgs = []
        for img in batch_imgs:
            batch_transformed_imgs.append(self.img_model_transform(img))
        
        batch_transformed_imgs = torch.stack(batch_transformed_imgs).to(self.device)
        
        with torch.no_grad():
            img_embeddings = self.img_features(batch_transformed_imgs)
        
        # 拆分批次嵌入為單獨的嵌入
        clip_embeddings_list = [emb for emb in clip_embeddings]
        img_embeddings_list = [emb for emb in img_embeddings]
        
        if self.dataset_name in ["flickr30k", "mscoco"]:
            text_embeddings_list = [emb for emb in text_embeddings]
        else:
            text_embeddings_list = [torch.tensor([int(t)], device=self.device) for t in batch_texts]
        
        return clip_embeddings_list, img_embeddings_list, text_embeddings_list

    def __len__(self):
        return len(self.clip_embeddings)
    
    def __getitem__(self, index):
        """直接從預計算的嵌入中獲取項目"""
        clip_embedding = self.clip_embeddings[index]
        img_embedding = self.img_embeddings[index]
        label = self.text_embeddings[index]
        
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