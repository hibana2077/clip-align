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
import os
import pickle
import hashlib

DATASET_DICT = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    # "flickr30k": FlickrDataset,  # 替換為您的自定義數據集
}

class EmbeddingDataset(Dataset):
    def __init__(self, dataset, img_model, ann_file=None, img_model_transform=None, device=None, batch_size=512, clip_model_name="openai/clip-vit-base-patch32", use_cache=True, cache_dir="./cache", **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.clip_model_name = clip_model_name

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # 初始化數據集
        if dataset == "flickr30k":
            self.dataset_name = dataset
            # 使用HuggingFace的load_dataset加載自定義數據集
            self.dataset = FlickrDataset()
            self.dataset.download_and_prepare()  # 需要手動下載和準備數據
            self.dataset = self.dataset.as_dataset()  # 轉換為可索引格式
            self.dataset = self.dataset['test'][:30000]
            self.dataset_size = len(self.dataset['image'])
        elif dataset == "mscoco":
            self.dataset_name = dataset
            self.dataset = MSCOCODataset(
                parquet_file_path="./data/mscoco_test2017.parquet",
                index_url="https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/resolve/main/mscoco.parquet?download=true",
                split="train2017",
                download=True,
                cache_dir="./data/mscoco_cache",
                sample=1000
            )
            self.dataset.download_all(num_workers=8)
            self.dataset_size = len(self.dataset)
        else:
            # 處理CIFAR等其他數據集
            self.dataset_name = dataset
            self.dataset = DATASET_DICT[dataset](root="./data", train=True, download=True)
            self.dataset = list(zip(self.dataset.data, self.dataset.targets))
            self.dataset_size = len(self.dataset)
        
        # 初始化模型
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.img_features = timm.create_model(img_model, pretrained=True, num_classes=0).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
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

    def _get_cache_path(self, data_type):
        """Generate a cache file path based on dataset and model info"""
        cache_key = f"{self.dataset_name}_{self.clip_model_name.replace('/', '_')}_{data_type}_{self.dataset_size}"
        if hasattr(self, 'img_features') and data_type == "img_embeddings":
            cache_key += f"_{self.img_features.__class__.__name__}"
        
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _load_cache(self, cache_path):
        """Load cached embeddings if they exist"""
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_cache(self, data, cache_path):
        """Save embeddings to cache"""
        print(f"Saving embeddings to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    def preprocess_dataset(self):
        """預處理整個數據集並將嵌入存儲在內存中，使用緩存機制"""
        print(f"預處理數據集 {self.dataset_name}...")
        
        clip_cache_path = self._get_cache_path("clip_embeddings")
        img_cache_path = self._get_cache_path("img_embeddings")
        text_cache_path = self._get_cache_path("text_embeddings")
        
        # Try to load from cache
        if self.use_cache:
            self.clip_embeddings = self._load_cache(clip_cache_path)
            self.img_embeddings = self._load_cache(img_cache_path)
            self.text_embeddings = self._load_cache(text_cache_path)
            
            # If all cache loaded successfully
            if all(x is not None for x in [self.clip_embeddings, self.img_embeddings, self.text_embeddings]):
                print("Successfully loaded all embeddings from cache.")
                return
        
        # Initialize embeddings lists and set flags for what needs processing
        self.clip_embeddings_cached = self.clip_embeddings is not None
        self.text_embeddings_cached = self.text_embeddings is not None
        
        # Initialize empty lists for embeddings that need processing
        if not self.clip_embeddings_cached:
            self.clip_embeddings = []
        if not self.text_embeddings_cached:
            self.text_embeddings = []
        
        # Always process img_embeddings as they may need to be recalculated
        self.img_embeddings = []
        
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
            
            # Process embeddings as needed
            batch_clip_embeddings, batch_img_embeddings, batch_text_embeddings = self.process_batch(
                batch_imgs, 
                batch_texts, 
                process_clip=not self.clip_embeddings_cached,
                process_text=not self.text_embeddings_cached
            )
            
            # Add embeddings only if they were processed
            if not self.clip_embeddings_cached:
                self.clip_embeddings.extend(batch_clip_embeddings)
            if not self.text_embeddings_cached:
                self.text_embeddings.extend(batch_text_embeddings)
                
            # Always add img embeddings
            self.img_embeddings.extend(batch_img_embeddings)
        
        # Convert to tensor if newly processed
        if not self.clip_embeddings_cached:
            self.clip_embeddings = torch.stack(self.clip_embeddings)
            if self.use_cache:
                self._save_cache(self.clip_embeddings, clip_cache_path)
                
        if not self.text_embeddings_cached:
            self.text_embeddings = torch.stack(self.text_embeddings) if isinstance(self.text_embeddings[0], torch.Tensor) else torch.tensor(self.text_embeddings)
            if self.use_cache:
                self._save_cache(self.text_embeddings, text_cache_path)
        
        # Always save img embeddings
        self.img_embeddings = torch.stack(self.img_embeddings)
        if self.use_cache:
            self._save_cache(self.img_embeddings, img_cache_path)
        
        print(f"預處理完成。數據集大小: {len(self.clip_embeddings)}")

    def process_batch(self, batch_imgs, batch_texts, process_clip=True, process_text=True):
        """批次處理圖像和文本以獲取嵌入"""
        clip_embeddings_list = []
        text_embeddings_list = []
        
        # 處理CLIP圖像嵌入和文本嵌入
        if process_clip or process_text:
            with torch.no_grad():
                if process_clip:
                    clip_inputs = self.clip_processor(images=batch_imgs, return_tensors="pt", padding=True).to(self.device)
                    clip_embeddings = self.clip_model.get_image_features(**clip_inputs)
                    clip_embeddings_list = [emb.cpu() for emb in clip_embeddings]
                
                if process_text:
                    if self.dataset_name in ["flickr30k", "mscoco"]:
                        text_inputs = self.clip_processor(
                            text=batch_texts, 
                            return_tensors="pt", 
                            max_length=77, 
                            padding='max_length', 
                            truncation=True
                        ).to(self.device)
                        text_embeddings = self.clip_model.get_text_features(**text_inputs)
                        text_embeddings_list = [emb.cpu() for emb in text_embeddings]
                    else:
                        text_embeddings_list = [torch.tensor([int(t)], device="cpu") for t in batch_texts]
        
        # 處理模型特定圖像嵌入 (always process)
        batch_transformed_imgs = []
        for img in batch_imgs:
            batch_transformed_imgs.append(self.img_model_transform(img))
        
        batch_transformed_imgs = torch.stack(batch_transformed_imgs).to(self.device)
        
        with torch.no_grad():
            img_embeddings = self.img_features(batch_transformed_imgs)
        
        img_embeddings_list = [emb.cpu() for emb in img_embeddings]
        
        return clip_embeddings_list, img_embeddings_list, text_embeddings_list

    def __len__(self):
        return len(self.clip_embeddings)
    
    def __getitem__(self, index):
        """直接從預計算的嵌入中獲取項目"""
        clip_embedding = self.clip_embeddings[index].to(self.device)
        img_embedding = self.img_embeddings[index].to(self.device)
        label = self.text_embeddings[index].to(self.device)
        
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