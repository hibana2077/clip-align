import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# self defined dataset
from clip_align.dataset import EmbeddingDataset
from clip_align.converter import Converter
from clip_align.loss import AlignLoss
from clip_align.vis import visualize_projection

# Config
# DATASET_NAME = "cifar10"
DATASET_NAME = "flickr30k"
MODEL_NAME = "resnet18"
# MODEL_NAME = "resnet50"

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def get_dataloader(batch_size=128, preload=True, cache_dir=None):
    # 加载原始数据集
    dataset = EmbeddingDataset(DATASET_NAME, MODEL_NAME)
    
    # 预加载数据到内存并支持缓存
    if preload and cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{DATASET_NAME}_{MODEL_NAME}_cache.pt")
        
        # 检查缓存文件是否存在
        if os.path.exists(cache_file):
            print("Loading data from cache...")
            loaded = torch.load(cache_file)
            clip_embeddings = loaded['clip_embeddings']
            resnet_embeddings = loaded['resnet_embeddings']
            labels = loaded['labels']
        else:
            print("Preloading data and saving to cache...")
            clip_embeddings = []
            resnet_embeddings = []
            labels = []
            for clip_emb, resnet_emb, label in tqdm(dataset, desc="Preloading data"):
                clip_embeddings.append(clip_emb)
                resnet_embeddings.append(resnet_emb)
                labels.append(label)
            
            # 转换为张量
            clip_embeddings = torch.stack(clip_embeddings)
            resnet_embeddings = torch.stack(resnet_embeddings)
            if DATASET_NAME != "flickr30k":
                labels = torch.tensor(labels)
            if DATASET_NAME == "flickr30k":
            
            # 保存缓存
            torch.save({
                'clip_embeddings': clip_embeddings,
                'resnet_embeddings': resnet_embeddings,
                'labels': labels,
            }, cache_file)
        
        # 创建新的TensorDataset
        print(f"clip: {type(clip_embeddings)}")
        print(f"img: {type(resnet_embeddings)}")
        print(f"label: {type(labels)}")
        dataset = TensorDataset(clip_embeddings, resnet_embeddings, labels)
        clip_model_embedding_size = clip_embeddings.size(1)
        img_model_embedding_size = resnet_embeddings.size(1)
    else:
        # 从原始数据集获取尺寸
        clip_model_embedding_size = dataset.clip_model_embedding_size
        img_model_embedding_size = dataset.img_model_embedding_size

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)  # 固定随机种子
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, clip_model_embedding_size, img_model_embedding_size

if __name__ == '__main__':
    # 创建数据加载器
    train_loader, val_loader, clip_model_embedding_size, img_model_embedding_size = get_dataloader(
        preload=True,
        cache_dir="./cache"
    )
    print(f"CLIP Model Embedding Size: {clip_model_embedding_size}")
    print(f"ResNet Model Embedding Size: {img_model_embedding_size}")
    # 测试数据加载器
    for clip_embedding, resnet_embedding, label in train_loader:
        print(f"CLIP Embedding Shape: {clip_embedding.shape}")
        print(f"ResNet Embedding Shape: {resnet_embedding.shape}")
        print(f"Label: {label}")
        break

    for clip_embedding, resnet_embedding, label in val_loader:
        print(f"CLIP Embedding Shape: {clip_embedding.shape}")
        print(f"ResNet Embedding Shape: {resnet_embedding.shape}")
        print(f"Label: {label}")
        break
    
    # 创建模型
    converter = Converter(img_model_embedding_size, clip_model_embedding_size).to(device)
    align_loss = AlignLoss().to(device)
    optimizer = torch.optim.AdamW(
        converter.parameters(), 
        lr=3e-4,
        weight_decay=1e-4  # 添加L2正则化
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    # 训练模型
    num_epochs = 30
    for epoch in range(num_epochs):
        converter.train()
        running_loss = 0.0
        for clip_embedding, resnet_embedding, label in tqdm(train_loader):
            clip_embedding = clip_embedding.to(device)
            resnet_embedding = resnet_embedding.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = converter(resnet_embedding)
            loss = align_loss(output, clip_embedding)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 验证模型
    converter.eval()
    val_loss = 0.0
    all_clip_embeddings = []
    all_resnet_embeddings = []
    all_labels = []

    with torch.no_grad():
        for clip_embedding, resnet_embedding, label in val_loader:
            clip_embedding = clip_embedding.to(device)
            resnet_embedding = resnet_embedding.to(device)
            label = label.to(device)

            output = converter(resnet_embedding)
            loss = align_loss(output, clip_embedding)
            val_loss += loss.item()

            all_clip_embeddings.append(clip_embedding.cpu())
            all_resnet_embeddings.append(resnet_embedding.cpu())
            all_labels.append(label.cpu())
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    all_clip_embeddings = torch.cat(all_clip_embeddings)
    all_resnet_embeddings = torch.cat(all_resnet_embeddings)
    all_labels = torch.cat(all_labels)
    print(f"All CLIP Embeddings Shape: {all_clip_embeddings.shape}")
    print(f"All ResNet Embeddings Shape: {all_resnet_embeddings.shape}")
    print(f"All Labels Shape: {all_labels.shape}")
    # 可视化
    visualize_projection(all_clip_embeddings, all_labels, save_name="clip_projection.png")
    visualize_projection(all_resnet_embeddings, all_labels, save_name="resnet_projection.png")