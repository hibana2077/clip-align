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
from clip_align.converter import Converter, Converter_Att, Converter_Linear, HilbertProjectionConverter, ProjectionConverter
from clip_align.loss import AlignLoss
from clip_align.vis import visualize_projection, visualize_similarity

# Config
# DATASET_NAME = "cifar10"
# DATASET_NAME = "flickr30k"
DATASET_NAME = "mscoco"
# MODEL_NAME = "resnet18"
# MODEL_NAME = "dla34"
# MODEL_NAME = "mobilenetv4_hybrid_medium"
MODEL_NAME = "vit_xsmall_patch16_clip_224"
# MODEL_NAME = "tiny_vit_11m_224.dist_in22k_ft_in1k"
# MODEL_NAME = "eva02_base_patch14_448"
# MODEL_NAME = "nextvit_small"
# MODEL_NAME = "resnet50"
SPLIT_RATIO = 0.9

# CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
# CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
# CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def get_dataloader(batch_size=128, preload=True, cache_dir=None):
    # Load the original dataset and set the batch size
    processing_batch_size = 300  # Batch size for preprocessing
    dataset = EmbeddingDataset(DATASET_NAME,
                               MODEL_NAME,
                               batch_size=processing_batch_size,
                               clip_model_name=CLIP_MODEL_NAME)
    
    # Get embedding sizes
    clip_model_embedding_size = dataset.clip_model_embedding_size
    img_model_embedding_size = dataset.img_model_embedding_size

    # Already preloaded into memory, supports caching
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{DATASET_NAME}_{MODEL_NAME}_cache.pt")
        
        # Check if the cache file exists
        if os.path.exists(cache_file) and preload:
            print("Loading data from cache...")
            loaded = torch.load(cache_file)
            clip_embeddings = loaded['clip_embeddings']
            resnet_embeddings = loaded['resnet_embeddings']
            labels = loaded['labels']
            
            # Create a new TensorDataset
            dataset = TensorDataset(clip_embeddings, resnet_embeddings, labels)
        else:
            print("Using preprocessed dataset...")
            # Save the embeddings to cache
            if preload:
                torch.save({
                    'clip_embeddings': dataset.clip_embeddings,
                    'resnet_embeddings': dataset.img_embeddings,
                    'labels': dataset.text_embeddings,
                }, cache_file)
                print(f"Cache saved to {cache_file}")

    # Split the dataset
    print(len(dataset))
    train_size = int(SPLIT_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)  # Fix random seed
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create DataLoaders
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
    # Create dataset and dataloader
    train_loader, val_loader, clip_model_embedding_size, img_model_embedding_size = get_dataloader(
        preload=False,
        cache_dir="./cache"
    )
    print(f"CLIP Model Embedding Size: {clip_model_embedding_size}")
    print(f"ResNet Model Embedding Size: {img_model_embedding_size}")

    # Data stats
    clip_embeddings_stats = []
    resnet_embeddings_stats = []
    labels_stats = []
    for clip_embedding, resnet_embedding, label in tqdm(train_loader):
        clip_embeddings_stats.append(clip_embedding)
        resnet_embeddings_stats.append(resnet_embedding)
        labels_stats.append(label)
    clip_embeddings_stats = torch.cat(clip_embeddings_stats)
    resnet_embeddings_stats = torch.cat(resnet_embeddings_stats)
    labels_stats = torch.cat(labels_stats)
    print(f"CLIP Embedding Stats: {clip_embeddings_stats.mean()}, {clip_embeddings_stats.std()}, {clip_embeddings_stats.min()}, {clip_embeddings_stats.max()}, {clip_embeddings_stats.median()}")
    print(f"ResNet Embedding Stats: {resnet_embeddings_stats.mean()}, {resnet_embeddings_stats.std()}, {resnet_embeddings_stats.min()}, {resnet_embeddings_stats.max()}, {resnet_embeddings_stats.median()}")
    print(f"Label Stats: {labels_stats.mean()}, {labels_stats.std()}, {labels_stats.min()}, {labels_stats.max()}, {labels_stats.median()}")

    # 测试数据加载器
    for clip_embedding, resnet_embedding, label in train_loader:
        print(f"CLIP Embedding Shape: {clip_embedding.shape}")
        # print(f"CLIP Embedding Stats: {clip_embedding.mean()}, {clip_embedding.std()}, {clip_embedding.min()}, {clip_embedding.max()}, {clip_embedding.median()}")
        print(f"ResNet Embedding Shape: {resnet_embedding.shape}")
        # print(f"ResNet Embedding Stats: {resnet_embedding.mean()}, {resnet_embedding.std()}, {resnet_embedding.min()}, {resnet_embedding.max()}, {resnet_embedding.median()}")
        print(f"Label: {label}")
        break

    for clip_embedding, resnet_embedding, label in val_loader:
        print(f"CLIP Embedding Shape: {clip_embedding.shape}")
        print(f"ResNet Embedding Shape: {resnet_embedding.shape}")
        print(f"Label: {label}")
        break
    
    # 创建模型
    converter = Converter(img_model_embedding_size, clip_model_embedding_size).to(device)
    # converter = HilbertProjectionConverter(
    #     img_model_embedding_size,
    #     clip_model_embedding_size).to(device)
    # converter = ProjectionConverter(
    #     img_model_embedding_size,
    #     clip_model_embedding_size).to(device)
    # converter = Converter_Att(
    #     img_model_embedding_size,
    #     clip_model_embedding_size,
    #     hidden_dim=1024
    # ).to(device)
    # converter = Converter_Linear(
    #     img_model_embedding_size,
    #     clip_model_embedding_size
    # ).to(device)
    print(f"Parameter Count: {sum(p.numel() for p in converter.parameters())/1e6:.2f}M")
    align_loss = AlignLoss(
        alpha=0.5,
        temperature=0.07,
        similarity_mode='cosine',
    ).to(device)
    # align_loss = nn.MSELoss().to(device)
    # align_loss = nn.L1Loss().to(device)
    optimizer = torch.optim.AdamW(
        converter.parameters(),
        lr=3e-4,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=3e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    # 训练模型
    num_epochs = 100
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training
        converter.train()
        running_loss = 0.0
        for clip_embedding, resnet_embedding, label in tqdm(train_loader):
            clip_embedding = clip_embedding.to(device)
            resnet_embedding = resnet_embedding.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            if isinstance(converter, HilbertProjectionConverter):
                output, reg_loss = converter(resnet_embedding)
                loss = align_loss(output, clip_embedding) + reg_loss
            else:
                output = converter(resnet_embedding)
                loss = align_loss(output, clip_embedding)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        
        # Validation after each epoch
        converter.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clip_embedding, resnet_embedding, label in val_loader:
                clip_embedding = clip_embedding.to(device)
                resnet_embedding = resnet_embedding.to(device)
                label = label.to(device)

                if isinstance(converter, HilbertProjectionConverter):
                    output, reg_loss = converter(resnet_embedding)
                    loss = align_loss(output, clip_embedding) + reg_loss
                else:
                    output = converter(resnet_embedding)
                    loss = align_loss(output, clip_embedding)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Best validation loss so far. Saving model...")
            torch.save(converter.state_dict(), f"converter_{DATASET_NAME}_{MODEL_NAME}.pth")
        else:
            print(f"Validation loss did not improve. Not saving model.")

    # Final comprehensive validation with embedding collection
    converter.eval()
    all_clip_embeddings = []
    all_resnet_embeddings = []
    all_convert_embeddings = []
    all_labels = []

    with torch.no_grad():
        for clip_embedding, resnet_embedding, label in val_loader:
            clip_embedding = clip_embedding.to(device)
            resnet_embedding = resnet_embedding.to(device)
            label = label.to(device)

            if isinstance(converter, HilbertProjectionConverter):
                output, reg_loss = converter(resnet_embedding)
            else:
                output = converter(resnet_embedding)
            
            all_clip_embeddings.append(clip_embedding.cpu())
            all_resnet_embeddings.append(resnet_embedding.cpu())
            all_convert_embeddings.append(output.cpu())
            all_labels.append(label.cpu())

    all_clip_embeddings = torch.cat(all_clip_embeddings)
    all_resnet_embeddings = torch.cat(all_resnet_embeddings)
    all_convert_embeddings = torch.cat(all_convert_embeddings)
    all_labels = torch.cat(all_labels)

    print(f"Final collected embeddings shapes:")
    print(f"CLIP embeddings: {all_clip_embeddings.shape}")
    print(f"ResNet embeddings: {all_resnet_embeddings.shape}")
    print(f"Converted embeddings: {all_convert_embeddings.shape}")
    print(f"Labels: {all_labels.shape}")

    print(f"All CLIP Embeddings Shape: {all_clip_embeddings.shape}")
    print(f"All ResNet Embeddings Shape: {all_resnet_embeddings.shape}")
    print(f"All Labels Shape: {all_labels.shape}")
    # 可视化
    visualize_projection(all_clip_embeddings, all_labels, save_name="clip_projection.png", label_type="tensor")
    visualize_projection(all_convert_embeddings, all_labels, save_name="convert_projection.png", label_type="tensor")
    if all_resnet_embeddings.shape[-1] == all_labels.shape[-1]:
        visualize_projection(all_resnet_embeddings, all_labels, save_name="resnet_projection.png", label_type="tensor")
        visualize_similarity(all_clip_embeddings, all_resnet_embeddings, all_convert_embeddings, save_prefix="similarity")

    # Similarity
    print(f"CLIP to Converted Similarity: {F.cosine_similarity(all_clip_embeddings, all_convert_embeddings).mean()}")