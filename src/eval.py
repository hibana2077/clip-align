import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json

# self defined dataset
from clip_align.cfg import DATASET_TYPE
from clip_align.infernece import original_clip_inference, converter_clip_inference
from clip_align.flickr30k import FlickrDataset
from clip_align.urban1k import Urban1k
from clip_align.mscoco import MSCOCODataset
from clip_align.converter import Converter, Converter_Att, Converter_Linear, HilbertProjectionConverter, ProjectionConverter
from clip_align.eval_utils import I2T, T2I

# Config
EVAL_DATASET_NAME = "urban1k"
# EVAL_DATASET_NAME = "mscoco"
# DATASET_NAME = "flickr30k"
DATASET_NAME = "mscoco"
MODEL_NAME = "resnet18"
# MODEL_NAME = "resnet50"
# MODEL_NAME = "mobilenetv4_hybrid_medium"
# MODEL_NAME = "vit_xsmall_patch16_clip_224"
# MODEL_NAME = "eva02_base_patch14_448"
# MODEL_NAME = "tiny_vit_11m_224.dist_in22k_ft_in1k"
CONVERTER_PT = f'./converter_{DATASET_NAME}_{MODEL_NAME}.pth'
CONVERTER_MODEL_TYPE = Converter

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def load_test_data(dataset_name:str):
    tasks = DATASET_TYPE[dataset_name]
    if dataset_name == "urban1k":
        test_dataset = Urban1k(root_dir="./data", download=True)
    elif dataset_name == "mscoco":
        test_dataset = MSCOCODataset(
            parquet_file_path="./data/mscoco_test2017.parquet",
            index_url="https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/resolve/main/mscoco.parquet?download=true",
            split="train2017",
            download=True,
            cache_dir="./data/mscoco_cache",
            sample=2000
        )
        test_dataset.download_all(num_workers=32)
    elif dataset_name == "flickr30k":
        test_dataset = FlickrDataset()
        test_dataset.download_and_prepare()
        test_dataset = test_dataset.as_dataset()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return test_dataset, tasks

def preprocess_data(dataset, clip_processor, img_model_transform):
    # Preprocess data for inference
    clip_images = []
    clip_texts = []
    img_images = []

    for img, text in tqdm(dataset, desc="Preprocessing data"):
        # Preprocess image for CLIP
        clip_image = clip_processor(images=img, return_tensors="pt")["pixel_values"]
        clip_images.append(clip_image)

        # Preprocess image for CNN
        img_image = img_model_transform(img).unsqueeze(0)
        img_images.append(img_image)

        # Preprocess text for CLIP
        clip_text = clip_processor(text=text, return_tensors="pt", max_length=77, padding='max_length', truncation=True)["input_ids"]
        clip_texts.append(clip_text)

    # Stack the tensors
    clip_images = torch.cat(clip_images, dim=0).to(device)
    img_images = torch.cat(img_images, dim=0).to(device)
    print(len(clip_texts))
    # stats tensor size clip_texts
    cnt = {}
    for i in range(len(clip_texts)):
        if clip_texts[i].shape[1] not in cnt:
            cnt[clip_texts[i].shape[1]] = 0
        cnt[clip_texts[i].shape[1]] += 1
    print(cnt)
    clip_texts = torch.cat(clip_texts, dim=0).to(device)

    return clip_images, img_images, clip_texts

if __name__ == "__main__":
    # Load dataset
    test_dataset, tasks = load_test_data(EVAL_DATASET_NAME)

    # Load CLIP model
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load CNN model
    img_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(img_model)
    img_model_transform = timm.data.create_transform(**data_config)

    # Preprocess data
    clip_images, img_images, clip_texts = preprocess_data(test_dataset, clip_processor, img_model_transform)
    print(f"clip_images: {clip_images.shape}")
    print(f"img_images: {img_images.shape}")
    print(f"clip_texts: {clip_texts.shape}")

    # Do the inference (Original CLIP)
    clip_image_embedding, clip_text_embedding = original_clip_inference(
        model_name="openai/clip-vit-base-patch32",
        image_set=clip_images,
        text_set=clip_texts,
        device=device
    )
    print(f"clip_image_embedding: {clip_image_embedding.shape}")
    print(f"clip_text_embedding: {clip_text_embedding.shape}")

    # Do the inference (Converter)
    converter_embedding, clip_text_embedding = converter_clip_inference(
        clip_model_name="openai/clip-vit-base-patch32",
        cnn_model_name=MODEL_NAME,
        converter_model_path=CONVERTER_PT,
        converter_model_type=CONVERTER_MODEL_TYPE,
        image_set=img_images,
        text_set=clip_texts,
        device=device
    )
    print(f"converter_embedding: {converter_embedding.shape}")
    print(f"clip_text_embedding: {clip_text_embedding.shape}")

    # Do evaluation
    print(f"{'='*20} Eval {MODEL_NAME} on {EVAL_DATASET_NAME} {'='*20}")
    # I2T
    i2t_recall_original = I2T(clip_image_embedding, clip_text_embedding, topk=(1, 5, 10))
    i2t_recall_converter = I2T(converter_embedding, clip_text_embedding, topk=(1, 5, 10))
    print(f"I2T Recall (Original): {i2t_recall_original}")
    print(f"I2T Recall (Converter): {i2t_recall_converter}")
    # T2I
    t2i_recall_original = T2I(clip_image_embedding, clip_text_embedding, topk=(1, 5, 10))
    t2i_recall_converter = T2I(converter_embedding, clip_text_embedding, topk=(1, 5, 10))
    print(f"T2I Recall (Original): {t2i_recall_original}")
    print(f"T2I Recall (Converter): {t2i_recall_converter}")
    # Save the results
    results = {
        "I2T Recall (Original)": i2t_recall_original,
        "I2T Recall (Converter)": i2t_recall_converter,
        "T2I Recall (Original)": t2i_recall_original,
        "T2I Recall (Converter)": t2i_recall_converter
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Evaluation results saved to evaluation_results.json")