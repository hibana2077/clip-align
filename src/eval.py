import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json

# self defined dataset
from clip_align.cfg import DATASET_TYPE
from clip_align.infernece import original_clip_inference, converter_clip_inference

from clip_align.eval_dataset.flickr1k import Flickr1k # torch like
from clip_align.eval_dataset.flux1k import Flux1k # torch like
from clip_align.eval_dataset.docci5k import DOCCI5k # torch like
from clip_align.eval_dataset.doodles1k import Doodles1k # torch like
from clip_align.eval_dataset.urban1k import Urban1k # torch like
from clip_align.eval_dataset.mscoco5k import MSCOCO5k # torch like
from clip_align.eval_dataset.roco10k import ROCOv2Dataset # torch like


from clip_align.converter import Converter, Converter_Att, Converter_Linear, HilbertProjectionConverter, ProjectionConverter
from clip_align.eval_utils import I2T, T2I

def abbreviate_number(n):
    """
    將大於等於 1000 的數字簡化為以 k 為單位的格式。
    範例:
      1000 -> "1k"
      5000 -> "5k"
      1500 -> "1.5k"
    """
    if n >= 1000:
        # 如果能整除 1000，就用整數表示
        if n % 1000 == 0:
            return f"{n // 1000}k"
        else:
            # 若有餘數，則保留一位小數
            return f"{n / 1000:.1f}k"
    else:
        return str(n)

# Config
import yaml
with open("cfg.yml", "r") as f:
    config = yaml.safe_load(f)
EVAL_DATASET_NAME = config["eval"]["EVAL_DATASET_NAME"]
DATASET_NAME = config["train"]["DATASET_NAME"]
MODEL_NAME = config["train"]["MODEL_NAME"]
CONVERTER_PT = f'./converter_{DATASET_NAME}_{MODEL_NAME}.pth'
CLIP_MODEL_NAME = config["train"]["CLIP_MODEL_NAME"]
CLIP_PRETRAINED = config["train"]["CLIP_PRETRAINED"]
SAMPLE_SIZE = config["train"]["SAMPLE"]
SAVE_FILE_NAME = f"{abbreviate_number(SAMPLE_SIZE)}_{DATASET_NAME}_on_{EVAL_DATASET_NAME}.json"

CONVERTER_MODEL_TYPE = Converter
# CONVERTER_MODEL_TYPE = HilbertProjectionConverter

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def load_test_data(dataset_name:str):
    tasks = DATASET_TYPE[dataset_name]
    if dataset_name == "urban1k":
        test_dataset = Urban1k(root_dir="./data", download=True)
    elif dataset_name == "mscoco5k":
        test_dataset = MSCOCO5k(split="test")
    elif dataset_name == "flickr1k":
        test_dataset = Flickr1k(split="test")
    elif dataset_name == "docci5k":
        test_dataset = DOCCI5k()
    elif dataset_name == "doodles1k":
        test_dataset = Doodles1k(split="train")
    elif dataset_name == "flux1k":
        test_dataset = Flux1k(split="train")
    elif dataset_name == "roco10k":
        test_dataset = ROCOv2Dataset(split="test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return test_dataset, tasks

def preprocess_data(dataset, clip_processor, img_model_transform, tokenizer):
    # Preprocess data for inference
    clip_images = []
    clip_texts = []
    img_images = []

    for img, text in tqdm(dataset, desc="Preprocessing data"):
        # Preprocess image for CLIP
        clip_image = clip_processor(img).unsqueeze(0)
        clip_images.append(clip_image)

        # Preprocess image for CNN
        img_image = img_model_transform(img).unsqueeze(0)
        img_images.append(img_image)

        # Preprocess text for CLIP
        # clip_text = clip_processor(text=text, return_tensors="pt", max_length=77, padding='max_length', truncation=True)["input_ids"]
        clip_text = tokenizer(text)
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

    # Load CLIP processor
    _, _, clip_processor = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED) if CLIP_PRETRAINED else open_clip.create_model_and_transforms(CLIP_MODEL_NAME)
    
    # Load tokenizer
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    # Load CNN model
    img_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(img_model)
    img_model_transform = timm.data.create_transform(**data_config)

    # Preprocess data
    clip_images, img_images, clip_texts = preprocess_data(test_dataset, clip_processor, img_model_transform, tokenizer)
    print(f"clip_images: {clip_images.shape}")
    print(f"img_images: {img_images.shape}")
    print(f"clip_texts: {clip_texts.shape}")

    # Do the inference (Original CLIP)
    clip_image_embedding, clip_text_embedding = original_clip_inference(
        model_name=CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        image_set=clip_images,
        text_set=clip_texts,
        device=str(device),
    )
    print(f"clip_image_embedding: {clip_image_embedding.shape}")
    print(f"clip_text_embedding: {clip_text_embedding.shape}")

    # Do the inference (Converter)
    converter_embedding, clip_text_embedding = converter_clip_inference(
        clip_model_name=CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        cnn_model_name=MODEL_NAME,
        converter_model_path=CONVERTER_PT,
        converter_model_type=CONVERTER_MODEL_TYPE,
        image_set=img_images,
        text_set=clip_texts,
        device=str(device),
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
    with open(SAVE_FILE_NAME, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {SAVE_FILE_NAME}")