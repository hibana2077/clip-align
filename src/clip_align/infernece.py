import torch
import timm
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import open_clip

from .converter import Converter, Converter_Att, Converter_Linear, HilbertProjectionConverter, ProjectionConverter

def original_clip_inference(
    model_name: str,
    pretrained: str,
    image_set: torch.Tensor, # (N, 3, 224, 224)
    text_set: torch.Tensor, # (N, 77) - Already tokenized text
    device: str = 'cuda',
):
    # Load the OpenCLIP model
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained) if pretrained else open_clip.create_model_and_transforms(model_name)
    model = model.to(device)
    model.eval()
    
    batch_size = 128
    clip_image_embedding_list = []
    clip_text_embedding_list = []
    
    with torch.no_grad(), torch.autocast(device):
        # Process in batches
        for i in range(0, len(image_set), batch_size):
            # Get batch
            image_batch = image_set[i:i+batch_size].to(device)
            text_batch = text_set[i:i+batch_size].to(device)
            
            # Get image features
            batch_clip_image_embedding = model.encode_image(image_batch)
            # Normalize image features
            batch_clip_image_embedding = batch_clip_image_embedding / batch_clip_image_embedding.norm(dim=-1, keepdim=True)
            clip_image_embedding_list.append(batch_clip_image_embedding)
            
            # Get text features
            batch_clip_text_embedding = model.encode_text(text_batch)
            # Normalize text features
            batch_clip_text_embedding = batch_clip_text_embedding / batch_clip_text_embedding.norm(dim=-1, keepdim=True)
            clip_text_embedding_list.append(batch_clip_text_embedding)
        
        # Concatenate all batch results
        clip_image_embedding = torch.cat(clip_image_embedding_list, dim=0)
        clip_text_embedding = torch.cat(clip_text_embedding_list, dim=0)

    return clip_image_embedding, clip_text_embedding

def converter_clip_inference(
        clip_model_name: str,
        pretrained: str,
        cnn_model_name: str,
        converter_model_path: str,
        converter_model_type: nn.Module,
        image_set: torch.Tensor, # (N, 3, 224, 224)
        text_set: torch.Tensor, # (N, 77)
        device: str = 'cuda',
):
    # Load the OpenCLIP model
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained) if pretrained else open_clip.create_model_and_transforms(clip_model_name)
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Load the CNN model
    cnn_model = timm.create_model(cnn_model_name, pretrained=True, num_classes=0).to(device)
    cnn_model.eval()

    # Load the Converter model
    print(f"converter_model_path: {converter_model_path}")
    converter_model = torch.load(converter_model_path, weight_only=False)
    converter_model.to(device)
    converter_model.eval()

    with torch.no_grad(), torch.autocast(device):
        batch_size = 8
        converter_embedding_list = []
        clip_text_embedding_list = []
        
        # Process in batches
        for i in range(0, len(image_set), batch_size):
            # Get batch
            image_batch = image_set[i:i+batch_size].to(device)
            text_batch = text_set[i:i+batch_size].to(device)
            
            # Get image features
            cnn_image_embedding = cnn_model(image_batch)
            if converter_model_type == HilbertProjectionConverter:
                batch_converter_embedding, _ = converter_model(cnn_image_embedding)
            else:
                batch_converter_embedding = converter_model(cnn_image_embedding)
            # Normalize features
            batch_converter_embedding = batch_converter_embedding / batch_converter_embedding.norm(dim=-1, keepdim=True)
            converter_embedding_list.append(batch_converter_embedding)
            
            # Get text features
            batch_clip_text_embedding = clip_model.encode_text(text_batch)
            # Normalize text features
            batch_clip_text_embedding = batch_clip_text_embedding / batch_clip_text_embedding.norm(dim=-1, keepdim=True)
            clip_text_embedding_list.append(batch_clip_text_embedding)
        
        # Concatenate all batch results
        converter_embedding = torch.cat(converter_embedding_list, dim=0)
        clip_text_embedding = torch.cat(clip_text_embedding_list, dim=0)

    return converter_embedding, clip_text_embedding