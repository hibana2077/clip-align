import torch
import timm
import torch.nn as nn
from transformers import CLIPModel
from torch.utils.data import DataLoader, random_split, TensorDataset

from .converter import Converter, Converter_Att, Converter_Linear, HilbertProjectionConverter, ProjectionConverter

def original_clip_inference(
    model_name: str,
    image_set: torch.Tensor, # (N, 3, 224, 224)
    text_set: torch.Tensor, # (N, 77)
    device: str = 'cuda',
):
    # Load the CLIP model
    model = CLIPModel.from_pretrained(model_name).to(device)

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Get image features
        clip_image_embedding = model.get_image_features(image_set.to(device))

        # Get text features
        clip_text_embedding = model.get_text_features(text_set.to(device))

    return clip_image_embedding, clip_text_embedding

def converter_clip_inference(
        clip_model_name: str,
        cnn_model_name: str,
        converter_model_path: str,
        converter_model_type: nn.Module,
        image_set: torch.Tensor, # (N, 3, 224, 224)
        text_set: torch.Tensor, # (N, 77)
        device: str = 'cuda',
):
    # Load the CLIP model
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)

    # Load the CNN model
    cnn_model = timm.create_model(cnn_model_name, pretrained=True, num_classes=0).to(device)
    cnn_model.eval()

    # Get the feature dimension
    test_size = (1, image_set.shape[1], image_set.shape[2], image_set.shape[3])
    dummy_input = torch.randn(test_size).to(device)
    feature_dim = cnn_model(dummy_input).shape[1]

    # Load the Converter model
    converter_model = converter_model_type(input_dim=feature_dim, output_dim=512)
    print(f"converter_model_path: {converter_model_path}")
    converter_model.load_state_dict(torch.load(converter_model_path, weights_only=True))
    converter_model.to(device)

    # Set the models to evaluation mode
    clip_model.eval()
    converter_model.eval()

    # Do the inference
    with torch.no_grad():
        batch_size = 64
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
            converter_embedding_list.append(batch_converter_embedding)
            
            # Get text features
            batch_clip_text_embedding = clip_model.get_text_features(text_batch)
            clip_text_embedding_list.append(batch_clip_text_embedding)
        
        # Concatenate all batch results
        converter_embedding = torch.cat(converter_embedding_list, dim=0)
        clip_text_embedding = torch.cat(clip_text_embedding_list, dim=0)

    return converter_embedding, clip_text_embedding