from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import io

class Flux1kCaptionDataset(Dataset):
    def __init__(self, split="train"):
        # Load the dataset
        self.ds = load_dataset("Kariander1/flux_1k_captions")
        self.data = self.ds[split].to_dict()
        self.images = self.data['image']
        self.captions = self.data['caption']
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and convert to PIL
        img_data = self.images[idx]
        pil_image = Image.open(io.BytesIO(img_data['bytes']))
        
        # Get caption as string
        caption = self.captions[idx]
        
        return pil_image, caption

# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = Flux1kCaptionDataset(split="train")
    
    # Check length
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    img, caption = dataset[0]
    print(f"Image type: {type(img)}")
    print(f"Image size: {img.size}")
    print(f"Caption: {caption}")