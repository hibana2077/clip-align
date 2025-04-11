from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import io
import random

# set the seed for reproducibility
random.seed(42)

class Flickr1kDataset(Dataset):
    def __init__(self, split="test"):
        # Load the dataset
        self.ds = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval")
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
        # Note: Each image in this dataset has multiple captions
        # Here we're randomly selecting one from the list
        captions_list = self.captions[idx]
        caption = random.choice(captions_list)
        
        return pil_image, caption

# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = Flickr1kDataset(split="test")
    
    # Check length
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    img, caption = dataset[0]
    print(f"Image type: {type(img)}")
    print(f"Image size: {img.size}")
    print(f"Caption: {caption}")