from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import io
import random

random.seed(42)  # Set the seed for reproducibility

class MSCOCO5k(Dataset):
    def __init__(self, split="test"):
        # Load the dataset
        self.ds = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval", split=split)
        self.data = self.ds.to_dict()
        self.images = self.data['image']
        self.captions = self.data['caption']
        self.name = "mscoco5k"
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and convert to PIL
        img_data = self.images[idx]
        pil_image = Image.open(io.BytesIO(img_data['bytes']))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Get caption as string
        # Note: Each image has multiple captions in a list
        captions_list = self.captions[idx]
        caption = random.choice(captions_list)  # Randomly select one caption
        
        return pil_image, caption

# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = MSCOCO5k(split="test")
    
    # Check length
    print(f"Dataset name: {dataset.name}")
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    img, caption = dataset[0]
    print(f"Image type: {type(img)}")
    print(f"Image size: {img.size}")
    # check channel
    print(f"Image channels: {len(img.getbands())}")
    print(f"Caption: {caption}")

    channel_stats = {}
    for i in range(len(dataset)):
        img, caption = dataset[i]
        channels = len(img.getbands())
        if channels not in channel_stats:
            channel_stats[channels] = 0
        channel_stats[channels] += 1
    print(channel_stats)