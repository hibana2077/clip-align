from .docci import DOCCI
from torch.utils.data import Dataset
from PIL import Image
import io

class DOCCI5k(Dataset):
    def __init__(self, split="test"):
        # Load the dataset
        builder = DOCCI()
        builder.download_and_prepare()
        docci_ds = builder.as_dataset(split=split)
        # Store the dataset
        self.data = docci_ds.to_dict()
        self.images = self.data['image']
        self.descriptions = self.data['description']
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and convert to PIL
        img_path = self.images[idx]['path']
        pil_image = Image.open(img_path).convert('RGB')
        
        # Get description as string
        description = self.descriptions[idx]
        
        return pil_image, description

# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = DOCCI1k()
    
    # Check length
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    img, description = dataset[0]
    print(f"Image type: {type(img)}")
    print(f"Image size: {img.size}")
    print(f"Description: {description[:50]}...")  # Print first 50 chars