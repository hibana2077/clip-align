from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import io

class ROCOv2Dataset(Dataset):
    def __init__(self, split="test"):
        # Load the dataset
        self.ds = load_dataset("eltorio/ROCOv2-radiology", split=split)
        self.data = self.ds.to_dict()
        self.images = self.data['image']
        self.captions = self.data['caption']
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and convert to PIL
        img_data = self.images[idx]
        pil_image = Image.open(io.BytesIO(img_data['bytes']))
        
        # Convert to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Get caption as string
        caption = self.captions[idx]
        
        return pil_image, caption

# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = ROCOv2Dataset(split="test")  # Can also use "train" or "val"
    
    # Check length
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    img, caption = dataset[0]
    print(f"Image type: {type(img)}")
    print(f"Image size: {img.size}")
    print(f"Caption: {caption}")