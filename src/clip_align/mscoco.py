import os
import requests
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import zipfile
import shutil

class MSCOCODataset(Dataset):
    """
    A PyTorch Dataset for MSCOCO images downloaded from the official zip files.
    The dataset handles images from train2017, test2017, and val2017 splits.
    """
    
    def __init__(self, 
                 parquet_file_path, 
                 index_url, 
                 split='train2017', 
                 transform=None, 
                 download=True,
                 cache_dir='./mscoco_cache',
                 sample=None):
        """
        Args:
            parquet_file_path (str): Path to the parquet file containing URLs and text
            index_url (str): URL to download the index file
            split (str): One of 'train2017', 'test2017', 'val2017'
            transform (callable, optional): Optional transform to be applied to the images
            download (bool): Whether to download the index file and images if not already available
            cache_dir (str): Directory to cache downloaded images
            sample (int, optional): If provided, limit the dataset to this many samples
        """
        self.parquet_file_path = parquet_file_path
        self.index_url = index_url
        self.split = split
        self.transform = transform
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set the official MSCOCO zip URL based on the split
        self.zip_url = f"http://images.cocodataset.org/zips/{self.split}.zip"
        self.images_dir = os.path.join(self.cache_dir, self.split)
        
        # Download index if needed
        if download:
            self._download_index()
            
            # Download and extract MSCOCO zip file if needed
            self._download_and_extract_zip()
        
        # Load the parquet file
        self.df = pd.read_parquet(parquet_file_path)
        print(f"Loaded dataframe with {len(self.df)} entries")
        
        # Filter only the desired split
        self.df = self.df[self.df['URL'].str.contains(self.split)]
        print(f"Filtered to {len(self.df)} entries for split: {self.split}")
        
        # Sample the dataset if requested
        self.sample = sample
        if self.sample is not None and self.sample < len(self.df):
            self.df = self.df.sample(self.sample, random_state=42)
            print(f"Sampled to {len(self.df)} entries")
        
    def _download_index(self):
        """Download the index file if it doesn't exist"""
        if not os.path.exists(self.parquet_file_path):
            print(f"Downloading index from {self.index_url}...")
            response = requests.get(self.index_url)
            if response.status_code == 200:
                with open(self.parquet_file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded index to {self.parquet_file_path}")
            else:
                raise RuntimeError(f"Failed to download index, status code: {response.status_code}")
        else:
            print(f"Index file already exists at {self.parquet_file_path}")
    
    def _download_and_extract_zip(self):
        """Download and extract the official MSCOCO zip file if it doesn't exist"""
        # Check if images directory already exists and has content
        if os.path.exists(self.images_dir) and os.listdir(self.images_dir):
            print(f"Images for {self.split} already exist in {self.images_dir}")
            return
        
        # Create a temporary directory for the zip file
        zip_dir = os.path.join(self.cache_dir, "zip")
        os.makedirs(zip_dir, exist_ok=True)
        zip_path = os.path.join(zip_dir, f"{self.split}.zip")
        
        # Download the zip file if it doesn't exist
        if not os.path.exists(zip_path):
            print(f"Downloading {self.split} zip from {self.zip_url}...")
            # Stream the download to handle large files
            with requests.get(self.zip_url, stream=True) as response:
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    with open(zip_path, 'wb') as f:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {self.split}.zip") as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    print(f"Downloaded {self.split}.zip to {zip_path}")
                else:
                    raise RuntimeError(f"Failed to download zip, status code: {response.status_code}")
        else:
            print(f"Zip file already exists at {zip_path}")
        
        # Extract the zip file
        print(f"Extracting {self.split}.zip to {self.cache_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)
        print(f"Extracted {self.split}.zip to {self.cache_dir}")
        
        # Optional: Remove the zip file to save space
        # os.remove(zip_path)
        # print(f"Removed zip file {zip_path}")
    
    def _get_image_filename(self, url):
        """Extract the image filename from the URL"""
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        return filename
    
    def _get_image(self, url):
        """Get an image from the extracted zip directory"""
        filename = self._get_image_filename(url)
        image_path = os.path.join(self.images_dir, filename)
        
        try:
            # Load the image from the extracted directory
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder image if there's an error
            return Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            tuple: (image, text) where image is a PIL Image and text is a string
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get the URL and text for this index
        url = self.df.iloc[idx]['URL']
        text = self.df.iloc[idx]['TEXT']
        
        # Get the image
        img = self._get_image(url)
        
        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)
            
        return img, text
    
    def download_all(self, num_workers=2):
        """
        Ensure all images in the dataset are available
        This is now a simple wrapper around the zip download and extract method
        """
        self._download_and_extract_zip()
        print(f"All images for {self.split} are available in {self.images_dir}")

# Example usage:
if __name__ == "__main__":
    from torchvision import transforms
    
    # Define a transform for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create a dataset instance
    dataset = MSCOCODataset(
        parquet_file_path="mscoco.parquet",
        index_url="https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/resolve/main/mscoco.parquet?download=true",
        split="train2017",
        transform=transform,
        download=True,
        cache_dir="./mscoco_cache",
        sample=1000  # Limit to 1000 samples for example
    )
    
    # Create a DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Example of iterating through the DataLoader
    for i, (images, texts) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"Images shape: {images.shape}")
        print(f"Sample text: {texts[0][:50]}...")
        
        if i >= 2:  # Just show first few batches
            break
            
    # You can also pre-download all images to cache
    # dataset.download_all(num_workers=8)