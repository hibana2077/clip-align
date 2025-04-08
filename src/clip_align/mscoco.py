import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import re


class MSCOCODataset(Dataset):
    """
    A PyTorch Dataset for MSCOCO images downloaded from URLs.
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
            download (bool): Whether to download the index file if not already available
            cache_dir (str): Directory to cache downloaded images
            sample (int, optional): If provided, limit the dataset to this many samples
        """
        self.parquet_file_path = parquet_file_path
        self.index_url = index_url
        self.split = split
        self.transform = transform
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(os.path.join(self.cache_dir, self.split), exist_ok=True)
        
        # Download index if needed
        if download:
            self._download_index()
        
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
    
    def _get_image_filename(self, url):
        """Extract the image filename from the URL"""
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        return filename
    
    def _download_image(self, url):
        """Download an image from a URL and convert to PIL Image"""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert('RGB')
            else:
                print(f"Failed to download image from {url}, status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None
    
    def _get_cached_image_path(self, url):
        """Get the path where the image should be cached"""
        filename = self._get_image_filename(url)
        return os.path.join(self.cache_dir, self.split, filename)
    
    def _get_image(self, url):
        """Get an image either from cache or by downloading"""
        cache_path = self._get_cached_image_path(url)
        
        # If the image is cached, load it from disk
        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path).convert('RGB')
            except Exception as e:
                print(f"Error loading cached image {cache_path}: {e}")
                # If there's an error loading the cached image, try downloading it again
                
        # If not cached or failed to load from cache, download it
        img = self._download_image(url)
        
        # Cache the downloaded image
        if img is not None:
            try:
                img.save(cache_path)
            except Exception as e:
                print(f"Error saving image to cache {cache_path}: {e}")
                
        return img
    
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
        
        # If image download failed, return a placeholder image
        if img is None:
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)
            
        return img, text
    
    def download_all(self, num_workers=4):
        """
        Download all images in the dataset to cache
        
        Args:
            num_workers (int): Number of threads to use for downloading
        """
        urls = self.df['URL'].tolist()
        print(f"Downloading {len(urls)} images with {num_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(self._get_image, urls), total=len(urls)))


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