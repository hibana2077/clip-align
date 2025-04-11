import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

class Urban1k(Dataset):
    # Thanks to Beichen Zhang for providing the dataset
    url = "https://huggingface.co/datasets/BeichenZhang/Urban1k/resolve/main/Urban1k.zip?download=true"
    filename = "data.zip"
    data_folder = "Urban1k"

    def __init__(self, root_dir, transform=None, download=False):
        self.root_dir = root_dir
        self.transform = transform
        self.data_path = os.path.join(self.root_dir, self.data_folder)
        
        if download:
            self.download()
            
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
            
        # Get the index of all files
        self.samples = list(range(1, 1001))  # 1 ~ 1000

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_num = self.samples[idx]

        # Construct file paths
        img_path = os.path.join(self.data_path, "image", f"{file_num}.jpg")
        txt_path = os.path.join(self.data_path, "caption", f"{file_num}.txt")

        # Read image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Read text content
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        return image, text

    def download(self):
        if self._check_exists():
            return
            
        os.makedirs(self.root_dir, exist_ok=True)
        download_and_extract_archive(
            self.url,
            download_root=self.root_dir,
            filename=self.filename,
            remove_finished=True
        )

    def _check_exists(self):
        # Check if the necessary files exist
        exists = True
        for i in range(1, 1001):
            img_path = os.path.join(self.data_path, "image", f"{i}.jpg")
            txt_path = os.path.join(self.data_path, "caption", f"{i}.txt")
            # print(img_path, txt_path)
            exists &= os.path.exists(img_path) and os.path.exists(txt_path)
            if not exists:
                break
        return exists

if __name__ == "__main__":
    from torchvision import transforms

    # 定義圖片轉換
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 初始化資料集
    dataset = Urban1k(
        root_dir="./data",
        transform=transform,
        download=True  # 第一次使用時需要設為True
    )

    # 取得單一樣本
    image, text = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Text content: {text}")

    print(f"Total samples: {len(dataset)}")
    print(dataset.samples)
