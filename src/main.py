import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from torchvision.models import resnet18
import numpy as np
from PIL import Image

# 1. 加载CLIP模型和预处理
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. 加载ResNet-18
resnet = resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层
resnet.eval()

# 3. 数据预处理
def get_transforms():
    # CLIP的预处理会自动处理尺寸
    clip_transform = lambda x: clip_processor(images=x, return_tensors="pt")['pixel_values'][0]
    
    # ResNet预处理需要调整尺寸到224x224
    resnet_transform = Compose([
        Resize(256),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return clip_transform, resnet_transform

# 4. 自定义CIFAR-10数据集
class CIFAR10Dataset(CIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train=train, download=True)
        self.clip_transform, self.resnet_transform = get_transforms()

    def __getitem__(self, index):
        img, _ = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        # 应用不同预处理
        clip_img = self.clip_transform(img)
        resnet_img = self.resnet_transform(img)
        
        return clip_img, resnet_img

# 5. 创建数据加载器
def get_dataloader(batch_size=128):
    dataset = CIFAR10Dataset(root="./data", train=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 6. 生成嵌入
def generate_embeddings(dataloader):
    clip_embeddings = []
    resnet_embeddings = []
    
    with torch.no_grad():
        for clip_imgs, resnet_imgs in dataloader:
            # CLIP嵌入
            clip_features = clip_model.get_image_features(clip_imgs)
            clip_embeddings.append(clip_features)
            
            # ResNet嵌入
            resnet_features = resnet(resnet_imgs)
            resnet_embeddings.append(resnet_features.view(resnet_features.size(0), -1))
    
    return torch.cat(clip_embeddings), torch.cat(resnet_embeddings)

# 7. 转换矩阵
class TransformationMatrix(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.V = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.V)

    def forward(self, x):
        return x @ self.V

# 8. 主训练流程
def main():
    # 创建数据加载器
    dataloader = get_dataloader(batch_size=128)
    
    # 生成嵌入
    clip_emb, resnet_emb = generate_embeddings(dataloader)
    
    # 初始化转换矩阵
    input_dim = clip_emb.shape[1]   # CLIP嵌入维度（512）
    output_dim = resnet_emb.shape[1] # ResNet嵌入维度（512）
    V = TransformationMatrix(input_dim, output_dim)
    
    # 训练设置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(V.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(20):
        optimizer.zero_grad()
        transformed = V(clip_emb)
        loss = criterion(transformed, resnet_emb)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

    # 9. 测试文本转换
    # CIFAR-10类别文本描述
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    text_descriptions = [f"A photo of a {c}" for c in classes]
    
    # 生成文本嵌入
    text_inputs = clip_processor(text=text_descriptions, 
                                return_tensors="pt", 
                                padding=True)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**text_inputs)
    
    # 转换文本嵌入
    transformed_text_emb = V(text_emb)
    
    # 10. 验证图像-文本匹配
    # 随机选择一个图像样本
    sample_idx = np.random.randint(len(resnet_emb))
    sample_image_emb = resnet_emb[sample_idx].unsqueeze(0)
    
    # 计算相似度
    similarities = torch.nn.functional.cosine_similarity(
        transformed_text_emb,
        sample_image_emb,
        dim=1
    )
    
    # 打印结果
    print("\nSimilarity scores:")
    for i, sim in enumerate(similarities):
        print(f"{classes[i]:>10}: {sim.item():.4f}")

if __name__ == "__main__":
    main()