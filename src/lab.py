# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision.datasets import CIFAR10
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
# from timm.models.resnet import resnet18
# import numpy as np
# from PIL import Image
# from tqdm import tqdm, trange

# # 1. 加载CLIP模型和预处理
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# # test clip output

fake_tensor = torch.randn(1, 3, 224, 224)
clip_output = clip_model.get_image_features(fake_tensor)
print(clip_output.shape)  # 应该是 (1, 512)

# # test resnet output

# ## feature extractor

# resnet = resnet18(pretrained=True)  # 修改输出维度为512
# print(resnet)
# out_feature = resnet(fake_tensor)
# print(out_feature.shape)  # 应该是 (1, 512)

import torch
import torch.nn as nn
# from torchvision.models import resnet18
from timm.models.resnet import resnet18, resnet50
from timm.models.densenet import densenet121
from timm.models.mobilenetv3 import mobilenetv3_small_050

# 加载预训练ResNet-18
# resnet = resnet18(pretrained=True)
resnet = resnet18(pretrained=True)  # 修改输出维度为512
# resnet = resnet50(pretrained=True)  # 修改输出维度为512
# resnet = densenet121(pretrained=True)  # 修改输出维度为512
# resnet = mobilenetv3_small_050(pretrained=True)  # 修改输出维度为512

# 打印原始模型结构（观察层结构）
print("Original ResNet structure:")
print(resnet)

# 正确截断模型：保留到全局池化层
# ResNet的结构顺序是：features -> avgpool -> fc
# 我们需要保留到avgpool层（即去掉最后的fc层）
resnet_features = nn.Sequential(*list(resnet.children())[:-1])

# 冻结特征提取器参数（可选）
for param in resnet_features.parameters():
    param.requires_grad = False

# 测试输入
dummy_input = torch.randn(1, 3, 224, 224)

# 获取特征输出
with torch.no_grad():
    output = resnet_features(dummy_input)

print("\nFeature output shape:", output.shape)  # 应该输出：torch.Size([1, 512, 1, 1])

# 展平后的特征
flattened = output.view(output.size(0), -1)
print("Flattened features shape:", flattened.shape)  # torch.Size([1, 512])