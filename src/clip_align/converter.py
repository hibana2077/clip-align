import torch
import torch.nn as nn
import torch.nn.functional as F

class Converter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, num_blocks=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 特征增强层
        self.augment = nn.Sequential(
            nn.Dropout(0.1),
            nn.LayerNorm(input_dim)
        )
        
        # 动态维度调整
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, output_dim)
        
        # 残差块（简化版）
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(num_blocks)
        ])
        
        # 自适应参数
        self.scale = nn.Parameter(torch.ones(output_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.scale, mean=1.0, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # 特征增强
        x = self.augment(x)
        
        # 维度提升
        x = self.up_proj(x)
        x = F.gelu(x)
        
        # 残差处理
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual  # 显式残差连接
            
        # 维度降下
        x = self.down_proj(x)
        
        # 自适应调整
        x = x * (1 + self.scale) + self.bias  # 改进的缩放方式
        
        return x