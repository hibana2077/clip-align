import torch
import torch.nn as nn

class Converter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 第一条路径的参数
        self.X1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W1 = nn.Parameter(torch.randn(hidden_dim, output_dim))
        
        # 第二条路径的参数
        self.X2 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim))

        # 偏置参数
        self.b = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        # 计算两个路径的矩阵乘积
        path1 = torch.matmul(x, self.X1) @ self.W1
        path2 = torch.matmul(x, self.X2) @ self.W2
        
        # Hadamard乘积（逐元素相乘）后加偏置
        return path1 * path2 + self.b