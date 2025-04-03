import torch
import torch.nn as nn
import torch.nn.functional as F

class Converter_Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, x):
        # 计算线性变换
        path1 = torch.matmul(x, self.W)
        
        # Hadamard乘积（逐元素相乘）
        return path1 * path1

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

class Converter_Att(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 双头自注意力机制
        self.head1 = SelfAttentionHead(input_dim, hidden_dim, output_dim)
        self.head2 = SelfAttentionHead(input_dim, hidden_dim, output_dim)
        
        # 偏置参数
        self.b = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        # 计算两个注意力头的输出
        path1 = self.head1(x)
        path2 = self.head2(x)
        
        # Hadamard乘积后加偏置
        return path1 * path2 + self.b

class SelfAttentionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.scale = hidden_dim ** -0.5  # 缩放因子

    def forward(self, x):
        # 计算Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 加权求和
        context = torch.matmul(attn_weights, V)
        return context.squeeze(1)  # 移除多余的维度