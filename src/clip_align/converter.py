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
    def __init__(self, input_dim=2048, output_dim=512, path1_hidden_dim=512, path2_hidden_dim=128, hidden_dim=256):
        super().__init__()
        
        # Bilinear low-rank decomposition (reducing the number of paths)
        self.path1 = nn.Sequential(
            nn.Linear(input_dim, path1_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(path1_hidden_dim, output_dim)
        )
        
        self.path2 = nn.Sequential(
            nn.Linear(input_dim, path2_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(path2_hidden_dim, path2_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(path2_hidden_dim, output_dim)
        )
        
        # Lightweight gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),
            nn.Linear(hidden_dim//2, 2)
        )
        
        # Fusion layer with residual connection
        self.fusion = nn.Linear(input_dim + output_dim*2, output_dim)

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x_norm = x
        
        # Gating weights
        gate_weights = F.softmax(self.gate(x_norm), dim=-1)
        w1, w2 = gate_weights.chunk(2, dim=-1)
        
        # Dual-path computation
        out1 = self.path1(x_norm)
        out2 = self.path2(x_norm)
        
        # Weighted fusion
        combined = torch.cat([x_norm, out1*w1, out2*w2], dim=-1)
        fused = self.fusion(combined)

        return fused

class ProjectionConverter(nn.Module):
    def __init__(self, input_dim=2048, output_dim=512, space_X_dim=783, space_Y_dim=783):
        """基于积空间自然投影概念的Converter
        
        将输入空间视为积空间X×Y，通过自然投影π_X和π_Y将高维特征映射到两个子空间，
        然后再融合这两个子空间的特征得到最终输出。
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            space_X_dim: X空间的维度
            space_Y_dim: Y空间的维度
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 定义积空间的维度分解
        self.space_X_dim = space_X_dim
        self.space_Y_dim = space_Y_dim
        
        # 自然投影π_X: X×Y → X
        self.proj_X = nn.Sequential(
            nn.Linear(input_dim, space_X_dim),
            nn.SiLU(inplace=True),
            nn.Linear(space_X_dim, output_dim // 2)
        )
        
        # 自然投影π_Y: X×Y → Y
        self.proj_Y = nn.Sequential(
            nn.Linear(input_dim, space_Y_dim),
            nn.ReLU(inplace=True),
            nn.Linear(space_Y_dim, output_dim // 2)
        )
        
        # 定义投影权重计算函数 - 类似于拓扑空间中的连续性保证
        self.proj_weights = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        
        # 融合层 - 将两个投影空间的特征融合，可以看作是从X和Y重构X×Y的过程
        self.fusion = nn.Linear(input_dim + output_dim, output_dim)
        
        # 输出归一化
        self.layer_norm = nn.GroupNorm(1, output_dim)
        
        # Dropout层 - 可选
        self.dropout = nn.Dropout(0.1)

        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征，视为来自积空间X×Y的点
            
        Returns:
            归一化后的输出特征
        """
        x_norm = self.input_norm(x)
        
        # 计算投影权重 - 确保连续映射的稳定性
        proj_weights = F.softmax(self.proj_weights(x_norm), dim=-1)
        w_X, w_Y = proj_weights.chunk(2, dim=-1)
        
        # 应用自然投影π_X和π_Y
        proj_X_out = self.proj_X(x_norm)
        proj_Y_out = self.proj_Y(x_norm)
        
        # 加权投影结果，保持投影的连续性质
        X_features = proj_X_out * w_X
        Y_features = proj_Y_out * w_Y
        
        # 将两个空间的投影特征拼接，重构为一个完整表示
        combined_projections = torch.cat([X_features, Y_features], dim=-1)
        
        # 融合原始特征和投影特征，类似于从投影重构原始积空间
        combined = torch.cat([x_norm, combined_projections], dim=-1)
        fused = self.fusion(combined)
        
        # 可选的Dropout层
        fused = self.dropout(fused)

        # 最终归一化
        output = self.layer_norm(fused)
        return F.normalize(output, p=2, dim=-1)

class HilbertProjectionConverter(nn.Module):
    def __init__(self, input_dim=2048, output_dim=512, hidden_dim=1024):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 正交投影層
        self.basis_projection = nn.Linear(input_dim, hidden_dim)
        self.projection_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # 凸集約束機制
        self.convex_constraint = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 正交初始化
        nn.init.orthogonal_(self.projection_layer.weight)
        
        # 正則化係數
        self.reg_lambda = 0.001

    def forward(self, x):
        # 步驟1：中間投影
        projected = self.basis_projection(x)
        
        # 步驟2：正交投影到目標空間
        hilbert_projection = self.projection_layer(projected)
        
        # 步驟3：動態約束係數
        constraint = self.convex_constraint(x).sigmoid()
        
        # 步驟4：融合原始特徵與投影特徵
        residual = x[:, :self.output_dim]  # 截取前output_dim維
        output = constraint * hilbert_projection + (1 - constraint) * residual
        
        # 正交正則化（改為返回正則化損失）
        reg_loss = self._orthogonal_regularization_loss()
        
        # 將正則化損失附加到輸出（需在訓練時處理）
        return F.normalize(output, p=2, dim=-1), reg_loss
    
    def _orthogonal_regularization_loss(self):
        weight = self.projection_layer.weight
        identity = torch.eye(weight.size(0), device=weight.device)
        corr = weight @ weight.t()
        return ((corr - identity) ** 2).sum() * self.reg_lambda

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