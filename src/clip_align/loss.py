import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AlignLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=0.07):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        
    def forward(self, pred, target):
        # 余弦相似度损失
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        cosine_loss = 1 - (pred_norm * target_norm).sum(dim=-1).mean()
        
        # 对比学习损失
        sim_matrix = torch.matmul(pred_norm, target_norm.T) / self.temperature
        labels = torch.arange(pred.size(0)).to(pred.device)
        contrastive_loss = (
            F.cross_entropy(sim_matrix, labels) + 
            F.cross_entropy(sim_matrix.T, labels)
        ) / 2
        
        return self.alpha * cosine_loss + (1 - self.alpha) * contrastive_loss