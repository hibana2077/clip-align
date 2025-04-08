import torch
import torch.nn as nn
import torch.nn.functional as F

def coral_loss(x, y):
    mean_x = x.mean(0)
    mean_y = y.mean(0)
    cov_x = torch.cov(x.T)
    cov_y = torch.cov(y.T)
    return F.mse_loss(mean_x, mean_y) + F.mse_loss(cov_x, cov_y)

class SimilarityLoss(nn.Module):
    def __init__(self, mode='cosine', temperature=0.07):
        """
        mode: 'cosine', 'euclidean', or 'manhattan'
        temperature: Used for contrastive loss calculation only in cosine mode
        """
        super().__init__()
        self.mode = mode
        self.temperature = temperature

    def forward(self, pred, target):
        if self.mode == 'cosine':
            # Cosine similarity loss
            pred_norm = F.normalize(pred, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            cosine_loss = 1 - (pred_norm * target_norm).sum(dim=-1).mean()
            return cosine_loss

        elif self.mode == 'euclidean':
            # Euclidean distance loss: using MSE as a substitute for Euclidean distance
            # Note: If you want to directly calculate L2 distance, you can implement it yourself
            euclidean_loss = torch.mean(torch.sum((pred - target) ** 2, dim=-1))
            return euclidean_loss

        elif self.mode == 'manhattan':
            # Manhattan distance loss
            manhattan_loss = torch.mean(torch.sum(torch.abs(pred - target), dim=-1))
            return manhattan_loss

        else:
            raise ValueError(f"Unsupported similarity mode: {self.mode}")

class AlignLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=0.07, similarity_mode='cosine'):
        """
        alpha: Weight to balance similarity loss and contrastive loss
        temperature: Temperature parameter in contrastive loss
        similarity_mode: Pass the mode parameter into SimilarityLoss
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.similarity_loss_fn = SimilarityLoss(mode=similarity_mode, temperature=temperature)
        
    def forward(self, pred, target):
        # Calculate similarity loss based on the selected similarity mode
        similarity_loss = self.similarity_loss_fn(pred, target)
        
        # contrastive loss part (still using the original cosine normalization method here)
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        sim_matrix = torch.matmul(pred_norm, target_norm.T) / self.temperature
        labels = torch.arange(pred.size(0)).to(pred.device)
        contrastive_loss = (
            F.cross_entropy(sim_matrix, labels) + 
            F.cross_entropy(sim_matrix.T, labels)
        ) / 2
        
        return self.alpha * similarity_loss + (1 - self.alpha) * contrastive_loss + coral_loss(pred, target)

# Test example
if __name__ == "__main__":
    # Randomly generate predictions and targets
    pred = torch.randn(8, 128)
    target = torch.randn(8, 128)

    # Use cosine similarity loss
    loss_fn_cosine = AlignLoss(alpha=0.5, temperature=0.07, similarity_mode='cosine')
    loss_cosine = loss_fn_cosine(pred, target)
    print("Cosine loss:", loss_cosine.item())

    # Use Euclidean distance loss
    loss_fn_euclidean = AlignLoss(alpha=0.5, temperature=0.07, similarity_mode='euclidean')
    loss_euclidean = loss_fn_euclidean(pred, target)
    print("Euclidean loss:", loss_euclidean.item())

    # Use Manhattan distance loss
    loss_fn_manhattan = AlignLoss(alpha=0.5, temperature=0.07, similarity_mode='manhattan')
    loss_manhattan = loss_fn_manhattan(pred, target)
    print("Manhattan loss:", loss_manhattan.item())
