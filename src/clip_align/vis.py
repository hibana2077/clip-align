from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns

def visualize_projection(embeddings, labels, save_name="projection.png", label_type="text"):
    # 确保输入为numpy数组
    embeddings = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    # 使用TSNE降维
    tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(embeddings)-1))
    projected = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 9))
    
    if label_type == "text":
        # 处理类别标签可视化
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(projected[mask, 0], projected[mask, 1], 
                       c=[color], label=label, alpha=0.7, s=50)
            
        plt.legend(title="Classes", bbox_to_anchor=(1.05, 1))
        
    elif label_type == "tensor":
        # 处理嵌入相似度可视化
        # 将numpy数组转回tensor进行归一化计算
        embeddings_tensor = torch.from_numpy(embeddings)
        labels_tensor = torch.from_numpy(labels)
        
        # L2归一化 + 余弦相似度计算
        with torch.no_grad():
            embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=1)
            labels_norm = F.normalize(labels_tensor, p=2, dim=1)
            similarities = (embeddings_norm * labels_norm).sum(dim=1).numpy()  # 点积即余弦相似度
        
        # 创建颜色映射（余弦相似度范围[-1,1]）
        norm = plt.Normalize(vmin=-1, vmax=1)
        cmap = plt.cm.RdBu_r  # 使用红蓝渐变色
        
        scatter = plt.scatter(projected[:, 0], projected[:, 1], 
                             c=similarities, cmap=cmap, norm=norm, 
                             alpha=0.7, s=50, edgecolor='w', linewidth=0.5)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['-1', '0', '1'])
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)
        
    else:
        raise ValueError("label_type must be 'text' or 'tensor'")
    
    # 添加标题和标签
    plt.title(f"t-SNE Projection ({save_name.split('.')[0]})", fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_similarity(clip_embeddings, img_embeddings, convert_embeddings, save_prefix=""):
    # 确保输入为numpy数组
    clip = clip_embeddings.cpu().numpy()
    img = img_embeddings.cpu().numpy()
    convert = convert_embeddings.cpu().numpy()
    
    # 计算相似度矩阵
    def calculate_sim_matrix(a, b):
        return cosine_similarity(a, b)
    
    # 1. 相似度分布对比直方图
    def plot_similarity_distributions():
        # Create subplots
        fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        
        # Calculate similarity
        clip_img_sim = np.diag(calculate_sim_matrix(clip, img))
        clip_convert_sim = np.diag(calculate_sim_matrix(clip, convert))
        convert_img_sim = np.diag(calculate_sim_matrix(convert, img))
        
        # Plot distributions
        sns.histplot(clip_img_sim, ax=ax[0], kde=True, color='blue', bins=20)
        ax[0].set_title('CLIP-ResNet Similarity')
        
        sns.histplot(clip_convert_sim, ax=ax[1], kde=True, color='green', bins=20)
        ax[1].set_title('CLIP-Converted Similarity')
        
        sns.histplot(convert_img_sim, ax=ax[2], kde=True, color='orange', bins=20)
        ax[2].set_title('Converted-ResNet Similarity')
        
        # Set overall title with adjusted layout to avoid cropping
        plt.suptitle('Similarity Distribution Comparison', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{save_prefix}_similarity_distributions.png")
        plt.close()

        
    # 2. 相似度散点图对比
    def plot_similarity_scatter():
        clip_img_sim = np.diag(calculate_sim_matrix(clip, img))
        clip_convert_sim = np.diag(calculate_sim_matrix(clip, convert))
        
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=clip_img_sim, y=clip_convert_sim, alpha=0.6)
        plt.plot([0,1], [0,1], 'r--')  # 对角线参考线
        plt.xlabel('CLIP-ResNet Similarity')
        plt.ylabel('CLIP-Converted Similarity')
        plt.title('Similarity Improvement Analysis')
        plt.grid(True)
        plt.savefig(f"{save_prefix}_similarity_scatter.png")
        plt.close()

    # 执行可视化
    plot_similarity_distributions()
    plot_similarity_scatter()