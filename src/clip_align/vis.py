from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_projection(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=0) # 设置random_state以保证结果可复现
    projected = tsne.fit_transform(embeddings.cpu())
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projected[:,0], projected[:,1], c=labels, cmap='tab10')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("2D t-SNE Projection of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    # plt.show()
    plt.savefig("tsne_projection.png")  # 保存图像