import time
import torch
import torch.nn.functional as F

def compute_similarity(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor):
    """
    Calculates the cosine similarity between each text embedding and all image embeddings.
    Assumes image_embeddings has shape (N, D) and text_embeddings has shape (N, D).
    The returned similarity_matrix has shape (N, N), where similarity_matrix[i, j]
    represents the similarity between the i-th text and the j-th image.
    """
    # Normalize vectors
    image_norm = F.normalize(image_embeddings, p=2, dim=1)
    text_norm = F.normalize(text_embeddings, p=2, dim=1)
    # Calculate similarity matrix
    similarity_matrix = text_norm @ image_norm.t()
    return similarity_matrix

def T2I(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, topk: tuple = (1, 5, 10)):
    """
    Text-to-Image Retrieval: For each text query, find the most similar images.
    Assumes that the ground truth image for each text corresponds to its index
    (i.e., the correct image for the i-th text is the i-th image).
    Returns a dictionary containing the recall values for each top-k.
    """
    similarity_matrix = compute_similarity(image_embeddings, text_embeddings)
    num_texts = similarity_matrix.shape[0]
    # Initialize hit counters
    hits = {k: 0 for k in topk}
    
    # For each text query
    for i in range(num_texts):
        sims = similarity_matrix[i]  # Similarity of the i-th text with all images
        sorted_indices = torch.argsort(sims, descending=True)
        # Find the correct image (assuming ground truth is i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
        for k in topk:
            if rank <= k:
                hits[k] += 1

    # Calculate recall (average hit rate)
    recall = {k: hits[k] / num_texts for k in topk}
    return recall

def I2T(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, topk: tuple = (1, 5, 10)):
    """
    Image-to-Text Retrieval: For each image query, find the most similar texts.
    Assumes that the ground truth text for each image corresponds to its index
    (i.e., the correct caption for the i-th image is the i-th text).
    Returns a dictionary containing the recall values for each top-k.
    """
    similarity_matrix = compute_similarity(image_embeddings, text_embeddings)
    num_images = similarity_matrix.shape[1]
    hits = {k: 0 for k in topk}

    # Use each image as a query, i.e., take each column of the similarity_matrix
    for j in range(num_images):
        sims = similarity_matrix[:, j]  # Similarity of each text with the j-th image
        sorted_indices = torch.argsort(sims, descending=True)
        rank = (sorted_indices == j).nonzero(as_tuple=True)[0].item() + 1  # Find the rank of the correct text
        for k in topk:
            if rank <= k:
                hits[k] += 1

    recall = {k: hits[k] / num_images for k in topk}
    return recall

# Test example (assuming 100 data points, each embedding with dimension 512)
if __name__ == "__main__":
    # Simulate random embeddings (use model-obtained embeddings in practice)
    N, D = 1000, 512
    image_embeddings = torch.randn(N, D)
    text_embeddings = torch.randn(N, D)

    # Call T2I and I2T
    ts = time.time()
    t2i_recall = T2I(image_embeddings, text_embeddings, topk=(1, 5, 10))
    te_t2i = time.time()
    i2t_recall = I2T(image_embeddings, text_embeddings, topk=(1, 5, 10))
    te_i2t = time.time()
    print(f"T2I Recall: {t2i_recall}, Time: {te_t2i - ts:.4f}s")
    print(f"I2T Recall: {i2t_recall}, Time: {te_i2t - te_t2i:.4f}s")
