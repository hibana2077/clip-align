# Contributions

## 1. AlignLoss

We proposed a novel loss function called AlignLoss, which combines two distinct loss components: SimilarityLoss and AlignLoss. The SimilarityLoss is designed to measure the similarity between the predicted features and the target features using a selected similarity metric (cosine, Euclidean, or Manhattan). The AlignLoss then combines this similarity loss with the cross-entropy loss obtained through contrastive learning, adjusting the weights of both components using an alpha parameter. By normalizing and scaling the feature vectors with a temperature parameter, this design leverages both direct similarity comparisons and contrastive learning advantages to enhance the model's feature alignment and recognition performance.

## 2. Converter

We propose a novel converter that employs two independent linear projection paths to map input features into a hidden space. The outputs of these two paths are then fused using an element-wise Hadamard product, followed by the addition of a bias term. This design captures complementary information between the two paths, effectively enhancing feature representation and improving learning performance for subsequent tasks.

## 3. Align Method

we propose a novel alignment method utilizing an extremely lightweight two-path linear projector. This projector efficiently aligns CNN-extracted image features into the domain of CLIP image encoder outputs, thereby eliminating the extensive and computationally expensive training required by existing methods such as TinyCLIP or CLIP-Adapter. Moreover, unlike methods like LiT that depend on paired image-text data, our approach requires only image data for adaptation, significantly simplifying data acquisition and preprocessing. We further provide comprehensive ablation experiments and employ t-SNE visualizations to empirically validate our theoretical insights.