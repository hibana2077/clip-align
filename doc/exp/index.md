# Experiment

## Code implementation

- Evaluate different datasets
    - Urban1k
    - Docci5k
    - Flickr1k
    - Flux1k

- Eval model
    - Lightweight(less than ViT-B/32)
        - mobilenetv3_small_075
        - lcnet_050
        - tinynet_e
    - Medium(less than ViT-B/16)
        - convnext_small
        - coatnet_rmlp_2_rw_224
        - efficientnet_b5
    - Large(less than ViT-L/14)
        - beitv2_large_patch16_224
        - eva_large_patch14_196
        - convnextv2_large
    - XLarge(less than ViT-H/14)
        - beitv2_large_patch16_224
        - regnety_1280
        - convnext_xxlarge
    - Special
        - xception41
        - xception65
        - xception71

## Record data

### 計算成本分析

get from [timm](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-infer-amp-nchw-pt240-cu124-rtx3090.csv)

- FLOPs
- Gmacs
- Parameters

### 輕量化模型的困境

> 若直接使用輕量CNN（如ResNet-18）提取圖像特徵，其表示空間與CLIP文本編碼器（CT）結構性不兼容（如圖示：IMB與CTB的相似度接近隨機）。

已經有了 [圖](../assets/similarity_similarity_distributions.png)
或是 tsne 圖

### 對比1： vs 傳統特徵蒸餾 (如FitNets)

- [x] 有拿到他的比較數據

這裡比較cifar-100

可能會需要多加一些code

### 對比2： vs 跨模態對齊 (如ALBEF)

- [x] 測 flickr30k (R@1 R@5 R@10) 的數據

### 對比3： vs 單損失對齊

- [ ] Flickr30k (R@1 R@5 R@10) 的數據
- [ ] CIFAR-100 (R@1 R@5 R@10) 的數據
- [ ] ImageNet-1k (R@1 R@5 R@10) 的數據
- [ ] coco (R@1 R@5 R@10) 的數據
- [ ] 切換模塊開關
- [ ] t-SNE 圖
- [ ] 相似度矩陣熱力圖