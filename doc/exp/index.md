# Experiment

## Code implementation

- Evaluate different datasets
    - Urban1k
    - Docci
    - Flickr30k
    - CIFAR-100
    - [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
    - [coco](https://huggingface.co/datasets/detection-datasets/coco)

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