# Experiment

## Code implementation

- Evaluate different datasets
    - genral
        - mscoco5k
        - Urban1k
        - Docci5k
        - Flickr1k
    - medical
        - ROCO10k
    - AI-generated
        - flux1k

- Eval model
    - Lightweight(less than ViT-B/32)
        - mobilenetv3_small_075
        - lcnet_050
        - tinynet_e
    - Small(less than ViT-B/32) (flops > 1)
        - convnextv2_nano, 4.06
        - rexnetr_200, 1.59
        - maxxvit_rmlp_nano_rw_256, 4.37
    - Medium(less than ViT-B/16)
        - convnext_small, 14.39
        - coatnet_rmlp_2_rw_224, 15.18
        - efficientnet_b5, 9.59
    - Large(less than ViT-L/14)
        - convnext_xxlarge, 151.66
        - eva_large_patch14_196, 61.57
        - beit_large_patch16_224, 61.60
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
- [ ] 相似度矩陣熱力圖(eval)

## 特定領域對齊

醫學影像案例：
利用已有的 MedICaT（專用於放射影像的 CLIP 模型，來自 hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO），探討以下兩種情況：

用一般預訓練模型去適配 MedICaT-ROCO，觀察在醫學影像領域的效果表現。

使用 MedICaT-ROCO 的圖像編碼器，再對齊一般 CLIP，進行效果的上下游比對，分析是否存在性能提升或下降。

## Ablation study

### Loss

消融設計（逐項去除或組合）：

| 編號 | 說明                              | 使用的 loss 組合                          |
|------|-----------------------------------|------------------------------------------|
| A    | **Full（baseline）**              | `similarity + contrastive + coral`       |
| B    | 去除 coral                         | `similarity + contrastive`               |
| C    | 去除 contrastive                  | `similarity + coral`                     |
| D    | 去除 similarity                   | `contrastive + coral`                    |
| E    | 僅用 similarity                   | `similarity`                             |
| F    | 僅用 contrastive                  | `contrastive`                            |
| G    | 僅用 coral                        | `coral`                                  |
| H    | similarity + weighted coral       | `similarity + λ * coral`（加個λ權重）     |
| I    | contrastive + weighted coral      | `contrastive + λ * coral`                |
| J    | 使用 cosine + contrastive (soft) | 用 soft cosine loss + contrastive        |

 **Top-1 accuracy / loss 值** 來畫 bar chart 或 radar chart 呈現效果貢

### Model

#### CP1

- Converter_Linear
- Converter
- ProjectionConverter
- HilbertProjectionConverter
- Converter_Att

#### CP2

- **Path1 wide / Path2 narrow**（baseline 情境）  
- **Path1 narrow / Path2 wide**（反向測試）  
- **Path1 和 Path2 相同寬度**（控制變因）  
- **增加/減少兩邊的 hidden_dim 同時看 scale 效果**


消融組合設計（[path1_hidden_dim, path2_hidden_dim]）

| 編號 | 說明                     | 組合             |
|------|--------------------------|------------------|
| A    | Baseline                 | `[512, 128]`     |
| B    | Reverse wide/narrow      | `[128, 512]`     |
| C    | 等寬                     | `[256, 256]`     |
| D    | 極端廣淺 vs 窄深         | `[1024, 64]`     |
| E    | 極端窄深 vs 廣淺         | `[64, 1024]`     |
| F    | 較小 scale（tiny model） | `[128, 64]`      |
| G    | 較大 scale（large model）| `[1024, 512]`    |
| H    | 偏平衡廣                 | `[512, 256]`     |
| I    | 偏平衡深                 | `[256, 512]`     |

效能統計圖（像 heatmap）