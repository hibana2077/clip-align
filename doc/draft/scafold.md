# 論文架構

## intro

- 點出問題:
    - clip 的 image encoder 太胖 太吃資源
        - 導致 edge device 難以使用
        - 導致 clip 的應用場景受限
        - 面對大量的資料時，clip 的 image encoder 會變得非常慢

- 相關工作:
    - CLIP-Adapter
    - LiT
    - TinyCLIP
    - Tip‑Adapter
    - CLIP‑KD
    - Domain Aligned CLIP

- 解決方法(本研究提出):
    - 使用更小的且預訓練的 image encoder(pretrained weight provide bt timm)
        - 這樣可以減少資源的使用
        - 這樣可以加快處理速度
        - 這樣可以擴大 clip 的應用場景
    - 使用一個 linear adapter(要實驗證明這個 domain adapter 是有用的)
        - 輔助 image encoder 對齊 clip image encoder 的特徵
    - 結合 similarity_loss + contrastive_loss + coral_loss 作為 align loss

- 原創性
    - 純影像監督
    - 結構感知對齊
    - 高靈活性
    - 顯著輕量化

## related work

上面提到的在講一遍

## method

- loss design
- adapter design
- model evaluation
- performance metrics
- data
    - Urban1k
    - Docci5k
    - Flickr1k
    - Flux1k

## experiment and result

- domain adapter
    - 證明 cnn model 與 clip image encoder 的特徵存在線性關係(or non-linear)
        - rexnetr_200 vs ViT-B/32
        1. 選定 N 張測試圖像 → 提取 X, Y  
        2. 線性回歸 & Procrustes → 計算 MSE, R^2, 殘差  
        3. CCA / SVCCA → 前 k 個 canonical correlations  
        4. 訓練小型 MLP 映射 → 比較 MSE 降低幅度  
        5. Kernel CCA & CKA → 非線性相似度  
        6. 置換測試 → p 值評估顯著性  

- data align
    - 測量多少對齊資料對recall@{1, 5, 10}的影響
        - Urban1k
        - Docci5k
        - Flickr1k
        - Flux1k

- model evaluation
    - compare with different methods
        - CLIP-Adapter
        - LiT
        - TinyCLIP
        - Tip‑Adapter
        - CLIP‑KD
        - Domain Aligned CLIP