# 論文架構
## Intro  
- **點出問題**  
  - CLIP 的 image encoder 過於龐大、資源消耗高  
    - Edge device 上難以部署，應用場景受限  
    - 面對大量資料時推論速度非常慢  
    - **量化說明**：例如 ResNet‑50 在某些 edge 平台每張圖約需 30 ms，ViT‑B/32 則需 150 ms，相差 5×  
- **核心貢獻與原創性**  
  - **純影像監督**：不需任何 paired text，只以影像資料做對齊  
  - **結構感知對齊**：雙路徑 linear adapter，顧及特徵結構差異  
  - **高靈活性與顯著輕量化**：額外參數與 FLOPs 微乎其微  
- **研究動機**  
  - 使用小型 pretrained image encoder（Timm 提供），降低資源需求、加速推論  
  - 設計一層 lightweight linear adapter，輔助 CNN 特徵映射至 CLIP 空間  
  - 結合 similarity_loss、contrastive_loss 與 CORAL_loss 作為對齊目標  

---

## Related Work

- **按資料需求分類**  
  - 需 image–text pairing：CLIP‑Adapter、LiT、Tip‑Adapter  
  - 純影像對齊／蒸餾：TinyCLIP、CLIP‑KD、Domain Aligned CLIP  
- **最新基線補充**  
  - CoOp、Tip‑Adapter‑F、SLIP 等近年方法  
- **比較維度**  
  - 資料需求、計算成本、推論延遲、對齊精度  
- **批判性分析**  
  - 需大量文本標注、預訓練成本高、收斂慢…  
  - 本研究純影像即可達同級性能  

---

## Method

- **Adapter 設計**  
  - 雙路徑結構：path1（寬淺）、path2（窄深
  - hidden_dim、depth、dropout、weight decay 等超參數說明
  - 參數量與額外 FLOPs 分析
- **Loss Design**  
  - **Similarity Loss**：拉近特徵向量單點距離  
  - **Contrastive Loss**：維持 batch 內硬負樣本對比  
  - **CORAL Loss**：對齊特徵統計分佈  
  - 損失函數完整公式與梯度行為簡要討論  
- **理論洞見**  
  - 線性映射在某條件下對齊誤差上界證明（簡要引理）  
- **複雜度分析**  
  - Adapter 增加的參數量、FLOPs 與 CLIP encoder 相比  

---

## Experiment & Results  
- **Domain Adapter 實驗**
    [link](https://colab.research.google.com/drive/1LqypVvd7VYuuT2OXgp20ZfEoF9zMT_Jl?usp=sharing)
    1. 線性回歸 & Procrustes → MSE、\(R^2\)、殘差  
    2. CCA / SVCCA → 前 k 個 canonical correlations  
    3. Kernel CCA & CKA → 非線性相似度  
    4. 小型 MLP 映射對比 → MSE 下降幅度  
    5. 置換測試 → p‑value 顯著性
    - 結論: 數據支持是 nonlinear projection 的，且用 polynomial kernel, CCA 可以達到 0.76 的 correlation
- **Data Align 敏感度**  
    - Urban1k、Docci5k、Flickr1k、Flux1k  
    - 對齊資料量（1k→5k→10k→50k）對 Recall@{1,5,10} 的變化曲線與斜率比較  
- **Baseline 比較**  
    - 與 CLIP‑Adapter、LiT、TinyCLIP、Tip‑Adapter、CLIP‑KD、Domain Aligned CLIP 等至少 5 種方法  
    - 三維比較：Recall、Latency、Memory  
- **Domain‑specific Evaluation**  
    - 醫療影像（MedICaT）外，加入工業或衛星影像案例驗證泛化能力

- **Hierarchical Recall Pipeline**  
    - 小模型 first‑stage（100% Recall@25）→ 大模型再精選  
    - End‑to‑end 加速比與資源節省量化

- **Ablation Study**  
    - Adapter size、loss 權重 $\alpha$、不同 backbone（ResNet‑18/50、MobileNet、EfficientNet） 

- **可視化與失敗案例**  
    - t‑SNE 對齊前後效果圖  
    - 少數失敗樣本分析，揭示方法局限  

---

## Conclusion & Future Work

- **研究亮點回顧**  
    - 無需文本、極輕量、結構感知對齊  
    - 加速 4×、精度下降 < 2%（具體數字）

- **開放問題與未來方向**  
    - 多模態共訓練架構探索  
    - 基於訓練動態自適應調整 adapter 結構  
    - 延伸至視覺–語言生成與強化學習場景