# Model Card — Skin Lesion Classifier

> 研究與教育用途之模型卡。本模型**並非**醫療器材，詳見[限制](#限制-limitations)與[免責聲明](#倫理與免責-ethical-considerations)。

## 模型概述 (Model Details)

| 項目 | 內容 |
|------|------|
| 任務 | HAM10000 皮膚鏡影像 7 類分類 |
| **部署模型** | **ViT-Tiny**（`vit_tiny_patch16_224`, timm）+ **weighted cross-entropy** |
| 其他訓練模型 | ResNet18、EfficientNet-B0（各 × ce / wce / focal，共 9 組對照） |
| 預訓練來源 | ImageNet（torchvision / timm） |
| 輸入 | 224×224 RGB 皮膚鏡影像 |
| 輸出 | 7 類 softmax 機率（akiec, bcc, bkl, df, mel, nv, vasc） |
| 框架 / 再現 | PyTorch；random seed 42；lesion-aware split |
| License | 程式碼 MIT；資料 HAM10000 CC BY-NC 4.0 |

## 適用範圍 (Intended Use)

- ✅ **研究 / 教育展示**：ML 方法比較（CNN vs Vision Transformer、類別不平衡下的 loss ablation）、可解釋性（Grad-CAM）展示。
- ❌ **不得**用於臨床診斷、篩檢或任何醫療決策。本模型**不是**經認證的醫療器材。

## 訓練資料 (Training Data)

HAM10000：10,015 張皮膚鏡影像，7 類，採**病灶感知分組切割**（lesion-aware split，防止同一病灶跨 split 造成資料洩漏）。

| 類別 | 全名 | 樣本數 | 佔比 |
|------|------|------:|-----:|
| nv | Melanocytic Nevi | 6705 | 66.9% |
| mel | Melanoma | 1113 | 11.1% |
| bkl | Benign Keratosis | 1099 | 11.0% |
| bcc | Basal Cell Carcinoma | 514 | 5.1% |
| akiec | Actinic Keratosis / IEC | 327 | 3.3% |
| vasc | Vascular Lesion | 142 | 1.4% |
| df | Dermatofibroma | 115 | 1.1% |

**最大不平衡比 58.3×**（nv : df）。

## 訓練設定 (Training)

- 30 epochs、AdamW（lr 1e-4、weight decay 1e-4）、cosine LR schedule、混合精度（AMP）、batch size 32、影像 224²。
- `weighted_ce`：以訓練集頻率反比作為類別權重；`focal`：Focal Loss γ=2.0。
- best checkpoint 以**驗證集 accuracy** 選取。

## 評估 (Evaluation)

於**留出測試集**（與訓練同 seed 的 lesion-aware split）評估，指標：accuracy、per-class sensitivity、macro F1、macro AUC、confusion matrix。

部署模型 **ViT-Tiny + weighted-CE** 的測試集表現：

| accuracy | macro F1 | macro AUC | melanoma sensitivity |
|---------:|---------:|----------:|---------------------:|
| 0.795 | 0.669 | 0.960 | 0.591 |

完整 9 組比較表與圖見 [README 實驗結果](README.md#實驗結果) 與 [`results/`](results/)。

## 限制 (Limitations)

> 這些限制是理解本模型可信度的關鍵，請務必閱讀。

- **少數類樣本嚴重不足**：`df`（115）與 `vasc`（142）樣本極少，其 sensitivity **極不穩定**（df 在不同設定下於 0.2–0.7 間劇烈跳動），**不可信賴**。
- **accuracy 會誤導**：多數類 `nv` 佔 66.9%，整體 accuracy 受其主導；單看 accuracy 會**高估**模型的臨床效用。應同時參考 per-class sensitivity 與 macro 指標。
- **melanoma 漏判風險**：即使採 weighted-CE / focal，melanoma sensitivity 仍 < 0.7，意即仍有相當比例的黑色素瘤會被漏判——**不足以用於任何臨床篩檢**。
- **loss 效果非一致**：再平衡 loss 不保證提升少數類；例如 ViT-Tiny + focal 的 melanoma sensitivity 反而最低（0.425）。
- **資料偏誤**：HAM10000 為皮膚鏡單一影像型態，且來源族群以歐洲膚色為主，對深膚色、其他成像條件的**泛化能力未知**。
- **統計穩健性**：僅單一 seed、單次 split，未報告跨 seed 的變異或信賴區間。

## 倫理與免責 (Ethical Considerations)

- 醫療 AI 有**被過度信任**的風險；假陰性（漏判 melanoma）後果嚴重。本模型僅作技術展示。
- **This model is NOT a certified medical device and MUST NOT be used for clinical diagnosis or medical decision-making. Always consult qualified medical professionals for actual medical concerns.**
- 任何實際醫療問題，請務必諮詢合格的醫療專業人員。
