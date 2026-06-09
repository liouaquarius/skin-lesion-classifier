# Skin Lesion Classifier — 醫療影像深度學習

[![CI](https://github.com/liouaquarius/skin-lesion-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/liouaquarius/skin-lesion-classifier/actions/workflows/ci.yml)

> Multi-class skin lesion classification on HAM10000: comparing CNN and
> Vision Transformer architectures under class-imbalanced conditions,
> with experiment tracking, automated testing, and end-to-end deployment.

## 免責聲明 / Disclaimer

本專案為**研究與教育用途之展示**。所訓練的模型**並非**經認證的醫療器材，
**不得**用於臨床診斷或任何醫療決策。實際醫療問題請務必諮詢合格的醫療專業人員。

> This project is a research and educational demonstration. The trained models
> are **NOT** certified medical devices and **MUST NOT** be used for clinical
> diagnosis or medical decision-making. Always consult qualified medical
> professionals for actual medical concerns.

HAM10000 資料集採 CC BY-NC 4.0 授權，本專案僅作非商業之教育用途使用。

---

## 專案簡介

在 HAM10000 皮膚鏡影像資料集（10,015 張，7 類，最大不平衡比 58.3×）上建立多類別分類器，
比較 **CNN（ResNet18 / EfficientNet-B0）** 與 **Vision Transformer（ViT-Tiny）** 架構，
並針對類別不平衡設計三組 loss 對照：

| Loss | 說明 |
|------|------|
| `cross_entropy` | 基線 |
| `weighted_ce` | 訓練集頻率的反比權重 |
| `focal` | Focal Loss（γ=2.0），抑制 easy negative |

每組實驗均以 **病灶感知分組切割（lesion-aware split）** 防止資料洩漏，
並以 **MLflow** 追蹤 per-class sensitivity、macro AUC 等指標。
推論結果整合 **Grad-CAM** 熱力圖，透過 **FastAPI + Vue 3** 提供互動式 demo。

## 技術棧

| 層面 | 技術 |
|------|------|
| 建模 / 訓練 | PyTorch · torchvision · timm · scikit-learn |
| 實驗追蹤 | MLflow |
| 可解釋性 | Grad-CAM（pytorch-grad-cam） |
| 資料 / EDA | pandas · matplotlib · seaborn |
| 後端 | FastAPI · Pydantic · Uvicorn |
| 前端 | Vue 3 · Vite · TypeScript |
| 測試 / 品質 | pytest · pytest-cov · ruff |
| CI / 容器 | GitHub Actions · Docker（CPU-only） |
| 環境管理 | uv（Python 3.11） |

## 快速開始（開發環境）

```bash
# 1. 安裝依賴
uv sync --extra dev

# 2. 下載資料集（需 Kaggle API token）
uv run --extra data python -u scripts/download_data.py

# 3. 執行測試
uv run pytest -v --cov=src --cov-report=term-missing

# 4. 訓練（以 ResNet18 + cross-entropy 為例）
uv run python -m src.train --config configs/resnet18.yaml

# 5. 評估並輸出報告
uv run python scripts/bake_results.py \
    --config  configs/resnet18.yaml \
    --checkpoint results/checkpoints/resnet18-ce-seed42_best.pt

# 6. 啟動推論服務（後端）
uv run uvicorn backend.main:app --reload

# 7. 啟動前端（另開 terminal）
cd frontend && npm install && npm run dev
```

瀏覽器開 `http://localhost:5173` 即可上傳影像、取得分類結果與 Grad-CAM 熱力圖。

## 專案架構

```
skin-lesion-classifier/
├── src/
│   ├── data/          # SkinLesionDataset · build_transforms · lesion_aware_split
│   ├── models/        # build_resnet18 · build_efficientnet_b0 · build_vit_tiny
│   ├── losses/        # FocalLoss · WeightedCrossEntropy · build_loss
│   ├── train.py       # 訓練迴圈（MLflow · AMP · cosine LR）
│   ├── evaluate.py    # accuracy · per-class sensitivity · macro F1/AUC · confusion matrix
│   └── inference.py   # Predictor（predict · explain with Grad-CAM）
├── backend/
│   ├── main.py        # FastAPI：POST /predict · POST /explain · GET /health
│   ├── schemas.py     # PredictionResponse · ExplainResponse
│   ├── Dockerfile     # CPU-only 推論 image（python:3.11-slim）
│   └── requirements.txt
├── frontend/          # Vue 3 + TypeScript（Vite dev server，proxy → :8000）
├── configs/           # resnet18.yaml · efficientnet_b0.yaml · vit_tiny.yaml
├── notebooks/         # 01_eda.ipynb（含輸出）
├── scripts/
│   ├── download_data.py   # Kaggle API 下載並整理 HAM10000
│   └── bake_results.py    # 測試集評估 → JSON metrics + PNG 視覺化
├── tests/             # 24 個測試；合成 fixture，CI 不需真實資料集
└── results/
    ├── checkpoints/   # 訓練產生的 .pt（不入庫）
    ├── metrics/       # bake_results 輸出的 JSON
    └── visualizations/  # 混淆矩陣、per-class sensitivity PNG
```

## 授權 License

- **程式碼 Code**：MIT License（見 [LICENSE](LICENSE)）
- **資料集 Dataset**：HAM10000 採 CC BY-NC 4.0，僅供非商業教育用途

## 致謝 Acknowledgements

- **HAM10000 資料集** — Tschandl, P., Rosendahl, C. & Kittler, H., *The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions* (Scientific Data, 2018)；採 CC BY-NC 4.0。
- **預訓練權重** — ImageNet 預訓練模型來自 [torchvision](https://github.com/pytorch/vision) 與 [timm](https://github.com/huggingface/pytorch-image-models)。
