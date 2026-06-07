# Skin Lesion Classifier — 醫療影像深度學習

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

在 HAM10000 皮膚鏡影像資料集上建立 **7 類皮膚病灶分類器**，於嚴重類別不平衡的條件下，
比較 **CNN（ResNet18 / EfficientNet-B0）** 與 **Vision Transformer（ViT-Tiny）** 架構，
並針對不平衡設計 **CrossEntropy / Class-weighted CE / Focal Loss** 三組 loss 對照。

全程以 **MLflow** 追蹤實驗、**pytest** 把關品質、**GitHub Actions** 跑 CI，
並透過 **FastAPI + Vue 3（TypeScript）** 提供端到端推論 demo。

## 技術棧

| 層面 | 技術 |
|------|------|
| 建模 / 訓練 | PyTorch · torchvision · timm · scikit-learn |
| 實驗追蹤 | MLflow |
| 可解釋性 | Grad-CAM |
| 資料 / EDA | pandas · matplotlib · seaborn |
| 後端 | FastAPI · Pydantic · Uvicorn |
| 前端 | Vue 3 · Vite · Pinia · TypeScript |
| 測試 / 品質 | pytest · pytest-cov · ruff |
| CI/CD · 容器 | GitHub Actions · Docker |
| 環境管理 | uv（Python 3.11） |

## 開發狀態

目前處於早期開發階段，核心功能（資料處理、模型訓練、評估與部署）正陸續建置中。

## 快速開始（開發環境）

```bash
# 建立環境並安裝依賴（含 dev 工具）
uv sync --extra dev

# 跑測試（含覆蓋率）
uv run pytest -v --cov=src --cov-report=term-missing

# Lint
uv run ruff check .
```

## 專案架構

```text
skin-lesion-classifier/
├── src/         # 核心程式：models / data / losses，及 train · evaluate · inference
├── configs/     # 各模型訓練設定（YAML）
├── notebooks/   # EDA 與實驗筆記
├── tests/       # pytest 測試
├── backend/     # FastAPI 推論服務（Docker 部署用）
├── scripts/     # 資料下載、結果烤製等工具
├── results/     # 評估指標、視覺化、checkpoint（大型產物不入庫）
├── docs/        # 對外文件（免責聲明等）
└── .github/     # GitHub Actions CI 設定
```

## 授權 License

- **程式碼 Code**：MIT License（見 [LICENSE](LICENSE)）
- **資料集 Dataset**：HAM10000 採 CC BY-NC 4.0，僅供非商業教育用途

## 致謝 Acknowledgements

- **HAM10000 資料集** — Tschandl, P., Rosendahl, C. & Kittler, H., *The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions* (Scientific Data, 2018)；採 CC BY-NC 4.0。
- **預訓練權重** — ImageNet 預訓練模型來自 [torchvision](https://github.com/pytorch/vision) 與 [timm](https://github.com/huggingface/pytorch-image-models)。
