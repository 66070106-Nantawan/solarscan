# 🛰️ SolarScan — Solar Panel Classification

> Binary Image Classification: มีแผงโซลาร์ / ไม่มีแผงโซลาร์  
> Fine-tuned ResNet50 บนภาพดาวเทียม | **SDG 7: Clean Energy**

---

## 🚀 Quick Start

```bash
# ติดตั้ง Git LFS ก่อน clone
git lfs install
git clone https://github.com/66070106-Nantawan/solarscan.git
cd solarscan/SolarScan_v2
docker compose up
```

| Service | URL |
|---|---|
| 🎨 Gradio UI | http://localhost:7860 |
| ⚡ API Docs | http://localhost:8080/docs |
| 📊 MLflow | http://localhost:5000 |

---

## 📁 Project Structure

```
SolarScan/
├── app/main.py                 ← FastAPI (POST /predict)
├── frontend/gradio_app.py      ← Gradio UI
├── scripts/
│   ├── prepare_data.py         ← แบ่ง train/val/test
│   ├── train.py                ← Fine-tune ResNet50 + MLflow
│   └── evaluate.py             ← Accuracy, F1, AUC-ROC
├── notebooks/01_EDA.ipynb      ← วิเคราะห์ dataset
├── model/weights/best.pt       ← trained weights
├── Dockerfile
├── Dockerfile.gradio
└── docker-compose.yml
```

---

## 🗂️ Dataset Structure ที่ต้องเตรียม

```
data/raw/
├── solar/        ← ภาพดาวเทียมที่มีแผงโซลาร์
└── no_solar/     ← ภาพดาวเทียมที่ไม่มีแผงโซลาร์
```

---

## ⚙️ Train Model

```bash
# 1. เตรียมข้อมูล
python scripts/prepare_data.py

# 2. Train + MLflow tracking
python scripts/train.py

# 3. Evaluate
python scripts/evaluate.py
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |
| AUC-ROC | TBD |

---

## 🛠️ Tech Stack

| Category | Tool |
|---|---|
| Model | ResNet50 (fine-tuned) |
| Backend | FastAPI |
| Frontend | Gradio |
| ML Tracking | MLflow |
| Deploy | Docker + Render |

---

## 👥 Team

| ชื่อ | บทบาท |
|------|-------|
| [ชื่อ 1] | ML Engineer |
| [ชื่อ 2] | Backend + DevOps |
